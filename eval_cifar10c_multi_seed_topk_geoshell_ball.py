"""
CIFAR-10-C evaluation on offline ResNet50 feature files with:
- multi-seed loop (e.g., seeds 0..9)
- forced Top-k masking (no radius thresholding)
- geo two modes: shell vs ball
- ridge retrain on clean per (seed, k_frac), then eval CIFAR-10-C

Assumptions:
- Clean train features: ./data/cifar10_r50_train.pt with keys {"X": [50000,H], "y": [50000]}
- CIFAR-10-C offline dir: contains {corruption}.pt each with {"X": [10000,H] or [50000,H], "y": [...]}
- Geo search results yaml per seed exists (contains k_frac + best_center + best_radius)
- pts_fixed .pt exists (contains {"pts_fixed": [H, pts_dim]}) and is shared by shell/ball for fairness
"""

import os
import time
import copy
import yaml
import numpy as np
import torch

from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# ===================== USER CONFIG =====================

CFG = {
    # Seeds to evaluate in one run
    "seeds": list(range(10)),  # 0..9

    # Methods to evaluate: "geo" | "kwta" | "random"
    # If you want both geo shell & geo ball, keep "geo" and set geo_modes below.
    "methods": ["geo"],

    # Geo scoring topology modes
    # - "shell": score = (sqrt(d2) - r)^2
    # - "ball":  score = sqrt(d2)  (radius ignored for ranking; still read if present)
    "geo_modes": ["ball"],

    # If True, use k list from each seed's results_yaml; else use manual list
    "use_k_list_from_yaml": True,
    "manual_k_fracs": [round(i * 0.05, 2) for i in range(1, 21)],  # 0.05..1.0

    # Paths
    "cifar10_r50_train_pt": "./data/cifar10_r50_train.pt",
    "cifar10c_r50_dir": "./data/cifar10c_r50_sev5",

    # These should point to your *SEARCH outputs* (per seed).
    # NOTE: for shell vs ball, you typically have different result YAMLs.
    # You can encode mode into the template if you have separate files.
    "results_yaml_template": "./results_kfrac_geo_r_shell_shuffle_c{seed}.yaml",
    # If you have a separate ball-search yaml, set:
    # "results_yaml_template_ball": "./cifar10_results_r50/results_kfrac_geo_r_ball_seed{seed}.yaml",

    # pts_fixed per seed (shared across shell/ball); should match the search run.
    "pts_fixed_template": "pts_fixed_seed{seed}_randproj.pt",

    # Train/val split for ridge
    "val_ratio": 0.1,
    "ridge_alpha": 1.0,

    # CIFAR-10-C severity slicing
    "severity": 5,  # 1..5

    # Output
    "out_dir": "./cifar10c_eval_multi_seed_c1_shuffled",
    "out_tag": "r50",  # used in filenames

    # Performance / determinism knobs
    "batch_eval_numpy": True,  # use numpy batch; (kept simple, scikit handles)
}

# ======================================================


# -------------------- Utilities --------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)

def save_csv(rows, path, header):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r.get(h, "") for h in header])

def get_corruptions_from_pt(cifar10c_r50_dir):
    corruptions = [
        "brightness", "contrast", "defocus_blur", "elastic_transform",
        "fog", "frost", "gaussian_blur", "gaussian_noise",
        "glass_blur", "impulse_noise", "jpeg_compression", "motion_blur",
        "pixelate", "saturate", "shot_noise", "snow",
        "spatter", "speckle_noise", "zoom_blur",
    ]
    out = []
    for c in corruptions:
        if os.path.exists(os.path.join(cifar10c_r50_dir, f"{c}.pt")):
            out.append(c)
    return out

def split_train_val_indices(n, val_ratio, seed):
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    n_val = max(1, int(n * float(val_ratio)))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    return tr_idx, val_idx

def one_hot(y_int, num_classes=10):
    y = np.asarray(y_int, dtype=np.int64)
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out

def build_fixed_perm(H, seed):
    rng = np.random.default_rng(int(seed))
    return rng.permutation(H)  # fixed nested order for random masks

def load_pts_fixed(path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and ("pts_fixed" in ckpt):
        pts = ckpt["pts_fixed"]
    elif isinstance(ckpt, torch.Tensor):
        pts = ckpt
    else:
        raise ValueError(f"Unrecognized pts_fixed file format: {path}")
    pts = pts.detach().cpu().numpy().astype(np.float32)
    return pts  # [H, pts_dim]

def load_cifar10_train_features(path):
    pack = torch.load(path, map_location="cpu")
    X = pack["X"].float().numpy().astype(np.float32)
    y = pack["y"].long().numpy().astype(np.int64)
    return X, y

def load_cifar10c_features(pt_path, severity):
    pack = torch.load(pt_path, map_location="cpu")
    X = pack["X"].float().numpy().astype(np.float32)
    y = pack["y"].long().numpy().astype(np.int64)

    if X.shape[0] == 50000:
        s = (int(severity) - 1) * 10000
        e = int(severity) * 10000
        X = X[s:e]
        y = y[s:e]
    elif X.shape[0] != 10000:
        raise ValueError(f"Unexpected X shape in {pt_path}: {X.shape} (expected 10000 or 50000)")
    return X, y

def get_run_k_frac(run):
    if "k_frac" in run:
        return float(run["k_frac"])
    if "kfrac" in run:
        return float(run["kfrac"])
    raise KeyError("run missing k_frac/kfrac")

def get_run_center_radius(run):
    # support multiple key spellings
    center = None
    radius = None

    for ck in ["best_center", "bestcenter", "bestCenter", "center"]:
        if ck in run:
            center = run[ck]
            break
    for rk in ["best_radius", "bestradius", "bestRadius", "radius"]:
        if rk in run:
            radius = run[rk]
            break

    if center is None:
        raise KeyError("run missing center (best_center/bestcenter/...)")
    # radius may be unused in ball mode; still try to parse if present
    if radius is None:
        radius = 0.0

    center = np.asarray(center, dtype=np.float32)
    radius = float(radius)
    return center, radius

# -------------------- Mask builders (FORCED TOP-K) --------------------

def topk_smallest_indices(scores, k_num):
    # scores: [H], smaller is better
    k_num = int(max(1, min(int(k_num), scores.shape[0])))
    # argpartition is O(H)
    idx = np.argpartition(scores, kth=k_num - 1)[:k_num]
    # optional: sort idx by score for determinism
    idx = idx[np.argsort(scores[idx])]
    return idx

def geo_topk_indices(pts_fixed, center, radius, k_frac, mode):
    # pts_fixed: [H, d]
    H = pts_fixed.shape[0]
    k_num = max(1, int(H * float(k_frac)))

    diff = pts_fixed - center.reshape(1, -1)
    d = np.sqrt(np.sum(diff * diff, axis=1) + 1e-12)  # [H]

    if mode == "shell":
        scores = (d - float(radius)) ** 2  # (sqrt(d2)-r)^2
    elif mode == "ball":
        scores = d  # nearest-to-center
    else:
        raise ValueError(f"Unknown geo mode: {mode}")

    return topk_smallest_indices(scores, k_num)

def kwta_topk_indices(X_train, k_frac):
    # global kWTA: rank by mean activation over train set, pick top-k
    H = X_train.shape[1]
    k_num = max(1, int(H * float(k_frac)))
    scores = -X_train.mean(axis=0)  # we want largest mean => smallest (-mean)
    return topk_smallest_indices(scores, k_num)

def random_topk_indices(fixed_perm, k_frac):
    H = fixed_perm.shape[0]
    k_num = max(1, int(H * float(k_frac)))
    return fixed_perm[:k_num].copy()

# -------------------- Ridge train/eval --------------------

def train_ridge_on_clean(Xtr, ytr_int, Xva, yva_int, feat_idx, alpha):
    """
    Fit StandardScaler on Xtr, transform, select columns, fit Ridge (multi-target).
    Return scaler, ridge, and clean val acc.
    """
    ytr = one_hot(ytr_int, 10)
    yva = one_hot(yva_int, 10)

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_s = scaler.fit_transform(Xtr)[:, feat_idx]
    Xva_s = scaler.transform(Xva)[:, feat_idx]

    ridge = Ridge(alpha=float(alpha), solver="cholesky")
    ridge.fit(Xtr_s, ytr)

    pred = ridge.predict(Xva_s)
    acc = float(np.mean(np.argmax(pred, axis=1) == np.argmax(yva, axis=1)))
    return scaler, ridge, acc

def eval_ridge_accuracy(ridge, scaler, X, y_int, feat_idx):
    Xs = scaler.transform(X)[:, feat_idx]
    pred = ridge.predict(Xs)
    yhat = np.argmax(pred, axis=1)
    return float(np.mean(yhat == y_int))

# -------------------- Main --------------------

def main():
    cfg = CFG
    ensure_dir(cfg["out_dir"])

    corruptions = get_corruptions_from_pt(cfg["cifar10c_r50_dir"])
    if not corruptions:
        raise FileNotFoundError(f"No CIFAR-10-C .pt found under: {cfg['cifar10c_r50_dir']}")

    X_all, y_all = load_cifar10_train_features(cfg["cifar10_r50_train_pt"])
    n, H = X_all.shape
    print("Loaded clean train:", X_all.shape, y_all.shape)
    print("Corruptions:", corruptions)

    all_rows = []  # flat records for csv
    # For summary CI across seeds
    # key: (method, geo_mode or "-", k_frac) -> list of cifar10c_mean_acc across seeds
    acc_pool = {}

    for seed in cfg["seeds"]:
        print(f"\n========== SEED {seed} ==========")
        tr_idx, va_idx = split_train_val_indices(n, cfg["val_ratio"], seed)
        Xtr, ytr = X_all[tr_idx], y_all[tr_idx]
        Xva, yva = X_all[va_idx], y_all[va_idx]

        fixed_perm = build_fixed_perm(H, seed)

        # pts_fixed per seed (shared across shell/ball)
        pts_path = cfg["pts_fixed_template"].format(seed=seed)
        pts_fixed = load_pts_fixed(pts_path)
        assert pts_fixed.shape[0] == H, f"pts_fixed H mismatch: {pts_fixed.shape} vs H={H}"
        pts_dim = pts_fixed.shape[1]
        print("Loaded pts_fixed:", pts_fixed.shape)

        # results yaml per seed
        results_yaml_path = cfg["results_yaml_template"].format(seed=seed)
        results = load_yaml(results_yaml_path)
        runs = results.get("runs", [])
        if not runs:
            raise ValueError(f"No runs found in results yaml: {results_yaml_path}")

        if cfg["use_k_list_from_yaml"]:
            k_list = [get_run_k_frac(r) for r in runs]
        else:
            k_list = list(cfg["manual_k_fracs"])

        # Build dict k->run for geo params
        run_by_k = {get_run_k_frac(r): r for r in runs}

        for method in cfg["methods"]:
            geo_modes = cfg["geo_modes"] if method == "geo" else ["-"]
            for geo_mode in geo_modes:
                print(f"\n--- Method={method}  GeoMode={geo_mode} ---")

                for k_frac in k_list:
                    k_frac = float(k_frac)
                    if method == "geo":
                        if k_frac not in run_by_k:
                            # if yaml had slightly different floats, try nearest
                            nearest = min(run_by_k.keys(), key=lambda kk: abs(kk - k_frac))
                            run_cfg = run_by_k[nearest]
                            k_used = nearest
                        else:
                            run_cfg = run_by_k[k_frac]
                            k_used = k_frac

                        center, radius = get_run_center_radius(run_cfg)
                        # safety: center dim must match pts_dim
                        if center.shape[0] != pts_dim:
                            raise ValueError(f"Center dim mismatch: {center.shape} vs pts_dim={pts_dim}")
                        feat_idx = geo_topk_indices(pts_fixed, center, radius, k_used, geo_mode)

                    elif method == "kwta":
                        feat_idx = kwta_topk_indices(Xtr, k_frac)
                        radius = None
                        center = None
                        k_used = k_frac

                    elif method == "random":
                        feat_idx = random_topk_indices(fixed_perm, k_frac)
                        radius = None
                        center = None
                        k_used = k_frac

                    else:
                        raise ValueError(f"Unknown method: {method}")

                    # Train ridge on clean (train split), evaluate on clean val
                    scaler, ridge, clean_val_acc = train_ridge_on_clean(
                        Xtr, ytr, Xva, yva, feat_idx, alpha=cfg["ridge_alpha"]
                    )

                    # Eval CIFAR-10-C
                    details = {}
                    for c in tqdm(corruptions, desc=f"Seed {seed} | {method}/{geo_mode} | k={k_used:.2f}", leave=False):
                        pt_path = os.path.join(cfg["cifar10c_r50_dir"], f"{c}.pt")
                        Xc, yc = load_cifar10c_features(pt_path, cfg["severity"])
                        acc = eval_ridge_accuracy(ridge, scaler, Xc, yc, feat_idx)
                        details[c] = float(acc)

                    cifar10c_mean = float(np.mean(list(details.values()))) if details else 0.0

                    row = {
                        "seed": seed,
                        "method": method,
                        "geo_mode": geo_mode,
                        "k_frac": float(k_used),
                        "clean_val_acc": float(clean_val_acc),
                        "cifar10c_mean_acc": float(cifar10c_mean),
                        "pts_dim": int(pts_dim),
                        "ridge_alpha": float(cfg["ridge_alpha"]),
                        "val_ratio": float(cfg["val_ratio"]),
                        "severity": int(cfg["severity"]),
                        "results_yaml": results_yaml_path,
                        "pts_fixed": pts_path,
                    }
                    all_rows.append(row)

                    key = (method, geo_mode, float(k_used))
                    acc_pool.setdefault(key, []).append(float(cifar10c_mean))

                # Save per-seed per-method yaml (optional, but helpful)
                out_seed_yaml = os.path.join(
                    cfg["out_dir"],
                    f"cifar10c_eval_{cfg['out_tag']}_seed{seed}_{method}_{geo_mode}.yaml"
                )
                payload = {
                    "meta": {
                        "seed": seed,
                        "method": method,
                        "geo_mode": geo_mode,
                        "severity": int(cfg["severity"]),
                        "val_ratio": float(cfg["val_ratio"]),
                        "ridge_alpha": float(cfg["ridge_alpha"]),
                        "pts_fixed": pts_path,
                        "results_yaml": results_yaml_path,
                        "eval_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "note": "Forced Top-k masking; ridge trained on clean train-split; evaluated on CIFAR-10-C offline features.",
                    },
                    "rows": [r for r in all_rows if r["seed"] == seed and r["method"] == method and r["geo_mode"] == geo_mode],
                }
                save_yaml(payload, out_seed_yaml)

    # Save flat CSV for everything
    out_csv = os.path.join(cfg["out_dir"], f"cifar10c_eval_{cfg['out_tag']}_allseeds.csv")
    header = ["seed", "method", "geo_mode", "k_frac", "clean_val_acc", "cifar10c_mean_acc",
              "pts_dim", "ridge_alpha", "val_ratio", "severity", "results_yaml", "pts_fixed"]
    save_csv(all_rows, out_csv, header)
    print("\nSaved CSV:", out_csv)

    # Summary across seeds: mean and 95% CI (normal approx)
    summary_rows = []
    for (method, geo_mode, k_frac), vals in sorted(acc_pool.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        v = np.asarray(vals, dtype=np.float64)
        mean = float(v.mean())
        if v.size >= 2:
            ci95 = float(1.96 * v.std(ddof=1) / np.sqrt(v.size))
        else:
            ci95 = 0.0
        summary_rows.append({
            "method": method,
            "geo_mode": geo_mode,
            "k_frac": float(k_frac),
            "mean_cifar10c_acc": mean,
            "ci95": ci95,
            "n_seeds": int(v.size),
        })

    out_summary_csv = os.path.join(cfg["out_dir"], f"cifar10c_eval_{cfg['out_tag']}_summary_mean_ci95.csv")
    sum_header = ["method", "geo_mode", "k_frac", "mean_cifar10c_acc", "ci95", "n_seeds"]
    save_csv(summary_rows, out_summary_csv, sum_header)
    print("Saved summary CSV:", out_summary_csv)


if __name__ == "__main__":
    main()
