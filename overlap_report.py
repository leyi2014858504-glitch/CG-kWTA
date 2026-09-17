"""Jaccard overlap between geometric selection sets and importance-ranked sets.

Motivation (Reviewer #4.2 / #3.3): a positive shuffle gap may indicate that
CG-kWTA finds a compact cluster of globally informative neurons rather than a
neighborhood of functionally similar ones. This test asks how much the
geometry-selected set overlaps the sets produced by the importance rankings used
as baselines (magnitude, variance, rate, ridge probe weight).

The archived YAML files record the optimized ``best_center`` but not the realized
selected indices, so the geometric selection is rebuilt exactly from the archived
center and the archived coordinate matrix (``pts_fixed``):

    d_i = ||p_i - c||_2,   geo_sel = argtop-k(-d)

Importance rankings are rebuilt from the cached backbone features. Two sample
budgets are reported because the baselines rank on the calibration subset while
the cached feature file holds the full training split:

  calib = full_train   rankings over every cached training sample
  calib = n4000_seed   rankings over a 4000-sample calibration draw per seed
                       (matches the calibration budget of the archived runs)

Baseline for the table: the expected Jaccard index of two independent uniform
k-subsets of H neurons is k / (2H - k), reported in ``jaccard_null_expected``.

Coordinate files and archived centers are taken from the same directories as
``functional_similarity.py`` so that the two reports describe identical runs.

Writes Reports/jaccard_overlap.csv
"""
import os

import numpy as np
import pandas as pd
import torch
import yaml

BASE = r"d:\my projects\PythonProject4"
DATA = os.path.join(BASE, "data")
OUT = r"d:\cg-kwta\Reports\jaccard_overlap.csv"

CALIB_N = 4000
KS = [0.05, 0.10, 0.20, 0.50]
CHUNK = 8192

# dataset -> (feature file stem, list of (arm, pts template, yaml templates, n_seeds))
CONFIGS = {
    "cifar10": ("cifar10_r50", [
        ("geo_m",
         r"cifar10_r50_meanstdrate\pts_fixed_seed{sd}_meanvar.pt",
         [r"cifar10_r50_meanstdrate\results_cifar10_r50_geo_m_sd{sd}.yaml"], 10),
        ("geo_r",
         r"rebuttal\cifar10_r50_geo_r\pts_fixed_seed{sd}_randproj.pt",
         [r"rebuttal_dim\cifar10_r50_d3\results_cifar10_r50_geo_r_sigma0.50_sd{sd}.yaml",
          r"rebuttal_dim\cifar10_r50_d3\results_cifar10_r50_geo_r_sd{sd}.yaml"], 10),
        ("geo_pca",
         r"cifar10_r50_pca\pts_fixed_seed{sd}_pca.pt",
         [r"dim_sweep\geo_pca\cifar10_r50_d3\results_cifar10_r50_geo_pca_sd{sd}.yaml"], 10),
    ]),
    "imagenet100": ("imagenet100_r50", [
        ("geo_m",
         r"imagenet100_r50_fixed_geo_m\pts_fixed_seed{sd}_meanvar.pt",
         [r"imagenet100_r50_fixed_geo_m\results_imagenet100_r50_geo_m_sd{sd}.yaml"], 5),
        ("geo_r",
         r"imagenet100_r50_fixed_geo_r\pts_fixed_seed{sd}_randproj.pt",
         [r"imagenet100_r50_fixed_geo_r\results_imagenet100_r50_geo_r_sigma0.50_sd{sd}.yaml",
          r"imagenet100_r50_fixed_geo_r\results_imagenet100_r50_geo_r_sd{sd}.yaml"], 5),
        ("geo_pca",
         r"imagenet100_r50_pca_d3_geo_pca\pts_fixed_seed{sd}_pca.pt",
         [r"imagenet100_r50_pca_d3_geo_pca\results_imagenet100_r50_geo_pca_sd{sd}.yaml"], 5),
    ]),
}
REFERENCE_MODES = ["magnitude", "variance", "rate", "probe_weight"]


def first_existing(patterns, sd):
    for pat in patterns:
        p = os.path.join(BASE, pat.format(sd=sd))
        if os.path.exists(p):
            return p
    return None


def load_features(stem):
    d = torch.load(os.path.join(DATA, stem + "_train.pt"), map_location="cpu")
    X = d["X"].float()
    y = d["y"].long()
    if y.dim() == 1:
        Y = torch.nn.functional.one_hot(y, int(y.max().item()) + 1).float()
    else:
        Y = y.float()
    return X, Y


def importance_scores(X, Y, alpha=1.0):
    """Per-neuron magnitude, variance, rate and ridge probe-weight scores."""
    H = X.shape[1]
    n = X.shape[0]
    s_abs = torch.zeros(H, dtype=torch.float64)
    s_sq = torch.zeros(H, dtype=torch.float64)
    s_pos = torch.zeros(H, dtype=torch.float64)
    s_sum = torch.zeros(H, dtype=torch.float64)
    XtX = torch.zeros(H, H, dtype=torch.float64)
    XtY = torch.zeros(H, Y.shape[1], dtype=torch.float64)

    for i in range(0, n, CHUNK):
        xb = X[i:i + CHUNK].double()
        Yb = Y[i:i + CHUNK].double()
        s_abs += xb.abs().sum(0)
        s_sq += xb.pow(2).sum(0)
        s_sum += xb.sum(0)
        s_pos += (xb > 0).double().sum(0)
        XtX += xb.T @ xb
        XtY += xb.T @ Yb

    mu = s_sum / n
    var = s_sq / n - mu ** 2
    sd = var.clamp_min(1e-12).sqrt()
    Ymu = Y.double().mean(0, keepdim=True)
    A_tA = (XtX - n * (mu.view(-1, 1) @ mu.view(1, -1))) / (sd.view(-1, 1) @ sd.view(1, -1))
    A_tYc = (XtY - n * (mu.view(-1, 1) @ Ymu)) / sd.view(-1, 1)
    G = A_tA + alpha * torch.eye(H, dtype=torch.float64)
    W = torch.linalg.solve(G, A_tYc)

    return {
        "magnitude": (s_abs / n).float(),
        "variance": var.float(),
        "rate": (s_pos / n).float(),
        "probe_weight": W.norm(dim=1).float(),
    }


rows = []
for dataset, (stem, arms) in CONFIGS.items():
    print(f"\n== {dataset} ({stem}) ==")
    X, Y = load_features(stem)
    H = X.shape[1]
    print(f"  features {tuple(X.shape)}  classes={Y.shape[1]}  neurons={H}")

    n_seeds = max(a[3] for a in arms)
    for calib in ["full_train"] + [f"n4000_seed{sd}" for sd in range(n_seeds)]:
        if calib == "full_train":
            Xc, Yc = X, Y
        else:
            sd = int(calib.replace("n4000_seed", ""))
            g = torch.Generator().manual_seed(sd)
            idx = torch.randperm(X.shape[0], generator=g)[:CALIB_N].sort().values
            Xc, Yc = X[idx], Y[idx]
        scores = importance_scores(Xc, Yc)
        print(f"  rankings from {calib} (n={Xc.shape[0]})")

        for arm, pts_pat, yaml_pats, arm_seeds in arms:
            for sd in range(arm_seeds):
                yp = first_existing(yaml_pats, sd)
                pp = first_existing([pts_pat], sd)
                if yp is None or pp is None:
                    print(f"  [skip] {arm} sd{sd}: missing {'yaml' if yp is None else 'pts'}")
                    continue
                pts = torch.load(pp, map_location="cpu")["pts_fixed"].float()
                y = yaml.safe_load(open(yp, encoding="utf-8"))
                for r in y["runs"]:
                    k = round(float(r["k_frac"]), 2)
                    if k not in KS or r.get("best_center") is None:
                        continue
                    c = torch.tensor(r["best_center"], dtype=torch.float32)
                    kn = max(1, int(pts.shape[0] * k))
                    d = torch.linalg.norm(pts - c, dim=1)
                    geo_sel = set(torch.topk(-d, k=kn).indices.tolist())
                    for ref in REFERENCE_MODES:
                        ref_sel = set(torch.topk(scores[ref], k=kn).indices.tolist())
                        rows.append(dict(
                            dataset=dataset,
                            arm=arm,
                            seed=sd,
                            k_frac=k,
                            k_neurons=kn,
                            reference=ref,
                            jaccard=len(geo_sel & ref_sel) / len(geo_sel | ref_sel),
                            jaccard_null_expected=kn / (2 * H - kn),
                            calib=calib,
                        ))
    del X, Y

df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df.to_csv(OUT, index=False)

print("\n== Jaccard(geo selection, importance ranking) ==")
print("   mean +/- sd over seeds, calib=full_train   [null in brackets]")
summary = (df[df.calib == "full_train"]
           .groupby(["dataset", "arm", "reference", "k_frac"])["jaccard"]
           .agg(["mean", "std"]).reset_index())
null = df.groupby("k_frac")["jaccard_null_expected"].first().to_dict()
hdr = f"{'dataset':<13}{'arm':<9}{'reference':<14}" + "".join(f"{'k=' + format(k, '.2f'):>16}" for k in KS)
print(hdr)
key = None
for _, r in summary.sort_values(["dataset", "arm", "reference", "k_frac"]).iterrows():
    gk = (r["dataset"], r["arm"], r["reference"])
    if gk != key:
        key = gk
        line = f"{r['dataset']:<13}{r['arm']:<9}{r['reference']:<14}"
    else:
        line = " " * 36
    print(line + f"{r['mean']:>9.3f}+/-{r['std']:.3f}")

print("\n  null (two uniform k-subsets): " + ", ".join(
    f"k={k:.2f}:{null[k]:.3f}" for k in KS))
print(f"\nsaved: {OUT}  ({len(df)} rows)")
