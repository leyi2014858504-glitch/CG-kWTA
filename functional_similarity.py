#!/usr/bin/env python3
"""Independent functional-similarity measure vs coordinate distance
(Reviewer #3.1, #4.2: "use the same framework to define and evaluate";
"add an independent measure of functional similarity versus coordinate distance").

Readout-free: every quantity comes from raw cached activations, so the measure
is independent of the CG-kWTA probe.

Definitions
-----------
Functional profile of neuron i: its z-scored activation vector over N_cal
calibration samples. Functional similarity of a pair (i,j) is the Pearson
correlation of their profiles, S_ij = corr(a_i, a_j)  (co-activation).
Coordinate distance: D_ij = || p_i - p_j ||_2 for the constructed coordinates.

Per-seed replicates
-------------------
Everything is repeated over seeds. Seed sd controls, exactly as in the main
pipeline: (i) the calibration subsample (4000 rows drawn with generator seed sd),
(ii) the coordinate file (pts_fixed_seed{sd}), (iii) the archived optimized
centers (results_..._sd{sd}.yaml). Reported values are mean +/- sd across seeds.

Tests
-----
A  Spearman(D_ij, S_ij) over sampled pairs (negative = closer => more similar),
   with a row-permutation null; repeated within importance quintiles to control
   for per-neuron response magnitude.
B  Internal cohesion of the SELECTED top-k set (nearest to the archived center),
   compared against (i) uniform random subsets and (ii) importance-matched
   random subsets that reproduce the selected set's per-quintile composition.

Usage:
    python functional_similarity.py
    python functional_similarity.py --only "cifar10 geo_m" --seeds 3
"""
import argparse
import os

import numpy as np
import torch
import yaml

BASE = r"d:\my projects\PythonProject4"
DATA = os.path.join(BASE, "data")
REPORTS = r"d:\cg-kwta\Reports"

CONFIGS = [
    # label, feature tag, pts template, yaml template, n_seeds
    ("cifar10 geo_m", "cifar10_r50",
     r"cifar10_r50_meanstdrate\pts_fixed_seed{sd}_meanvar.pt",
     r"cifar10_r50_meanstdrate\results_cifar10_r50_geo_m_sd{sd}.yaml", 10),
    ("cifar10 geo_r", "cifar10_r50",
     r"rebuttal\cifar10_r50_geo_r\pts_fixed_seed{sd}_randproj.pt",
     r"rebuttal_dim\cifar10_r50_d3\results_cifar10_r50_geo_r_sigma0.50_sd{sd}.yaml", 10),
    ("cifar10 geo_pca", "cifar10_r50",
     r"cifar10_r50_pca\pts_fixed_seed{sd}_pca.pt",
     r"dim_sweep\geo_pca\cifar10_r50_d3\results_cifar10_r50_geo_pca_sd{sd}.yaml", 10),
    ("imagenet100 geo_m", "imagenet100_r50",
     r"imagenet100_r50_fixed_geo_m\pts_fixed_seed{sd}_meanvar.pt",
     r"imagenet100_r50_fixed_geo_m\results_imagenet100_r50_geo_m_sd{sd}.yaml", 10),
    ("imagenet100 geo_r", "imagenet100_r50",
     r"imagenet100_r50_fixed_geo_r\pts_fixed_seed{sd}_randproj.pt",
     r"imagenet100_r50_fixed_geo_r\results_imagenet100_r50_geo_r_sigma0.50_sd{sd}.yaml", 10),
    ("imagenet100 geo_pca", "imagenet100_r50",
     r"imagenet100_r50_pca_d3_geo_pca\pts_fixed_seed{sd}_pca.pt",
     r"imagenet100_r50_pca_d3_geo_pca\results_imagenet100_r50_geo_pca_sd{sd}.yaml", 10),
]

K_FRACS = [0.05, 0.10, 0.20]


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else np.nan


def calib_slice(X, n_cal, sd):
    """Row subsample, matching the pipeline's per-seed calibration draw."""
    n = X.shape[0]
    g = torch.Generator().manual_seed(sd)
    idx = torch.randperm(n, generator=g)[:min(n_cal, n)].sort().values
    A = np.asarray(X[idx], dtype=np.float32)
    return (A - A.mean(0, keepdims=True)) / (A.std(0, keepdims=True) + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cal", type=int, default=4000)
    ap.add_argument("--pairs", type=int, default=300000)
    ap.add_argument("--n-perm", type=int, default=999)
    ap.add_argument("--n-rand-sets", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=None, help="cap seeds per config")
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--inputs", choices=["calib", "test"], default="calib",
                    help="calib = the 4000-row calibration draw that also builds "
                         "the coordinates (shared source); test = all held-out "
                         "evaluation images, which never enter coordinate "
                         "construction or the gate search (label-free)")
    ap.add_argument("--out-name", default=None,
                    help="override the CSV basename under Reports/")
    args = ap.parse_args()

    per_seed_rows = []
    for label, tag, pts_t, yml_t, n_seeds in CONFIGS:
        if args.only and label not in args.only:
            continue
        if args.seeds:
            n_seeds = min(n_seeds, args.seeds)
        path = os.path.join(DATA, f"{tag}_train.pt")
        try:
            pack = torch.load(path, map_location="cpu", mmap=True)
        except TypeError:
            pack = torch.load(path, map_location="cpu")
        X = pack["X"]
        Xte = None
        if args.inputs == "test":
            pte = os.path.join(DATA, f"{tag}_test.pt")
            if not os.path.exists(pte):
                print(f"  [skip] {label}: no held-out features at {pte}")
                continue
            Xte = torch.load(pte, map_location="cpu")["X"]

        print(f"\n{'='*78}\n{label}  (seeds 0..{n_seeds-1})\n{'='*78}", flush=True)
        for sd in range(n_seeds):
            pts_p = os.path.join(BASE, pts_t.format(sd=sd))
            yml_p = os.path.join(BASE, yml_t.format(sd=sd))
            if not (os.path.exists(pts_p) and os.path.exists(yml_p)):
                print(f"  sd{sd}: missing artifact, skip")
                continue
            if args.inputs == "test":
                # Held-out evaluation images: never used for coordinates or gates.
                # Profiles (and hence S) are then identical across seeds; the
                # seed-to-seed spread reflects coordinate construction only.
                A = np.asarray(Xte, dtype=np.float32)
                A = (A - A.mean(0, keepdims=True)) / (A.std(0, keepdims=True) + 1e-6)
            else:
                A = calib_slice(X, args.n_cal, sd)
            N, H = A.shape
            pts = torch.load(pts_p, map_location="cpu")["pts_fixed"].float().numpy()
            d = pts.shape[1]

            S = (A.T @ A) / (N - 1)
            sq = (pts ** 2).sum(1)
            D = np.sqrt(np.clip(sq[:, None] + sq[None, :] - 2.0 * (pts @ pts.T), 0, None))
            imp = np.abs(A).mean(0)

            rng = np.random.default_rng(1000 + sd)
            iu = np.triu_indices(H, k=1)
            m = len(iu[0])
            sel = rng.choice(m, size=min(args.pairs, m), replace=False)
            I, J = iu[0][sel], iu[1][sel]
            d_ij, s_ij = D[I, J], S[I, J]
            rho = spearman(d_ij, s_ij)

            null = np.empty(args.n_perm)
            for t in range(args.n_perm):
                perm = rng.permutation(H)
                # row permutation of the points: D'[i,j] = D[perm[i], perm[j]],
                # evaluated only on the sampled pairs
                null[t] = spearman(D[perm[I], perm[J]], s_ij)
            p_two = (1.0 + float((np.abs(null) >= abs(rho)).sum())) / (1.0 + len(null))
            p_one = (1.0 + float((null <= rho).sum())) / (1.0 + len(null))

            pmean = (imp[I] + imp[J]) / 2.0
            qs = np.quantile(pmean, [0.2, 0.4, 0.6, 0.8])
            bins = np.digitize(pmean, qs)
            strata = [spearman(d_ij[bins == b], s_ij[bins == b])
                      for b in range(5) if (bins == b).sum() > 1000]

            q_edges = np.quantile(imp, [0.2, 0.4, 0.6, 0.8])
            lab = np.digitize(imp, q_edges)
            pools = [np.where(lab == b)[0] for b in range(5)]

            y = yaml.safe_load(open(yml_p, encoding="utf-8"))
            centers = {round(float(r["k_frac"]), 2): r.get("best_center")
                       for r in y.get("runs", [])}
            for kf in K_FRACS:
                c = centers.get(kf)
                if c is None:
                    continue
                c = np.asarray(c, dtype=np.float32)
                kk = max(2, int(H * kf))
                sel_idx = np.argsort(np.sqrt(((pts - c.reshape(1, -1)) ** 2).sum(1)))[:kk]
                sub = S[np.ix_(sel_idx, sel_idx)]
                kiu = np.triu_indices(kk, k=1)
                coh = float(sub[kiu].mean())

                rnd = np.array([float(S[np.ix_(r, r)][np.triu_indices(kk, k=1)].mean())
                                for r in (rng.choice(H, size=kk, replace=False)
                                          for _ in range(args.n_rand_sets))])
                cnt = np.bincount(lab[sel_idx], minlength=5)
                mat = []
                for _ in range(args.n_rand_sets):
                    r = np.concatenate([rng.choice(pools[b], size=cnt[b], replace=False)
                                        for b in range(5) if cnt[b] > 0])
                    ii = np.triu_indices(len(r), k=1)
                    mat.append(float(S[np.ix_(r, r)][ii].mean()))
                mat = np.array(mat)

                per_seed_rows.append({
                    "config": label, "coord_dim": d, "seed": sd, "k_frac": kf,
                    "spearman_D_S": rho, "perm_null_mean": float(null.mean()),
                    "perm_null_sd": float(null.std(ddof=1)),
                    "perm_p": p_two, "perm_p_one_sided": p_one,
                    "n_perm": int(args.n_perm), "n_inputs": int(N),
                    "n_pairs": int(len(I)), "inputs": args.inputs,
                    "within_imp_mean_rho": float(np.mean(strata)),
                    "cohesion_selected": coh,
                    "cohesion_random_mean": float(rnd.mean()),
                    "cohesion_random_sd": float(rnd.std()),
                    "cohesion_z": float((coh - rnd.mean()) / (rnd.std() + 1e-12)),
                    "cohesion_matched_mean": float(mat.mean()),
                    "cohesion_matched_sd": float(mat.std()),
                    "cohesion_matched_z": float((coh - mat.mean()) / (mat.std() + 1e-12)),
                    "imp_selected": float(imp[sel_idx].mean()), "imp_all": float(imp.mean()),
                })
            print(f"  sd{sd}: rho={rho:+.4f}  within-imp rho={np.mean(strata):+.4f}", flush=True)
            del A, S, D

    import pandas as pd
    df = pd.DataFrame(per_seed_rows)
    os.makedirs(REPORTS, exist_ok=True)
    out = os.path.join(REPORTS, args.out_name or (
        "functional_similarity_vs_coord_distance.csv" if args.inputs == "calib"
        else "functional_similarity_vs_coord_distance_heldout.csv"))
    df.to_csv(out, index=False)
    print(f"\nsaved {out}  ({len(df)} per-seed rows, inputs={args.inputs}, "
          f"n_perm={args.n_perm})")

    # ---- aggregate: mean +/- sd across seeds ----
    print("\n=== mean +/- sd across seeds ===")
    print(f"{'config':<22} {'k':>5} {'rho_D_S':>16} {'within-imp rho':>16} "
          f"{'cohesion z (matched)':>22} {'coh sel':>16}")
    for (cfg, kf), g in df.groupby(["config", "k_frac"], sort=False):
        r = g["spearman_D_S"]
        w = g["within_imp_mean_rho"]
        z = g["cohesion_matched_z"]
        c = g["cohesion_selected"]
        print(f"{cfg:<22} {kf:>5.2f} {r.mean():>+9.4f}+-{r.std(ddof=1):<6.4f} "
              f"{w.mean():>+9.4f}+-{w.std(ddof=1):<6.4f} "
              f"{z.mean():>+9.2f}+-{z.std(ddof=1):<6.2f}    "
              f"{c.mean():>+8.4f}+-{c.std(ddof=1):<6.4f}")


if __name__ == "__main__":
    main()