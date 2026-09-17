"""Wall-clock aggregation for Reviewer #4.5.

Reports, per (dataset, model, coordinate), the end-to-end cost of ONE k-point
(one full pass: gate search + final ridge refit + test evaluation), taken from
the `elapsed_sec` field written by the main script.

Also reports the number of ridge solves implied by the search configuration,
so the per-solve cost can be read off the totals:
    N_solves_per_k = 1 (initial) + popsize*maxiter (search candidates)
                     + 1 (final refit on train+val) + 1 (baseline val eval)
with popsize = 4 + floor(3*ln(1+d)) from CMA-ES defaults.
"""
import glob
import math
import os

import numpy as np
import pandas as pd
import yaml

BASE = r"d:\my projects\PythonProject4"

# (label, folder, filename pattern, coord dim)
JOBS = [
    ("cifar10 / r50 / geo_m", "cifar10_r50_meanstdrate", "results_cifar10_r50_geo_m_sd*.yaml", 3),
    ("cifar10 / r50 / geo_r", "rebuttal_dim/cifar10_r50_d3", "results_cifar10_r50_geo_r_sigma0.50_sd*.yaml", 3),
    ("cifar10 / r50 / geo_pca", "dim_sweep/geo_pca/cifar10_r50_d3", "results_cifar10_r50_geo_pca_sd*.yaml", 3),
    ("cifar10 / vit / geo_m", "rebuttal_geo_m_orig/cifar10_vit", "results_cifar10_vit_geo_m_sd*.yaml", 3),
    ("cifar10 / vitmae / geo_r", "rebuttal_mae_verify/cifar10_vitmae", "results_cifar10_vit_mae_geo_r_sigma0.50_sd*.yaml", 3),
    ("stl10 / r50 / geo_m", "rebuttal_geo_m_orig/stl10_r50", "results_stl10_r50_geo_m_sd*.yaml", 3),
    ("imagenet100 / r50 / geo_m", "imagenet100_r50_fixed_geo_m", "results_imagenet100_r50_geo_m_sd*.yaml", 3),
    ("imagenet100 / r50 / geo_r", "imagenet100_r50_fixed_geo_r", "results_imagenet100_r50_geo_r_sigma0.50_sd*.yaml", 3),
    ("imagenet100 / r50 / geo_pca", "imagenet100_r50_pca_d3_geo_pca", "results_imagenet100_r50_geo_pca_sd*.yaml", 3),
    # NOTE: the imagenet100 maxiter=40 run belongs to the discarded 512-MLP era
    # (total_neurons=512) and its timings are NOT comparable. The valid
    # longer-budget run is the CIFAR-10 one below.
    ("cifar10 / r50 / geo_m maxiter40", "rebuttal_maxiter40/cifar10_r50_geo_m",
     "results_cifar10_r50_geo_m_sd*.yaml", 3),
]


def popsize(dim):
    return 4 + math.floor(3 * math.log(1 + dim))


def solves_per_k(dim, maxiter):
    return 1 + popsize(dim) * maxiter + 1 + 1


rows = []
for label, folder, pat, dim in JOBS:
    paths = sorted(glob.glob(os.path.join(BASE, folder, pat)))
    if not paths:
        continue
    per_k = []
    meta_seen = set()
    for p in paths:
        try:
            d = yaml.safe_load(open(p, encoding="utf-8"))
        except Exception:
            continue
        m = d.get("meta") or {}
        mi = int(m.get("maxiter", 20) or 20)
        meta_seen.add((dim, mi))
        for r in d.get("runs", []):
            if r.get("elapsed_sec") is not None:
                per_k.append(float(r["elapsed_sec"]))
    if not per_k:
        continue
    a = np.array(per_k)
    dim_s, mi_s = next(iter(meta_seen))
    n_solves = solves_per_k(dim_s, mi_s)
    rows.append({
        "config": label,
        "n_k_points": len(a),
        "maxiter": mi_s,
        "popsize": popsize(dim_s),
        "ridge_solves_per_k": n_solves,
        "mean_s": a.mean(),
        "median_s": float(np.median(a)),
        "sd_s": a.std(ddof=1) if len(a) > 1 else np.nan,
        "min_s": a.min(),
        "max_s": a.max(),
        "sum_hours": a.sum() / 3600,
        "ms_per_solve": 1000 * a.mean() / n_solves,
    })

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.round(3).to_string(index=False))
out = r"d:\cg-kwta\Reports\wallclock_timing.csv"
os.makedirs(os.path.dirname(out), exist_ok=True)
df.to_csv(out, index=False)
print(f"\nsaved {out}")
print(f"\ntotal measured wall-clock across the listed runs: {df['sum_hours'].sum():.2f} h")
print("\nNOTE: elapsed_sec covers ONE k-point end to end (search + final refit + "
      "test eval). ms_per_solve is an upper-bound average that also absorbs "
      "data movement and coordinate lookups, not just the linear solve.")