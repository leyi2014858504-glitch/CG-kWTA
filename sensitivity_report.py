#!/usr/bin/env python3
"""Isolated sensitivity summary (Reviewer #4.2).

For each variant of the knob under test, the data seed is held FIXED at 0, so
the train/val split, the CMA-ES trajectory, the shuffle permutation and the
final refit are identical across variants. Any variation in the table below is
therefore attributable to the knob alone, not to run-to-run seed noise.

Columns:
    gap_mean      mean over the 20 k-fracs of (original - shuffled) test accuracy
    gap_lowk      mean over the low-sparsity k-fracs (<= 0.30)
    gap_at_0.05   the smallest k-frac (most interpretable single point)
    orig_k05      absolute original test accuracy at k_frac = 0.05
    The *_sd columns are the spread ACROSS VARIANTS of the knob (not across seeds).

Usage:
    python sensitivity_report.py
"""
import glob
import os
import re

import numpy as np
import pandas as pd
import yaml

BASE = r"d:\my projects\PythonProject4"
REPORTS = r"d:\cg-kwta\Reports"

BLOCKS = [
    ("calibration split", "cifar10 / r50 / geo_m", "sens_calib_c10_geo_m"),
    ("calibration split", "imagenet100 / r50 / geo_m", "sens_calib_im100_geo_m"),
    ("calibration split", "imagenet100 / r50 / geo_r", "sens_calib_im100_geo_r"),
    ("projection seed", "cifar10 / r50 / geo_r", "sens_proj_c10_geo_r"),
    ("projection seed", "imagenet100 / r50 / geo_r", "sens_proj_im100_geo_r"),
]

LOWK = 0.30


def load_acc(folder):
    """Return {k_frac: (orig_acc, shuf_acc)} for one variant folder."""
    out = {}
    for p in glob.glob(os.path.join(folder, "results_*.yaml")):
        is_shuf = os.path.basename(p).endswith("_shuf_sigma0.50_sd0.yaml") or \
                  re.search(r"_shuf_sd0\.yaml$", os.path.basename(p))
        try:
            y = yaml.safe_load(open(p, encoding="utf-8"))
        except Exception as e:
            print(f"  [skip] {p}: {e}")
            continue
        for r in y.get("runs", []):
            kf, acc = r.get("k_frac"), r.get("test_acc")
            if kf is None or acc is None:
                continue
            cur = out.setdefault(round(float(kf), 2), [None, None])
            cur[1 if is_shuf else 0] = float(acc)
    return out


rows = []
for knob, label, prefix in BLOCKS:
    for v in range(5):
        folder = os.path.join(BASE, f"{prefix}_{v}")
        if not os.path.isdir(folder):
            print(f"[missing] {folder}")
            continue
        d = load_acc(folder)
        ks = sorted(k for k, ab in d.items() if ab[0] is not None and ab[1] is not None)
        if not ks:
            print(f"[empty] {folder}")
            continue
        gaps = np.array([d[k][0] - d[k][1] for k in ks])
        lowk = np.array([d[k][0] - d[k][1] for k in ks if k <= LOWK + 1e-9])
        rows.append({
            "knob": knob, "config": label, "value": v,
            "n_k": len(ks),
            "gap_mean": gaps.mean(), "gap_lowk": lowk.mean() if len(lowk) else np.nan,
            "gap_at_0.05": d[0.05][0] - d[0.05][1] if 0.05 in d else np.nan,
            "orig_k05": d[0.05][0] if 0.05 in d else np.nan,
            "orig_mean": float(np.mean([d[k][0] for k in ks])),
        })

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("no data")

pd.set_option("display.width", 200)
print("=== per-variant ===")
print(df.round(4).to_string(index=False))

print("\n=== spread ACROSS VARIANTS (knob varies, seed fixed at 0) ===")
hdr = (f"{'knob':<18} {'config':<26} {'gap_lowk mean+-sd':>22} "
       f"{'gap_mean mean+-sd':>22} {'gap@0.05 mean+-sd':>22} {'orig@0.05 mean+-sd':>22}")
print(hdr)
summary = []
for (knob, label), g in df.groupby(["knob", "config"], sort=False):
    r = {
        "knob": knob, "config": label, "n_variants": len(g),
        "gap_lowk_mean": g["gap_lowk"].mean(), "gap_lowk_sd": g["gap_lowk"].std(ddof=1),
        "gap_mean_mean": g["gap_mean"].mean(), "gap_mean_sd": g["gap_mean"].std(ddof=1),
        "gap_k05_mean": g["gap_at_0.05"].mean(), "gap_k05_sd": g["gap_at_0.05"].std(ddof=1),
        "orig_k05_mean": g["orig_k05"].mean(), "orig_k05_sd": g["orig_k05"].std(ddof=1),
        "gap_lowk_min": g["gap_lowk"].min(), "gap_lowk_max": g["gap_lowk"].max(),
    }
    summary.append(r)
    print(f"{knob:<18} {label:<26} "
          f"{r['gap_lowk_mean']:>+13.4f}+-{r['gap_lowk_sd']:<7.4f} "
          f"{r['gap_mean_mean']:>+13.4f}+-{r['gap_mean_sd']:<7.4f} "
          f"{r['gap_k05_mean']:>+13.4f}+-{r['gap_k05_sd']:<7.4f} "
          f"{r['orig_k05_mean']:>13.4f}+-{r['orig_k05_sd']:<7.4f}")

sm = pd.DataFrame(summary)
os.makedirs(REPORTS, exist_ok=True)
df.to_csv(os.path.join(REPORTS, "sensitivity_variants.csv"), index=False)
sm.to_csv(os.path.join(REPORTS, "sensitivity_summary.csv"), index=False)
print(f"\nsaved {REPORTS}\\sensitivity_variants.csv")
print(f"saved {REPORTS}\\sensitivity_summary.csv")