#!/usr/bin/env python3
"""Cost breakdown for Reviewer #4.5: wall-clock AND regression-solve costs.

The main script now records, per k-point, in addition to `elapsed_sec`:
    search_sec       time inside CmaEngine.f() = the search-time ridge solves
    n_search_solves  number of such solves
    search_frac      search_sec / elapsed_sec
The remainder of elapsed_sec is setup, coordinate construction/lookups, the
final refit on train+val and the one-shot test evaluation.

Runs a small set of representative settings in a sandbox-writable directory,
so the historical result folders are untouched. Outputs land in
Reports/cost_breakdown.csv.

Usage:
    python run_cost_breakdown.py
    python run_cost_breakdown.py --configs c10_r50_geo_m
"""
import argparse
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

BASE_DIR = r"d:\my projects\PythonProject4"
DATA_DIR = os.path.join(BASE_DIR, "data")
MAIN_SCRIPT = os.path.join(BASE_DIR, "sphere_kwta_dimension_resnet50_MAE.py")
WORK_ROOT = os.path.join(r"d:\cg-kwta", "_costbreak")
LOG_PATH = os.path.join(r"d:\cg-kwta", "cost_breakdown.log")

# key: (dataset tag, exp_mode, seed, maxiter, cma_train_n, cma_val_n, label)
CONFIGS = {
    "c10_r50_geo_m": ("cifar10_r50", "geo_m", 0, 20, 4000, 800,
                      "cifar10 / r50 / geo_m"),
    "c10_vit_geo_m": ("cifar10_vit", "geo_m", 0, 20, 4000, 800,
                      "cifar10 / vit / geo_m"),
    "c10_r50_geo_m_max40": ("cifar10_r50", "geo_m", 0, 40, 4000, 800,
                            "cifar10 / r50 / geo_m maxiter40"),
    "c10_r50_geo_m_cma20k": ("cifar10_r50", "geo_m", 0, 20, 20000, 4000,
                             "cifar10 / r50 / geo_m cma_train_n=20000"),
    "im100_r50_geo_m": ("imagenet100_r50", "geo_m", 0, 20, 4000, 800,
                        "imagenet100 / r50 / geo_m"),
}

LOG = open(LOG_PATH, "a", encoding="utf-8", buffering=1)


def log(msg):
    print(msg, flush=True)
    LOG.write(str(msg) + "\n")
    LOG.flush()


def make_temp(folder, tag, exp_mode, seed, maxiter, cma_train_n, cma_val_n):
    os.makedirs(folder, exist_ok=True)
    with open(MAIN_SCRIPT, encoding="utf-8") as f:
        content = f.read()
    content = re.sub(r'EXP_MODE\s*=\s*"[^"]*"', f'EXP_MODE = "{exp_mode}"', content)
    content = re.sub(r'DATASET\s*=\s*"[^"]*"', f'DATASET = "{tag}"', content)
    content = re.sub(r'Shuffle_mode\s*=\s*(True|False)', 'Shuffle_mode = False', content)
    content = re.sub(r'N_SHUFFLE_RUNS\s*=\s*\d+', 'N_SHUFFLE_RUNS = 1', content)
    content = re.sub(r'^\s*MAXITER\s*=\s*\d+', f'    MAXITER = {maxiter}', content, flags=re.M)
    content = re.sub(r'^\s*CMA_TRAIN_N\s*=\s*\d+', f'    CMA_TRAIN_N = {cma_train_n}', content, flags=re.M)
    content = re.sub(r'^\s*CMA_VAL_N\s*=\s*\d+', f'    CMA_VAL_N = {cma_val_n}', content, flags=re.M)
    content = re.sub(r'^\s*SEED_START\s*=\s*\d+', f'    SEED_START = {seed}', content, flags=re.M)
    content = re.sub(r'^\s*SEED_END\s*=\s*\d+', f'    SEED_END = {seed}', content, flags=re.M)
    dd = DATA_DIR.replace("\\", "\\\\")
    content = re.sub(r'DATA_DIR\s*=\s*"[^"]*"', f'DATA_DIR = r"{dd}"', content)
    bd = BASE_DIR.replace("\\", "\\\\")
    content = f'import sys; sys.path.insert(0, r"{bd}")\n' + content
    ts = os.path.join(folder, "temp_script_cost.py")
    with open(ts, "w", encoding="utf-8") as f:
        f.write(content)
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    args = ap.parse_args()

    rows = []
    for key in args.configs:
        if key not in CONFIGS:
            log(f"[warn] unknown config {key}")
            continue
        tag, exp_mode, seed, maxiter, ctn, cvn, label = CONFIGS[key]
        folder = os.path.join(WORK_ROOT, key)
        ts = make_temp(folder, tag, exp_mode, seed, maxiter, ctn, cvn)
        log(f"=== {label} (maxiter={maxiter}, cma_train_n={ctn}) ===")
        r = subprocess.run([sys.executable, ts], cwd=folder, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            log(f"[FAIL] {label}\n{(r.stderr or '')[-1200:]}")
            continue
        # collect the yaml just written
        cand = [p for p in os.listdir(folder) if p.startswith("results_") and p.endswith(".yaml")]
        if not cand:
            log(f"[FAIL] {label}: no yaml produced")
            continue
        y = yaml.safe_load(open(os.path.join(folder, cand[0]), encoding="utf-8"))
        el = [float(x["elapsed_sec"]) for x in y["runs"] if x.get("elapsed_sec") is not None]
        se = [float(x.get("search_sec", 0.0)) for x in y["runs"]]
        ns = [int(x.get("n_search_solves", 0)) for x in y["runs"]]
        el_a, se_a = np.array(el), np.array(se)
        rows.append({
            "config": label, "maxiter": maxiter, "cma_train_n": ctn,
            "n_k_points": len(el),
            "elapsed_mean_s": el_a.mean(), "elapsed_median_s": float(np.median(el_a)),
            "search_mean_s": se_a.mean(), "search_median_s": float(np.median(se_a)),
            "search_frac_mean": float((se_a / el_a).mean()),
            "n_search_solves": int(np.median(ns)) if ns else 0,
            "ms_per_search_solve": 1000 * se_a.mean() / max(1, int(np.median(ns))) if ns else np.nan,
            "rest_mean_s": float((el_a - se_a).mean()),
        })
        log(f"[OK] {label}: elapsed {el_a.mean():.2f}s, search {se_a.mean():.2f}s "
            f"({(se_a/el_a).mean()*100:.1f}%)")
        if os.path.exists(ts):
            os.remove(ts)

    if not rows:
        log("no rows produced")
        return
    df = pd.DataFrame(rows)
    os.makedirs(r"d:\cg-kwta\Reports", exist_ok=True)
    out = r"d:\cg-kwta\Reports\cost_breakdown.csv"
    df.to_csv(out, index=False)
    print("\n" + df.round(3).to_string(index=False))
    print(f"\nsaved {out}")
    log(f"saved {out}")


if __name__ == "__main__":
    main()