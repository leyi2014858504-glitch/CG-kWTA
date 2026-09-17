#!/usr/bin/env python3
"""ImageNet-100 geo_pca dimension sweep.

Rationale (reviewer-aligned): dimension d is a free hyperparameter only for
projection-based coordinates (geo_r random projection, geo_pca). geo_m's
coordinates are hand-defined statistics (mean/std/rate), so d-sweeping it is
not a dimension-sensitivity question and is intentionally NOT run.

Small datasets already have full geo_r/geo_pca d sweeps (dim_sweep/, 60 combos).
The scale check was missing geo_pca entirely; this adds its d-sensitivity on
ImageNet-100 (orig + shuffled, 5 seeds, d in {1,2,3,5,8}). geo_r keeps its
existing d3 run; extend later only if reviewers push.

Each dim runs under its own out-root (filenames do not encode pd).
Resumable. ~3 h total on this machine.

Usage:
    python run_dimsweep_extra.py
    python run_dimsweep_extra.py --dims 1 2 8     # d3 exists via block A run
"""
import argparse
import os
import subprocess
import sys

BASE_DIR = r"d:\my projects\PythonProject4"
RUNNER = os.path.join(r"d:\cg-kwta", "run_imagenet100.py")
LOG_PATH = os.path.join(r"d:\cg-kwta", "dimsweep_extra.log")

LOG = open(LOG_PATH, "a", encoding="utf-8", buffering=1)


def log(msg):
    print(msg, flush=True)
    LOG.write(str(msg) + "\n")
    LOG.flush()


def run(cmd, label):
    log(f"=== {label} ===")
    log("  cmd: " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=BASE_DIR, stdout=LOG, stderr=subprocess.STDOUT,
                       text=True, encoding="utf-8", errors="replace")
    log(f"  exit={r.returncode}")
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", nargs="+", type=int, default=[1, 2, 3, 5, 8])
    args = ap.parse_args()

    log("")
    log("#" * 60)
    log(f"ImageNet-100 geo_pca d-sweep started (dims={args.dims})")

    for d in args.dims:
        run([sys.executable, RUNNER, "--tag", "imagenet100_r50",
             "--coords", "geo_pca", "--seeds", "5", "--pd", str(d),
             "--out-root", f"imagenet100_r50_pca_d{d}"],
            f"imagenet100 geo_pca d={d}")

    log("ImageNet-100 geo_pca d-sweep finished.")


if __name__ == "__main__":
    main()
