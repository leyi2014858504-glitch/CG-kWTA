#!/usr/bin/env python3
"""Selector-geometry ablation under the NEW configuration (sigma0=0.5, 20 k).

The paper's appendix (app:selectors) shows L1/shell vs L2-center only on the
March-era setup (sigma0=0.2, 6 k points, CIFAR-10). Reviewers may ask whether
"gap holds across selector families" survives the new configuration and the
scale-up. This reruns the two alternative selectors with current settings:

    shell       gate = k neurons with | ||p-c|| - r | smallest  (radius active)
    center_l1   gate = k neurons with smallest L1 distance to c
    (center/L2 = main method, already archived everywhere)

Each selector runs under its own out-root because result filenames do not
encode GEOSCORE_MODE. Single permutation null per seed (appendix-level claim;
the multi-permutation requirement applies to the main table).

Combos:
  imagenet100_r50  geo_r + geo_m   5 seeds   (headline scale result)
  cifar10_r50      geo_m            10 seeds  (main effective config)

Usage:
    python run_geometry_ablation.py
    python run_geometry_ablation.py --only shell
"""
import argparse
import os
import subprocess
import sys

BASE_DIR = r"d:\my projects\PythonProject4"
RUNNER = os.path.join(r"d:\cg-kwta", "run_imagenet100.py")
LOG_PATH = os.path.join(r"d:\cg-kwta", "geometry_ablation.log")

SELECTORS = ["shell", "center_l1"]

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
    ap.add_argument("--only", nargs="+", default=SELECTORS, choices=SELECTORS)
    args = ap.parse_args()

    log("")
    log("#" * 60)
    log(f"Geometry ablation started (selectors={args.only})")

    for sel in args.only:
        # ImageNet-100, both coords, 5 seeds
        run([sys.executable, RUNNER,
             "--tag", "imagenet100_r50", "--coords", "geo_r", "geo_m",
             "--seeds", "5", "--geoscore", sel,
             "--out-root", f"imagenet100_r50_{sel}"],
            f"{sel}: imagenet100_r50 geo_r+geo_m")
        # CIFAR-10 r50 geo_m, 10 seeds. run_imagenet100.py takes DATASET via
        # --tag, so the same runner works for cifar10_r50 (identity backbone).
        run([sys.executable, RUNNER,
             "--tag", "cifar10_r50", "--coords", "geo_m",
             "--seeds", "10", "--geoscore", sel,
             "--out-root", f"cifar10_r50_{sel}"],
            f"{sel}: cifar10_r50 geo_m")

    log("Geometry ablation finished.")


if __name__ == "__main__":
    main()
