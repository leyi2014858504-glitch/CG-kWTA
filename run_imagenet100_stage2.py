#!/usr/bin/env python3
"""Stage-2 ImageNet-100 runs (deadline batch): P1 + P2.

P1  budget / data-starvation control
      geo_r + geo_m, 5 seeds, maxiter 20 -> 40 and CMA subsample 4000 -> 20000.
      Outputs go to separate roots (imagenet100_r50_max40_cma20k_{coord}) so the
      original maxiter=20 / cma_train_n=4000 evidence is NOT overwritten.
P2  seven-condition table at scale
      4 informative baselines (magnitude/variance/rate/probe_weight) and
      best-of-random (N = popsize * maxiter = 160, budget-matched to the
      baseline geo run) for imagenet100_r50, 5 seeds.

Everything is resumable: each underlying runner skips completed seeds/combos.

Usage:
    python run_imagenet100_stage2.py
    python run_imagenet100_stage2.py --only P1
"""
import argparse
import os
import subprocess
import sys

BASE_DIR = r"d:\my projects\PythonProject4"
RUNNER = os.path.join(r"d:\cg-kwta", "run_imagenet100.py")
BASELINES = os.path.join(r"d:\cg-kwta", "run_baselines_only.py")
BESTOF = os.path.join(r"d:\cg-kwta", "run_bestof_random.py")
LOG_PATH = os.path.join(r"d:\cg-kwta", "imagenet100_stage2.log")

P1_OUT_ROOT = "imagenet100_r50_max40_cma20k"

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


def step_p1():
    return run([sys.executable, RUNNER,
                "--coords", "geo_r", "geo_m",
                "--seeds", "5",
                "--maxiter", "40",
                "--cma-train-n", "20000",
                "--cma-val-n", "4000",
                "--out-root", P1_OUT_ROOT],
               "P1 budget/data control (maxiter=40, cma_train_n=20000)")


def step_p2_baselines():
    return run([sys.executable, BASELINES,
                "--datasets", "imagenet100",
                "--models", "r50",
                "--seeds", "5"],
               "P2a informative baselines (imagenet100_r50)")


def step_p2_bestof():
    return run([sys.executable, BESTOF,
                "--datasets", "imagenet100",
                "--models", "r50",
                "--seeds", "5"],
               "P2b best-of-random (imagenet100_r50)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=["P1", "P2"],
                    choices=["P1", "P2"])
    args = ap.parse_args()

    log("")
    log("#" * 60)
    log(f"ImageNet-100 stage-2 started (only={args.only})")

    if "P1" in args.only:
        step_p1()
    if "P2" in args.only:
        step_p2_baselines()
        step_p2_bestof()

    log("ImageNet-100 stage-2 finished.")


if __name__ == "__main__":
    main()
