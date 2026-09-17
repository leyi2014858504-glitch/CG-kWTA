#!/usr/bin/env python3
"""Stage-3: information-vs-geometry controls on ImageNet-100 (fixed wiring).

Why: the shuffled-null gap proves the neuron<->coordinate PAIRING carries
information, but the pairing could act THROUGH per-neuron informativeness.
The four activation baselines (magnitude/variance/rate/probe_weight) use no
geometry at all; best-of-random uses neither geometry nor scores. If geo beats
them at matched k, geometry contributes beyond "picking informative neurons".

IMPORTANT: every existing imagenet100_r50_{baseline,bestof} folder was produced
during the 512-MLP bug (total_neurons=512) and is INVALID. Step 0 quarantines
them (rename with _buggy512 suffix) so the runners' resume checks do not
silently skip them.

Steps
  0  quarantine buggy baseline/bestof dirs
  1  4 informative baselines, 5 seeds (no search; fast)
  2  best-of-random N=160, 5 seeds (budget-matched to geo@maxiter20)

Run AFTER run_imagenet100_fix.py finishes (GPU/RAM contention), or any time
afterwards; resumable.

Usage:
    python run_imagenet100_stage3.py
"""
import argparse
import os
import shutil
import subprocess
import sys

import yaml

BASE_DIR = r"d:\my projects\PythonProject4"
BASELINES = os.path.join(r"d:\cg-kwta", "run_baselines_only.py")
BESTOF = os.path.join(r"d:\cg-kwta", "run_bestof_random.py")
LOG_PATH = os.path.join(r"d:\cg-kwta", "imagenet100_stage3.log")

MODES = ["magnitude", "variance", "rate", "probe_weight"]

LOG = open(LOG_PATH, "a", encoding="utf-8", buffering=1)


def log(msg):
    print(msg, flush=True)
    LOG.write(str(msg) + "\n")
    LOG.flush()


def is_buggy(path):
    """True if any result yaml in the folder reports total_neurons != 2048."""
    for fn in os.listdir(path):
        if fn.startswith("results_") and fn.endswith(".yaml"):
            try:
                d = yaml.safe_load(open(os.path.join(path, fn), encoding="utf-8"))
                runs = d.get("runs") or []
                if runs and runs[0].get("total_neurons") not in (2048,):
                    return True
            except Exception:
                pass
    return False


def quarantine():
    log("=== Step 0: quarantine 512-bug baseline/bestof dirs ===")
    targets = [f"imagenet100_r50_{m}" for m in MODES] + ["imagenet100_r50_bestof_random"]
    for name in targets:
        p = os.path.join(BASE_DIR, name)
        if not os.path.isdir(p):
            continue
        if is_buggy(p):
            dst = p + "_buggy512"
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.move(p, dst)
            log(f"  moved {name} -> {name}_buggy512")
        else:
            log(f"  kept {name} (looks fixed)")


def run(cmd, label):
    log(f"=== {label} ===")
    log("  cmd: " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=BASE_DIR, stdout=LOG, stderr=subprocess.STDOUT,
                       text=True, encoding="utf-8", errors="replace")
    log(f"  exit={r.returncode}")
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-quarantine", action="store_true")
    args = ap.parse_args()

    log("")
    log("#" * 60)
    log("ImageNet-100 stage-3 (information vs geometry) started")

    if not args.skip_quarantine:
        quarantine()

    ok = run([sys.executable, BASELINES,
              "--datasets", "imagenet100", "--models", "r50", "--seeds", "5"],
             "Step 1: informative baselines (fixed wiring)")
    if not ok:
        log("[warn] baselines failed; continuing to best-of-random")
    run([sys.executable, BESTOF,
         "--datasets", "imagenet100", "--models", "r50", "--seeds", "5"],
        "Step 2: best-of-random N=160 (fixed wiring)")

    log("ImageNet-100 stage-3 finished.")


if __name__ == "__main__":
    main()
