#!/usr/bin/env python3
"""ImageNet-100 FIX + T1/T2 driver.

Context: every earlier ImageNet-100 run is INVALID — `imagenet100_r50` was
missing from the main script's `_cached_feature_datasets`, so backbone_type
fell back to "mlp" and the gate selected 512 MLP hidden units instead of the
2048 frozen ResNet-50 neurons (see total_neurons=512 in those yamls, vs 2048
for every other r50 dataset). That bug is now fixed, so everything is re-run.

Steps
  A. T2 features: deterministic 45k-row subsample of the cached ImageNet-100
     train features -> data/imagenet100_r50_sub45k_{train,test}.pt
  B. corrected baseline: geo_r + geo_m, 5 seeds, maxiter=20, cma_train_n=4000
     -> imagenet100_r50_fixed_{geo_r,geo_m}
  C. T1 features: re-extract with images downscaled to 96x96 (matching STL-10's
     effective detail) -> data/imagenet100_r50_ds96_{train,test}.pt
  D. T1 runs  -> imagenet100_r50_ds96_{geo_r,geo_m}
  E. T2 runs  -> imagenet100_r50_sub45k_{geo_r,geo_m}

Resumable: feature files and completed seed yamls are skipped.

Usage:
    python run_imagenet100_fix.py
    python run_imagenet100_fix.py --only B
"""
import argparse
import os
import subprocess
import sys

import torch

BASE_DIR = r"d:\my projects\PythonProject4"
DATA_DIR = os.path.join(BASE_DIR, "data")
RUNNER = os.path.join(r"d:\cg-kwta", "run_imagenet100.py")
EXTRACT = os.path.join(r"d:\cg-kwta", "extract_imagenet100_r50.py")
LOG_PATH = os.path.join(r"d:\cg-kwta", "imagenet100_fix.log")

T2_TAG = "imagenet100_r50_sub45k"
T1_TAG = "imagenet100_r50_ds96"
T2_N = 45000
T1_DOWNSCALE = 96

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


def feats_exist(tag):
    return (os.path.exists(os.path.join(DATA_DIR, f"{tag}_train.pt"))
            and os.path.exists(os.path.join(DATA_DIR, f"{tag}_test.pt")))


def build_t2_features():
    """Deterministic subsample of the cached train features (seed 0)."""
    if feats_exist(T2_TAG):
        log(f"[skip] {T2_TAG} features already present")
        return True
    src = os.path.join(DATA_DIR, "imagenet100_r50_train.pt")
    src_test = os.path.join(DATA_DIR, "imagenet100_r50_test.pt")
    if not os.path.exists(src):
        log(f"[abort] missing {src}")
        return False
    log(f"=== A. build {T2_TAG} features ({T2_N} train rows, seed 0) ===")
    try:
        pack = torch.load(src, map_location="cpu", mmap=True)
    except TypeError:
        pack = torch.load(src, map_location="cpu")
    X, y = pack["X"], pack["y"]
    n = X.shape[0]
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(n, generator=g)[:min(T2_N, n)]
    idx = idx.sort().values
    out_train = os.path.join(DATA_DIR, f"{T2_TAG}_train.pt")
    torch.save({"X": X[idx].contiguous(), "y": y[idx].contiguous()}, out_train)
    log(f"[saved] {out_train}  X={tuple(X[idx].shape)}")
    del X, y, pack
    # same test split as the parent features
    try:
        pte = torch.load(src_test, map_location="cpu", mmap=True)
    except TypeError:
        pte = torch.load(src_test, map_location="cpu")
    out_test = os.path.join(DATA_DIR, f"{T2_TAG}_test.pt")
    torch.save({"X": pte["X"].contiguous(), "y": pte["y"].contiguous()}, out_test)
    log(f"[saved] {out_test}  X={tuple(pte['X'].shape)}")
    return True


def extract_t1():
    if feats_exist(T1_TAG):
        log(f"[skip] {T1_TAG} features already present")
        return True
    return run([sys.executable, EXTRACT,
                "--tag", T1_TAG, "--downscale", str(T1_DOWNSCALE),
                "--data_root", DATA_DIR, "--out_dir", DATA_DIR,
                "--batch_size", "256", "--num_workers", "2"],
               f"C. T1 features ({T1_TAG}, downscale={T1_DOWNSCALE})")


def run_coords(tag, out_root, label):
    return run([sys.executable, RUNNER,
                "--tag", tag, "--coords", "geo_r", "geo_m",
                "--seeds", "5", "--out-root", out_root],
               label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=["A", "B", "C", "D", "E"],
                    choices=["A", "B", "C", "D", "E"])
    args = ap.parse_args()

    log("")
    log("#" * 60)
    log(f"ImageNet-100 FIX + T1/T2 started (only={args.only})")

    if "A" in args.only and not build_t2_features():
        log("[abort] step A failed")
        return
    if "B" in args.only:
        run_coords("imagenet100_r50", "imagenet100_r50_fixed",
                   "B. corrected baseline (identity backbone, 2048 neurons)")
    if "C" in args.only and not extract_t1():
        log("[abort] step C failed")
        return
    if "D" in args.only:
        run_coords(T1_TAG, T1_TAG, "D. T1 runs (96x96 detail)")
    if "E" in args.only:
        run_coords(T2_TAG, T2_TAG, "E. T2 runs (45k train)")

    log("ImageNet-100 FIX + T1/T2 finished.")


if __name__ == "__main__":
    main()
