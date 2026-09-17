#!/usr/bin/env python3
"""One-shot ImageNet-100 pipeline for the scalability check (Reviewer R1.2).

Step 0  unpack archive.zip and rearrange into the ImageFolder layout the
        feature extractor expects:
            data/imagenet100/train/<wnid>/<file>.JPEG
            data/imagenet100/val/<wnid>/<file>.JPEG
        (the archive stores train.X1..X4 shards of 25 classes each, plus val.X)

Step 1  feature extraction with the frozen ResNet-50 backbone
        -> data/imagenet100_r50_{train,test}.pt
Step 2  geo_r and geo_m experiments (original + shuffled, 5 seeds) via
        run_imagenet100.py

Resumable: a sentinel file marks a finished extraction, and the feature
extractor / experiment runner skip work whose outputs already exist.

Usage:
    python run_imagenet100_pipeline.py
    python run_imagenet100_pipeline.py --skip-extract     # dataset already arranged
    python run_imagenet100_pipeline.py --coords geo_r     # single coord
"""
import argparse
import os
import shutil
import subprocess
import sys
import zipfile

BASE_DIR = r"d:\my projects\PythonProject4"
DATA_DIR = os.path.join(BASE_DIR, "data")
ARCHIVE = os.path.join(r"d:\cg-kwta", "archive.zip")
DS_ROOT = os.path.join(DATA_DIR, "imagenet100")
SENTINEL = os.path.join(DS_ROOT, "_EXTRACT_DONE")
RUNNER = os.path.join(r"d:\cg-kwta", "run_imagenet100.py")
FEATURE_SCRIPT = os.path.join(r"d:\cg-kwta", "extract_imagenet100_r50.py")
FEAT_TRAIN = os.path.join(DATA_DIR, "imagenet100_r50_train.pt")
FEAT_TEST = os.path.join(DATA_DIR, "imagenet100_r50_test.pt")
LOG_PATH = os.path.join(r"d:\cg-kwta", "imagenet100_pipeline.log")

LOG = open(LOG_PATH, "a", encoding="utf-8", buffering=1)


def log(msg):
    print(msg, flush=True)
    LOG.write(str(msg) + "\n")
    LOG.flush()


def target_rel(name):
    """Map an archive member to its destination relative to DS_ROOT."""
    parts = name.split("/")
    if len(parts) == 1 and name.lower().endswith(".json"):
        return name
    if len(parts) < 3:
        return None
    top = parts[0]
    if top.startswith("train."):
        return os.path.join("train", parts[1], parts[2])
    if top.startswith("val."):
        return os.path.join("val", parts[1], parts[2])
    return None


def extract_and_arrange():
    if os.path.exists(SENTINEL):
        log(f"[skip] extraction (sentinel present: {SENTINEL})")
        return True
    if not os.path.exists(ARCHIVE):
        log(f"[abort] archive not found: {ARCHIVE}")
        return False

    log(f"=== Step 0: extracting {ARCHIVE} -> {DS_ROOT} ===")
    os.makedirs(DS_ROOT, exist_ok=True)
    made_dirs = set()
    n_files = 0
    n_skip = 0
    with zipfile.ZipFile(ARCHIVE) as z:
        infos = z.infolist()
        total = len(infos)
        log(f"  entries: {total}  (compressed {os.path.getsize(ARCHIVE)/1e9:.2f} GB)")
        for info in infos:
            if info.is_dir():
                continue
            rel = target_rel(info.filename)
            if rel is None:
                n_skip += 1
                continue
            dst = os.path.join(DS_ROOT, rel)
            d = os.path.dirname(dst)
            if d not in made_dirs:
                os.makedirs(d, exist_ok=True)
                made_dirs.add(d)
            if os.path.exists(dst) and os.path.getsize(dst) == info.file_size:
                n_files += 1
                continue
            with z.open(info) as src, open(dst, "wb") as out:
                shutil.copyfileobj(src, out, length=1024 * 1024)
            n_files += 1
            if n_files % 5000 == 0:
                log(f"  ... {n_files}/{total} written")

    # verify layout
    tr = os.path.join(DS_ROOT, "train")
    va = os.path.join(DS_ROOT, "val")
    n_tr_classes = len([d for d in os.listdir(tr) if os.path.isdir(os.path.join(tr, d))]) if os.path.isdir(tr) else 0
    n_va_classes = len([d for d in os.listdir(va) if os.path.isdir(os.path.join(va, d))]) if os.path.isdir(va) else 0
    log(f"  written={n_files} skipped={n_skip} train_classes={n_tr_classes} val_classes={n_va_classes}")
    if n_tr_classes < 100 or n_va_classes < 100:
        log("[abort] unexpected class counts; expected 100/100")
        return False
    with open(SENTINEL, "w", encoding="utf-8") as f:
        f.write("done\n")
    log("  sentinel written; extraction complete.")
    return True


def ensure_features():
    """Feature extraction with the frozen ResNet-50 (torchvision IMAGENET1K_V2,
    locally cached -> no network). run_imagenet100.py skips its own (timm/HF
    based) extraction once these files exist."""
    missing = []
    if not os.path.exists(FEAT_TRAIN):
        missing.append("train")
    if not os.path.exists(FEAT_TEST):
        missing.append("test")
    if not missing:
        log("[skip] feature extraction (cached .pt present)")
        return True
    log(f"=== Step 1: ResNet-50 feature extraction (splits: {missing}) ===")
    cmd = [sys.executable, FEATURE_SCRIPT,
           "--data_root", DATA_DIR, "--out_dir", DATA_DIR,
           "--batch_size", "256", "--num_workers", "2",
           "--splits", *missing]
    log("  cmd: " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=BASE_DIR, stdout=LOG, stderr=subprocess.STDOUT,
                       text=True, encoding="utf-8", errors="replace")
    log(f"  exit={r.returncode}")
    ok = r.returncode == 0 and os.path.exists(FEAT_TRAIN) and os.path.exists(FEAT_TEST)
    if not ok:
        log("[abort] feature extraction failed.")
    return ok


def run_runner(coord):
    log(f"=== run_imagenet100.py --coord {coord} ===")
    cmd = [sys.executable, RUNNER, "--coord", coord, "--seeds", "5"]
    log("  cmd: " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=BASE_DIR, stdout=LOG, stderr=subprocess.STDOUT,
                       text=True, encoding="utf-8", errors="replace")
    log(f"  exit={r.returncode}")
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", nargs="+", default=["geo_r", "geo_m"])
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    log("")
    log("#" * 60)
    log("ImageNet-100 pipeline started")
    log(f"  coords={args.coords} skip_extract={args.skip_extract}")

    if args.skip_extract:
        log("[skip] extraction (requested)")
    elif not extract_and_arrange():
        log("[abort] step 0 failed.")
        return

    if not ensure_features():
        return

    for coord in args.coords:
        ok = run_runner(coord)
        if not ok:
            log(f"[warn] coord {coord} returned non-zero; continuing")

    log("ImageNet-100 pipeline finished.")


if __name__ == "__main__":
    main()
