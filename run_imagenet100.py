#!/usr/bin/env python3
"""ImageNet-100 scalability validation for Reviewer R1.2.

Pipeline:
  Step 1 (optional): extract cached features with the frozen ResNet-50 backbone
        from an ImageFolder ImageNet-100 subset -> data/imagenet100_r50_{train,test}.pt
  Step 2: run geo experiments (original + shuffled) on the cached features by
        generating temp scripts from the main script with DATASET="imagenet100_r50".

ImageNet-100 layout (download & arrange manually):
   d:\\my projects\\PythonProject4\\data\\imagenet100\\train\\<class>\\*.{jpg,jpeg,png}
   d:\\my projects\\PythonProject4\\data\\imagenet100\\val\\<class>\\*.{jpg,jpeg,png}

Resumable: completed seed yamls are skipped (checks 20-k_frac completeness).

Usage:
    python run_imagenet100.py                     # extract (if needed) + run geo_r
    python run_imagenet100.py --extract-only
    python run_imagenet100.py --coord geo_m
"""
import argparse
import os
import re
import sys
import subprocess
import yaml

BASE_DIR = r"d:\my projects\PythonProject4"
DATA_DIR = os.path.join(BASE_DIR, "data")
MAIN_SCRIPT = os.path.join(BASE_DIR, "sphere_kwta_dimension_resnet50_MAE.py")
EXTRACT_SCRIPT = os.path.join(BASE_DIR, "extract_features.py")

TAG = "imagenet100_r50"
DATASET_KEY = "imagenet100_r50"
# user can override either coord (geo_r or geo_m) for the scalability check
SUPPORTED_COORD = {"geo_r": "geo_r", "geo_m": "geo_m"}
SEED_START = 0
SEED_END = 4  # 5 seeds is enough for a scalability check


def is_yaml_complete(path, min_runs=20):
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return bool(d) and len(d.get("runs", [])) >= min_runs
    except Exception:
        return False


def all_seeds_done(out_folder, shuffle):
    sfx = "_shuf" if shuffle else ""
    return all(is_yaml_complete(os.path.join(
        out_folder, f"results_{TAG}_{coord}{sfx}_sd{sd}.yaml")) for sd in range(SEED_START, SEED_END + 1))


def feature_files_exist():
    return (os.path.exists(os.path.join(DATA_DIR, f"{TAG}_train.pt"))
            and os.path.exists(os.path.join(DATA_DIR, f"{TAG}_test.pt")))


def extract():
    print("=== Step 1: feature extraction (ImageNet-100, frozen ResNet-50) ===")
    cmd = [sys.executable, EXTRACT_SCRIPT,
           "--model", "resnet50_sup", "--tag", TAG,
           "--dataset", "imagenet100", "--data_dir", DATA_DIR,
           "--out_dir", DATA_DIR, "--batch_size", "256"]
    print("Running:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print("[FAIL] extraction:\n", r.stderr[-2000:])
        return False
    print(r.stdout[-1500:])
    return True


def run_experiment(coord, shuffle):
    exp_mode = SUPPORTED_COORD[coord]
    out_folder = os.path.join(BASE_DIR, f"{TAG}_{coord}")
    os.makedirs(out_folder, exist_ok=True)

    sfx = "_shuf" if shuffle else ""
    if all(is_yaml_complete(os.path.join(
            out_folder, f"results_{TAG}_{exp_mode}{sfx}_sd{sd}.yaml"))
            for sd in range(SEED_START, SEED_END + 1)):
        print(f"[skip] {coord}/shuf={shuffle}")
        return

    with open(MAIN_SCRIPT, "r", encoding="utf-8") as f:
        content = f.read()
    content = re.sub(r'EXP_MODE\s*=\s*"[^"]*"', f'EXP_MODE = "{exp_mode}"', content)
    content = re.sub(r'DATASET\s*=\s*"[^"]*"', f'DATASET = "{DATASET_KEY}"', content)
    content = re.sub(r'Shuffle_mode\s*=\s*(True|False)', f'Shuffle_mode = {shuffle}', content)
    content = re.sub(r'^SEED_START\s*=\s*\d+', f'SEED_START = {SEED_START}', content, flags=re.M)
    content = re.sub(r'^SEED_END\s*=\s*\d+', f'SEED_END = {SEED_END}', content, flags=re.M)
    data_dir_escaped = DATA_DIR.replace("\\", "\\\\")
    content = re.sub(r'DATA_DIR\s*=\s*"[^"]*"', f'DATA_DIR = r"{data_dir_escaped}"', content)
    base_escaped = BASE_DIR.replace("\\", "\\\\")
    content = f'import sys; sys.path.insert(0, r"{base_escaped}")\n' + content

    temp_script = os.path.join(out_folder, "temp_script_imagenet100.py")
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(content)
    try:
        r = subprocess.run([sys.executable, temp_script], cwd=out_folder,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(f"[FAIL] {coord}/shuf={shuffle}\n{r.stderr[-1500:]}")
        else:
            print(f"[OK] {coord}/shuf={shuffle}")
    finally:
        if os.path.exists(temp_script):
            os.remove(temp_script)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coord", default="geo_r", choices=list(SUPPORTED_COORD))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--extract-only", action="store_true")
    args = ap.parse_args()

    global SEED_END
    SEED_END = max(SEED_START, args.seeds - 1)

    if not feature_files_exist() and not args.extract_only:
        if not extract():
            print("[abort] feature extraction failed; check ImageNet-100 layout.")
            return

    if args.extract_only:
        print("extraction requested; done.")
        return

    if not feature_files_exist():
        print("[abort] features missing:", f"{TAG}_train.pt / {TAG}_test.pt")
        return

    print(f"=== Step 2: experiment {args.coord} (5 seeds: {SEED_START}-{SEED_END}) ===")
    run_experiment(args.coord, shuffle=False)
    run_experiment(args.coord, shuffle=True)
    print("ImageNet-100 validation scheduled.")


if __name__ == "__main__":
    main()
