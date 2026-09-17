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
SUPPORTED_COORD = {"geo_r": "geo_r", "geo_m": "geo_m", "geo_pca": "geo_pca"}
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


def result_name(exp_mode, sd, shuffle):
    """Actual output filenames produced by the main script differ per coord:
    geo_r carries a '_sigma0.50' segment, geo_m/geo_pca do not."""
    sfx = "_shuf" if shuffle else ""
    if exp_mode == "geo_r":
        return f"results_{TAG}_geo_r{sfx}_sigma0.50_sd{sd}.yaml"
    return f"results_{TAG}_{exp_mode}{sfx}_sd{sd}.yaml"


def seed_done(out_folder, exp_mode, sd, shuffle, shuffle_runs):
    """Resume check. With shuffle_runs>1 the main script writes per-repeat files
    plus a '_multi_shuffle.yaml' index instead of a single yaml."""
    if shuffle and shuffle_runs > 1:
        sfx = "_shuf"
        tag = f"results_{TAG}_geo_r{sfx}_sigma0.50_sd{sd}" if exp_mode == "geo_r" \
            else f"results_{TAG}_{exp_mode}{sfx}_sd{sd}"
        if not os.path.exists(os.path.join(out_folder, f"{tag}_multi_shuffle.yaml")):
            return False
        return all(is_yaml_complete(os.path.join(out_folder, f"{tag}_rep{rep}.yaml"))
                   for rep in range(shuffle_runs))
    return is_yaml_complete(os.path.join(out_folder, result_name(exp_mode, sd, shuffle)))


def run_experiment(coord, shuffle, args):
    exp_mode = SUPPORTED_COORD[coord]
    # When out_root is given, keep coord in the folder name so multiple coords
    # in one invocation do not share a directory.
    folder_name = f"{args.out_root}_{coord}" if args.out_root else f"{TAG}_{coord}"
    out_folder = os.path.join(BASE_DIR, folder_name)
    os.makedirs(out_folder, exist_ok=True)

    todo = [sd for sd in range(SEED_START, SEED_END + 1)
            if not seed_done(out_folder, exp_mode, sd, shuffle, args.shuffle_runs)]
    if not todo:
        print(f"[skip] {coord}/shuf={shuffle}: all seeds done")
        return

    with open(MAIN_SCRIPT, "r", encoding="utf-8") as f:
        content = f.read()
    content = re.sub(r'EXP_MODE\s*=\s*"[^"]*"', f'EXP_MODE = "{exp_mode}"', content)
    content = re.sub(r'DATASET\s*=\s*"[^"]*"', f'DATASET = "{DATASET_KEY}"', content)
    content = re.sub(r'Shuffle_mode\s*=\s*(True|False)', f'Shuffle_mode = {shuffle}', content)
    content = re.sub(r'N_SHUFFLE_RUNS\s*=\s*\d+', f'N_SHUFFLE_RUNS = {args.shuffle_runs}', content)
    # Selector geometry: "center" (L2, main method), "shell", "center_l1".
    # Module-level constant; results filenames do NOT encode it, so each
    # selector MUST run under its own --out-root.
    content = re.sub(r'^\s*GEOSCORE_MODE\s*=\s*"[^"]*"',
                     f'GEOSCORE_MODE = "{args.geoscore}"', content, flags=re.M)
    # Coordinate dimension. Module-level `pd` (main script). For geo_m the
    # coordinate builder only distinguishes pd in {1,2,>=3}: pd=1 -> [mean],
    # pd=2 -> [mean,std], pd>=3 -> [mean,std,rate] (extra dims are dead). So a
    # geo_m "dimension sweep" is really a statistic ablation over pd=1/2/3.
    content = re.sub(r'^\s*pd\s*=\s*\d+', f'pd = {args.pd}', content, flags=re.M)
    # These constants are indented inside __main__; tolerate leading whitespace
    # or the substitution silently no-ops and the file's own values are used.
    content = re.sub(r'^\s*MAXITER\s*=\s*\d+', f'    MAXITER = {args.maxiter}', content, flags=re.M)
    content = re.sub(r'^\s*CMA_TRAIN_N\s*=\s*\d+', f'    CMA_TRAIN_N = {args.cma_train_n}', content, flags=re.M)
    content = re.sub(r'^\s*CMA_VAL_N\s*=\s*\d+', f'    CMA_VAL_N = {args.cma_val_n}', content, flags=re.M)
    # Run only the seeds still missing, one process per invocation.
    content = re.sub(r'^\s*SEED_START\s*=\s*\d+', f'    SEED_START = {min(todo)}', content, flags=re.M)
    content = re.sub(r'^\s*SEED_END\s*=\s*\d+', f'    SEED_END = {max(todo)}', content, flags=re.M)
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
    global SEED_END, TAG, DATASET_KEY

    ap = argparse.ArgumentParser()
    ap.add_argument("--coord", default="geo_r", choices=list(SUPPORTED_COORD))
    ap.add_argument("--coords", nargs="+", default=None,
                    help="run several coords in one go (overrides --coord)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--maxiter", type=int, default=20,
                    help="CMA-ES iterations per k-point (budget control)")
    ap.add_argument("--cma-train-n", type=int, default=4000,
                    help="samples used to calibrate coordinates and score each gate candidate")
    ap.add_argument("--cma-val-n", type=int, default=800)
    ap.add_argument("--shuffle-runs", type=int, default=1,
                    help="independent coordinate permutations per seed (null strength)")
    ap.add_argument("--out-root", default=None,
                    help="directory name under BASE_DIR for outputs. REQUIRED when "
                         "changing budget, because filenames do not encode maxiter "
                         "or cma_train_n (otherwise the baseline run is overwritten).")
    ap.add_argument("--tag", default=TAG,
                    help="feature tag / dataset key, e.g. imagenet100_r50_ds96 "
                         "(default: imagenet100_r50)")
    ap.add_argument("--geoscore", default="center",
                    choices=["center", "shell", "center_l1"],
                    help="selector geometry (main method = center/L2)")
    ap.add_argument("--pd", type=int, default=3,
                    help="coordinate dimension (geo_m: only 1/2/3 differ; "
                         "geo_r/geo_pca: 1..8 meaningful)")
    ap.add_argument("--extract-only", action="store_true")
    args = ap.parse_args()

    SEED_END = max(SEED_START, args.seeds - 1)
    TAG = args.tag
    DATASET_KEY = args.tag
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

    coords = args.coords or [args.coord]
    print(f"=== Step 2: experiment {coords} (seeds {SEED_START}-{SEED_END}, "
          f"maxiter={args.maxiter}, cma_train_n={args.cma_train_n}, "
          f"shuffle_runs={args.shuffle_runs}, out_root={args.out_root}) ===")
    for coord in coords:
        run_experiment(coord, shuffle=False, args=args)
        run_experiment(coord, shuffle=True, args=args)
    print("ImageNet-100 validation scheduled.")


if __name__ == "__main__":
    main()
