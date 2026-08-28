#!/usr/bin/env python3
"""Unified rebuttal-experiment scheduler.

Runs on a small representative subset, in one go:
  (A) Multi-shuffle geo runs   -> d:\\my projects\\PythonProject4\\rebuttal\\{ds}_{m}_{coord}\\
  (B) Coordinate-dimension d sweep (geo_r, original + shuffled)
                                -> d:\\my projects\\PythonProject4\\rebuttal_dim\\{ds}_{m}_d{d}\\

Every new run automatically records CMA-ES convergence history
(`cma_fbest_history` in each k_frac run). Resumable: completed combos are skipped.

Defaults are deliberately cheap (r50 + vit, cifar10 + stl10, geo_r + geo_m,
d in 1,2,3,5,8, 5 shuffle repeats). Override via CLI.

Usage:
    python run_rebuttal_experiments.py
    python run_rebuttal_experiments.py --models r50 vit vitmae --datasets cifar10 cifar100 stl10 --seeds 5
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

MODELS = {
    "r50": "r50",
    "r50mocov2": "mocov2",
    "convnext": "convnext_base_sup",
    "convnextv2": "convnextv2_base_mae",
    "vitmae": "vit_mae",
    "vit": "vit",
    "vitdino": "dino",
    "swin": "swin",
}
COORD_EXP = {"geo_r": "geo_r", "geo_m": "geo_m", "geo_pca": "geo_pca"}


def check_features(ds, tag):
    return (os.path.exists(os.path.join(DATA_DIR, f"{ds}_{tag}_train.pt"))
            and os.path.exists(os.path.join(DATA_DIR, f"{ds}_{tag}_test.pt")))


def make_temp(out_folder, exp_mode, ds, tag, shuffle, n_shuffle_runs, pd=None,
              seed_start=0, seed_end=9, maxiter=20):
    os.makedirs(out_folder, exist_ok=True)
    with open(MAIN_SCRIPT, "r", encoding="utf-8") as f:
        content = f.read()
    content = re.sub(r'EXP_MODE\s*=\s*"[^"]*"', f'EXP_MODE = "{exp_mode}"', content)
    content = re.sub(r'DATASET\s*=\s*"[^"]*"', f'DATASET = "{ds}_{tag}"', content)
    content = re.sub(r'Shuffle_mode\s*=\s*(True|False)', f'Shuffle_mode = {shuffle}', content)
    content = re.sub(r'N_SHUFFLE_RUNS\s*=\s*\d+', f'N_SHUFFLE_RUNS = {n_shuffle_runs}', content)
    content = re.sub(r'^MAXITER\s*=\s*\d+', f'MAXITER = {maxiter}', content, flags=re.M)
    content = re.sub(r'^SEED_START\s*=\s*\d+', f'SEED_START = {seed_start}', content, flags=re.M)
    content = re.sub(r'^SEED_END\s*=\s*\d+', f'SEED_END = {seed_end}', content, flags=re.M)
    if pd is not None:
        content = re.sub(r'^pd\s*=\s*\d+', f'pd = {pd}', content, flags=re.M)
    data_dir_escaped = DATA_DIR.replace("\\", "\\\\")
    content = re.sub(r'DATA_DIR\s*=\s*"[^"]*"', f'DATA_DIR = r"{data_dir_escaped}"', content)
    base_escaped = BASE_DIR.replace("\\", "\\\\")
    content = f'import sys; sys.path.insert(0, r"{base_escaped}")\n' + content

    temp_script = os.path.join(out_folder, "temp_script_rebuttal.py")
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(content)
    return temp_script


def run_script(temp_script, out_folder, label):
    try:
        r = subprocess.run([sys.executable, temp_script], cwd=out_folder,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(f"[FAIL] {label}\n{r.stderr[-1500:]}")
        else:
            print(f"[OK] {label}")
    finally:
        if os.path.exists(temp_script):
            os.remove(temp_script)


def is_yaml_complete(path, min_runs=20):
    """Resume safety: file exists AND contains all expected k_frac runs.
    (The main script rewrites the yaml incrementally per k_frac, so a hard-killed
    process can leave a truncated file that must not count as 'done'.)"""
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return bool(d) and len(d.get("runs", [])) >= min_runs
    except Exception:
        return False


# ---- (A) multi-shuffle geo ----
def multi_shuffle_done(out_folder, ds, tag, exp_mode, n_seeds):
    for sd in range(n_seeds):
        idx = os.path.join(out_folder, f"results_{ds}_{tag}_{exp_mode}_shuf_sd{sd}_multi_shuffle.yaml")
        if not os.path.exists(idx):
            return False
    return True


def run_multi_shuffle(ds, m, tag, coord, args):
    exp_mode = COORD_EXP[coord]
    out_folder = os.path.join(BASE_DIR, "rebuttal", f"{ds}_{m}_{coord}")
    if multi_shuffle_done(out_folder, ds, tag, exp_mode, args.seeds):
        print(f"[skip] multi-shuffle {ds}/{m}/{coord}")
        return
    ts = make_temp(out_folder, exp_mode, ds, tag, shuffle=True,
                   n_shuffle_runs=args.shuffle_runs, seed_end=args.seeds - 1, maxiter=args.maxiter)
    run_script(ts, out_folder, f"multi-shuffle {ds}/{m}/{coord} (runs={args.shuffle_runs})")


# ---- (B) d-sweep (geo_r, orig + shuf) ----
def dim_done(out_folder, ds, tag, d, shuffle, n_seeds):
    sfx = "_shuf" if shuffle else ""
    return all(is_yaml_complete(
        os.path.join(out_folder, f"results_{ds}_{tag}_geo_r{sfx}_sigma0.50_sd{sd}.yaml"))
        for sd in range(n_seeds))


def run_dim(ds, m, tag, d, args, shuffle):
    out_folder = os.path.join(BASE_DIR, "rebuttal_dim", f"{ds}_{m}_d{d}")
    if dim_done(out_folder, ds, tag, d, shuffle, args.seeds):
        print(f"[skip] dim {ds}/{m}/d={d}/shuf={shuffle}")
        return
    ts = make_temp(out_folder, "geo_r", ds, tag, shuffle=shuffle, n_shuffle_runs=1,
                   pd=d, seed_end=args.seeds - 1, maxiter=args.maxiter)
    run_script(ts, out_folder, f"dim {ds}/{m}/d={d}/shuf={shuffle}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["r50", "vit"],
                    help="model keys, e.g. r50 vit vitmae")
    ap.add_argument("--datasets", nargs="+", default=["cifar10", "stl10"])
    ap.add_argument("--coords", nargs="+", default=["geo_r", "geo_m"],
                    choices=list(COORD_EXP))
    ap.add_argument("--dims", nargs="+", type=int, default=[1, 2, 3, 5, 8])
    ap.add_argument("--shuffle-runs", type=int, default=5,
                    help="number of independent coordinate permutations (R4.3)")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--maxiter", type=int, default=20)
    args = ap.parse_args()

    print(f"Rebuttal schedule: models={args.models} datasets={args.datasets} "
          f"coords={args.coords} dims={args.dims} shuffle_runs={args.shuffle_runs} seeds={args.seeds}")
    for ds in args.datasets:
        for m in args.models:
            tag = MODELS[m]
            if not check_features(ds, tag):
                print(f"[warn] missing features {ds}_{tag}, skip")
                continue
            for coord in args.coords:
                run_multi_shuffle(ds, m, tag, coord, args)
            for d in args.dims:
                run_dim(ds, m, tag, d, args, shuffle=False)
                run_dim(ds, m, tag, d, args, shuffle=True)
    print("All rebuttal experiments scheduled.")


if __name__ == "__main__":
    main()
