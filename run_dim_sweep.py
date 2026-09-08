#!/usr/bin/env python3
"""Coordinate-dimension (d) sensitivity sweep for Reviewer 2.3 / 4.2.

Generates temp scripts from the main script with `pd = N` replaced, runs
geo (RandProj or PCA) original + shuffled for each d in a separate folder, so the
shuffle gap can be compared across d. CMA-ES convergence history is recorded per
run (`cma_fbest_history`).

Scope (default): r50 + vit x cifar10 + cifar100 + stl10 x d in {1,2,3,5,8}.
  original + shuffled. Use --models/--datasets/--coords/--dims to change.

Output folders (kept separate per coord so RandProj vs PCA don't collide):
  d:\\my projects\\PythonProject4\\dim_sweep\\{coord}\\{ds}_{m}_d{d}\\

Resumable: a combo is skipped only when all 10 (or chosen) seed yamls are complete.
"""
import os
import re
import sys
import subprocess
import argparse
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

# coord -> (EXP_MODE, output-name pattern template with {sfx},{tag},{sd})
# geo_r (RandProj) writes "..._geo_r{sfx}_sigma0.50_sd{sd}.yaml"
# geo_pca (PCA)     writes "..._geo_pca{sfx}_sd{sd}.yaml"
COORD_SPEC = {
    "geo_r": ("geo_r", "results_{tag}_geo_r{sfx}_sigma0.50_sd{sd}.yaml"),
    "geo_pca": ("geo_pca", "results_{tag}_geo_pca{sfx}_sd{sd}.yaml"),
}

DEFAULT_MODELS = ["r50", "vit"]
DEFAULT_DATASETS = ["cifar10", "cifar100", "stl10"]
DEFAULT_DIMS = [1, 2, 3, 5, 8]
SEED_START = 0
SEED_END = 9


def check_features(ds, tag):
    return (os.path.exists(os.path.join(DATA_DIR, f"{ds}_{tag}_train.pt"))
            and os.path.exists(os.path.join(DATA_DIR, f"{ds}_{tag}_test.pt")))


def is_yaml_complete(path, min_runs=20):
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return bool(d) and len(d.get("runs", [])) >= min_runs
    except Exception:
        return False


def all_seeds_done(out_folder, ds, tag, coord, shuffle, n_seeds):
    exp_mode, pat = COORD_SPEC[coord]
    sfx = "_shuf" if shuffle else ""
    pattern = pat.format(tag=f"{ds}_{tag}", sfx=sfx, sd="{sd}")
    return all(is_yaml_complete(os.path.join(out_folder, pattern.format(sd=sd)))
               for sd in range(SEED_START, SEED_START + n_seeds))


def run_one(ds, m, tag, coord, d, shuffle, args):
    exp_mode, _ = COORD_SPEC[coord]
    out_folder = os.path.join(BASE_DIR, "dim_sweep", coord, f"{ds}_{m}_d{d}")
    os.makedirs(out_folder, exist_ok=True)
    if all_seeds_done(out_folder, ds, tag, coord, shuffle, args.seeds):
        print(f"[skip] {ds}/{m}/{coord}/d={d}/shuf={shuffle}")
        return

    with open(MAIN_SCRIPT, "r", encoding="utf-8") as f:
        content = f.read()
    content = re.sub(r'EXP_MODE\s*=\s*"[^"]*"', f'EXP_MODE = "{exp_mode}"', content)
    content = re.sub(r'DATASET\s*=\s*"[^"]*"', f'DATASET = "{ds}_{tag}"', content)
    content = re.sub(r'Shuffle_mode\s*=\s*(True|False)', f'Shuffle_mode = {shuffle}', content)
    content = re.sub(r'^pd\s*=\s*\d+', f'pd = {d}', content, flags=re.M)
    content = re.sub(r'^SEED_START\s*=\s*\d+', f'SEED_START = {SEED_START}', content, flags=re.M)
    content = re.sub(r'^SEED_END\s*=\s*\d+', f'SEED_END = {SEED_START + args.seeds - 1}', content, flags=re.M)
    data_dir_escaped = DATA_DIR.replace("\\", "\\\\")
    content = re.sub(r'DATA_DIR\s*=\s*"[^"]*"', f'DATA_DIR = r"{data_dir_escaped}"', content)
    base_escaped = BASE_DIR.replace("\\", "\\\\")
    content = f'import sys; sys.path.insert(0, r"{base_escaped}")\n' + content

    temp_script = os.path.join(out_folder, f"temp_script_d{d}.py")
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(content)
    try:
        r = subprocess.run([sys.executable, temp_script], cwd=out_folder,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(f"[FAIL] {ds}/{m}/{coord}/d={d}/shuf={shuffle}\n{r.stderr[-1500:]}")
        else:
            print(f"[OK] {ds}/{m}/{coord}/d={d}/shuf={shuffle}")
    finally:
        if os.path.exists(temp_script):
            os.remove(temp_script)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--coords", nargs="+", default=["geo_r"], choices=list(COORD_SPEC))
    ap.add_argument("--dims", nargs="+", type=int, default=DEFAULT_DIMS)
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()

    total = (len(args.models) * len(args.datasets) * len(args.coords)
             * len(args.dims) * args.seeds * 2)
    print(f"d-sweep: models={args.models} datasets={args.datasets} coords={args.coords} "
          f"dims={args.dims} seeds={args.seeds} -> seed-runs={total}")

    for ds in args.datasets:
        for m in args.models:
            tag = MODELS[m]
            if not check_features(ds, tag):
                print(f"[warn] missing features {ds}_{tag}, skip")
                continue
            for coord in args.coords:
                for d in args.dims:
                    run_one(ds, m, tag, coord, d, shuffle=False, args=args)
                    run_one(ds, m, tag, coord, d, shuffle=True, args=args)
    print("d-sweep done.")


if __name__ == "__main__":
    main()
