#!/usr/bin/env python3
"""Coordinate-dimension (d) sensitivity sweep for Reviewer 2.3 / 4.2.

Generates temp scripts from the main script with `pd = N` replaced, runs geo_r
(original + shuffled) for each d in a separate output folder, so the shuffle gap
can be compared across d. CMA-ES convergence history is recorded per run
(`cma_fbest_history`).

Defaults are scoped to keep cost manageable (r50 + vit); extend via --models.

Usage:
    python run_dim_sweep.py
    python run_dim_sweep.py --dims 1 2 3 5 8 --models r50 vit r50mocov2

Output: d:\\my projects\\PythonProject4\\dim_sweep\\{dataset}_{model}_d{d}\\
"""
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

DEFAULT_MODELS = ["r50", "vit"]
DATASETS = ["cifar10", "cifar100", "stl10"]
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


def all_seeds_done(folder, ds, tag, shuffle):
    sfx = "_shuf" if shuffle else ""
    return all(is_yaml_complete(
        os.path.join(folder, f"results_{ds}_{tag}_geo_r{sfx}_sigma0.50_sd{sd}.yaml"))
        for sd in range(SEED_START, SEED_END + 1))


def run_one(ds, m, tag, d, shuffle):
    out_folder = os.path.join(BASE_DIR, "dim_sweep", f"{ds}_{m}_d{d}")
    os.makedirs(out_folder, exist_ok=True)
    if all_seeds_done(out_folder, ds, tag, shuffle):
        print(f"[skip] {ds}/{m}/d={d}/shuf={shuffle}")
        return

    with open(MAIN_SCRIPT, "r", encoding="utf-8") as f:
        content = f.read()
    content = re.sub(r'EXP_MODE\s*=\s*"[^"]*"', 'EXP_MODE = "geo_r"', content)
    content = re.sub(r'DATASET\s*=\s*"[^"]*"', f'DATASET = "{ds}_{tag}"', content)
    content = re.sub(r'Shuffle_mode\s*=\s*(True|False)', f'Shuffle_mode = {shuffle}', content)
    content = re.sub(r'^pd\s*=\s*\d+', f'pd = {d}', content, flags=re.M)
    content = re.sub(r'^SEED_START\s*=\s*\d+', f'SEED_START = {SEED_START}', content, flags=re.M)
    content = re.sub(r'^SEED_END\s*=\s*\d+', f'SEED_END = {SEED_END}', content, flags=re.M)
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
            print(f"[FAIL] {ds}/{m}/d={d}/shuf={shuffle}\n{r.stderr[-1500:]}")
        else:
            print(f"[OK] {ds}/{m}/d={d}/shuf={shuffle}")
    finally:
        os.remove(temp_script)


def main():
    model_keys = DEFAULT_MODELS
    dims = DEFAULT_DIMS
    for ds in DATASETS:
        for m in model_keys:
            tag = MODELS[m]
            if not check_features(ds, tag):
                print(f"[warn] missing features {ds}_{tag}, skip")
                continue
            for d in dims:
                run_one(ds, m, tag, d, shuffle=False)
                run_one(ds, m, tag, d, shuffle=True)
    print("dim sweep done.")


if __name__ == "__main__":
    main()
