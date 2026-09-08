#!/usr/bin/env python3
"""Best-of-N unstructured random mask baseline (Reviewer R4.2).

Matches the geo CMA-ES search budget (B = lambda * maxiter, lambda from cma
default popsize for dim N=1+3), sampling N random top-k masks and selecting the
one with min J = (1 - acc) + eps * mse on val — identical selection rule to geo.

One seed per (dataset, model) is enough for the budget-match demonstration;
run with --seeds to extend.

Usage:
    python run_bestof_random.py
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
    "vit": "vit",
    "vitmae": "vit_mae",
}
DATASETS = ["cifar10", "cifar100", "stl10"]


def is_yaml_complete(path, min_runs=20):
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return bool(d) and len(d.get("runs", [])) >= min_runs
    except Exception:
        return False


def check_features(ds, tag):
    return (os.path.exists(os.path.join(DATA_DIR, f"{ds}_{tag}_train.pt"))
            and os.path.exists(os.path.join(DATA_DIR, f"{ds}_{tag}_test.pt")))


def all_seeds_done(out_folder, ds, tag, n_seeds):
    return all(is_yaml_complete(os.path.join(
        out_folder, f"results_{ds}_{tag}_sd{sd}.yaml")) for sd in range(n_seeds))


def run_one(ds, m, tag, args):
    out_folder = os.path.join(BASE_DIR, f"{ds}_{m}_bestof_random")
    os.makedirs(out_folder, exist_ok=True)
    if all_seeds_done(out_folder, ds, tag, args.seeds):
        print(f"[skip] {ds}/{m}")
        return

    with open(MAIN_SCRIPT, "r", encoding="utf-8") as f:
        content = f.read()
    content = re.sub(r'EXP_MODE\s*=\s*"[^"]*"', 'EXP_MODE = "bestof_random"', content)
    content = re.sub(r'DATASET\s*=\s*"[^"]*"', f'DATASET = "{ds}_{tag}"', content)
    content = re.sub(r'Shuffle_mode\s*=\s*(True|False)', "Shuffle_mode = False", content)
    content = re.sub(r'^SEED_START\s*=\s*\d+', "SEED_START = 0", content, flags=re.M)
    content = re.sub(r'^SEED_END\s*=\s*\d+', f'SEED_END = {args.seeds - 1}', content, flags=re.M)
    data_dir_escaped = DATA_DIR.replace("\\", "\\\\")
    content = re.sub(r'DATA_DIR\s*=\s*"[^"]*"', f'DATA_DIR = r"{data_dir_escaped}"', content)
    base_escaped = BASE_DIR.replace("\\", "\\\\")
    content = f'import sys; sys.path.insert(0, r"{base_escaped}")\n' + content

    temp_script = os.path.join(out_folder, "temp_script_bestofrandom.py")
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(content)
    try:
        r = subprocess.run([sys.executable, temp_script], cwd=out_folder,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(f"[FAIL] {ds}/{m}\n{r.stderr[-1500:]}")
        else:
            print(f"[OK] {ds}/{m}")
    finally:
        if os.path.exists(temp_script):
            os.remove(temp_script)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    for ds in args.datasets:
        for m in args.models:
            tag = MODELS[m]
            if not check_features(ds, tag):
                print(f"[warn] missing features {ds}_{tag}, skip")
                continue
            run_one(ds, m, tag, args)
    print("best-of-random done.")


if __name__ == "__main__":
    main()
