#!/usr/bin/env python3
"""Run ONLY the 4 reviewer-requested baselines (magnitude/variance/rate/probe_weight)
across all models and datasets, using the same main script as the geo experiments.

Skipped automatically when all 10 seed result files already exist (resumable).
"""
import os
import subprocess
import sys
import re
import yaml

BASE_DIR = r"d:\my projects\PythonProject4"
DATA_DIR = os.path.join(BASE_DIR, "data")
MAIN_SCRIPT = os.path.join(BASE_DIR, "sphere_kwta_dimension_resnet50_MAE.py")

MODELS = {
    "r50": ("resnet50_sup", "r50"),
    "r50mocov2": ("mocov2", "mocov2"),
    "convnext": ("convnext_base_sup", "convnext_base_sup"),
    "convnextv2": ("convnextv2_base_mae", "convnextv2_base_mae"),
    "vitmae": ("vit_base_mae", "vit_mae"),
    "vit": ("vit_base", "vit"),
    "vitdino": ("vit_base_dino", "dino"),
    "swin": ("swin_base_sup", "swin"),
}

DATASETS = ["cifar10", "cifar100", "stl10"]

# Only the 4 new baselines requested by Reviewer 2.2 (kwta already in the paper).
BASELINE_MODES = ["magnitude", "variance", "rate", "probe_weight"]

SEED_START = 0
SEED_END = 9


def check_features_exist(dataset, model_tag):
    train_path = os.path.join(DATA_DIR, f"{dataset}_{model_tag}_train.pt")
    test_path = os.path.join(DATA_DIR, f"{dataset}_{model_tag}_test.pt")
    return os.path.exists(train_path) and os.path.exists(test_path)


def is_yaml_complete(path, min_runs=20):
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return bool(d) and len(d.get("runs", [])) >= min_runs
    except Exception:
        return False


def all_seeds_done(out_folder, dataset_tag, mode):
    for sd in range(SEED_START, SEED_END + 1):
        if not is_yaml_complete(os.path.join(out_folder, f"results_{dataset_tag}_{mode}_sd{sd}.yaml")):
            return False
    return True


def run_single_baseline(dataset, model_short, mode):
    model_name, model_tag = MODELS[model_short]
    dataset_tag = f"{dataset}_{model_tag}"
    out_folder = os.path.join(BASE_DIR, f"{dataset}_{model_short}_{mode}")
    os.makedirs(out_folder, exist_ok=True)

    if all_seeds_done(out_folder, dataset_tag, mode):
        print(f"[skip] {dataset_tag} / {mode}: all seeds already done")
        return

    print(f"\n=== {dataset} | {model_short} | {mode} -> {out_folder} ===")
    temp_script = os.path.join(out_folder, "temp_script.py")

    with open(MAIN_SCRIPT, "r", encoding="utf-8") as f:
        content = f.read()

    content = re.sub(r'EXP_MODE\s*=\s*"[^"]*"', f'EXP_MODE = "{mode}"', content)
    content = re.sub(r'DATASET\s*=\s*"[^"]*"', f'DATASET = "{dataset_tag}"', content)
    content = re.sub(r'Shuffle_mode\s*=\s*(True|False)', "Shuffle_mode = False", content)
    data_dir_escaped = DATA_DIR.replace("\\", "\\\\")
    content = re.sub(r'DATA_DIR\s*=\s*"[^"]*"', f'DATA_DIR = r"{data_dir_escaped}"', content)

    project_root_escaped = BASE_DIR.replace("\\", "\\\\")
    content = f'import sys; sys.path.insert(0, r"{project_root_escaped}")\n' + content

    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(content)

    try:
        result = subprocess.run(
            [sys.executable, temp_script],
            cwd=out_folder,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        if result.returncode != 0:
            print(f"[FAIL] {dataset_tag} / {mode}")
            print(f"STDOUT: {result.stdout[-3000:]}")
            print(f"STDERR: {result.stderr[-3000:]}")
        else:
            print(f"[OK] {dataset_tag} / {mode}")
    finally:
        os.remove(temp_script)


def main():
    print("=" * 80)
    print(f"Baseline-only runs: {len(BASELINE_MODES)} modes x {len(MODELS)} models x {len(DATASETS)} datasets")
    print("=" * 80)

    for dataset in DATASETS:
        for model_short, (model_name, model_tag) in MODELS.items():
            if not check_features_exist(dataset, model_tag):
                print(f"[warn] missing features: {dataset}_{model_tag}, skip")
                continue
            for mode in BASELINE_MODES:
                run_single_baseline(dataset, model_short, mode)

    print("\nAll baseline experiments finished.")


if __name__ == "__main__":
    main()
