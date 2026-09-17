#!/usr/bin/env python3
"""Isolated sensitivity checks requested by Reviewer #4:
"test sensitivity to coordinate dimension, calibration split, random-projection
seed, distance, and selector".

Dimension / distance / selector are already covered elsewhere. This script
covers the two remaining nuisance parameters, one at a time:

  calibration split  (CALIB_SEED)         - which D_cal rows build the coordinates
  random-projection seed (PROJ_SEED_OVERRIDE) - the RandProj directions

Isolation: the data seed is held FIXED at 0, so the train/val split, the CMA-ES
search trajectory, the shuffle permutation and the final refit are identical
across variants; only the knob under test changes. Both arms (original and
shuffled) are run so the shuffle gap can be recomputed for every variant.

Each variant writes into its own folder because result filenames do not encode
these knobs. Resumable: a variant is skipped when both arms have a complete
20-k yaml.

Usage:
    python run_sensitivity_checks.py
    python run_sensitivity_checks.py --values 0 1 2
    python run_sensitivity_checks.py --only calib_im100_geo_m
"""
import argparse
import os
import re
import subprocess
import sys

import yaml

BASE_DIR = r"d:\my projects\PythonProject4"
DATA_DIR = os.path.join(BASE_DIR, "data")
MAIN_SCRIPT = os.path.join(BASE_DIR, "sphere_kwta_dimension_resnet50_MAE.py")
LOG_PATH = os.path.join(r"d:\cg-kwta", "sensitivity_checks.log")

SEED = 0  # held fixed so only the knob under test varies

CONFIGS = [
    # key, dataset tag, exp_mode, knob constant, out prefix
    ("calib_c10_geo_m", "cifar10_r50", "geo_m", "CALIB_SEED", "sens_calib_c10_geo_m"),
    ("calib_im100_geo_m", "imagenet100_r50", "geo_m", "CALIB_SEED", "sens_calib_im100_geo_m"),
    ("calib_im100_geo_r", "imagenet100_r50", "geo_r", "CALIB_SEED", "sens_calib_im100_geo_r"),
    ("proj_c10_geo_r", "cifar10_r50", "geo_r", "PROJ_SEED_OVERRIDE", "sens_proj_c10_geo_r"),
    ("proj_im100_geo_r", "imagenet100_r50", "geo_r", "PROJ_SEED_OVERRIDE", "sens_proj_im100_geo_r"),
]

LOG = open(LOG_PATH, "a", encoding="utf-8", buffering=1)


def log(msg):
    print(msg, flush=True)
    LOG.write(str(msg) + "\n")
    LOG.flush()


def is_complete(path, min_runs=20):
    if not os.path.exists(path):
        return False
    try:
        d = yaml.safe_load(open(path, encoding="utf-8"))
        runs = d.get("runs") or []
        return len(runs) >= min_runs and all(r.get("test_acc") is not None for r in runs)
    except Exception:
        return False


def yaml_names(tag, exp_mode, shuffle, sd):
    sfx = "_shuf" if shuffle else ""
    if exp_mode == "geo_r":
        return f"results_{tag}_geo_r{sfx}_sigma0.50_sd{sd}.yaml"
    return f"results_{tag}_{exp_mode}{sfx}_sd{sd}.yaml"


def make_temp(folder, tag, exp_mode, knob, value, shuffle):
    os.makedirs(folder, exist_ok=True)
    with open(MAIN_SCRIPT, encoding="utf-8") as f:
        content = f.read()
    content = re.sub(r'EXP_MODE\s*=\s*"[^"]*"', f'EXP_MODE = "{exp_mode}"', content)
    content = re.sub(r'DATASET\s*=\s*"[^"]*"', f'DATASET = "{tag}"', content)
    content = re.sub(r'Shuffle_mode\s*=\s*(True|False)', f'Shuffle_mode = {shuffle}', content)
    content = re.sub(r'N_SHUFFLE_RUNS\s*=\s*\d+', 'N_SHUFFLE_RUNS = 1', content)
    content = re.sub(r'^\s*SEED_START\s*=\s*\d+', f'    SEED_START = {SEED}', content, flags=re.M)
    content = re.sub(r'^\s*SEED_END\s*=\s*\d+', f'    SEED_END = {SEED}', content, flags=re.M)
    # the knob under test
    content = re.sub(rf'^\s*{knob}\s*=\s*(None|\d+)', f'{knob} = {int(value)}', content, flags=re.M)
    dd = DATA_DIR.replace("\\", "\\\\")
    content = re.sub(r'DATA_DIR\s*=\s*"[^"]*"', f'DATA_DIR = r"{dd}"', content)
    bd = BASE_DIR.replace("\\", "\\\\")
    content = f'import sys; sys.path.insert(0, r"{bd}")\n' + content
    ts = os.path.join(folder, "temp_script_sens.py")
    with open(ts, "w", encoding="utf-8") as f:
        f.write(content)
    return ts


def run_variant(key, tag, exp_mode, knob, value, prefix):
    for shuffle in (False, True):
        folder = os.path.join(BASE_DIR, f"{prefix}_{value}")
        if is_complete(os.path.join(folder, yaml_names(tag, exp_mode, shuffle, SEED))):
            log(f"[skip] {key} knob={value} shuf={shuffle}")
            continue
        ts = make_temp(folder, tag, exp_mode, knob, value, shuffle)
        log(f"=== {key} {knob}={value} shuf={shuffle} ===")
        r = subprocess.run([sys.executable, ts], cwd=folder, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            log(f"[FAIL] {key} {knob}={value} shuf={shuffle}\n{(r.stderr or '')[-1200:]}")
        else:
            log(f"[OK] {key} {knob}={value} shuf={shuffle}")
        if os.path.exists(ts):
            os.remove(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                    help="values of the knob to sweep (default 0..4)")
    ap.add_argument("--only", nargs="+", default=None,
                    help="subset of config keys")
    args = ap.parse_args()

    log("")
    log("#" * 60)
    log(f"Isolated sensitivity checks started (values={args.values})")
    for key, tag, exp_mode, knob, prefix in CONFIGS:
        if args.only and key not in args.only:
            continue
        for v in args.values:
            run_variant(key, tag, exp_mode, knob, v, prefix)
    log("Isolated sensitivity checks finished.")


if __name__ == "__main__":
    main()