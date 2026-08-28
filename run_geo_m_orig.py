#!/usr/bin/env python3
"""Backfill two gaps with the CURRENT main script (sigma0=0.5):

  Part 1: non-shuffled geo_m originals for combos that lacked a same-script orig
          (cifar10 vit, stl10 r50, stl10 vit)  -> rebuttal_geo_m_orig\\
  Part 2: vitmae geo_r original + shuffled (cifar10, stl10) to VERIFY whether the
          "MAE > informative-neuron baselines" claim holds under a clean same-script,
          sigma0=0.5 comparison  -> rebuttal_mae_verify\\

Resumable (checks complete 20-k_frac yamls). Run in terminal:
    python run_geo_m_orig.py
"""
import os
import re
import sys
import subprocess
import yaml

BASE_DIR = r"d:\my projects\PythonProject4"
DATA_DIR = os.path.join(BASE_DIR, "data")
MAIN_SCRIPT = os.path.join(BASE_DIR, "sphere_kwta_dimension_resnet50_MAE.py")

MODELS = {"r50": "r50", "vit": "vit", "vitmae": "vit_mae"}

# (dataset, model_short, exp_mode, shuffle)
JOBS = [
    # Part 1: missing same-script geo_m originals
    ("cifar10", "vit", "geo_m", False),
    ("stl10", "r50", "geo_m", False),
    ("stl10", "vit", "geo_m", False),
    # Part 2: vitmae geo_r verification (original + shuffled)
    ("cifar10", "vitmae", "geo_r", False),
    ("cifar10", "vitmae", "geo_r", True),
    ("stl10", "vitmae", "geo_r", False),
    ("stl10", "vitmae", "geo_r", True),
]

EXP_GEOR = "geo_r"
EXP_GEOM = "geo_m"


def is_yaml_complete(path, min_runs=20):
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return bool(d) and len(d.get("runs", [])) >= min_runs
    except Exception:
        return False


def run_job(ds, m, exp_mode, shuffle):
    tag = MODELS[m]
    if exp_mode == "geo_r":
        out_folder = os.path.join(BASE_DIR, "rebuttal_mae_verify", f"{ds}_{m}")
    else:
        out_folder = os.path.join(BASE_DIR, "rebuttal_geo_m_orig", f"{ds}_{m}")
    sfx = "_shuf" if shuffle else ""
    if exp_mode == "geo_r":
        pat = f"results_{ds}_{tag}_geo_r{sfx}_sigma0.50_sd{{sd}}.yaml"
    else:
        pat = f"results_{ds}_{tag}_geo_m{sfx}_sd{{sd}}.yaml"
    if all(is_yaml_complete(os.path.join(out_folder, pat.format(sd=sd))) for sd in range(10)):
        print(f"[skip] {ds}/{m}/{exp_mode}/shuf={shuffle}")
        return
    os.makedirs(out_folder, exist_ok=True)

    with open(MAIN_SCRIPT, "r", encoding="utf-8") as f:
        content = f.read()
    content = re.sub(r'EXP_MODE\s*=\s*"[^"]*"', f'EXP_MODE = "{exp_mode}"', content)
    content = re.sub(r'DATASET\s*=\s*"[^"]*"', f'DATASET = "{ds}_{tag}"', content)
    content = re.sub(r'Shuffle_mode\s*=\s*(True|False)', f'Shuffle_mode = {shuffle}', content)
    content = re.sub(r'N_SHUFFLE_RUNS\s*=\s*\d+', "N_SHUFFLE_RUNS = 1", content)
    data_dir_escaped = DATA_DIR.replace("\\", "\\\\")
    content = re.sub(r'DATA_DIR\s*=\s*"[^"]*"', f'DATA_DIR = r"{data_dir_escaped}"', content)
    base_escaped = BASE_DIR.replace("\\", "\\\\")
    content = f'import sys; sys.path.insert(0, r"{base_escaped}")\n' + content

    temp_script = os.path.join(out_folder, "temp_script_backfill.py")
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(content)
    try:
        r = subprocess.run([sys.executable, temp_script], cwd=out_folder,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(f"[FAIL] {ds}/{m}/{exp_mode}/shuf={shuffle}\n{r.stderr[-1200:]}")
        else:
            print(f"[OK] {ds}/{m}/{exp_mode}/shuf={shuffle}")
    finally:
        if os.path.exists(temp_script):
            os.remove(temp_script)


def main():
    print(f"backfill jobs: {len(JOBS)}")
    for ds, m, exp_mode, shuffle in JOBS:
        run_job(ds, m, exp_mode, shuffle)
    print("done.")


if __name__ == "__main__":
    main()
