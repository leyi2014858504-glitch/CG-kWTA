# CG-kWTA Experiments

This repository contains the experiment pipeline for CG-kWTA gating experiments across datasets and backbones.

## 1. Core scripts and roles

- `cg_kwta.py`  
  Main experiment engine (renamed from `sphere_kwta_dimension_resnet50_MAE.py`).

- `run_all_experiments.py`  
  Full-matrix automation (feature check/extract + run + plotting script injection).

- `extract_features.py`  
  Builds cached feature files in `./data`, e.g. `{dataset}_{tag}_{train|test}.pt`.

- `plot_4modes_ci95.py` (must be in project root)  
  Global plot template used by `run_all_experiments.py`.

- `merge_all_results.py`  
  Merge CSV summaries from experiment folders.

- `plot_all_results.py`  
  Optional root-level aggregation utility.

---

## 2. Environment

Recommended:
- Python 3.10+
- torch, torchvision, timm
- numpy, pyyaml, cma, networkx
- pandas, openpyxl

Install:

```bash
pip install torch torchvision timm numpy pyyaml cma networkx pandas openpyxl
```

---

## 3. Full experiment matrix

```bash
python run_all_experiments.py
```

This script:
1. checks missing cached features in `./data`
2. auto-runs feature extraction if needed
3. runs all configured combinations
4. copies root `plot_4modes_ci95.py` into each output folder as `plot_ci95.py`
5. patches per-folder YAML filename patterns

---

## 4. Run a single experiment via main script (important)

Edit user switches in `cg_kwta.py` (bottom `if __name__ == "__main__":` block), then run:

```bash
python cg_kwta.py
```

To run only one seed:
- set `SEED_START = 3`
- set `SEED_END = 3`

This gives one single-seed experiment.

---

## 5. Main-script tunable hyperparameters

### A) Global behavior (top of file)

- `sigma0`  
  CMA-ES initial step size (search scale).

- `pd`  
  Coordinate dimension for geometric gate points.

- `GEOSCORE_MODE`  
  Geometry scoring mode (`center`, `shell`, etc., depending on implementation).

- `GEO_PTS_INIT`  
  Point initialization mode (`randproj`, `meanvar`, `pca`, `external`, ...).

- `PTS_EXTERNAL_PATH`  
  Path to external point file when `GEO_PTS_INIT="external"`.

- `Shuffle_mode`  
  Whether to shuffle neuron-point assignment for shuffle-control experiments.

- `FITNESS_SHUFFLE_MODE`, `FITNESS_SHUFFLE_SEED`  
  Optional shuffle on CMA fitness values (ablation/control use).

### B) Main run switches (`__main__` block)

- `EXP_MODE`  
  Experiment type. Typical options:
  - `geo_r` (geo + randproj points)
  - `geo_m` (geo + mean/std/rate points)
  - `geo_pca`
  - `random`
  - `kwta`
  - `bestof_random`
  - `semantic_test`
  - `geo_r_external`

- `SEED_START`, `SEED_END`  
  Seed range (inclusive).

- `MAXITER`  
  CMA optimization iterations.

- `DATASET`  
  Dataset key used by dataset registry in script (e.g. `cifar10_r50`, `cifar10_vit`, `cifar100_r50`, `stl10_r50`, ...).

- `DATA_DIR`  
  Data/cache directory.

### C) Sweep-level params (inside function call)

In `sweep_kfrac_to_yaml(...)`, defaults include:
- `ridge_alpha`
- `cma_train_n`
- `cma_val_n`
- `maxiter`
- output YAML naming pattern

---

## 6. Single-experiment quick recipes

### Example 1: one run, CIFAR10-ResNet50 cached features, geo-randproj

Set in `cg_kwta.py`:
- `EXP_MODE = "geo_r"`
- `SEED_START = 0`
- `SEED_END = 0`
- `MAXITER = 20`
- `DATASET = "cifar10_r50"`
- `DATA_DIR = "./data"`

Then run:

```bash
python cg_kwta.py
```

### Example 2: random baseline (no CMA search emphasis)

Set:
- `EXP_MODE = "random"`
- `MAXITER = 0` (or keep low)
- single seed range

Run same command.

---

## 7. Per-folder plotting

After experiments, each output folder contains `plot_ci95.py`.

```bash
cd <experiment_folder>
python plot_ci95.py
```

---

## 8. Merge all results

```bash
python merge_all_results.py
```

---

## 9. Naming/rename note

If the main file has been renamed to `cg_kwta.py`, ensure this line in `run_all_experiments.py` is updated:

- `MAIN_SCRIPT = os.path.join(BASE_DIR, "cg_kwta.py")`

Also keep this in root:

- `plot_ci95.py`

# Known Limitations

Some printed words in the result files may not change when hyper‑parameters are altered in the main script; this is a leftover issue from earlier versions, but in practice it does not affect the results. 

# License

This project is licensed under the MIT License.