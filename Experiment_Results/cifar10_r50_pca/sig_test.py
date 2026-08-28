import os
import glob
import re
import yaml
import numpy as np
from scipy import stats

# ---- config ----
# match results_kfrac_geo_r_*.yaml
file_pattern = "results_cifar10_vit_mae_geo_pca_sd*.yaml"
files = glob.glob(file_pattern)

data = {}  # key=(seed, k_frac), value={'original': acc, 'shuffled': acc}

for fpath in files:
    fname = os.path.basename(fpath)
    # logic: files containing 'shuffled' are the control group, otherwise the experiment group
    is_shuffled = "_shuffled_" in fname

    # extract seed: match the digits after 'sd'
    match = re.search(r'sd(\d+)', fname)
    if not match:
        continue
    seed = int(match.group(1))

    with open(fpath, 'r') as f:
        yaml_data = yaml.safe_load(f)

    for run in yaml_data.get('runs', []):
        k_frac = round(float(run['k_frac']), 2)
        test_acc = run['test_acc']
        key = (seed, k_frac)
        if key not in data:
            data[key] = {}
        data[key]['shuffled' if is_shuffled else 'original'] = test_acc

# ---- extract paired data ----
seeds = sorted(set(s for s, _ in data.keys()))
k_fracs = sorted(set(k for _, k in data.keys()))

# ---- one-sided test stratified by k_frac ----
print("=" * 65)
print(f"{'k_frac':<8} {'n':>4} {'Original':>10} {'Shuffled':>10} {'Diff':>8} {'p_t':>9} {'p_W':>9} {'Sig'}")
print("-" * 65)

for k in k_fracs:
    orig_k = []
    shuf_k = []
    for s in seeds:
        entry = data.get((s, k), {})
        if 'original' in entry and 'shuffled' in entry:
            orig_k.append(entry['original'])
            shuf_k.append(entry['shuffled'])

    n = len(orig_k)
    if n < 3:  # too few samples for statistics
        if n > 0:
            print(f"{k:<8.2f} {n:>4} {'insufficient data (n<3)':>40}")
        continue

    orig_arr = np.array(orig_k)
    shuf_arr = np.array(shuf_k)
    diff_mean = orig_arr.mean() - shuf_arr.mean()

    # paired t-test (one-sided: original > shuffled)
    # alternative='greater' means H1: original - shuffled > 0
    t_stat, p_t = stats.ttest_rel(orig_arr, shuf_arr, alternative='greater')

    # Wilcoxon signed-rank test (one-sided)
    try:
        # wilcoxon errors if all differences are identical; handle as exception
        res_w = stats.wilcoxon(orig_arr, shuf_arr, alternative='greater')
        p_w = res_w.pvalue
    except ValueError:
        p_w = float('nan')

    # significance markers
    sig = "***" if p_t < 0.001 else ("**" if p_t < 0.01 else ("*" if p_t < 0.05 else ("†" if p_t < 0.1 else "")))

    print(f"{k:<8.2f} {n:>4} {orig_arr.mean():>10.4f} {shuf_arr.mean():>10.4f} {diff_mean:>8.4f} "
          f"{p_t:>9.4f} {p_w:>9.4f} {sig}")

print("-" * 65)
print("Note: p-values are one-sided (H1: Original > Shuffled)")
print("* p<0.05, ** p<0.01, *** p<0.001, † p<0.1")
