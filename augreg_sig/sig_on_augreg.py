import os
import glob
import re
import yaml
import numpy as np
from scipy import stats

# ---- 文件加载 ----
file_pattern = "results_kfrac_geo_r_augreg_cls*.yaml"
files = glob.glob(file_pattern)

data = {}  # key=(seed, k_frac), value={'non_shuffled': acc, 'shuffled': acc}

for fpath in files:
    fname = os.path.basename(fpath)
    shuffled = "_shuffled_" in fname

    match = re.search(r'sd(\d+)', fname)
    if not match:
        print(f"跳过（无法提取seed）: {fname}")
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
        data[key]['shuffled' if shuffled else 'non_shuffled'] = test_acc

# ---- 构建配对 ----
seeds = sorted(set(s for s, _ in data.keys()))
k_fracs = sorted(set(k for _, k in data.keys()))

non_shuffled_all, shuffled_all = [], []
for key, vals in data.items():
    if 'non_shuffled' in vals and 'shuffled' in vals:
        non_shuffled_all.append(vals['non_shuffled'])
        shuffled_all.append(vals['shuffled'])
    else:
        print(f"缺失配对: {key} -> {list(vals.keys())}")

ns_arr = np.array(non_shuffled_all)
sh_arr = np.array(shuffled_all)
diffs_all = ns_arr - sh_arr

# ---- 总体统计 ----
print("=" * 55)
print("总体统计（所有seed × 所有k_frac合并）")
print("=" * 55)
print(f"总配对数       : {len(ns_arr)}")
print(f"non-shuffled均值: {ns_arr.mean():.6f}")
print(f"shuffled均值    : {sh_arr.mean():.6f}")
print(f"差值均值        : {diffs_all.mean():.6f}")
print(f"差值标准差      : {diffs_all.std(ddof=1):.6f}")

t_all, p_all = stats.ttest_rel(ns_arr, sh_arr)
w_all, pw_all = stats.wilcoxon(ns_arr, sh_arr)
print(f"\n配对t检验  : t={t_all:.4f}, p={p_all:.4e}")
print(f"Wilcoxon检验: W={w_all:.1f},  p={pw_all:.4e}")
print("注：此处各观测不独立（同seed跨k_frac相关），仅供参考")

# ---- 按k_frac分层检验 ----
print("\n" + "=" * 55)
print("按k_frac分层配对检验（每点n=seed数，观测独立）")
print("=" * 55)
print(f"{'k_frac':<8} {'n':>4} {'mean_diff':>10} {'t':>7} {'p_t':>9} {'W':>8} {'p_W':>9} {'sig':>4}")
print("-" * 55)

for k in k_fracs:
    ns_k = [data[(s,k)]['non_shuffled'] for s in seeds
            if (s,k) in data and 'non_shuffled' in data[(s,k)] and 'shuffled' in data[(s,k)]]
    sh_k = [data[(s,k)]['shuffled']     for s in seeds
            if (s,k) in data and 'non_shuffled' in data[(s,k)] and 'shuffled' in data[(s,k)]]

    if len(ns_k) < 2:
        print(f"{k:<8.2f} {'<2对，跳过':>40}")
        continue

    diff_k = np.array(ns_k) - np.array(sh_k)
    t_k, p_t = stats.ttest_rel(ns_k, sh_k)
    try:
        w_k, p_w = stats.wilcoxon(ns_k, sh_k)
    except ValueError:
        w_k, p_w = float('nan'), float('nan')  # 所有差值为0时报错

    sig = "*" if p_t < 0.05 else ("†" if p_t < 0.10 else "")
    print(f"{k:<8.2f} {len(ns_k):>4} {diff_k.mean():>10.4f} "
          f"{t_k:>7.3f} {p_t:>9.4f} {w_k:>8.1f} {p_w:>9.4f} {sig:>4}")

print("-" * 55)
print("* p<0.05  † p<0.10")