import os
import glob
import re
import yaml
import numpy as np
import matplotlib.pyplot as plt

file_pattern = "results_kfrac_geo_r_0.05_*.yaml"

data = {}  # data[maxiter] = [val1, val2, ...]

for fpath in glob.glob(file_pattern):
    fname = os.path.basename(fpath)

    # 匹配如 "5maxiter5" → maxiter=5, seed=5
    m = re.search(r'_(\d+)maxiter(\d+)\.yaml', fname)
    if not m:
        print(f"跳过: {fname}")
        continue

    maxiter = int(m.group(1))
    # seed = int(m.group(2))  # 备用

    with open(fpath, 'r') as f:
        yaml_data = yaml.safe_load(f)

    # 根据你yaml里的实际key修改这里
    val = yaml_data['runs'][0]['best_acc']  # 或 yaml_data['best_acc'] 等

    if maxiter not in data:
        data[maxiter] = []
    data[maxiter].append(val)

# ---- 统计 ----
maxiters = sorted(data.keys())
means = np.array([np.mean(data[m]) for m in maxiters])
stds  = np.array([np.std(data[m], ddof=1) for m in maxiters])
ns    = np.array([len(data[m]) for m in maxiters])
se    = stds / np.sqrt(ns)

for m, mean, std, n in zip(maxiters, means, stds, ns):
    print(f"maxiter={m:3d}: n={n}, mean={mean:.4f}, std={std:.4f}")

# ---- 画图 ----
plt.figure(figsize=(5, 3.5))
plt.plot(maxiters, means, 'o-', color='steelblue', linewidth=2, markersize=6)
plt.fill_between(maxiters, means - 1.96*se, means + 1.96*se,
                 alpha=0.2, color='steelblue', label='95% CI')
plt.axvline(x=10, color='gray', linestyle='--', linewidth=1.2, label='Used (maxiter=10)')
plt.xscale('log', base=2)
plt.xticks(maxiters, labels=[str(m) for m in maxiters])
plt.xlabel('CMA-ES maxiter')
plt.ylabel('Test accuracy')
plt.title(r'Budget convergence ($k_{\rm frac}=0.05$)')
plt.legend()
plt.tight_layout()
plt.savefig('budget_convergence.png', dpi=150)
plt.show()
