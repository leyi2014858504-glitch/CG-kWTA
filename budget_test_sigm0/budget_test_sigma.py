import os
import glob
import re
import yaml
import numpy as np
import matplotlib.pyplot as plt

file_pattern = "results_sigma*.yaml"

data = {}  # data[sigma] = [val1, val2, ...]

for fpath in glob.glob(file_pattern):
    fname = os.path.basename(fpath)

    # Match e.g., "sigma0.05_sd3" → sigma=0.05, seed=3
    m = re.search(r'sigma(\d+\.\d+)_sd(\d+)\.yaml', fname)
    if not m:
        print(f"Skipping: {fname}")
        continue

    sigma = float(m.group(1))
    # seed = int(m.group(2))  

    with open(fpath, 'r') as f:
        yaml_data = yaml.safe_load(f)

    val = yaml_data['runs'][0]['best_acc']  # Use validation set

    if sigma not in data:
        data[sigma] = []
    data[sigma].append(val)

# ---- Statistics ----
sigmas = sorted(data.keys())
means = np.array([np.mean(data[s]) for s in sigmas])
stds  = np.array([np.std(data[s], ddof=1) for s in sigmas])
ns    = np.array([len(data[s]) for s in sigmas])
se    = stds / np.sqrt(ns)

for s, mean, std, n in zip(sigmas, means, stds, ns):
    print(f"sigma={s:.2f}: n={n}, mean={mean:.4f}, std={std:.4f}")

# ---- Plotting ----
plt.figure(figsize=(5, 3.5))
plt.plot(sigmas, means, 'o-', color='steelblue', linewidth=2, markersize=6)
plt.fill_between(sigmas, means - 1.96*se, means + 1.96*se,
                 alpha=0.2, color='steelblue', label='95% CI')
plt.axvline(x=0.2, color='gray', linestyle='--', linewidth=1.2, label='Used (σ₀=0.2)')
plt.xlabel('CMA-ES initial step size $\\sigma_0$')
plt.ylabel('Val accuracy')
plt.title(r'Sensitivity to $\sigma_0$ ($k_{\rm frac}=0.05$)')
plt.legend()
plt.tight_layout()
plt.savefig('sigma_sensitivity.png', dpi=150)
plt.show()
