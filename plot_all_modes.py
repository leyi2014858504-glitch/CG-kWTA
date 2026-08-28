#!/usr/bin/env python3
"""Plot combined per-(dataset, model) figures: 3 geo coords + 4 informative-neuron baselines,
plus a shuffle-gap panel for the geo modes.

Reads result YAMLs from d:\\my projects\\PythonProject4 (and d:\\cg-kwta\\Experiment_Results as fallback).
Output: d:\\cg-kwta\\Figures\\{dataset}_{model}_all_modes.png  /  _shuffle_gap.png
"""
import os
import glob
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

BASE = r"d:\my projects\PythonProject4"
EXP = r"d:\cg-kwta\Experiment_Results"
OUT = r"d:\cg-kwta\Figures"

DATASETS = ["cifar10", "cifar100", "stl10"]
MODELS = {
    "r50": "r50", "r50mocov2": "mocov2", "convnext": "convnext_base_sup",
    "convnextv2": "convnextv2_base_mae", "vitmae": "vit_mae", "vit": "vit",
    "vitdino": "dino", "swin": "swin",
}
GEO = {"geo_r": "randproj", "geo_m": "meanstdrate", "geo_pca": "pca"}
BASELINES = ["magnitude", "variance", "rate", "probe_weight"]
ALL_MODES = ["geo_r", "geo_m", "geo_pca"] + BASELINES

COLORS = {
    "geo_r": "#1f77b4", "geo_m": "#ff7f0e", "geo_pca": "#2ca02c",
    "magnitude": "#d62728", "variance": "#9467bd", "rate": "#8c564b",
    "probe_weight": "#e377c2",
}
STYLES = {m: "-" for m in ["geo_r", "geo_m", "geo_pca"]}
STYLES.update({m: "--" for m in BASELINES})


def find_files(ds, m, mode):
    """Return (seed->path) map for a mode, dedup by basename, excluding nothing."""
    folders = []
    if mode in GEO:
        folders = [os.path.join(BASE, f"{ds}_{m}_{GEO[mode]}"),
                   os.path.join(EXP, f"{ds}_{m}_{GEO[mode]}")]
    else:
        folders = [os.path.join(BASE, f"{ds}_{m}_{mode}")]
    tag = f"{ds}_{MODELS[m]}"
    seen = {}
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for f in glob.glob(os.path.join(folder, f"results_{tag}_{mode}*_sd*.yaml")):
            base = os.path.basename(f)
            if base in seen:
                continue
            m2 = __import__("re").search(r"_sd(\d+)\.ya?ml$", base)
            if m2:
                seen[int(m2.group(1))] = f
    return seen


def load_accs(files):
    """files: seed->path. Return dict k_frac -> {seed: acc}."""
    out = {}
    for sd, f in files.items():
        try:
            with open(f, encoding="utf-8") as fh:
                d = yaml.safe_load(fh)
        except Exception:
            continue
        for r in d.get("runs", []):
            if r.get("test_acc") is None:
                continue
            out.setdefault(float(r["k_frac"]), {})[sd] = float(r["test_acc"])
    return out


def ci95_series(acc_by_k, k_fracs, seeds):
    xs, lo, hi = [], [], []
    for k in k_fracs:
        vals = np.array([acc_by_k.get(k, {}).get(s, np.nan) for s in seeds])
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            xs.append(k); lo.append(np.nan); hi.append(np.nan); continue
        mean = vals.mean()
        sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        crit = student_t.ppf(0.975, len(vals) - 1) if len(vals) > 1 else 0.0
        xs.append(k); lo.append(mean - crit * sem); hi.append(mean + crit * sem)
    return np.array(xs), np.array(lo), np.array(hi)


def main():
    os.makedirs(OUT, exist_ok=True)
    for ds in DATASETS:
        for m in MODELS:
            tag = f"{ds}_{MODELS[m]}"
            data = {}          # mode -> acc_by_k
            shuf_data = {}     # geo mode -> acc_by_k (shuffled)
            for mode in ALL_MODES:
                files = find_files(ds, m, mode)
                if files:
                    data[mode] = load_accs(files)
            for mode in GEO:
                files = find_files(ds, m, mode)
                shuf_files = {sd: f for sd, f in files.items() if "_shuf" in os.path.basename(f)}
                if shuf_files:
                    shuf_data[mode] = load_accs(shuf_files)

            if not data:
                print(f"[skip] {tag}: no data")
                continue

            seeds = sorted({sd for d in data.values() for accs in d.values() for sd in accs})
            k_fracs = sorted({k for d in data.values() for k in d})
            if len(seeds) == 0:
                continue

            # ---- Figure 1: all modes ----
            plt.figure(figsize=(8.5, 5.2))
            ax = plt.gca()
            for mode in ALL_MODES:
                if mode not in data:
                    continue
                acc = data[mode]
                xs, lo, hi = ci95_series(acc, k_fracs, seeds)
                mvals = np.array([np.nanmean([acc.get(k, {}).get(s, np.nan) for s in seeds])
                                  for k in xs])
                ax.plot(xs, mvals, STYLES[mode], color=COLORS[mode], label=mode, linewidth=2)
                ax.fill_between(xs, lo, hi, color=COLORS[mode], alpha=0.15)
            ax.set_xlabel("k_frac")
            ax.set_ylabel("test_acc (mean over seeds)")
            ax.set_title(f"{tag}: geo coordinates vs informative-neuron baselines (95% CI)")
            ax.grid(True, alpha=0.3)
            ax.legend(ncol=2, fontsize=8)
            plt.tight_layout()
            p1 = os.path.join(OUT, f"{tag}_all_modes.png")
            plt.savefig(p1, dpi=200)
            plt.close()
            print("saved:", p1)

            # ---- Figure 2: shuffle gap (geo only) ----
            if shuf_data:
                plt.figure(figsize=(8.5, 5.2))
                ax = plt.gca()
                for mode in GEO:
                    if mode not in data or mode not in shuf_data:
                        continue
                    orig, shuf = data[mode], shuf_data[mode]
                    ks = sorted(set(orig) & set(shuf))
                    diffs = []
                    for k in ks:
                        d = np.array([orig[k].get(s, np.nan) - shuf[k].get(s, np.nan) for s in seeds])
                        d = d[~np.isnan(d)]
                        if len(d):
                            diffs.append((k, d))
                    if not diffs:
                        continue
                    ks = np.array([x[0] for x in diffs])
                    means = np.array([x[1].mean() for x in diffs])
                    sems = np.array([x[1].std(ddof=1) / np.sqrt(len(x[1])) if len(x[1]) > 1 else 0 for x in diffs])
                    crits = np.array([student_t.ppf(0.975, len(x[1]) - 1) if len(x[1]) > 1 else 0 for x in diffs])
                    ax.plot(ks, means, "-", color=COLORS[mode], label=f"{mode} (orig-shuf)", linewidth=2)
                    ax.fill_between(ks, means - crits * sems, means + crits * sems, color=COLORS[mode], alpha=0.15)
                ax.axhline(0, color="k", linewidth=1, linestyle=":")
                ax.set_xlabel("k_frac")
                ax.set_ylabel("test_acc gap (original - shuffled)")
                ax.set_title(f"{tag}: shuffle gap for geo coordinates (95% CI)")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=9)
                plt.tight_layout()
                p2 = os.path.join(OUT, f"{tag}_shuffle_gap.png")
                plt.savefig(p2, dpi=200)
                plt.close()
                print("saved:", p2)

    print("done.")


if __name__ == "__main__":
    main()
