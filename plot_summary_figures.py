#!/usr/bin/env python3
"""Consolidated rebuttal summary figures (line charts).

Outputs to d:\\cg-kwta\\Figures\\summary\\:
  1. shuffle_gap_by_combo.png    low-k shuffle gap per (dataset, model, coord) + Holm stars
  2. d_sensitivity.png           low-k gap vs coordinate dimension d
  3. gap_vs_kfrac.png            shuffle gap vs k_frac (panels by model)
  4. geo_vs_baselines.png        test_acc vs k_frac: geo + 4 baselines (r50, vitmae panels)
  5. cma_convergence.png         mean CMA-ES fbest trajectory, orig vs shuffled
"""
import os
import glob
import re
import sys
import traceback
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

_LOG = open(r"d:\cg-kwta\_plot_log.txt", "w", encoding="utf-8")
def log(*a):
    _LOG.write(" ".join(str(x) for x in a) + "\n"); _LOG.flush()

BASE = r"d:\my projects\PythonProject4"
REB = os.path.join(BASE, "rebuttal")
REBDIM = os.path.join(BASE, "rebuttal_dim")
GMORIG = os.path.join(BASE, "rebuttal_geo_m_orig")
OUT = r"d:\cg-kwta\Figures\summary"
os.makedirs(OUT, exist_ok=True)

TAG = {"r50": "r50", "vit": "vit", "vitmae": "vit_mae"}
DATASETS = ["cifar10", "stl10"]
MODELS = ["r50", "vit"]
C_GEOR = "#1f77b4"
C_GEOM = "#ff7f0e"
C_BL = {"magnitude": "#d62728", "variance": "#9467bd", "rate": "#8c564b", "probe_weight": "#e377c2"}


def load_accs(path):
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return {r["k_frac"]: r["test_acc"] for r in d.get("runs", []) if r.get("test_acc") is not None}


def load_seed_map(folder, pat):
    out = {}
    for f in glob.glob(os.path.join(folder, pat)):
        md = re.search(r"_sd(\d+)\.ya?ml$", os.path.basename(f))
        if md:
            out[int(md.group(1))] = load_accs(f)
    return out


def load_null(folder):
    out = {}
    for f in glob.glob(os.path.join(folder, "*_multi_shuffle.yaml")):
        md = re.search(r"_sd(\d+)_multi_shuffle\.ya?ml$", os.path.basename(f))
        if not md:
            continue
        sd = int(md.group(1))
        with open(f, encoding="utf-8") as fh:
            idx = yaml.safe_load(fh)
        reps = []
        for rel in idx.get("runs_by_repeat", []):
            p = rel if os.path.isabs(rel) else os.path.join(folder, rel)
            if os.path.exists(p):
                reps.append(load_accs(p))
        per_k = {}
        for k in reps[0]:
            vals = [r.get(k) for r in reps if r.get(k) is not None]
            if vals:
                per_k[k] = float(np.mean(vals))
        out[sd] = per_k
    return out


def get_orig(ds, m, gm):
    tag = f"{ds}_{TAG[m]}"
    if gm == "geo_r":
        return load_seed_map(os.path.join(REBDIM, f"{ds}_{m}_d3"),
                             f"results_{tag}_geo_r_sigma0.50_sd[0-9].yaml")
    if m == "r50" and ds == "cifar10":
        return load_seed_map(os.path.join(BASE, "cifar10_r50_meanstdrate"),
                             "results_cifar10_r50_geo_m_sd[0-9].yaml")
    return load_seed_map(os.path.join(GMORIG, f"{ds}_{m}"),
                         f"results_{tag}_geo_m_sd[0-9].yaml")


def per_seed_lowk_gap(orig, null, hi=0.30):
    seeds = sorted(set(orig) & set(null))
    gaps = []
    for s in seeds:
        g = [orig[s][k] - null[s][k] for k in orig[s] if k in null[s] and k <= hi]
        if g:
            gaps.append(np.mean(g))
    return np.array(gaps)


def holm(pv):
    n = len(pv); order = np.argsort(pv); adj = np.ones(n)
    for rank, i in enumerate(order):
        adj[i] = min(1.0, pv[i] * (n - rank))
    for i in range(1, n):
        a, b = order[i-1], order[i]
        if adj[b] < adj[a]:
            adj[b] = adj[a]
    return adj


# ---------------- 1. shuffle gap by combo ----------------
combos = []
log("== section 1: shuffle_gap_by_combo ==")
for ds in DATASETS:
    for m in MODELS:
        for gm in ["geo_r", "geo_m"]:
            orig = get_orig(ds, m, gm)
            null = load_null(os.path.join(REB, f"{ds}_{m}_{gm}"))
            gaps = per_seed_lowk_gap(orig, null)
            # Holm significance across k_fracs
            seeds = sorted(set(orig) & set(null))
            ks = sorted({k for sd in seeds for k in orig[sd] if k in null[sd] and k <= 0.30})
            pv = []
            for k in ks:
                d = np.array([orig[sd][k] - null[sd][k] for sd in seeds if k in orig[sd] and k in null[sd]])
                if len(d) >= 2:
                    # two-sided Wilcoxon primary analysis (no one-sided halving)
                    p2 = stats.wilcoxon(d).pvalue if len(d) >= 10 else stats.ttest_1samp(d, 0.0).pvalue
                    pv.append(p2 if np.isfinite(p2) else 1.0)
            sig = int((holm(np.array(pv)) < 0.05).sum()) if pv else 0
            combos.append({"ds": ds, "m": m, "gm": gm, "gaps": gaps,
                           "mean": gaps.mean() if len(gaps) else 0.0,
                           "sem": gaps.std(ddof=1) / np.sqrt(len(gaps)) if len(gaps) > 1 else 0.0,
                           "sig_lowk": sig, "n_lowk": len(ks)})

labels = [f"{c['ds'][:5]}_{c['m']}" for c in combos]
xs = np.arange(len(combos))
fig, ax = plt.subplots(figsize=(11, 4.6))
w = 0.38
for off, gm in [(0.0, "geo_r"), (w, "geo_m")]:
    sel = [c for c in combos if c["gm"] == gm]
    x = [xs[i] + off for i, c in enumerate(combos) if c["gm"] == gm]
    mu = [c["mean"] for c in sel]
    se = [c["sem"] for c in sel]
    col = C_GEOR if gm == "geo_r" else C_GEOM
    ax.bar(x, mu, width=w, yerr=se, capsize=3, color=col, alpha=0.85, label=gm)
    for c, xx in zip(sel, x):
        star = "*" if c["sig_lowk"] == c["n_lowk"] and c["n_lowk"] > 0 else ("+" if c["sig_lowk"] > 0 else "ns")
        ax.text(xx, c["mean"] + c["sem"] + 0.004, star, ha="center", fontsize=9)
ax.axhline(0, color="k", linewidth=0.8)
ax.set_xticks(xs + w / 2)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("low-k shuffle gap (orig - mean of 5 perms)")
ax.set_title("Shuffle gap by combo (Holm-corrected low-k: * all sig, + some, ns none)")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
p1 = os.path.join(OUT, "shuffle_gap_by_combo.png")
plt.savefig(p1, dpi=200); plt.close(); print("saved:", p1)

# ---------------- 2. d sensitivity (2x2: coord x model, 3 datasets per panel) ----------------
DIMS = os.path.join(BASE, "dim_sweep")
COORD_TMPL = {"geo_r": "results_{tag}_geo_r{sfx}_sigma0.50_sd*.yaml",
              "geo_pca": "results_{tag}_geo_pca{sfx}_sd*.yaml"}
DS_COL = {"cifar10": "#1f77b4", "cifar100": "#ff7f0e", "stl10": "#2ca02c"}
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6))
for row, gm in enumerate(["geo_r", "geo_pca"]):
    for col, m in enumerate(["r50", "vit"]):
        ax = axes[row, col]
        for ds in ["cifar10", "cifar100", "stl10"]:
            tag = f"{ds}_{TAG[m]}"
            vals = []
            for d in [1, 2, 3, 5, 8]:
                folder = os.path.join(DIMS, gm, f"{ds}_{m}_d{d}")
                if not os.path.isdir(folder):
                    vals.append(np.nan)
                    continue
                orig = load_seed_map(folder, COORD_TMPL[gm].format(tag=tag, sfx=""))
                shuf = load_seed_map(folder, COORD_TMPL[gm].format(tag=tag, sfx="_shuf"))
                g = per_seed_lowk_gap(orig, shuf)
                vals.append(g.mean() if len(g) else np.nan)
            ax.plot([1, 2, 3, 5, 8], vals, marker="o", color=DS_COL[ds], label=ds, linewidth=1.8)
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set_xlabel("coordinate dimension d")
        ax.set_title(f"{'RandProj' if gm == 'geo_r' else 'PCA'} — {m}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
axes[1, 0].set_ylabel("low-k shuffle gap")
axes[0, 0].set_ylabel("low-k shuffle gap")
fig.suptitle("d-sensitivity: low-k shuffle gap vs dimension (same-run orig - shuffled)", y=0.99)
plt.tight_layout()
p2 = os.path.join(OUT, "d_sensitivity.png")
plt.savefig(p2, dpi=200); plt.close(); print("saved:", p2)

# ---------------- 3. gap vs k_frac (panels by model) ----------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
for ax, m in zip(axes, ["r50", "vit"]):
    for ds, gm, col, ls in [("cifar10", "geo_r", C_GEOR, "-"), ("stl10", "geo_r", C_GEOR, "--"),
                            ("cifar10", "geo_m", C_GEOM, "-"), ("stl10", "geo_m", C_GEOM, "--")]:
        orig = get_orig(ds, m, gm)
        null = load_null(os.path.join(REB, f"{ds}_{m}_{gm}"))
        seeds = sorted(set(orig) & set(null))
        if len(seeds) < 2:
            continue
        ks = sorted({k for sd in seeds for k in orig[sd] if k in null[sd]})
        ys = []
        for k in ks:
            d = np.array([orig[sd][k] - null[sd][k] for sd in seeds if k in orig[sd] and k in null[sd]])
            ys.append(d.mean())
        ax.plot(ks, ys, color=col, linestyle=ls, marker="o", ms=3,
                label=f"{ds} {gm}", linewidth=1.6)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xlabel("k_frac")
    ax.set_title(f"{m}: shuffle gap vs k_frac")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_ylabel("gap (orig - null)")
plt.tight_layout()
p3 = os.path.join(OUT, "gap_vs_kfrac.png")
plt.savefig(p3, dpi=200); plt.close(); print("saved:", p3)

# ---------------- 4. geo vs baselines (r50, vitmae) ----------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for ax, (ds, m) in zip(axes, [("cifar10", "r50"), ("cifar10", "vitmae")]):
    tag = f"{ds}_{TAG[m]}"
    curves = {}
    if m == "r50":
        curves["geo_r"] = load_seed_map(os.path.join(REBDIM, "cifar10_r50_d3"),
                                        "results_cifar10_r50_geo_r_sigma0.50_sd[0-9].yaml")
        curves["geo_m"] = load_seed_map(os.path.join(BASE, "cifar10_r50_meanstdrate"),
                                        "results_cifar10_r50_geo_m_sd[0-9].yaml")
    else:
        curves["geo_r"] = load_seed_map(os.path.join(BASE, "rebuttal_mae_verify", f"{ds}_{m}"),
                                        f"results_{tag}_geo_r_sigma0.50_sd[0-9].yaml")
    for b in ["magnitude", "variance", "rate", "probe_weight"]:
        curves[b] = load_seed_map(os.path.join(BASE, f"{ds}_{m}_{b}"),
                                  f"results_{tag}_{b}_sd[0-9].yaml")
    for name, cmap in [("geo_r", C_GEOR), ("geo_m", C_GEOM),
                       ("magnitude", C_BL["magnitude"]), ("variance", C_BL["variance"]),
                       ("rate", C_BL["rate"]), ("probe_weight", C_BL["probe_weight"])]:
        if name not in curves or not curves[name]:
            continue
        seeds = sorted(curves[name])
        ks = sorted({k for sd in seeds for k in curves[name][sd]})
        mu, lo, hi = [], [], []
        for k in ks:
            v = np.array([curves[name][sd][k] for sd in seeds if k in curves[name][sd]])
            if len(v) == 0:
                continue
            mu.append(v.mean())
            sem = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
            crit = stats.t.ppf(0.975, len(v) - 1) if len(v) > 1 else 0
            lo.append(v.mean() - crit * sem); hi.append(v.mean() + crit * sem)
        ls = "-" if name.startswith("geo") else "--"
        ax.plot(ks, mu, ls, color=cmap, label=name, linewidth=1.8)
        ax.fill_between(ks, lo, hi, color=cmap, alpha=0.12)
    ax.set_xlabel("k_frac")
    ax.set_ylabel("test_acc (mean, 95% CI)")
    ax.set_title(f"{ds} {m}: geo vs informative-neuron baselines")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
p4 = os.path.join(OUT, "geo_vs_baselines.png")
plt.savefig(p4, dpi=200); plt.close(); print("saved:", p4)

# ---------------- 5. CMA-ES convergence (full dim_sweep, per condition panels) ----------------
log("== section 5: cma_convergence ==")
GM_COL = {"geo_r": C_GEOR, "geo_pca": "#2ca02c"}
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
improvement_stats = {}   # (gm, cond) -> (mean, median, n)
for panel, cond, sfx in [(0, "orig", ""), (1, "shuffled", "_shuf")]:
    ax = axes[panel]
    for gm in ["geo_r", "geo_pca"]:
        hist = []
        glob_pat = (f"results_*_geo_r{sfx}_sigma0.50_sd*.yaml" if gm == "geo_r"
                    else f"results_*_geo_pca{sfx}_sd*.yaml")
        for f in glob.glob(os.path.join(DIMS, gm, "**", glob_pat), recursive=True):
            if "_multi_shuffle" in f:
                continue
            try:
                with open(f, encoding="utf-8") as fh:
                    d = yaml.safe_load(fh)
            except Exception:
                continue
            for r in d.get("runs", []):
                h = r.get("cma_fbest_history")
                if h and len(h) >= 5:
                    hist.append(np.array(h, dtype=float))
        if not hist:
            continue
        L = min(len(h) for h in hist)
        arr = np.vstack([h[:L] for h in hist])
        norm = arr / np.maximum(arr[:, 0:1], 1e-9)
        mu = norm.mean(axis=0)
        sem = norm.std(axis=0, ddof=1) / np.sqrt(len(norm))
        crit = stats.t.ppf(0.975, len(norm) - 1)
        ax.plot(range(L), mu, color=GM_COL[gm], label=f"{gm} (n={len(hist)})", linewidth=2)
        ax.fill_between(range(L), mu - crit * sem, mu + crit * sem, color=GM_COL[gm], alpha=0.12)
        rel = (arr[:, 0] - arr[:, -1]) / np.maximum(np.abs(arr[:, 0]), 1e-9)
        improvement_stats[(gm, cond)] = (float(rel.mean()), float(np.median(rel)), len(hist))
    ax.axhline(1.0, color="k", linewidth=0.6, linestyle=":")
    ax.set_xlabel("CMA-ES iteration")
    ax.set_title(cond)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_ylabel("best-so-far fbest / fbest(iter 0)")
fig.suptitle("CMA-ES best-so-far trajectories (dim_sweep, mean +/- 95% CI)", y=0.99)
plt.tight_layout()
p5 = os.path.join(OUT, "cma_convergence.png")
plt.savefig(p5, dpi=200); plt.close(); print("saved:", p5)

# ---------------- 6. optimization headroom: orig vs shuffled improvement ----------------
keys = ["geo_r", "geo_pca"]
x = np.arange(len(keys))
w = 0.35
fig, ax = plt.subplots(figsize=(6.5, 4.2))
mean_m = [improvement_stats.get((g, "orig"), (np.nan,) * 3)[0] * 100 for g in keys]
mean_s = [improvement_stats.get((g, "shuffled"), (np.nan,) * 3)[0] * 100 for g in keys]
med_m = [improvement_stats.get((g, "orig"), (np.nan,) * 3)[1] * 100 for g in keys]
med_s = [improvement_stats.get((g, "shuffled"), (np.nan,) * 3)[1] * 100 for g in keys]
ax.bar(x - w / 2 - 0.04, mean_m, width=w * 0.9, color=GM_COL["geo_r"], alpha=0.9, label="original coords")
ax.bar(x + w / 2 + 0.04, mean_s, width=w * 0.9, color="#c3c9f2", label="shuffled coords")
for i, g in enumerate(keys):
    n_o = improvement_stats.get((g, "orig"), (0, 0, 0))[2]
    print(f"improvement {g}: orig mean={mean_m[i]:.1f}% (n={n_o})  shuf mean={mean_s[i]:.1f}%")
ax.set_xticks(x)
ax.set_xticklabels(["RandProj", "PCA"])
ax.set_ylabel("relative improvement over 20 iters (%)")
ax.set_title("Search headroom: best-so-far improvement,\noriginal vs shuffled coordinates")
ax.grid(True, alpha=0.3, axis="y")
ax.legend(fontsize=9)
plt.tight_layout()
p6 = os.path.join(OUT, "search_headroom.png")
plt.savefig(p6, dpi=200); plt.close(); print("saved:", p6)

print("all summary figures done.")
