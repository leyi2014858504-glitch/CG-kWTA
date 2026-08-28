# plot_4modes.py
import argparse
import glob
import os
import re
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import t as student_t
except Exception:
    student_t = None

SEED_RE = re.compile(r"(\d+)\.ya?ml$")

MODE_PATTERNS = {
    "non-shuffled": "results_cifar10_dino_geo_pca_sd*.yaml",
    "shuffled": "results_cifar10_dino_geo_pca_shuf_sd*.yaml",
}

def read_one_yaml(path, mode):
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)

    m = SEED_RE.search(os.path.basename(path))
    seed = int(m.group(1)) if m else None

    rows = []
    for r in obj["runs"]:
        rows.append({
            "mode": mode,
            "seed": seed,
            "k_frac": float(r["k_frac"]),
            "test_acc": None if r.get("test_acc", None) is None else float(r["test_acc"]),
            "elapsed_sec": None if r.get("elapsed_sec", None) is None else float(r["elapsed_sec"]),
        })
    return rows

def load_all(data_dir):
    all_rows = []
    for mode, pat in MODE_PATTERNS.items():
        paths = sorted(glob.glob(os.path.join(data_dir, pat)))
        if not paths:
            print(f"[WARN] no files for mode={mode}, pattern={pat}")
        for p in paths:
            all_rows.extend(read_one_yaml(p, mode))
    df = pd.DataFrame(all_rows)
    df = df.dropna(subset=["k_frac"])
    return df

def agg_ci95(df, value_col):
    """Aggregate mean with 95% confidence interval over seeds (t-interval)."""
    d = df.dropna(subset=[value_col]).copy()
    g = d.groupby(["mode", "k_frac"])[value_col].agg(["mean", "std", "count"]).reset_index()
    g = g.rename(columns={"count": "n"})
    # standard error
    g["sem"] = g["std"] / np.sqrt(g["n"].clip(lower=1))
    if student_t is None:
        # fallback: normal approx
        crit = 1.96
        g["ci_lo"] = g["mean"] - crit * g["sem"]
        g["ci_hi"] = g["mean"] + crit * g["sem"]
    else:
        # t critical value depends on df = n-1; for n=1 -> CI degenerates to mean
        dfs = (g["n"] - 1).clip(lower=1)
        crit = student_t.ppf(0.975, dfs)
        g["ci_lo"] = g["mean"] - crit * g["sem"]
        g["ci_hi"] = g["mean"] + crit * g["sem"]
        g.loc[g["n"] <= 1, ["ci_lo", "ci_hi"]] = g.loc[g["n"] <= 1, ["mean", "mean"]].to_numpy()
    return g[["mode", "k_frac", "mean","std","sem", "ci_lo", "ci_hi", "n"]]

def plot_curve_with_band(ax, g_mode, x="k_frac", y_mean="mean",
                         y_lo="ci_lo", y_hi="ci_hi", label=None,
                         marker="o", markersize=4):
    g_mode = g_mode.sort_values(x)
    ax.plot(g_mode[x], g_mode[y_mean], label=label,
            linewidth=2.2, marker=marker, markersize=markersize)
    ax.fill_between(g_mode[x], g_mode[y_lo], g_mode[y_hi], alpha=0.18)

def main():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "legend.title_fontsize": 13,
        "lines.linewidth": 2.2,
    })

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=".", help="directory containing yaml files")
    ap.add_argument("--out_dir", type=str, default=".", help="output directory for png")
    args = ap.parse_args()

    df = load_all(args.data_dir)

    # --- Figure 1: k_frac vs test_acc ---
    g_acc = agg_ci95(df, "test_acc")
    plt.figure(figsize=(6.0, 4.0))
    ax = plt.gca()
    for mode in MODE_PATTERNS.keys():
        gm = g_acc[g_acc["mode"] == mode]
        if len(gm) == 0:
            continue
        plot_curve_with_band(ax, gm, label=mode)
    ax.set_xlabel("k_frac")
    ax.set_ylabel("test_acc")
    ax.set_title("k_frac vs test_acc")
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.9, edgecolor="0.8")
    out1 = os.path.join(args.out_dir, "kfrac_vs_testacc.png")
    plt.tight_layout()
    plt.savefig(out1, dpi=200)
    print("saved:", out1)
    plt.close()

    # --- Figure 2: time for low sparsity region (k_frac <= 0.2) ---
    low = df[df["k_frac"] <= 0.2].copy()
    g_t = agg_ci95(low, "elapsed_sec")
    plt.figure(figsize=(8.5, 5.0))
    ax = plt.gca()
    for mode in MODE_PATTERNS.keys():
        gm = g_t[g_t["mode"] == mode]
        if len(gm) == 0:
            continue
        plot_curve_with_band(ax, gm, label=mode)
    ax.set_xlabel("k_frac (<= 0.2)")
    ax.set_ylabel("elapsed_sec (mean over seeds)")
    ax.set_title("Low-k_frac runtime (mean, 95% CI band over seeds)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out2 = os.path.join(args.out_dir, "lowk_time.png")
    plt.tight_layout()
    plt.savefig(out2, dpi=200)
    print("saved:", out2)
    plt.close()
    df.to_csv(os.path.join(args.out_dir, "kfrac_long.csv"), index=False)

    g_acc.to_csv(os.path.join(args.out_dir, "kfrac_testacc_ci95.csv"), index=False)

    low = df[df["k_frac"] <= 0.2].copy()
    g_t = agg_ci95(low, "elapsed_sec")
    g_t.to_csv(os.path.join(args.out_dir, "kfrac_time_ci95_lowk.csv"), index=False)

if __name__ == "__main__":
    main()
