import argparse
import glob
import os
import re
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as student_t


RE_SHUFFLE = re.compile(r"shuffle(\d+)\.ya?ml$", re.IGNORECASE)
RE_RANDOM  = re.compile(r"random_r50(\d+)\.ya?ml$", re.IGNORECASE)
RE_R50     = re.compile(r"r_r50(\d+)\.ya?ml$", re.IGNORECASE)
RE_BESTOF = re.compile(r"sd(\d+)\.ya?ml$", re.IGNORECASE)


def infer_seed_from_filename(path: str):
    base = os.path.basename(path)
    for rgx in (RE_SHUFFLE, RE_RANDOM, RE_R50 , RE_BESTOF):
        m = rgx.search(base)
        if m:
            return int(m.group(1))
    return None


def read_one_yaml(path: str, group: str, metric: str):
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}

    meta = obj.get("meta", {}) or {}
    seed = meta.get("seed", None)
    seed = int(seed) if seed is not None else infer_seed_from_filename(path)

    rows = []
    runs = obj.get("runs", []) or []
    for r in runs:
        if "k_frac" not in r:
            continue
        if metric not in r:
            continue
        rows.append({
            "group": group,
            "seed": seed,
            "k_frac": float(r["k_frac"]),
            "acc": float(r[metric]),
            "file": os.path.basename(path),
        })
    return rows


def load_all(data_dir: str, metric: str, nonshuffle_glob: str, shuffle_glob: str, random_glob: str, bestof_glob):
    patterns = [
        ("nonshuffle", nonshuffle_glob),
        ("shuffle", shuffle_glob),
        ("random", random_glob),
        ("bestof_random", bestof_glob)
    ]

    all_rows = []
    matched = {}

    for group, pat in patterns:
        files = sorted(glob.glob(os.path.join(data_dir, pat)))
        matched[group] = files
        for p in files:
            all_rows.extend(read_one_yaml(p, group=group, metric=metric))

    df = pd.DataFrame(all_rows)
    if len(df) == 0:
        return df, matched

    # Ensure all three columns are present, otherwise dropna will cause a group to disappear
    df = df.dropna(subset=["seed", "k_frac", "acc"]).copy()
    df["seed"] = df["seed"].astype(int)
    df["k_frac"] = df["k_frac"].astype(float)
    df["acc"] = df["acc"].astype(float)

    return df, matched


def add_ci_95(sub: pd.DataFrame) -> pd.DataFrame:
    # sub columns: k_frac, mean, std, n
    sub = sub.sort_values("k_frac").copy()
    mu = sub["mean"].to_numpy(dtype=float)
    sd = sub["std"].to_numpy(dtype=float)
    n = sub["n"].to_numpy(dtype=int)

    lo = np.zeros_like(mu)
    hi = np.zeros_like(mu)

    for i in range(len(mu)):
        if n[i] <= 1 or np.isnan(sd[i]):
            lo[i] = mu[i]
            hi[i] = mu[i]
        else:
            sem = sd[i] / np.sqrt(n[i])
            tcrit = float(student_t.ppf(0.975, df=n[i] - 1))
            lo[i] = mu[i] - tcrit * sem
            hi[i] = mu[i] + tcrit * sem

    sub["ci_lo"] = lo
    sub["ci_hi"] = hi
    return sub


CURVE_STYLE = {
    "nonshuffle":    dict(label="geo_r (non-shuffle)", linestyle="-",  linewidth=2.0, marker="o"),
    "shuffle":       dict(label="geo_r (shuffle)",     linestyle="-",  linewidth=2.0, marker="s"),
    "random":        dict(label="random",              linestyle="--", linewidth=1.5, marker=None),
    "bestof_random": dict(label="best-of-160 random",  linestyle="--", linewidth=1.5, marker=None),
}

def plot_curves(df: pd.DataFrame, out_png: str, title: str):
    g = (
        df.groupby(["group", "k_frac"])["acc"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "n"})
    )
    plt.figure(figsize=(7.2, 4.2))
    for group, style in CURVE_STYLE.items():
        sub = g[g["group"] == group]
        if len(sub) == 0:
            continue
        sub = add_ci_95(sub)
        x  = sub["k_frac"].to_numpy()
        y  = sub["mean"].to_numpy()
        lo = sub["ci_lo"].to_numpy()
        hi = sub["ci_hi"].to_numpy()
        plt.plot(x, y, **style)
        plt.fill_between(x, lo, hi, alpha=0.13)
    plt.xlabel("$k_{\\mathrm{frac}}$")
    plt.ylabel("Test Accuracy")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=".")
    ap.add_argument("--metric", type=str, default="test_acc",
                    choices=["test_acc", "best_acc", "initial_acc"])
    ap.add_argument("--kmax", type=float, default=0.5)

    ap.add_argument("--nonshuffle_glob", type=str, default="results_kfrac_geo_r_r50[0-9].yaml")
    ap.add_argument("--shuffle_glob", type=str, default="results_kfrac_geo_r_r50_shuffle[0-9].yaml")
    ap.add_argument("--random_glob", type=str, default="results_kfrac_random_r50[0-9].yaml")


    ap.add_argument("--out_long_csv", type=str, default="long_3curves.csv")
    ap.add_argument("--bestof_glob", type=str,
                    default="results_kfrac_bestof_random_sd*.yaml")
    ap.add_argument("--out_plot", type=str,
                    default="plot_shuffle_vs_nonshuffle_kle0p5.png")

    args = ap.parse_args()

    df, matched = load_all(
        args.data_dir,
        metric=args.metric,
        nonshuffle_glob=args.nonshuffle_glob,
        shuffle_glob=args.shuffle_glob,
        random_glob=args.random_glob,
        bestof_glob=args.bestof_glob
    )

    # Print matching status to avoid cases where the plot remains the same because no files were read
    print("[Matched files]")
    for k, v in matched.items():
        print(f"  {k}: {len(v)} files")
        if len(v) > 0:
            print(f"    e.g. {os.path.basename(v[0])}")

    if len(df) == 0:
        raise RuntimeError("No rows loaded. Check --data_dir and glob patterns.")

    print("\n[Loaded rows by group]")
    print(df["group"].value_counts())

    df.to_csv(args.out_long_csv, index=False)

    df_plot = df[df["k_frac"] <= args.kmax].copy()
    title = f"{args.metric}: nonshuffle vs shuffle vs random (k_frac <= {args.kmax})"
    plot_curves(df_plot, out_png=args.out_plot, title=title)

    print(f"\nSaved plot: {args.out_plot}")
    print(f"Saved long csv: {args.out_long_csv}")


if __name__ == "__main__":
    main()
