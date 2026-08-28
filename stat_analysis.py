#!/usr/bin/env python3
"""
Statistical significance analysis with multiple comparison correction.

Reads original (non-shuffled) and multiple shuffle-repeat YAML files,
computes one-sided paired tests at each k_frac, and applies Holm and
Benjamini-Hochberg FDR corrections.

Usage:
    python stat_analysis.py --original <yaml> --shuffled <yaml> [<yaml> ...]
    python stat_analysis.py --multi-shuffle <multi_shuffle_yaml>
    python stat_analysis.py --dir <experiment_folder>
"""
import argparse
import os
import sys
import yaml
import numpy as np
from scipy import stats


def load_test_accs(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    runs = data.get("runs", [])
    return {r["k_frac"]: r.get("test_acc") for r in runs if r.get("test_acc") is not None}


def holm_correction(pvals):
    """Holm (step-down) correction. Returns adjusted p-values."""
    n = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.ones(n)
    for rank, idx in enumerate(order):
        adjusted[idx] = min(1.0, pvals[idx] * (n - rank))
    # enforce monotonicity
    for i in range(1, n):
        idx_prev = order[i - 1]
        idx_curr = order[i]
        if adjusted[idx_curr] < adjusted[idx_prev]:
            adjusted[idx_curr] = adjusted[idx_prev]
    return adjusted


def bh_fdr_correction(pvals):
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.ones(n)
    for rank, idx in enumerate(order):
        adjusted[idx] = min(1.0, pvals[idx] * n / (rank + 1))
    # enforce monotonicity (step-up)
    for i in range(n - 2, -1, -1):
        idx_curr = order[i]
        idx_next = order[i + 1]
        if adjusted[idx_curr] > adjusted[idx_next]:
            adjusted[idx_curr] = adjusted[idx_next]
    return adjusted


def paired_test_one_sided(original_acc, shuffle_accs, k_frac):
    """
    One-sided paired test: H1: original > shuffled.
    Uses Wilcoxon signed-rank if N >= 10, otherwise paired t-test.
    Returns (test_stat, p_value, n_shuffle, mean_delta, std_delta).
    """
    orig_vals = np.array([original_acc[k] for k in sorted(original_acc.keys()) if k == k_frac])
    if len(orig_vals) == 0:
        return None

    orig = orig_vals[0]
    shuf_vals = np.array([shuffle_accs[s].get(k_frac, np.nan) for s in sorted(shuffle_accs.keys())])
    shuf_vals = shuf_vals[~np.isnan(shuf_vals)]
    if len(shuf_vals) == 0:
        return None

    deltas = orig - shuf_vals
    mean_d = float(np.mean(deltas))
    std_d = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0

    if len(deltas) < 2:
        return (0.0, 1.0, len(deltas), mean_d, std_d)

    if len(deltas) >= 10:
        stat, p_two = stats.wilcoxon(deltas, alternative="two-sided")
    else:
        stat, p_two = stats.ttest_1samp(deltas, 0.0)

    p_one = p_two / 2.0 if mean_d > 0 else 1.0 - p_two / 2.0
    return (float(stat), float(p_one), len(deltas), mean_d, std_d)


def analyze_folder(folder, original_pattern="*_sd*.yaml", shuffle_pattern="*_shuf_sd*.yaml"):
    """Analyze all seeds in a folder."""
    import glob

    orig_files = sorted(glob.glob(os.path.join(folder, original_pattern)))
    shuf_files = sorted(glob.glob(os.path.join(folder, shuffle_pattern)))

    if not orig_files:
        print(f"No original YAML files found in {folder}")
        return None
    if not shuf_files:
        print(f"No shuffle YAML files found in {folder}")
        return None

    print(f"Found {len(orig_files)} original, {len(shuf_files)} shuffle files")

    # Aggregate: for each k_frac, collect test_acc across all original seeds and shuffle seeds
    all_k_fracs = set()
    orig_by_k = {}
    shuf_by_k = {i: {} for i in range(len(shuf_files))}

    for f in orig_files:
        accs = load_test_accs(f)
        for k, v in accs.items():
            all_k_fracs.add(k)
            orig_by_k.setdefault(k, []).append(v)

    for i, f in enumerate(shuf_files):
        accs = load_test_accs(f)
        for k, v in accs.items():
            all_k_fracs.add(k)
            shuf_by_k[i].setdefault(k, []).append(v)

    k_fracs = sorted(all_k_fracs)

    # For each k_frac, compute mean original acc and mean shuffle accs per seed
    # Then do paired test
    results = []
    for k in k_fracs:
        orig_vals = orig_by_k.get(k, [])
        if not orig_vals:
            continue
        orig_mean = np.mean(orig_vals)

        shuf_means = []
        for i in range(len(shuf_files)):
            vals = shuf_by_k[i].get(k, [])
            if vals:
                shuf_means.append(np.mean(vals))

        if not shuf_means:
            continue

        shuf_arr = np.array(shuf_means)
        deltas = orig_mean - shuf_arr

        # One-sided test: original > shuffled
        if len(deltas) >= 10:
            stat, p_two = stats.wilcoxon(deltas, alternative="two-sided")
        else:
            stat, p_two = stats.ttest_1samp(deltas, 0.0)

        p_one = float(p_two / 2.0) if np.mean(deltas) > 0 else float(1.0 - p_two / 2.0)

        results.append({
            "k_frac": k,
            "orig_mean": float(orig_mean),
            "shuf_mean": float(np.mean(shuf_arr)),
            "mean_delta": float(np.mean(deltas)),
            "raw_p": p_one,
            "n_shuffle": len(deltas),
        })

    if not results:
        print("No valid k_frac results found.")
        return None

    # Apply multiple comparison corrections
    raw_p = np.array([r["raw_p"] for r in results])
    holm_adj = holm_correction(raw_p)
    bh_adj = bh_fdr_correction(raw_p)

    for i, r in enumerate(results):
        r["holm_p"] = float(holm_adj[i])
        r["bh_fdr_p"] = float(bh_adj[i])
        r["significant_holm"] = holm_adj[i] < 0.05
        r["significant_bh"] = bh_adj[i] < 0.05

    return results


def analyze_multi_shuffle(folder):
    """Multi-permutation analysis for R4.3.

    Expects, per experiment seed sd:
      results_..._sd{sd}.yaml            (original, non-shuffled)
      results_..._shuf_sd{sd}_multi_shuffle.yaml   (index: list of rep files)
    Each index's runs_by_repeat are resolved relative to `folder`.

    For each k_frac: delta = orig(seed) - mean(reps(seed)) over seeds;
    one-sided paired test, then Holm + BH-FDR over k_fracs.
    """
    import glob
    orig_files = sorted(glob.glob(os.path.join(folder, "*_sd[0-9].yaml")))
    idx_files = sorted(glob.glob(os.path.join(folder, "*_multi_shuffle.yaml")))
    orig_files = [f for f in orig_files if "_shuf" not in os.path.basename(f)
                  and "multi_shuffle" not in os.path.basename(f)]
    if not orig_files or not idx_files:
        print(f"multi-shuffle: need original *_sdN.yaml and *_multi_shuffle.yaml in {folder}")
        return None

    # orig per seed: seed -> k_frac -> acc
    orig = {}
    for f in orig_files:
        m = __import__("re").search(r"_sd(\d+)\.ya?ml$", os.path.basename(f))
        if not m:
            continue
        orig[int(m.group(1))] = load_test_accs(f)

    # null per seed: seed -> k_frac -> mean over rep files
    null = {}
    for f in idx_files:
        m = __import__("re").search(r"_sd(\d+)_multi_shuffle\.ya?ml$", os.path.basename(f))
        if not m:
            continue
        sd = int(m.group(1))
        with open(f, "r", encoding="utf-8") as fh:
            idx = yaml.safe_load(fh)
        reps = []
        for rel in idx.get("runs_by_repeat", []):
            p = rel if os.path.isabs(rel) else os.path.join(folder, rel)
            if os.path.exists(p):
                reps.append(load_test_accs(p))
        if not reps:
            continue
        per_k = {}
        for k in reps[0]:
            vals = [r.get(k) for r in reps if r.get(k) is not None]
            if vals:
                per_k[k] = float(np.mean(vals))
        null[sd] = per_k

    seeds = sorted(set(orig) & set(null))
    if not seeds:
        print("multi-shuffle: no seed matched between original and indexes")
        return None

    k_fracs = sorted({k for sd in seeds for k in orig[sd] if k in null[sd]})
    results = []
    for k in k_fracs:
        deltas = np.array([orig[sd][k] - null[sd][k] for sd in seeds
                           if k in orig[sd] and k in null[sd]])
        if len(deltas) < 2:
            continue
        if len(deltas) >= 10:
            stat, p_two = stats.wilcoxon(deltas, alternative="two-sided")
        else:
            stat, p_two = stats.ttest_1samp(deltas, 0.0)
        p_one = float(p_two / 2.0) if np.mean(deltas) > 0 else float(1.0 - p_two / 2.0)
        results.append({
            "k_frac": k,
            "orig_mean": float(np.mean([orig[sd][k] for sd in seeds if k in orig[sd]])),
            "shuf_mean": float(np.mean([null[sd][k] for sd in seeds if k in null[sd]])),
            "mean_delta": float(np.mean(deltas)),
            "raw_p": p_one,
            "n_seeds": int(len(deltas)),
            "n_shuffle": int(idx.get("meta", {}).get("n_shuffle_runs", len(idx.get("runs_by_repeat", [])))),
        })

    if not results:
        print("multi-shuffle: no valid k_frac results")
        return None

    raw_p = np.array([r["raw_p"] for r in results])
    holm_adj = holm_correction(raw_p)
    bh_adj = bh_fdr_correction(raw_p)
    for i, r in enumerate(results):
        r["holm_p"] = float(holm_adj[i])
        r["bh_fdr_p"] = float(bh_adj[i])
        r["significant_holm"] = holm_adj[i] < 0.05
        r["significant_bh"] = bh_adj[i] < 0.05
    return results


def print_results(results, title="Statistical Analysis"):
    if not results:
        return
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"{'k_frac':>8} {'orig':>8} {'shuf':>8} {'delta':>8} {'raw_p':>10} {'holm_p':>10} {'bh_fdr':>10} {'sig(H)':>8} {'sig(B)':>8}")
    print("-" * 90)
    for r in results:
        sig_h = "YES" if r["significant_holm"] else "no"
        sig_b = "YES" if r["significant_bh"] else "no"
        print(f"{r['k_frac']:8.2f} {r['orig_mean']:8.4f} {r['shuf_mean']:8.4f} {r['mean_delta']:+8.4f} "
              f"{r['raw_p']:10.6f} {r['holm_p']:10.6f} {r['bh_fdr_p']:10.6f} {sig_h:>8} {sig_b:>8}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Statistical significance with multiple comparison correction")
    parser.add_argument("--dir", type=str, help="Experiment folder containing YAML files")
    parser.add_argument("--original", type=str, nargs="+", help="Original (non-shuffled) YAML files")
    parser.add_argument("--shuffled", type=str, nargs="+", help="Shuffled YAML files")
    parser.add_argument("--multi-shuffle", type=str,
                        help="Folder containing *_sdN.yaml originals and *_multi_shuffle.yaml indexes")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    if args.dir:
        results = analyze_folder(args.dir)
        if results:
            print_results(results, title=f"Folder: {args.dir}")
            if args.output:
                import csv
                with open(args.output, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                    w.writeheader()
                    w.writerows(results)
                print(f"saved: {args.output}")
    elif args.multi_shuffle:
        results = analyze_multi_shuffle(args.multi_shuffle)
        if results:
            print_results(results, title=f"Multi-shuffle folder: {args.multi_shuffle}")
            if args.output:
                import csv
                with open(args.output, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                    w.writeheader()
                    w.writerows(results)
                print(f"saved: {args.output}")
    elif args.original and args.shuffled:
        # Manual mode
        all_k = set()
        orig_data = {}
        for f in args.original:
            accs = load_test_accs(f)
            for k, v in accs.items():
                all_k.add(k)
                orig_data.setdefault(k, []).append(v)

        shuf_data = {}
        for i, f in enumerate(args.shuffled):
            accs = load_test_accs(f)
            for k, v in accs.items():
                shuf_data.setdefault(k, []).append(v)

        results = []
        for k in sorted(all_k):
            orig_vals = orig_data.get(k, [])
            shuf_vals = shuf_data.get(k, [])
            if not orig_vals or not shuf_vals:
                continue
            orig_mean = np.mean(orig_vals)
            shuf_mean = np.mean(shuf_vals)
            delta = orig_mean - shuf_mean

            if len(shuf_vals) >= 10:
                stat, p_two = stats.mannwhitneyu([orig_mean], shuf_vals, alternative="two-sided")
            else:
                stat, p_two = stats.ttest_1samp(np.array(shuf_vals) - orig_mean, 0.0)

            p_one = float(p_two / 2.0) if delta > 0 else float(1.0 - p_two / 2.0)

            results.append({
                "k_frac": k,
                "orig_mean": float(orig_mean),
                "shuf_mean": float(shuf_mean),
                "mean_delta": float(delta),
                "raw_p": p_one,
                "n_shuffle": len(shuf_vals),
            })

        if results:
            raw_p = np.array([r["raw_p"] for r in results])
            holm_adj = holm_correction(raw_p)
            bh_adj = bh_fdr_correction(raw_p)
            for i, r in enumerate(results):
                r["holm_p"] = float(holm_adj[i])
                r["bh_fdr_p"] = float(bh_adj[i])
                r["significant_holm"] = holm_adj[i] < 0.05
                r["significant_bh"] = bh_adj[i] < 0.05
            print_results(results)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
