#!/usr/bin/env python3
"""CMA-ES convergence statistics for Reviewer R4.5 / R4.6.

Reads result YAMLs (that contain `cma_fbest_history` per k_frac run) from a
d-sweep root, and reports:

  1. exact combo composition  (n_models x n_datasets x n_dims x n_seeds x n_kfrac)
  2. per-run ridge-fit count (separate from the nominal "budget" B = lambda * T)
  3. best-so-far objective trajectory (normalized to its first iter) + CI band
  4. CAUTION: the "small last-step" metric is a *best-so-far plateau* indicator
     only. Without a larger-budget control it must NOT be claimed as
     evidence of a near-optimal solution.

Usage:
    python cma_convergence_stats.py --root d:\\my projects\\PythonProject4\\dim_sweep
    python cma_convergence_stats.py --root d:\\...\\dim_sweep --csv stats.csv
"""
import argparse
import glob
import os
import re
import yaml
import numpy as np
from scipy import stats


def find_runs(root):
    """Return a list of (file, list_of_fbest_history) for all geo-yamls under root."""
    runs = []          # (path, k_frac, history)
    files = sorted(glob.glob(os.path.join(root, "**", "results_*.yaml"), recursive=True))
    for f in files:
        b = os.path.basename(f)
        if "multi_shuffle" in b:
            continue
        try:
            with open(f, encoding="utf-8") as fh:
                d = yaml.safe_load(fh)
        except Exception:
            continue
        for r in d.get("runs", []):
            h = r.get("cma_fbest_history")
            if h:
                runs.append((f, r.get("k_frac"), [float(v) for v in h]))
    return files, runs


def combo_tuple(path, root):
    """Derive (dataset, model, dim, seed, shuffle) from relative path + filename."""
    rel = os.path.relpath(path, root)
    parts = rel.replace("\\", "/").split("/")
    # .../<dataset>_<model>_d<dim>/results_<tag>_geo_r<sfx>[_sigma0.50]_sd<seed>.yaml
    dim = -1
    for p in parts:
        m = re.search(r"_d(\d+)$", p)
        if m:
            dim = int(m.group(1))
            ds_model_dir = p
    name = os.path.basename(path)
    m = re.search(r"^results_([a-z0-9]+)_([a-z0-9_]+?)_geo_", name)
    ds, model = (m.group(1), m.group(2)) if m else ("?", "?")
    sm = re.search(r"_sd(\d+)\.ya?ml$", name)
    seed = int(sm.group(1)) if sm else -1
    sfx = "_shuf" if ("_shuf_" in name or name.count("shuf")) else ""
    return ds, model, dim, seed, ("shuf" if "_shuf" in name else "orig")


def summarize(root, args):
    files, runs = find_runs(root)
    if not runs:
        print(f"[no cma_fbest_history runs] under {root}")
        return

    # combo decomposition from actual file names
    combos = {}
    n_kfrac_total = 0
    hist_lens = []
    for f, kfrac, h in runs:
        ds, model, dim, seed, cond = combo_tuple(f, root)
        combos.setdefault((ds, model, dim, seed, cond), []).append(kfrac)
        n_kfrac_total += 1
        hist_lens.append(len(h))

    uniq = sorted({(ds, model, dim, seed, cond) for (ds, model, dim, seed, cond) in combos})
    n_ds = len({c[0] for c in uniq})
    n_mod = len({c[1] for c in uniq})
    n_dim = len({c[2] for c in uniq})
    n_seed = len({c[3] for c in uniq})
    n_cond = len({c[4] for c in uniq})

    print("=" * 80)
    print(f"Root: {root}")
    print(f"Result files scanned: {len(files)}  runs (k_frac) with history: {n_kfrac_total}")
    print(f"Combo decomposition:")
    print(f"  datasets   = {n_ds}   {sorted({c[0] for c in uniq})}")
    print(f"  models     = {n_mod}   {sorted({c[1] for c in uniq})}")
    print(f"  dims       = {n_dim}   {sorted({c[2] for c in uniq})}")
    print(f"  seeds      = {n_seed}")
    print(f"  conditions = {n_cond}  (orig/shuffled)")
    print(f"  k_frac per combo = {sorted({combos[c].__len__() for c in combos})}")
    print(f"  => implied total runs (check vs reported 4000/6000): {n_kfrac_total}")
    print(f"  history length distribution: min={min(hist_lens)} max={max(hist_lens)}")

    # ---- per-run ridge fit count (separate from nominal budget) ----
    # run() in the main script:
    #   1 (initial evaluate) + lambda*T (each f() = one ridge fit) + 1 (best-params final eval)
    # lambda = CMA-default popsize = 4 + floor(3*ln(N)), N = 1 + d
    print("\nPer-run ridge fit count (nominal budget B = lambda*T is NOT the full count):")
    for d in sorted({c[2] for c in uniq}):
        N = 1 + d
        lam = 4 + int(np.floor(3 * np.log(max(N, 1))))
        T = int(round(np.median(hist_lens)))
        ridge = 1 + lam * T + 1
        print(f"  d={d}: N={N} param, lambda={lam}, T={T} iters -> "
              f"ridge fits/run = 1+{lam}*{T}+1 = {ridge}  (nominal budget B={lam*T})")

    # ---- convergence trajectory (best-so-far, normalized to iter0) ----
    # split by condition (orig vs shuffled) because they converge differently
    from collections import defaultdict
    by_cond = defaultdict(list)
    for f, kfrac, h in runs:
        ds, model, dim, seed, cond = combo_tuple(f, root)
        by_cond[cond].append([float(v) for v in h])

    for cond in ["orig", "shuf"]:
        if cond not in by_cond:
            continue
        hist = [np.array(h, dtype=float) for h in by_cond[cond]]
        L = min(len(h) for h in hist)
        arr = np.vstack([h[:L] for h in hist])
        base = np.maximum(arr[:, 0:1], 1e-9)
        norm = arr / base
        mu = norm.mean(axis=0)
        sem = norm.std(axis=0, ddof=1) / np.sqrt(len(norm))
        crit = stats.t.ppf(0.975, len(norm) - 1)
        rel = (arr[:, 0] - arr[:, -1]) / np.maximum(np.abs(arr[:, 0]), 1e-9)
        last = np.abs(arr[:, -1] - arr[:, -2]) / np.maximum(np.abs(arr[:, 0]), 1e-9)

        print(f"\n[{cond}] runs={len(norm)} best-so-far trajectory (normalized to iter0 = 1.0):")
        for i in range(L):
            print(f"  iter {i:2d}: {mu[i]:.4f}  (CI {mu[i]-crit*sem[i]:.4f} .. {mu[i]+crit*sem[i]:.4f})")
        print(f"  Relative improvement (f0 - fT)/|f0|  : mean={rel.mean()*100:.1f}%  median={np.median(rel)*100:.1f}%")
        print(f"  Last-step |fT - f(T-1)|/|f0|          : mean={last.mean()*1e4:.2f}e-4  median={np.median(last)*1e4:.2f}e-4")
        print(f"  Runs with last-step < 0.1% of |f0|    : {(last < 1e-3).mean()*100:.1f}%")
        if args.csv:
            with open(args.csv.replace(".csv", f"_{cond}.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["iter", "mean", "sem", "ci_lo", "ci_hi", "n"])
                for i in range(L):
                    w.writerow([i, f"{mu[i]:.6f}", f"{sem[i]:.6f}",
                                f"{mu[i]-crit*sem[i]:.6f}", f"{mu[i]+crit*sem[i]:.6f}", len(norm)])
            print(f"  saved trajectory: {args.csv.replace('.csv', f'_{cond}.csv')}")

    print("\n" + "=" * 80)
    print("CAUTION (must be worded this way in the paper):")
    print("  - 'last step' is a BEST-SO-FAR plateau indicator, NOT the CMA-ES step size sigma.")
    print("  - A plateau in best-so-far does NOT prove a near-optimal solution: without a")
    print("    larger-budget (T>20) control, budget adequacy is not established.")
    print("=" * 80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="root dir containing dim_sweep result folders")
    ap.add_argument("--csv", default=None, help="optional csv to save the mean trajectory")
    args = ap.parse_args()
    summarize(args.root, args)


if __name__ == "__main__":
    main()
