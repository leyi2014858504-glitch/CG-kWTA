#!/usr/bin/env python3
"""Plot CMA-ES convergence curves from `cma_fbest_history` recorded in result YAMLs.

Usage:
    python plot_cma_convergence.py --dir <folder> [--seed 0] [--kfrac 0.2] [--out out.png]
"""
import argparse
import glob
import os
import re
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_history(path):
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    out = {}
    for r in d.get("runs", []):
        h = r.get("cma_fbest_history")
        if h:
            out[float(r["k_frac"])] = [float(v) for v in h]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="folder with results_*.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--kfrac", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "results_*.yaml")))
    files = [f for f in files if "_shuf" not in os.path.basename(f)
             and "multi_shuffle" not in os.path.basename(f)]
    if not files:
        print("no yaml files found")
        return

    plt.figure(figsize=(8, 5))
    n_lines = 0
    for f in files:
        m = re.search(r"_sd(\d+)\.ya?ml$", os.path.basename(f))
        sd = int(m.group(1)) if m else None
        if args.seed is not None and sd != args.seed:
            continue
        hist = load_history(f)
        for k, h in sorted(hist.items()):
            if args.kfrac is not None and abs(k - args.kfrac) > 1e-6:
                continue
            label = f"seed={sd}, k={k:.2f}" if args.seed is None else f"k={k:.2f}"
            plt.plot(range(len(h)), h, label=label, linewidth=1.5)
            n_lines += 1
    if n_lines == 0:
        print("no cma_fbest_history found (results were produced before history recording)")
        return
    plt.xlabel("CMA-ES iteration")
    plt.ylabel("best objective fbest")
    plt.title(f"CMA-ES convergence ({os.path.basename(args.dir)})")
    plt.grid(True, alpha=0.3)
    if n_lines <= 12:
        plt.legend(fontsize=8)
    plt.tight_layout()
    out = args.out or os.path.join(args.dir, "cma_convergence.png")
    plt.savefig(out, dpi=200)
    print("saved:", out)


if __name__ == "__main__":
    main()
