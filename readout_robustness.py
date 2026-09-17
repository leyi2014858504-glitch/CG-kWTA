#!/usr/bin/env python3
"""Readout robustness of the shuffle gap (Reviewer: "is the diagnostic just an
artifact of your linear ridge readout?").

The gate selection is fully recoverable from archived artifacts:
    idx = top-k smallest || pts_fixed - best_center ||        (GEOSCORE_MODE="center")
so for every (seed, arm, k) we rebuild the SAME selected neuron set that the
original run used, then re-measure test accuracy with several readouts:

    ridge  a=1     (the original readout; equality with the archived yaml is a
                    hard correctness gate for the reconstruction)
    ridge  a=0.1 / a=10
    logistic       (multinomial, L2, torch LBFGS)
    kNN            (cosine, k=20)

Arms: original pts, and the permutation null pts[perm(seed=sd)] which reproduces
exactly what `shuffle_pts_rows_inplace` did in the main script.

Output: Reports/readout_robustness.csv  + per-config console table.

Usage:
    python readout_robustness.py --configs cifar10_r50_geo_m
    python readout_robustness.py              # all configs
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

BASE = r"d:\my projects\PythonProject4"
DATA = os.path.join(BASE, "data")
REPORTS = r"d:\cg-kwta\Reports"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50]

CONFIGS = {
    # name: (feature_tag, orig_folder, orig_yaml, shuf_folder, shuf_yaml, pts_kind)
    "cifar10_r50_geo_m": (
        "cifar10_r50", os.path.join(BASE, "cifar10_r50_meanstdrate"),
        "results_cifar10_r50_geo_m_sd{sd}.yaml",
        os.path.join(BASE, "rebuttal", "cifar10_r50_geo_m"),
        "results_cifar10_r50_geo_m_shuf_sd{sd}_rep0.yaml", "meanvar", 10),
    "cifar10_r50_geo_r": (
        "cifar10_r50", os.path.join(BASE, "rebuttal_dim", "cifar10_r50_d3"),
        "results_cifar10_r50_geo_r_sigma0.50_sd{sd}.yaml",
        os.path.join(BASE, "rebuttal", "cifar10_r50_geo_r"),
        "results_cifar10_r50_geo_r_shuf_sigma0.50_sd{sd}_rep0.yaml", "randproj", 10),
    "imagenet100_r50_geo_r": (
        "imagenet100_r50", os.path.join(BASE, "imagenet100_r50_fixed_geo_r"),
        "results_imagenet100_r50_geo_r_sigma0.50_sd{sd}.yaml",
        os.path.join(BASE, "imagenet100_r50_fixed_geo_r"),
        "results_imagenet100_r50_geo_r_shuf_sigma0.50_sd{sd}.yaml", "randproj", 5),
    "imagenet100_r50_geo_m": (
        "imagenet100_r50", os.path.join(BASE, "imagenet100_r50_fixed_geo_m"),
        "results_imagenet100_r50_geo_m_sd{sd}.yaml",
        os.path.join(BASE, "imagenet100_r50_fixed_geo_m"),
        "results_imagenet100_r50_geo_m_shuf_sd{sd}.yaml", "meanvar", 5),
    "vitmae_cifar10_geo_r": (
        "cifar10_vit_mae", os.path.join(BASE, "rebuttal_mae_verify", "cifar10_vitmae"),
        "results_cifar10_vit_mae_geo_r_sigma0.50_sd{sd}.yaml",
        os.path.join(BASE, "rebuttal_mae_verify", "cifar10_vitmae"),
        "results_cifar10_vit_mae_geo_r_shuf_sigma0.50_sd{sd}.yaml", "randproj", 10),
}


def load_feats(tag):
    """Load feature tensors fully into RAM.

    NOTE: mmap=True looks tempting but advanced indexing (X[:, sel]) on an
    mmap-backed tensor re-reads the whole file per k-point, which is ~1 GB x
    (seeds x k x arms) of disk traffic and gets killed by the sandbox.
    """
    tr = torch.load(os.path.join(DATA, f"{tag}_train.pt"), map_location="cpu")
    te = torch.load(os.path.join(DATA, f"{tag}_test.pt"), map_location="cpu")
    return tr["X"], tr["y"], te["X"], te["y"]


def selected_idx(pts, center, k_frac):
    d = torch.linalg.norm(pts - center.reshape(1, -1), dim=1)
    k = max(1, int(pts.shape[0] * k_frac))
    return torch.topk(-d, k=k).indices


def prep(sel, Xtr, Xte):
    """Selected columns, z-scored with train statistics (same as the ridge path)."""
    A = Xtr[:, sel].to(DEV).float()
    B = Xte[:, sel].to(DEV).float()
    mu = A.mean(0, keepdim=True)
    sd = A.std(0, keepdim=True).clamp_min(1e-6)
    return (A - mu) / sd, (B - mu) / sd


def acc_ridge(A, B, Yc, Ymu, yte_lab, alpha):
    G = A.T @ A + alpha * torch.eye(A.shape[1], device=A.device)
    W = torch.linalg.solve(G, A.T @ Yc)
    return float(((B @ W + Ymu).argmax(1) == yte_lab).float().mean())


def acc_logistic(A, B, ytr_lab, yte_lab, classes, wd=1e-4, iters=200):
    torch.manual_seed(0)
    W = torch.zeros(A.shape[1], classes, device=DEV, requires_grad=True)
    b = torch.zeros(classes, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=iters, history_size=10)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(A @ W + b, ytr_lab) + wd * (W.pow(2).sum() + b.pow(2).sum())
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return float(((B @ W + b).argmax(1) == yte_lab).float().mean())


@torch.no_grad()
def acc_knn(A, B, ytr_lab, yte_lab, k=20, chunk=512):
    """Cosine kNN (features already z-scored, so cosine distance is meaningful)."""
    An = F.normalize(A, dim=1)
    correct = 0
    for i in range(0, B.shape[0], chunk):
        Bc = F.normalize(B[i:i + chunk], dim=1)
        sim = Bc @ An.T
        topk = sim.topk(k, dim=1).indices
        votes = ytr_lab[topk]
        pred = torch.mode(votes, dim=1).values
        correct += int((pred == yte_lab[i:i + chunk]).sum())
        del Bc, sim, topk, votes
    return correct / B.shape[0]


def permute_pts(pts, seed):
    """Reproduce `shuffle_pts_rows_inplace` exactly.

    The original runs shuffled model.pts on the CUDA device, and torch.randperm
    with a seeded generator is DEVICE-DEPENDENT: verified on cifar10/geo_m/sd0
    (archived shuf 0.7463 / 0.8058 / 0.8577 at k=0.05/0.1/0.2) that a CUDA perm
    matches exactly while a CPU perm does not (0.7085 / 0.7887 / 0.8527).
    """
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    g = torch.Generator(device=dev)
    g.manual_seed(int(seed))
    perm = torch.randperm(pts.shape[0], generator=g, device=dev).cpu()
    return pts[perm]


def run_config(name, args):
    tag, ofolder, opat, sfolder, spat, pts_kind, n_seeds = CONFIGS[name]
    print(f"\n{'='*78}\n{name}  (features={tag}, seeds={n_seeds})\n{'='*78}")
    Xtr, ytr, Xte, yte = load_feats(tag)
    ytr = ytr.long().to(DEV)
    yte = yte.long().to(DEV)
    nc = int(max(ytr.max().item(), yte.max().item())) + 1
    Yc_all = F.one_hot(ytr, nc).float()
    Ymu = Yc_all.mean(0, keepdim=True)
    Yc = Yc_all - Ymu

    rows = []
    for sd in range(n_seeds):
        op = os.path.join(ofolder, opat.format(sd=sd))
        sp = os.path.join(sfolder, spat.format(sd=sd))
        ptsp = os.path.join(ofolder, f"pts_fixed_seed{sd}_{pts_kind}.pt")
        if not (os.path.exists(op) and os.path.exists(sp) and os.path.exists(ptsp)):
            print(f"  [skip] sd{sd}: missing artifact")
            continue
        pts = torch.load(ptsp, map_location="cpu")["pts_fixed"].float()
        oy = yaml.safe_load(open(op, encoding="utf-8"))
        sy = yaml.safe_load(open(sp, encoding="utf-8"))
        o_runs = {round(float(r["k_frac"]), 2): r for r in oy["runs"]}
        s_runs = {round(float(r["k_frac"]), 2): r for r in sy["runs"]}
        pts_shuf = permute_pts(pts, sd)

        # correctness gate: the rebuilt ridge must reproduce BOTH archived arms
        # (the shuffled arm also validates the permutation reproduction)
        gate_ok = True
        for k in KS:
            if k not in o_runs or k not in s_runs:
                continue
            for arm, pmat, yml in (("orig", pts, o_runs), ("shuf", pts_shuf, s_runs)):
                sel = selected_idx(pmat, torch.tensor(yml[k]["best_center"], dtype=torch.float32), k)
                A, B = prep(sel, Xtr, Xte)
                got = acc_ridge(A, B, Yc, Ymu, yte, 1.0)
                arch = float(yml[k]["test_acc"])
                if abs(got - arch) > 2e-3:
                    gate_ok = False
                    print(f"  [WARN] sd{sd} k={k} {arm}: rebuilt {got:.4f} != archived {arch:.4f}")
                del A, B
        if args.validate_only:
            print(f"  sd{sd}: orig reconstruction {'OK' if gate_ok else 'MISMATCH'}")
            continue
        if not gate_ok:
            print(f"  [skip] sd{sd}: reconstruction gate failed")
            continue

        for k in KS:
            if k not in o_runs or k not in s_runs:
                continue
            for arm, pmat, yml in (("orig", pts, o_runs), ("shuf", pts_shuf, s_runs)):
                sel = selected_idx(pmat, torch.tensor(yml[k]["best_center"], dtype=torch.float32), k)
                A, B = prep(sel, Xtr, Xte)
                rows.append({
                    "config": name, "seed": sd, "k_frac": k, "arm": arm,
                    "total_neurons": pts.shape[0],
                    "ridge_a1": acc_ridge(A, B, Yc, Ymu, yte, 1.0),
                    "ridge_a01": acc_ridge(A, B, Yc, Ymu, yte, 0.1),
                    "ridge_a10": acc_ridge(A, B, Yc, Ymu, yte, 10.0),
                    "logistic": acc_logistic(A, B, ytr, yte, nc),
                    "knn20": acc_knn(A, B, ytr, yte, k=20),
                    "archived": float(yml[k]["test_acc"]),
                })
                del A, B
                print(f"  sd{sd} k={k:<5} {arm:<4} done", flush=True)
        del pts, pts_shuf
    del Xtr, Xte
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    all_rows = []
    for name in args.configs:
        all_rows += run_config(name, args)

    if not all_rows:
        print("no rows produced")
        return
    df = pd.DataFrame(all_rows)
    os.makedirs(REPORTS, exist_ok=True)
    out = os.path.join(REPORTS, "readout_robustness.csv")
    # merge with existing rows so per-config runs accumulate instead of overwrite
    if os.path.exists(out):
        old = pd.read_csv(out)
        df = pd.concat([old, df], ignore_index=True)
        df = df.drop_duplicates(subset=["config", "seed", "k_frac", "arm"], keep="last")
    df.to_csv(out, index=False)
    print(f"\nsaved {out}")

    # headline: does the shuffle gap survive across readouts?
    readouts = ["ridge_a1", "ridge_a01", "ridge_a10", "logistic", "knn20"]
    print("\nlow-k (<=0.30) mean shuffle gap per readout:")
    for cfg, g in df.groupby("config"):
        line = [f"{cfg:<24}"]
        for r in readouts:
            lo = g[g.k_frac <= 0.30]
            o = lo[lo.arm == "orig"].groupby("k_frac")[r].mean()
            s = lo[lo.arm == "shuf"].groupby("k_frac")[r].mean()
            line.append(f"{r}={float((o-s).mean()):+.4f}")
        print("  " + "  ".join(line))


if __name__ == "__main__":
    main()
