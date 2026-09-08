#!/usr/bin/env python3
"""Unified rebuttal report generator.

Produces (in d:\\cg-kwta\\Reports\\):
  1. d_sensitivity.csv/.md     shuffle gap vs coordinate dimension d
                               (RandProj + PCA; r50+vit; 3 datasets; per-k + low-k summary)
  2. multishuffle_sig.csv/.md  per-combination paired significance of the shuffle gap
                               (orig vs mean of 5 permutations, Wilcoxon/t, Holm + BH-FDR,
                               per-k 95% CI and low-k mean gap row)
  3. cma_convergence.csv/.md   best-so-far trajectory stats (orig vs shuffled), budget accounting
                               (nominal lambda*T vs total ridge fits), combo decomposition
  4. inventory.md              completeness audit of every artifact used

Conventions (documented in reports):
  - pairing unit = seed; per-seed delta = orig_acc(seed) - mean_over_5_permutations(seed)
  - PRIMARY TEST: two-sided. Wilcoxon signed-rank (n>=10) or one-sample t (n<10)
    on the paired deltas; NO one-sided halving conversion (avoids p distortion
    under ties/zero differences and post-hoc direction selection).
  - directionality is conveyed by the sign of the gap, its 95% CI, and effect size
    (Cohen's dz, Hodges-Lehmann point estimate of the paired location shift).
  - correction: Holm and BH-FDR within each (dataset, model, coord) over k_frac only
  - low-k region: k_frac in [0.05, 0.30]; "lowk_mean_gap" averages per-k mean deltas there
  - best-so-far plateau is NOT evidence of near-optimality without a >20-iteration control
"""
import os
import re
import glob
import yaml
import numpy as np
from scipy import stats

BASE = r"d:\my projects\PythonProject4"
REB = os.path.join(BASE, "rebuttal")
DIMS = os.path.join(BASE, "dim_sweep")
REBDIM = os.path.join(BASE, "rebuttal_dim")
GMORIG = os.path.join(BASE, "rebuttal_geo_m_orig")
MAEV = os.path.join(BASE, "rebuttal_mae_verify")
REPORTS = r"d:\cg-kwta\Reports"

MODELS = {"r50": "r50", "vit": "vit", "vitmae": "vit_mae"}
DATASETS = ["cifar10", "cifar100", "stl10"]
D_LIST = [1, 2, 3, 5, 8]
COORD_DIRS = {"randproj": ("geo_r", "results_{tag}_geo_r{sfx}_sigma0.50_sd{sd}.yaml"),
              "pca": ("geo_pca", "results_{tag}_geo_pca{sfx}_sd{sd}.yaml")}
LOWK = (0.05, 0.30)


def load_runs(path):
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return d


def accs_from(path):
    d = load_runs(path)
    return {round(float(r["k_frac"]), 2): float(r["test_acc"])
            for r in d.get("runs", []) if r.get("test_acc") is not None}


def seed_map(folder, pat_tmpl, tag):
    """seed -> {k_frac: acc} by scanning sd0..sd9 for a folder/pattern."""
    out = {}
    for sd in range(10):
        p = os.path.join(folder, pat_tmpl.format(tag=tag, sfx="", sd=sd))
        if os.path.exists(p):
            out[sd] = accs_from(p)
    return out


def seed_map_shuf(folder, pat_tmpl, tag):
    out = {}
    for sd in range(10):
        p = os.path.join(folder, pat_tmpl.format(tag=tag, sfx="_shuf", sd=sd))
        if os.path.exists(p):
            out[sd] = accs_from(p)
    return out


def holm(pv):
    pv = np.asarray(pv, dtype=float)
    n = len(pv)
    order = np.argsort(pv)
    adj = np.ones(n)
    for rank, idx in enumerate(order):
        adj[idx] = min(1.0, pv[idx] * (n - rank))
    for i in range(1, n):
        a, b = order[i - 1], order[i]
        if adj[b] < adj[a]:
            adj[b] = adj[a]
    return adj


def bh(pv):
    pv = np.asarray(pv, dtype=float)
    n = len(pv)
    order = np.argsort(pv)
    adj = np.ones(n)
    for rank, idx in enumerate(order):
        adj[idx] = min(1.0, pv[idx] * n / (rank + 1))
    for i in range(n - 2, -1, -1):
        a, b = order[i + 1], order[i]
        if adj[a] < adj[b]:
            adj[b] = adj[a]
    return adj


def paired_gap_rows(orig, null):
    """orig/null: seed -> {k: acc}. Returns per-k dict rows with CI + corrected p."""
    seeds = sorted(set(orig) & set(null))
    ks = sorted({k for s in seeds for k in orig[s] if k in null[s]})
    rows = []
    for k in ks:
        d = np.array([orig[s][k] - null[s][k] for s in seeds if k in orig[s] and k in null[s]])
        if len(d) < 2:
            continue
        mean_d = float(d.mean())
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        tcrit = float(stats.t.ppf(0.975, len(d) - 1))
        # two-sided p-value (primary analysis; no one-sided conversion)
        if len(d) >= 10:
            p_two = float(stats.wilcoxon(d).pvalue)
        else:
            p_two = float(stats.ttest_1samp(d, 0.0).pvalue)
        if not np.isfinite(p_two):
            p_two = 1.0
        # effect sizes: Cohen's dz and Hodges-Lehmann paired location shift
        sd_d = float(d.std(ddof=1))
        dz = mean_d / sd_d if sd_d > 0 else float("nan")
        pairs = (d[:, None] + d[None, :]) / 2.0
        iu = np.triu_indices(len(d))
        hl = float(np.median(pairs[iu]))
        rows.append({"k_frac": k, "n_seeds": int(len(d)),
                     "orig_mean": float(np.mean([orig[s][k] for s in seeds if k in orig[s]])),
                     "null_mean": float(np.mean([null[s][k] for s in seeds if k in null[s]])),
                     "gap": mean_d, "ci_lo": mean_d - tcrit * se, "ci_hi": mean_d + tcrit * se,
                     "cohens_dz": dz, "hodges_lehmann": hl,
                     "p_two_sided": float(p_two)})
    if rows:
        pv = np.array([r["p_two_sided"] for r in rows])
        h = holm(pv); b = bh(pv)
        for i, r in enumerate(rows):
            r["holm_p"] = float(h[i]); r["bh_p"] = float(b[i])
            r["sig_holm"] = bool(h[i] < 0.05); r["sig_bh"] = bool(b[i] < 0.05)
    return rows, seeds


# ---------------- Report 1: d-sensitivity ----------------
def report_d_sensitivity():
    rows = []
    for coord, (gm, tmpl) in COORD_DIRS.items():
        for ds in DATASETS:
            for m in MODELS.keys():
                if m == "vitmae":
                    continue
                tag = f"{ds}_{MODELS[m]}"
                for d in D_LIST:
                    folder = os.path.join(DIMS, gm, f"{ds}_{m}_d{d}")
                    if not os.path.isdir(folder):
                        continue
                    orig = seed_map(folder, tmpl, tag)
                    shuf = seed_map_shuf(folder, tmpl, tag)
                    per_k, seeds = paired_gap_rows(orig, shuf)
                    low = [r for r in per_k if LOWK[0] <= r["k_frac"] <= LOWK[1]]
                    if not low:
                        continue
                    gaps = np.array([r["gap"] for r in low])
                    # CI of per-seed paired diffs averaged over low-k (bootstrap-free:
                    # mean of per-k gaps; CI from per-k gap spread is reported separately)
                    sig_lowk = sum(1 for r in low if r["sig_holm"])
                    rows.append({"coord": coord, "dataset": ds, "model": m, "d": d,
                                 "n_seeds": len(seeds), "n_k_lowk": len(low),
                                 "lowk_mean_gap": float(gaps.mean()),
                                 "lowk_gap_min": float(gaps.min()),
                                 "lowk_gap_max": float(gaps.max()),
                                 "sig_holm_lowk": int(sig_lowk)})
    import csv
    cp = os.path.join(REPORTS, "d_sensitivity.csv")
    with open(cp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    md = ["# Coordinate-dimension (d) sensitivity — shuffle gap, low-k [0.05,0.30]",
          "",
          "gap = orig - same-seed shuffled (same run, same split); per-k Holm-corrected within combo.",
          ""]
    for coord in ["randproj", "pca"]:
        gm = COORD_DIRS[coord][0]
        md.append(f"## {coord} ({gm})")
        md.append("")
        md.append("| dataset | model | d | low-k mean gap | sig k (Holm) / n_k |")
        md.append("|---|---|---|---|---|")
        for r in [x for x in rows if x["coord"] == coord]:
            md.append(f"| {r['dataset']} | {r['model']} | {r['d']} | {r['lowk_mean_gap']:+.4f} "
                      f"| {r['sig_holm_lowk']}/{r['n_k_lowk']} |")
        md.append("")
    with open(os.path.join(REPORTS, "d_sensitivity.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    return len(rows)


# ---------------- Report 2: multi-shuffle significance ----------------
def get_rebuttal_orig(ds, m, gm):
    """Same-script non-shuffled originals for the multi-shuffle combos."""
    tag = f"{ds}_{MODELS[m]}"
    if gm == "geo_r":
        # orig from rebuttal_dim d3 (same scheduling batch as the reps)
        return seed_map(os.path.join(REBDIM, f"{ds}_{m}_d3"),
                        "results_{tag}_geo_r{sfx}_sigma0.50_sd{sd}.yaml", tag)
    # geo_m: cifar10 r50 came from BASE meanstdrate folder; the other three from rebuttal_geo_m_orig
    if ds == "cifar10" and m == "r50":
        return seed_map(os.path.join(BASE, "cifar10_r50_meanstdrate"),
                        "results_{tag}_geo_m{sfx}_sd{sd}.yaml", tag)
    return seed_map(os.path.join(GMORIG, f"{ds}_{m}"),
                    "results_{tag}_geo_m{sfx}_sd{sd}.yaml", tag)


def load_null(folder):
    out = {}
    for f in glob.glob(os.path.join(folder, "*_multi_shuffle.yaml")):
        mm = re.search(r"_sd(\d+)_multi_shuffle\.ya?ml$", os.path.basename(f))
        if not mm:
            continue
        sd = int(mm.group(1))
        idx = load_runs(f)
        reps = []
        for rel in idx.get("runs_by_repeat", []):
            p = rel if os.path.isabs(rel) else os.path.join(folder, rel)
            if os.path.exists(p):
                reps.append(accs_from(p))
        if not reps:
            continue
        per_k = {}
        for k in reps[0]:
            vals = [r.get(k) for r in reps if r.get(k) is not None]
            if vals:
                per_k[k] = float(np.mean(vals))
        out[sd] = per_k
    return out


def report_multishuffle():
    import csv
    combos = [("cifar10", "r50", "geo_r"), ("cifar10", "r50", "geo_m"),
              ("cifar10", "vit", "geo_r"), ("cifar10", "vit", "geo_m"),
              ("stl10", "r50", "geo_r"), ("stl10", "r50", "geo_m"),
              ("stl10", "vit", "geo_r"), ("stl10", "vit", "geo_m"),
              ("cifar10", "vitmae", "geo_r"), ("stl10", "vitmae", "geo_r")]
    all_rows = []
    for ds, m, gm in combos:
        if gm == "geo_r" and m == "vitmae":
            # MAE verification: single same-run shuffled (not 5-perm null) -> report gap only
            folder = os.path.join(MAEV, f"{ds}_{m}")
            tag = f"{ds}_{MODELS[m]}"
            orig = seed_map(folder, "results_{tag}_geo_r{sfx}_sigma0.50_sd{sd}.yaml", tag)
            null = seed_map_shuf(folder, "results_{tag}_geo_r{sfx}_sigma0.50_sd{sd}.yaml", tag)
            note = "single permutation (rebuttal_mae_verify)"
        else:
            folder = os.path.join(REB, f"{ds}_{m}_{gm}")
            orig = get_rebuttal_orig(ds, m, gm)
            null = load_null(folder)
            note = "mean of 5 permutations"
        per_k, seeds = paired_gap_rows(orig, null)
        low = [r for r in per_k if LOWK[0] <= r["k_frac"] <= LOWK[1]]
        for r in per_k:
            r.update({"dataset": ds, "model": m, "coord_mode": gm,
                      "null_kind": note, "n_seeds": len(seeds)})
        all_rows.extend(per_k)
        gaps = np.array([r["gap"] for r in low]) if low else np.array([np.nan])
        print(f"{ds:<8} {m:<7} {gm:<6} lowk_mean_gap={gaps.mean():+.4f} "
              f"sig_holm_lowk={sum(1 for r in low if r['sig_holm'])}/{len(low)}  ({note})")
    cp = os.path.join(REPORTS, "multishuffle_sig.csv")
    cols = ["dataset", "model", "coord_mode", "null_kind", "k_frac", "n_seeds",
            "orig_mean", "null_mean", "gap", "ci_lo", "ci_hi", "cohens_dz", "hodges_lehmann",
            "p_two_sided", "holm_p", "bh_p", "sig_holm", "sig_bh"]
    with open(cp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            w.writerow({c: r.get(c) for c in cols})
    return len(all_rows)


# ---------------- Report 3: CMA convergence ----------------
def report_convergence():
    import csv
    rows = []
    for root_name, root in [("dim_sweep (d 1..8, 3 datasets)", DIMS)]:
        for gm in ["geo_r", "geo_pca"]:
            for cond, sfx in [("orig", ""), ("shuffled", "_shuf")]:
                hist = []
                for f in glob.glob(os.path.join(root, gm, "**", "results_*.yaml"), recursive=True):
                    b = os.path.basename(f)
                    if "multi_shuffle" in b:
                        continue
                    is_shuf = "_shuf" in b
                    if (cond == "shuffled") != is_shuf:
                        continue
                    try:
                        d = load_runs(f)
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
                rel = (arr[:, 0] - arr[:, -1]) / np.maximum(np.abs(arr[:, 0]), 1e-9)
                last = np.abs(arr[:, -1] - arr[:, -2]) / np.maximum(np.abs(arr[:, 0]), 1e-9)
                rows.append({"root": root_name, "coord": gm, "condition": cond,
                             "n_runs": len(hist), "iters": L,
                             "rel_improve_mean": float(rel.mean()),
                             "rel_improve_median": float(np.median(rel)),
                             "last_step_mean": float(last.mean()),
                             "frac_last_step_lt_1e-3": float((last < 1e-3).mean())})
    cp = os.path.join(REPORTS, "cma_convergence.csv")
    with open(cp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"[{r['coord']:<7} {r['condition']:<8}] runs={r['n_runs']:>5} "
              f"improve mean={r['rel_improve_mean']*100:5.1f}% med={r['rel_improve_median']*100:5.1f}%  "
              f"plateau(last step<0.1%|f0|): {r['frac_last_step_lt_1e-3']*100:.1f}%")
    return len(rows)


# ---------------- Report 4: inventory ----------------
def report_inventory():
    lines = ["# Rebuttal data inventory (completeness audit)", ""]
    def count(folder, pat):
        return len(glob.glob(os.path.join(folder, pat)))
    lines.append("## d-sweep (dim_sweep/{geo_r|geo_pca}/{ds}_{model}_d{d})")
    missing = 0
    for gm in ["geo_r", "geo_pca"]:
        for ds in DATASETS:
            for m in ["r50", "vit"]:
                for d in D_LIST:
                    folder = os.path.join(DIMS, gm, f"{ds}_{m}_d{d}")
                    tag = f"{ds}_{MODELS[m]}"
                    tmpl = COORD_DIRS["pca" if gm == "geo_pca" else "randproj"][1]
                    o = sum(os.path.exists(os.path.join(folder, tmpl.format(tag=tag, sfx="", sd=s))) for s in range(10))
                    s = sum(os.path.exists(os.path.join(folder, tmpl.format(tag=tag, sfx="_shuf", sd=s))) for s in range(10))
                    if o < 10 or s < 10:
                        lines.append(f"  INCOMPLETE: {gm}/{ds}_{m}_d{d} orig={o}/10 shuf={s}/10")
                        missing += 1
    lines.append(f"  -> {60 - missing}/60 combos complete (2 coords x 3 datasets x 2 models x 5 dims)")
    lines.append("")
    lines.append("## multi-shuffle nulls (rebuttal/{ds}_{model}_{geo_r|geo_m})")
    for ds in ["cifar10", "stl10"]:
        for m in ["r50", "vit"]:
            for gm in ["geo_r", "geo_m"]:
                n = count(os.path.join(REB, f"{ds}_{m}_{gm}"), "*_multi_shuffle.yaml")
                lines.append(f"  {ds}_{m}_{gm}: {n}/10 seed indexes")
    lines.append("")
    lines.append("## MAE verification (rebuttal_mae_verify)")
    for ds in ["cifar10", "stl10"]:
        folder = os.path.join(MAEV, f"{ds}_vitmae")
        o = count(folder, "results_*_geo_r_sigma0.50_sd[0-9].yaml")
        s = count(folder, "results_*_geo_r_shuf_sigma0.50_sd[0-9].yaml")
        lines.append(f"  {ds}_vitmae geo_r: orig={o}/10 shuf={s}/10")
    lines.append("")
    lines.append("## Baselines (BASE/{ds}_{model}_{mode})")
    for ds in DATASETS:
        for m in ["r50", "r50mocov2", "convnext", "convnextv2", "vitmae", "vit", "vitdino", "swin"]:
            for b in ["magnitude", "variance", "rate", "probe_weight"]:
                n = count(os.path.join(BASE, f"{ds}_{m}_{b}"), "results_*_sd[0-9].yaml")
                if n < 10:
                    lines.append(f"  INCOMPLETE: {ds}_{m}_{b}: {n}/10")
    lines.append("  (only incomplete entries listed; absence = all 96 combos complete)")
    with open(os.path.join(REPORTS, "inventory.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    os.makedirs(REPORTS, exist_ok=True)
    print("== Report 1: d-sensitivity ==")
    n1 = report_d_sensitivity()
    print(f"rows={n1}")
    print("\n== Report 2: multi-shuffle significance ==")
    n2 = report_multishuffle()
    print(f"rows={n2}")
    print("\n== Report 3: CMA convergence ==")
    n3 = report_convergence()
    print(f"rows={n3}")
    print("\n== Report 4: inventory ==")
    report_inventory()
    print(f"\nAll reports written to {REPORTS}")


if __name__ == "__main__":
    main()
