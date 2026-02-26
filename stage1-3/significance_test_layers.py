# significance_test_layers.py
import pandas as pd
import numpy as np
from scipy import stats

# ====== 在这里改路径（不需要命令行参数）======
NON_CSV = "test_acc_by_layer_raw.csv"
SHUF_CSV = "test_acc_by_layer_shuffle_raw.csv"
OUT_CSV = "significance_by_layer.csv"
ALPHA = 0.05
# ===========================================

def one_sided_p_from_two_sided(p_two, stat, greater=True):
    """
    把 two-sided p 转成 one-sided p（方向由 greater 控制）
    greater=True 表示 H1: stat > 0（non - shuffle > 0）
    """
    if np.isnan(p_two) or np.isnan(stat):
        return np.nan
    if greater:
        return p_two / 2 if stat > 0 else 1 - p_two / 2
    else:
        return p_two / 2 if stat < 0 else 1 - p_two / 2

def paired_ttest_greater(x, y):
    """
    H1: mean(x - y) > 0
    兼容不同 SciPy 版本：优先用 alternative；否则手动由 two-sided 转 one-sided。
    """
    try:
        res = stats.ttest_rel(x, y, alternative="greater", nan_policy="omit")
        return float(res.statistic), float(res.pvalue)
    except TypeError:
        res = stats.ttest_rel(x, y, nan_policy="omit")
        p_one = one_sided_p_from_two_sided(res.pvalue, res.statistic, greater=True)
        return float(res.statistic), float(p_one)

def wilcoxon_greater(d):
    """
    对差值 d = x - y 做 Wilcoxon signed-rank，H1: median(d) > 0
    """
    try:
        # 注意：当全为0或样本太特殊时可能报错，下面会捕获
        w_stat, p = stats.wilcoxon(d, alternative="greater", zero_method="wilcox", correction=False)
        return float(w_stat), float(p)
    except ValueError:
        return np.nan, np.nan

def main():
    non = pd.read_csv(NON_CSV)
    shuf = pd.read_csv(SHUF_CSV)

    # 强制列名一致（如果你改过 metric 名，可在这里改）
    metric = "test_acc"

    # 对齐：同一 layer, k_frac, seed_idx 的 non/shuffle 配对
    df = non.merge(
        shuf,
        on=["layer", "k_frac", "seed_idx"],
        suffixes=("_non", "_shuf"),
        how="inner",
    )

    # 差值：non - shuffle
    df["delta"] = df[f"{metric}_non"] - df[f"{metric}_shuf"]

    rows = []
    for (layer, k), sub in df.groupby(["layer", "k_frac"], sort=True):
        x = sub[f"{metric}_non"].to_numpy(dtype=float)
        y = sub[f"{metric}_shuf"].to_numpy(dtype=float)
        d = sub["delta"].to_numpy(dtype=float)
        n = int(len(d))

        t_stat, p_t = paired_ttest_greater(x, y)
        w_stat, p_w = wilcoxon_greater(d)

        rows.append({
            "layer": layer,
            "k_frac": float(k),
            "n": n,
            "mean_non": float(np.mean(x)),
            "mean_shuf": float(np.mean(y)),
            "mean_delta(non-shuf)": float(np.mean(d)),
            "median_delta": float(np.median(d)),
            "t_stat": t_stat,
            "p_ttest_one_sided": p_t,
            "wilcoxon_stat": w_stat,
            "p_wilcoxon_one_sided": p_w,
            "ttest_sig@0.05": (p_t < ALPHA) if not np.isnan(p_t) else False,
            "wilcoxon_sig@0.05": (p_w < ALPHA) if not np.isnan(p_w) else False,
        })

    out = pd.DataFrame(rows).sort_values(["k_frac", "layer"]).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")
    print(out)

if __name__ == "__main__":
    main()
