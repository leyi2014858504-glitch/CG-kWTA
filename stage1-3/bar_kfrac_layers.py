# plot_testacc_by_layer.py
import os
import math
import yaml
import numpy as np
import matplotlib.pyplot as plt
import csv

# =========================
# 所有参数都在这里调节（无需运行时传参）
# =========================
BASE_DIR = r"C:\Users\20148\PycharmProjects\PythonProject4\stage1-3"  # 改成你的目录
FILE_PATTERN = "results_kfrac_geo_r_l{layer}_shuffle{seed}.yaml"            # 按你的命名规则

LAYERS = [1, 2, 3, 4]
SEEDS = list(range(10))  # 0~9
KFRACS = [0.05,0.2, 0.5]      # 需要画的两个点
METRIC_KEY = "test_acc"
KFRAC_KEY = "k_frac" # 纵轴指标：testacc
OUTPUT_PNG = os.path.join(BASE_DIR, "test_acc_bar_by_layer_shuffle.png")

# 画图风格参数
FIGSIZE = (7.5, 4.5)
BAR_WIDTH = 0.25
SHOW_ERRORBAR = True      # 是否画 seed 间标准差
ERRORBAR_CAPSIZE = 3
DPI = 200

OUTPUT_CSV_RAW = os.path.join(BASE_DIR, "test_acc_by_layer_shuffle_raw.csv")
OUTPUT_CSV_SUMMARY = os.path.join(BASE_DIR, "test_acc_by_layer_shuffle_summary.csv")

def _almost_equal(a, b, tol=1e-12):
    return abs(float(a) - float(b)) <= tol


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def extract_metric_for_kfrac(yaml_obj, kfrac, metric_key=METRIC_KEY):
    runs = yaml_obj.get("runs", [])
    for r in runs:
        if KFRAC_KEY in r and _almost_equal(r[KFRAC_KEY], kfrac):
            return float(r[metric_key])
    raise KeyError(f"{KFRAC_KEY}={kfrac} not found in runs")

    for r in runs:
        if "kfrac" in r and _almost_equal(r["kfrac"], k_frac):
            if metric_key not in r:
                raise KeyError(f"'{metric_key}' not found for kfrac={k_frac}")
            return float(r[metric_key])

    raise KeyError(f"k_frac={k_frac} not found in runs")


def collect_values():
    """
    返回：
    values[layer][kfrac] = [v_seed0, v_seed1, ...]  (不存在的 seed 文件会跳过)
    """
    values = {layer: {k: [] for k in KFRACS} for layer in LAYERS}

    for layer in LAYERS:
        for seed in SEEDS:
            fname = FILE_PATTERN.format(layer=layer, seed=seed)
            fpath = os.path.join(BASE_DIR, fname)
            if not os.path.isfile(fpath):
                # 文件不存在就跳过（方便你只放一部分 seed 时也能画）
                continue

            obj = load_yaml(fpath)
            for k in KFRACS:
                v = extract_metric_for_kfrac(obj, k, METRIC_KEY)
                values[layer][k].append(v)

    return values


def summarize(values):
    """
    返回 mean/std 的二维数组，shape=(len(LAYERS), len(KFRACS))
    """
    means = np.full((len(LAYERS), len(KFRACS)), np.nan, dtype=float)
    stds = np.full((len(LAYERS), len(KFRACS)), np.nan, dtype=float)
    counts = np.zeros((len(LAYERS), len(KFRACS)), dtype=int)

    for i, layer in enumerate(LAYERS):
        for j, k in enumerate(KFRACS):
            arr = np.array(values[layer][k], dtype=float)
            counts[i, j] = arr.size
            if arr.size > 0:
                means[i, j] = float(arr.mean())
                stds[i, j] = float(arr.std(ddof=1)) if arr.size >= 2 else 0.0

    return means, stds, counts

def save_csv(values, means, stds, counts):
    # 1) raw：每个 seed 的 test_acc
    with open(OUTPUT_CSV_RAW, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layer", "k_frac", "seed_idx", METRIC_KEY])
        for i, layer in enumerate(LAYERS):
            for j, k in enumerate(KFRACS):
                arr = values[layer][k]
                for seed_idx, v in enumerate(arr):
                    w.writerow([f"layer{layer}", k, seed_idx, v])

    # 2) summary：mean/std/n
    with open(OUTPUT_CSV_SUMMARY, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layer", "k_frac", "mean", "std", "n"])
        for i, layer in enumerate(LAYERS):
            for j, k in enumerate(KFRACS):
                w.writerow([f"layer{layer}", k, means[i, j], stds[i, j], int(counts[i, j])])

    print(f"Saved CSV raw to: {OUTPUT_CSV_RAW}")
    print(f"Saved CSV summary to: {OUTPUT_CSV_SUMMARY}")

def plot_bar(means, stds, counts):
    x = np.arange(len(LAYERS), dtype=float)

    plt.figure(figsize=FIGSIZE)

    for j, k in enumerate(KFRACS):
        offset = (j - (len(KFRACS) - 1) / 2) * BAR_WIDTH
        y = means[:, j]
        yerr = stds[:, j] if SHOW_ERRORBAR else None

        label = f"k_frac={k}"
        plt.bar(
            x + offset,
            y,
            width=BAR_WIDTH,
            yerr=yerr,
            capsize=ERRORBAR_CAPSIZE if SHOW_ERRORBAR else 0,
            label=label,
            alpha=0.9,
        )

        # 在柱子上标注均值和seed数量（可删）
        for i in range(len(LAYERS)):
            if not math.isnan(y[i]):
                plt.text(
                    x[i] + offset,
                    y[i] + 0.002,
                    f"{y[i]:.3f}\n(n={counts[i, j]})",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.xticks(x, [f"layer{l}" for l in LAYERS])
    plt.ylabel("test_acc")
    plt.xlabel("layer type")
    plt.title("test_acc at k_frac=0.05,0.20,0.50")
    plt.ylim(0, 1.0)  # 如果你想自动缩放可注释掉
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=DPI)
    plt.show()
    print(f"Saved to: {OUTPUT_PNG}")


def main():
    values = collect_values()
    means, stds, counts = summarize(values)
    save_csv(values, means, stds, counts)
    # 打印一下汇总，方便核对
    print("Layer-wise summary (mean ± std, n):")
    for i, layer in enumerate(LAYERS):
        parts = []
        for j, k in enumerate(KFRACS):
            parts.append(f"k={k}: {means[i, j]:.6f} ± {stds[i, j]:.6f} (n={counts[i, j]})")
        print(f"layer{layer}: " + " | ".join(parts))

    plot_bar(means, stds, counts)


if __name__ == "__main__":
    main()