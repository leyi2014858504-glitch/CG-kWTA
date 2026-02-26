import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import yaml
import os
from scipy import stats


def parse_file(filename, group_label):
    with open(filename, 'r') as f:
        data = yaml.safe_load(f)

    records = []
    # 尝试自动获取seed，如果没有meta信息则从文件名解析
    seed = data.get('meta', {}).get('seed', None)
    if seed is None:
        try:
            # 假设文件名格式类似 results_...0.yaml
            seed = int(os.path.splitext(filename)[0][-1])
        except:
            seed = 0

    for run in data['runs']:
        records.append({
            'k_frac': run['k_frac'],
            'test_acc': run['test_acc'],
            'Group': group_label,
            'Seed': seed
        })
    return records


# 1. 读取所有文件
all_data = []

# 定义文件匹配模式和对应的图例名称
file_patterns = {
    'results_kfrac_geo_r_r50_shuffle*.yaml': 'Original (Shuffle)',
    'results_kfrac_geo_r_r50c_shuffle*.yaml': 'Center (Shuffle)',
    'results_kfrac_geo_r_r50[0-9].yaml': 'Original (No Shuffle)',  # 注意区分文件名模式
    'results_kfrac_geo_r_r50c[0-9].yaml': 'Center (No Shuffle)'
}

for pattern, label in file_patterns.items():
    files = glob.glob(pattern)
    print(f"Found {len(files)} files for {label}")
    for f in files:
        all_data.extend(parse_file(f, label))

df = pd.DataFrame(all_data)

# 2. 绘图
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# ---- 显著性检验：4.3的核心假设 ----
# H0: Δ_shell = Δ_ball（geometry is interchangeable）
# H1: Δ_shell > Δ_ball（Shell的alignment gap更大，单侧）

kfracs = sorted(df['k_frac'].unique())
rows = []

for kf in kfracs:
    sub = df[df['k_frac'] == kf]

    # 按seed对齐，取每个seed的test_acc
    shell_ns   = sub[sub['Group'] == 'Original (No Shuffle)'].set_index('Seed')['test_acc']
    shell_shuf = sub[sub['Group'] == 'Original (Shuffle)'].set_index('Seed')['test_acc']
    ball_ns    = sub[sub['Group'] == 'Center (No Shuffle)'].set_index('Seed')['test_acc']
    ball_shuf  = sub[sub['Group'] == 'Center (Shuffle)'].set_index('Seed')['test_acc']

    # 对齐seed
    seeds = shell_ns.index.intersection(shell_shuf.index)\
                          .intersection(ball_ns.index)\
                          .intersection(ball_shuf.index)

    delta_shell = (shell_ns - shell_shuf)[seeds].values   # Δ_shell per seed
    delta_ball  = (ball_ns  - ball_shuf )[seeds].values   # Δ_ball  per seed
    diff        = delta_shell - delta_ball                 # 差值，H1: >0

    t, p_t = stats.ttest_1samp(diff, popmean=0, alternative='greater')

    # 追加 Wilcoxon（单侧，alternative='greater' 即 H1: diff > 0）
    w, p_w = stats.wilcoxon(diff, alternative='greater')

    rows.append({
        'k_frac': kf,
        'Δ_shell (mean)': round(delta_shell.mean(), 4),
        'Δ_ball (mean)': round(delta_ball.mean(), 4),
        'Δ_diff (mean)': round(diff.mean(), 4),
        't': round(t, 3),
        'p_t': round(p_t, 4),
        'p_wilcoxon': round(p_w, 4),
        'sig': '***' if min(p_t, p_w) < 0.001 else (
            '**' if min(p_t, p_w) < 0.01 else ('*' if min(p_t, p_w) < 0.05 else 'ns'))
    })

result_df = pd.DataFrame(rows)
print(result_df.to_string(index=False))
result_df.to_csv('sec43_significance.csv', index=False)

# 使用 lineplot 自动绘制均值线和置信区间(阴影)
sns.lineplot(data=df, x='k_frac', y='test_acc', hue='Group', marker='o')

plt.title('Test Accuracy Comparison: Shuffle vs No-Shuffle (Original vs Center)', fontsize=14)
plt.xlabel('Fraction of Data (k_frac)', fontsize=12)
plt.ylabel('Test Accuracy', fontsize=12)
plt.legend(title='Configuration')
plt.tight_layout()

# 保存或显示
plt.savefig('comparison_plot.png', dpi=300)
plt.show()
