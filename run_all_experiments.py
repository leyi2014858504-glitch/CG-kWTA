#!/usr/bin/env python3
import os
import subprocess
import sys
import shutil
import re

BASE_DIR = r"d:\my projects\PythonProject4"
DATA_DIR = os.path.join(BASE_DIR, "data")
MAIN_SCRIPT = os.path.join(BASE_DIR, "sphere_kwta_dimension_resnet50_MAE.py")
PLOT_TEMPLATE = os.path.join(BASE_DIR, "plot_4modes_ci95.py")
EXTRACT_SCRIPT = os.path.join(BASE_DIR, "extract_features.py")

# Experiment configuration
MODELS = {
    "r50": ("resnet50_sup", "r50"),                              # ResNet-50 supervised
    "r50mocov2": ("mocov2", "mocov2"),                            # MoCo v2
    "convnext": ("convnext_base_sup", "convnext_base_sup"),
    "convnextv2": ("convnextv2_base_mae", "convnextv2_base_mae"),
    "vitmae": ("vit_base_mae", "vit_mae"),
    "vit": ("vit_base", "vit"),                                   # Standard ViT-B/16
    "vitdino": ("vit_base_dino", "dino"),                         # ViT-DINO
    "swin": ("swin_base_sup", "swin"),                            # Swin Transformer
}

DATASETS = ["cifar10"]

COORD_SYSTEMS = {
    "randproj": "geo_r",
    "meanstdrate": "geo_m",
    "pca": "geo_pca",
   # "som": "geo_som",
}


def check_features_exist(dataset, model_tag):
    """Check whether cached feature files exist."""
    train_path = os.path.join(DATA_DIR, f"{dataset}_{model_tag}_train.pt")
    test_path = os.path.join(DATA_DIR, f"{dataset}_{model_tag}_test.pt")
    return os.path.exists(train_path) and os.path.exists(test_path)


def extract_features(dataset, model_name, model_tag):
    """Extract and save feature files."""
    print(f"\n{'='*70}")
    print(f"开始提取特征：{dataset} - {model_name}")
    print(f"{'='*70}")
    
    cmd = [
        sys.executable,
        EXTRACT_SCRIPT,
        "--model", model_name,
        "--tag", f"{dataset}_{model_tag}",
        "--dataset", dataset,
        "--data_dir", DATA_DIR,
        "--out_dir", DATA_DIR,
        "--batch_size", "128"
    ]
    
    print(f"运行命令：{' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True, encoding='utf-8', errors='replace')
    
    if result.returncode != 0:
        print(f"错误：特征提取失败!")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    
    print(f"特征提取成功：{dataset}_{model_tag}")
    print(result.stdout)
    return True


def run_single_experiment(dataset, model_short, coord_system):
    model_name, model_tag = MODELS[model_short]
    
    if dataset == "cifar10":
        dataset_tag = f"cifar10_{model_tag}"
    else:
        dataset_tag = f"{dataset}_{model_tag}"
    
    exp_mode = COORD_SYSTEMS[coord_system]
    
    out_folder = os.path.join(BASE_DIR, f"{dataset}_{model_short}_{coord_system}")
    os.makedirs(out_folder, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"实验组合: {dataset} | {model_short} | {coord_system}")
    print(f"输出目录: {out_folder}")
    print(f"{'='*70}\n")
    
    for shuffle in [False, True]:
        shuffle_str = "shuffled" if shuffle else "non-shuffled"
        print(f"\n--- 运行 {shuffle_str} ---")
        
        temp_script = os.path.join(out_folder, f"temp_script_{shuffle}.py")
        
        with open(MAIN_SCRIPT, "r", encoding="utf-8") as f:
            content = f.read()
        
        content = re.sub(r'EXP_MODE\s*=\s*"[^"]*"', f'EXP_MODE = "{exp_mode}"', content)
        content = re.sub(r'DATASET\s*=\s*"[^"]*"', f'DATASET = "{dataset_tag}"', content)
        content = re.sub(r'Shuffle_mode\s*=\s*(True|False)', f'Shuffle_mode = {shuffle}', content)
        data_dir_escaped = DATA_DIR.replace("\\", "\\\\")
        content = re.sub(r'DATA_DIR\s*=\s*"[^"]*"', f'DATA_DIR = r"{data_dir_escaped}"', content)
        
        with open(temp_script, "w", encoding="utf-8") as f:
            f.write(content)
        
        try:
            result = subprocess.run(
                [sys.executable, temp_script],
                cwd=out_folder,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            if result.returncode != 0:
                print(f"错误: {shuffle_str} 运行失败!")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
            else:
                print(f"{shuffle_str} 运行成功!")
                
        finally:
            os.remove(temp_script)
    
    plot_script_dst = os.path.join(out_folder, "plot_ci95.py")
    shutil.copy2(PLOT_TEMPLATE, plot_script_dst)
    
    with open(plot_script_dst, "r", encoding="utf-8") as f:
        plot_content = f.read()
    
    new_patterns = f'''MODE_PATTERNS = {{
    "non-shuffled": "results_{dataset_tag}_{exp_mode}_sd*.yaml",
    "shuffled": "results_{dataset_tag}_{exp_mode}_shuf_sd*.yaml",
}}'''
    
    plot_content = re.sub(
        r'MODE_PATTERNS\s*=\s*\{[^}]+\}',
        new_patterns,
        plot_content,
        flags=re.DOTALL
    )
    
    with open(plot_script_dst, "w", encoding="utf-8") as f:
        f.write(plot_content)
    
    print(f"\n已配置画图脚本: {plot_script_dst}")
    print(f"运行画图: cd {out_folder} ; python plot_ci95.py")


def main():
    print("="*80)
    print("实验矩阵自动化运行（含自动特征提取）")
    print("="*80)
    
    print(f"\n将运行:")
    print(f"  数据集：{DATASETS}")
    print(f"  模型：{list(MODELS.keys())}")
    print(f"  坐标系：{list(COORD_SYSTEMS.keys())}")
    print(f"  每个组合运行：shuffle=False 和 shuffle=True")
    print(f"\n总计：{len(DATASETS) * len(MODELS) * len(COORD_SYSTEMS)} 个实验组合")
    
    # Step 1: check and extract missing features
    print("\n" + "="*80)
    print("第一步：检查特征文件")
    print("="*80)
    
    missing_features = []
    for dataset in DATASETS:
        for model_short, (model_name, model_tag) in MODELS.items():
            if not check_features_exist(dataset, model_tag):
                missing_features.append((dataset, model_name, model_tag))
    
    if missing_features:
        print(f"\n发现 {len(missing_features)} 个缺失的特征文件:")
        for dataset, model_name, model_tag in missing_features:
            print(f"  - {dataset}_{model_tag}")
        
        print(f"\n开始自动提取特征...")
        print(f"预计磁盘空间：每个特征文件约 200-500MB，总计约 {len(missing_features) * 2 * 350 / 1024:.1f}GB")
        
        for dataset, model_name, model_tag in missing_features:
            if not extract_features(dataset, model_name, model_tag):
                print(f"警告：{dataset} - {model_name} 特征提取失败，跳过相关实验")
    else:
        print("\n所有特征文件已存在，无需提取。")
    
    # Step 2: run experiments
    print("\n" + "="*80)
    print("第二步：运行实验")
    print("="*80)
    
    for dataset in DATASETS:
        for model_short in MODELS.keys():
            for coord_system in COORD_SYSTEMS.keys():
                try:
                    run_single_experiment(dataset, model_short, coord_system)
                except Exception as e:
                    print(f"错误：{dataset} - {model_short} - {coord_system} 失败：{e}")
    
    print("\n" + "="*80)
    print("所有实验完成!")
    print("="*80)


if __name__ == "__main__":
    main()
