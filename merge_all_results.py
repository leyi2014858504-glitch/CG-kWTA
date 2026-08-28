#!/usr/bin/env python3
"""
Merge all experiment results into one summary table.
Group by dataset: cifar10, cifar100, stl10.
Within each dataset, order by model and coordinate system.
"""
import os
import pandas as pd
import glob

BASE_DIR = r"d:\my projects\PythonProject4"
OUTPUT_FILE = os.path.join(BASE_DIR, "all_experiment_results_summary.xlsx")

# Model alias mapping (for sorting and display)
MODEL_ORDER = {
    "r50": 1,
    "r50mocov2": 2,
    "convnext": 3,
    "convnextv2": 4,
    "vitmae": 5,
    "vit": 6,
    "vitdino": 7,
    "swin": 8,
}

COORD_ORDER = {
    "randproj": 1,
    "meanstdrate": 2,
    "pca": 3,
}

def parse_folder_name(folder_name):
    """
    Parse folder name and extract dataset/model/coordinate-system info.
    Format: {dataset}_{model}_{coord}
    Example: cifar10_r50_pca
    """
    parts = folder_name.split('_')
    if len(parts) < 3:
        return None
    
    # Dataset is the prefix
    if parts[0] == "cifar10":
        dataset = "cifar10"
        model = parts[1]
        coord = parts[2]
    elif parts[0] == "cifar100":
        dataset = "cifar100"
        model = parts[1]
        coord = parts[2]
    elif parts[0] == "stl10":
        dataset = "stl10"
        model = parts[1]
        coord = parts[2]
    else:
        return None
    
    return {
        "dataset": dataset,
        "model": model,
        "coord": coord,
        "folder": folder_name
    }

def load_experiment_data(folder_path):
    """Load result data from a single experiment folder."""
    # Find CSV files
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        return None
    
    results = {}
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        try:
            df = pd.read_csv(csv_file)
            results[filename] = df
        except Exception as e:
            print(f"  [warn] failed to read {filename}: {e}")
    
    return results

def create_summary_table(all_data):
    """Build the summary table."""
    summary_rows = []
    
    for folder_name, data_dict in all_data.items():
        info = parse_folder_name(folder_name)
        if not info:
            continue
        
        dataset = info["dataset"]
        model = info["model"]
        coord = info["coord"]
        
        # Extract key metrics
        kfrac_testacc = data_dict.get("kfrac_testacc_ci95.csv")
        kfrac_time = data_dict.get("kfrac_time_ci95_lowk.csv")
        
        # Best accuracy
        best_acc = None
        best_k = None
        if kfrac_testacc is not None:
            if "test_acc_mean" in kfrac_testacc.columns:
                best_acc_idx = kfrac_testacc["test_acc_mean"].idxmax()
                best_acc = kfrac_testacc.loc[best_acc_idx, "test_acc_mean"]
                best_k = kfrac_testacc.loc[best_acc_idx, "k_frac"]
        
        # Runtime
        avg_time = None
        if kfrac_time is not None:
            if "time_mean" in kfrac_time.columns:
                avg_time = kfrac_time["time_mean"].mean()
        
        # Shuffle status (cannot be inferred from folder name directly)
        shuffle_status = "unknown"
        
        summary_rows.append({
            "dataset": dataset,
            "model": model,
            "coord": coord,
            "best_acc": best_acc,
            "best_k_frac": best_k,
            "avg_time_s": avg_time,
            "folder": folder_name,
            "model_order": MODEL_ORDER.get(model, 99),
            "coord_order": COORD_ORDER.get(coord, 99),
        })
    
    return pd.DataFrame(summary_rows)

def main():
    print("="*80)
    print("Merge all experiment results")
    print("="*80)
    
    # Find all experiment folders
    patterns = ["cifar10_*_*", "cifar100_*_*", "stl10_*_*"]
    all_folders = []
    
    for pattern in patterns:
        matches = glob.glob(os.path.join(BASE_DIR, pattern))
        all_folders.extend(matches)
    
    print(f"\nFound {len(all_folders)} experiment folders")
    
    # Load all data
    all_data = {}
    for folder in all_folders:
        folder_name = os.path.basename(folder)
        print(f"\nProcessing: {folder_name}")
        
        data = load_experiment_data(folder)
        if data:
            all_data[folder_name] = data
            print(f"  [ok] loaded ({len(data)} files)")
        else:
            print(f"  [warn] no data")
    
    print(f"\nSuccessfully loaded data from {len(all_data)} experiments")
    
    # Build summary table
    print("\nBuilding summary table...")
    summary_df = create_summary_table(all_data)
    
    # Sort by dataset, then model, then coordinate system
    summary_df = summary_df.sort_values(
        by=["dataset", "model_order", "coord_order"],
        ascending=[True, True, True]
    )
    
    # Drop helper columns
    summary_df = summary_df.drop(columns=["model_order", "coord_order"])
    
    # Save to Excel
    print(f"\nSaving to: {OUTPUT_FILE}")
    
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        # Overview sheet
        summary_df.to_excel(writer, sheet_name="Overview", index=False)
        
        # Per-dataset sheets
        for dataset in ["cifar10", "cifar100", "stl10"]:
            dataset_df = summary_df[summary_df["dataset"] == dataset].copy()
            dataset_df = dataset_df.drop(columns=["dataset"])
            dataset_df.to_excel(writer, sheet_name=f"{dataset}", index=False)
        
        # Detailed data for each experiment
        for folder_name, data_dict in all_data.items():
            for csv_name, df in data_dict.items():
                sheet_name = f"{folder_name}_{csv_name.replace('.csv', '')}"
                # Excel sheet name length limit
                sheet_name = sheet_name[:31]
                try:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                except Exception as e:
                    print(f"  [warn] cannot write sheet {sheet_name}: {e}")
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80)
    print(f"\nSummary file: {OUTPUT_FILE}")
    print(f"\nSheets included:")
    print(f"  - Overview: summary of all experiments")
    print(f"  - cifar10, cifar100, stl10: grouped by dataset")
    print(f"  - detailed data: raw CSV data of each experiment")
    
    # Show summary statistics
    print(f"\nStatistics:")
    print(f"  Total experiments: {len(summary_df)}")
    for dataset in ["cifar10", "cifar100", "stl10"]:
        count = len(summary_df[summary_df["dataset"] == dataset])
        print(f"  {dataset}: {count} experiments")

if __name__ == "__main__":
    main()
