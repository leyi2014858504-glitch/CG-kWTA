import pandas as pd
from pathlib import Path

DATASETS  = ["cifar100", "cifar10", "stl10"]   # Keep longer prefix first
BACKBONES = ["r50mocov2","convnextv2","convnext","vitmae","vitdino","vit","swin","r50"]
COORDS    = ["meanstdrate", "randproj", "pca"]

ROOT = "."   # Script is in project root, so "." is enough
records = []

for folder in sorted(Path(ROOT).iterdir()):
    if not folder.is_dir():
        continue
    name     = folder.name
    csv_path = folder / "kfrac_testacc_ci95.csv"
    if not csv_path.exists():
        continue

    dataset  = next((d for d in DATASETS if name.startswith(d + "_")), None)
    if dataset is None:
        continue
    rest     = name[len(dataset)+1:]
    coord    = next((c for c in COORDS if rest.endswith("_" + c)), None)
    if coord is None:
        continue
    backbone = rest[:-(len(coord)+1)]

    df = pd.read_csv(csv_path)
    df["dataset"]  = dataset
    df["backbone"] = backbone
    df["coord"]    = coord
    records.append(df)
    print(f"ok {dataset:8s}  {backbone:12s}  {coord:12s}  rows={len(df)}")

master = pd.concat(records, ignore_index=True)
master.to_csv("master.csv", index=False)
print(f"\nTotal rows: {len(master)}")
print(f"Unique combos: {master[['dataset','backbone','coord']].drop_duplicates().shape[0]}")
