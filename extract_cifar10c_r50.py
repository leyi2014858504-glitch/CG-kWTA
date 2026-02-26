import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.inference_mode()
def extract_cifar10c_resnet50_features(
    cifar10c_dir="./data/CIFAR-10-C",
    out_dir="./data/cifar10c_r50_sev5",
    severity=5,
    batch_size=32,   # 8GB 建议从 32 开始
):
    cifar10c_dir = Path(cifar10c_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # labels.npy: 10000 或 50000（二者都兼容）
    labels_all = np.load(cifar10c_dir / "labels.npy")
    s, e = (severity - 1) * 10000, severity * 10000
    y_np = labels_all if labels_all.shape[0] == 10000 else labels_all[s:e]
    y = torch.from_numpy(y_np.astype(np.int64))

    # ResNet50 -> 2048d
    try:
        weights = models.ResNet50_Weights.DEFAULT
        net = models.resnet50(weights=weights)
    except Exception:
        net = models.resnet50(pretrained=True)
    net.fc = nn.Identity()
    net.eval().to(device)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    corr_files = sorted([p for p in cifar10c_dir.glob("*.npy") if p.name != "labels.npy"])
    for p in corr_files:
        corr = p.stem
        out_path = out_dir / f"{corr}.pt"
        if out_path.exists():
            print("[skip]", out_path)
            continue

        # 关键：mmap + 每次只取一个 batch，避免整块上 GPU
        x_mmap = np.load(p, mmap_mode="r")  # shape [50000,32,32,3] uint8

        feats = []
        for i in range(s, e, batch_size):
            xb = torch.from_numpy(x_mmap[i:i+batch_size].copy())  # copy 防止 mmap stride 问题
            xb = xb.permute(0, 3, 1, 2).float() / 255.0          # NCHW, float
            xb = xb.to(device, non_blocking=True)

            xb = F.interpolate(xb, size=(224, 224), mode="bilinear", align_corners=False)
            xb = (xb - mean) / std

            # 关键：AMP 半精度省显存（仅 CUDA 时启用）
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                fb = net(xb)  # [B, 2048]

            feats.append(fb.float().cpu())  # 立刻下 GPU

        X = torch.cat(feats, 0).contiguous()  # [10000,2048]
        torch.save({"X": X, "y": y}, out_path)
        print("[save]", out_path, tuple(X.shape), tuple(y.shape))
if __name__ == "__main__":
    extract_cifar10c_resnet50_features(
        cifar10c_dir="./data/CIFAR-10-C",
        out_dir="./data/cifar10c_r50_sev5",
        severity=5,
        batch_size=32,
    )