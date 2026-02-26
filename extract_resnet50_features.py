import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

device = torch.device("cuda" )


@torch.no_grad()
def extract_split(split: str, data_dir: str, batch_size: int = 256, num_workers: int = 1):
    assert split in ["train", "test"]
    train_flag = (split == "train")

    # CIFAR-10 -> ResNet-50 expects ImageNet-style normalization & 224
    tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    ds = datasets.CIFAR10(root=data_dir, train=train_flag, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())

    # Pretrained ResNet-50, remove fc -> 2048-d features
    try:
        weights = models.ResNet50_Weights.DEFAULT
        net = models.resnet50(weights=weights)
    except Exception:
        net = models.resnet50(pretrained=True)
    net.fc = nn.Identity()
    net.eval().to(device)

    feats = []
    ys = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        f = net(x)                 # [B, 2048]
        feats.append(f.float().cpu())
        ys.append(y.cpu())

    X = torch.cat(feats, dim=0).contiguous()   # [N, 2048]
    y = torch.cat(ys, dim=0).contiguous()      # [N]
    return X, y


def main():
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)

    for split in ["train", "test"]:
        out_path = os.path.join(data_dir, f"cifar10_r50_{split}.pt")
        if os.path.exists(out_path):
            print(f"[skip] exists: {out_path}")
            continue
        X, y = extract_split(split, data_dir=data_dir)
        torch.save({"X": X, "y": y}, out_path)
        print(f"[save] {out_path}  X={tuple(X.shape)}  y={tuple(y.shape)}")


if __name__ == "__main__":
    main()