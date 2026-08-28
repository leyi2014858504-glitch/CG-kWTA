# extract_features.py  ── supports 8 backbones
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import timm
import huggingface_hub

# Enable huggingface_hub resume/retry behavior
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Backbones directly supported by timm ─────────────────────────────────
TIMM_MODELS = {
    # tag                      timm model string                          feat_dim
    "resnet50_sup":            "resnet50.a1_in1k",                       # 2048
    "mocov2":                  "resnet50.mocov2_1x",                     # 2048 (MoCo v2)
    "convnext_base_sup":       "convnext_base.fb_in1k",                  # 1024
    "convnextv2_base_mae":     "convnextv2_base.fcmae_ft_in1k",          # 1024
    "vit_base_augreg":         "vit_base_patch16_224.augreg_in1k",       # 768
    "vit_base_mae":            "vit_base_patch16_224.mae",               # 768
    "vit_base_dino":           "vit_base_patch16_224.dino",              # 768
    "vit_base":                "vit_base_patch16_224",                   # 768 (standard ViT-B/16)
    "swin_base_sup":           "swin_base_patch4_window7_224.ms_in1k",   # 1024
}

def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

def build_dataset(name, root, transform):
    name = name.lower()
    if name == "cifar10":
        return (datasets.CIFAR10(root, train=True,  download=True, transform=transform),
                datasets.CIFAR10(root, train=False, download=True, transform=transform))
    elif name == "cifar100":
        return (datasets.CIFAR100(root, train=True,  download=True, transform=transform),
                datasets.CIFAR100(root, train=False, download=True, transform=transform))
    elif name == "stl10":
        return (datasets.STL10(root, split="train", download=True, transform=transform),
                datasets.STL10(root, split="test",  download=True, transform=transform))
    raise ValueError(f"Unknown dataset: {name}")

def build_mocov2(ckpt_path: str) -> nn.Module:
    """
    Load the official Facebook MoCo v2 checkpoint.
    Download URL:
    https://dl.fbaipublicfiles.com/moco/moco_checkpoints/
            moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"]

    # Remove the "module.encoder_q." prefix and skip projection-head fc
    new_sd = {}
    for k, v in sd.items():
        if not k.startswith("module.encoder_q."):
            continue
        new_key = k.replace("module.encoder_q.", "")
        if new_key.startswith("fc."):       # Drop the projection head
            continue
        new_sd[new_key] = v

    net = models.resnet50(weights=None)
    net.fc = nn.Identity()                  # Remove the classification head
    missing, unexpected = net.load_state_dict(new_sd, strict=False)
    print(f"[MoCo v2] missing={missing}, unexpected={unexpected}")
    return net

@torch.no_grad()
def extract(net, loader):
    feats, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        feats.append(net(x).float().cpu())
        ys.append(y.cpu() if isinstance(y, torch.Tensor) else torch.tensor(y))
    return torch.cat(feats), torch.cat(ys)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    type=str, required=True,
                    help=f"one of: {list(TIMM_MODELS)} or 'mocov2'")
    ap.add_argument("--tag",      type=str, required=True,
                    help="output file tag, e.g. cifar10_mocov2")
    ap.add_argument("--dataset",  type=str, default="cifar10")
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--out_dir",  type=str, default="./data")
    ap.add_argument("--batch_size", type=int, default=64,
                    help="batch size (default: 64 for 8GB GPU)")
    ap.add_argument("--moco_ckpt", type=str, default="./moco_v2_800ep_pretrain.pth.tar",
                    help="used only when --model mocov2")
    args = ap.parse_args()

    # ── Build network ───────────────────────────────────────────────────
    if args.model == "mocov2":
        net = build_mocov2(args.moco_ckpt)
    elif args.model in TIMM_MODELS:
        net = timm.create_model(TIMM_MODELS[args.model], pretrained=True, num_classes=0)
    else:
        # Also allow passing a raw timm model string
        net = timm.create_model(args.model, pretrained=True, num_classes=0)

    net.eval().to(device)

    # ── Dataset ────────────────────────────────────────────────────────
    tfm = get_transform()
    train_set, test_set = build_dataset(args.dataset, args.data_dir, tfm)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Extract and save ────────────────────────────────────────────────
    for split, loader in [("train", train_loader), ("test", test_loader)]:
        X, y = extract(net, loader)
        out_path = os.path.join(args.out_dir, f"{args.tag}_{split}.pt")
        torch.save({"X": X, "y": y}, out_path)
        print(f"[saved] {out_path}  X={tuple(X.shape)}  y={tuple(y.shape)}")

if __name__ == "__main__":
    main()

