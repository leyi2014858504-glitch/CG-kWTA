#!/usr/bin/env python3
"""Extract frozen ResNet-50 features for ImageNet-100 (scalability check, R1.2).

Recipe mirrored from the other r50 datasets (extract_resnet50_features_*.py):
    torchvision resnet50(weights=ResNet50_Weights.DEFAULT)  -> IMAGENET1K_V2
    net.fc = nn.Identity()                                  -> 2048-d avgpool
    eval transform: Resize(256) + CenterCrop(224) + ImageNet normalization

Why torchvision instead of timm's "resnet50.a1_in1k": the IMAGENET1K_V2 weights
are already in the local torch hub cache, so this runs fully offline, and it
matches the backbone actually used for cifar10/cifar100/stl10.

Outputs (keys {"X","y"}):
    {out_dir}/imagenet100_r50_train.pt   [N_train, 2048]
    {out_dir}/imagenet100_r50_test.pt    [N_test,  2048]

Usage:
    python extract_imagenet100_r50.py
    python extract_imagenet100_r50.py --limit 32 --out_dir d:\\cg-kwta\\_smoke   # smoke test
"""
import argparse
import gc
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

DEFAULT_ROOT = r"d:\my projects\PythonProject4\data"
TAG = "imagenet100_r50"


def get_transform(downscale=None):
    """Eval transform. `downscale` first shrinks the image to N x N, so the
    subsequent Resize(256)+CenterCrop(224) upsamples it back — this reproduces
    the effective image detail of the low-res benchmarks (CIFAR 32->224,
    STL-10 96->224) on ImageNet-100 (T1 control)."""
    ops = []
    if downscale:
        ops.append(transforms.Resize((int(downscale), int(downscale))))
    ops += [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(ops)


def build_net():
    net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    net.fc = nn.Identity()
    return net


@torch.no_grad()
def extract(net, loader, device, label):
    """Preallocate the output and fill it in place.

    Avoids the list-of-batches + torch.cat pattern, whose peak is ~2x the final
    tensor — that blew the host RAM budget on the 130k x 2048 train split.
    """
    n_total = len(loader.dataset)
    X = None
    ys = []
    pos = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        f = net(x).float().cpu()
        if X is None:
            X = torch.empty(n_total, f.shape[1], dtype=torch.float32)
        X[pos:pos + f.shape[0]] = f
        pos += f.shape[0]
        del f
        ys.append(y.cpu())
        if pos % 5120 < x.size(0):
            print(f"  [{label}] {pos}/{n_total} images", flush=True)
    return X[:pos].contiguous(), torch.cat(ys, dim=0).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=DEFAULT_ROOT)
    ap.add_argument("--out_dir", default=DEFAULT_ROOT)
    ap.add_argument("--tag", default=TAG,
                    help="output tag / dataset key, e.g. imagenet100_r50_ds96")
    ap.add_argument("--downscale", type=int, default=0,
                    help="T1 control: shrink each image to N x N before the "
                         "standard 224 pipeline (0 = native resolution)")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--splits", nargs="+", default=["train", "test"],
                    choices=["train", "test"],
                    help="which splits to extract (use 'test' to top up a run "
                         "whose train split already succeeded)")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: cap each split at N images (0 = use all)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  torch={torch.__version__}  tag={args.tag}  "
          f"downscale={args.downscale or 'native'}")

    tfm = get_transform(args.downscale or None)
    in100 = os.path.join(args.data_root, "imagenet100")
    paths = {"train": os.path.join(in100, "train"),
             "test": os.path.join(in100, "val")}
    sets = {k: datasets.ImageFolder(paths[k], transform=tfm) for k in args.splits}
    for k, ds in sets.items():
        print(f"{k}: {len(ds)} images, {len(ds.classes)} classes")

    if args.limit:
        rng = random.Random(0)
        for k, ds in sets.items():
            idx = rng.sample(range(len(ds)), min(args.limit, len(ds)))
            sets[k] = Subset(ds, idx)

    net = build_net().eval().to(device)
    os.makedirs(args.out_dir, exist_ok=True)

    for split, ds in sets.items():
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=(device.type == "cuda"))
        X, y = extract(net, loader, device, split)
        out = os.path.join(args.out_dir, f"{args.tag}_{split}.pt")
        torch.save({"X": X, "y": y}, out)
        print(f"[saved] {out}  X={tuple(X.shape)}  y={tuple(y.shape)}  "
              f"classes={int(y.max().item()) + 1}", flush=True)
        # free before the next split (host RAM is the binding constraint)
        del X, y, loader, ds
        gc.collect()


if __name__ == "__main__":
    main()
