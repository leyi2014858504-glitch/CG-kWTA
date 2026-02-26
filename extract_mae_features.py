# extract_vitmae_features.py

import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import timm
from timm.data import resolve_model_data_config, create_transform
from huggingface_hub import HfApi
from contextlib import nullcontext

print("HF endpoint =", HfApi().endpoint)


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_dataset(name: str, root: str, transform):
    name = name.lower()
    if name == "cifar10":
        train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    elif name == "cifar100":
        train_set = datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR100(root=root, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {name} (use cifar10 or cifar100)")
    return train_set, test_set


@torch.inference_mode()
def extract_split_layerwise_cls(
    model,
    loader,
    device,
    amp: bool,
    out_dtype: str = "float32",
    num_layers: int = 12,
    pool = "cls"
):
    """
    Return:
      x_layers: list length=num_layers, each Tensor [N, C] (CLS at that layer)
      y: Tensor [N]
    """
    x_layers_chunks = [[] for _ in range(num_layers)]
    y_chunks = []

    # cache per minibatch
    layer_out = {}

    def make_hook(layer_idx: int):
        def hook(_module, _inp, out):
            layer_out[layer_idx] = out
        return hook

    # register hooks once
    hooks = []
    for i in range(num_layers):
        hooks.append(model.blocks[i].register_forward_hook(make_hook(i)))

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if (amp and device.type == "cuda")
        else nullcontext()
    )

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        layer_out.clear()

        with autocast_ctx:
            _ = model.forward_features(xb)

        for l in range(num_layers):
            z = layer_out[l]
            if isinstance(z, dict):
                z = z.get("x", next(iter(z.values())))

            # expected: [B, tokens, C]
            if z.dim() == 3:
                if pool == "cls":
                    x = z[:, 0, :]
                else:  # mean，跳过CLS token(index 0)，只对patch tokens做均值
                    x = z[:, 1:, :].mean(dim=1)
            elif z.dim() == 2:
                # fallback if some model returns pooled [B, C]
                x = z
            else:
                raise RuntimeError(f"Unexpected feature shape at layer {l}: {tuple(z.shape)}")

            x = x.detach().cpu()
            x = x.half() if out_dtype == "float16" else x.float()
            x_layers_chunks[l].append(x)

        y_chunks.append(yb.detach().cpu().long())

    # remove hooks
    for h in hooks:
        h.remove()

    x_layers = [torch.cat(chunks, dim=0) for chunks in x_layers_chunks]
    y = torch.cat(y_chunks, dim=0)
    return x_layers, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--out_dir", type=str, default="./data")
    parser.add_argument("--model", type=str, default="vit_base_patch16_224.mae")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dtype", type=str, default="float32", choices=["float32", "float16"])
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--pool", type=str, default="cls", choices=["cls", "mean"])
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # feature extractor
    model = timm.create_model(args.model, pretrained=True, num_classes=0)
    model.eval().to(device)

    data_cfg = resolve_model_data_config(model)
    transform = create_transform(**data_cfg, is_training=False)

    train_set, test_set = build_dataset(args.dataset, args.data_dir, transform)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    xtr_layers, ytr = extract_split_layerwise_cls(
        model=model,
        loader=train_loader,
        device=device,
        amp=args.amp,
        out_dtype=args.out_dtype,
        num_layers=args.num_layers,
    )
    xte_layers, yte = extract_split_layerwise_cls(
        model=model,
        loader=test_loader,
        device=device,
        amp=args.amp,
        out_dtype=args.out_dtype,
        num_layers=args.num_layers,
    )

    # IMPORTANT: match main-script loader convention: {tag}train.pt / {tag}test.pt
    tag_prefix = f"{args.dataset}_{args.model.replace('.', '_')}_{args.pool}"
    for l in range(args.num_layers):
        tag = f"{tag_prefix}_layer{l:02d}"
        train_path = os.path.join(args.out_dir, f"{tag}_train.pt")
        test_path = os.path.join(args.out_dir, f"{tag}_test.pt")
        torch.save({"X": xtr_layers[l], "y": ytr}, train_path)
        torch.save({"X": xte_layers[l], "y": yte}, test_path)
        print("Saved:", train_path, xtr_layers[l].shape, ytr.shape, xtr_layers[l].dtype)
        print("Saved:", test_path, xte_layers[l].shape, yte.shape, xte_layers[l].dtype)


if __name__ == "__main__":
    main()
