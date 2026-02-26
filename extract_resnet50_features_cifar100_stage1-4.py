import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.nn.functional as F

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

    acts = {}  # name -> feature map [B,C,H,W]

    feats_l1, feats_l2, feats_l3, feats_l4 = [], [], [], []
    feats_l4b0, feats_l4b1, feats_l4b2 = [], [], []
    def _make_list_hook(store_list):
        def hook(module, inp, out):
            # out: [B, C, H, W]
            pooled = F.adaptive_avg_pool2d(out, (1, 1)).flatten(1)  # [B, C]
            store_list.append(pooled.detach().float().cpu())

        return hook

    h1 = net.layer1.register_forward_hook(_make_list_hook(feats_l1))
    h2 = net.layer2.register_forward_hook(_make_list_hook(feats_l2))
    h3 = net.layer3.register_forward_hook(_make_list_hook(feats_l3))
    h4 = net.layer4.register_forward_hook(_make_list_hook(feats_l4))

    hb0 = net.layer4[0].register_forward_hook(_make_list_hook(feats_l4b0))
    hb1 = net.layer4[1].register_forward_hook(_make_list_hook(feats_l4b1))
    hb2 = net.layer4[2].register_forward_hook(_make_list_hook(feats_l4b2))

    feats = []
    ys = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        f = net(x)  # [B, 2048] (fc=Identity)
        feats.append(f.detach().float().cpu())
        ys.append(y.cpu())

    X = torch.cat(feats, dim=0).contiguous()  # [N,2048]
    X1 = torch.cat(feats_l1, dim=0).contiguous()  # [N,256]
    X2 = torch.cat(feats_l2, dim=0).contiguous()  # [N,512]
    X3 = torch.cat(feats_l3, dim=0).contiguous()  # [N,1024]
    X4 = torch.cat(feats_l4, dim=0).contiguous()  # [N,2048]
    X4b0 = torch.cat(feats_l4b0, dim=0).contiguous()
    X4b1 = torch.cat(feats_l4b1, dim=0).contiguous()
    X4b2 = torch.cat(feats_l4b2, dim=0).contiguous()

    assert X4b0.shape[1] == 2048 and X4b1.shape[1] == 2048 and X4b2.shape[1] == 2048

    y = torch.cat(ys, dim=0).contiguous()  # [N]

    for h in [h1, h2, h3, h4, hb0, hb1, hb2]:
        h.remove()
    return {"final": X, "layer1": X1, "layer2": X2, "layer3": X3, "layer4": X4, "layer4b0": X4b0,
            "layer4b1": X4b1, "layer4b2": X4b2}, y


def main():
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)

    for split in ["train", "test"]:
        out_path = os.path.join(data_dir, f"cifar10_r50_{split}.pt")
        if os.path.exists(out_path):
            print(f"[skip] exists: {out_path}")
            continue
        Xdict, y = extract_split(split, data_dir=data_dir)

        # keep old final feature name
        out_path = os.path.join(data_dir, f"cifar10_r50_{split}.pt")
        torch.save({"X": Xdict["final"], "y": y}, out_path)

        for lname in ["layer1", "layer2", "layer3", "layer4", "layer4b0", "layer4b1", "layer4b2"]:
            if lname not in Xdict: continue  # 容错，防止以后删了key报错
            out_path_l = os.path.join(data_dir, f"cifar10_r50_{lname}_{split}.pt")
            torch.save({"X": Xdict[lname], "y": y}, out_path_l)
            print(f"[save] {out_path_l} X={tuple(Xdict[lname].shape)} y={tuple(y.shape)}")



if __name__ == "__main__":
    main()
