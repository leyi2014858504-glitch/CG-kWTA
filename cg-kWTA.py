import os
print("Script started...")
import random
import time
from concurrent.futures import ThreadPoolExecutor

import cma
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from networkx.algorithms.distance_measures import radius, center
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision import models
from shuffle import shuffle_pts_rows_inplace
from shuffle import  fitness_shuffle
from minisom import MiniSom


print("Imports done.")

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NOTE: Keep this global random tensor unchanged to avoid altering any implicit behavior.
pts = torch.randn(1024, 3)

# CMA-ES step size
sigma0 = 0.50

#pts dimension
pd = 3

# ---- scoring mode ----
# "shell": original distance-to-sphere-shell (uses Radius)
# "center": Euclidean distance to Center (ignores Radius)
GEOSCORE_MODE = "center"   # "shell" or "center"
# "randn"    -> original random pts (kept)
GEO_PTS_INIT = "external"
PTS_EXTERNAL_PATH = "./data/pts_stl10_r50_randproj.pt"

#The shuffle test
Shuffle_mode = False
FITNESS_SHUFFLE_MODE = False
FITNESS_SHUFFLE_SEED = 0   # True to enable



def set_seed(seed: int):
    """Set seeds for reproducibility (kept as-is)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _zscore(v: torch.Tensor) -> torch.Tensor:
    return (v - v.mean()) / (v.std() + 1e-6)


def compute_pts_meanvar_once(
    backbone: nn.Module,
    calib_data: torch.Tensor,
    pts_dim: int = pd,
    batch_size: int = 512,
) -> torch.Tensor:
    """
    Compute fixed per-neuron coordinates using simple activation statistics:
    mean, std, and activation rate (for pts_dim=3).
    Returns: pts tensor [hidden_dim, pts_dim] on `device`.
    """
    backbone.eval()
    ds = TensorDataset(calib_data)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    s1 = None
    s2 = None
    sp = None
    n_total = 0

    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            feats = backbone(x)  # [B, H]

            if s1 is None:
                H = feats.shape[1]
                s1 = torch.zeros(H, device=device, dtype=feats.dtype)
                s2 = torch.zeros(H, device=device, dtype=feats.dtype)
                sp = torch.zeros(H, device=device, dtype=feats.dtype)

            s1 += feats.sum(dim=0)
            s2 += (feats * feats).sum(dim=0)
            sp += (feats > 0).to(feats.dtype).sum(dim=0)
            n_total += feats.shape[0]

    mean = s1 / max(1, n_total)
    var = s2 / max(1, n_total) - mean * mean
    std = torch.sqrt(torch.clamp(var, min=1e-12))
    act_rate = sp / max(1, n_total)

    mean_z = _zscore(mean)
    std_z = _zscore(std)
    rate_z = _zscore(act_rate)

    if pts_dim == 1:
        out = mean_z[:, None]
    elif pts_dim == 2:
        out = torch.stack([mean_z, std_z], dim=1)
    else:
        out = torch.stack([mean_z, std_z, rate_z], dim=1)

    return out.detach()


def compute_pts_randproj_once(
    backbone: nn.Module,
    calib_data: torch.Tensor,
    pts_dim: int = 3,
    batch_size: int = 512,
    proj_seed: int = 0,
) -> torch.Tensor:
    """
    Data-dependent random projection baseline:
    Let A = backbone(calib_data) in R^{N x H}. Sample a fixed random matrix R in R^{N x d}
    (generated on the fly with a deterministic RNG), then pts = A^T R in R^{H x d}.
    """
    backbone.eval()
    ds = TensorDataset(calib_data)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(proj_seed))

    acc = None
    n_total = 0

    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            feats = backbone(x)  # [B, H]
            B, H = feats.shape

            if acc is None:
                acc = torch.zeros(H, pts_dim, device=device, dtype=feats.dtype)

            r = torch.randn(B, pts_dim, generator=gen, device=device, dtype=feats.dtype)
            acc += feats.transpose(0, 1).matmul(r)  # [H, d]
            n_total += B

    pts_out = acc / max(1, n_total)
    # per-dimension z-score (stabilizes geometry)
    for j in range(pts_out.shape[1]):
        pts_out[:, j] = _zscore(pts_out[:, j])

    return pts_out.detach()

def compute_pts_pca_once(
    backbone: nn.Module,
    calib_data: torch.Tensor,
    pts_dim: int = pd,
    batch_size: int = 512,
) -> torch.Tensor:
    backbone.eval()
    ds = TensorDataset(calib_data)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    feats_all = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            feats_all.append(backbone(x))
    A = torch.cat(feats_all, dim=0)          # [N, H]
    A = A - A.mean(dim=0, keepdim=True)

    q = max(pts_dim, 2)
    U, S, V = torch.pca_lowrank(A, q=q)      # V: [H, q]
    pts_out = V[:, :pts_dim].contiguous()    # [H, d]

    for j in range(pts_out.shape[1]):
        pts_out[:, j] = _zscore(pts_out[:, j])

    return pts_out.detach()

def compute_pts_som_once(
    backbone: nn.Module,
    calib_data: torch.Tensor,
    pts_dim: int = pd,
    batch_size: int = 512,
    som_side: int | None = None,
    som_sigma: float = 1.0,
    som_lr: float = 0.5,
    som_steps: int = 5000,
) -> torch.Tensor:
    assert pts_dim == 2
    backbone.eval()
    ds = TensorDataset(calib_data)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    feats_all = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            feats_all.append(backbone(x))
    A = torch.cat(feats_all, dim=0)      # [N, H]
    X = A.transpose(0, 1).detach().cpu().numpy()   # [H, N]

    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-6)

    H = X.shape[0]
    if som_side is None:
        som_side = int(np.ceil(np.sqrt(H)))


    som = MiniSom(som_side, som_side, X.shape[1], sigma=som_sigma, learning_rate=som_lr, random_seed=0)
    som.random_weights_init(X)
    som.train_random(X, som_steps)

    coords = np.array([som.winner(x) for x in X], dtype=np.float32)   # [H, 2]
    pts_out = torch.from_numpy(coords).to(device)

    for j in range(2):
        pts_out[:, j] = _zscore(pts_out[:, j])

    return pts_out.detach()




class Geo_kWTA(torch.nn.Module):
    """
    Spherical gate with 4 parameters: Radius (1) + Center (pts_dim=3).
    We use distance-to-sphere-shell so Radius actually affects the top-k selection.
    """

    def __init__(self, k_frac: float = 0.2, learnable: bool = True, pts_dim: int = 3):
        super().__init__()
        self.k_frac = float(k_frac)
        self.pts_dim = int(pts_dim)

        if learnable:
            self.Radius = nn.Parameter(torch.ones(1))
            self.Center = nn.Parameter(torch.zeros(self.pts_dim))
        else:
            self.Radius = torch.ones(1)
            self.Center = torch.zeros(self.pts_dim)

    def calculate_distance(self, pts: torch.Tensor) -> torch.Tensor:
        r = self.Radius.clamp(min=1e-6)
        c = self.Center.to(device=pts.device, dtype=pts.dtype)
        diff = pts - c
        d = torch.sqrt(torch.sum(diff * diff, dim=1) + 1e-12)  # Euclidean distance to center
        d_l1 = torch.sum(torch.abs(diff), dim=1)
        global GEOSCORE_MODE
        if GEOSCORE_MODE == "center":
            return d  # smaller is better; forward() uses topk(-d)
        elif GEOSCORE_MODE == "shell":
            return (d - r).pow(2)  # original distance-to-shell (squared)
        elif GEOSCORE_MODE == "center_l1":
            return d_l1
        else:
            raise ValueError(f"Unknown GEOSCORE_MODE: {GEOSCORE_MODE}")

    def forward(self, pts):
        d2 = self.calculate_distance(pts)  # [H]
        knum = max(1, int(d2.numel() * self.k_frac))
        idx = torch.topk(-d2, k=knum).indices  # exactly knum
        gate = torch.zeros_like(d2)
        gate[idx] = 1.0
        return gate


class MainModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 2048,
        output_dim: int = 10,
        k_frac: float = 0.4,
        gate_mode="geo",
        pts_dim: int = 3,
        backbone_type: str = "mlp"
    ):
        super().__init__()

        if backbone_type == "identity":
            # Gate will operate directly on input features (e.g., 2048-d ResNet50 cached features)
            assert input_dim == hidden_dim, "For identity backbone, require input_dim == hidden_dim"
            self.backbone = nn.Identity()
        else:
            # Original 3-layer frozen MLP
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

        # gate_mode: "geo" | "random" | "kwta"
        self.gate_mode = gate_mode
        self.k_frac = float(k_frac)
        self.pts_dim = pts_dim

        # Geo-kWTA uses the same sparsity ratio as other gates.
        self.geo_kwta = Geo_kWTA(k_frac=self.k_frac, learnable=True, pts_dim=self.pts_dim)

        # Random gate: a global fixed mask over hidden_dim for fair comparison to global geo gate.
        k_num = max(1, int(hidden_dim * self.k_frac))
        idx = torch.randperm(hidden_dim)[:k_num]
        rand_gate = torch.zeros(hidden_dim)
        rand_gate[idx] = 1.0
        self.register_buffer("random_gate", rand_gate)
        # Best-of-N random gate (filled later by selection; initialized as zeros)
        self.register_buffer("best_random_gate", torch.zeros(hidden_dim))
        self.register_buffer("cls_w", torch.zeros(hidden_dim, output_dim))
        self.register_buffer("cls_b", torch.zeros(output_dim))

        # Per-neuron coordinates (kept as-is for reproducibility under set_seed control).
        self.register_buffer("pts", torch.randn(hidden_dim, self.pts_dim) * 2)

        self.train_features = None
        self.train_labels = None
        self.val_features = None
        self.val_labels = None

        self.freeze_backbone()
        self.ridge_alpha = 1.0

    def get_activation_stats(self):
        with torch.no_grad():
            gate = self._make_gate(self._cached_train_features)
            active = gate.sum().item()
            total = gate.numel()
            return {
                "active_neurons": int(active),
                "total_neurons": int(total),
                "active_ratio": active / total,
            }

    def set_data(self, train_data, train_labels, val_data, val_labels):
        """Set train/val data tensors and clear caches."""
        self.train_features = train_data
        self.train_labels = train_labels
        self.val_features = val_data
        self.val_labels = val_labels

        # Clear torch-ridge (GPU) caches to avoid shape mismatch across different set_data() calls
        for name in [
            "_cached_train_features", "_cached_val_features",
            "_Xtr_z_t", "_Xva_z_t",
            "_x_mu", "_x_sig",
            "_y_mu", "_Ytr_c",
        ]:
            if hasattr(self, name):
                delattr(self, name)

    def _make_gate(self, features: torch.Tensor) -> torch.Tensor:
        # features: [N, hidden_dim]
        if self.gate_mode == "geo":
            return self.geo_kwta(self.pts)  # [hidden_dim] global gate
        if self.gate_mode == "random":
            return self.random_gate  # [hidden_dim] global gate
        if self.gate_mode == "bestof_random":
            return self.best_random_gate  # [hidden_dim] global gate chosen on val
        if self.gate_mode == "kwta":
            # Classic kWTA: select top-k neurons by activation scores.
            # Here we build a global gate using mean activation over training features.
            scores = features.mean(dim=0)  # [hidden_dim]
            k_num = max(1, int(scores.numel() * self.k_frac))
            topk_idx = torch.topk(scores, k=k_num).indices
            gate = torch.zeros_like(scores)
            gate[topk_idx] = 1.0
            return gate
        raise ValueError(f"Unknown gate_mode: {self.gate_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)  # [batch_size, hidden_dim]
        gate = self._make_gate(features)  # [hidden_dim]

        # Apply gate
        gated_features = features * gate.unsqueeze(0)  # [batch_size, hidden_dim]

        # logits = gated_features @ W + b
        logits = torch.matmul(gated_features, self.cls_w) + self.cls_b
        return logits, gated_features, gate

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def get_trainable_params_count(self) -> tuple[int, int, int]:
        """Return trainable parameter counts (kept as-is)."""
        backbone_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        geo_params = sum(p.numel() for p in self.geo_kwta.parameters() if p.requires_grad)
        output_params = sum(p.numel() for p in [self.cls_w, self.cls_b] if p.requires_grad)
        print("未冻结参数（应为0）:", backbone_params, "可训练几何参数:", geo_params)
        return backbone_params, geo_params, output_params


    def evaluate_with_torch_ridge(self, alpha: float = None):
        """
        GPU Ridge (closed-form) using torch.linalg.cholesky + torch.cholesky_solve.
        Works best when self.train_features/self.val_features are already on CUDA.
        """
        if self.train_features is None or self.val_features is None:
            raise ValueError("请先使用 set_data() 设置训练和验证数据")

        if alpha is None:
            alpha = self.ridge_alpha

        self.eval()
        with torch.no_grad():
            # 1) backbone features cache (torch)
            if not hasattr(self, "_cached_train_features"):
                self._cached_train_features = self.backbone(self.train_features)
                self._cached_val_features = self.backbone(self.val_features)
            Xtr = self._cached_train_features
            Xva = self._cached_val_features

            # 2) standardize on GPU once (like StandardScaler: center + scale)
            if not hasattr(self, "_Xtr_z_t"):
                mu = Xtr.mean(dim=0, keepdim=True)
                sig = Xtr.std(dim=0, keepdim=True).clamp_min(1e-6)
                self._x_mu = mu
                self._x_sig = sig
                self._Xtr_z_t = (Xtr - mu) / sig
                self._Xva_z_t = (Xva - mu) / sig

                # labels: keep on device, and implement intercept by centering y
                Ytr = self.train_labels
                self._y_mu = Ytr.mean(dim=0, keepdim=True)  # [1, C]
                self._Ytr_c = (Ytr - self._y_mu)  # centered labels
            else:
                # reuse cached standardized tensors
                pass

            # 3) gate -> column selection
            gate = self._make_gate(Xtr)  # [H]
            idx = gate.bool()
            Xtr_sel = self._Xtr_z_t[:, idx]
            Xva_sel = self._Xva_z_t[:, idx]

            # 4) Ridge closed form on selected dims (solve (XtX + alpha I) W = XtY)
            # do solve in float32 for numerical stability
            A = (Xtr_sel.float().T @ Xtr_sel.float())  # [k, k]
            k = A.shape[0]
            A = A + float(alpha) * torch.eye(k, device=A.device, dtype=A.dtype)

            B = (Xtr_sel.float().T @ self._Ytr_c.float())  # [k, C]

            L = torch.linalg.cholesky(A)  # [k, k]
            W = torch.cholesky_solve(B, L)  # [k, C]
            b = self._y_mu  # [1, C] since X is centered

            # 5) predict
            Yva = self.val_labels
            Ypred = Xva_sel.float() @ W + b

            mse_loss = torch.mean((Yva.float() - Ypred) ** 2).item()

            # acc (one-hot)
            if Yva.dim() == 2 and Yva.size(1) > 1:
                acc = (Ypred.argmax(dim=1) == Yva.argmax(dim=1)).float().mean().item()
            else:
                acc = None

            # 6) update classifier weights buffers (keep same semantics)
            # note: your cls_w expects [hidden_dim, C], but W is [k, C] after gating
            # so only selected dimensions are updated; others stay at 0
            dev = self.cls_w.device
            cls_w_full = torch.zeros_like(self.cls_w, device=dev, dtype=torch.float32)
            cls_w_full[idx] = W.to(device=dev, dtype=torch.float32)
            self.cls_w.copy_(cls_w_full)
            self.cls_b.copy_(b.squeeze(0).to(device=dev, dtype=torch.float32))

            return mse_loss, acc, None

    def evaluatewithtorchridge(self, alpha: float = None):
        return self.evaluate_with_torch_ridge(alpha=alpha)

def _compute_obj_J(acc: float, mse: float, eps: float = 1e-3) -> float:
    # Match your CMA objective: J = (1 - Acc_val) + eps * MSE_val
    return float(1.0 - acc + eps * mse)

def _ensure_ridge_caches_underscore(model, alpha=None):
    """
    Ensure ridge caches exist: _Xtr_z_t, _Xva_z_t, _Ytr_c, _y_mu.
    We call evaluate_with_torch_ridge() once to build them.
    IMPORTANT: if current gate_mode could yield k=0, temporarily switch to 'random'.
    """
    old_mode = getattr(model, "gate_mode", None)
    model.gate_mode = "random"
    try:
        model.evaluate_with_torch_ridge(alpha=alpha if alpha is not None else model.ridge_alpha)
    finally:
        if old_mode is not None:
            model.gate_mode = old_mode


def _ridge_val_metrics_given_gate_underscore(model, gate: torch.Tensor, alpha=None):
    if alpha is None:
        alpha = model.ridge_alpha

    Xtr_sel = model._Xtr_z_t[:, gate.bool()]
    Xva_sel = model._Xva_z_t[:, gate.bool()]
    Yva = model.val_labels

    A = (Xtr_sel.float().T @ Xtr_sel.float())
    k = A.shape[0]
    A = A + float(alpha) * torch.eye(k, device=A.device, dtype=A.dtype)
    B = (Xtr_sel.float().T @ model._Ytr_c.float())

    L = torch.linalg.cholesky(A)
    W = torch.cholesky_solve(B, L)
    b = model._y_mu

    Ypred = Xva_sel.float() @ W + b
    mse = torch.mean((Yva.float() - Ypred) ** 2).item()
    acc = (Ypred.argmax(dim=1) == Yva.argmax(dim=1)).float().mean().item()
    return mse, acc


def select_bestofN_unstructured_random_masks(model, N: int, seed: int, alpha=None, eps_obj: float = 1e-3):
    """
    Best-of-N unstructured random masks on current train/val split:
    - sample N random global top-k masks (exactly k = int(H*k_frac))
    - score each on val using the same ridge math
    - pick the one with minimum J=(1-acc)+eps*mse
    - store to model.best_random_gate
    """
    if alpha is None:
        alpha = model.ridge_alpha

    # build caches if missing
    if not hasattr(model, "_Xtr_z_t"):
        _ensure_ridge_caches_underscore(model, alpha=alpha)

    H = model._Xtr_z_t.shape[1]
    k_num = max(1, int(H * model.k_frac))

    gen = torch.Generator(device=model._Xtr_z_t.device)
    gen.manual_seed(int(seed))

    best_J = float("inf")
    best_gate = None
    best_acc = None
    best_mse = None

    for _ in range(int(N)):
        perm = torch.randperm(H, generator=gen, device=model._Xtr_z_t.device)
        idx = perm[:k_num]
        gate = torch.zeros(H, device=model._Xtr_z_t.device, dtype=model._Xtr_z_t.dtype)
        gate[idx] = 1.0

        mse, acc = _ridge_val_metrics_given_gate_underscore(model, gate, alpha=alpha)
        J = _compute_obj_J(acc=acc, mse=mse, eps=eps_obj)

        if J < best_J:
            best_J, best_gate, best_acc, best_mse = J, gate, acc, mse

    model.best_random_gate.copy_(best_gate)
    return best_gate, best_J, best_acc, best_mse


class CmaEngine:
    def __init__(self, model: MainModel):
        self.model = model
        self.layer = model.geo_kwta

        r0 = self.layer.Radius.detach().cpu().numpy().tolist()  # [r]
        c0 = self.layer.Center.detach().cpu().numpy().tolist()  # [cx, cy, cz]
        self.x0 = r0 + c0  # total dim = 1 + pts_dim

        self.es = cma.CMAEvolutionStrategy(self.x0, sigma0)
        self.pts = model.pts

        print(f"初始球体参数: Radius={self.x0[0]:.4f}, Center={self.x0[1:]}")

    def f(self, x):
        # x: [radius, center_x, center_y, center_z] (when pts_dim=3)
        dev = self.layer.Radius.device

        radius = torch.tensor([x[0]], dtype=torch.float32, device=dev)
        center = torch.tensor(x[1 : 1 + self.layer.pts_dim], dtype=torch.float32, device=dev)

        self.layer.Radius.data = radius
        self.layer.Center.data = center

        try:
            mse_loss, accuracy, _ = self.model.evaluate_with_torch_ridge()
            if accuracy is None:
                return mse_loss
            eps = 1e-3
            return float((1.0 - accuracy) / eps + mse_loss)
        except Exception as e:
            print(f"评估失败: {e}")
            return 1e6

    def step(self, maxiter: int = 20, verbose: bool = True):
        """
        Run CMA-ES optimization.
        Returns: best_params, best_loss, best_accuracy
        """
        total_start_time = time.time()

        best_loss = float("inf")
        best_params = None
        best_accuracy = 0.0

        loop_start_time = time.time()

        for i in range(maxiter):
            solutions = self.es.ask()
            n_workers = min(len(solutions), 1)  #Set Parallel Count

            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                values = list(ex.map(self.f, solutions))
            global FITNESS_SHUFFLE_MODE, FITNESS_SHUFFLE_SEED
            if FITNESS_SHUFFLE_MODE:
                values = fitness_shuffle(values, seed=FITNESS_SHUFFLE_SEED + i)

            self.es.tell(solutions, values)

            if self.es.result.fbest < best_loss:
                best_loss = self.es.result.fbest
                best_params = self.es.result.xbest.copy()

                dev = self.model.geo_kwta.Radius.device
                radius = torch.tensor([best_params[0]], dtype=torch.float32, device=dev)
                center = torch.tensor(
                    best_params[1 : 1 + self.model.geo_kwta.pts_dim],
                    dtype=torch.float32,
                    device=dev,
                )
                self.model.geo_kwta.Radius.data = radius
                self.model.geo_kwta.Center.data = center



            if verbose and i % 10 == 0:
                current_best = self.es.result.fbest
                current_params = self.es.result.xbest
                print(
                    f"迭代 {i:3d}: 最佳损失 = {current_best:.6f}, "
                    f"参数 = [{current_params[0]:.4f}, {current_params[1]:.4f}], "
                    f"最佳准确率 = {best_accuracy:.4f}"
                )

        loop_end_time = time.time() - loop_start_time

        if best_params is not None:
            dev = self.model.geo_kwta.Radius.device
            radius = torch.tensor([best_params[0]], dtype=torch.float32, device=dev)
            center = torch.tensor(
                best_params[1 : 1 + self.model.geo_kwta.pts_dim],
                dtype=torch.float32,
                device=dev,
            )
            self.model.geo_kwta.Radius.data = radius
            self.model.geo_kwta.Center.data = center

            final_loss, final_accuracy, _ = self.model.evaluate_with_torch_ridge()
            if final_accuracy is not None:
                best_accuracy = final_accuracy
            total_end_time = time.time() - total_start_time

            print("\n优化完成!")
            print(f"最佳球体参数: Major={best_params[0]:.6f}, Minor={best_params[1]:.6f}")
            print(f"最佳岭回归损失: {best_loss:.6f}")
            print(f"最佳准确率: {best_accuracy:.4f}")
            print(f"CMA优化用时:{loop_end_time:.2f}")
            print(f"总训练时长:{total_end_time:.2f}")

        return best_params, best_loss, best_accuracy


def load_cifar10_via_torchvision(train_samples=1000, test_samples=200, data_dir="./data"):
    """
    Load CIFAR-10 using torchvision and return flattened tensors + one-hot labels.
    Returns:
        train_data/test_data: [N, 3072]
        train_labels/test_labels: [N, 10] one-hot
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    full_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    full_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    train_data_all, train_labels_all = next(iter(full_train_loader))
    test_data_all, test_labels_all = next(iter(full_test_loader))

    if train_samples is not None and train_samples < train_data_all.size(0):
        idx = torch.randperm(train_data_all.size(0))[:train_samples]
        train_data = train_data_all[idx]
        train_labels = train_labels_all[idx]
    else:
        train_data = train_data_all
        train_labels = train_labels_all

    if test_samples is not None and test_samples < test_data_all.size(0):
        idx = torch.randperm(test_data_all.size(0))[:test_samples]
        test_data = test_data_all[idx]
        test_labels = test_labels_all[idx]
    else:
        test_data = test_data_all
        test_labels = test_labels_all

    num_classes = 10
    train_labels_onehot = F.one_hot(train_labels, num_classes=num_classes).float()
    test_labels_onehot = F.one_hot(test_labels, num_classes=num_classes).float()

    print("[数据加载完成 - CIFAR-10]")
    print(f" 训练集: {train_data.shape} -> 特征维度 {train_data.shape[1]}, 标签 {train_labels_onehot.shape}")
    print(f" 测试集: {test_data.shape} -> 特征维度 {test_data.shape[1]}, 标签 {test_labels_onehot.shape}")
    print(f" 标签示例: 原始 {train_labels[:5].tolist()}, One-Hot 形状 {train_labels_onehot[:1].shape}")

    return train_data, train_labels_onehot, test_data, test_labels_onehot


def load_mnist_via_torchvision(train_samples=1000, test_samples=200, data_dir="./MNIST", download=True):
    """
    Load MNIST using torchvision and return flattened tensors + one-hot labels.
    Returns:
        train_data/test_data: [N, 784]
        train_labels/test_labels: [N, 10] one-hot
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )

    train_dataset = datasets.MNIST(root=data_dir, train=True, download=download, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=download, transform=transform)

    full_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    full_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    train_data_all, train_labels_all = next(iter(full_train_loader))
    test_data_all, test_labels_all = next(iter(full_test_loader))

    if train_samples is not None and train_samples < train_data_all.size(0):
        idx = torch.randperm(train_data_all.size(0))[:train_samples]
        train_data = train_data_all[idx]
        train_labels = train_labels_all[idx]
    else:
        train_data = train_data_all
        train_labels = train_labels_all

    if test_samples is not None and test_samples < test_data_all.size(0):
        idx = torch.randperm(test_data_all.size(0))[:test_samples]
        test_data = test_data_all[idx]
        test_labels = test_labels_all[idx]
    else:
        test_data = test_data_all
        test_labels = test_labels_all

    num_classes = 10
    train_labels_onehot = F.one_hot(train_labels, num_classes=num_classes).float()
    test_labels_onehot = F.one_hot(test_labels, num_classes=num_classes).float()

    print("[Data ready - MNIST]")
    print(f" Train: {train_data.shape} -> dim {train_data.shape[1]}, labels {train_labels_onehot.shape}")
    print(f" Test: {test_data.shape} -> dim {test_data.shape[1]}, labels {test_labels_onehot.shape}")

    return train_data, train_labels_onehot, test_data, test_labels_onehot

def load_cifar10_r50_from_pt(train_samples=None, test_samples=None, data_dir="./data", download=False):
    """
    Reads offline cached ResNet-50 (2048-d) features.
    Expected files:
      {data_dir}/cifar10_r50_train.pt  with keys: 'X' [N,2048], 'y' [N]
      {data_dir}/cifar10_r50_test.pt   with keys: 'X' [N,2048], 'y' [N]
    Returns:
      train_data: [Ntr,2048] float32
      train_labels: [Ntr,10] one-hot float32
      test_data:  [Nte,2048] float32
      test_labels:[Nte,10] one-hot float32
    """
    train_path = os.path.join(data_dir, "cifar10_r50_train.pt")
    test_path  = os.path.join(data_dir, "cifar10_r50_test.pt")

    pack_tr = torch.load(train_path, map_location="cpu")
    pack_te = torch.load(test_path, map_location="cpu")

    Xtr, ytr = pack_tr["X"], pack_tr["y"]
    Xte, yte = pack_te["X"], pack_te["y"]

    # optional subsample (keep same style as existing loaders)
    if train_samples is not None and train_samples < Xtr.size(0):
        idx = torch.randperm(Xtr.size(0))[:train_samples]
        Xtr, ytr = Xtr[idx], ytr[idx]

    if test_samples is not None and test_samples < Xte.size(0):
        idx = torch.randperm(Xte.size(0))[:test_samples]
        Xte, yte = Xte[idx], yte[idx]

    num_classes = 10
    ytr_oh = F.one_hot(ytr, num_classes=num_classes).float()
    yte_oh = F.one_hot(yte, num_classes=num_classes).float()

    print("Data ready - CIFAR10 ResNet50 cached features")
    print("Train:", Xtr.shape, ytr_oh.shape)
    print("Test :", Xte.shape, yte_oh.shape)

    return Xtr.float(), ytr_oh, Xte.float(), yte_oh
def loadcachedpt(tag: str, trainsamples=None, testsamples=None, datadir=".data", download=False):
    # Reads offline cached features: {datadir}/{tag}_train.pt and {tag}_test.pt
    trainpath = os.path.join(datadir, f"{tag}_train.pt")
    testpath  = os.path.join(datadir, f"{tag}_test.pt")

    packtr = torch.load(trainpath, map_location="cpu")
    packte = torch.load(testpath, map_location="cpu")
    Xtr, ytr = packtr["X"], packtr["y"]
    Xte, yte = packte["X"], packte["y"]

    if trainsamples is not None and trainsamples < Xtr.size(0):
        idx = torch.randperm(Xtr.size(0))[:trainsamples]
        Xtr, ytr = Xtr[idx], ytr[idx]
    if testsamples is not None and testsamples < Xte.size(0):
        idx = torch.randperm(Xte.size(0))[:testsamples]
        Xte, yte = Xte[idx], yte[idx]

    num_classes = max(ytr.max().item(), yte.max().item()) + 1
    ytroh = F.one_hot(ytr.long(), num_classes=num_classes).float()
    yteoh = F.one_hot(yte.long(), num_classes=num_classes).float()

    input_dim = int(Xtr.shape[1])
    return Xtr.float(), ytroh, Xte.float(), yteoh

def load_cifar10_vitmae_from_pt(
    train_samples=None,
    test_samples=None,
    data_dir="./data",
    tag="cifar10_vit_base_patch16_224_mae",
):
    """
    Reads offline cached ViT-MAE features saved by extract_vitmae_features.py.

    Expected files:
      {data_dir}/{tag}_train.pt with keys: 'X' [N,H], 'y' [N]
      {data_dir}/{tag}_test.pt  with keys: 'X' [N,H], 'y' [N]
    """
    train_path = os.path.join(data_dir, f"{tag}_train.pt")
    test_path = os.path.join(data_dir, f"{tag}_test.pt")

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            f"Cannot find ViT-MAE cached features:\n  {train_path}\n  {test_path}"
        )

    pack_tr = torch.load(train_path, map_location="cpu")
    pack_te = torch.load(test_path, map_location="cpu")

    Xtr, ytr = pack_tr["X"], pack_tr["y"]
    Xte, yte = pack_te["X"], pack_te["y"]

    # ensure tensors
    if not torch.is_tensor(Xtr): Xtr = torch.tensor(Xtr)
    if not torch.is_tensor(Xte): Xte = torch.tensor(Xte)
    if not torch.is_tensor(ytr): ytr = torch.tensor(ytr)
    if not torch.is_tensor(yte): yte = torch.tensor(yte)

    # optional subsample (same style as your ResNet50 loader)
    if train_samples is not None and train_samples < Xtr.size(0):
        idx = torch.randperm(Xtr.size(0))[:train_samples]
        Xtr, ytr = Xtr[idx], ytr[idx]

    if test_samples is not None and test_samples < Xte.size(0):
        idx = torch.randperm(Xte.size(0))[:test_samples]
        Xte, yte = Xte[idx], yte[idx]

    num_classes = 10
    ytr_oh = F.one_hot(ytr.long(), num_classes=num_classes).float()
    yte_oh = F.one_hot(yte.long(), num_classes=num_classes).float()

    print("Data ready - CIFAR10 ViT-MAE cached features")
    print("  tag  :", tag)
    print("  Train:", Xtr.shape, ytr_oh.shape)
    print("  Test :", Xte.shape, yte_oh.shape)

    return Xtr.float(), ytr_oh, Xte.float(), yte_oh



# Dataset registry (easy to extend without touching training logic)
_DATASET_SPECS = {
    "cifar10": {
        "input_dim": 3072,
        "load": lambda train_samples, test_samples, data_dir, download: load_cifar10_via_torchvision(
            train_samples=train_samples, test_samples=test_samples, data_dir=data_dir
        ),
    },
    "mnist": {
        "input_dim": 784,
        "load": lambda train_samples, test_samples, data_dir, download: load_mnist_via_torchvision(
            train_samples=train_samples, test_samples=test_samples, data_dir=data_dir, download=download
        ),
    },
    "cifar10_r50": {
        "input_dim": 2048,
        "load": lambda train_samples, test_samples, data_dir, download: load_cifar10_r50_from_pt(
            train_samples=train_samples,
            test_samples=test_samples,
            data_dir=data_dir,
            download=download,
        ),
    },
 "cifar10_r50_layer1": dict(inputdim=256,  load=lambda trainsamples, testsamples, datadir, download: loadcachedpt("cifar10_r50_layer1", trainsamples, testsamples, datadir, download)),
"cifar10_r50_layer2": dict(inputdim=512,  load=lambda trainsamples, testsamples, datadir, download: loadcachedpt("cifar10_r50_layer2", trainsamples, testsamples, datadir, download)),
"cifar10_r50_layer3": dict(inputdim=1024, load=lambda trainsamples, testsamples, datadir, download: loadcachedpt("cifar10_r50_layer3", trainsamples, testsamples, datadir, download)),
"cifar10_r50_layer4": dict(inputdim=2048, load=lambda trainsamples, testsamples, datadir, download: loadcachedpt("cifar10_r50_layer4", trainsamples, testsamples, datadir, download)),
    "cifar10_r50_layer4b0": dict(inputdim=2048, load=lambda trainsamples, testsamples, datadir, download: loadcachedpt(
        "cifar10_r50_layer4b0", trainsamples, testsamples, datadir, download)),
    "cifar10_r50_layer4b1": dict(inputdim=2048, load=lambda trainsamples, testsamples, datadir, download: loadcachedpt(
        "cifar10_r50_layer4b1", trainsamples, testsamples, datadir, download)),
    "cifar10_r50_layer4b2": dict(inputdim=2048, load=lambda trainsamples, testsamples, datadir, download: loadcachedpt(
        "cifar10_r50_layer4b2", trainsamples, testsamples, datadir, download)),




    "cifar100_r50": dict(
        input_dim=2048,
        load=lambda train_samples, test_samples, data_dir, download: loadcachedpt(
            "cifar100_r50", train_samples, test_samples, data_dir, download
        ),
    ),
    "cifar100_r50_layer1": dict(
        input_dim=256,
        load=lambda train_samples, test_samples, data_dir, download: loadcachedpt(
            "cifar100_r50_layer1", train_samples, test_samples, data_dir, download
        ),
    ),
    "cifar100_r50_layer2": dict(
        input_dim=512,
        load=lambda train_samples, test_samples, data_dir, download: loadcachedpt(
            "cifar100_r50_layer2", train_samples, test_samples, data_dir, download
        ),
    ),
    "cifar100_r50_layer3": dict(
        input_dim=1024,
        load=lambda train_samples, test_samples, data_dir, download: loadcachedpt(
            "cifar100_r50_layer3", train_samples, test_samples, data_dir, download
        ),
    ),
    "cifar100_r50_layer4": dict(
        input_dim=2048,
        load=lambda train_samples, test_samples, data_dir, download: loadcachedpt(
            "cifar100_r50_layer4", train_samples, test_samples, data_dir, download
        ),
    ),
    "cifar100_r50_layer4b0": dict(
        input_dim=2048,
        load=lambda train_samples, test_samples, data_dir, download: loadcachedpt(
            "cifar100_r50_layer4b0", train_samples, test_samples, data_dir, download
        ),
    ),
    "cifar100_r50_layer4b1": dict(
        input_dim=2048,
        load=lambda train_samples, test_samples, data_dir, download: loadcachedpt(
            "cifar100_r50_layer4b1", train_samples, test_samples, data_dir, download
        ),
    ),
    "cifar100_r50_layer4b2": dict(
        input_dim=2048,
        load=lambda train_samples, test_samples, data_dir, download: loadcachedpt(
            "cifar100_r50_layer4b2", train_samples, test_samples, data_dir, download
        ),
    ),
"stl10_r50": dict(
    input_dim=2048,
    load=lambda train_samples, test_samples, data_dir, download: loadcachedpt(
        "stl10_r50", train_samples, test_samples, data_dir, download
    ),
),
"cifar10_mocov2":       dict(input_dim=2048, load=lambda tr,te,d,dl: loadcachedpt("cifar10_mocov2",      tr,te,d,dl)),
"cifar10_convnext_base_sup": dict(input_dim=1024, load=lambda tr,te,d,dl: loadcachedpt("cifar10_convnext_base_sup",tr,te,d,dl)),
"cifar10_convnextv2_base_mae":dict(input_dim=1024,load=lambda tr,te,d,dl: loadcachedpt("cifar10_convnextv2_base_mae",tr,te,d,dl)),
"cifar10_vit_augreg":   dict(input_dim=768,  load=lambda tr,te,d,dl: loadcachedpt("cifar10_vit_augreg",  tr,te,d,dl)),
"cifar10_vit_mae":      dict(input_dim=768,  load=lambda tr,te,d,dl: loadcachedpt("cifar10_vit_mae",     tr,te,d,dl)),
"cifar10_dino":         dict(input_dim=768,  load=lambda tr,te,d,dl: loadcachedpt("cifar10_dino",        tr,te,d,dl)),
"cifar10_swin":         dict(input_dim=1024, load=lambda tr,te,d,dl: loadcachedpt("cifar10_swin",        tr,te,d,dl)),
"cifar10_vit":          dict(input_dim=768,  load=lambda tr,te,d,dl: loadcachedpt("cifar10_vit",         tr,te,d,dl)),


"cifar100_vit_mae":     dict(input_dim=768,  load=lambda tr,te,d,dl: loadcachedpt("cifar100_vit_mae",    tr,te,d,dl)),
"cifar100_vit_augreg":  dict(input_dim=768,  load=lambda tr,te,d,dl: loadcachedpt("cifar100_vit_augreg", tr,te,d,dl)),
"cifar100_vit_dino":    dict(input_dim=768,  load=lambda tr,te,d,dl: loadcachedpt("cifar100_vit_dino",   tr,te,d,dl)),
"cifar100_vit":         dict(input_dim=768,  load=lambda tr,te,d,dl: loadcachedpt("cifar100_vit",        tr,te,d,dl)),
"cifar100_convnext_base_sup": dict(input_dim=1024, load=lambda tr,te,d,dl: loadcachedpt("cifar100_convnext_base_sup", tr,te,d,dl)),
"cifar100_convnextv2_base_mae": dict(input_dim=1024, load=lambda tr,te,d,dl: loadcachedpt("cifar100_convnextv2_base_mae", tr,te,d,dl)),
"cifar100_mocov2":      dict(input_dim=2048, load=lambda tr,te,d,dl: loadcachedpt("cifar100_mocov2",     tr,te,d,dl)),
"cifar100_swin":        dict(input_dim=1024, load=lambda tr,te,d,dl: loadcachedpt("cifar100_swin",       tr,te,d,dl)),
"cifar100_dino":        dict(input_dim=768,  load=lambda tr,te,d,dl: loadcachedpt("cifar100_dino",       tr,te,d,dl)),


"stl10_vit_mae":        dict(input_dim=768,  load=lambda tr,te,d,dl: loadcachedpt("stl10_vit_mae",       tr,te,d,dl)),
"stl10_vit_augreg":     dict(input_dim=768,  load=lambda tr,te,d,dl: loadcachedpt("stl10_vit_augreg",    tr,te,d,dl)),
"stl10_vit_dino":       dict(input_dim=768,  load=lambda tr,te,d,dl: loadcachedpt("stl10_vit_dino",      tr,te,d,dl)),
"stl10_vit":            dict(input_dim=768,  load=lambda tr,te,d,dl: loadcachedpt("stl10_vit",           tr,te,d,dl)),
"stl10_convnext_base_sup": dict(input_dim=1024, load=lambda tr,te,d,dl: loadcachedpt("stl10_convnext_base_sup", tr,te,d,dl)),
"stl10_convnextv2_base_mae": dict(input_dim=1024, load=lambda tr,te,d,dl: loadcachedpt("stl10_convnextv2_base_mae", tr,te,d,dl)),
"stl10_mocov2":         dict(input_dim=2048, load=lambda tr,te,d,dl: loadcachedpt("stl10_mocov2",        tr,te,d,dl)),
"stl10_swin":           dict(input_dim=1024, load=lambda tr,te,d,dl: loadcachedpt("stl10_swin",          tr,te,d,dl)),
"stl10_dino":           dict(input_dim=768,  load=lambda tr,te,d,dl: loadcachedpt("stl10_dino",          tr,te,d,dl)),



}
VITMAE_NUM_LAYERS = 12
VITMAE_LAYER_TAG_PREFIX = "cifar10_vit_base_patch16_224_mae"  # must match extractor's tag_prefix

for layer_idx in range(VITMAE_NUM_LAYERS):
    ds_key = f"cifar10vitmae_layer{layer_idx:02d}"
    tag = f"{VITMAE_LAYER_TAG_PREFIX}_layer{layer_idx:02d}"
    _DATASET_SPECS[ds_key] = dict(
        inputdim=768,
        load=lambda trainsamples, testsamples, data_dir, download, _tag=tag: loadcachedpt(
            tag=_tag,
            trainsamples=trainsamples,
            testsamples=testsamples,
            datadir=data_dir,
            download=download,
        ),
    )
VITMAE_MEAN_TAG_PREFIX = "cifar10_vit_base_patch16_224_mae_mean"

for layer_idx in range(VITMAE_NUM_LAYERS):
    ds_key = f"cifar10vitmae_mean_layer{layer_idx:02d}"
    tag = f"{VITMAE_MEAN_TAG_PREFIX}_layer{layer_idx:02d}"
    _DATASET_SPECS[ds_key] = dict(
        inputdim=768,
        load=lambda tr, te, d, dl, _tag=tag: loadcachedpt(
            tag=_tag, trainsamples=tr, testsamples=te, datadir=d, download=dl
        ),
    )

# ---- Augreg ViT-B/16 layers ----
AUGREG_NUM_LAYERS = 12
AUGREG_LAYER_TAG_PREFIX = "cifar10_vit_base_patch16_224_augreg_in1k_cls"

for layer_idx in range(AUGREG_NUM_LAYERS):
    ds_key = f"cifar10augreg_layer{layer_idx:02d}"
    tag = f"{AUGREG_LAYER_TAG_PREFIX}_layer{layer_idx:02d}"
    _DATASET_SPECS[ds_key] = dict(
        inputdim=768,
        load=lambda tr, te, d, dl, _tag=tag: loadcachedpt(
            tag=_tag, trainsamples=tr, testsamples=te,
            datadir=d, download=dl
        ),
    )

# ---- Augreg ViT-B/16 mean-pooled layers ----
AUGREG_MEAN_NUM_LAYERS = 12
AUGREG_MEAN_TAG_PREFIX = "cifar10_augreg_mean"

for layer_idx in range(AUGREG_MEAN_NUM_LAYERS):
    ds_key = f"cifar10augreg_mean_layer{layer_idx:02d}"
    tag = f"{AUGREG_MEAN_TAG_PREFIX}_layer{layer_idx:02d}"
    _DATASET_SPECS[ds_key] = dict(
        inputdim=768,
        load=lambda tr, te, d, dl, _tag=tag: loadcachedpt(
            tag=_tag, trainsamples=tr, testsamples=te,
            datadir=d, download=dl
        ),
    )

def load_dataset(dataset: str, train_samples, test_samples, data_dir: str, download: bool = False):
    """Generic dataset entry point."""
    key = str(dataset).lower()
    if key not in _DATASET_SPECS:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(_DATASET_SPECS.keys())}")

    spec = _DATASET_SPECS[key]
    train_data, train_labels, test_data, test_labels = spec["load"](train_samples, test_samples, data_dir, download)

    # Use actual feature dimension if we're loading cached features
    input_dim = int(train_data.shape[1])
    return train_data, train_labels, test_data, test_labels, input_dim

def split_train_val(train_data: torch.Tensor, train_labels: torch.Tensor, val_ratio: float = 0.1, seed: int = 0):
    """Split a training set into (train, val) without touching the official test set."""
    set_seed(seed)
    n = train_data.size(0)
    n_val = max(1, int(n * val_ratio))
    perm = torch.randperm(n)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    return train_data[tr_idx], train_labels[tr_idx], train_data[val_idx], train_labels[val_idx]


def sweep_kfrac_to_yaml(
    out_path="results_kfrac.yaml",
    seed=0,
    gate_mode="geo",
    sigma0_override=None,
    maxiter=20,
    hidden_dim=512,
    ridge_alpha=1.0,
    data_dir="./data",
    cma_train_n=4000,
    cma_val_n=800,
    dataset="cifar10",
    dataset_download=False,
):
    if sigma0_override is not None:
        global sigma0
        sigma0 = float(sigma0_override)

    set_seed(seed)

    # Load the full dataset once (sampling happens later for CMA).
    train_all, labels_all, test_data, test_labels, input_dim = load_dataset(
        dataset=dataset,
        train_samples=None,
        test_samples=None,
        data_dir=data_dir,
        download=dataset_download,


    )
    num_classes = labels_all.shape[1]
    _cached_feature_datasets = {"cifar10_r50", "cifar10_vitmae", "cifar10_r50_layer1", "cifar10_r50_layer2",
                                "cifar10_r50_layer3", "cifar10_r50_layer4","cifar10_r50_layer4b0", "cifar10_r50_layer4b1",
                                "cifar10_r50_layer4b2", "cifar100_r50",
    "cifar100_r50_layer1", "cifar100_r50_layer2", "cifar100_r50_layer3", "cifar100_r50_layer4",
    "cifar100_r50_layer4b0", "cifar100_r50_layer4b1", "cifar100_r50_layer4b2", "stl10_r50", "cifar10_mocov2",
    "cifar10_convnext_base_sup",
    "cifar10_convnextv2_base_mae",
    "cifar10_vit_augreg",
    "cifar10_vit_mae",
    "cifar10_dino",
    "cifar10_swin","cifar100_vit_mae","stl10_vit_mae","cifar100_dino","stl10_dino","cifar100_convnext_base_sup",
                                "cifar100_convnextv2_base_mae","stl10_convnextv2_base_mae","stl10_convnext_base_sup",
                                "cifar100_swin","stl10_swin","cifar10_vit" , "cifar100_vit", "stl10_vit","stl10_mocov2","cifar100_mocov2"}
    for layer_idx in range(VITMAE_NUM_LAYERS):
        _cached_feature_datasets.add(f"cifar10vitmae_layer{layer_idx:02d}")
    for layer_idx in range(VITMAE_NUM_LAYERS):
        _cached_feature_datasets.add(f"cifar10vitmae_mean_layer{layer_idx:02d}")
    for layer_idx in range(AUGREG_NUM_LAYERS):
        _cached_feature_datasets.add(f"cifar10augreg_layer{layer_idx:02d}")
    for layer_idx in range(AUGREG_MEAN_NUM_LAYERS):
        _cached_feature_datasets.add(f"cifar10augreg_mean_layer{layer_idx:02d}")
    backbone_type = "identity" if str(dataset).lower() in _cached_feature_datasets else "mlp"
    hidden_dim_eff = input_dim if backbone_type == "identity" else hidden_dim
    # IMPORTANT: create a validation split from the official training set.
    # The official test set must be untouched during hyperparameter/model selection.
    _VAL_RATIO = 0.1
    train_data, train_labels, val_data, val_labels = split_train_val(train_all, labels_all, val_ratio=_VAL_RATIO, seed=seed)

    results = {
        "meta": {
            "seed": seed,
            "gate_mode": gate_mode,
            "geo_gate_type": "sphere_center_radius",
            "pts_dim": pd,
            "sigma0": sigma0,
            "maxiter": maxiter,
            "hidden_dim": hidden_dim,
            "ridge_alpha": ridge_alpha,
            "data_dir": data_dir,
            "val_ratio": _VAL_RATIO,
        },
        "runs": [],
    }

    # ---- NEW: fixed pts for geo mode (shared across all k_frac) ----
    pts_fixed = None
    if gate_mode == "geo" and GEO_PTS_INIT in ["meanvar", "randproj", "external","pca","som"]:
        set_seed(seed)

        if GEO_PTS_INIT == "external":

            pack = torch.load(PTS_EXTERNAL_PATH, map_location="cpu")
            pts_fixed = pack["pts"].to(device)
            print(f"External coordinates loaded: {pts_fixed.shape} from {PTS_EXTERNAL_PATH}")

        else:
            # meanvar / randproj need a probe model and calib_data
            calib_n = min(cma_train_n, train_data.shape[0])
            calib_idx = torch.randperm(train_data.shape[0])[:calib_n]
            calib_data = train_data[calib_idx].to(device)

            probe = MainModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim_eff,
                output_dim=num_classes,
                k_frac=0.5,
                gate_mode="geo",
                pts_dim=pd,
                backbone_type=backbone_type,
            ).to(device)

            if GEO_PTS_INIT == "meanvar":
                pts_fixed = compute_pts_meanvar_once(
                    backbone=probe.backbone,
                    calib_data=calib_data,
                    pts_dim=pd,
                    batch_size=512,
                )
            elif GEO_PTS_INIT == "randproj":
                pts_fixed = compute_pts_randproj_once(
                    backbone=probe.backbone,
                    calib_data=calib_data,
                    pts_dim=pd,
                    batch_size=512,
                    proj_seed=seed,
                )
            elif GEO_PTS_INIT == "pca":
                pts_fixed = compute_pts_pca_once(
                    backbone=probe.backbone,
                    calib_data=calib_data,
                    pts_dim=pd,
                    batch_size=512,
                )
            elif GEO_PTS_INIT == "som":
                pts_fixed = compute_pts_som_once(
                    backbone=probe.backbone,
                    calib_data=calib_data,
                    pts_dim=pd,  # make sure the pd = 2
                    batch_size=512,
                )

            print("CWD:", os.getcwd())
            print("dataset=", dataset, "backbone_type=", backbone_type, "hidden_dim_eff=", hidden_dim_eff)
            print("probe.backbone:", probe.backbone)
            print("calib_data:", calib_data.shape)
            print("pts_fixed:", pts_fixed.shape)
            del probe



    # ---------------------------------------------------------------

    k_list = [round(i * 0.05, 2) for i in range(1, 21)]
    #k_list = [0.05,0.10,0.20,0.30,0.40,0.50]     #per k_frac test
    if pts_fixed is not None:
        torch.save(
            {"seed": seed, "GEO_PTS_INIT": GEO_PTS_INIT, "pts_fixed": pts_fixed.detach().cpu()},
            f"pts_fixed_seed{seed}_{GEO_PTS_INIT}.pt"
        )
    for k_frac in k_list:
        # Keep randomness identical across k_frac (controlled variables).
        t0 = time.time()
        set_seed(seed)

        Ntr = train_data.shape[0]
        Nva = val_data.shape[0]

        tr_idx = torch.randperm(Ntr)[:cma_train_n]
        va_idx = torch.randperm(Nva)[:cma_val_n]

        train_data_cma = train_data[tr_idx]
        train_labels_cma = train_labels[tr_idx]
        val_data_cma = val_data[va_idx]
        val_labels_cma = val_labels[va_idx]

        model = MainModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim_eff,
            output_dim=num_classes,
            k_frac=k_frac,
            gate_mode=gate_mode,
            pts_dim=pd,
            backbone_type=backbone_type,
        ).to(device)

        if pts_fixed is not None:
            model.pts.copy_(pts_fixed)

        if Shuffle_mode == True:
            shuffle_pts_rows_inplace(model, seed=seed)

        model.ridge_alpha = ridge_alpha

        # (A)/(B) Gate selection on train/val (NO test used)
        best_params, best_loss, best_acc = None, None, None

        if gate_mode == "geo":
            model.set_data(
                train_data.to(device),
                train_labels.to(device),
                val_data.to(device),
                val_labels.to(device),
            )
            initial_loss, initial_acc, _ = model.evaluate_with_torch_ridge()
            optimizer = CmaEngine(model)
            best_params, best_loss, best_acc = optimizer.step(maxiter=maxiter, verbose=False)

        elif gate_mode == "bestof_random":
            # Use full train/val split for selection (NO test), and match *total* gate-evaluation budget to geo's two-stage CMA
            model.set_data(
                train_data.to(device),
                train_labels.to(device),
                val_data.to(device),
                val_labels.to(device),
            )

            # Compute CMA popsize for dimension (1 + pd) so budget matches your geo search.
            es_tmp = cma.CMAEvolutionStrategy([1.0] + [0.0] * pd, sigma0, {"verbose": -9})
            popsize = int(es_tmp.popsize)
            del es_tmp

            # Geo does two optimizer.step() calls; each step evaluates about popsize*maxiter + 1 candidates.
            N_total = popsize * int(maxiter)

            # Select best random mask using the same objective form J = (1-acc) + 1e-3*mse
            _, best_loss, best_acc, _ = select_bestofN_unstructured_random_masks(
                model, N=N_total, seed=seed, alpha=ridge_alpha, eps_obj=1e-3
            )

            # initial_* for logging only (optional)
            initial_loss, initial_acc, _ = model.evaluate_with_torch_ridge()

        else:
            # No search: just compute val metrics once, then proceed to final test evaluation.
            model.set_data(
                train_data.to(device),
                train_labels.to(device),
                val_data.to(device),
                val_labels.to(device),
            )
            initial_loss, initial_acc, _ = model.evaluate_with_torch_ridge()
            best_loss = _compute_obj_J(acc=initial_acc, mse=initial_loss, eps=1e-3)
            best_acc = initial_acc

        # (C) Final evaluation on the untouched official test set (one-shot)
        # Refit Ridge on all available training data (train+val) and evaluate on test.
        model.set_data(
            train_all.to(device),
            labels_all.to(device),
            test_data.to(device),
            test_labels.to(device),
        )

        if best_params is not None:
            dev = model.geo_kwta.Radius.device
            model.geo_kwta.Radius.data = torch.tensor([best_params[0]], dtype=torch.float32, device=dev)
            model.geo_kwta.Center.data = torch.tensor(
                best_params[1 : 1 + model.geo_kwta.pts_dim],
                dtype=torch.float32,
                device=dev,
            )

        test_loss, test_acc, _ = model.evaluate_with_torch_ridge()
        stats = model.get_activation_stats()
        dt = time.time() - t0
        results["runs"].append(
            {
                "k_frac": float(k_frac),
                "initial_acc": None if initial_acc is None else float(initial_acc),
                "best_acc": None if best_acc is None else float(best_acc),
                "best_loss": float(best_loss),
                "best_radius": float(best_params[0]) if best_params is not None else None,
                "best_center": [float(v) for v in (best_params[1 : 1 + model.geo_kwta.pts_dim])] if best_params is not None else None,
                "test_acc": None if test_acc is None else float(test_acc),
                "test_loss": float(test_loss),
                "active_ratio": float(stats["active_ratio"]),
                "active_neurons": int(stats["active_neurons"]),
                "total_neurons": int(stats["total_neurons"]),
                "elapsed_sec": float(dt)
            }
        )

        print(f"k_frac={k_frac:.2f} best_acc={best_acc:.4f}")

        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(results, f, allow_unicode=True, sort_keys=False)

        print(f"Saved: {out_path}")

        print(f"[time] k_frac={k_frac:.2f} total={dt:.2f}s")

    return results


def get_cifar10_semantic_subsets(data, labels_onehot):
    """
    Split CIFAR10 data into Artificial and Biological subsets.
    Artificial: 0(airplane), 1(automobile), 8(ship), 9(truck)
    Biological: 2(bird), 3(cat), 4(deer), 5(dog), 6(frog), 7(horse)
    """
    # labels_onehot is [N, 10]
    labels = torch.argmax(labels_onehot, dim=1)
    
    artificial_indices = []
    biological_indices = []
    
    # CIFAR10 class mapping
    art_classes = {0, 1, 8, 9}
    bio_classes = {2, 3, 4, 5, 6, 7}
    
    for i, label in enumerate(labels):
        l = label.item()
        if l in art_classes:
            artificial_indices.append(i)
        elif l in bio_classes:
            biological_indices.append(i)
            
    artificial_indices = torch.tensor(artificial_indices, dtype=torch.long, device=data.device)
    biological_indices = torch.tensor(biological_indices, dtype=torch.long, device=data.device)
    
    return (data[artificial_indices], labels_onehot[artificial_indices]), \
           (data[biological_indices], labels_onehot[biological_indices])


def run_semantic_experiment(
    out_path="results_semantic.yaml",
    seed=0,
    gate_mode="geo",
    hidden_dim=512,
    ridge_alpha=1.0,
    data_dir="./data",
    dataset="cifar10",
    maxiter=20,
):
    """
    Refactored Experiment to avoid Test-Set Leakage.
    1. Train model (Gate+Ridge) using ONLY Train+Val data.
    2. Identify 'Artificial' neurons using trained weights.
    3. Perturb them (Zero/Shuffle).
    4. Measure impact on Val (primary) and Test (reporting) subsets.
    """
    set_seed(seed)
    
    # 1. Load Data (Split Train/Test)
    train_all, labels_all, test_data, test_labels, input_dim = load_dataset(
        dataset=dataset,
        train_samples=None,
        test_samples=None,
        data_dir=data_dir,
    )
    
    # 2. Split Train -> Train/Val
    train_data, train_labels, val_data, val_labels = split_train_val(
        train_all, labels_all, val_ratio=0.1, seed=seed
    )
    
    # 3. Setup Model
    _cached_feature_datasets = {"cifar10_r50", "cifar10_vitmae", "cifar10_r50_layer1", "cifar10_r50_layer2",
                                "cifar10_r50_layer3", "cifar10_r50_layer4","cifar10_r50_layer4b0", "cifar10_r50_layer4b1",
                                "cifar10_r50_layer4b2"}
    backbone_type = "identity" if str(dataset).lower() in _cached_feature_datasets else "mlp"
    hidden_dim_eff = input_dim if backbone_type == "identity" else hidden_dim
    
    # Use k_frac=0.2 as requested
    k_frac = 0.2
    
    model = MainModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim_eff,
        output_dim=10,
        k_frac=k_frac,
        gate_mode=gate_mode,
        pts_dim=pd, # global pd
        backbone_type=backbone_type,
    ).to(device)
    
    # Initialize pts if needed
    if gate_mode == "geo" and GEO_PTS_INIT == "randproj":
        # Use train_data for calibration (not train_all, just to be safe, though train_all is fine as it excludes test)
        calib_n = min(6400, train_data.shape[0])
        calib_idx = torch.randperm(train_data.shape[0])[:calib_n]
        calib_data = train_data[calib_idx].to(device)
        pts_fixed = compute_pts_randproj_once(
            backbone=model.backbone,
            calib_data=calib_data,
            pts_dim=pd,
            batch_size=512,
            proj_seed=seed,
        )
        model.pts.copy_(pts_fixed)
    
    # 4. Train (Optimize Gate) on Train/Val
    model.set_data(train_data.to(device), train_labels.to(device), val_data.to(device), val_labels.to(device))
    
    print(f"Training gate on Train/Val with k_frac={k_frac}...")
    optimizer = CmaEngine(model)
    best_params, best_loss, best_acc = optimizer.step(maxiter=maxiter, verbose=False)
    
    # Set best params
    if best_params is not None:
        dev = model.geo_kwta.Radius.device
        model.geo_kwta.Radius.data = torch.tensor([best_params[0]], dtype=torch.float32, device=dev)
        model.geo_kwta.Center.data = torch.tensor(
            best_params[1 : 1 + model.geo_kwta.pts_dim],
            dtype=torch.float32,
            device=dev,
        )
    
    # 5. Final Ridge Training on Train+Val
    # We use train_all (Train+Val) to learn the classifier weights.
    # We evaluate on val_data to get a baseline validation accuracy.
    # NO TEST DATA used here.
    model.set_data(train_all.to(device), labels_all.to(device), val_data.to(device), val_labels.to(device))
    base_loss, base_val_acc, _ = model.evaluate_with_torch_ridge()
    print(f"Baseline Val Acc: {base_val_acc:.4f}")
    
    # 6. Semantic Analysis Preparation
    # Get Subsets for Val and Test
    (val_art, val_art_labels), (val_bio, val_bio_labels) = get_cifar10_semantic_subsets(val_data.to(device), val_labels.to(device))
    (test_art, test_art_labels), (test_bio, test_bio_labels) = get_cifar10_semantic_subsets(test_data.to(device), test_labels.to(device))
    
    results = {}
    results["baseline_val_acc"] = float(base_val_acc)

    def eval_subset(data, labels, name):
        model.eval()
        with torch.no_grad():
            logits, _, _ = model(data)
            preds = logits.argmax(dim=1)
            targets = labels.argmax(dim=1)
            acc = (preds == targets).float().mean().item()
        print(f"{name} Acc: {acc:.4f}")
        return acc

    print("--- Baseline Evaluation ---")
    results["baseline_val_art_acc"] = eval_subset(val_art, val_art_labels, "Val Artificial")
    results["baseline_val_bio_acc"] = eval_subset(val_bio, val_bio_labels, "Val Biological")
    results["baseline_test_art_acc"] = eval_subset(test_art, test_art_labels, "Test Artificial")
    results["baseline_test_bio_acc"] = eval_subset(test_bio, test_bio_labels, "Test Biological")
    
    # 7. Identify Artificial Neurons
    # Use trained weights (model.cls_w) to identify neurons
    gate = model._make_gate(None) # Input ignored in geo mode
    
    W = model.cls_w # [H, 10]
    art_classes = [0, 1, 8, 9]
    bio_classes = [2, 3, 4, 5, 6, 7]
    
    art_score = W[:, art_classes].abs().sum(dim=1)
    bio_score = W[:, bio_classes].abs().sum(dim=1)
    
    is_art_neuron = (art_score > bio_score) & (gate > 0)
    art_neuron_indices = torch.nonzero(is_art_neuron).squeeze()
    
    is_bio_neuron = (bio_score > art_score) & (gate > 0)
    bio_neuron_indices = torch.nonzero(is_bio_neuron).squeeze()
    
    print(f"Identified {len(art_neuron_indices)} Artificial Neurons and {len(bio_neuron_indices)} Biological Neurons out of {int(gate.sum())} active neurons.")
    results["num_art_neurons"] = int(len(art_neuron_indices))
    results["num_bio_neurons"] = int(len(bio_neuron_indices))
    results["num_active_neurons"] = int(gate.sum())
    
    # Perturbation Helper
    def run_perturbation(indices, pert_type, target_name):
        saved_pts = model.pts.clone()
        
        if pert_type == "zero":
            model.pts[indices] = 0.0
        elif pert_type == "shuffle":

            all_indices = torch.randperm(model.pts.shape[0], device=device)
            random_pts = model.pts[all_indices[:len(indices)]] # Take random positions from the pool
            model.pts[indices] = random_pts
            
        print(f"--- Perturbation: {pert_type.capitalize()} {target_name} Neurons Pts ---")
        
        # Eval
        prefix = f"{pert_type}_{target_name.lower()[:3]}" # e.g. zero_art, shuffle_bio
        
        res_val_art = eval_subset(val_art, val_art_labels, f"Val Art ({pert_type} {target_name})")
        res_val_bio = eval_subset(val_bio, val_bio_labels, f"Val Bio ({pert_type} {target_name})")
        res_test_art = eval_subset(test_art, test_art_labels, f"Test Art ({pert_type} {target_name})")
        res_test_bio = eval_subset(test_bio, test_bio_labels, f"Test Bio ({pert_type} {target_name})")
        
        results[f"{prefix}_val_art_acc"] = res_val_art
        results[f"{prefix}_val_bio_acc"] = res_val_bio
        results[f"{prefix}_test_art_acc"] = res_test_art
        results[f"{prefix}_test_bio_acc"] = res_test_bio
        
        # Restore
        model.pts.copy_(saved_pts)

    # 8. Run Experiments
    # Zero Artificial
    run_perturbation(art_neuron_indices, "zero", "Artificial")
    # Shuffle Artificial
    run_perturbation(art_neuron_indices, "shuffle", "Artificial")
    
    # Zero Biological
    run_perturbation(bio_neuron_indices, "zero", "Biological")
    # Shuffle Biological
    run_perturbation(bio_neuron_indices, "shuffle", "Biological")
    
    # Save Results
    print(f"Saving results to {out_path}...")
    with open(out_path, "w") as f:
        yaml.dump(results, f)



if __name__ == "__main__":
    # ----------------- USER SWITCHES  -----------------
    # EXP_MODE:
    #   "geo_r"  -> Geo gate + random projection pts   => results_kfrac_geo_r{seed}.yaml
    #   "random" -> pure random mask baseline          => results_kfrac_random{seed}.yaml
    #   "geo_m"  -> Geo gate + meanvar pts             => results_kfrac_geo_m{seed}.yaml
    #   "kwta"   -> Global kWTA method                 => results_kfrac_geo
    #   "semantic_test"
    EXP_MODE = "geo_r"

    SEED_START = 0
    SEED_END = 9 # inclusive


    MAXITER = 20 #You can set it as "0" if you wanna run random or kwta mode.
    DATASET = "cifar10_r50" #mnist cifar10 cifar10_r50 cifar10_vitmae
    DATA_DIR = "./data"
    # ---------------------------------------------------------------

    print(f"Starting experiment loop. EXP_MODE={EXP_MODE}, SEED_START={SEED_START}, SEED_END={SEED_END}")

    for sd in range(SEED_START, SEED_END + 1):
        print(f"Running seed {sd}")
        if EXP_MODE == "geo_r":
            GEO_PTS_INIT = "randproj"
            gate_mode = "geo"
            shuffle_suffix = "_shuf" if Shuffle_mode else ""
            out_path = f"results_{DATASET}_geo_r{shuffle_suffix}_sigma0.50_sd{sd}.yaml"
            sweep_kfrac_to_yaml(
                out_path=out_path,
                seed=sd,
                gate_mode=gate_mode,
                maxiter=MAXITER,
                dataset=DATASET,
                data_dir=DATA_DIR,
            )
        elif EXP_MODE == "random":
            GEO_PTS_INIT = "randn"  # irrelevant for random gate, kept explicit
            gate_mode = "random"
            out_path = f"results_kfrac_random_cifar100_sd{sd}.yaml"
            sweep_kfrac_to_yaml(
                out_path=out_path,
                seed=sd,
                gate_mode=gate_mode,
                maxiter=MAXITER,
                dataset=DATASET,
                data_dir=DATA_DIR,
            )
        elif EXP_MODE == "geo_m":
            GEO_PTS_INIT = "meanvar"
            gate_mode = "geo"
            shuffle_suffix = "_shuf" if Shuffle_mode else ""
            out_path = f"results_{DATASET}_geo_m{shuffle_suffix}_sd{sd}.yaml"
            sweep_kfrac_to_yaml(
                out_path=out_path,
                seed=sd,
                gate_mode=gate_mode,
                maxiter=MAXITER,
                dataset=DATASET,
                data_dir=DATA_DIR,
            )
        elif EXP_MODE == "kwta":
            GEO_PTS_INIT = "randn"  # pts is useless in this mode
            gate_mode = "kwta"
            out_path = f"results_kfrac_kwta_sd{sd}.yaml"
            sweep_kfrac_to_yaml(
                out_path=out_path,
                seed=sd,
                gate_mode=gate_mode,
                maxiter=MAXITER,
                dataset=DATASET,
                data_dir=DATA_DIR,
            )
        elif EXP_MODE == "semantic_test":
            GEO_PTS_INIT = "randproj"
            gate_mode = "geo"
            out_path = f"results_semantic_test{sd}.yaml"
            run_semantic_experiment(
                out_path=out_path,
                seed=sd,
                gate_mode=gate_mode,
                maxiter=MAXITER,
                dataset=DATASET,
                data_dir=DATA_DIR,
            )
        elif EXP_MODE == "bestof_random":
            # best-of-N unstructured random masks (budget-matched to geo CMA-ES)
            GEO_PTS_INIT = "randn"  # irrelevant here; keep explicit for consistency
            gate_mode = "bestof_random"
            out_path = f"results_{DATASET}_sd{sd}.yaml"
            sweep_kfrac_to_yaml(
                out_path=out_path,
                seed=sd,
                gate_mode=gate_mode,
                maxiter=MAXITER,
                dataset=DATASET,
                data_dir=DATA_DIR,
            )
        elif EXP_MODE == "geo_r_external":
            GEO_PTS_INIT = "external"
            gate_mode = "geo"
            out_path = f"results_kfrac_geo_r_external_shuffled_sd{sd}.yaml"
            sweep_kfrac_to_yaml(
                out_path=out_path,
                seed=sd,
                gate_mode=gate_mode,
                maxiter=MAXITER,
                dataset=DATASET,
                data_dir=DATA_DIR,
            )

        elif EXP_MODE == "geo_pca":
            GEO_PTS_INIT = "pca"
            gate_mode = "geo"
            shuffle_suffix = "_shuf" if Shuffle_mode else ""
            out_path = f"results_{DATASET}_geo_pca{shuffle_suffix}_sd{sd}.yaml"
            sweep_kfrac_to_yaml(out_path=out_path,
                seed=sd,
                gate_mode=gate_mode,
                maxiter=MAXITER,
                dataset=DATASET,
                data_dir=DATA_DIR,)

        elif EXP_MODE == "geo_som":
            GEO_PTS_INIT = "som"
            gate_mode = "geo"
            out_path = f"results_{DATASET}_geo_som_shuffled_sd{sd}.yaml"
            sweep_kfrac_to_yaml(out_path=out_path,
                seed=sd,
                gate_mode=gate_mode,
                maxiter=MAXITER,
                dataset=DATASET,
                data_dir=DATA_DIR,)

        else:
            raise ValueError(f"Unknown EXP_MODE: {EXP_MODE}")
