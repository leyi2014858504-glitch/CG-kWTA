import torch
@torch.no_grad()
def shuffle_pts_rows_inplace(model, seed: int = 0):
    """
    Shuffle neuron coordinates by permuting rows of model.pts (H, ptsdim).
    This preserves each neuron's 3D coordinate as a tuple, but breaks neuron-to-coordinate assignment.
    """
    dev = model.pts.device
    g = torch.Generator(device=dev)
    g.manual_seed(int(seed))
    perm = torch.randperm(model.pts.shape[0], generator=g, device=dev)
    model.pts.copy_(model.pts[perm])

@torch.no_grad()
def shuffleptsrowsinplace(model, seed: int = 0):
    dev = model.pts.device
    g = torch.Generator(device=dev)
    g.manual_seed(int(seed))
    perm = torch.randperm(model.pts.shape[0], generator=g, device=dev)
    model.pts.copy_(model.pts[perm])

def fitness_shuffle(values, seed: int = 0):
    """
    Permute the fitness list (values) but keep solutions order unchanged.
    This breaks the optimization signal of CMA-ES while preserving
    evaluation budget and code path.
    """
    if values is None or len(values) <= 1:
        return values
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    perm = torch.randperm(len(values), generator=g).tolist()
    return [values[i] for i in perm]