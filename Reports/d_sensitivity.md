# Coordinate-dimension (d) sensitivity — shuffle gap, low-k [0.05,0.30]

gap = orig - same-seed shuffled (same run, same split); per-k Holm-corrected within combo.

## randproj (geo_r)

| dataset | model | d | low-k mean gap | sig k (Holm) / n_k |
|---|---|---|---|---|
| cifar10 | r50 | 1 | +0.0196 | 5/6 |
| cifar10 | r50 | 2 | +0.0252 | 6/6 |
| cifar10 | r50 | 3 | +0.0261 | 6/6 |
| cifar10 | r50 | 5 | +0.0285 | 6/6 |
| cifar10 | r50 | 8 | +0.0253 | 6/6 |
| cifar10 | vit | 1 | -0.0004 | 0/6 |
| cifar10 | vit | 2 | -0.0006 | 0/6 |
| cifar10 | vit | 3 | +0.0016 | 0/6 |
| cifar10 | vit | 5 | -0.0008 | 0/6 |
| cifar10 | vit | 8 | -0.0023 | 0/6 |
| cifar100 | r50 | 1 | +0.0314 | 6/6 |
| cifar100 | r50 | 2 | +0.0439 | 6/6 |
| cifar100 | r50 | 3 | +0.0483 | 6/6 |
| cifar100 | r50 | 5 | +0.0535 | 6/6 |
| cifar100 | r50 | 8 | +0.0454 | 6/6 |
| cifar100 | vit | 1 | -0.0017 | 0/6 |
| cifar100 | vit | 2 | -0.0001 | 0/6 |
| cifar100 | vit | 3 | -0.0012 | 0/6 |
| cifar100 | vit | 5 | -0.0028 | 0/6 |
| cifar100 | vit | 8 | -0.0025 | 0/6 |
| stl10 | r50 | 1 | +0.0199 | 3/6 |
| stl10 | r50 | 2 | +0.0205 | 3/6 |
| stl10 | r50 | 3 | +0.0256 | 6/6 |
| stl10 | r50 | 5 | +0.0237 | 5/6 |
| stl10 | r50 | 8 | +0.0176 | 3/6 |
| stl10 | vit | 1 | +0.0004 | 0/6 |
| stl10 | vit | 2 | -0.0017 | 0/6 |
| stl10 | vit | 3 | -0.0006 | 0/6 |
| stl10 | vit | 5 | -0.0012 | 0/6 |
| stl10 | vit | 8 | -0.0023 | 0/6 |

## pca (geo_pca)

| dataset | model | d | low-k mean gap | sig k (Holm) / n_k |
|---|---|---|---|---|
| cifar10 | r50 | 1 | -0.0031 | 1/6 |
| cifar10 | r50 | 2 | +0.0012 | 0/6 |
| cifar10 | r50 | 3 | +0.0093 | 2/6 |
| cifar10 | r50 | 5 | +0.0075 | 2/6 |
| cifar10 | r50 | 8 | +0.0090 | 1/6 |
| cifar10 | vit | 1 | +0.0022 | 0/6 |
| cifar10 | vit | 2 | +0.0060 | 2/6 |
| cifar10 | vit | 3 | +0.0044 | 1/6 |
| cifar10 | vit | 5 | +0.0041 | 2/6 |
| cifar10 | vit | 8 | +0.0019 | 0/6 |
| cifar100 | r50 | 1 | +0.0056 | 0/6 |
| cifar100 | r50 | 2 | +0.0206 | 4/6 |
| cifar100 | r50 | 3 | +0.0256 | 6/6 |
| cifar100 | r50 | 5 | +0.0341 | 6/6 |
| cifar100 | r50 | 8 | +0.0386 | 6/6 |
| cifar100 | vit | 1 | +0.0009 | 0/6 |
| cifar100 | vit | 2 | +0.0005 | 0/6 |
| cifar100 | vit | 3 | +0.0006 | 0/6 |
| cifar100 | vit | 5 | -0.0013 | 0/6 |
| cifar100 | vit | 8 | -0.0005 | 0/6 |
| stl10 | r50 | 1 | -0.0186 | 0/6 |
| stl10 | r50 | 2 | -0.0054 | 0/6 |
| stl10 | r50 | 3 | -0.0012 | 0/6 |
| stl10 | r50 | 5 | -0.0019 | 0/6 |
| stl10 | r50 | 8 | -0.0075 | 0/6 |
| stl10 | vit | 1 | +0.0004 | 0/6 |
| stl10 | vit | 2 | +0.0022 | 0/6 |
| stl10 | vit | 3 | +0.0013 | 0/6 |
| stl10 | vit | 5 | +0.0010 | 0/6 |
| stl10 | vit | 8 | -0.0006 | 0/6 |
