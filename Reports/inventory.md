# Rebuttal data inventory (completeness audit)

## d-sweep (dim_sweep/{geo_r|geo_pca}/{ds}_{model}_d{d})
  -> 60/60 combos complete (2 coords x 3 datasets x 2 models x 5 dims)

## multi-shuffle nulls (rebuttal/{ds}_{model}_{geo_r|geo_m})
  cifar10_r50_geo_r: 10/10 seed indexes
  cifar10_r50_geo_m: 10/10 seed indexes
  cifar10_vit_geo_r: 10/10 seed indexes
  cifar10_vit_geo_m: 10/10 seed indexes
  stl10_r50_geo_r: 10/10 seed indexes
  stl10_r50_geo_m: 10/10 seed indexes
  stl10_vit_geo_r: 10/10 seed indexes
  stl10_vit_geo_m: 10/10 seed indexes

## MAE verification (rebuttal_mae_verify)
  cifar10_vitmae geo_r: orig=10/10 shuf=10/10
  stl10_vitmae geo_r: orig=10/10 shuf=10/10

## Baselines (BASE/{ds}_{model}_{mode})
  (only incomplete entries listed; absence = all 96 combos complete)