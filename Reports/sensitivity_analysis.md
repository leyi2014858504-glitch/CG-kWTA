# Isolated sensitivity checks (Reviewer #4.2)

Reviewer request: *"The paper should test sensitivity to coordinate dimension,
calibration split, random-projection seed, distance, and selector."*

Dimension, distance and selector are covered elsewhere
(`Reports/d_sensitivity.md`, `Reports/geometry_ablation_summary.csv`).
This note covers the two remaining nuisance parameters.

## Design

The **data seed is held fixed at 0** in every variant, so the train/val split,
the CMA-ES trajectory (same seed, same budget), the shuffle permutation
(`shuffle_seed_offset = 0`, so the permutation is identical in all variants) and
the final refit are unchanged. Only the knob under test varies:

| Knob | What varies | What is held fixed |
|---|---|---|
| `CALIB_SEED` | which `cma_train_n` = 4000 rows of the training set are used to build the coordinates | everything else |
| `PROJ_SEED_OVERRIDE` | the RandProj projection matrix (`geo_r` only) | everything else |

5 values per knob (0..4), both arms (original and shuffled) for each, so a shuffle
gap can be recomputed for every variant. 50 runs total. The spread reported below
is **across variants of the knob**, not across seeds.

## Result

| Knob | Configuration | low-k gap (mean ± sd) | gap @ k=0.05 (mean ± sd) | orig @ k=0.05 (mean ± sd) |
|---|---|---|---|---|
| calibration split | cifar10 / r50 / geo_m | **+0.0478 ± 0.0034** | +0.1050 ± 0.0093 | 0.8479 ± 0.0040 |
| calibration split | imagenet100 / r50 / geo_m | **+0.2482 ± 0.0051** | +0.4829 ± 0.0151 | 0.8211 ± 0.0070 |
| calibration split | imagenet100 / r50 / geo_r | **+0.1411 ± 0.0178** | +0.2694 ± 0.0392 | 0.5950 ± 0.0413 |
| projection seed | cifar10 / r50 / geo_r | **+0.0222 ± 0.0081** | +0.0490 ± 0.0117 | 0.7921 ± 0.0101 |
| projection seed | imagenet100 / r50 / geo_r | **+0.1515 ± 0.0220** | +0.2727 ± 0.0311 | 0.6124 ± 0.0265 |

## Reading

1. **No sign flips anywhere.** All 50 variants produce a positive low-k gap.
   For the headline configuration (imagenet100 / r50 / geo_m) the isolated spread
   is +0.2482 ± 0.0051, i.e. a coefficient of variation of ~2%; the value also
   coincides with the multi-seed (+0.247) and multi-permutation (+0.248) estimates.
2. **Calibration split matters less than the projection seed, in relative terms.**
   geo_m (which averages over the calibration set through mean/std/rate) is
   insensitive by construction: CV 2% at ImageNet scale, 7% on CIFAR-10.
   geo_r is more exposed (CV 13–15% at scale), because a random projection can
   land close to or far from the discriminative directions.
3. **The largest relative sensitivity is cifar10 / r50 / geo_r under the
   projection seed (CV ~36%), where the effect itself is smallest (+0.022).**
   The absolute spread is only ±0.008 and the sign is preserved; one variant
   (projection 2) reduces the all-k mean gap to ≈0 (−0.0019) while its low-k gap
   stays positive (+0.0083), consistent with the known high-k behaviour of geo_r.
4. **Absolute accuracy is stable too** — original test accuracy at k=0.05 moves
   by 0.4–4.1 points across variants, i.e. the coordinate construction itself is
   not being destabilised by either knob.

## Conclusion for the response letter

Conclusions are invariant to the calibration split and to the random-projection
seed: no variant reverses the sign of the shuffle gap, and the spread introduced
by these two arbitrary choices is small relative to the effect (and, for geo_m,
comparable to the between-seed spread already reported). Sensitivity is largest
for the weakest configuration (CIFAR-10 / geo_r), where the effect is smallest to
begin with.

## Reproducing

```bash
python run_sensitivity_checks.py          # sweep (isolated; seed fixed at 0)
python sensitivity_report.py              # this table
```

Raw per-variant values: `Reports/sensitivity_variants.csv`.
Aggregated: `Reports/sensitivity_summary.csv`.
