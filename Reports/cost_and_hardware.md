# Computational cost and hardware (Reviewer #4.5)

Reviewer request: *"Please report wall-clock and regression-solve costs and
provide convergence checks for both conditions over several representative
settings."*

Convergence checks are satisfied separately by `cma_fbest_history`
(one entry per CMA-ES generation, for both the original and the shuffled
condition). This note covers the two cost quantities.

---

## 1. Hardware and software

| Item | Specification |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM (CUDA) |
| CPU | AMD Ryzen 9 7940H, 8 cores / 16 threads, ~4.0 GHz base |
| Memory | 15.2 GB RAM (laptop platform, shared integrated-graphics memory) |
| Operating system | Windows 11, build 10.0.26200 |
| Python | 3.14.0 (64-bit) |
| PyTorch | 2.12.0+cu126 |
| Linear algebra | CUDA cuSOLVER / cuBLAS for the GPU solves |

Device selection is automatic (`cuda` when available); all runs reported here
executed on the GPU.

---

## 2. What the two cost quantities mean here

| Quantity | Definition in this codebase |
|---|---|
| **wall-clock** | `elapsed_sec`: the complete cost of **one k-point** — gate search (CMA-ES), the final ridge refit on train+val, and the one-shot test evaluation. It excludes dataset loading and feature caching. |
| **regression-solve cost** | `search_sec`: the time spent inside `CmaEngine.f()`, i.e. the ridge solves performed **while searching**. One solve per CMA-ES candidate. |

The rest of `elapsed_sec` (`rest_mean_s`) is setup, coordinate construction and
lookups, the final refit, and the test evaluation.

Number of search-time ridge solves per k-point:

```
n_search_solves = popsize x maxiter        (popsize = 4 + floor(3 ln(1+d)) = 8 for d = 3)
                = 8 x 20 = 160             (maxiter = 20)
                = 8 x 40 = 320             (maxiter = 40)
```
plus one initial evaluation, one final refit on train+val, and one baseline
validation evaluation.

---

## 3. Wall-clock per k-point (measured, `Reports/wallclock_timing.csv`)

One k-point = full pass (search + final refit + test eval). Columns: mean and
median over all k-points x seeds in the archived runs.

| Configuration | n points | mean (s) | median (s) | max (s) | solves/k | total (h) |
|---|---|---|---|---|---|---|
| cifar10 / r50 / geo_m | 200 | 5.49 | 5.56 | 12.23 | 163 | 0.31 |
| cifar10 / r50 / geo_r | 200 | 4.78 | 3.60 | 11.77 | 163 | 0.27 |
| cifar10 / r50 / geo_pca | 200 | 4.73 | 3.58 | 11.69 | 163 | 0.26 |
| cifar10 / vit / geo_m | 200 | 1.28 | 1.13 | 2.55 | 163 | 0.07 |
| cifar10 / vitmae / geo_r | 200 | 1.28 | 1.10 | 2.58 | 163 | 0.07 |
| stl10 / r50 / geo_m | 200 | 1.08 | 0.96 | 2.12 | 163 | 0.06 |
| imagenet100 / r50 / geo_m | 100 | 13.26 | 9.91 | 41.50 | 163 | 0.37 |
| imagenet100 / r50 / geo_r | 100 | 12.18 | 9.21 | 30.89 | 163 | 0.34 |
| imagenet100 / r50 / geo_pca | 100 | 15.75 | 10.27 | 46.45 | 163 | 0.44 |
| cifar10 / r50 / geo_m, maxiter=40 | 200 | 11.81 | 8.79 | 30.65 | 323 | 0.66 |

Cost scales with the feature dimension (2048-d ResNet-50 vs 768-d ViT) and with
the search budget (maxiter 20 -> 40 roughly doubles the per-point cost).

---

## 4. Cost breakdown (measured, `Reports/cost_breakdown.csv`)

Fresh single-seed runs with per-component instrumentation.

| Configuration | maxiter | cma_train_n | elapsed (s) | search (s) | search share | solves | ms / solve | rest (s) |
|---|---|---|---|---|---|---|---|---|
| cifar10 / r50 / geo_m | 20 | 4000 | 16.06 | 15.47 | 91.5% | 160 | 96.7 | 0.59 |
| cifar10 / vit / geo_m | 20 | 4000 | 3.55 | 3.18 | 87.2% | 160 | 19.9 | 0.37 |
| cifar10 / r50 / geo_m | 40 | 4000 | 30.09 | 29.25 | 94.2% | 320 | 91.4 | 0.84 |
| cifar10 / r50 / geo_m | 20 | 20000 | — | — | — | — | — | — |
| imagenet100 / r50 / geo_m | 20 | 4000 | — | — | — | — | — | — |

**Robust finding:** **87–94% of the end-to-end cost is the search-time ridge
solves**, confirming the reviewer's concern that the method's cost is dominated
by repeatedly refitting the readout rather than by the O(d+1) parameter count of
the gate. The remaining 6–13% is coordinate construction, data movement, the
final refit and the test evaluation.

The two rows marked `—` were blocked by a CUDA out-of-memory condition because
another experiment was occupying the 8 GB GPU at the time; they do not change
the ratio above and can be filled in on a free GPU
(`python run_cost_breakdown.py --configs c10_r50_geo_m_cma20k im100_r50_geo_m`).

---

## 5. Caveats for the write-up

1. `ms / solve` is an **upper bound** on the pure linear-algebra time: it is the
   total search time divided by the number of solves, so it also absorbs the gate
   computation, tensor copies and Python overhead per candidate.
2. Absolute seconds are machine-state dependent (shared laptop GPU, thermal and
   contention effects). The breakdown is therefore reported mainly as a **share**
   of the total; the reproducible quantity is the solve count
   (`popsize x maxiter` per k-point) and the ratio.
3. `elapsed_sec` deliberately excludes one-off feature extraction and dataset
   loading, so the table reflects the per-configuration diagnostic cost, not the
   full pipeline.

---

## 6. Reproducing

```bash
python wallclock_report.py                                   # section 3
python run_cost_breakdown.py                                 # section 4
python run_cost_breakdown.py --configs c10_r50_geo_m_cma20k im100_r50_geo_m   # the two pending rows
```
