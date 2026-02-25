[README.md](https://github.com/user-attachments/files/25547571/README.md)
# # Shuffle Counterfactuals with O(1)-in-H Geometric Masks

This repository contains the code for our TMLR submission on
geometric global top‑k masking and shuffle‑based alignment probes.
We implement Geo‑kWTA, a constant‑parameter geometric family of
global top‑k masks, and the shuffle counterfactual protocol used in
the experiments.

In the geo_r mode with the default experimental budget used in the paper, 
a single main sweep takes approximately 9–10 minutes on an RTX 4060 Laptop GPU.

## 1. Environment Setup

Install the necessary dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## 2. Data Preparation (Feature Extraction)

Before running experiments, you must extract features from pre-trained backbones (ResNet-50, ViT-MAE, etc.).

- **ResNet-50 Features**:
  - `extract_resnet50_features.py`: Extract CIFAR-10 features.
  - `extract_resnet50_features_stage1-4.py`: Extract layer-wise features for CIFAR-10.
  - `extract_resnet50_features_cifar100_stage1-4.py`: Extract layer-wise features for CIFAR-100.
- **ViT Features**:
  - `extract_mae_features.py`: Extract features from pre-trained ViT-MAE.
  - `extract_augreg_features.py`: Extract features from ViT-AugReg.

## 3. Running Experiments

The main experiment logic is implemented in `sphere_kwta_dimension_resnet50_MAE.py`.

### Experiment Modes:
Modify the `EXP_MODE` constant at the bottom of the script (line 1489) to select the experiment type:
- `"geo_r"`: Main Geometric k-WTA method with random projection initialization.
- `"random"`: Unstructured random masking baseline.
- `"bestof_random"`：Unstructured random masking with N=176 budget.
- `"kwta"`: Standard k-WTA baseline.
- `"semantic_test"`: Runs the synthetic semantic tests described in Appendix A.1 .

- To reproduce the main CIFAR‑10 ResNet‑50 results (Fig. 2 in the paper),
first run `sphere_kwta_dimension_resnet50_MAE.py` with `EXPMODE="geo_r"`,
`"random"`, `"kwta"`, and `"bestofrandom"` on `DATASET="cifar10r50layer4"`,
then call `cifar10_results_r50/plot_4modes_ci95.py`.

### Hyperparameter Tuning:
Key hyperparameters mentioned in the paper can be adjusted in `sphere_kwta_dimension_resnet50_MAE.py`:
- **CMA-ES Step Size (`sigma0`)**: Adjusted on line 33.
- **Latent Space Dimension (`pd`)**: Adjusted on line 36.
- **Scoring Mode (`GEOSCORE_MODE`)**: Set to `"center"` (Euclidean) or `"shell"` (Sphere surface) on line 41.
- **Optimization Budget (`MAXITER`)**: Set on line 1494.
- **Dataset/Layer Selection (`DATASET`)**: Choose the target feature set on line 1495.

## 4. Evaluation and Plotting

### Significance and Sensitivity Analysis:
- `augreg_sig/sig_on_augreg.py`: Significance tests for AugReg models.
- `budget_test_maxiter/budget_test.py`: Sensitivity to CMA-ES iteration budget.
- `budget_test_sigm0/budget_test_sigma.py`: Sensitivity to the initial step size `sigma0`.

### Robustness Evaluation:
- `eval_cifar10c_multi_seed_topk_geoshell_ball.py`: Multi-seed evaluation on CIFAR-10C (Corrupted data).

### Plotting Figures:
Plotting scripts are organized by experiment type. Run them to generate the figures used in the paper:
- `cifar10_results_r50/plot_4modes_ci95.py`: Generates the main comparison plots (Accuracy vs. Sparsity).
- `stage1-3/bar_kfrac_layers.py`: Generates layer-wise sparsity analysis plots.
- `stage4_b1-b2/bar_kfrac_blocks.py`: Generates block-wise sparsity analysis plots.
- `shuffle_comparison_cifar10/plot_geo_vs_shuffle_ci95_kmax04.py`: Generates plots comparing geometric selection with random shuffling.


## License

This project is licensed under the MIT License – see the `LICENSE` file for details.
