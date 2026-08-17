# Deep Learning for Extreme Class-Imbalanced Credit Card Fraud Detection

Studies how deep learning models behave under extreme class imbalance (~0.17% positive
rate), using the ULB Credit Card Fraud dataset.


## Approach

- **Model:** Dense layers → self-attention block → fraud probability; trained using **BCE, class-weighted BCE, and Focal Loss** variants
- **Baselines:** Plain MLP + BCE and MLP + class-weighted BCE
- **Experiment axis:** Train each architecture/loss combination at several imbalance ratios (**1:1, 1:10, 1:100, full ~1:577**) via controlled undersampling, to isolate the effects of imbalance ratio, loss function, and architecture
- **Metrics:** PR-AUC, F1, F2, recall @ fixed precision, and MCC — **not plain accuracy** as the primary metric, given the extreme class imbalance
- **Interpretability:** SHAP analysis on **true positives, false positives, and false negatives** from the test set to characterize where the model succeeds and fails
- 
## Setup

```bash
python -m venv venv
source venv/bin/activate        
pip install -r requirements.txt
```

## Dataset

1. `creditcard.csv` from the ULB Credit Card Fraud dataset on Kaggle.
   
## Repo structure

```
fraud-detection-imbalance/                
├── notebooks/
│   └── 01_eda.ipynb      
├── src/
│   ├── data_prep.py       # load, scale, split, build imbalance-ratio subsets
│   ├── models.py          # MLP, MLP+Attention architectures
│   ├── losses.py          # Focal Loss implementation
│   ├── train.py           # training loop, CLI
│   ├── evaluate.py        # metrics + confusion matrix + PR curve
│   └── explain.py         # SHAP analysis on TP/FP/FN
├── results/
│   ├── metrics_table.csv
│   └── figures/
├── requirements.txt
├── .gitignore
└── README.md
```

## Results

Full sweep: 2 architectures (`mlp`, `mlp_attention`) × 3 losses (`bce`, `weighted_bce`,
`focal`) × 4 imbalance ratios (`1to1`, `1to10`, `1to100`, `full`) = 24 runs, evaluated
on a held-out test set fixed at the natural imbalance ratio (74 fraud / ~42,800
legitimate transactions), never resampled.

### Best model

**`mlp_attention` + `bce` + `full` ratio** — PR-AUC 0.839, F1 0.837, precision 0.881,
recall 0.797, MCC 0.838. The attention-augmented MLP trained on the natural imbalance
ratio with plain binary cross-entropy outperformed every other configuration on PR-AUC,
including all focal-loss variants.

### Key findings

1. **ROC-AUC is not a useful metric at this imbalance level.** Every one of the 24 runs
   scores between 0.954–0.983 ROC-AUC regardless of quality — a spread of under 3
   points. PR-AUC spans 0.58–0.84, a real 26-point spread that actually distinguishes
   good models from bad ones. Any evaluation of this dataset that leads with ROC-AUC or
   accuracy is not measuring what matters.

2. **Focal loss does not uniformly outperform plain BCE.** At the natural imbalance
   ratio, `mlp_bce_full` (F1 0.844) and `mlp_attention_bce_full` (F1 0.837) both beat
   their focal-loss counterparts (`mlp_focal_full` F1 0.753, `mlp_attention_focal_full`
   F1 0.837 — tied only after attention is added). Focal loss's advantage shows up more
   clearly at extreme undersampling ratios (e.g., `mlp_attention_focal_1to100` reaches
   0.794 PR-AUC vs `mlp_attention_weighted_bce_1to100` at 0.774), suggesting its
   down-weighting of easy examples matters most when the training set is already
   artificially balanced, not when it's left at its natural extreme skew.

3. **Naive class-weighting (`weighted_bce`) overcorrects.** Using the standard heuristic
   `pos_weight = n_negative / n_positive` (≈577 at the full ratio) produces recall as
   high as 0.905 but precision as low as 0.056 — the model floods almost every
   borderline transaction with a fraud flag. This is a genuine failure mode, not just an
   underperforming baseline: naive cost-sensitive weighting needs threshold calibration
   or a capped weight to be usable, and simply maximizing recall via loss weighting is
   not sufficient on its own.

4. **Self-attention gives a modest, consistent PR-AUC lift over the plain MLP** when
   comparing matched loss/ratio pairs — e.g., BCE at the full ratio: 0.839 (attention)
   vs 0.831 (plain); focal at 1:100: 0.794 (attention) vs 0.755 (plain). The gain is
   real but not dramatic, which is a more credible finding than claiming attention
   transforms performance.

5. **Undersampling trades recall for precision predictably.** For plain MLP + BCE,
   moving from a 1:1 to the natural ~1:577 ratio raises precision from 0.080 to 0.934
   while recall falls from 0.865 to 0.770 — the classic imbalance-ratio trade-off,
   cleanly isolated here because the test set never changes.

See `results/metrics_table.csv` for the full 24-row table and `results/figures/grids/`
for the corresponding confusion-matrix and SHAP visualizations.

### SHAP failure analysis 
(best model: `mlp_attention_bce_full`)

`V14`, `V10`, `V12`, `V4`, and `V17` are consistently the most influential features
across true positives, false positives, and false negatives — consistent with prior
published analyses of this dataset. The interesting difference is in magnitude: SHAP
values for true positives reach as high as ±0.6, while false negatives rarely exceed
±0.2. This suggests the 15 missed frauds are not cases the model gets confidently
wrong — they sit closer to the legitimate-transaction distribution on the features that
matter most, i.e., genuine class overlap that loss reweighting alone cannot resolve.


