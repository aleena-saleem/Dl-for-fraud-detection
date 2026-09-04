# Deep Learning for Extreme Class-Imbalanced Credit Card Fraud Detection

An empirical study of how deep learning models behave under extreme class imbalance
(~0.17% positive rate), using the ULB Credit Card Fraud dataset. Rather than reporting
a single "best" model, the goal here is to isolate how three factors — **loss
function**, **imbalance ratio in training**, and **architecture** — independently
affect fraud detection performance, and where each one breaks down.

## Approach

- **Model:** dense layers → self-attention block → fraud probability, trained with
  **BCE**, **class-weighted BCE**, and **Focal Loss**
- **Baseline:** plain MLP with the same three losses
- **Experiment axis:** each architecture/loss pair is trained at four imbalance ratios
  — **1:1, 1:10, 1:100, and the natural ~1:577** — via controlled undersampling, to
  isolate the effect of imbalance ratio independently of loss choice and architecture
- **Metrics:** PR-AUC, F1, MCC, and recall @ fixed precision — **not accuracy or
  ROC-AUC**, both of which are close to uninformative at this level of imbalance
- **Validation:** 7-fold cross-validation for every one of the 24 configurations, plus
  a single held-out test evaluation, so results are reported as CV mean ± std rather
  than a single number
- **Interpretability:** SHAP analysis on true positives, false positives, and false
  negatives to characterize *where* the model succeeds and fails, not just how often

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset

`creditcard.csv` from the ULB Credit Card Fraud dataset on Kaggle.

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
│   ├── cv_results.csv         # 7-fold CV aggregates, all 24 configs
│   ├── metrics_table.csv      # single held-out-run metrics + CV aggregates
│   └── figures/
├── requirements.txt
├── .gitignore
└── README.md
```

## Results

Full sweep: 2 architectures (`mlp`, `mlp_attention`) × 3 losses (`bce`, `weighted_bce`,
`focal`) × 4 imbalance ratios (`1to1`, `1to10`, `1to100`, `full`) = **24
configurations**, each evaluated with 7-fold CV and a fixed held-out test set (74 fraud
/ ~42,800 legitimate transactions, never resampled).

### Best-performing configuration

**Plain MLP + BCE, full imbalance ratio** — PR-AUC 0.837 ± 0.041 (CV mean ± std), F1
0.823 ± 0.034, MCC 0.825 ± 0.033. `mlp_attention_bce_full` is a close second (PR-AUC
0.832, F1 0.819, MCC 0.819) and edges ahead on the single held-out test split for F1,
MCC, and recall. No config wins on every metric.

### Key findings

1. **PR-AUC over ROC-AUC.** CV-mean PR-AUC ranges from **0.558** (`mlp_focal_1to1`) to
   **0.837** (`mlp_bce_full`) — a real 28-point spread. Accuracy/ROC-AUC would hide
   almost all of it.

2. **Attention's lift is inconsistent.** At BCE + full ratio, plain MLP actually beats
   attention (0.837 vs. 0.832 PR-AUC). At BCE + 1:100, attention wins clearly (0.826 vs.
   0.761, test set). It shifts the precision/recall balance, not a guaranteed +PR-AUC.

3. **Focal loss underperforms, badly, at the full ratio.** `mlp_focal_full`: F1 0.096
   (CV mean). `mlp_attention_focal_full`: F1 0.104. Both far below BCE's 0.823. Focal's
   easy-example down-weighting backfires when training data already reflects the real
   skew.

4. **Weighted BCE overcorrects.** Precision drops to **0.068** at 1:10 (recall 0.865);
   recovers only to 0.432 at the full ratio — still well below BCE's 0.892 at similar
   recall. Needs threshold calibration to be usable.

5. **Undersampling trades recall for precision.** MLP + BCE, 1:1 → full: precision
   **0.082 → 0.892**, recall **0.878 → 0.784**. Test set held fixed throughout.

See [`results/cv_results.csv`](results/cv_results.csv) for the full 24-row
cross-validated table, [`results/metrics_table.csv`](results/metrics_table.csv) for
per-run test-set metrics alongside the same CV aggregates, and
`results/figures/grids/` for the corresponding confusion-matrix and SHAP
visualizations.

### Figures

![PR-AUC across imbalance ratio, by loss and architecture](results/figures/pr_auc_by_ratio.png)

*CV-mean PR-AUC (± std) as training imbalance ratio moves from 1:1 to the natural
~1:577, split by loss function and architecture. BCE is the only loss that keeps
improving toward the full ratio; focal loss and weighted BCE plateau or decline.*

![Precision/recall tradeoff for undersampling](results/figures/precision_recall_tradeoff_mlp_bce.png)

*MLP + BCE: precision climbs from 0.08 to 0.89 as the ratio moves toward the natural
imbalance, while recall drifts down from 0.88 to 0.78 — the tradeoff described in
finding 5.*

![Weighted BCE overcorrection](results/figures/weighted_bce_overcorrection.png)

*MLP + weighted BCE: recall stays roughly flat (~0.85–0.87) across every ratio, but
precision swings wildly — the class-weighting term dominates the loss regardless of
how much real imbalance the model sees during training.*

### SHAP failure analysis (best model: `mlp_bce_full`)

`V14`, `V10`, `V12`, `V4`, and `V17` are consistently the most influential features
across true positives, false positives, and false negatives — consistent with prior
published analyses of this dataset. The interesting difference is in magnitude: SHAP
values for true positives reach substantially higher magnitudes than for false
negatives, which cluster much closer to zero. This suggests the missed fraud cases
aren't ones the model gets confidently wrong — they sit closer to the legitimate
transaction distribution on the features that matter most, i.e., genuine class overlap
that loss reweighting alone cannot resolve.

## Limitations

- The held-out test set contains only **74 fraud cases**. Single-run test metrics
  swing noticeably between configurations for this reason, which is exactly why CV
  mean ± std is reported as the primary number throughout rather than a single split.
- Splits are random, not time-based. Since this is transaction data, a time-ordered
  split (train on earlier transactions, test on later ones) would be a more realistic
  simulation of deployment and is a natural next step.
- All four imbalance ratios are built by undersampling the majority class; no
  oversampling (SMOTE, ADASYN, etc.) or hybrid resampling is tested here.
- Thresholds are left at the default 0.5 cutoff throughout. Several findings above
  (particularly the weighted-BCE precision collapse) would likely look different under
  a calibrated decision threshold — that comparison is left for future work.

## Acknowledgments

Dataset: Machine Learning Group, Université Libre de Bruxelles (ULB) — *Credit Card
Fraud Detection*, available on Kaggle.
