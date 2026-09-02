"""
Stratified k-fold cross-validation for all (model × loss × ratio) combinations.

"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from models import build_model
from losses import build_loss

DATA_DIR = os.path.join("data", "processed")
RESULTS_DIR = os.path.join("results")
CV_CSV = os.path.join(RESULTS_DIR, "cv_results.csv")


RANDOM_STATE = 42
N_FOLDS = 7          # ~24 fraud cases per test fold; change via --folds
EPOCHS = 20
BATCH_SIZE = 256
LR = 1e-3

IMBALANCE_RATIOS = {
    "1to1": 1,
    "1to10": 10,
    "1to100": 100,
    "full": None,
}



def load_trainval() -> tuple[np.ndarray, np.ndarray]:
    """
    Load and concatenate the train (full) + val splits.

    The val split was carved out of the original train pool with the same
    stratification, so concatenating them gives back the full 85 % pre-test
    pool with the correct class proportions.
    """
    X_train = pd.read_csv(
        os.path.join(DATA_DIR, "X_train_full.csv")
    ).values.astype(np.float32)
    y_train = pd.read_csv(
        os.path.join(DATA_DIR, "y_train_full.csv")
    ).values.ravel().astype(np.float32)

    X_val = pd.read_csv(
        os.path.join(DATA_DIR, "X_val.csv")
    ).values.astype(np.float32)
    y_val = pd.read_csv(
        os.path.join(DATA_DIR, "y_val.csv")
    ).values.ravel().astype(np.float32)

    X = np.concatenate([X_train, X_val], axis=0)
    y = np.concatenate([y_train, y_val], axis=0)
    return X, y


def apply_ratio_subset(
    X: np.ndarray, y: np.ndarray, ratio_multiplier
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mirror of data_prep.build_ratio_subset but operating on numpy arrays.

    Keeps all fraud rows; randomly subsamples legit rows to achieve the
    requested fraud-to-legit ratio multiplier (None → keep all legit).
    """
    fraud_mask = y == 1
    legit_mask = ~fraud_mask

    fraud_idx = np.where(fraud_mask)[0]
    legit_idx = np.where(legit_mask)[0]

    n_fraud = len(fraud_idx)

    if ratio_multiplier is None:
        keep_legit = legit_idx
    else:
        n_keep = min(len(legit_idx), n_fraud * ratio_multiplier)
        rng = np.random.RandomState(RANDOM_STATE)
        keep_legit = rng.choice(legit_idx, size=n_keep, replace=False)

    keep_idx = np.concatenate([fraud_idx, keep_legit])
    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(keep_idx)

    return X[keep_idx], y[keep_idx]


def recall_at_precision(
    y_true: np.ndarray, probs: np.ndarray,
    target_precision: float = 0.90
) -> float:
    """
    Max recall at any threshold where precision >= target_precision.

    Uses sklearn's precision_recall_curve for a single vectorised pass
    instead of iterating over every unique threshold.
    """
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, _ = precision_recall_curve(y_true, probs)
    mask = precisions >= target_precision
    if not mask.any():
        return 0.0
    return float(recalls[mask].max())


def score_fold(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """Compute all headline metrics for a single fold."""
    preds = (probs >= 0.5).astype(int)
    return {
        "pr_auc": average_precision_score(y_true, probs),
        "f1": f1_score(y_true, preds, zero_division=0),
        "mcc": matthews_corrcoef(y_true, preds),
        "recall_at_90pct_prec": recall_at_precision(y_true, probs, 0.90),
    }



def train_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    model_name: str,
    loss_name: str,
    device: torch.device,
) -> torch.nn.Module:
    """Train a fresh model on one fold's training split, return it."""
    model = build_model(model_name, input_dim=X_tr.shape[1]).to(device)
    criterion = build_loss(loss_name, y_tr)
    if hasattr(criterion, "to"):
        criterion = criterion.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=1e-5
    )

    ds = TensorDataset(
        torch.from_numpy(X_tr),
        torch.from_numpy(y_tr),
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    return model


def infer_fold(
    model: torch.nn.Module,
    X_te: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Run inference and return sigmoid probabilities."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_te).to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs




def run_cv(
    model_name: str,
    loss_name: str,
    ratio_name: str,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    n_folds: int,
    device: torch.device,
    verbose: bool = True,
) -> dict:
    """
    Full stratified k-fold CV for one (model, loss, ratio) configuration.

    Returns a dict with mean and std for each metric, plus per-fold raw scores
    (useful for debugging / plotting).
    """
    ratio_multiplier = IMBALANCE_RATIOS[ratio_name]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                          random_state=RANDOM_STATE)

    fold_scores: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        skf.split(X_pool, y_pool), start=1
    ):
        X_tr_raw, y_tr_raw = X_pool[train_idx], y_pool[train_idx]
        X_te, y_te = X_pool[test_idx], y_pool[test_idx]

        # Apply ratio subsampling to the *training* portion of this fold
        X_tr, y_tr = apply_ratio_subset(X_tr_raw, y_tr_raw, ratio_multiplier)

        n_pos_tr = int(y_tr.sum())
        n_pos_te = int(y_te.sum())

        if verbose:
            print(
                f"  fold {fold_idx}/{n_folds} | "
                f"train={len(y_tr):,} (fraud={n_pos_tr}) | "
                f"test={len(y_te):,} (fraud={n_pos_te})"
            )

        trained_model = train_fold(X_tr, y_tr, model_name, loss_name, device)
        probs = infer_fold(trained_model, X_te, device)
        scores = score_fold(y_te, probs)

        if verbose:
            print(
                f"         PR-AUC={scores['pr_auc']:.4f}  "
                f"F1={scores['f1']:.4f}  "
                f"MCC={scores['mcc']:.4f}  "
                f"R@90P={scores['recall_at_90pct_prec']:.4f}"
            )

        fold_scores.append(scores)

    # Aggregate across folds
    metric_keys = list(fold_scores[0].keys())
    result = {
        "model": model_name,
        "loss": loss_name,
        "ratio": ratio_name,
        "n_folds": n_folds,
    }
    for k in metric_keys:
        vals = np.array([fs[k] for fs in fold_scores])
        result[f"{k}_mean"] = float(vals.mean())
        result[f"{k}_std"] = float(vals.std(ddof=1))  # sample std

    # Store per-fold breakdown for debugging
    result["_per_fold"] = fold_scores

    return result


METRICS_CSV = os.path.join(RESULTS_DIR, "metrics_table.csv")

# The four headline CV metrics that get merged in
CV_METRIC_KEYS = ["pr_auc", "f1", "mcc", "recall_at_90pct_prec"]


def merge_cv_into_metrics_table(df_cv: pd.DataFrame) -> None:
    """
    Join CV mean ± std columns into metrics_table.csv.

    For each row in metrics_table.csv that matches a (model, loss, ratio)
    triple in df_cv, the following columns are added / updated:

        cv_<metric>_mean   – mean across folds
        cv_<metric>_std    – sample std across folds (ddof=1)
        cv_n_folds         – number of folds used

    for metric in {pr_auc, f1, mcc, recall_at_90pct_prec}.

    Rows in metrics_table.csv with no matching CV result are left unchanged
    (their cv_* columns are NaN).  Existing cv_* columns are overwritten.
    """
    if not os.path.exists(METRICS_CSV):
        print(f"  metrics_table.csv not found at {METRICS_CSV} — skipping merge.")
        return

    df_metrics = pd.read_csv(METRICS_CSV)

    # Build the CV lookup keyed on (model, loss, ratio)
    cv_cols = {"n_folds": "cv_n_folds"}
    for k in CV_METRIC_KEYS:
        cv_cols[f"{k}_mean"] = f"cv_{k}_mean"
        cv_cols[f"{k}_std"]  = f"cv_{k}_std"

    df_cv_slim = df_cv[["model", "loss", "ratio"] + list(cv_cols.keys())].copy()
    df_cv_slim = df_cv_slim.rename(columns=cv_cols)

    # Drop any stale cv_* columns already in metrics table so we get a
    # clean replace rather than duplicate-suffix columns from pandas merge
    stale = [c for c in df_metrics.columns if c.startswith("cv_")]
    if stale:
        df_metrics = df_metrics.drop(columns=stale)

    df_merged = df_metrics.merge(
        df_cv_slim, on=["model", "loss", "ratio"], how="left"
    )

    # Re-order: all original columns first, then cv_* columns grouped by metric
    orig_cols = list(df_metrics.columns)
    new_cv_cols = [c for c in df_merged.columns if c not in orig_cols]
    df_merged = df_merged[orig_cols + new_cv_cols]

    df_merged.to_csv(METRICS_CSV, index=False)

    n_matched = df_merged["cv_n_folds"].notna().sum()
    print(f"  Merged CV results for {n_matched}/{len(df_merged)} rows "
          f"into {METRICS_CSV}")




def main():
    parser = argparse.ArgumentParser(
        description="Stratified k-fold CV across model × loss × ratio."
    )
    parser.add_argument(
        "--model",
        choices=["mlp", "mlp_attention", "all"],
        default="all",
        help="Which model to evaluate (default: all)",
    )
    parser.add_argument(
        "--loss",
        choices=["bce", "weighted_bce", "focal", "all"],
        default="all",
        help="Which loss to evaluate (default: all)",
    )
    parser.add_argument(
        "--ratio",
        choices=["1to1", "1to10", "1to100", "full", "all"],
        default="all",
        help="Which ratio condition to evaluate (default: all)",
    )
    parser.add_argument(
        "--folds", type=int, default=N_FOLDS,
        help=f"Number of CV folds (default: {N_FOLDS})",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-fold output",
    )
    parser.add_argument(
        "--merge-only", action="store_true",
        help=(
            "Skip training entirely. Read the existing cv_results.csv and "
            "merge its CV columns into metrics_table.csv, then exit."
        ),
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.merge_only:
        if not os.path.exists(CV_CSV):
            print(f"ERROR: {CV_CSV} not found. Run CV first.")
            raise SystemExit(1)
        df_cv = pd.read_csv(CV_CSV)
        merge_cv_into_metrics_table(df_cv)
        print("Done.")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve 'all' expansions
    models = (
        ["mlp", "mlp_attention"] if args.model == "all" else [args.model]
    )
    losses = (
        ["bce", "weighted_bce", "focal"] if args.loss == "all" else [args.loss]
    )
    ratios = (
        ["1to1", "1to10", "1to100", "full"] if args.ratio == "all"
        else [args.ratio]
    )

    print(f"\nLoading train+val pool …")
    X_pool, y_pool = load_trainval()
    n_fraud = int(y_pool.sum())
    print(
        f"Pool: {len(y_pool):,} rows | {n_fraud} fraud "
        f"({100 * n_fraud / len(y_pool):.4f}%)"
    )
    print(
        f"Using {args.folds}-fold stratified CV → "
        f"~{n_fraud // args.folds} fraud cases per test fold\n"
    )

    all_results = []

    total = len(models) * len(losses) * len(ratios)
    done = 0

    for model_name in models:
        for loss_name in losses:
            for ratio_name in ratios:
                done += 1
                run_id = f"{model_name}_{loss_name}_{ratio_name}"
                print(
                    f"\n[{done}/{total}] === {run_id} "
                    f"({args.folds}-fold CV) ==="
                )

                result = run_cv(
                    model_name=model_name,
                    loss_name=loss_name,
                    ratio_name=ratio_name,
                    X_pool=X_pool,
                    y_pool=y_pool,
                    n_folds=args.folds,
                    device=device,
                    verbose=not args.quiet,
                )

                # Pretty-print summary
                print(f"\n  Summary for {run_id}:")
                metric_keys = ["pr_auc", "f1", "mcc", "recall_at_90pct_prec"]
                for k in metric_keys:
                    m = result[f"{k}_mean"]
                    s = result[f"{k}_std"]
                    print(f"    {k:28s}: {m:.4f} ± {s:.4f}")

                all_results.append(result)

    rows = []
    for r in all_results:
        row = {k: v for k, v in r.items() if k != "_per_fold"}
        rows.append(row)

    df_new = pd.DataFrame(rows)

    # Column order for readability
    id_cols = ["model", "loss", "ratio", "n_folds"]
    metric_cols = [c for c in df_new.columns if c not in id_cols]
    df_new = df_new[id_cols + metric_cols]

    # Merge with any previously saved CV results (different configs)
    if os.path.exists(CV_CSV):
        df_old = pd.read_csv(CV_CSV)
        # Remove rows being replaced
        key_cols = ["model", "loss", "ratio"]
        merge_key = df_new[key_cols].apply(tuple, axis=1)
        old_key = df_old[key_cols].apply(tuple, axis=1)
        df_old = df_old[~old_key.isin(merge_key)]
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(CV_CSV, index=False)
    print(f"\n\nCV results saved to {CV_CSV}")

    merge_cv_into_metrics_table(df_new)


    print("\n=== Cross-Validation Summary (mean ± std) ===\n")
    metric_keys = ["pr_auc", "f1", "mcc", "recall_at_90pct_prec"]
    header = f"{'model':<20} {'loss':<14} {'ratio':<8}"
    for k in metric_keys:
        header += f"  {k:>28}"
    print(header)
    print("-" * len(header))

    for r in all_results:
        line = f"{r['model']:<20} {r['loss']:<14} {r['ratio']:<8}"
        for k in metric_keys:
            cell = f"{r[f'{k}_mean']:.4f} ± {r[f'{k}_std']:.4f}"
            line += f"  {cell:>28}"
        print(line)

    # Optionally save per-fold detail as JSON for later inspection
    fold_detail_path = os.path.join(RESULTS_DIR, "cv_fold_detail.json")
    detail = [
        {
            "run": f"{r['model']}_{r['loss']}_{r['ratio']}",
            "folds": r["_per_fold"],
        }
        for r in all_results
    ]
    with open(fold_detail_path, "w") as fh:
        json.dump(detail, fh, indent=2)
    print(f"Per-fold detail saved to {fold_detail_path}")


if __name__ == "__main__":
    main()
