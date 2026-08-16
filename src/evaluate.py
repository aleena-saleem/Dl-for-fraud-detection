import argparse
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score, f1_score, fbeta_score, recall_score,
    precision_score, matthews_corrcoef, confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt

from models import build_model

DATA_DIR = os.path.join("data", "processed")
CKPT_DIR = os.path.join("results", "checkpoints")
FIG_DIR = os.path.join("results", "figures")
METRICS_CSV = os.path.join("results", "metrics_table.csv")


def load_test():

    X = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv")).values.astype(np.float32)
    y = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel().astype(np.float32)
    return X, y


def recall_at_precision(y_true, probs, target_precision=0.90):
    
    thresholds = np.unique(probs)

    best_recall = 0.0

    for t in thresholds:

        preds = (probs >= t).astype(int)

        if preds.sum() == 0:

            continue

        p = precision_score(y_true, preds, zero_division=0)

        if p >= target_precision:

            r = recall_score(y_true, preds, zero_division=0)

            best_recall = max(best_recall, r)

    return best_recall


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", choices=["mlp", "mlp_attention"], default="mlp_attention")

    parser.add_argument("--loss", choices=["bce", "weighted_bce", "focal"], default="focal")

    parser.add_argument("--ratio", choices=["1to1", "1to10", "1to100", "full"], default="full")
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)

    run_name = f"{args.model}_{args.loss}_{args.ratio}"

    ckpt_path = os.path.join(CKPT_DIR, f"{run_name}.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device)

    model = build_model(args.model, input_dim=ckpt["input_dim"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    X_test, y_test = load_test()

    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "run": run_name,
        "model": args.model,
        "loss": args.loss,
        "ratio": args.ratio,
        "pr_auc": average_precision_score(y_test, probs),
        "roc_auc": roc_auc_score(y_test, probs),
        "f1": f1_score(y_test, preds, zero_division=0),
        "f2": fbeta_score(y_test, preds, beta=2, zero_division=0),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "mcc": matthews_corrcoef(y_test, preds),
        "recall_at_90pct_precision": recall_at_precision(y_test, probs, 0.90),
    }

    print(f"\n=== Results: {run_name} ===")

    for k, v in metrics.items():

        if isinstance(v, float):
            print(f"  {k:28s}: {v:.4f}")

    
    cm = confusion_matrix(y_test, preds)

    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(4.2, 4.2))

    ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)

    for i in range(2):

        for j in range(2):

            ax.text(j, i - 0.08, f"{cm_pct[i, j]:.1f}%", ha="center", va="center",
                     fontsize=13, fontweight="bold")
            ax.text(j, i + 0.18, f"(n={cm[i, j]})", ha="center", va="center",
                     fontsize=8, color="dimgray")
            
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Legit", "Fraud"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Legit", "Fraud"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(run_name, fontsize=10) 
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{run_name}_confusion.png"), dpi=150)
    plt.close(fig)

    
    row = pd.DataFrame([metrics])
    if os.path.exists(METRICS_CSV):
        existing = pd.read_csv(METRICS_CSV)
        existing = existing[existing["run"] != run_name]  # replace if re-run
        combined = pd.concat([existing, row], ignore_index=True)
    else:
        combined = row
    combined.to_csv(METRICS_CSV, index=False)
    print(f"\nAppended to {METRICS_CSV}")


if __name__ == "__main__":
    main()
