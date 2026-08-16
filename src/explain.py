

import argparse
import os
import numpy as np
import pandas as pd
import torch
import shap
import matplotlib.pyplot as plt

from models import build_model

DATA_DIR = os.path.join("data", "processed")
CKPT_DIR = os.path.join("results", "checkpoints")
FIG_DIR = os.path.join("results", "figures")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", choices=["mlp", "mlp_attention"], default="mlp_attention")

    parser.add_argument("--loss", choices=["bce", "weighted_bce", "focal"], default="focal")

    parser.add_argument("--ratio", choices=["1to1", "1to10", "1to100", "full"], default="full")
    parser.add_argument("--background_size", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)

    run_name = f"{args.model}_{args.loss}_{args.ratio}"

    ckpt_path = os.path.join(CKPT_DIR, f"{run_name}.pt")

    device = torch.device("cpu")  

    ckpt = torch.load(ckpt_path, map_location=device)

    model = build_model(args.model, input_dim=ckpt["input_dim"]).to(device)

    model.load_state_dict(ckpt["model_state"])

    model.eval()

    feature_names = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"), nrows=0).columns.tolist()

    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))

    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

    def predict_fn(x_numpy):

        with torch.no_grad():

            logits = model(torch.from_numpy(x_numpy.astype(np.float32)))

            return torch.sigmoid(logits).numpy()

    background = shap.sample(X_test, args.background_size, random_state=42)

    explainer = shap.KernelExplainer(predict_fn, background)

    with torch.no_grad():

        probs = predict_fn(X_test.values)

    preds = (probs >= 0.5).astype(int)

    tp_idx = np.where((preds == 1) & (y_test == 1))[0]
    fp_idx = np.where((preds == 1) & (y_test == 0))[0]
    fn_idx = np.where((preds == 0) & (y_test == 1))[0]

    print(f"TP={len(tp_idx)}  FP={len(fp_idx)}  FN={len(fn_idx)}")

    def explain_subset(idx, label, n=10):
        if len(idx) == 0:
           
            return
        sample_idx = idx[:n]
        subset = X_test.iloc[sample_idx]
        shap_values = explainer.shap_values(subset, nsamples=100)

        fig = plt.figure()
        shap.summary_plot(shap_values, subset, feature_names=feature_names, show=False)
        plt.title(f"SHAP summary — {label} ({run_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"{run_name}_shap_{label}.png"), dpi=150)
        plt.close(fig)
        print(f"Saved SHAP summary for {label} ({len(sample_idx)} examples)")

    explain_subset(tp_idx, "true_positives")
    explain_subset(fp_idx, "false_positives")
    explain_subset(fn_idx, "false_negatives")

    print(f"\nAll SHAP figures written to {FIG_DIR}/")


if __name__ == "__main__":
    main()
