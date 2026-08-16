

import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from models import build_model
from losses import build_loss

DATA_DIR = os.path.join("data", "processed")
CKPT_DIR = os.path.join("results", "checkpoints")


def load_split(name: str):
    X = pd.read_csv(os.path.join(DATA_DIR, f"X_train_{name}.csv")).values.astype(np.float32)
    y = pd.read_csv(os.path.join(DATA_DIR, f"y_train_{name}.csv")).values.ravel().astype(np.float32)
    return X, y


def load_val():
    X = pd.read_csv(os.path.join(DATA_DIR, "X_val.csv")).values.astype(np.float32)
    y = pd.read_csv(os.path.join(DATA_DIR, "y_val.csv")).values.ravel().astype(np.float32)
    return X, y


def evaluate_loader(model, loader, device):

    model.eval()

    all_logits, all_targets = [], []

    with torch.no_grad():

        for xb, yb in loader:

            xb = xb.to(device)

            logits = model(xb)

            all_logits.append(logits.cpu())

            all_targets.append(yb)
    logits = torch.cat(all_logits)

    targets = torch.cat(all_targets)

    probs = torch.sigmoid(logits)

    from sklearn.metrics import average_precision_score, f1_score

    preds = (probs >= 0.5).float()
    pr_auc = average_precision_score(targets.numpy(), probs.numpy())
    f1 = f1_score(targets.numpy(), preds.numpy(), zero_division=0)
    return pr_auc, f1


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("model", choices=["mlp", "mlp_attention"], default="mlp_attention")

    parser.add_argument("loss", choices=["bce", "weighted_bce", "focal"], default="focal")

    parser.add_argument("ratio", choices=["1to1", "1to10", "1to100", "full"], default="full")

    parser.add_argument("epochs", type=int, default=30)

    parser.add_argument("batch_size", type=int, default=256)

    parser.add_argument("lr", type=float, default=1e-3)

    args = parser.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    X_train, y_train = load_split(args.ratio)

    X_val, y_val = load_val()

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))

    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = build_model(args.model, input_dim=X_train.shape[1]).to(device)

    criterion = build_loss(args.loss, y_train)

    if hasattr(criterion, "to"):

        criterion = criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_pr_auc = -1.0

    run_name = f"{args.model}_{args.loss}_{args.ratio}"

    ckpt_path = os.path.join(CKPT_DIR, f"{run_name}.pt")
    history = []

    for epoch in range(1, args.epochs + 1):

        model.train()

        epoch_loss = 0.0

        for xb, yb in train_loader:

            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            logits = model(xb)

            loss = criterion(logits, yb)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(train_ds)

        val_pr_auc, val_f1 = evaluate_loader(model, val_loader, device)

        history.append({"epoch": epoch, "train_loss": epoch_loss,
                        
                         "val_pr_auc": val_pr_auc, "val_f1": val_f1})
        
        print(f"[{run_name}] epoch {epoch:02d}  loss={epoch_loss:.4f}  "
              
              f"val_PR-AUC={val_pr_auc:.4f}  val_F1={val_f1:.4f}")

        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            torch.save({"model_state": model.state_dict(),
                        "input_dim": X_train.shape[1],
                        "args": vars(args)}, ckpt_path)

    with open(os.path.join(CKPT_DIR, f"{run_name}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val PR-AUC: {best_pr_auc:.4f}  (checkpoint saved to {ckpt_path})")


if __name__ == "__main__":
    main()
