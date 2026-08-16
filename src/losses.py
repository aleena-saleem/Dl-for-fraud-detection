"""
Loss functions for extreme class imbalance.

FocalLoss down-weights easy (well-classified) examples and focuses gradient
signal on hard/minority examples, controlled by:
  - alpha: class-balancing weight (higher -> more weight on positive/fraud class)
  - gamma: focusing parameter (higher -> more down-weighting of easy examples;
           gamma=0 reduces to plain weighted BCE)

Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017),
adapted here from detection to tabular binary classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_term = alpha_t * (1 - p_t) ** self.gamma * bce

        if self.reduction == "mean":
            return focal_term.mean()
        elif self.reduction == "sum":
            return focal_term.sum()
        return focal_term


class WeightedBCELoss(nn.Module):
    """Standard cost-sensitive baseline: weight positive class by pos_weight."""

    def __init__(self, pos_weight: float):
        super().__init__()
        self.pos_weight = torch.tensor(pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight.to(logits.device)
        )


def build_loss(name: str, y_train) -> nn.Module:
    """name in {'bce', 'weighted_bce', 'focal'}"""
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_weight = n_neg / max(n_pos, 1)

    if name == "bce":
        return nn.BCEWithLogitsLoss()
    elif name == "weighted_bce":
        return WeightedBCELoss(pos_weight=pos_weight)
    elif name == "focal":
        return FocalLoss(alpha=0.75, gamma=2.0)
    else:
        raise ValueError(f"Unknown loss name: {name}")
