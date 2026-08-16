"""
Model architectures.

- MLP: baseline dense network
- MLPAttention: dense layers -> lightweight self-attention over the feature
  embedding -> classification head. Attention here treats each learned hidden
  unit as a "token" so the model can learn to weight which latent features
  matter most for a given transaction, which we then also use qualitatively
  alongside SHAP for interpretability.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Plain baseline MLP."""

    def __init__(self, input_dim: int, hidden_dims=(128, 64, 32), dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # raw logits


class SelfAttentionBlock(nn.Module):
    """Single-head self-attention over the hidden representation.

    We reshape the hidden vector into (seq_len, head_dim) "pseudo-tokens" so a
    plain tabular embedding can still benefit from attention: the model learns
    which chunks of the latent representation to emphasize per-transaction,
    rather than attending over raw input features directly.
    """

    def __init__(self, hidden_dim: int, n_chunks: int = 8):
        super().__init__()
        assert hidden_dim % n_chunks == 0, "hidden_dim must be divisible by n_chunks"
        self.n_chunks = n_chunks
        self.chunk_dim = hidden_dim // n_chunks
        self.query = nn.Linear(self.chunk_dim, self.chunk_dim)
        self.key = nn.Linear(self.chunk_dim, self.chunk_dim)
        self.value = nn.Linear(self.chunk_dim, self.chunk_dim)
        self.scale = self.chunk_dim ** 0.5
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        batch_size = x.size(0)
        chunks = x.view(batch_size, self.n_chunks, self.chunk_dim)

        q = self.query(chunks)
        k = self.key(chunks)
        v = self.value(chunks)

        attn_scores = torch.bmm(q, k.transpose(1, 2)) / self.scale  # (B, n_chunks, n_chunks)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attended = torch.bmm(attn_weights, v)  # (B, n_chunks, chunk_dim)

        attended = attended.reshape(batch_size, -1)
        out = self.out_proj(attended)
        return self.norm(out + x), attn_weights  # residual connection + weights for inspection


class MLPAttention(nn.Module):
    """Dense layers -> self-attention -> classification head."""

    def __init__(self, input_dim: int, hidden_dims=(128, 64), attn_chunks: int = 8,
                 dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            prev_dim = h
        self.encoder = nn.Sequential(*layers)

        self.attn = SelfAttentionBlock(prev_dim, n_chunks=attn_chunks)
        self.classifier = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(prev_dim // 2, 1),
        )

    def forward(self, x, return_attn: bool = False):
        h = self.encoder(x)
        h_attn, attn_weights = self.attn(h)
        logits = self.classifier(h_attn).squeeze(-1)
        if return_attn:
            return logits, attn_weights
        return logits


def build_model(name: str, input_dim: int) -> nn.Module:
    if name == "mlp":
        return MLP(input_dim)
    elif name == "mlp_attention":
        return MLPAttention(input_dim)
    else:
        raise ValueError(f"Unknown model name: {name}")
