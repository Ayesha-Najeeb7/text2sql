"""Scaled dot-product and multi-head attention (Vaswani et al. 2017, Section 3.2)."""
import math
import torch
import torch.nn as nn


def scaled_dot_product_attention(q, k, v, mask=None):
    """Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V.

    q: (..., Lq, dk)  k: (..., Lk, dk)  v: (..., Lk, dv)
    mask: bool, broadcastable to (..., Lq, Lk); True = may attend, False = hidden.
    Returns (output, attention weights).
    """
    dk = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    # a query row with every key masked would give NaN; zero it (cannot happen with our masks)
    weights = torch.nan_to_num(weights, nan=0.0)
    return torch.matmul(weights, v), weights


class MultiHeadAttention(nn.Module):
    """Project Q,K,V with W^Q, W^K, W^V, run h heads in parallel, concatenate, project with W^O."""

    def __init__(self, d_model=256, n_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attn = None  # last attention weights (B, h, Lq, Lk), kept for plotting

    def _split(self, x):  # (B, L, d) -> (B, h, L, dk)
        B, L, _ = x.shape
        return x.view(B, L, self.h, self.dk).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)  # (B, Lq, Lk) -> (B, 1, Lq, Lk) for all heads
        qh, kh, vh = self._split(self.w_q(q)), self._split(self.w_k(k)), self._split(self.w_v(v))
        out, w = scaled_dot_product_attention(qh, kh, vh, mask)
        self.attn = w.detach()
        B, _, Lq, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.h * self.dk)
        return self.w_o(out)
