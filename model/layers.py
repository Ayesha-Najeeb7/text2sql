"""Feed-forward, encoder layer, decoder layer (post-norm: LayerNorm(x + Sublayer(x)))."""
import torch.nn as nn
from .attention import MultiHeadAttention


class PositionwiseFeedForward(nn.Module):
    """FFN(x) = max(0, x W1 + b1) W2 + b2"""

    def __init__(self, d_model=256, d_ff=1024, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(self.relu(self.w1(x))))


class EncoderLayer(nn.Module):
    """self-attention -> add & norm -> FFN -> add & norm"""

    def __init__(self, d_model=256, n_heads=4, d_ff=1024, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, src_mask)))
        return self.norm2(x + self.drop(self.ffn(x)))


class DecoderLayer(nn.Module):
    """masked self-attention -> add & norm -> cross-attention -> add & norm -> FFN -> add & norm"""

    def __init__(self, d_model=256, n_heads=4, d_ff=1024, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(d_model) for _ in range(3))
        self.drop = nn.Dropout(dropout)

    def forward(self, y, memory, tgt_mask, mem_mask):
        y = self.norm1(y + self.drop(self.self_attn(y, y, y, tgt_mask)))
        y = self.norm2(y + self.drop(self.cross_attn(y, memory, memory, mem_mask)))
        return self.norm3(y + self.drop(self.ffn(y)))


class Encoder(nn.Module):
    def __init__(self, n_layers=3, **kw):
        super().__init__()
        self.layers = nn.ModuleList(EncoderLayer(**kw) for _ in range(n_layers))

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, n_layers=3, **kw):
        super().__init__()
        self.layers = nn.ModuleList(DecoderLayer(**kw) for _ in range(n_layers))

    def forward(self, y, memory, tgt_mask, mem_mask):
        for layer in self.layers:
            y = layer(y, memory, tgt_mask, mem_mask)
        return y
