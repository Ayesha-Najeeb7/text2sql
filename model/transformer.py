"""Full encoder-decoder Transformer with masks and shared embedding / output projection."""
import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "starter"))
from embeddings import TokenEmbedding, InputLayer  # noqa: E402  (starter code, unchanged)
from .layers import Encoder, Decoder  # noqa: E402

PAD_ID = 0


def padding_mask(ids, pad_id=PAD_ID):
    """(B, L) ids -> (B, 1, L) bool; True where the key is a real token."""
    return (ids != pad_id).unsqueeze(1)


def causal_mask(L, device=None):
    """(1, L, L) bool lower-triangular; position t may see positions <= t."""
    return torch.tril(torch.ones(L, L, dtype=torch.bool, device=device)).unsqueeze(0)


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=3, d_ff=1024,
                 dropout=0.1, pad_id=PAD_ID, max_len=512):
        super().__init__()
        self.pad_id = pad_id
        self.shared = TokenEmbedding(vocab_size, d_model, pad_id)
        self.src_in = InputLayer(self.shared, d_model, max_len, dropout)
        self.tgt_in = InputLayer(self.shared, d_model, max_len, dropout)
        kw = dict(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
        self.encoder = Encoder(n_layers, **kw)
        self.decoder = Decoder(n_layers, **kw)
        self.out = nn.Linear(d_model, vocab_size, bias=False)
        self.out.weight = self.shared.emb.weight  # weight sharing: the SAME tensor
        self._init()

    def _init(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and "emb" not in name:
                nn.init.xavier_uniform_(p)
        # embedding std d^-0.5 so that (x * sqrt(d)) has unit scale; keep the pad row at zero
        d = self.shared.emb.embedding_dim
        nn.init.normal_(self.shared.emb.weight, mean=0.0, std=d ** -0.5)
        with torch.no_grad():
            self.shared.emb.weight[self.pad_id].zero_()

    def encode(self, src):
        src_mask = padding_mask(src, self.pad_id)
        return self.encoder(self.src_in(src), src_mask), src_mask

    def decode(self, tgt_in, memory, src_mask):
        T = tgt_in.size(1)
        tgt_mask = padding_mask(tgt_in, self.pad_id) & causal_mask(T, tgt_in.device)
        return self.out(self.decoder(self.tgt_in(tgt_in), memory, tgt_mask, src_mask))

    def forward(self, src, tgt_in):
        memory, src_mask = self.encode(src)
        return self.decode(tgt_in, memory, src_mask)  # (B, T, V) logits

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
