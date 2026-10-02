import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "starter"))
import torch
import sentencepiece as spm
from model.transformer import Transformer

CFG = dict(d_model=256, n_heads=4, n_layers=3, d_ff=1024, dropout=0.1)


ROOT = Path(__file__).resolve().parent


def load_sp(path=None):
    return spm.SentencePieceProcessor(model_file=str(path or ROOT / "sql_sp.model"))


def load_model(ckpt="checkpoints/best.pt", sp=None, device="cpu"):
    sp = sp or load_sp()
    model = Transformer(sp.get_piece_size(), **CFG).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    return model.eval(), sp
