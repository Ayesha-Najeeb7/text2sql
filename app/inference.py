"""Model wrapper for the front end: question + column names -> SQL with the real column names."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "starter")]
import torch
from common import load_model
from data_prep import encode_source
from decode import beam_search, parse_ids, to_readable_sql
from tokenizer import EOS_ID

_cache = {}


def _get(ckpt):
    if ckpt not in _cache:
        _cache[ckpt] = load_model(str(ROOT / ckpt), sp=None)
        # sql_sp.model lives in the repo root
    return _cache[ckpt]


def generate_sql(question, columns, ckpt="checkpoints/best.pt"):
    """columns: list of column names. Returns (sql or None, raw model string, parsed dict or None)."""
    model, sp = _get(ckpt)
    ids = sp.encode(encode_source(question, columns)) + [EOS_ID]
    out = beam_search(model, torch.tensor([ids]))[0]
    parsed = parse_ids(sp, out)
    return to_readable_sql(parsed, columns), sp.decode(out), parsed
