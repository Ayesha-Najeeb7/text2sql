"""Task 4: greedy decoding, beam search (size 4), parser to WikiSQL format, readable SQL."""
import re
import torch
from tokenizer import PAD_ID, BOS_ID, EOS_ID
from data_prep import AGG_OPS, COND_OPS

MAX_LEN = 64


@torch.no_grad()
def greedy_decode(model, src, max_len=MAX_LEN):
    """src: (B, S) -> list of token-id lists (without <s> and </s>)."""
    B = src.size(0)
    memory, src_mask = model.encode(src)
    ys = torch.full((B, 1), BOS_ID, dtype=torch.long, device=src.device)
    done = torch.zeros(B, dtype=torch.bool, device=src.device)
    for _ in range(max_len):
        nxt = model.decode(ys, memory, src_mask)[:, -1].argmax(-1)
        nxt = nxt.masked_fill(done, PAD_ID)
        ys = torch.cat([ys, nxt[:, None]], 1)
        done |= nxt == EOS_ID
        if done.all():
            break
    return [_strip(r.tolist()) for r in ys[:, 1:]]


@torch.no_grad()
def beam_search(model, src, beam=4, max_len=MAX_LEN, alpha=0.6):
    """Batched beam search with the length penalty of the paper (alpha=0.6)."""
    B, dev = src.size(0), src.device
    memory, src_mask = model.encode(src)
    memory = memory.repeat_interleave(beam, 0)
    src_mask = src_mask.repeat_interleave(beam, 0)
    seqs = torch.full((B, beam, 1), BOS_ID, dtype=torch.long, device=dev)
    scores = torch.full((B, beam), float("-inf"), device=dev)
    scores[:, 0] = 0.0
    finished = torch.zeros(B, beam, dtype=torch.bool, device=dev)
    lengths = torch.zeros(B, beam, device=dev)
    for _ in range(max_len):
        logp = torch.log_softmax(model.decode(seqs.view(B * beam, -1), memory, src_mask)[:, -1], -1)
        V = logp.size(-1)
        logp = logp.view(B, beam, V)
        # finished beams can only emit <pad> at zero cost, so their score is frozen
        frozen = torch.full_like(logp, float("-inf")); frozen[..., PAD_ID] = 0.0
        logp = torch.where(finished[..., None], frozen, logp)
        cand = (scores[..., None] + logp).view(B, beam * V)
        scores, idx = cand.topk(beam, -1)
        src_beam, tok = idx // V, idx % V
        seqs = torch.cat([seqs.gather(1, src_beam[..., None].expand(-1, -1, seqs.size(2))), tok[..., None]], 2)
        was_done = finished.gather(1, src_beam)
        lengths = lengths.gather(1, src_beam) + (~was_done).float()
        finished = was_done | (tok == EOS_ID)
        if finished.all():
            break
    norm = scores / (((5 + lengths) / 6) ** alpha)
    best = norm.argmax(-1)
    return [_strip(seqs[b, best[b], 1:].tolist()) for b in range(B)]


def _strip(ids):
    out = []
    for t in ids:
        if t in (EOS_ID, PAD_ID):
            break
        out.append(t)
    return out


# ---------------------------------------------------------------- parsing
_HEAD = re.compile(r"^select\s+(?:(max|min|count|sum|avg)\s+)?<c(\d+)>\s*(.*)$", re.S)
_COND = re.compile(r"<c(\d+)>\s*([=><])\s*(.*)$", re.S)
_SPLIT = re.compile(r"\s+and\s+(?=<c\d+>\s*[=><](?:\s|$))")


def parse_sql_string(text):
    """'select count <c3> where <c1> = kim manners' -> {'sel': 3, 'agg': 3, 'conds': [[1, 0, 'kim manners']]}
    Returns None when the string is not a well-formed query."""
    m = _HEAD.match(text.strip())
    if not m:
        return None
    agg, sel, rest = m.group(1), int(m.group(2)), m.group(3).strip()
    q = {"sel": sel, "agg": AGG_OPS.index(agg.upper()) if agg else 0, "conds": []}
    if not rest:
        return q
    if not rest.startswith("where"):
        return None
    for part in _SPLIT.split(rest[len("where"):].strip()):
        c = _COND.match(part.strip())
        if not c:
            return None
        q["conds"].append([int(c.group(1)), COND_OPS.index(c.group(2)), c.group(3).strip()])
    return q


def parse_ids(sp, ids):
    return parse_sql_string(sp.decode(ids))


def to_line(query):
    """The prediction-file record for one example (never skip a line)."""
    return {"error": "parse"} if query is None else {"query": query}


def to_readable_sql(query, header, table_name="table"):
    """Parsed query + real column names -> SQL string. Returns None if a column index is out of range."""
    if query is None or not 0 <= query["sel"] < len(header):
        return None
    col = header[query["sel"]]
    sel = f"{AGG_OPS[query['agg']]}({col})" if query["agg"] else col
    sql = f"SELECT {sel} FROM {table_name}"
    conds = []
    for c, op, val in query["conds"]:
        if not 0 <= c < len(header):
            return None
        v = str(val)
        try:
            float(v); lit = v
        except ValueError:
            lit = "'" + v.replace("'", "''") + "'"
        conds.append(f"{header[c]} {COND_OPS[op]} {lit}")
    if conds:
        sql += " WHERE " + " AND ".join(conds)
    return sql
