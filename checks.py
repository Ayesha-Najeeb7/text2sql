"""Correctness checks (Section 3.2) + Task 1 stats/heat-map + Fig 2/3. Output goes to results/checks.txt."""
import json, subprocess, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common
from common import CFG, load_sp
from data_prep import load_split, encode_target
from decode import parse_ids, to_line
from model.attention import scaled_dot_product_attention
from model.transformer import Transformer, causal_mask
from embeddings import PositionalEncoding
from tokenizer import read_pairs, PAD_ID, BOS_ID, EOS_ID

R = Path("results"); R.mkdir(exist_ok=True)
lines = []
def say(s=""):
    print(s); lines.append(s)


def table1(sp):
    say("== Table 1: data ==")
    t1 = {}
    for split in ["train", "dev", "test"]:
        pairs = read_pairs(f"{split}_pairs.jsonl")
        S = [len(sp.encode(p["src"])) + 1 for p in pairs]; T = [len(sp.encode(p["tgt"])) + 2 for p in pairs]
        dropped = sum(1 for s, t in zip(S, T) if s > 160 or t > 64) if split == "train" else 0
        t1[split] = dict(pairs=len(pairs), src_mean=float(np.mean(S)), src_max=max(S), tgt_mean=float(np.mean(T)),
                         tgt_max=max(T), dropped=dropped)
        say(f"{split:5s} pairs={len(pairs)} src mean/max={np.mean(S):.1f}/{max(S)} tgt mean/max={np.mean(T):.1f}/{max(T)} dropped={dropped}")
    json.dump(t1, open(R / "table1.json", "w"), indent=2)


def pe_heatmap():
    pe = PositionalEncoding(256, 512, 0.0).pe[0, :100].numpy()
    plt.figure(figsize=(9, 4)); plt.imshow(pe, aspect="auto", cmap="RdBu"); plt.colorbar()
    plt.xlabel("dimension (0-255)"); plt.ylabel("position (0-99)"); plt.title("Sinusoidal positional encoding")
    plt.tight_layout(); plt.savefig(R / "fig1_positional_encoding.png", dpi=150); plt.close()


def lr_plot():
    d, w = 256, 4000
    steps = np.arange(1, 20001); lr = d ** -0.5 * np.minimum(steps ** -0.5, steps * w ** -1.5)
    plt.figure(figsize=(6, 3.5)); plt.plot(steps, lr); plt.axvline(w, ls="--", c="gray")
    plt.xlabel("step"); plt.ylabel("learning rate"); plt.title("Learning-rate schedule (warmup 4000)")
    plt.tight_layout(); plt.savefig(R / "fig3_lr_schedule.png", dpi=150); plt.close()
    up = np.all(np.diff(lr[:w]) > 0); down = np.all(np.diff(lr[w:]) < 0)
    ratio = lr[-1] / lr[w]; expect = (20000 / w) ** -0.5
    say(f"LR: rises for {w} steps: {up}; decays after: {down}; lr(20000)/lr(4000)={ratio:.4f} vs (20000/4000)^-0.5={expect:.4f}")


def loss_plot():
    p = R / "train_log.csv"
    if not p.exists():
        return
    import csv
    rows = list(csv.DictReader(open(p)))
    ep = [int(r["epoch"]) for r in rows]
    plt.figure(figsize=(6, 3.5))
    plt.plot(ep, [float(r["train_loss"]) for r in rows], label="train"); plt.plot(ep, [float(r["dev_loss"]) for r in rows], label="dev")
    plt.xlabel("epoch"); plt.ylabel("label-smoothed CE"); plt.legend(); plt.title("Loss per epoch")
    plt.tight_layout(); plt.savefig(R / "fig2_loss.png", dpi=150); plt.close()


def model_checks(sp):
    torch.manual_seed(0)
    m = Transformer(sp.get_piece_size(), **CFG).eval()  # random init is enough: these are structural properties
    say(f"== trainable parameters: {m.num_params():,} ==")
    src = torch.randint(4, 500, (3, 12)); tgt = torch.randint(4, 500, (3, 9))
    # causal mask
    out1 = m(src, tgt)
    tgt2 = tgt.clone(); tgt2[:, -1] = (tgt2[:, -1] + 7) % 500 + 4
    out2 = m(src, tgt2)
    d = (out1[:, :-1] - out2[:, :-1]).abs().max().item()
    say(f"Causal mask: max change at earlier positions after editing last token = {d:.2e}  -> {'PASS' if d < 1e-5 else 'FAIL'}"
        f" (last position changed by {(out1[:, -1]-out2[:, -1]).abs().max().item():.2e})")
    # padding mask
    src_pad = torch.cat([src, torch.full((3, 5), PAD_ID)], 1)
    d = (m(src, tgt) - m(src_pad, tgt)).abs().max().item()
    say(f"Padding mask: max output change after adding 5 <pad> to the source = {d:.2e}  -> {'PASS' if d < 1e-5 else 'FAIL'}")
    # attention rows sum to 1 over unmasked positions
    mem, sm = m.encode(src_pad)
    w = m.encoder.layers[0].self_attn.attn  # (B,h,S,S)
    sums = w.sum(-1); masked_w = (w * (~sm.unsqueeze(1))).sum().item()
    say(f"Attention rows: min/max row sum = {sums.min():.6f}/{sums.max():.6f}; weight on <pad> keys = {masked_w:.1e}"
        f"  -> {'PASS' if abs(sums-1).max() < 1e-5 and masked_w == 0 else 'FAIL'}")
    q = torch.randn(2, 4, 5, 64); mask = causal_mask(5)
    _, aw = scaled_dot_product_attention(q, q, q, mask)
    say(f"Causal attention rows sum to 1: {torch.allclose(aw.sum(-1), torch.ones(2,4,5), atol=1e-6)}; "
        f"upper-triangle weight = {aw.triu(1).sum().item():.1e}")
    # weight sharing
    same = m.out.weight is m.shared.emb.weight and m.src_in.tok.emb.weight is m.tgt_in.tok.emb.weight
    say(f"Weight sharing: output.weight is embedding.weight -> {same}  "
        f"(counted once: {m.num_params():,} parameters)")


def gold_round_trip(sp):
    ex, _ = load_split("dev")
    with open(R / "dev_gold_roundtrip.jsonl", "w") as f:
        for e in ex:  # gold target -> BPE -> ids -> decode -> parser
            ids = sp.encode(encode_target(e["sql"]))
            f.write(json.dumps(to_line(parse_ids(sp, ids))) + "\n")
    r = subprocess.run([sys.executable, "evaluate.py", "data/dev.jsonl", "data/dev.db", "../results/dev_gold_roundtrip.jsonl"],
                       cwd="WikiSQL", capture_output=True, text=True)
    js = json.loads(r.stdout[r.stdout.index("{"):])
    say(f"Gold round-trip (dev): logical form {100*js['lf_accuracy']:.2f}%  execution {100*js['ex_accuracy']:.2f}%"
        f"  -> {'PASS' if js['ex_accuracy'] > 0.99 else 'FAIL'}")


if __name__ == "__main__":
    sp = load_sp()
    table1(sp); pe_heatmap(); lr_plot(); loss_plot(); model_checks(sp); gold_round_trip(sp)
    (R / "checks.txt").write_text("\n".join(lines) + "\n")
