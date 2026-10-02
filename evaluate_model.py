"""Task 5: prediction files, component accuracy, attention map, plus the Task 1/2/3 figures and correctness checks."""
import argparse, json, subprocess, sys
from pathlib import Path
import torch
import common
from common import load_model
from dataset import make_loader
from decode import greedy_decode, beam_search, parse_ids, parse_sql_string, to_line, to_readable_sql
from data_prep import load_split, build_pairs
from tokenizer import BOS_ID, EOS_ID

R = Path("results")


def predict(model, sp, split, mode, device, bs=128, limit=0):
    dl = make_loader(f"{split}_pairs.jsonl", sp, False, bs)
    if limit:
        dl.dataset.items = dl.dataset.items[:limit]
    dl = torch.utils.data.DataLoader(dl.dataset, batch_size=bs, shuffle=False, collate_fn=dl.collate_fn)
    out = []
    for i, (src, _) in enumerate(dl):
        ids = (greedy_decode if mode == "greedy" else beam_search)(model, src.to(device))
        out += [parse_ids(sp, x) for x in ids]
        print(f"\r{split}/{mode}: {len(out)}", end="", flush=True)
    print()
    return out


def write_preds(queries, path):
    with open(path, "w") as f:
        for q in queries:
            f.write(json.dumps(to_line(q)) + "\n")  # same order as <split>.jsonl, never skip a line


def official(split, pred_path):
    """Run the official WikiSQL evaluator (not re-implemented)."""
    r = subprocess.run([sys.executable, "evaluate.py", f"data/{split}.jsonl", f"data/{split}.db", str(Path("..") / pred_path)],
                       cwd="WikiSQL", capture_output=True, text=True)
    last = [l for l in r.stdout.splitlines() if l.strip()]
    js = json.loads("\n".join(r.stdout[r.stdout.index("{"):].splitlines()))
    return js["lf_accuracy"] * 100, js["ex_accuracy"] * 100


def component_acc(split, queries, limit=0):
    examples, _ = load_split(split)
    examples = examples[: len(queries)]
    sel = agg = where = 0
    for ex, q in zip(examples, queries):
        g = ex["sql"]
        if q is None:
            continue
        sel += q["sel"] == g["sel"]
        agg += q["agg"] == g["agg"]
        norm = lambda cs: {(c, o, str(v).lower()) for c, o, v in cs}
        where += norm(q["conds"]) == norm(g["conds"])
    n = len(examples)
    return 100 * sel / n, 100 * agg / n, 100 * where / n


def samples(split, queries, n_each=5):
    examples, tables = load_split(split)
    pairs = build_pairs(split)
    good, bad = [], []
    for ex, q, p in zip(examples, queries, pairs):
        g = ex["sql"]; h = tables[ex["table_id"]]["header"]
        norm = lambda cs: {(c, o, str(v).lower()) for c, o, v in cs}
        ok = q is not None and q["sel"] == g["sel"] and q["agg"] == g["agg"] and norm(q["conds"]) == norm(g["conds"])
        rec = (ex["question"], to_readable_sql(g, h), to_readable_sql(q, h) if q else None, failure(q, g))
        (good if ok else bad).append(rec)
    return good[:n_each], bad[:n_each]


def failure(q, g):
    if q is None:
        return "parse failure"
    if q["sel"] != g["sel"]:
        return "wrong select column"
    if q["agg"] != g["agg"]:
        return "wrong aggregation"
    pc, gc = {(c, o) for c, o, _ in q["conds"]}, {(c, o) for c, o, _ in g["conds"]}
    if len(q["conds"]) < len(g["conds"]):
        return "missing condition"
    if len(q["conds"]) > len(g["conds"]):
        return "extra condition"
    if pc != gc:
        return "wrong condition column/operator"
    return "wrong value"


def write_samples(good, bad):
    with open(R / "samples.md", "w") as f:
        f.write("# Qualitative samples (dev)\n\n## Correct\n\n")
        for q, g, p, _ in good:
            f.write(f"- **Q:** {q}\n  - gold: `{g}`\n  - ours: `{p}`\n")
        f.write("\n## Wrong\n\n")
        for q, g, p, why in bad:
            f.write(f"- **Q:** {q}\n  - gold: `{g}`\n  - ours: `{p}`\n  - failure: **{why}**\n")


def attention_map(model, sp, device, idx=0):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pairs = [json.loads(l) for l in open("dev_pairs.jsonl")]
    p = pairs[idx]
    src = torch.tensor([sp.encode(p["src"]) + [EOS_ID]], device=device)
    ids = greedy_decode(model, src)[0]
    tgt = torch.tensor([[BOS_ID] + ids + [EOS_ID]], device=device)
    with torch.no_grad():
        model(src, tgt[:, :-1])
    w = model.decoder.layers[-1].cross_attn.attn[0].mean(0).cpu()  # (T, S), averaged over heads
    ys = sp.id_to_piece(tgt[0, 1:].tolist()); xs = sp.id_to_piece(src[0].tolist())
    fig, ax = plt.subplots(figsize=(max(8, len(xs) * 0.3), max(3, len(ys) * 0.35)))
    ax.imshow(w.numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, rotation=90, fontsize=7)
    ax.set_yticks(range(len(ys))); ax.set_yticklabels(ys, fontsize=8)
    ax.set_xlabel("source tokens"); ax.set_ylabel("generated tokens")
    ax.set_title(f"Last-layer decoder cross-attention (mean of heads), dev example {idx}")
    fig.tight_layout(); fig.savefig(R / "fig4_attention_map.png", dpi=150); plt.close(fig)
    print("saved attention map; generated:", sp.decode(ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N dev examples only")
    ap.add_argument("--test", action="store_true", help="run on test ONCE with --final decoding")
    ap.add_argument("--final", default="beam", choices=["greedy", "beam"])
    ap.add_argument("--skip_dev", action="store_true")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, sp = load_model(a.ckpt, device=device); model.to(device)
    R.mkdir(exist_ok=True)
    rows = json.load(open(R / "table3.json")) if (R / "table3.json").exists() else {}
    if not a.skip_dev:
        for mode in ["greedy", "beam"]:
            qs = predict(model, sp, "dev", mode, device, limit=a.limit)
            path = R / f"dev_{mode}.jsonl"; write_preds(qs, path)
            if a.limit:
                print("limit set: official evaluator needs the full file; skipping it")
                rows[f"dev_{mode}"] = {"parse_fail": 100 * sum(q is None for q in qs) / len(qs)}
            else:
                lf, ex = official("dev", path)
                rows[f"dev_{mode}"] = {"lf": lf, "ex": ex, "parse_fail": 100 * sum(q is None for q in qs) / len(qs)}
            print(mode, rows[f"dev_{mode}"])
            if mode == "beam":
                s, ag, wh = component_acc("dev", qs)
                rows["components"] = {"sel": s, "agg": ag, "where": wh}
                print("components", rows["components"])
                write_samples(*samples("dev", qs))
        attention_map(model, sp, device)
    if a.test:
        qs = predict(model, sp, "test", a.final, device)
        path = R / "test.jsonl"; write_preds(qs, path)
        lf, ex = official("test", path)
        rows["test"] = {"decoding": a.final, "lf": lf, "ex": ex, "parse_fail": 100 * sum(q is None for q in qs) / len(qs)}
        print(rows["test"])
    json.dump(rows, open(R / "table3.json", "w"), indent=2)


if __name__ == "__main__":
    main()
