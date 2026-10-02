"""Task 3: teacher forcing, label smoothing 0.1, Adam(0.9, 0.98, 1e-9), warmup 4000, batch 64, 20 epochs."""
import argparse, csv, json, time
from pathlib import Path
import torch
import torch.nn.functional as F
import common
from common import CFG, load_sp
from dataset import make_loader
from model.transformer import Transformer
from tokenizer import PAD_ID


def noam(d_model, warmup):
    return lambda step: d_model ** -0.5 * min(max(step, 1) ** -0.5, max(step, 1) * warmup ** -1.5)


def run_epoch(model, dl, device, opt=None, sched=None):
    train = opt is not None
    model.train(train)
    tot_ls = tot_nll = n_tok = 0.0
    with torch.set_grad_enabled(train):
        for src, tgt in dl:
            src, tgt = src.to(device), tgt.to(device)
            logits = model(src, tgt[:, :-1])  # teacher forcing: input tgt[:, :-1] ...
            gold = tgt[:, 1:]  # ... predicts tgt[:, 1:]
            flat, flat_gold = logits.reshape(-1, logits.size(-1)), gold.reshape(-1)
            loss = F.cross_entropy(flat, flat_gold, ignore_index=PAD_ID, label_smoothing=0.1)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step(); sched.step()
            n = (flat_gold != PAD_ID).sum().item()
            tot_ls += loss.item() * n
            tot_nll += F.cross_entropy(flat, flat_gold, ignore_index=PAD_ID, reduction="sum").item()
            n_tok += n
    return tot_ls / n_tok, tot_nll / n_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=4000)
    ap.add_argument("--max_train", type=int, default=0, help="debug: use only N train examples")
    ap.add_argument("--max_dev", type=int, default=0, help="debug: use only N dev examples")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp = load_sp()
    train_dl = make_loader("train_pairs.jsonl", sp, True, a.batch_size)
    dev_dl = make_loader("dev_pairs.jsonl", sp, False, a.batch_size)
    if a.max_train:
        train_dl.dataset.items = train_dl.dataset.items[: a.max_train]
    if a.max_dev:
        dev_dl.dataset.items = dev_dl.dataset.items[: a.max_dev]

    model = Transformer(sp.get_piece_size(), **CFG).to(device)
    n_params = model.num_params()
    print(f"device={device}  trainable parameters={n_params:,}")
    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, noam(CFG["d_model"], a.warmup))

    out = Path(a.out); out.mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    log = open("results/train_log.csv", "w", newline="")
    w = csv.writer(log); w.writerow(["epoch", "train_loss", "dev_loss", "train_nll", "dev_nll", "lr", "seconds"])
    best, best_ep, t0 = float("inf"), 0, time.time()
    for ep in range(1, a.epochs + 1):
        t1 = time.time()
        tr, tr_nll = run_epoch(model, train_dl, device, opt, sched)
        dv, dv_nll = run_epoch(model, dev_dl, device)
        lr = sched.get_last_lr()[0]
        w.writerow([ep, f"{tr:.4f}", f"{dv:.4f}", f"{tr_nll:.4f}", f"{dv_nll:.4f}", f"{lr:.6f}", f"{time.time()-t1:.0f}"])
        log.flush()
        print(f"epoch {ep:2d}  train {tr:.4f}  dev {dv:.4f}  lr {lr:.6f}  ({time.time()-t1:.0f}s)")
        if dv < best:  # the dev set is used only to choose the checkpoint
            best, best_ep = dv, ep
            torch.save({"model": model.state_dict(), "epoch": ep, "dev_loss": dv}, out / "best.pt")
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    json.dump({"params": n_params, "epochs": a.epochs, "best_epoch": best_ep, "best_dev_loss": best,
               "train_seconds": time.time() - t0, "device": gpu, "warmup": a.warmup,
               "batch_size": a.batch_size}, open("results/train_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
