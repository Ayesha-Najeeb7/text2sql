"""Render results/tables.md (Tables 1-4) from the json/csv files written by the other scripts."""
import json
from pathlib import Path
R = Path("results")
t1 = json.load(open(R / "table1.json")); s = json.load(open(R / "train_summary.json")); t3 = json.load(open(R / "table3.json"))
o = ["## Table 1 - Data", "", "| | Train | Dev | Test |", "|---|---|---|---|"]
o.append("| Pairs | " + " | ".join(f"{t1[k]['pairs']:,}" for k in t1) + " |")
o.append("| Mean / max source length | " + " | ".join(f"{t1[k]['src_mean']:.1f} / {t1[k]['src_max']}" for k in t1) + " |")
o.append("| Mean / max target length | " + " | ".join(f"{t1[k]['tgt_mean']:.1f} / {t1[k]['tgt_max']}" for k in t1) + " |")
o.append("| Pairs dropped as too long | " + " | ".join(str(t1[k]['dropped']) for k in t1) + " |")
o += ["", "## Table 2 - Model and training", "", "| | |", "|---|---|",
      f"| Trainable parameters | {s['params']:,} |", f"| Epochs trained / best epoch | {s['epochs']} / {s['best_epoch']} |",
      f"| Best dev loss | {s['best_dev_loss']:.4f} |", f"| Training time and GPU | {s['train_seconds']/60:.0f} min, {s['device']} |"]
o += ["", "## Table 3 - Official metrics", "", "| Split | Decoding | Logical form (%) | Execution (%) | Parse failures (%) |", "|---|---|---|---|---|"]
for key, name, dec in [("dev_greedy", "Dev", "greedy"), ("dev_beam", "Dev", "beam (4)"), ("test", "Test", None)]:
    if key in t3:
        r = t3[key]; d = dec or r.get("decoding", "")
        o.append(f"| {name} | {d} | {r['lf']:.2f} | {r['ex']:.2f} | {r['parse_fail']:.2f} |")
if "components" in t3:
    c = t3["components"]
    o += ["", "## Table 4 - Component accuracy (dev, beam)", "", "| sel column (%) | agg (%) | WHERE clause (%) |", "|---|---|---|",
          f"| {c['sel']:.2f} | {c['agg']:.2f} | {c['where']:.2f} |"]
(R / "tables.md").write_text("\n".join(o) + "\n"); print("\n".join(o))
