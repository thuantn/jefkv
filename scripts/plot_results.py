"""
Summarise runs/: results table (mean ± std over seeds) and two plots.

  runs/<config>_s<seed>/{train_log.jsonl, eval.json}   trained runs
  runs/zs_<name>/eval.json                              zero-shot baselines

Writes <out_dir>/results.md, training_dynamics.png (clean vs RandomKV) and layer_retrieval.png.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLS = [("ood", "acc"), ("multi", "acc"), ("alias", "acc"), ("ood", "ece"), ("trap", "mean_conf"),
        ("abstain_auroc", None), ("ood", "avg_ms")]


def fmt(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "–"
    return f"{np.mean(xs):.2f}" if len(xs) == 1 else f"{np.mean(xs):.2f} ± {np.std(xs):.2f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out_dir", default="outputs")
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    table = defaultdict(lambda: defaultdict(list))   # row -> col -> values over seeds
    layers = defaultdict(lambda: defaultdict(list))  # row -> layer -> acc@1 over seeds
    for f in sorted(Path(args.runs).glob("*/eval.json")):
        config = re.sub(r"_s\d+$", "", f.parent.name)
        for variant, res in json.loads(f.read_text(encoding="utf-8"))["variants"].items():
            row = f"{config} / {variant}"
            for split, metric in COLS:
                table[row][(split, metric)].append(res[split] if metric is None else res[split].get(metric))
            for layer, r in res["ood"].get("retrieval", {}).items():
                layers[row][int(layer)].append(r["acc@1"])

    head = ["run / variant", "OOD acc", "multi acc", "alias acc", "OOD ECE", "trap conf (->50)", "abstain AUROC", "ms/case"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for row, cols in table.items():
        lines.append("| " + " | ".join([row] + [fmt(cols[c]) for c in COLS]) + " |")
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    clean = sorted(Path(args.runs).glob("kv_jef_s*/train_log.jsonl"))
    rand = sorted(Path(args.runs).glob("kv_jef_randomkv_s*/train_log.jsonl"))
    if clean and rand:
        plt.figure(figsize=(7, 4.2))
        for files, label, style in [(clean, "Clean KG", "-"), (rand, "RandomKV", "--")]:
            for i, fp in enumerate(files):
                recs = [r for r in (json.loads(x) for x in fp.read_text(encoding="utf-8").splitlines() if x.strip())
                        if "ema_loss" in r]  # skips --restart_on_collapse markers
                plt.plot([r["step"] for r in recs], [r["ema_loss"] for r in recs], style,
                         color="C0" if label == "Clean KG" else "C3", alpha=0.8, label=label if i == 0 else None)
        plt.xlabel("step"), plt.ylabel("EMA training loss"), plt.title("Retrieval vs. RandomKV")
        plt.legend(), plt.grid(alpha=0.3), plt.tight_layout()
        plt.savefig(out / "training_dynamics.png", dpi=160)
        plt.close()

    if layers:
        plt.figure(figsize=(7, 4.0))
        for row, d in layers.items():
            ks = sorted(d)
            plt.plot(ks, [np.mean(d[k]) for k in ks], marker="o", label=row)
        plt.xlabel("global attention layer"), plt.ylabel("retrieval ACC@1 (%)"), plt.title("Which layer retrieves?")
        plt.legend(fontsize=7), plt.grid(alpha=0.3), plt.tight_layout()
        plt.savefig(out / "layer_retrieval.png", dpi=160)
        plt.close()
    print(f"wrote {out}/results.md and plots")


if __name__ == "__main__":
    main()
