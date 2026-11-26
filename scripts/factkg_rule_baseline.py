"""Non-neural FactKG baseline: how far does plain string matching against the same KG memory get?

    python scripts/factkg_rule_baseline.py data/factkg/test.json

Rule: a claim is True iff some KG triple whose head occurs in the claim also has its tail in the claim. It reads the
same triples the models get (test["triples"]), no training. A model that does not beat this does not show reasoning
beyond surface matching (review round 2, 26/11/2026).
"""
import json
import sys
from pathlib import Path


def main():
    test = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    by_head = {}
    for t in test["triples"]:
        by_head.setdefault(t["head"].lower(), []).append(t["tail"].lower())
    heads = sorted(by_head, key=len, reverse=True)
    rows = []
    for s in test["splits"]["ood"]:
        c = s["state"].lower()
        pred = any(tail and tail in c for h in heads if h in c for tail in by_head[h])
        rows.append((s["label"], int(pred)))
    acc = 100 * sum(y == p for y, p in rows) / len(rows)
    out = {"n": len(rows), "acc": round(acc, 2)}
    for lab, nm in ((1, "acc_label_true"), (0, "acc_label_false")):
        sub = [y == p for y, p in rows if y == lab]
        out[nm] = round(100 * sum(sub) / len(sub), 2)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
