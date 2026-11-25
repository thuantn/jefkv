"""Name overlap between a training file and a test file (entities, tiers, any KG head/tail word).
A fresh test set must share no entity name with the training data.
  python scripts/check_overlap.py data/processed/train.json data/fresh/test.json"""
import json, re, sys

def names(path):
    d = json.load(open(path, encoding="utf-8"))
    heads = {t["head"].lower() for t in d["triples"]}
    words = {w for t in d["triples"] for w in re.findall(r"[a-z]{4,}", (t["head"] + " " + t["tail"]).lower())}
    return heads, words

tr_h, tr_w = names(sys.argv[1])
te_h, te_w = names(sys.argv[2])
print(f"train heads {len(tr_h)}, test heads {len(te_h)}, shared heads {len(tr_h & te_h)}")
print(f"shared head/tail words (>=4 letters): {len(tr_w & te_w)}; e.g. {sorted(tr_w & te_w)[:8]}")
