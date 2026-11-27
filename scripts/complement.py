"""Complementarity of two models on the same test set (post-hoc, 27/11/2026): python scripts/complement.py <preds A> <preds B>
(preds_*.jsonl from scripts/eval.py; A is called "KV" in the output).
Prints acc / cf of each, 'either right' upper bound, KV-only-right share, and a confidence gate (pick the surer model)."""
import json, sys
L = lambda f: {r["i"]: r for r in map(json.loads, open(f, encoding="utf-8"))}
k, r = L(sys.argv[1]), L(sys.argv[2]); ids = sorted(set(k) & set(r)); n = len(ids)
ok = lambda x: (x["p"] >= 0.5) == (x["y"] == 1)
conf = lambda x: abs(x["p"] - 0.5)
acc_k = sum(ok(k[i]) for i in ids) / n; acc_r = sum(ok(r[i]) for i in ids) / n
either = sum(ok(k[i]) or ok(r[i]) for i in ids) / n
konly = sum(ok(k[i]) and not ok(r[i]) for i in ids) / n
gate = sum(ok(k[i] if conf(k[i]) > conf(r[i]) else r[i]) for i in ids) / n
cf = [i for i in ids if k[i].get("cf_pair") is not None and r[i].get("cf_pair") is not None]
c = lambda f: sum(f(i) for i in cf) / max(len(cf), 1)
print(f"n={n} acc KV {acc_k:.3f} RAG {acc_r:.3f} either {either:.3f} KV-only-right {konly:.3f} conf-gate {gate:.3f} | "
      f"cf n={len(cf)} KV {c(lambda i: k[i]['cf_pair']):.3f} RAG {c(lambda i: r[i]['cf_pair']):.3f} "
      f"either {c(lambda i: k[i]['cf_pair'] or r[i]['cf_pair']):.3f} KV-only {c(lambda i: k[i]['cf_pair'] and not r[i]['cf_pair']):.3f}")
