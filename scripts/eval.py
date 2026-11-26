"""
Evaluate one trained run (--run runs/x) or a zero-shot Jef baseline (--zero_shot base|rag) on
data/processed/test.json (splits: ood, multi, alias, trap). Writes <out>/eval.json.

Per variant and split: accuracy and ECE (rule-present splits), mean confidence, latency avg/p95 ms
(batch 1, including retrieval). Also: abstention AUROC (confidence separates ood from trap; higher =
the model knows when the rule is missing) and, for KV variants, per-layer retrieval ACC@1/ACC@5
(AtlasKV Section 5.2), and on ood a counterfactual tail swap (pair accuracy / flip rate): does the
model actually use the gold triple, or only a case-only prior?
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from jefkv.memory import BM25Retriever, DenseRetriever, kv_text, triple_text
from jefkv.model import JefKVModel
from scripts.train import amp_dtype, build_memory


def ece(p_true, labels, n_bins=10):
    p, y = np.asarray(p_true), np.asarray(labels)
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    acc = (pred == y).astype(float)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    return float(sum(abs(acc[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean()
                     for b in range(n_bins) if (bins == b).any()))


@torch.no_grad()
def run_split(model, samples, variant, memory, retrievers, device, dt):
    model.eval()
    probs, lat = [], []
    hits = {i: {"acc1": [], "acc5": [], "score5": []} for i in model.kg_modules}
    for s in tqdm(samples, desc=variant["name"], leave=False):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        ctx, kg, cand = None, None, None
        if variant.get("retriever"):
            r = retrievers[variant["retriever"]]
            ctx = [" ".join(triple_text(r.triples[i]) for i in r.search(s["state"], variant["k"])[0])]
        if variant.get("kv") == "full":
            kg = (memory.keys, memory.vals, memory.slot_mask)
        elif variant.get("kv") == "pruned":
            h, m = memory.encode([s["state"]], 128)
            q = (h.float() * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True)
            cand = memory.prune(q, variant["top_clusters"])
            kg = memory.subset(cand)
        packed = model.batch([s], ctx)
        with torch.autocast(device_type=device, dtype=dt, enabled=(dt != torch.float32)):
            logits, sl, tl = model(packed, kg)  # tl: post-softmax attention per triple (AtlasKV ACC@k); sl: retrieval scores
        p_true = F.softmax(logits.float(), -1)[0, 1].item()
        if device == "cuda":
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
        probs.append(p_true)

        if tl and s["gold_idx"]:
            gold = s["gold_idx"][0]  # the entity rule
            if cand is not None:
                where = (cand == gold).nonzero(as_tuple=True)[0]
                gold = int(where[0]) if where.numel() else -1
            for i, x in tl.items():
                top5 = x[0].topk(min(5, x.shape[-1])).indices.tolist()
                hits[i]["acc1"].append(float(gold >= 0 and top5[0] == gold))
                hits[i]["acc5"].append(float(gold in top5))
                hits[i]["score5"].append(float(gold in sl[i][0].topk(min(5, x.shape[-1])).indices.tolist()))

    n_nan = sum(p != p for p in probs)
    probs = [0.5 if p != p else p for p in probs]  # NaN = no answer; counted, never crashes a long run
    labels = [s["label"] for s in samples]
    conf = [max(p, 1 - p) for p in probs]
    out = {
        "n_nan": n_nan,
        "mean_conf": round(100 * float(np.mean(conf)), 2),
        "avg_ms": round(float(np.mean(lat[5:] or lat)), 2),
        "p95_ms": round(float(np.percentile(lat[5:] or lat, 95)), 2),
    }
    if samples[0]["rule_present"]:
        out["acc"] = round(100 * float(np.mean([(p >= 0.5) == bool(y) for p, y in zip(probs, labels)])), 2)
        out["ece"] = round(ece(probs, labels), 4)
    if any(h["acc1"] for h in hits.values()):
        out["retrieval"] = {str(i): {"acc@1": round(100 * float(np.mean(h["acc1"])), 2),
                                     "acc@5": round(100 * float(np.mean(h["acc5"])), 2),
                                     "score_acc@5": round(100 * float(np.mean(h["score5"])), 2)} for i, h in hits.items()}
    return out, conf


def _set_row(memory, gi, t):
    """Re-encode KG row gi as triple t, in place (no copy of the 10K-triple memory). Returns an undo closure."""
    # RandomKV: the gold triple's VALUE sits where shuffle_values put it (vals[vj] = value of triple perm[vj]).
    vj = gi if memory.perm is None else int((memory.perm == gi).nonzero()[0, 0])
    old = memory.keys[gi].clone(), memory.vals[vj].clone(), memory.slot_mask[gi].clone(), memory.slot_mask[vj].clone()
    S = memory.keys.shape[1]
    if memory.atlaskv:  # MiniLM KG2KV: key = "the <rel> of <head>" (unchanged), value = tail
        v, _ = memory.encode([t["tail"]])
        memory.vals[vj] = v[0].to(memory.vals.dtype)
    else:  # token-level: the sentence's states are its key and (possibly moved) value
        h, m = memory.encode([kv_text(t)], S)
        h, m = F.pad(h, (0, 0, 0, S - h.shape[1]))[0].to(memory.keys.dtype), F.pad(m, (0, S - m.shape[1]))[0]
        memory.keys[gi] = h
        memory.vals[vj] = h  # same tensor as keys when not shuffled
        if memory.perm is None:
            memory.slot_mask[gi] = m

    def undo():
        memory.keys[gi], memory.vals[vj], memory.slot_mask[gi], memory.slot_mask[vj] = old
    return undo


@torch.no_grad()
def counterfactual(model, samples, variant, memory, retrievers, triples, device, dt):
    """Causal check that the model USES the gold triple: same case, gold tail changed so the label flips
    (on ood, label = case number <= tail for every relation). A case-only shortcut gets at most one of
    each pair right, so pair accuracy is ~0 for a model that ignores the knowledge; flip rate = share of
    cases whose prediction changes. Only for variants that see the whole KG or retrieved text."""
    if variant.get("kv") == "pruned" or not (variant.get("kv") or variant.get("retriever")):
        return None
    model.eval()
    both = flips = n = 0
    for s in tqdm(samples, desc=variant["name"] + "/cf", leave=False):
        nums = re.findall(r"\d+", s["state"])
        if not s["gold_idx"] or not (nums or "cf_tail" in s):
            continue
        gi = s["gold_idx"][0]
        if "cf_tail" in s:  # real data (FactKG): the converter picked a tail that flips the label
            swapped = dict(triples[gi], tail=s["cf_tail"])
        else:
            x = int(nums[-1])
            d = max(1, round(0.1 * x))
            swapped = dict(triples[gi], tail=str(x - d) if s["label"] else str(x + d))
        preds = []
        for t, label in ((triples[gi], s["label"]), (swapped, 1 - s["label"])):
            ctx, kg, undo = None, None, None
            if variant.get("retriever"):
                r = retrievers[variant["retriever"]]
                ctx = [" ".join(triple_text(t if i == gi else r.triples[i]) for i in r.search(s["state"], variant["k"])[0])]
            if variant.get("kv") == "full":
                undo = _set_row(memory, gi, t) if t is swapped else None
                kg = (memory.keys, memory.vals, memory.slot_mask)
            try:
                with torch.autocast(device_type=device, dtype=dt, enabled=(dt != torch.float32)):
                    logits, _, _ = model(model.batch([s], ctx), kg)
            finally:
                if undo:
                    undo()
            pred = int(logits[0, 1] > logits[0, 0])
            preds.append((pred, pred == label))
        n += 1
        both += preds[0][1] and preds[1][1]
        flips += preds[0][0] != preds[1][0]
    if not n:
        return None
    return {"pair_acc": round(100 * both / n, 2), "flip_rate": round(100 * flips / n, 2), "n": n,
            "pair_acc_ci95": round(196 * float(np.sqrt(max(both / n * (1 - both / n), 1e-12) / n)), 2)}


def variants_for(mode):
    if mode == "base":
        return [{"name": "jef_zero_shot"}]
    if mode == "rag_zs":
        return [{"name": "jef_rag_dense_k3", "retriever": "dense", "k": 3},
                {"name": "jef_rag_bm25_k3", "retriever": "bm25", "k": 3}]
    if mode == "kv":
        return [{"name": "kv_full", "kv": "full"}, {"name": "kv_pruned", "kv": "pruned", "top_clusters": 8}]
    if mode == "rag":
        return [{"name": f"rag_dense_k{k}", "retriever": "dense", "k": k} for k in (1, 3, 10)] + \
               [{"name": "rag_bm25_k3", "retriever": "bm25", "k": 3}]
    if mode == "hybrid":
        return [{"name": "hybrid_dense_k3", "retriever": "dense", "k": 3, "kv": "full"}]
    raise ValueError(mode)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", help="runs/<name> containing adapter.pt")
    p.add_argument("--zero_shot", choices=["base", "rag"])
    p.add_argument("--test", default="data/processed/test.json")
    p.add_argument("--out_dir", help="default: --run, or runs/zs_<zero_shot>")
    p.add_argument("--limit", type=int, default=0, help="max samples per split (0 = all)")
    p.add_argument("--amp", default="auto")
    args = p.parse_args()
    assert bool(args.run) ^ bool(args.zero_shot), "pass exactly one of --run / --zero_shot"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dt = amp_dtype(device, args.amp)
    test = json.loads(Path(args.test).read_text(encoding="utf-8"))
    triples, splits = test["triples"], test["splits"]
    if args.limit:
        splits = {k: v[: args.limit] for k, v in splits.items()}

    if args.run:
        ck = torch.load(Path(args.run) / "adapter.pt", map_location="cpu", weights_only=True)
        targs = ck["args"]
        mode = targs["mode"]
        model = JefKVModel(kg_embed_dim=384 if targs["kv_encoder"] == "minilm" else 768,
                           use_kv=mode in ("kv", "hybrid"), unfreeze_top=targs["unfreeze_top"], device=device,
                           freeze_kg_v=targs.get("freeze_kg_v", False))
        model.load_trainable(ck["state"])
    else:
        mode = "base" if args.zero_shot == "base" else "rag_zs"
        targs = {"mode": mode, "kv_encoder": "jef", "max_slots": 16, "random_kv": False, "seed": 0}
        model = JefKVModel(use_kv=False, device=device)

    memory = None
    if mode in ("kv", "hybrid"):
        memory = build_memory(model, triples, targs["kv_encoder"], device, dt if device == "cuda" else torch.float32, targs["max_slots"])
        if targs["random_kv"]:
            memory.shuffle_values(targs["seed"])
        memory.build_clusters(128)
    retrievers = {}
    if mode in ("rag", "rag_zs", "hybrid"):
        retrievers = {"dense": DenseRetriever(triples, device), "bm25": BM25Retriever(triples)}

    results = {"train_args": targs, "variants": {}}
    for v in variants_for(mode):
        res, confs = {}, {}
        for name, samples in splits.items():
            res[name], confs[name] = run_split(model, samples, v, memory, retrievers, device, dt)
        cf = counterfactual(model, splits["ood"], v, memory, retrievers, triples, device, dt)
        if cf:
            res["ood"]["cf"] = cf
        if "trap" in confs:
            y = [1] * len(confs["ood"]) + [0] * len(confs["trap"])
            res["abstain_auroc"] = round(float(roc_auc_score(y, confs["ood"] + confs["trap"])), 4)
        results["variants"][v["name"]] = res
        print(v["name"], json.dumps({k: {m: x for m, x in r.items() if m != "retrieval"} if isinstance(r, dict) else r
                                     for k, r in res.items()}))

    out = Path(args.out_dir or args.run or f"runs/zs_{args.zero_shot}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {out / 'eval.json'}")


if __name__ == "__main__":
    main()
