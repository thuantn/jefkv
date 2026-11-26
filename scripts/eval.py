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

from jefkv.memory import _TOK, BM25Retriever, DenseRetriever, kv_text, triple_text
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


def gold_set(s):
    """Triples that decide the claim: True -> the claimed fact (gold_idx[0]); False -> any real tail refutes it."""
    g = s.get("gold_idx") or []
    return set(g[:1]) if s["label"] else set(g)


def sel_layer(model):
    return next((i for i, m in model.kg_modules.items() if m.select_k), None)


def edited_search(r, query, k, gi, t):
    """Retrieve as if KG row gi had been edited to triple t AND re-indexed (what a real KG edit does to RAG)."""
    if isinstance(r, DenseRetriever):  # the edited row's score only (embeddings are inference tensors: no in-place edit)
        enc = lambda x: r.model.encode(x, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)  # noqa: E731
        q = enc([query])
        scores = (q @ r.emb.T)[0].clone()
        scores[gi] = (q @ enc([triple_text(t)]).to(q.dtype).T)[0, 0]
        return scores.topk(min(k, len(r.triples))).indices.tolist()
    from rank_bm25 import BM25Okapi
    docs = list(r.docs)
    docs[gi] = _TOK.findall(triple_text(t).lower())
    scores = torch.tensor(BM25Okapi(docs).get_scores(_TOK.findall(query.lower())))
    return scores.topk(min(k, len(docs))).indices.tolist()


def rag_ids(s, variant, r, edit=None):
    """Retrieved triple ids; oracle = the case's gold triples first (retrieval ceiling for the text reader)."""
    k = variant["k"]
    ids = edited_search(r, s["state"], k, *edit) if edit else r.search(s["state"], k)[0]
    if variant.get("oracle"):
        gold = list(s["gold_idx"])
        ids = (gold + [i for i in ids if i not in gold])[:k]
    return list(ids)


def kv_input(s, variant, memory, retrievers, edit=None):
    """(kg tensors, candidate ids or None) for the KV variants; prefilter = RAG's dense retriever picks the candidates."""
    if variant.get("kv") == "full":
        return (memory.keys, memory.vals, memory.slot_mask), None
    if variant.get("kv") == "prefilter":
        r = retrievers["dense"]
        ids = edited_search(r, s["state"], variant["k_pre"], *edit) if edit else r.search(s["state"], variant["k_pre"])[0]
        cand = torch.tensor(ids, device=memory.keys.device)
        return memory.subset(cand), cand
    if variant.get("kv") == "pruned":
        h, m = memory.encode([s["state"]], 128)
        q = (h.float() * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True)
        cand = memory.prune(q, variant["top_clusters"])
        return memory.subset(cand), cand
    return None, None


def local(ids, cand):  # global triple ids -> positions in the candidate subset (missing ones dropped)
    if cand is None:
        return list(ids)
    pos = {int(c): j for j, c in enumerate(cand.tolist())}
    return [pos[i] for i in ids if i in pos]


def run_model(model, packed, kg, device, dt, force=None):
    i = sel_layer(model)
    if force is not None:
        model.kg_modules[i].route["force"] = force
    try:
        with torch.autocast(device_type=device, dtype=dt, enabled=(dt != torch.float32)):
            return model(packed, kg)
    finally:
        if force is not None:
            model.kg_modules[i].route.pop("force", None)


def oracle_force(model, packed, kg, gold_local, device, dt):
    """The selector's top-k with the case's gold triples forced in: the reading ceiling of the KV path."""
    i = sel_layer(model)
    _, sl, _ = run_model(model, packed, kg, device, dt)
    k = model.kg_modules[i].select_k
    order = sl[i][0].topk(min(sl[i].shape[-1], k + len(gold_local))).indices.tolist()
    ids = (list(gold_local) + [x for x in order if x not in gold_local])[:k]
    return torch.tensor([ids], device=sl[i].device)


@torch.no_grad()
def run_split(model, samples, variant, memory, retrievers, device, dt):
    model.eval()
    probs, lat = [], []
    hits = {i: {"acc1": [], "acc5": [], "score5": []} for i in model.kg_modules}
    rec = []  # (label, {k: gold among the selector's / retriever's top k})
    for s in tqdm(samples, desc=variant["name"], leave=False):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        ctx, force = None, None
        ids = None
        if variant.get("retriever"):
            r = retrievers[variant["retriever"]]
            ids = rag_ids(s, variant, r)
            ctx = [" ".join(triple_text(r.triples[i]) for i in ids)]
        kg, cand = kv_input(s, variant, memory, retrievers)
        packed = model.batch([s], ctx)
        if variant.get("oracle") and kg is not None:
            force = oracle_force(model, packed, kg, local(s["gold_idx"], cand), device, dt)
        logits, sl, tl = run_model(model, packed, kg, device, dt, force)  # tl: post-softmax attention per triple; sl: scores
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
        gs = gold_set(s) if s.get("gold_idx") and "cf_tail" in s else set()  # real-data sets (FactKG) only
        if gs and not variant.get("oracle"):
            i = sel_layer(model)
            if sl and i in sl:  # KV: the selector's ranking over its candidates (prefilter misses count as misses)
                g = set(local(gs, cand))
                order = sl[i][0].topk(min(16, sl[i].shape[-1])).indices.tolist()
                rec.append((s["label"], {k: float(bool(g & set(order[:k]))) for k in (1, 5, 16)}))
            elif ids is not None:  # RAG: the retrieved rows
                rec.append((s["label"], {variant["k"]: float(bool(gs & set(ids)))}))

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
        for lab, nm in ((1, "acc_label_true"), (0, "acc_label_false")):
            sel = [(p >= 0.5) == bool(y) for p, y in zip(probs, labels) if y == lab]
            if sel and "cf_tail" in samples[0]:  # real-data sets only: keeps the synthetic eval.json unchanged
                out[nm] = round(100 * float(np.mean(sel)), 2)
    if any(h["acc1"] for h in hits.values()):
        out["retrieval"] = {str(i): {"acc@1": round(100 * float(np.mean(h["acc1"])), 2),
                                     "acc@5": round(100 * float(np.mean(h["acc5"])), 2),
                                     "score_acc@5": round(100 * float(np.mean(h["score5"])), 2)} for i, h in hits.items()}
    if rec:  # recall of the decisive gold triple(s): True claims need gold_idx[0], False ones any real tail
        ks = sorted(rec[0][1])
        out["recall"] = {f"@{k}": round(100 * float(np.mean([h[k] for _, h in rec])), 2) for k in ks}
        for lab, nm in ((1, "true"), (0, "false")):
            sub = [h for y, h in rec if y == lab]
            if sub:
                out["recall"].update({f"{nm}@{k}": round(100 * float(np.mean([h[k] for h in sub])), 2) for k in ks})
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
    cases whose prediction changes. Only for variants that see the whole KG or retrieved text.
    The edit is a real KG edit for every reader: the KV row is re-encoded and, on real data (cf_tail), the
    retrievers re-index the edited row before searching (the synthetic tables keep their original protocol)."""
    if variant.get("kv") == "pruned" or not (variant.get("kv") or variant.get("retriever")):
        return None
    model.eval()
    both = flips = n = 0
    by_label = {0: [], 1: []}  # pair correct, per original label (True claims vs False claims)
    hits = []  # was the gold triple among the retrieved / pre-filtered ones (else the edit cannot reach the model)
    for s in tqdm(samples, desc=variant["name"] + "/cf", leave=False):
        nums = re.findall(r"\d+", s["state"])
        if not s["gold_idx"] or not (nums or "cf_tail" in s):
            continue
        gi = s["gold_idx"][0]
        real = "cf_tail" in s
        if real:  # real data (FactKG): the converter picked a tail that flips the label
            swapped = dict(triples[gi], tail=s["cf_tail"])
        else:
            x = int(nums[-1])
            d = max(1, round(0.1 * x))
            swapped = dict(triples[gi], tail=str(x - d) if s["label"] else str(x + d))
        preds = []
        for t, label in ((triples[gi], s["label"]), (swapped, 1 - s["label"])):
            edit = (gi, t) if (t is swapped and real) else None
            ctx, kg, cand, undo, force = None, None, None, None, None
            if variant.get("retriever"):
                r = retrievers[variant["retriever"]]
                ids = rag_ids(s, variant, r, edit)
                ctx = [" ".join(triple_text(t if i == gi else r.triples[i]) for i in ids)]
                if t is swapped:
                    hits.append(gi in ids)
            try:
                if variant.get("kv") in ("full", "prefilter"):
                    undo = _set_row(memory, gi, t) if t is swapped else None
                    kg, cand = kv_input(s, variant, memory, retrievers, edit)
                    if cand is not None and t is swapped and not variant.get("retriever"):
                        hits.append(bool((cand == gi).any()))
                packed = model.batch([s], ctx)
                if variant.get("oracle") and kg is not None:
                    force = oracle_force(model, packed, kg, local(s["gold_idx"], cand), device, dt)
                logits, _, _ = run_model(model, packed, kg, device, dt, force)
            finally:
                if undo:
                    undo()
            pred = int(logits[0, 1] > logits[0, 0])
            preds.append((pred, pred == label))
        n += 1
        both += preds[0][1] and preds[1][1]
        flips += preds[0][0] != preds[1][0]
        by_label[s["label"]].append(preds[0][1] and preds[1][1])
    if not n:
        return None
    out = {"pair_acc": round(100 * both / n, 2), "flip_rate": round(100 * flips / n, 2), "n": n,
           "pair_acc_ci95": round(196 * float(np.sqrt(max(both / n * (1 - both / n), 1e-12) / n)), 2)}
    for lab, name in ((1, "pair_acc_label_true"), (0, "pair_acc_label_false")):
        if by_label[lab]:
            out[name] = round(100 * float(np.mean(by_label[lab])), 2)
    if hits:
        out["gold_retrieved"] = round(100 * float(np.mean(hits)), 2)
    return out


def variants_for(mode, diag=False):
    """diag (real-data diagnostics): drop the broken K-Means variant, add retrieval ceilings (oracle = gold triples
    forced into the KV top-k / the retrieved text) and KV reading with RAG's dense retriever as pre-filter."""
    if mode == "base":
        return [{"name": "jef_zero_shot"}]
    if mode == "rag_zs":
        return [{"name": "jef_rag_dense_k3", "retriever": "dense", "k": 3},
                {"name": "jef_rag_bm25_k3", "retriever": "bm25", "k": 3}] + \
               ([{"name": "jef_rag_dense_k3_oracle", "retriever": "dense", "k": 3, "oracle": True}] if diag else [])
    if mode == "kv":
        if diag:
            return [{"name": "kv_full", "kv": "full"}, {"name": "kv_oracle", "kv": "full", "oracle": True},
                    {"name": "kv_prefilter64", "kv": "prefilter", "k_pre": 64}]
        return [{"name": "kv_full", "kv": "full"}, {"name": "kv_pruned", "kv": "pruned", "top_clusters": 8}]
    if mode == "rag":
        return [{"name": f"rag_dense_k{k}", "retriever": "dense", "k": k} for k in (1, 3, 10)] + \
               [{"name": "rag_bm25_k3", "retriever": "bm25", "k": 3}] + \
               ([{"name": "rag_dense_k3_oracle", "retriever": "dense", "k": 3, "oracle": True}] if diag else [])
    if mode == "hybrid":
        return [{"name": "hybrid_dense_k3", "retriever": "dense", "k": 3, "kv": "full"}] + \
               ([{"name": "hybrid_oracle", "retriever": "dense", "k": 3, "kv": "full", "oracle": True}] if diag else [])
    raise ValueError(mode)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", help="runs/<name> containing adapter.pt")
    p.add_argument("--zero_shot", choices=["base", "rag"])
    p.add_argument("--test", default="data/processed/test.json")
    p.add_argument("--out_dir", help="default: --run, or runs/zs_<zero_shot>")
    p.add_argument("--limit", type=int, default=0, help="max samples per split (0 = all)")
    p.add_argument("--amp", default="auto")
    p.add_argument("--diag", action="store_true", help="real-data diagnostics: oracle retrieval ceilings, KV with a dense"
                   " pre-filter, no K-Means variant (its clustering copies the whole KG in fp32: OOM at 20k+ triples)")
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
        if not args.diag:
            memory.build_clusters(128)
    retrievers = {}
    if mode in ("rag", "rag_zs", "hybrid"):
        retrievers = {"dense": DenseRetriever(triples, device), "bm25": BM25Retriever(triples)}
    elif mode == "kv" and args.diag:  # the dense pre-filter variant
        retrievers = {"dense": DenseRetriever(triples, device)}

    results = {"train_args": targs, "variants": {}}
    for v in variants_for(mode, args.diag):
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
