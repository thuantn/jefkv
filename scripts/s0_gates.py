"""Apply the S0 decision rules of docs/PREREG_factkg.md to the per-claim predictions.

    python scripts/s0_gates.py <dir with fk_*_s<seed>/preds_*.jsonl> data/factkg/train.json data/factkg/test.json

Populations: ALL, RAG-MISS (the dense top-3 misses the deciding gold triple: `rag_hit` of rag_dense_k3, the same for
every seed because the retriever is not trained), UNSEEN-HEAD (claim's head entity not in the train KG).
Prints a table (acc / cf pair accuracy per variant, population and seed) and the H / K decisions.
"""
import json
import sys
from pathlib import Path
from statistics import mean


def load(p):
    return {x["i"]: x for x in map(json.loads, p.read_text(encoding="utf-8").splitlines())}


def metric(rows, ids, key):
    v = [rows[i][key] for i in ids if i in rows and key in rows[i]]
    if not v:
        return None
    if key == "p":  # accuracy from p_true
        v = [(rows[i]["p"] >= 0.5) == bool(rows[i]["y"]) for i in ids if i in rows and "p" in rows[i]]
    return 100 * mean(map(float, v))


def main():
    root, train, test = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    tr = json.loads(train.read_text(encoding="utf-8"))
    te = json.loads(test.read_text(encoding="utf-8"))
    ents = {t["head"].lower() for t in tr["triples"]} | {t["tail"].lower() for t in tr["triples"]}
    ood = te["splits"]["ood"]
    runs = {}  # (family, seed) -> {variant: rows}
    for d in sorted(root.glob("fk_*_s*")):
        fam, seed = d.name.rsplit("_s", 1)
        runs[(fam.removeprefix("fk_"), int(seed))] = {p.stem.removeprefix("preds_"): load(p) for p in d.glob("preds_*.jsonl")}
    rag_rows = next((v["rag_dense_k3"] for (f, _), v in sorted(runs.items()) if f == "rag" and "rag_dense_k3" in v), None)
    assert rag_rows, "need rag_dense_k3 predictions (rag_hit defines RAG-MISS)"
    pops = {"ALL": [s_i for s_i in range(len(ood))],
            "RAG-MISS": [i for i, r in rag_rows.items() if r.get("rag_hit") is False],
            "UNSEEN-HEAD": [i for i, s in enumerate(ood) if s["gold_idx"] and
                            te["triples"][s["gold_idx"][0]]["head"].lower() not in ents]}
    print({k: len(v) for k, v in pops.items()})
    table = {}  # (family, variant, pop, metric) -> {seed: value}
    for (fam, seed), vs in runs.items():
        for vn, rows in vs.items():
            for pn, ids in pops.items():
                for mn, key in (("acc", "p"), ("cf", "cf_pair")):
                    x = metric(rows, ids, key)
                    if x is not None:
                        table.setdefault((fam, vn, pn, mn), {})[seed] = x
    for k in sorted(table):
        v = table[k]
        print(f"{k[0]:16s} {k[1]:24s} {k[2]:11s} {k[3]:3s} mean {mean(v.values()):6.2f}  " +
              " ".join(f"s{s}={x:.1f}" for s, x in sorted(v.items())))

    def m(fam, vn, pn, mn):
        v = table.get((fam, vn, pn, mn))
        return (mean(v.values()), v) if v else (None, {})

    print("\n== decisions (docs/PREREG_factkg.md) ==")
    h_acc, _ = m("hybrid", "hybrid_qclaim", "RAG-MISS", "acc")
    r_acc, _ = m("rag", "rag_dense_k3", "RAG-MISS", "acc")
    h_cf, _ = m("hybrid", "hybrid_qclaim_cfkv", "RAG-MISS", "cf")
    full_acc, _ = m("hybrid", "hybrid_dense_k3", "RAG-MISS", "acc")
    if None not in (h_acc, r_acc, h_cf):
        i, ii = h_acc >= r_acc + 5, h_cf >= 25
        print(f"H(i)  hybrid_qclaim acc on RAG-MISS {h_acc:.1f} vs rag {r_acc:.1f} + 5 -> {i}")
        print(f"H(ii) hybrid_qclaim_cfkv cf on RAG-MISS {h_cf:.1f} >= 25 -> {ii}")
        if full_acc is not None and full_acc >= r_acc + 5 and not i:
            print("      note: the full-mask hybrid passes (i) but qclaim does not: the gain is the selector reading the RAG text")
        print("H:", "GO" if i and ii else "FAIL")
    kf, _ = m("kv_jef", "kv_full", "ALL", "acc")
    ko, _ = m("kv_jef", "kv_oracle", "ALL", "acc")
    kg, _ = m("kv_jef", "kv_oracle_goldonly", "ALL", "acc")
    if None not in (kf, ko, kg):
        print(f"K selector:    kv_oracle {ko:.1f} - kv_full {kf:.1f} >= 10 -> {ko - kf >= 10}")
        print(f"K distractors: kv_oracle_goldonly {kg:.1f} - kv_oracle {ko:.1f} >= 5 -> {kg - ko >= 5}")
        print(f"K reader:      kv_oracle_goldonly {kg:.1f} < 85 -> {kg < 85}")


if __name__ == "__main__":
    main()
