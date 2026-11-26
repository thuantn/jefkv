"""Turn FactKG one-hop claims (Kim et al., ACL 2023) into a JefKV benchmark (train.json / test.json, same schema as
data/generate_benchmark.py). FactKG is CC BY-NC-SA 4.0 and its KG is DBpedia 2015: the outputs are derived data and
must stay OUT of the repo (default: a folder you choose outside it).

    python -I scripts/factkg_convert.py <factkg folder> <out folder> [--neighbours 3] [--max_train 12000] [--seed 0] [--grow 20000,50000]

Claims: FactKG "num1" one-hop claims about two entities linked by ONE relation (Evidence {e1: [[rel]], e2: [[~rel]]}).
Gold triples = every KG triple (head, rel, t), so a True claim's tail is among them and a False (substituted) claim
is refuted by them; claims whose head has no such triple (KG cannot decide) are dropped. Memory = gold triples plus
up to --neighbours other triples of each entity (same-entity hard distractors). Test = FactKG dev (test has no
Evidence). Counterfactual tail per sample (eval `cf`): True claim -> a real tail of the same relation from another
head (label flips to False); False claim -> the claimed entity (label flips to True). Needs ~20+ GB RAM for the KG.
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.factkg_inspect import load  # noqa: E402  restricted Unpickler

QUESTION = "Is this claim true according to the knowledge base?"
OPTIONS = {"true": "The knowledge base supports this claim.", "false": "The knowledge base contradicts this claim."}


def name(x: str) -> str:  # DBpedia key or quoted literal -> text
    return x.strip('"').replace("_", " ").strip()


def rel_text(r: str) -> str:  # camelCase -> words
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", r).lower()


def one_hop(claims: dict) -> list:
    out = []
    for text, v in claims.items():
        ev, ents = v.get("Evidence") or {}, v.get("Entity_set") or []
        if "num1" not in v.get("types", []) or "negation" in v["types"] or len(ents) != 2:
            continue
        links = set()
        for e, paths in ev.items():
            for p in paths:
                if len(p) != 1:
                    links = None
                    break
                r = p[0]
                other = ents[1] if e == ents[0] else ents[0]
                links.add((other, r[1:], e) if r.startswith("~") else (e, r, other))
            if links is None:
                break
        if links and len(links) == 1:
            (h, r, t), = links
            out.append((text, bool(v["Label"][0]), h, r, t))
    return out


def build(claims, kg, rng, neighbours, rel_tails):
    triples, index, samples = [], {}, []

    def add(h, r, t):
        k = (name(h).lower(), rel_text(r), name(t).lower())  # text, not raw key: two raw keys with one name gave two
        # identical gold rows, and cf edits only gold[0] (79/2,327 test claims, review 26/11/2026)
        if k not in index:
            index[k] = len(triples)
            triples.append({"head": name(h), "relation": rel_text(r), "tail": name(t)})
        return index[k]

    for text, label, h, r, t in claims:
        tails = list(dict.fromkeys(kg.get(h, {}).get(r, [])))
        tn = {name(x).lower() for x in tails}  # written text: two raw keys can share one name ("United_States" / '"United States"')
        if not tails or label != (t in tails) or (not label and name(t).lower() in tn):  # undecidable or contradicted: drop
            continue
        first = t if label else tails[0]
        gold = list(dict.fromkeys([add(h, r, first)] + [add(h, r, x) for x in tails if x != first][:7]))  # ponytail: 8 gold max
        for e in (h, t):  # same-entity distractors
            nb = [(e, rr, x) for rr, xs in kg.get(e, {}).items() if not rr.startswith("~") and rr != r for x in xs]
            for k in rng.sample(nb, min(neighbours, len(nb))):
                add(*k)
        if label:
            alts = [x for x in rel_tails.get(r, ()) if name(x).lower() not in tn]
            if not alts:
                continue
            cf = name(rng.choice(alts))
        else:
            cf = name(t)
        samples.append({"state": text, "question": QUESTION, "options": OPTIONS, "label": int(label),
                        "gold_idx": gold, "rule_present": True, "kind": "factkg_onehop", "cf_tail": cf})
    return triples, samples


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    a = sys.argv[3:]
    opt = lambda k, d: type(d)(a[a.index(k) + 1]) if k in a else d  # noqa: E731
    neighbours, max_train, seed = opt("--neighbours", 3), opt("--max_train", 12000), opt("--seed", 0)
    rng = random.Random(seed)
    train, dev = one_hop(load(src / "factkg_train.pickle")), one_hop(load(src / "factkg_dev.pickle"))
    print(f"one-hop claims: train {len(train)}, dev {len(dev)}", flush=True)
    kg = load(src / "dbpedia_2015_undirected_light.pickle")
    rels = {r for c in train + dev for r in [c[3]]}
    rel_tails = {}  # relation -> sample of real tails (for counterfactual swaps)
    for e, rs in kg.items():
        for r in rels & rs.keys():
            lst = rel_tails.setdefault(r, [])
            if len(lst) < 2000:
                lst.extend(rs[r][:2])
    rng.shuffle(train)
    rng.shuffle(dev)  # FactKG lists claims grouped by label: --limit must not see one label only
    out.mkdir(parents=True, exist_ok=True)
    for split, claims, cap in (("train", train, max_train), ("test", dev, 0)):
        triples, samples = build(claims[:cap] if cap else claims, kg, rng, neighbours, rel_tails)
        body = {"triples": triples, "samples": samples} if split == "train" else {"triples": triples, "splits": {"ood": samples}}
        (out / f"{split}.json").write_text(json.dumps(body), encoding="utf-8")
        pos = sum(s["label"] for s in samples)
        print(f"{split}: {len(samples)} samples ({pos} true), {len(triples)} triples", flush=True)
    # --grow 20000,50000: the same test claims with a larger KG (random real DBpedia triples appended after the test
    # triples, so gold_idx is unchanged; heads of test claims are excluded so no appended triple can decide a claim).
    grow = [int(x) for x in str(opt("--grow", "")).split(",") if x]
    if grow:
        used = {name(c[2]) for c in dev} | {name(c[4]) for c in dev}
        ents, grng, extra, seen = list(kg), random.Random(seed + 1), [], {(t["head"], t["relation"], t["tail"]) for t in triples}
        for _ in range(50 * max(grow)):  # bounded: a small KG may not have enough distinct triples
            if len(triples) + len(extra) >= max(grow):
                break
            e = grng.choice(ents)
            fw = [(r, x) for r, xs in kg[e].items() if not r.startswith("~") for x in xs]
            if not fw or name(e) in used:
                continue
            r, x = grng.choice(fw)
            t = {"head": name(e), "relation": rel_text(r), "tail": name(x)}
            if (t["head"], t["relation"], t["tail"]) not in seen:
                seen.add((t["head"], t["relation"], t["tail"]))
                extra.append(t)
        for n in grow:
            body = {"triples": triples + extra[: n - len(triples)], "splits": {"ood": samples}}
            (out / f"test_kg{n // 1000}k.json").write_text(json.dumps(body), encoding="utf-8")
            print(f"test_kg{n // 1000}k: {len(body['triples'])} triples", flush=True)


if __name__ == "__main__":
    main()
