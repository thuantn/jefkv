"""
Synthetic rule-verification benchmark for JefKV.

Every case asks one yes/no question ("is this request within policy?") whose answer depends on a
numeric limit stored only in the knowledge graph. Splits:

  train.json   2K entities + tier rules, 15K samples: mix of plain / multi-rule / alias / missing-rule.
  test.json    10K unseen entities + tier rules (one KG shared by all test splits) and four sample sets:
                 ood    single rule, canonical entity name
                 multi  decision needs two rules (entity limit + customer-tier bonus)
                 alias  entity written in a different surface form than in the KG
                 trap   entity has no rule in the KG (ideal confidence ~50%)

Design notes (fixes over v0):
  - Entity names are pronounceable pseudo-words, not numeric IDs, and missing-rule entities are drawn
    from the same generator, so nothing in the text reveals that a rule is missing.
  - Train and test entities, and the missing-rule names, are disjoint.
"""

import argparse
import json
import random
from pathlib import Path

DOMAINS = [
    {
        "relation": "return window in days",
        "state_tpl": "Customer requested a refund for item {entity} {val_obs} days after delivery.",
        "question": "Is this refund request eligible under the item's return policy?",
        "opt_true": "The request falls within the allowed return window and can be approved.",
        "opt_false": "The request exceeds the allowed return window and must be rejected.",
        "rule_range": (7, 60),
    },
    {
        "relation": "auto-approval expense limit in USD",
        "state_tpl": "Employee submitted an expense claim for category {entity} totaling ${val_obs}.",
        "question": "Can this expense claim be auto-approved without director sign-off?",
        "opt_true": "The claimed amount is at or below the auto-approval limit.",
        "opt_false": "The claimed amount exceeds the auto-approval limit and needs escalation.",
        "rule_range": (50, 2000),
    },
    {
        "relation": "warranty coverage period in months",
        "state_tpl": "Buyer reported a hardware fault for device {entity} after {val_obs} months of use.",
        "question": "Is the device still covered under standard warranty?",
        "opt_true": "The usage duration is within the warranty coverage period.",
        "opt_false": "The warranty coverage period has expired.",
        "rule_range": (6, 36),
    },
    {
        "relation": "maximum complimentary baggage weight in kg",
        "state_tpl": "Passenger checked in a bag for fare code {entity} weighing {val_obs} kg.",
        "question": "Does the bag qualify for complimentary check-in without overweight fees?",
        "opt_true": "The bag weight is within the complimentary baggage allowance.",
        "opt_false": "The bag exceeds the complimentary limit and incurs an overweight fee.",
        "rule_range": (15, 40),
    },
]

SYLLABLES = (
    "ba be bo da de do fa fe fi ga go ka ke ki ko la le li lo lu ma me mi mo na ne ni no "
    "pa pe po ra re ri ro sa se si so ta te ti to va ve vi vo za ze zo"
).split()


def name_pool(n: int, rng: random.Random, exclude=frozenset()):
    """n unique pseudo-word names, each kept as its syllable list (needed for alias variants).
    `exclude`: lower-case names that must not appear (e.g. every name of another benchmark's files)."""
    seen, out = set(exclude), []
    while len(out) < n:
        syl = [rng.choice(SYLLABLES) for _ in range(rng.choice((3, 3, 4)))]
        key = "".join(syl)
        if key not in seen:
            seen.add(key)
            out.append(syl)
    return out


def canonical(syl):
    return "".join(syl).capitalize()


def alias(syl, rng: random.Random):
    """Same entity, different surface form (what a human might type)."""
    return rng.choice([
        lambda s: "".join(s).lower(),
        lambda s: " ".join(x.capitalize() for x in s),
        lambda s: "-".join(s).upper(),
    ])(syl)


def build_kg(entity_syls, tier_names, rng: random.Random):
    """One rule per entity, plus one bonus rule per (tier, domain)."""
    triples, entities, tiers = [], [], {}
    for syl in entity_syls:
        dom_id = rng.randrange(len(DOMAINS))
        limit = rng.randint(*DOMAINS[dom_id]["rule_range"])
        entities.append({"syl": syl, "dom": dom_id, "limit": limit, "idx": len(triples)})
        triples.append({"head": canonical(syl), "relation": DOMAINS[dom_id]["relation"], "tail": str(limit)})
    for t in tier_names:
        for dom_id, dom in enumerate(DOMAINS):
            hi = dom["rule_range"][1]
            bonus = rng.randint(2, max(3, hi // 4))
            tiers[(t, dom_id)] = (bonus, len(triples))
            triples.append({"head": f"{t} tier", "relation": f"bonus {dom['relation']}", "tail": str(bonus)})
    return triples, entities, tiers


def make_sample(kind, ent, tiers, tier_names, missing_syl, rng: random.Random):
    dom = DOMAINS[ent["dom"]]
    limit, gold = ent["limit"], [ent["idx"]]
    is_valid = rng.random() < 0.5
    prefix = ""

    if kind == "multi":
        tier = rng.choice(tier_names)
        bonus, t_idx = tiers[(tier, ent["dom"])]
        # The observed value always exceeds the base limit, so the tier rule decides the label.
        val_obs = rng.randint(limit + 1, limit + bonus) if is_valid else limit + bonus + rng.randint(1, max(2, limit // 3))
        prefix = f"The customer is a {tier} tier member. "
        gold = [ent["idx"], t_idx]
    else:
        delta = rng.randint(1, max(2, limit // 3))
        val_obs = max(1, limit - delta) if is_valid else limit + delta

    if kind == "missing":
        surface, gold = canonical(missing_syl), []
    elif kind == "alias":
        surface = alias(ent["syl"], rng)
    else:
        surface = canonical(ent["syl"])

    return {
        "state": prefix + dom["state_tpl"].format(entity=surface, val_obs=val_obs),
        "question": dom["question"],
        "options": {"false": dom["opt_false"], "true": dom["opt_true"]},
        "label": int(is_valid),
        "gold_idx": gold,
        "rule_present": kind != "missing",
        "kind": kind,
    }


def sample_set(n, mix, entities, tiers, tier_names, missing_pool, rng):
    kinds, weights = zip(*mix.items())
    out = []
    for _ in range(n):
        kind = rng.choices(kinds, weights)[0]
        out.append(make_sample(kind, rng.choice(entities), tiers, tier_names, rng.choice(missing_pool), rng))
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate JefKV rule-verification benchmark splits.")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument("--quick", action="store_true", help="Tiny splits for smoke testing.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="benchmark JSON files whose names must not be reused (fresh test set: pass the train file)")
    args = parser.parse_args()
    exclude = set()
    for f in args.exclude:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        exclude |= {t["head"].lower() for t in d["triples"]} | {t["tail"].lower() for t in d["triples"]}

    if args.quick:
        n_train_ent, n_test_ent, n_train, n_test = 300, 1000, 1200, 100
    else:
        n_train_ent, n_test_ent, n_train, n_test = 2000, 10000, 15000, 500
    n_tiers, n_missing = 20, 2000

    rng = random.Random(args.seed)
    pool = name_pool(n_train_ent + n_test_ent + 2 * n_tiers + n_missing, rng, exclude)
    train_syl, pool = pool[:n_train_ent], pool[n_train_ent:]
    test_syl, pool = pool[:n_test_ent], pool[n_test_ent:]
    train_tiers = [canonical(s) for s in pool[:n_tiers]]
    test_tiers = [canonical(s) for s in pool[n_tiers:2 * n_tiers]]
    missing_pool = pool[2 * n_tiers:]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    triples, ents, tiers = build_kg(train_syl, train_tiers, rng)
    train = {
        "triples": triples,
        "samples": sample_set(n_train, {"plain": 0.35, "multi": 0.3, "alias": 0.2, "missing": 0.15},
                              ents, tiers, train_tiers, missing_pool[: n_missing // 2], rng),
    }

    triples, ents, tiers = build_kg(test_syl, test_tiers, rng)
    test_missing = missing_pool[n_missing // 2:]
    test = {
        "triples": triples,
        "splits": {
            name: sample_set(n_test if name != "ood" else 2 * n_test, {kind: 1.0}, ents, tiers, test_tiers, test_missing, rng)
            for name, kind in [("ood", "plain"), ("multi", "multi"), ("alias", "alias"), ("trap", "missing")]
        },
    }

    for name, payload in [("train", train), ("test", test)]:
        target = out / f"{name}.json"
        target.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
        n = len(payload["samples"]) if "samples" in payload else {k: len(v) for k, v in payload["splits"].items()}
        print(f"Wrote {target} ({len(payload['triples'])} triples, samples: {n})")


if __name__ == "__main__":
    main()
