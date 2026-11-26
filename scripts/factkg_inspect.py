"""Summary statistics of the FactKG files (Kim et al., ACL 2023; CC BY-NC-SA 4.0: the data is never committed here).

    python scripts/factkg_inspect.py <folder with factkg_{train,dev,test}.pickle and dbpedia_2015_undirected_light.pickle> [--claims_only]

The files are pickles, so they are read with a restricted Unpickler that only allows plain containers, strings and
numbers: anything else in the file (a class, a function) raises instead of running code.
"""
import builtins
import collections
import pickle
import sys
from pathlib import Path

ALLOWED = {"dict", "list", "set", "frozenset", "tuple", "str", "int", "float", "bool"}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ("builtins", "__builtin__") and name in ALLOWED:  # __builtin__: protocol 0-2 (Python 2 names)
            return getattr(builtins, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def load(path: Path):
    with path.open("rb") as f:
        return SafeUnpickler(f).load()


def claims_summary(name: str, d: dict) -> None:
    labels = collections.Counter(str(v.get("Label")) for v in d.values())
    types = collections.Counter(t for v in d.values() for t in v.get("types", []))
    n_ent = collections.Counter(len(v.get("Entity_set", [])) for v in d.values())
    hops = collections.Counter(max((len(p) for paths in v.get("Evidence", {}).values() for p in paths), default=0) for v in d.values())
    print(f"{name}: {len(d)} claims | labels {dict(labels)}")
    print(f"   types {dict(types.most_common())}")
    print(f"   entities per claim {dict(sorted(n_ent.items()))} | longest evidence path {dict(sorted(hops.items()))}")
    print(f"   mean claim length {sum(len(c.split()) for c in d) / max(1, len(d)):.1f} words")


def main() -> None:
    root = Path(sys.argv[1])
    for split in ("train", "dev", "test"):
        p = root / f"factkg_{split}.pickle"
        if p.exists():
            claims_summary(split, load(p))
    p = root / "dbpedia_2015_undirected_light.pickle"
    if p.exists() and "--claims_only" not in sys.argv:  # the light KG needs several GB of RAM
        kg = load(p)
        rels = collections.Counter(r for e in kg.values() for r in e if not r.startswith("~"))
        n_trip = sum(len(t) for e in kg.values() for r, t in e.items() if not r.startswith("~"))
        print(f"light KG: {len(kg)} entities, {len(rels)} forward relations, {n_trip} forward triples")
        print(f"   top relations {rels.most_common(12)}")


if __name__ == "__main__":
    main()
