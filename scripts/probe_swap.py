"""Which trained component carries a 50/50 collapse? Load adapter A, overwrite one group of parameters with
adapter B's, and measure mean |logit margin| and accuracy on n OOD cases (KG = gold + distractors).
Also prints, per encoder layer, the mean pairwise cosine of the [DEC] hidden state across cases.
  PYTHONPATH=. python scripts/probe_swap.py <good_adapter.pt> <collapsed_adapter.pt> <test.json> [n] [n_distractors]"""
import json, random, sys
import torch
import torch.nn.functional as F
from jefkv.model import JefKVModel
from jefkv.memory import KGMemory

good_p, bad_p, test_p = sys.argv[1:4]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 40
ndist = int(sys.argv[5]) if len(sys.argv) > 5 else 300
good = torch.load(good_p, map_location="cpu", weights_only=True)["state"]
bad = torch.load(bad_p, map_location="cpu", weights_only=True)["state"]
model = JefKVModel(device="cpu").eval()
te = json.load(open(test_p, encoding="utf-8"))
T = te["triples"]
ss = [s for s in te["splits"]["ood"] if s["gold_idx"]][:n]
gold = sorted({g for s in ss for g in s["gold_idx"]})
rng = random.Random(0)
ids = gold + rng.sample([i for i in range(len(T)) if i not in set(gold)], ndist)
mem = KGMemory([T[i] for i in ids], lambda t, max_len=16: model.encode_kg_texts(t, max_len), "cpu", torch.float32)
kg = (mem.keys, mem.vals, mem.slot_mask)


def layer_of(name):
    return int(name.split("layers.")[1].split(".")[0]) if "layers." in name else None


GROUPS = {
    "kg_v": lambda k: ".kg_v_proj." in k,
    "kg_q+kg_k": lambda k: ".kg_q_proj." in k or ".kg_k_proj." in k,
    "gate+prior": lambda k: k.endswith("kg_gate") or k.endswith("kg_prior"),
    "temp": lambda k: k.startswith("temp."),
    "layers 3-6 (all)": lambda k: layer_of(k) is not None and layer_of(k) <= 6,
    "layers 9-21 (all)": lambda k: layer_of(k) is not None and layer_of(k) >= 9,
}


@torch.no_grad()
def score(state, hidden=False):
    model.load_trainable(state)
    marg, acc, dec_states = [], 0, {}
    for s in ss:
        packed = model.batch([s])
        hs = []
        hooks = [l.register_forward_hook(lambda m, i, o, hs=hs: hs.append((o[0] if isinstance(o, tuple) else o).detach()))
                 for l in model.jef.encoder.layers] if hidden else []
        try:
            lg = model(packed, kg)[0][0]
        finally:
            for h in hooks:
                h.remove()
        d = (lg[1] - lg[0]).item()
        marg.append(abs(d)); acc += int((d > 0) == bool(s["label"]))
        if hidden:
            p = int(packed.dec_pos[0]) if torch.is_tensor(packed.dec_pos) else int(packed.dec_pos[0])
            for li, h in enumerate(hs):
                dec_states.setdefault(li, []).append(h[0, p].float())
    out = (sum(marg) / len(marg), 100 * acc / len(ss))
    if hidden:
        cos = {}
        for li, xs in dec_states.items():
            x = F.normalize(torch.stack(xs), dim=-1); c = x @ x.T
            cos[li] = ((c.sum() - c.trace()) / (len(xs) * (len(xs) - 1))).item()
        return out, cos
    return out


(gm, ga), gcos = score(good, hidden=True)
(bm, ba), bcos = score(bad, hidden=True)
print(f"good       margin {gm:.2f} acc {ga:.0f}")
print(f"collapsed  margin {bm:.2f} acc {ba:.0f}")
print("cos([DEC]) across cases per layer, good | collapsed:")
print("  " + " ".join(f"{li}:{gcos[li]:.2f}|{bcos[li]:.2f}" for li in sorted(gcos)))
for g, sel in GROUPS.items():
    keys = [k for k in good if sel(k)]
    m1, a1 = score({**bad, **{k: good[k] for k in keys}})   # collapsed adapter, this group from the good one
    m2, a2 = score({**good, **{k: bad[k] for k in keys}})   # good adapter, this group from the collapsed one
    print(f"{g:18s} ({len(keys):2d} tensors)  collapsed+good-{g}: margin {m1:.2f} acc {a1:.0f} | good+collapsed-{g}: margin {m2:.2f} acc {a2:.0f}")
