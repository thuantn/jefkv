"""Does the KG branch drown the sequence? Per KV layer, at the [DEC] token: ||y_kg|| / ||y_seq|| and the
mean pairwise cosine of y_kg across cases (≈1 = same vector for every case = no case information).
  PYTHONPATH=. python scripts/probe_drown.py <adapter.pt> <test.json> <n> <n_distractors>"""
import json, random, sys
import torch
import torch.nn.functional as F
from jefkv.model import JefKVModel
from jefkv.memory import KGMemory

adapter, test_p, n, ndist = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
model = JefKVModel(device="cpu").eval()
model.load_trainable(torch.load(adapter, map_location="cpu", weights_only=True)["state"])
te = json.load(open(test_p, encoding="utf-8"))
T = te["triples"]
ss = [s for s in te["splits"]["ood"] if s["gold_idx"]][:n]
gold = sorted({g for s in ss for g in s["gold_idx"]})
rng = random.Random(0)
ids = gold + rng.sample([i for i in range(len(T)) if i not in set(gold)], ndist)
mem = KGMemory([T[i] for i in ids], lambda t, max_len=16: model.encode_kg_texts(t, max_len), "cpu", torch.float32)
kg = (mem.keys, mem.vals, mem.slot_mask)
for m in model.kg_modules.values():
    m.debug = {}

acc = {i: {"ratio": [], "kg": [], "seq": []} for i in model.kg_modules}
with torch.no_grad():
    for s in ss:
        packed = model.batch([s])
        model(packed, kg)
        d = packed.dec_pos[0] if hasattr(packed, "dec_pos") else 0
        d = int(d[0]) if torch.is_tensor(d) and d.dim() else int(d)
        for i, m in model.kg_modules.items():
            ys, yk = m.debug["y_seq"][0, :, d].flatten(), m.debug["y_kg"][0, :, d].flatten()
            acc[i]["ratio"].append((yk.norm() / ys.norm().clamp(min=1e-8)).item())
            acc[i]["kg"].append(yk); acc[i]["seq"].append(ys)
for i, a in acc.items():
    def cos(xs):
        x = F.normalize(torch.stack(xs), dim=-1); c = x @ x.T
        return ((c.sum() - c.trace()) / (len(xs) * (len(xs) - 1))).item()
    r = torch.tensor(a["ratio"])
    print(f"layer {i:>2}: |y_kg|/|y_seq| median {r.median():.2f} max {r.max():.2f} | cos(y_kg across cases) {cos(a['kg']):.2f} | cos(y_seq) {cos(a['seq']):.2f}")
