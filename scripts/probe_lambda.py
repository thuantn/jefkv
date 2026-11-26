"""
Diagnose KG attention mass: for a few OOD cases compare no KG / train-size sub-KG (gold + 120) / full test KG.
Prints Jef raw margin, temperature factor, lambda_kg per layer (= mean KG gate g, at most 0.5 since the gated mixing; it was the joint-softmax KG mass before).
  PYTHONPATH=. python scripts/probe_lambda.py <run dir with adapter.pt | -> data/processed/test.json 12
"""
import json, random, sys
import torch
import torch.nn.functional as F
from jefkv.model import JefKVModel
from jefkv.memory import KGMemory

run, test_path, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
model = JefKVModel(device="cpu").eval()
if run != "-":  # "-" = freshly initialised heads
    model.load_trainable(torch.load(f"{run}/adapter.pt", map_location="cpu", weights_only=True)["state"])
print("temp weight:", [round(x, 3) for x in model.temp.weight[0].tolist()], "bias", round(model.temp.bias.item(), 3))

test = json.loads(open(test_path, encoding="utf-8").read())
triples = test["triples"]
samples = test["splits"]["ood"][:n]
full = KGMemory(triples, lambda t, max_len=16: model.encode_kg_texts(t, max_len), "cpu", torch.float32)
rng = random.Random(0)


@torch.no_grad()
def probe(s, kg):
    packed = model.batch([s])
    if kg is not None:
        model.set_kg(*kg, packed.attention_mask.bool())
    raw = model.jef(packed).float()
    feats = []
    for m in model.kg_modules.values():
        if m.triple_logits is None:
            feats.append(torch.zeros(1, 2))
        else:
            tl = m.triple_logits.topk(min(model.feat_k, m.triple_logits.shape[-1]), dim=-1).values  # as model.py
            feats.append(torch.stack([m.lambda_kg, F.softmax(tl, -1).max(-1).values], -1))
    lam = [round(m.lambda_kg.item(), 3) if m.lambda_kg is not None else 0 for m in model.kg_modules.values()]
    factor = F.softplus(model.temp(torch.cat(feats, -1))).item()
    model.set_kg(None, None, None)
    margin = (raw[0, 1] - raw[0, 0]).item()
    return margin, factor, lam, (margin * factor > 0) == bool(s["label"])


for name in ("none", "sub", "full"):
    rows = []
    for s in samples:
        if name == "none":
            kg = None
        elif name == "full":
            kg = (full.keys, full.vals, full.slot_mask)
        else:
            ids = sorted(set(s["gold_idx"]) | set(rng.sample(range(len(triples)), 120)))
            sub = full.subset(torch.tensor(ids))
            kg = sub
        rows.append(probe(s, kg))
    m = sum(abs(r[0]) for r in rows) / len(rows)
    f = sum(r[1] for r in rows) / len(rows)
    acc = sum(r[3] for r in rows) / len(rows)
    lam = [round(sum(r[2][i] for r in rows) / len(rows), 3) for i in range(len(rows[0][2]))]
    print(f"{name:5} mean|raw margin|={m:.3f} temp factor={f:.3f} acc={acc:.2f} lambda_kg per layer={lam}")
