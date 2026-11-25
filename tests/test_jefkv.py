"""
Self-check (CPU, ~1 min after weights are cached):  PYTHONPATH=. python tests/test_jefkv.py

1. With no KG attached, JefKV == stock Decision-Jef (wrapper is an exact pass-through).
2. With a KG whose slots are all masked, the re-implemented sequence branch (RoPE, Jef's segment
   masks, separated softmax) reproduces stock Jef: lambda_kg must be ~0 and logits unchanged.
3. With a real KG, the output moves and lambda_kg > 0 (the KG branch is live).
"""

import torch
from decision_jef import Decider

from jefkv.model import JefKVModel

SAMPLES = [
    {"state": "The customer is a Gold tier member. Customer requested a refund for item Velorakin 12 days after delivery.",
     "question": "Is this refund request eligible under the item's return policy?",
     "options": {"false": "The request exceeds the window.", "true": "The request is within the window."}},
    {"state": "Passenger checked in a bag for fare code Mabe weighing 31 kg.",
     "question": "Does the bag qualify for complimentary check-in?",
     "options": {"false": "Overweight fee.", "true": "Within allowance."}},
]


def main():
    torch.manual_seed(0)
    stock = Decider.from_pretrained("BarraHome/Decision-Jef-0.1", device="cpu").model.eval()
    model = JefKVModel(device="cpu").eval()

    with torch.no_grad():
        packed = model.batch(SAMPLES)
        ref = stock(packed).float()

        out, _, _ = model(packed, None)
        assert torch.allclose(out, ref, atol=1e-4), (out, ref)

        h, m = model.encode_kg_texts(["Velorakin | return window in days: 30", "Mabe | bag kg: 20"])
        masked, _, _ = model(packed, (h, h, torch.zeros_like(m)))
        assert torch.allclose(masked, ref, atol=2e-3), (masked, ref)

        model.set_kg(h, h, torch.zeros_like(m), packed.attention_mask.bool())
        stock_like = model.jef(packed).float()
        lam = max(mod.lambda_kg.max().item() for mod in model.kg_modules.values())
        model.set_kg(None, None, None)
        assert lam < 1e-6, lam
        assert torch.allclose(stock_like, ref, atol=2e-3)

        live, tl, _ = model(packed, (h, h, m))
        assert not torch.allclose(live, ref, atol=1e-3), "KG branch had no effect"
        assert set(tl) == set(model.kg_modules) and all(x.shape == (2, 2) for x in tl.values())

        # 4. Under fp16 autocast (what the T4 runs) the KG path must still run and stay finite.
        #    CPU-only tests missed this once: autocast recast the "fp32" scores to fp16 and overflowed.
        with torch.autocast(device_type="cpu", dtype=torch.float16):
            amp_out, amp_tl, _ = model(packed, (h, h, m))
        assert amp_out.isfinite().all() and all(x.isfinite().all() for x in amp_tl.values())

        # 5. Top-k routing: with k=1, every later KV layer attends only to the triple the first layer
        #    picked for that case (exactly one attended triple per case, the same one).
        first, *later = model.kg_modules.values()
        k0, first.select_k = first.select_k, 1
        _, tl, _ = model(packed, (h, h, m))
        first.select_k = k0
        pick = list(tl.values())[0].argmax(-1)
        for x in list(tl.values())[1:]:
            seen = x > -1e30
            assert (seen.sum(-1) == 1).all() and (seen.float().argmax(-1) == pick).all(), x
    print("ok: pass-through exact, masked-KG == stock Jef, live KG changes output, fp16 autocast finite, top-k routing")


if __name__ == "__main__":
    main()
