"""
JefKV: Decision-Jef-0.1 with AtlasKV-style KG memory in its global-attention layers.

The decision path is Jef's own: one `noul` question per case, answer read at [DEC] by Jef's frozen
q_proj/o_proj heads, so Jef's calibration is kept. What is added:
  - AtlasKVModernBertAttention in global layers (default 3,6,...,21; ModernBERT has global attention
    every 3 layers, matching AtlasKV's K=3 optimum, AtlasKV Appendix B.3).
  - A retrieval-conditioned temperature: logits * softplus(w . f + b), where f = per-layer
    [lambda_kg, max triple probability]. When no triple matches, the model can flatten towards 50%.
    RAG modes get the bias only (a learned global temperature).

References: Decision-Jef-0.1 (BarraHome, 2026, MIT, `decision-jef` package); AtlasKV (Huang et al.,
ICLR 2026); mmBERT (Marone et al., 2025).
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from decision_jef import Decider, Question, Request
from decision_jef.pack import pack

from .attention import AtlasKVModernBertAttention

JEF = "BarraHome/Decision-Jef-0.1"
DEFAULT_INJECTION_LAYERS = (3, 6, 9, 12, 15, 18, 21)


class JefKVModel(nn.Module):
    def __init__(self, kg_embed_dim: int = 768, injection_layers=DEFAULT_INJECTION_LAYERS,
                 use_kv: bool = True, unfreeze_top: int = 0, max_len: int = 512, device: str = "cpu",
                 kg_topk: int = 16, freeze_kg_v: bool = False):
        super().__init__()
        decider = Decider.from_pretrained(JEF, device=device)
        self.jef = decider.model
        self.tok = decider.tok
        self.max_len = max_len
        self.jef.set_trainable(unfreeze_top)        # 0 = encoder frozen; n = top n layers trainable
        if unfreeze_top == 0:                       # set_trainable(0) still unfreezes final_norm
            self.jef.encoder.final_norm.requires_grad_(False)
        for p in list(self.jef.q_proj.parameters()) + list(self.jef.o_proj.parameters()) + [self.jef.logit_scale]:
            p.requires_grad_(False)                 # keep Jef's decision head (and its calibration)

        layers = self.jef.encoder.layers
        types = self.jef.encoder.config.layer_types
        self.kg_modules: Dict[int, AtlasKVModernBertAttention] = {}
        self.feat_k = kg_topk or 16
        # Backbone-encoded KG (kg_embed_dim == hidden size): layer-matched keys, see encode_kg_texts.
        self.layer_kv = kg_embed_dim == self.jef.encoder.config.hidden_size
        if use_kv:
            for j, i in enumerate(injection_layers):
                if types[i] != "full_attention":
                    raise ValueError(f"layer {i} is {types[i]}; inject only into global layers")
                sl = slice(j * kg_embed_dim, (j + 1) * kg_embed_dim) if self.layer_kv else None
                layers[i].attn = AtlasKVModernBertAttention(layers[i].attn, kg_embed_dim, sl).to(device)
                self.kg_modules[i] = layers[i].attn
            # Retrieve-then-attend: the first KV layer (best untrained retriever: OOD ACC@5 97.5% among
            # 2K triples) picks kg_topk triples per case; every KV layer attends only to those.
            route = {}
            for m in self.kg_modules.values():
                m.route = route
            if kg_topk:
                sel = next(iter(self.kg_modules.values()))
                sel.select_k = kg_topk
                # Its Q/K stay at identity when the KV is layer-matched: training them on the ~130-triple
                # train KGs made layer 3 a worse retriever on the 10K test KG (gate run 4: OOD ACC@5 97.5%
                # untrained -> 77.5% trained, alias 52.5% -> 50%). Its V and bias still train.
                if self.layer_kv:
                    sel.kg_q_proj.requires_grad_(False)
                    sel.kg_k_proj.requires_grad_(False)
            if freeze_kg_v:  # W_V^kg stays at its init (the layer's own W_V): collapsed runs carried the collapse in W_V^kg
                for m in self.kg_modules.values():
                    m.kg_v_proj.requires_grad_(False)
        self.temp = nn.Linear(2 * len(self.kg_modules), 1).to(device)
        nn.init.zeros_(self.temp.weight)
        nn.init.constant_(self.temp.bias, 0.5413)   # softplus(0.5413) = 1.0: start at Jef's own scale

    # ---------- KG ----------
    def set_kg(self, keys, vals, slot_mask, query_mask=None):
        for m in self.kg_modules.values():
            m.set_kg(keys, vals, slot_mask, query_mask)

    @torch.no_grad()
    def encode_kg_texts(self, texts: List[str], max_slots: int = 16):
        """Token-level KV: frozen Jef states of each triple sentence (KG detached), layer-matched: the
        input each KV layer's attention sees, concatenated over KV layers -> [n, S, L*D]. Final-layer
        keys did not match the case's layer-i queries (OOD ACC@5 10-57% vs 95-98% layer-matched)."""
        self.set_kg(None, None, None)
        dev = next(self.parameters()).device
        enc = self.tok(texts, padding=True, truncation=True, max_length=max_slots, return_tensors="pt").to(dev)
        was_training = self.jef.encoder.training
        self.jef.encoder.eval()  # memory must not depend on dropout state
        seen = {}
        hooks = [m.register_forward_pre_hook(
            lambda mod, a, kw, i=i: seen.__setitem__(i, kw["hidden_states"] if "hidden_states" in kw else a[0]),
            with_kwargs=True) for i, m in self.kg_modules.items()]
        try:
            h = self.jef.encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
        finally:
            for hk in hooks:
                hk.remove()
            self.jef.encoder.train(was_training)
        if self.layer_kv:
            h = torch.cat([seen[i] for i in self.kg_modules], dim=-1)
        return h, enc["attention_mask"].bool()

    # ---------- decisions ----------
    def batch(self, samples: List[Dict], contexts: Optional[List[str]] = None):
        reqs = []
        for i, s in enumerate(samples):
            state = s["state"] if not contexts else f"Policy context: {contexts[i]}\nCase: {s['state']}"
            q = Question("noul", s["question"], {"false": s["options"]["false"], "true": s["options"]["true"]})
            reqs.append(Request(state, {"q": q}))
        dev = next(self.parameters()).device
        return pack(reqs, self.tok, self.max_len).to(dev)

    def forward(self, packed, kg=None):
        """Returns (logits [B,2], per-layer {layer: [B,M]} retrieval logits (grad), per-layer post-softmax
        attention per triple (no grad, for ACC@k)). Both dicts are empty without a KG."""
        if kg is not None and self.kg_modules:
            keys, vals, slot_mask = kg
            self.set_kg(keys, vals, slot_mask, packed.attention_mask.bool())
        try:
            logits = self.jef(packed).float()
            feats, triple_logits, triple_attn = [], {}, {}
            for i, m in self.kg_modules.items():
                if m.triple_logits is None:
                    feats.append(torch.zeros(logits.shape[0], 2, device=logits.device))
                    continue
                triple_logits[i], triple_attn[i] = m.triple_logits, m.triple_attn
                # max-prob over the top-k scores only: a softmax over the whole KG depends on its size
                # (~130 triples in training vs 10K at eval) and shifted this feature between the two.
                tl = m.triple_logits.topk(min(self.feat_k, m.triple_logits.shape[-1]), dim=-1).values
                feats.append(torch.stack([m.lambda_kg, F.softmax(tl, -1).max(-1).values], -1))
            f = torch.cat(feats, -1) if feats else logits.new_zeros(logits.shape[0], 0)
            if kg is not None or not self.kg_modules:  # a KV model without a KG stays exactly stock Jef
                logits = logits * F.softplus(self.temp(f))
        finally:
            self.set_kg(None, None, None)
        return logits, triple_logits, triple_attn

    def trainable_state(self) -> Dict[str, torch.Tensor]:
        return {n: p.detach().cpu() for n, p in self.named_parameters() if p.requires_grad}

    def load_trainable(self, state: Dict[str, torch.Tensor]) -> None:
        missing = [n for n, p in self.named_parameters() if p.requires_grad and n not in state]
        unknown = set(state) - {n for n, _ in self.named_parameters()}
        if missing or unknown:
            raise KeyError(f"checkpoint mismatch: missing={missing[:5]} unknown={sorted(unknown)[:5]}")
        self.load_state_dict(state, strict=False)
