"""
Separated-softmax KG attention for ModernBERT global-attention layers (transformers >= 5).

References:
  - Separated softmax & W_Q^kg initialisation: Huang et al., "AtlasKV", ICLR 2026 (arXiv:2510.17934),
    Section 4.2 (Eq. 3-6), Appendix A.1, Appendix C.
  - Rectangular KB attention: Wang et al., "KBLaM", arXiv:2410.10450.
  - ModernBERT layer layout: Warner et al., ACL 2025 (arXiv:2412.13663).

KG memory is token-level: each triple contributes S slots (S=1 for a sentence encoder, up to ~16
token states when the triple sentence is encoded by the backbone itself). Slots are flattened to
M*S attention columns; `slot_mask` marks real slots.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb


KG_SOFTCAP = 50.0  # same value as Gemma 2's attention soft-cap
# Gated mixing: out = (1-g) * sequence attention + g * KG attention, g = KG_GATE_MAX * sigmoid(w) per head.
# The separated softmax let trained KG scores take 63-98% of the attention in layers 6-21 (gate run 5 probe),
# drowning the case; a bounded gate decouples "how much KG" from "which slot".
KG_GATE_MAX = 0.5
KG_GATE_INIT = 0.15


def _neg(dtype: torch.dtype) -> float:
    return torch.finfo(dtype).min


class AtlasKVModernBertAttention(nn.Module):
    """Wraps one ModernBertAttention. Only kg_q/k/v_proj are new; base_attn is left as is."""

    def __init__(self, base_attn: nn.Module, kg_embed_dim: int, kg_slice: Optional[slice] = None):
        """kg_slice: this layer's part of the last dim of layer-matched KG states (None = shared keys,
        e.g. a sentence encoder)."""
        super().__init__()
        self.base_attn = base_attn
        self.head_dim = base_attn.head_dim
        self.all_head_size = base_attn.Wo.in_features
        self.num_heads = self.all_head_size // self.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.kg_slice = kg_slice

        self.kg_q_proj = nn.Linear(self.all_head_size, self.all_head_size, bias=False)
        self.kg_k_proj = nn.Linear(kg_embed_dim, self.all_head_size, bias=False)
        self.kg_v_proj = nn.Linear(kg_embed_dim, self.all_head_size, bias=False)
        with torch.no_grad():
            d = self.all_head_size
            if kg_slice is None:
                # W_Q^kg starts as the frozen Q slice of the fused Wqkv (AtlasKV Appendix A.1).
                self.kg_q_proj.weight.copy_(base_attn.Wqkv.weight[:d])
            else:
                # Keys are this layer's own input states of the triple tokens. Identity Q/K start the
                # KG scores at token-state similarity, which already finds the entity named in the case
                # (measured, gate run 3); Jef's Wq/Wk and random K did not. V = the layer's own Wv:
                # a matched KG token contributes what it would if it were in the sequence.
                self.kg_q_proj.weight.copy_(torch.eye(d))
                self.kg_k_proj.weight.copy_(torch.eye(d))
                self.kg_v_proj.weight.copy_(base_attn.Wqkv.weight[2 * d:])
        p = KG_GATE_INIT / KG_GATE_MAX
        self.kg_gate = nn.Parameter(torch.full((self.num_heads,), math.log(p / (1 - p))))
        # Sharpness of the selecting layer's P(triple | case), added as log P to every token's KG scores: [DEC]
        # and the case's number put 80-94% of their KG attention on distractor triples without it (gate 5 probe).
        # The scores are z-scored over the top-k first: raw ColBERT scores average ~100 case tokens, so the
        # gold-vs-distractor gap is small and a raw log_softmax was nearly flat (gate-7 review).
        self.kg_prior = nn.Parameter(torch.tensor(3.0))

        # Retrieve-then-attend: the layer with select_k > 0 scores the whole KG and writes its top-k
        # triple ids per case to route["idx"]; later KV layers attend only to those (shared dict).
        self.select_k = 0
        self.route: dict = {}
        self.debug: Optional[dict] = None

        self.kg = None  # (keys [M,S,De], vals [M,S,De], slot_mask [M,S] bool, query_mask [B,N] bool)
        self.triple_logits: Optional[torch.Tensor] = None  # [B, M], keeps grad (retrieval loss); not-attended = finfo.min
        self.lambda_kg: Optional[torch.Tensor] = None      # [B], keeps grad (calibration features)
        self.triple_attn: Optional[torch.Tensor] = None    # [B, M], no grad (ACC@k metric)

    def set_kg(self, keys, vals, slot_mask, query_mask):
        self.kg = None if keys is None else (keys, vals, slot_mask, query_mask)
        self.triple_logits = self.lambda_kg = self.triple_attn = None
        self.route.pop("idx", None)
        self.route.pop("z", None)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
        if self.kg is None:  # exact pass-through: Jef zero-shot is unchanged
            return self.base_attn(hidden_states, position_embeddings=position_embeddings,
                                  attention_mask=attention_mask, **kwargs)

        keys, vals, slot_mask, query_mask = self.kg
        if self.kg_slice is not None:
            keys, vals = keys[..., self.kg_slice], vals[..., self.kg_slice]
        B, N, _ = hidden_states.shape
        M, S, _ = keys.shape
        H, Dh = self.num_heads, self.head_dim
        dtype = hidden_states.dtype

        # 1. Frozen sequence branch (same maths as ModernBERT eager attention).
        qkv = self.base_attn.Wqkv(hidden_states).view(B, N, 3, H, Dh)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        # Attention scores in fp32: with fp16 autocast, q.k overflowed (|q.k| = 64,418 vs fp16 max 65,504
        # in the first gate run) and produced NaN. autocast must be off here: it recasts matmul inputs
        # to fp16 even after .float().
        no_amp = torch.autocast(device_type=hidden_states.device.type, enabled=False)
        with no_amp:
            logits_seq = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                logits_seq = logits_seq.masked_fill(~attention_mask, _neg(logits_seq.dtype))
            else:
                logits_seq = logits_seq + attention_mask

        # 2. KG branch. No RoPE: triples are unordered. The whole KG is shared by the batch (Bk = 1);
        #    after selection each case has its own k triples (Bk = B).
        idx = None if self.select_k else self.route.get("idx")               # [B,k] triple ids
        if idx is not None:
            keys, vals, slot_mask = keys[idx], vals[idx], slot_mask[idx]     # [B,k,S,De], [B,k,S]
        else:
            keys, vals, slot_mask = keys[None], vals[None], slot_mask[None]  # [1,M,S,De], [1,M,S]
        Bk, Mk = keys.shape[:2]
        q_kg = self.kg_q_proj(hidden_states).view(B, N, H, Dh).transpose(1, 2)
        k_kg = self.kg_k_proj(keys.reshape(Bk, Mk * S, -1).to(dtype)).view(Bk, Mk * S, H, Dh).transpose(1, 2)
        v_kg = self.kg_v_proj(vals.reshape(Bk, Mk * S, -1).to(dtype)).view(Bk, Mk * S, H, Dh).transpose(1, 2)
        with no_amp:
            logits_kg = torch.matmul(q_kg.float(), k_kg.float().transpose(-1, -2)) * self.scale  # [B,H,N,Mk*S]
        # Soft-cap (Gemma 2 style): the retrieval loss otherwise drives KG scores to ~8,000, so KG
        # attention swamps the sequence and the decision collapses to 50/50 (first gate run).
        logits_kg = KG_SOFTCAP * torch.tanh(logits_kg / KG_SOFTCAP)
        # (KG size no longer changes the KG's share: the gate sets it, the softmax below is KG-only.)
        sm = slot_mask.reshape(Bk, 1, 1, Mk * S)
        logits_kg = logits_kg.masked_fill(~sm, _neg(torch.float32))

        # ColBERT-style triple score: each query token picks its best slot of the triple (max), then
        # average over tokens. Averaging tokens first (old code) scored triples against the whole case,
        # i.e. by relation type only: the 2-3 entity tokens were drowned (ACC@5 ~0 in gate run 3).
        # logsumexp over slots instead of max: 37% vs 100% OOD ACC@5 at init (generic slots add up).
        qm = query_mask.to(torch.float32)                                     # [B,N]
        denom = qm.sum(-1).clamp(min=1.0)
        live = slot_mask.any(-1)                                              # [Bk,Mk]
        tok = logits_kg.mean(1).view(B, N, Mk, S).amax(-1)                   # [B,N,Mk]
        tok = tok.masked_fill(~live.unsqueeze(1), 0.0)
        trip = ((tok * qm.unsqueeze(-1)).sum(1) / denom.unsqueeze(-1)).masked_fill(~live, _neg(torch.float32))  # [B,Mk]

        if self.select_k:
            # Attention spread over all ~161K slots of the 10K test KG let almost no value of the right
            # triple through (gate run 4: retrieval 95%, decisions 64% vs trained RAG 78%), so every
            # KV layer, this one included, attends only to the top-k triples of this case.
            nat = trip.detach().topk(min(self.select_k, M), dim=-1)          # natural routing [B,k]
            top = self.route.get("force")  # eval-only oracle routing (scripts/eval.py): given triple ids [B,k]
            raw = nat.values if top is None else trip.detach().gather(1, top)
            if top is None:
                top = nat.indices
            else:  # the first force_gold ids are gold: score them as the natural top-1 (an ideal selector ranks them
                raw = raw.clone()  # first), else their low layer-3 score gives a negative prior (review 26/11/2026)
                raw[:, : self.route.get("force_gold", 0)] = nat.values[:, :1]
            # z over the natural top-k's statistics, so forcing gold in does not shift the other triples' priors
            if raw.shape[1] > 1:
                z = (raw - nat.values.mean(-1, keepdim=True)) / nat.values.std(-1, keepdim=True).clamp(min=1e-4)
            else:
                z = raw * 0
            self.route.update(idx=top, z=z)
            keep = torch.zeros(B, M, dtype=torch.bool, device=trip.device).scatter_(1, top, True)
            keep = keep.unsqueeze(-1) & slot_mask                             # [B,M,S]
            logits_kg = logits_kg.masked_fill(~keep.view(B, 1, 1, M * S), _neg(torch.float32))
            z_full = z.new_zeros(B, M).scatter(1, top, z)                     # non-selected are masked anyway
        else:
            z_full = self.route.get("z")                                      # [B,k] or None (no routing)
        if z_full is not None:  # every token leans towards the triples the case is about: + log P(triple)
            prior = self.kg_prior.float() * z_full  # = log P(triple) up to a per-case constant (softmax-invariant)
            pr = prior.unsqueeze(-1).expand(B, prior.shape[1], S).reshape(B, 1, 1, -1)
            logits_kg = torch.where(sm, logits_kg + pr, logits_kg)

        # 3. Gated mixing of the frozen sequence attention and a KG-only softmax, fp32 softmaxes.
        attn_seq = torch.softmax(logits_seq, dim=-1).to(dtype)
        attn_kg = torch.softmax(logits_kg, dim=-1)                            # [B,H,N,Mk*S], sums to 1
        has = sm.flatten(1).any(-1).view(Bk, 1, 1, 1)                         # no real slot -> no KG at all
        g = KG_GATE_MAX * torch.sigmoid(self.kg_gate.float()).view(1, H, 1, 1) * has
        # No cap on ||v_kg||: capping the KG read-out at the sequence's RMS (gate 8) cost ~11 pts OOD, since
        # good runs read the KG ~10x louder than the sequence before g. scripts/probe_drown.py measures it.
        y_seq, y_kg = (1 - g).to(dtype) * torch.matmul(attn_seq, v), g.to(dtype) * torch.matmul(attn_kg.to(dtype), v_kg)
        out = y_seq + y_kg
        if self.debug is not None:  # diagnostics only (probes); None in training / eval
            self.debug.update(attn_kg=(g * attn_kg).detach().float(), idx=idx, Mk=Mk, S=S,
                              y_seq=y_seq.detach().float(), y_kg=y_kg.detach().float())  # [B,H,N,Dh]

        # 4. Diagnostics, indexed over the whole KG [B,M]: retrieval score (grad), KG weight (grad),
        #    KG attention per triple (no grad, AtlasKV-style ACC@k).
        self.lambda_kg = g.mean(1).view(-1).expand(B)                         # [B]
        with torch.no_grad():
            a = (attn_kg.float().mean(1) * qm.unsqueeze(-1)).sum(1) / denom.unsqueeze(-1)
            a = a.view(B, Mk, S).sum(-1)                                      # [B,Mk]
        if idx is None:
            self.triple_logits, self.triple_attn = trip, a
        else:  # scatter the k candidates back to KG positions; the rest were not attended
            self.triple_logits = trip.new_full((B, M), _neg(torch.float32)).scatter(1, idx, trip)
            self.triple_attn = a.new_zeros(B, M).scatter(1, idx, a)

        out = out.transpose(1, 2).reshape(B, N, self.all_head_size)
        return self.base_attn.out_drop(self.base_attn.Wo(out)), None
