"""
VRAM-resident KG memory (KG2KV) plus the retrievers used by the RAG baselines.

References:
  - KG2KV triple representation: Huang et al., "AtlasKV", ICLR 2026, Section 4.1 & Appendix B.2.
  - RandomKV diagnostic: AtlasKV Appendix E.
  - Sentence encoder `all-MiniLM-L6-v2`: Reimers & Gurevych 2019; Wang et al., NeurIPS 2020.
  - BM25: Robertson & Zaragoza 2009 (via `rank_bm25`).
"""

import re
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

MINILM = "sentence-transformers/all-MiniLM-L6-v2"


def triple_text(t: Dict[str, str]) -> str:
    return f"The {t['relation']} of {t['head']} is {t['tail']}."


def kv_text(t: Dict[str, str]) -> str:
    # Token-level mode: one short sentence, every token is a memory slot (used as key and value).
    return f"{t['head']} | {t['relation']}: {t['tail']}"


def atlaskv_key(t: Dict[str, str]) -> str:
    # AtlasKV Sec 4.1 / App B.2: key "the <relation> of <head>", value = the tail entity only.
    return f"the {t['relation']} of {t['head']}"


def _encode_all(encode, texts, device, dtype, batch_size):
    states, masks = [], []
    for i in range(0, len(texts), batch_size):
        s, m = encode(texts[i:i + batch_size])
        states.append(s.to(device=device, dtype=dtype))
        masks.append(m.to(device))
    S = max(s.shape[1] for s in states)
    keys = torch.cat([F.pad(s, (0, 0, 0, S - s.shape[1])) for s in states])
    mask = torch.cat([F.pad(m, (0, S - m.shape[1]), value=False) for m in masks])
    return keys, mask


class KGMemory:
    """
    keys/vals [M, S, De] and slot_mask [M, S] for M triples with up to S slots each.
    Built from `encode(texts) -> (states [n, S, De], mask [n, S])`, so the same class serves the
    MiniLM sentence encoder (S=1) and the backbone token encoder (S<=max_slots).

    atlaskv_kv=True: AtlasKV's KG2KV (key and value encoded from different strings, S must be 1).
    atlaskv_kv=False: one sentence per triple; its token states are both keys and values.
    """

    def __init__(self, triples: List[Dict[str, str]], encode: Callable, device: str,
                 dtype: torch.dtype, atlaskv_kv: bool = False, batch_size: int = 64):
        self.triples = triples
        self.encode = encode  # reused to embed queries for cluster pruning
        self.atlaskv = atlaskv_kv
        self.perm = None
        if atlaskv_kv:
            self.keys, self.slot_mask = _encode_all(encode, [atlaskv_key(t) for t in triples], device, dtype, batch_size)
            self.vals, _ = _encode_all(encode, [t["tail"] for t in triples], device, dtype, batch_size)
            assert self.keys.shape[1] == 1, "AtlasKV-style KV needs one slot per triple"
        else:
            self.keys, self.slot_mask = _encode_all(encode, [kv_text(t) for t in triples], device, dtype, batch_size)
            self.vals = self.keys
        self.device = device
        self.embed_dim = self.keys.shape[-1]
        self.centroids: Optional[torch.Tensor] = None
        self.members: Optional[List[torch.Tensor]] = None

    def __len__(self):
        return self.keys.shape[0]

    def shuffle_values(self, seed: int) -> None:
        """RandomKV sanity check: pair every key with another triple's value (AtlasKV Appendix E)."""
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(self), generator=g).to(self.device)
        self.perm = perm  # vals[j] is the value of triple perm[j]
        self.vals = self.vals[perm]
        # A slot is usable only if both its key and its (shuffled) value exist.
        self.slot_mask = self.slot_mask & self.slot_mask[perm]

    def subset(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.keys[idx], self.vals[idx], self.slot_mask[idx]

    def pooled(self) -> torch.Tensor:
        m = self.slot_mask.unsqueeze(-1).float()
        return F.normalize((self.keys.float() * m).sum(1) / m.sum(1).clamp(min=1.0), dim=-1)

    @torch.no_grad()
    def build_clusters(self, num_clusters: int = 128, n_iters: int = 12, seed: int = 0) -> None:
        """Cosine K-Means on pooled keys, entirely on the device."""
        x = self.pooled()
        k = min(num_clusters, x.shape[0])
        g = torch.Generator().manual_seed(seed)
        c = x[torch.randperm(x.shape[0], generator=g)[:k].to(x.device)]
        for _ in range(n_iters):
            assign = (x @ c.T).argmax(-1)
            new_c = torch.zeros_like(c).index_add_(0, assign, x)
            counts = torch.bincount(assign, minlength=k).unsqueeze(-1)
            c = torch.where(counts > 0, F.normalize(new_c, dim=-1), c)
        self.centroids = c
        self.members = [(assign == i).nonzero(as_tuple=True)[0] for i in range(k)]

    @torch.no_grad()
    def prune(self, query_vec: torch.Tensor, top_clusters: int = 8) -> torch.Tensor:
        """Candidate triple indices from the top clusters for one pooled query vector."""
        top = (F.normalize(query_vec.float().reshape(-1), dim=-1) @ self.centroids.T).topk(min(top_clusters, len(self.members))).indices
        return torch.cat([self.members[i] for i in top.tolist()])


def minilm_encoder(device: str):
    """encode(texts) -> ([n,1,384], [n,1]) with a frozen sentence encoder."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MINILM, device=device)

    @torch.no_grad()
    def encode(texts, max_len=None):  # max_len: same signature as the backbone encoder; unused
        e = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        return e.unsqueeze(1), torch.ones(e.shape[0], 1, dtype=torch.bool, device=e.device)

    encode.dim = model.get_sentence_embedding_dimension()
    return encode


class DenseRetriever:
    """Standard RAG retriever: MiniLM cosine over triple sentences."""

    def __init__(self, triples, device: str):
        from sentence_transformers import SentenceTransformer

        self.triples = triples
        self.model = SentenceTransformer(MINILM, device=device)
        self.emb = self.model.encode([triple_text(t) for t in triples], convert_to_tensor=True,
                                     normalize_embeddings=True, show_progress_bar=False)

    @torch.no_grad()
    def search(self, query: str, k: int) -> Tuple[List[int], List[float]]:
        q = self.model.encode([query], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        s, i = (q @ self.emb.T)[0].topk(min(k, len(self.triples)))
        return i.tolist(), s.tolist()


_TOK = re.compile(r"[a-z0-9]+")


class BM25Retriever:
    def __init__(self, triples):
        from rank_bm25 import BM25Okapi

        self.triples = triples
        self.docs = [_TOK.findall(triple_text(t).lower()) for t in triples]  # kept: eval re-indexes an edited row
        self.bm25 = BM25Okapi(self.docs)

    def search(self, query: str, k: int) -> Tuple[List[int], List[float]]:
        scores = torch.tensor(self.bm25.get_scores(_TOK.findall(query.lower())))
        s, i = scores.topk(min(k, len(self.triples)))
        return i.tolist(), s.tolist()
