"""
JefKV: AtlasKV-style in-attention KG memory for the Decision-Jef-0.1 decision model.

Builds on AtlasKV (Huang et al., ICLR 2026, arXiv:2510.17934), KBLaM (Wang et al., 2024),
Decision-Jef-0.1 (BarraHome, 2026) and mmBERT (Marone et al., 2025).
"""

from .attention import AtlasKVModernBertAttention
from .memory import BM25Retriever, DenseRetriever, KGMemory
from .model import JefKVModel

__all__ = ["AtlasKVModernBertAttention", "BM25Retriever", "DenseRetriever", "KGMemory", "JefKVModel"]
