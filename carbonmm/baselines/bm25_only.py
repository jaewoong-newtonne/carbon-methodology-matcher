"""Baseline 3: methodology-document-level BM25 (NOT clause-level).

Differs from `graphrag.retrieve.HybridRetriever` which is clause-granular
and combined with dense kNN. This baseline computes BM25 against one
concatenated document per methodology code — a classic IR comparator and the
single-signal baseline §5 needs.
"""
from __future__ import annotations

import importlib
from .base import Baseline, code_to_text_corpus, load_clauses_df

_bm25_mod = importlib.import_module("carbonmm.redact.bm25_sanity")
BM25 = _bm25_mod.BM25
tokenize = _bm25_mod.tokenize


class BM25OnlyBaseline(Baseline):
    name = "bm25-only"

    def __init__(self):
        corpus = code_to_text_corpus()
        self.codes = sorted(corpus.keys())
        docs = [tokenize(corpus[c]) for c in self.codes]
        self.bm25 = BM25(docs)

    def predict(self, pdd_text: str, top_k: int = 5) -> list[tuple[str, float]]:
        q = tokenize(pdd_text)
        scores = [(self.codes[i], self.bm25.score(q, i)) for i in range(len(self.codes))]
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]
