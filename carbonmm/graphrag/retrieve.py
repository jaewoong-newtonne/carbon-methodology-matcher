"""Hybrid clause-level retrieval: BM25 ∪ dense kNN → RRF → top-K.

Reuses the existing BM25 implementation in `redact.bm25_sanity` (offline, no
external dep). Adds an HNSW dense index over PCA-reduced
text-embedding-3-large vectors.

Usage:
    from carbonmm.graphrag.retrieve import HybridRetriever
    R = HybridRetriever.load_default()
    hits = R.query("PDD project text...", top_k=30)
    for hit, score in hits:
        print(hit["code"], hit["clause_type"], score)

CLI:
    python -m carbonmm.graphrag.retrieve --pdd-id GS10529 --top-k 30
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Reuse existing offline BM25 + tokenizer
import importlib
_bm25_mod = importlib.import_module("carbonmm.redact.bm25_sanity")
BM25 = _bm25_mod.BM25
tokenize = _bm25_mod.tokenize

logger = logging.getLogger(__name__)

EMBED_DIR = Path(__file__).resolve().parents[1] / "data" / "embeddings"
EVAL_PDD_ROOT = Path(__file__).resolve().parents[1] / "data" / "eval-pdds"

DEFAULT_BM25_K = 50
DEFAULT_DENSE_K = 50
DEFAULT_RRF_K = 60  # standard RRF smoothing constant
DEFAULT_TOP_K = 30


@dataclass
class RetrievalConfig:
    bm25_k: int = DEFAULT_BM25_K
    dense_k: int = DEFAULT_DENSE_K
    rrf_k: int = DEFAULT_RRF_K
    top_k: int = DEFAULT_TOP_K


class HybridRetriever:
    def __init__(
        self,
        meta_df: pd.DataFrame,
        vectors_1024d: np.ndarray,
        pca_model,
        openai_client,
        config: RetrievalConfig | None = None,
    ):
        assert len(meta_df) == vectors_1024d.shape[0], "meta/vectors row count mismatch"
        self.df = meta_df.reset_index(drop=True)
        self.n = len(self.df)
        self.config = config or RetrievalConfig()

        logger.info("building BM25 over %d clauses ...", self.n)
        self.bm25 = BM25([tokenize(t) for t in self.df["clause_text"].fillna("")])

        logger.info("building HNSW index (cosine, M=32, ef_construction=200) ...")
        import hnswlib
        dim = vectors_1024d.shape[1]
        self.hnsw = hnswlib.Index(space="cosine", dim=dim)
        self.hnsw.init_index(max_elements=self.n, ef_construction=200, M=32)
        self.hnsw.add_items(vectors_1024d.astype(np.float32), np.arange(self.n))
        self.hnsw.set_ef(max(64, self.config.dense_k * 2))

        self.pca = pca_model
        self.openai = openai_client

    # ─── Loading factory ────────────────────────────────────────────────

    @classmethod
    def load_default(
        cls,
        embed_dir: Path = EMBED_DIR,
        config: RetrievalConfig | None = None,
    ) -> "HybridRetriever":
        from openai import OpenAI
        import joblib

        meta_path = embed_dir / "clauses-meta.parquet"
        vec_path = embed_dir / "clauses-1024d.npy"
        pca_path = embed_dir / "pca-3072to1024.joblib"

        for p in [meta_path, vec_path, pca_path]:
            if not p.exists():
                raise FileNotFoundError(f"missing artifact: {p} — run embed.py first")

        meta_df = pd.read_parquet(meta_path)
        vectors = np.load(vec_path)
        pca = joblib.load(pca_path)

        # OpenAI key — try env, then sibling .env
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            from .embed import load_openai_key
            key = load_openai_key()
        client = OpenAI(api_key=key)
        return cls(meta_df, vectors, pca, client, config)

    # ─── Single-query retrieval ─────────────────────────────────────────

    def _embed_query(self, text: str) -> np.ndarray:
        """Embed text via OpenAI text-embedding-3-large + PCA project to 1024d."""
        from .embed import MODEL, MAX_TOKENS_PER_INPUT, truncate_tokens
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        text = truncate_tokens(text, MAX_TOKENS_PER_INPUT, enc)
        resp = self.openai.embeddings.create(model=MODEL, input=[text])
        raw = np.asarray(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
        return self.pca.transform(raw)[0].astype(np.float32)

    def _retrieve_raw(
        self,
        text: str,
        allow_clause_idx: set[int] | None = None,
    ) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        """Return raw (bm25_top, dense_top) as lists of (clause_idx, score) tuples.

        When `allow_clause_idx` is provided, filter both channels to that subset.
        Per-channel k is bumped 2× to compensate for the smaller candidate pool;
        HNSW is oversampled 3× then post-filtered.
        """
        cfg = self.config
        masked = allow_clause_idx is not None and len(allow_clause_idx) > 0
        bm25_k = cfg.bm25_k * 2 if masked else cfg.bm25_k
        dense_k = cfg.dense_k * 2 if masked else cfg.dense_k

        q_tokens = tokenize(text)
        if masked:
            bm25_scores = [(i, self.bm25.score(q_tokens, i)) for i in allow_clause_idx]
        else:
            bm25_scores = [(i, self.bm25.score(q_tokens, i)) for i in range(self.n)]
        bm25_top = sorted(bm25_scores, key=lambda x: -x[1])[:bm25_k]

        qv = self._embed_query(text)
        if masked:
            # Oversample 3× then post-filter to the allowlist
            oversample = min(self.n, dense_k * 3)
            labels, dists = self.hnsw.knn_query(qv, k=oversample)
            dense_top = []
            for i, d in zip(labels[0], dists[0]):
                if int(i) in allow_clause_idx:
                    dense_top.append((int(i), 1.0 - float(d)))
                    if len(dense_top) >= dense_k:
                        break
        else:
            labels, dists = self.hnsw.knn_query(qv, k=dense_k)
            dense_top = [(int(i), 1.0 - float(d)) for i, d in zip(labels[0], dists[0])]
        return bm25_top, dense_top

    def query(
        self,
        text: str,
        top_k: int | None = None,
        allow_clause_idx: set[int] | None = None,
    ) -> list[tuple[dict, float]]:
        """Return top-K clause hits as (meta_row_dict, fused_score) tuples."""
        cfg = self.config
        k = top_k or cfg.top_k

        bm25_top, dense_top = self._retrieve_raw(text, allow_clause_idx=allow_clause_idx)

        # RRF fusion
        rrf: dict[int, float] = {}
        for rank, (i, _) in enumerate(bm25_top, 1):
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (cfg.rrf_k + rank)
        for rank, (i, _) in enumerate(dense_top, 1):
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (cfg.rrf_k + rank)

        fused = sorted(rrf.items(), key=lambda x: -x[1])[:k]
        hits = []
        for idx, score in fused:
            row = self.df.iloc[idx].to_dict()
            hits.append((row, float(score)))
        return hits

    def query_with_raw(
        self,
        text: str,
        allow_clause_idx: set[int] | None = None,
    ) -> dict:
        """Return raw + RRF-fused for downstream fusion. Used by GraphRAG.

        `allow_clause_idx` (when supplied) restricts both BM25 and dense channels
        to the given clause indices, with 2× k-bump + 3× HNSW oversample.
        """
        cfg = self.config
        bm25_top, dense_top = self._retrieve_raw(text, allow_clause_idx=allow_clause_idx)

        # Map clause-idx to methodology code (small lookup table built once)
        if not hasattr(self, "_idx_to_code"):
            self._idx_to_code = dict(enumerate(self.df["code"].tolist()))

        return {
            "bm25_top": bm25_top,
            "dense_top": dense_top,
            "idx_to_code": self._idx_to_code,
            "df": self.df,
        }


# ─── CLI smoke test ────────────────────────────────────────────────────

def _load_pdd_text(pdd_id: str) -> str | None:
    """Try Section A first (preferred — short, redacted target), fall back to full body."""
    section_a = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "section-a-redacted"
    )
    if section_a.exists():
        for f in section_a.rglob(f"{pdd_id}.*.json"):
            d = json.loads(f.read_text())
            if d.get("redacted_text"):
                return d["redacted_text"]
            if d.get("section_a_text"):
                return d["section_a_text"]
    # body fallback
    for f in EVAL_PDD_ROOT.rglob(f"{pdd_id}.body.json"):
        d = json.loads(f.read_text())
        return d.get("full_text") or d.get("text") or ""
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdd-id", required=True)
    ap.add_argument("--top-k", type=int, default=30)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    text = _load_pdd_text(args.pdd_id)
    if not text:
        raise SystemExit(f"PDD {args.pdd_id} not found")
    logger.info("PDD %s text length: %d chars", args.pdd_id, len(text))

    R = HybridRetriever.load_default(config=RetrievalConfig(top_k=args.top_k))
    hits = R.query(text, top_k=args.top_k)

    print(f"\n=== top-{args.top_k} clauses for {args.pdd_id} ===")
    # group by methodology code for compactness
    from collections import defaultdict
    by_code = defaultdict(list)
    for row, score in hits:
        by_code[row["code"]].append((row["clause_type"], score))
    for code in sorted(by_code, key=lambda c: -sum(s for _, s in by_code[c])):
        clauses = by_code[code]
        total = sum(s for _, s in clauses)
        types = ",".join(t for t, _ in clauses)
        print(f"  {code:14s}  agg={total:.4f}  n={len(clauses):2d}  types={types}")


if __name__ == "__main__":
    main()
