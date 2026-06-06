"""Base interface for all baselines + helpers for label space + corpus loaders."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

EMBED_DIR = Path(__file__).resolve().parents[1] / "data" / "embeddings"
META_PATH = EMBED_DIR / "clauses-meta.parquet"


def load_clauses_df() -> pd.DataFrame:
    if not META_PATH.exists():
        raise FileNotFoundError(
            f"{META_PATH} missing — run `python -m carbonmm.graphrag.embed` first"
        )
    return pd.read_parquet(META_PATH)


def label_space(df: pd.DataFrame | None = None) -> list[str]:
    """All unique methodology codes in the corpus = candidate labels."""
    df = load_clauses_df() if df is None else df
    return sorted(df["code"].unique())


def code_to_text_corpus(df: pd.DataFrame | None = None) -> dict[str, str]:
    """One concatenated document per methodology code (for doc-level BM25)."""
    df = load_clauses_df() if df is None else df
    out: dict[str, list[str]] = {}
    for row in df.itertuples(index=False):
        out.setdefault(row.code, []).append(row.clause_text or "")
    return {code: " ".join(texts) for code, texts in out.items()}


def code_to_registry(df: pd.DataFrame | None = None) -> dict[str, str]:
    df = load_clauses_df() if df is None else df
    # take the most common registry per code (should be unique anyway)
    out = {}
    for row in df.itertuples(index=False):
        out.setdefault(row.code, row.registry or "")
    return out


class Baseline(ABC):
    """Common contract: predict top-K (code, score) tuples for a PDD."""

    name: str = "baseline"

    @abstractmethod
    def predict(
        self,
        pdd_text: str,
        top_k: int = 5,
        pdd_meta: dict | None = None,
    ) -> list[tuple[str, float]]:
        """Return top-k (methodology_code, score) sorted by descending score.

        `pdd_meta` (when supplied) carries reliable PDD-side metadata
        ({"registry": "GS"|"VCS", "creditingPeriodStartDate": "YYYY-MM-DD"})
        for baselines that use retrieval-side filtering (e.g., graphrag-v28).
        Baselines that don't need it ignore the argument.
        """

    def predict_batch(
        self,
        pdd_texts: Iterable[str],
        top_k: int = 5,
        pdd_metas: Iterable[dict | None] | None = None,
    ) -> list[list[tuple[str, float]]]:
        if pdd_metas is None:
            return [self.predict(t, top_k=top_k) for t in pdd_texts]
        return [self.predict(t, top_k=top_k, pdd_meta=m) for t, m in zip(pdd_texts, pdd_metas)]
