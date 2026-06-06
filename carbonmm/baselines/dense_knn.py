"""Baseline 4: dense kNN with mean-pooled methodology-doc embeddings.

For each methodology code, pool the 1024d PCA-reduced embeddings of all its
clauses (mean) → one vector per methodology. PDD text is embedded with the
same model/projection and matched by cosine similarity.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .base import Baseline, load_clauses_df

EMBED_DIR = Path(__file__).resolve().parents[1] / "data" / "embeddings"


class DenseKNNBaseline(Baseline):
    name = "dense-knn"

    def __init__(self):
        import joblib
        from openai import OpenAI

        df = load_clauses_df()
        vectors = np.load(EMBED_DIR / "clauses-1024d.npy")
        assert len(df) == vectors.shape[0]

        # Mean-pool per methodology code
        df = df.reset_index(drop=True)
        codes = sorted(df["code"].unique())
        pool = np.zeros((len(codes), vectors.shape[1]), dtype=np.float32)
        for i, c in enumerate(codes):
            idxs = df.index[df["code"] == c].values
            pool[i] = vectors[idxs].mean(axis=0)
        # L2-normalize for fast cosine via dot product
        norms = np.linalg.norm(pool, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.pool_norm = (pool / norms).astype(np.float32)
        self.codes = codes

        self.pca = joblib.load(EMBED_DIR / "pca-3072to1024.joblib")
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            # Reuse the same loader as embed.py
            import importlib
            embed_mod = importlib.import_module("carbonmm.graphrag.embed")
            key = embed_mod.load_openai_key()
        self.openai = OpenAI(api_key=key)

    def _embed(self, text: str) -> np.ndarray:
        import tiktoken
        import importlib
        embed_mod = importlib.import_module("carbonmm.graphrag.embed")

        enc = tiktoken.get_encoding("cl100k_base")
        text = embed_mod.truncate_tokens(text, embed_mod.MAX_TOKENS_PER_INPUT, enc)
        resp = self.openai.embeddings.create(model=embed_mod.MODEL, input=[text])
        raw = np.asarray(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
        q = self.pca.transform(raw)[0].astype(np.float32)
        q /= max(np.linalg.norm(q), 1e-9)
        return q

    def predict(self, pdd_text: str, top_k: int = 5) -> list[tuple[str, float]]:
        q = self._embed(pdd_text)
        sims = self.pool_norm @ q  # (n_codes,)
        order = np.argsort(-sims)[:top_k]
        return [(self.codes[int(i)], float(sims[int(i)])) for i in order]
