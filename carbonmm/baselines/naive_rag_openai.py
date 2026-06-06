"""Naive RAG with an OpenAI chat model (default V28_OPENAI_MODEL) via the OpenAI API.

Identical pipeline to baselines/naive_rag.py and naive_rag_sonnet.py — hybrid
retrieve + LLM rerank with NO graph filter, NO extract_features, NO popularity
hint. ONLY the LLM endpoint differs (OpenAI chat.completions), so this is the
fair model-controlled naive baseline for the OpenAI side (e.g. GPT-5.5),
paralleling naive-rag-sonnet vs graphrag-v281-sonnet.

The model id is selected by V28_OPENAI_MODEL (default gpt-4o-mini); set it to
gpt-5.5 to reproduce the GPT-5.5 naive baseline. Reuses recommend_v28_openai's
_make_client (optional LAN source-address bypass via V28_LOCAL_ADDR).
"""
from __future__ import annotations

import logging

from .base import Baseline, label_space
from .naive_rag import _build_candidate_block, _extract_json_array, PROMPT_TEMPLATE
from ..graphrag.recommend_v28_openai import OPENAI_MODEL, _make_client, _effort_kwargs

logger = logging.getLogger(__name__)


class NaiveRAGOpenAIBaseline(Baseline):
    name = "naive-rag-openai"

    def __init__(self, retriever=None):
        if retriever is None:
            import importlib
            mod = importlib.import_module("carbonmm.graphrag.retrieve")
            retriever = mod.HybridRetriever.load_default()
        self.R = retriever
        # Bypass unavailable network routes for the OpenAI embedding query (dense retrieval)
        self.R.openai = _make_client()
        self.openai = _make_client()
        self.valid_codes = set(label_space())

    def predict(self, pdd_text: str, top_k: int = 5, pdd_meta: dict | None = None) -> list[tuple[str, float]]:
        try:
            hits = self.R.query(pdd_text, top_k=30)
        except Exception as e:
            logger.warning("retriever failed: %s", e)
            return []
        if not hits:
            return []

        candidate_block = _build_candidate_block(hits)
        prompt = PROMPT_TEMPLATE.format(
            project_text=pdd_text[:6_000], candidate_clauses=candidate_block
        )

        try:
            resp = self.openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                timeout=60,
                **_effort_kwargs(OPENAI_MODEL),
            )
            text = resp.choices[0].message.content or ""
        except Exception as e:
            logger.warning("OpenAI rerank failed: %s", e)
            return []

        parsed = _extract_json_array(text)
        out: list[tuple[str, float]] = []
        seen = set()
        for rank, item in enumerate(parsed, 1):
            code = (item or {}).get("code")
            if not code or code in seen:
                continue
            if code not in self.valid_codes:
                continue
            score = 1.0 / rank
            out.append((code, score))
            seen.add(code)
            if len(out) >= top_k:
                break

        if not out:
            by_code_score: dict[str, float] = {}
            for row, score in hits:
                by_code_score[row["code"]] = max(by_code_score.get(row["code"], 0), score)
            for code, score in sorted(by_code_score.items(), key=lambda x: -x[1])[:top_k]:
                out.append((code, score))
        return out
