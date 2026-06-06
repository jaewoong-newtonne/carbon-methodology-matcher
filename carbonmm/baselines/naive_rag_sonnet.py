"""Naive RAG with claude-sonnet-4-6 via local CLI subprocess.

Identical pipeline to baselines/naive_rag.py — hybrid retrieve + LLM rerank
with NO graph filter, NO extract_features, NO popularity hint. Only the LLM
model is changed for fair model-controlled comparison vs graphrag-v281-sonnet.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from collections import defaultdict

from .base import Baseline, label_space
from .naive_rag import _build_candidate_block, _extract_json_array, PROMPT_TEMPLATE
from ..graphrag.recommend_v281_sonnet import (
    CLAUDE_BIN,
    SONNET_MODEL,
    SONNET_TIMEOUT,
    _invoke_claude_subprocess,
)

logger = logging.getLogger(__name__)


class NaiveRAGSonnetBaseline(Baseline):
    name = "naive-rag-sonnet"

    def __init__(self, retriever=None):
        if retriever is None:
            import importlib
            mod = importlib.import_module("carbonmm.graphrag.retrieve")
            retriever = mod.HybridRetriever.load_default()
        self.R = retriever
        # Bypass unavailable network routes for OpenAI embedding query
        from ..graphrag.recommend_v28_openai import _make_client
        self.R.openai = _make_client()
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
            text = _invoke_claude_subprocess(prompt, model=SONNET_MODEL, timeout=SONNET_TIMEOUT)
        except Exception as e:
            logger.warning("Sonnet rerank failed: %s", e)
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
