"""GraphRAG v2.8 with OpenAI gpt-4o-mini for LLM rerank (daemon fallback).

When the self-hosted inference endpoint is unreachable, this variant uses OpenAI
gpt-4o-mini directly for structured-output extraction and rerank.

Architecture (identical to v28 except LLM endpoint):
  - Retrieval-side Stage A + B filter (registry × family × scale)
  - extract_features via OpenAI gpt-4o-mini
  - LLM rerank via OpenAI gpt-4o-mini
  - Optional source-address binding via V28_LOCAL_ADDR
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from datetime import date
from pathlib import Path

import httpx
from openai import OpenAI

from ..baselines.base import label_space, load_clauses_df
from .candidate_filter import build_allowlist, codes_to_clause_idx, parse_date
from .extract_features import (
    PROMPT_TEMPLATE as FEATURE_PROMPT,
    PDDFeatures,
    _extract_json_object,
    _validate as _validate_features,
)
from .graph_filter import filter_candidates
from .recommend_v2 import _build_candidate_block_v2, _extract_json_array
from .recommend_v25 import GENERATION_PROMPT
from .score_fusion import FusionConfig, fuse

logger = logging.getLogger(__name__)
_MAXCHARS = int(os.environ.get("EVAL_TEXT_MAXCHARS", "6000"))  # rich-mode window; default 6000 = baseline

OPENAI_MODEL = os.environ.get("V28_OPENAI_MODEL", "gpt-4o-mini")
# Reasoning effort for GPT-5.x models (minimal = fair "base reader" comparison + low cost).
_REASONING_EFFORT = os.environ.get("V28_REASONING_EFFORT", "minimal")
def _effort_kwargs(model: str) -> dict:
    return {"reasoning_effort": _REASONING_EFFORT} if model.startswith("gpt-5") else {}
# Optional source-address binding: set V28_LOCAL_ADDR to a host LAN IP only when
# the machine's default network route is unavailable. Empty = normal routing.
LOCAL_ADDR = os.environ.get("V28_LOCAL_ADDR", "")


def _load_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    # A local ``.env`` (or ``backend/.env``) is read only as a fallback.
    candidates = [Path(".env"), Path("backend") / ".env"]
    for p in candidates:
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    raise RuntimeError("OPENAI_API_KEY missing")


def _make_client() -> OpenAI:
    """OpenAI client with an explicit LAN source address (bypasses unavailable network routes)."""
    transport = httpx.HTTPTransport(local_address=LOCAL_ADDR) if LOCAL_ADDR else None
    http_client = httpx.Client(
        transport=transport,
        timeout=httpx.Timeout(60.0, connect=10.0),
    )
    # max_retries: exponential backoff on 429/5xx so transient rate-limits recover
    # instead of falling back to fusion/retrieval (which silently corrupts scores).
    return OpenAI(api_key=_load_openai_key(), http_client=http_client, max_retries=int(os.environ.get("V28_MAX_RETRIES", "8")))


class GraphRAGv28OpenAIBaseline:
    name = "graphrag-v28-openai"

    def __init__(self, retriever=None, fusion: FusionConfig | None = None, top_k_after_fusion: int = 30):
        if retriever is None:
            from .retrieve import HybridRetriever
            retriever = HybridRetriever.load_default()
        self.R = retriever
        # Replace retriever's OpenAI client to bypass unavailable network routes
        self.R.openai = _make_client()
        self.fusion_cfg = fusion or FusionConfig(alpha=0.5, beta=0.5, gamma=0.0)
        self.top_k_after_fusion = top_k_after_fusion
        self.valid_codes = set(label_space())
        self._meta_df = load_clauses_df()
        self._corpus_codes = self._meta_df["code"].unique().tolist()
        self.openai = _make_client()
        self._feat_cache: dict[str, PDDFeatures] = {}
        # Popularity prior from val split
        try:
            from ..eval.splits import load_splits
            splits = load_splits()
            self._freq_count = dict(Counter(p["gt"] for p in splits["val"]["pdds"]))
        except Exception:
            self._freq_count = {}

    def _extract_features_openai(self, pdd_text: str) -> PDDFeatures:
        # Cache by first 200 chars of PDD (a rough proxy; we re-feed full text below)
        cache_key = pdd_text[:200]
        if cache_key in self._feat_cache:
            return self._feat_cache[cache_key]
        prompt = FEATURE_PROMPT.format(project_text=pdd_text[:_MAXCHARS])
        try:
            resp = self.openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                **_effort_kwargs(OPENAI_MODEL),
                timeout=60,
            )
            text = resp.choices[0].message.content or ""
            parsed = _extract_json_object(text)
            feats = _validate_features(parsed)
            feats.raw_response = text[:500]
        except Exception as e:
            logger.warning("extract_features failed: %s", e)
            feats = PDDFeatures()
        self._feat_cache[cache_key] = feats
        return feats

    def _build_candidate_block(self, fused, df):
        # Same as v25: clean prompt + popularity annotation
        lines = []
        used = 0
        max_clauses_per_code = 2
        max_chars_per_clause = 400
        max_total_chars = 12_000
        for code, _score, _comp in fused:
            cnt = self._freq_count.get(code, 0)
            rows = df[df["code"] == code]
            n_emitted = 0
            for row in rows.itertuples(index=False):
                txt = (row.clause_text or "").replace("\n", " ")[:max_chars_per_clause]
                line = f"- [{code}] ({cnt} projects) ({row.clause_type}) {txt}"
                if used + len(line) > max_total_chars:
                    return "\n".join(lines) + "\n- ... (truncated)"
                lines.append(line)
                used += len(line) + 1
                n_emitted += 1
                if n_emitted >= max_clauses_per_code:
                    break
        return "\n".join(lines)

    def predict(self, pdd_text: str, top_k: int = 5, pdd_meta: dict | None = None) -> list[tuple[str, float]]:
        meta = pdd_meta or {}
        pdd_registry = meta.get("registry")
        pdd_credit_start = parse_date(meta.get("creditingPeriodStartDate"))

        feats = self._extract_features_openai(pdd_text)

        allowed_codes, _diag = build_allowlist(
            pdd_registry=pdd_registry,
            pdd_credit_start=pdd_credit_start,
            pdd_scale=feats.scale,
            pdd_scope=feats.sectoral_scope,
            corpus_codes=self._corpus_codes,
        )
        allow_idx = codes_to_clause_idx(allowed_codes, self._meta_df) if allowed_codes else None

        try:
            raw = self.R.query_with_raw(pdd_text, allow_clause_idx=allow_idx)
        except Exception as e:
            logger.warning("retrieve failed: %s", e)
            return []

        candidate_codes = list(
            {raw["idx_to_code"][i] for i, _ in raw["bm25_top"]} |
            {raw["idx_to_code"][i] for i, _ in raw["dense_top"]}
        )
        filter_results = filter_candidates(candidate_codes, feats)
        fused = fuse(
            raw["bm25_top"], raw["dense_top"], raw["idx_to_code"],
            filter_results, config=self.fusion_cfg, top_k=self.top_k_after_fusion,
        )
        if not fused:
            return []

        block = self._build_candidate_block(fused, raw["df"])
        prompt = GENERATION_PROMPT.format(
            ghg_species=", ".join(feats.ghg_species) or "unknown",
            sectoral_scope=feats.sectoral_scope or "unknown",
            technology=feats.technology or "unknown",
            country_iso=feats.country_iso or "unknown",
            scale=feats.scale,
            project_text=pdd_text[:_MAXCHARS],
            candidate_clauses=block,
        )

        try:
            resp = self.openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                **_effort_kwargs(OPENAI_MODEL),
                timeout=60,
            )
            text = resp.choices[0].message.content or ""
        except Exception as e:
            logger.warning("LLM rerank failed: %s; falling back to fusion", e)
            return [(c, s) for c, s, _ in fused[:top_k]]

        parsed = _extract_json_array(text)
        out: list[tuple[str, float]] = []
        seen = set()
        for rank, item in enumerate(parsed, 1):
            code = (item or {}).get("code")
            if not code or code in seen:
                continue
            if code not in self.valid_codes:
                continue
            out.append((code, 1.0 / rank))
            seen.add(code)
            if len(out) >= top_k:
                break

        if not out:
            return [(c, s) for c, s, _ in fused[:top_k]]
        return out
