"""GraphRAG v2.8.1 reranker with claude-sonnet-4-6.

Supports three interchangeable LLM transports, selected by environment variable:
  - direct vendor API (``V281_API=claude|gemini``; the transport used for the
    reported results),
  - an optional self-hosted inference endpoint (``V281_USE_DAEMON=1``,
    ``V281_DAEMON_URL``),
  - a local ``claude`` CLI subprocess (fallback).

Pipeline (identical to v2.8 except LLM model + transport):
  - Stage A + B retrieval filter
  - Extract features via the reranker model (LLM call 1)
  - Hybrid retrieve + score fusion (unchanged)
  - LLM rerank with the v2.8.1 scale-aware prompt (LLM call 2)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import threading
from collections import Counter

from .candidate_filter import build_allowlist, codes_to_clause_idx, parse_date
from .extract_features import (
    PDDFeatures,
    PROMPT_TEMPLATE as FEATURE_PROMPT,
    _extract_json_object,
    _validate as _validate_features,
)
from .graph_filter import filter_candidates
from .recommend_v2 import _extract_json_array
from .recommend_v281_openai import GENERATION_PROMPT_V281, _guidance_flags
from .score_fusion import FusionConfig, fuse
from . import guidance as _guidance

from ..baselines.base import label_space, load_clauses_df

logger = logging.getLogger(__name__)
_MAXCHARS = int(os.environ.get("EVAL_TEXT_MAXCHARS", "6000"))  # rich-mode window; default 6000 = baseline

CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")  # resolve via PATH; override with CLAUDE_BIN
SONNET_MODEL = os.environ.get("V281_SONNET_MODEL", "claude-sonnet-4-6")
SONNET_TIMEOUT = 120.0
# Optional self-hosted inference endpoint transport. When V281_USE_DAEMON=1, the LLM
# call is POSTed to the daemon (default http://localhost:8765/chat) instead of
# spawning a local `claude` CLI subprocess. Shared by naive_rag_sonnet (imports
# this fn), so both naive + v281 route through the daemon with the same V281_SONNET_MODEL.
USE_DAEMON = os.environ.get("V281_USE_DAEMON", "") == "1"
DAEMON_URL = os.environ.get("V281_DAEMON_URL", "http://localhost:8765/chat")

# Direct vendor-API transport (used for the reported results).
# V281_API=claude -> Anthropic API; V281_API=gemini -> Google Gemini API.
# Keys read from env (CLAUDE_API_KEY / GEMINI_API_KEY), never logged. No extended
# thinking (thinkingBudget=0) for a fair "base reader" comparison. Model = V281_SONNET_MODEL.
API_VENDOR = os.environ.get("V281_API", "")
_API_MAXTOK = int(os.environ.get("V281_API_MAXTOK", "2048"))
_API_RETRIES = int(os.environ.get("V281_API_RETRIES", "6"))


_HTTP = None  # module-level pooled client (keep-alive) — avoids a fresh TLS handshake per call.


def _http_client():
    global _HTTP
    if _HTTP is None:
        import httpx
        _HTTP = httpx.Client(
            timeout=httpx.Timeout(SONNET_TIMEOUT, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=16, max_connections=32),
        )
    return _HTTP


def _api_retry(fn):
    import time
    last = None
    for i in range(_API_RETRIES):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(1.5 ** i, 5.0))  # short backoff (429s are transient/few)
    raise RuntimeError(f"api call failed after {_API_RETRIES} retries: {last}")


def _invoke_anthropic_api(prompt: str, model: str, timeout: float) -> str:
    import httpx
    def _call():
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": os.environ["CLAUDE_API_KEY"],
                     "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": model, "max_tokens": _API_MAXTOK,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=timeout,
        )
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json().get("content", [])
                       if b.get("type") == "text")
    return _api_retry(_call)


def _invoke_gemini_api(prompt: str, model: str, timeout: float) -> str:
    import httpx
    def _call():
        r = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"], "Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}],
                  "generationConfig": {"maxOutputTokens": _API_MAXTOK,
                                       "thinkingConfig": {"thinkingBudget": 0}}},
            timeout=timeout,
        )
        r.raise_for_status()
        cands = r.json().get("candidates", [])
        if not cands:
            raise RuntimeError("gemini: empty candidates")
        return "".join(p.get("text", "") for p in cands[0].get("content", {}).get("parts", []))
    return _api_retry(_call)


def _invoke_claude_subprocess(prompt: str, model: str = SONNET_MODEL, timeout: float = SONNET_TIMEOUT) -> str:
    """Run the rerank/extract LLM call. Transport precedence: direct vendor API
    (V281_API=claude|gemini) -> self-hosted inference endpoint (V281_USE_DAEMON=1) ->
    local `claude` CLI subprocess (fallback).
    """
    _vendor = os.environ.get("V281_API", API_VENDOR)  # re-read at call time (robust across workers)
    if _vendor == "claude":
        return _invoke_anthropic_api(prompt, model, timeout)
    if _vendor == "gemini":
        return _invoke_gemini_api(prompt, model, timeout)
    if USE_DAEMON:
        import httpx
        try:
            r = httpx.post(
                DAEMON_URL,
                json={"prompt": prompt, "model": model,
                      "timeout_s": min(timeout - 5.0, 90.0), "use_cache": True},
                timeout=timeout,
            )
            r.raise_for_status()
            return r.json()["text"]
        except Exception as e:
            raise RuntimeError(f"daemon call failed: {e}")
    try:
        proc = subprocess.run(
            [CLAUDE_BIN, "-p", "--model", model, "--output-format", "text"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            stderr = proc.stderr[:500] if proc.stderr else ""
            raise RuntimeError(f"claude rc={proc.returncode}: {stderr}")
        return proc.stdout
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"claude timeout after {timeout}s")


class GraphRAGv281SonnetBaseline:
    name = "graphrag-v281-sonnet"
    _lock = threading.Lock()  # serialize subprocess to avoid spurious CLI state

    def __init__(self, retriever=None, fusion: FusionConfig | None = None, top_k_after_fusion: int = 30):
        if retriever is None:
            from .retrieve import HybridRetriever
            retriever = HybridRetriever.load_default()
        self.R = retriever
        # OpenAI client for query embeddings only (no special network dependency — direct LAN)
        from .recommend_v28_openai import _make_client
        self.R.openai = _make_client()
        self.fusion_cfg = fusion or FusionConfig(alpha=0.5, beta=0.5, gamma=0.0)
        self.top_k_after_fusion = top_k_after_fusion
        self.valid_codes = set(label_space())
        self._meta_df = load_clauses_df()
        self._corpus_codes = self._meta_df["code"].unique().tolist()
        self._feat_cache: dict[str, PDDFeatures] = {}
        try:
            from ..eval.splits import load_splits
            splits = load_splits()
            self._freq_count = dict(Counter(p["gt"] for p in splits["val"]["pdds"]))
        except Exception:
            self._freq_count = {}

    def _call_sonnet(self, prompt: str) -> str:
        # Lock-free: subprocess is process-level isolated, CLI handles concurrency safely
        return _invoke_claude_subprocess(prompt)

    def _extract_features_sonnet(self, pdd_text: str) -> PDDFeatures:
        cache_key = pdd_text[:200]
        if cache_key in self._feat_cache:
            return self._feat_cache[cache_key]
        prompt = FEATURE_PROMPT.format(project_text=pdd_text[:_MAXCHARS])
        try:
            text = self._call_sonnet(prompt)
            parsed = _extract_json_object(text)
            feats = _validate_features(parsed)
            feats.raw_response = text[:500]
        except Exception as e:
            logger.warning("extract failed: %s", e)
            feats = PDDFeatures()
        self._feat_cache[cache_key] = feats
        return feats

    def _build_candidate_block(self, fused, df):
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

        feats = self._extract_features_sonnet(pdd_text)

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

        # Guidance-KG connection (flag-gated; all OFF → identical baseline).
        g_filter, g_scale, g_clauses = _guidance_flags()
        pdd_value = meta.get("capacity_value")
        pdd_conf = meta.get("capacity_conf", 1.0)
        smults = (_guidance.scale_multipliers(candidate_codes, pdd_value, pdd_conf)
                  if g_filter else None)

        fused = fuse(
            raw["bm25_top"], raw["dense_top"], raw["idx_to_code"],
            filter_results, config=self.fusion_cfg, top_k=self.top_k_after_fusion,
            scale_multipliers=smults,
        )
        if not fused:
            return []

        block = self._build_candidate_block(fused, raw["df"])
        if g_scale or g_clauses:
            gblock = _guidance.rerank_block(
                [c for c, _, _ in fused], pdd_value, pdd_conf, pdd_text, pdd_registry or "",
                want_scale=g_scale, want_clauses=g_clauses,
            )
            if gblock:
                block = block + "\n\n" + gblock
        prompt = GENERATION_PROMPT_V281.format(
            ghg_species=", ".join(feats.ghg_species) or "unknown",
            sectoral_scope=feats.sectoral_scope or "unknown",
            technology=feats.technology or "unknown",
            country_iso=feats.country_iso or "unknown",
            scale=feats.scale,
            project_text=pdd_text[:_MAXCHARS],
            candidate_clauses=block,
        )

        try:
            text = self._call_sonnet(prompt)
        except Exception as e:
            logger.warning("Sonnet rerank failed: %s; falling back to fusion", e)
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
