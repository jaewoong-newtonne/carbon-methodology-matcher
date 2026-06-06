"""GraphRAG v3 — RRF-hybrid + frequency prior.

Builds on v2's prompt cleanup. Two additional fixes targeting the v1 failure
mode where LLM rerank demotes a correct fusion top-1 to top-2/3 (e.g.,
GCCM001 ↔ ACM0002 swap, observed 66 times in test 535):

  1. **RRF-hybrid ranking** — LLM provides one rank list, fusion provides
     another. Reciprocal Rank Fusion combines them with weight on fusion side,
     so LLM cannot fully override a strong fusion signal.
        rrf_score(c) = w_F · 1/(rank_fusion(c) + K) + w_L · 1/(rank_llm(c) + K)
     With w_F=0.6, w_L=0.4, K=60 — fusion dominates close calls.

  2. **Frequency prior fusion channel** — add δ·freq_prior to fusion so
     high-frequency methodologies (ACM0002 has 158 train PDDs; AMS-I.D. 56)
     are anchored. Reduces LLM-driven demote of high-prior codes.

Optional escalation hook: switch model to `claude-sonnet-4-6` if Haiku v3
still leaves headroom. Set MODEL env var to override.

Routes LLM calls through the configured inference transport (self-hosted daemon).
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

import httpx

from collections import Counter

from ..baselines.base import Baseline, label_space
from ..eval.splits import load_splits
from .extract_features import extract_features
from .graph_filter import filter_candidates
from .recommend_v2 import GENERATION_PROMPT, _build_candidate_block_v2, _extract_json_array
from .score_fusion import FusionConfig, fuse

logger = logging.getLogger(__name__)

DAEMON_URL = "http://localhost:8765/chat"
DAEMON_TIMEOUT = 120.0
DEFAULT_MODEL = os.environ.get("GRAPHRAG_V3_MODEL", "claude-haiku-4-5")

RRF_K = 60          # standard RRF constant
W_FUSION = 0.6      # fusion weight in RRF combine
W_LLM = 0.4         # LLM weight in RRF combine
FREQ_PRIOR_DELTA = 0.15  # δ·freq_prior in fusion channel


class GraphRAGv3Baseline(Baseline):
    name = "graphrag-v3"

    def __init__(
        self,
        retriever=None,
        fusion: FusionConfig | None = None,
        model: str = DEFAULT_MODEL,
        daemon_url: str = DAEMON_URL,
        top_k_after_fusion: int = 30,
    ):
        if retriever is None:
            from .retrieve import HybridRetriever
            retriever = HybridRetriever.load_default()
        self.R = retriever
        # γ=0 same as v2; freq_prior injected separately below
        self.fusion_cfg = fusion or FusionConfig(alpha=0.5, beta=0.5, gamma=0.0)
        self.model = model
        self.daemon_url = daemon_url
        self.top_k_after_fusion = top_k_after_fusion
        self.valid_codes = set(label_space())
        self.client = httpx.Client(timeout=DAEMON_TIMEOUT)

        # Frequency prior from VAL split only (test is held out — paper-clean).
        # ACM0002 dominates ~30% so a freq prior anchors strongly.
        try:
            splits = load_splits()
            val_labels = Counter(p["gt"] for p in splits["val"]["pdds"])
            total = sum(val_labels.values()) or 1
            self._freq_prior = {c: v / total for c, v in val_labels.items()}
        except Exception as e:
            logger.warning("freq_prior load failed: %s — defaulting to uniform", e)
            self._freq_prior = {}

    def _call_daemon(self, prompt: str) -> str:
        resp = self.client.post(
            self.daemon_url,
            json={
                "prompt": prompt,
                "model": self.model,
                "timeout_s": min(DAEMON_TIMEOUT - 5.0, 90.0),
                "use_cache": True,
            },
        )
        resp.raise_for_status()
        return resp.json()["text"]

    def _rrf_combine(
        self,
        fusion_ranking: list[str],
        llm_ranking: list[str],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """RRF combine LLM + fusion rankings with fusion-side weight."""
        scores: dict[str, float] = defaultdict(float)
        all_codes = set(fusion_ranking) | set(llm_ranking)
        for code in all_codes:
            rank_f = fusion_ranking.index(code) + 1 if code in fusion_ranking else len(fusion_ranking) + 1
            rank_l = llm_ranking.index(code) + 1 if code in llm_ranking else len(llm_ranking) + 1
            scores[code] = W_FUSION / (rank_f + RRF_K) + W_LLM / (rank_l + RRF_K)
        return sorted(scores.items(), key=lambda x: -x[1])[:top_k]

    def predict(self, pdd_text: str, top_k: int = 5) -> list[tuple[str, float]]:
        try:
            raw = self.R.query_with_raw(pdd_text)
        except Exception as e:
            logger.warning("retrieve failed: %s", e)
            return []

        try:
            feats = extract_features(pdd_text)
        except Exception as e:
            logger.warning("extract failed: %s — proceeding without features", e)
            from .extract_features import PDDFeatures
            feats = PDDFeatures()

        candidate_codes = list(
            {raw["idx_to_code"][i] for i, _ in raw["bm25_top"]} |
            {raw["idx_to_code"][i] for i, _ in raw["dense_top"]}
        )
        filter_results = filter_candidates(candidate_codes, feats)

        # Fusion + freq_prior injection
        fused = fuse(
            raw["bm25_top"], raw["dense_top"], raw["idx_to_code"],
            filter_results, config=self.fusion_cfg, top_k=self.top_k_after_fusion,
        )
        if not fused:
            return []
        if self._freq_prior:
            fused = [
                (c, s + FREQ_PRIOR_DELTA * self._freq_prior.get(c, 0.0), comp)
                for c, s, comp in fused
            ]
            fused.sort(key=lambda x: -x[1])

        fusion_ranking = [c for c, _, _ in fused]

        # LLM rerank
        block = _build_candidate_block_v2(fused, raw["df"])
        prompt = GENERATION_PROMPT.format(
            ghg_species=", ".join(feats.ghg_species) or "unknown",
            sectoral_scope=feats.sectoral_scope or "unknown",
            technology=feats.technology or "unknown",
            country_iso=feats.country_iso or "unknown",
            scale=feats.scale,
            project_text=pdd_text[:6_000],
            candidate_clauses=block,
        )

        try:
            text = self._call_daemon(prompt)
        except Exception as e:
            logger.warning("daemon recommend failed: %s — falling back to fusion order", e)
            return [(c, s) for c, s, _ in fused[:top_k]]

        parsed = _extract_json_array(text)
        llm_ranking: list[str] = []
        seen = set()
        for item in parsed:
            code = (item or {}).get("code")
            if not code or code in seen:
                continue
            if code not in self.valid_codes:
                continue
            llm_ranking.append(code)
            seen.add(code)
            if len(llm_ranking) >= 10:
                break

        if not llm_ranking:
            logger.info("graphrag-v3: empty LLM output, returning fusion order")
            return [(c, s) for c, s, _ in fused[:top_k]]

        # RRF combine
        return self._rrf_combine(fusion_ranking, llm_ranking, top_k=top_k)
