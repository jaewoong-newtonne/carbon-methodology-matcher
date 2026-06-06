"""GraphRAG v2.8 — v2.5 + retrieval-side Stage A + B candidate filter.

Stage A: registry/family pre-filter (PDD.registry × code prefix).
Stage B: date validity (conservative) + AMS-large hard-exclude.

The filter narrows the clause pool from 3,334 to ~2,100 before retrieval,
surfacing rare/under-represented methodologies that were drowned out by
off-registry distractors (numeric GS codes 407/408 etc. in VCS PDDs,
BCR/ACR/CER/PURO in GS PDDs).

Inherits v2.5's prompt (popularity-aware), Haiku model, retry logic.
Goal: push test top-1 from 0.512 (v2.5) toward Oracle ceiling 0.632
by closing the 87-PDD retrieval-failure cap.
"""
from __future__ import annotations

import logging
from datetime import date

from ..baselines.base import label_space, load_clauses_df
from .candidate_filter import build_allowlist, codes_to_clause_idx, parse_date
from .extract_features import PDDFeatures, extract_features
from .graph_filter import filter_candidates
from .recommend_v2 import _build_candidate_block_v2, _extract_json_array
from .recommend_v25 import GraphRAGv25Baseline, GENERATION_PROMPT
from .score_fusion import FusionConfig, fuse

logger = logging.getLogger(__name__)


class GraphRAGv28Baseline(GraphRAGv25Baseline):
    name = "graphrag-v28"

    def __init__(self, **kw):
        super().__init__(**kw)
        # Cache clause-meta for the codes_to_clause_idx adapter
        self._meta_df = load_clauses_df()
        self._corpus_codes = self._meta_df["code"].unique().tolist()

    def predict(self, pdd_text: str, top_k: int = 5, pdd_meta: dict | None = None) -> list[tuple[str, float]]:
        meta = pdd_meta or {}
        pdd_registry = meta.get("registry")
        pdd_credit_start = parse_date(meta.get("creditingPeriodStartDate"))

        # Stage 1: extract features (still useful for AMS-large rule + LLM prompt)
        try:
            feats = extract_features(pdd_text)
        except Exception as e:
            logger.warning("extract failed: %s", e)
            feats = PDDFeatures()

        # Build the Stage A + B allowlist
        allowed_codes, diag = build_allowlist(
            pdd_registry=pdd_registry,
            pdd_credit_start=pdd_credit_start,
            pdd_scale=feats.scale,
            pdd_scope=feats.sectoral_scope,
            corpus_codes=self._corpus_codes,
        )
        if not allowed_codes:
            # Catastrophic miss — fall back to full corpus
            logger.warning("v28: empty allowlist for registry=%s; falling back to full corpus",
                           pdd_registry)
            allow_idx = None
        else:
            allow_idx = codes_to_clause_idx(allowed_codes, self._meta_df)

        # Retrieve with the mask
        try:
            raw = self.R.query_with_raw(pdd_text, allow_clause_idx=allow_idx)
        except Exception as e:
            logger.warning("retrieve failed: %s", e)
            return []

        # Stage 3: rule-based filter (existing — kept for downstream prompt info)
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
            logger.warning("daemon failed: %s", e)
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
