"""GraphRAG v2.8.1 — v28-openai + scale-aware prompt rule.

Diagnostic finding from v28-gpt4o per-class analysis:
  - WINS BIG: ACM0001 +37pp, AMS-II.G. +57pp (popularity hint helps rare-decent codes)
  - LOSES BIG: AMS-I.D. -43pp, AMS-I.E. -53pp (popularity push toward ACM/AM
    misroutes small-scale renewable PDDs)

Fix: add explicit scale-disambiguation instruction to the rerank prompt.
When extracted scale=small AND project mentions small-scale capacity markers,
strongly prefer AMS-* (small-scale CDM) over ACM/AM (large-scale).

Inherits v28-openai's:
  - Stage A + B retrieval-side filter
  - OpenAI gpt-4o LLM (with local_address bypass)
  - Popularity hint
  - Catalog-derived metadata
"""
from __future__ import annotations

import logging
import os
from collections import Counter

from .recommend_v28_openai import (
    GraphRAGv28OpenAIBaseline,
    OPENAI_MODEL,
    _make_client,
)
from .candidate_filter import build_allowlist, codes_to_clause_idx, parse_date
from .extract_features import PDDFeatures, _extract_json_object, _validate as _validate_features, PROMPT_TEMPLATE as FEATURE_PROMPT
from .graph_filter import filter_candidates
from .recommend_v2 import _extract_json_array
from .score_fusion import FusionConfig, fuse
from . import guidance as _guidance

logger = logging.getLogger(__name__)
_MAXCHARS = int(os.environ.get("EVAL_TEXT_MAXCHARS", "6000"))  # rich-mode window; default 6000 = baseline


def _guidance_flags() -> tuple[bool, bool, bool]:
    """(filter, rerank_scale, rerank_clauses) from env — all OFF = baseline. Read per-call
    so the ablation harness can toggle conditions without re-import."""
    return (
        os.environ.get("V281_GUIDANCE_FILTER") == "1",
        os.environ.get("V281_GUIDANCE_RERANK_SCALE") == "1",
        os.environ.get("V281_GUIDANCE_RERANK_CLAUSES") == "1",
    )


GENERATION_PROMPT_V281 = """You are a carbon-credit methodology classifier.

A carbon-project's design document is below, along with structured features extracted from it and the most-relevant clauses retrieved from a corpus of registered methodologies (Gold Standard / CDM / Verra etc.).

# Project features (extracted)

- GHG species: {ghg_species}
- Sectoral scope: {sectoral_scope}
- Technology: {technology}
- Country: {country_iso}
- Scale: {scale}

# Project text (truncated)

{project_text}

# Candidate methodology clauses (retrieved, ordered by relevance)

Each candidate is annotated with `(N projects)` indicating how many other carbon-credit projects have applied that methodology.

**IMPORTANT — popularity is a CRITICAL signal:**
- HIGH project counts (>50) = well-established, broadly applicable. Default for typical projects.
- LOW project counts (<10) = niche / registry-specific. Only when project text contains specific markers.

{candidate_clauses}

# Decision rules (in priority order)

**Rule 1 — SCALE DISCRIMINATION (CRITICAL for renewable energy projects):**

If the extracted `scale` field is `small` OR the project text explicitly mentions any of:
- "small-scale", "small scale CDM", "SSC-CDM"
- installed capacity ≤ 15 MW (e.g., "5 MW wind", "10 MW solar")
- annual emission reductions ≤ 60 kt CO2e/yr

→ STRONGLY PREFER **AMS-*** (small-scale CDM) methodologies over ACM/AM (large-scale).

For renewable electricity specifically: **AMS-I.D.** is the small-scale equivalent of ACM0002. Both apply to grid-connected renewable generation; the scale determines which is correct.

Similarly:
- AMS-I.E. (small-scale wood-fuel switch) vs AM0072 (large-scale)
- AMS-II.G. (small-scale energy efficiency in thermal biomass) vs ACM0021 (large)

**Rule 2 — SPECIFIC METHODOLOGY MENTION:**

If the project text explicitly names a specific methodology code, registry, or niche framework (e.g., "Verra Global Carbon Council", "Gold Standard EN-002"), pick that.

**Rule 3 — POPULARITY DEFAULT:**

For ambiguous large-scale projects without scale-specific markers, prefer high-popularity methodologies. A 158-project methodology is far more likely than a 5-project alternative for typical projects.

**Rule 4 — RETRIEVAL ORDER:**

When all else equal, the candidates are ordered by retrieval relevance — treat this as a soft prior.

# Output

ONLY a JSON array of EXACTLY 5 objects:

  [
    {{"code": "AMS-I.D.", "rationale": "<1 sentence>"}},
    ...
  ]

Use codes EXACTLY as they appear in the candidate list. Do not invent codes.
"""


class GraphRAGv281OpenAIBaseline(GraphRAGv28OpenAIBaseline):
    name = "graphrag-v281-openai"

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
                want_scale=g_scale, want_clauses=g_clauses, openai_client=self.openai,
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
            resp = self.openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
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
