"""GraphRAG v2.5 — explicit popularity hint in candidate block.

Hypothesis: LLM's confusion of ACM0002 ↔ GCCM001 is because the LLM has no
sense of which methodology is COMMONLY APPLICABLE vs which is NICHE. Adding
project-count annotations to each candidate makes the popularity prior explicit:

  - [ACM0002] (158 projects) (applicability) Grid-connected ...
  - [GCCM001] (5 projects) (applicability) Renewable energy ...

This pushes the LLM to prefer the high-prior methodology when textual signals
are ambiguous. The counts come from val split (paper-clean — no test leakage).

Inherits v2's other improvements:
  - 30 candidates
  - 400-char clauses
  - no score noise
  - structured features hint
  - γ=0 (no graph_match score channel; hard-exclude retained)
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter

import httpx

from ..baselines.base import Baseline, label_space
from ..eval.splits import load_splits
from .extract_features import extract_features, PDDFeatures
from .graph_filter import filter_candidates
from .recommend_v2 import _extract_json_array
from .score_fusion import FusionConfig, fuse

logger = logging.getLogger(__name__)

DAEMON_URL = "http://localhost:8765/chat"
DAEMON_TIMEOUT = 120.0
DEFAULT_MODEL = "claude-haiku-4-5"


GENERATION_PROMPT = """You are a carbon-credit methodology classifier.

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

Each candidate is annotated with `(N projects)` indicating how many other
carbon-credit projects in our reference set have applied that methodology. A
high project count signals a widely-applicable methodology; a low count
signals a niche or specialty one.

{candidate_clauses}

# Instructions

Pick the FIVE most likely methodology codes. The candidates are listed in
approximate order of retrieval relevance. **When the project text is ambiguous
between two semantically similar methodologies, prefer the one with the higher
project count.** Otherwise stay close to the retrieval order.

Output ONLY a JSON array of EXACTLY 5 objects:

  [
    {{"code": "ACM0002", "rationale": "<1 sentence>"}},
    ...
  ]

Use codes EXACTLY as they appear in the candidate list. Do not invent codes.
"""


def _build_candidate_block_v25(
    fused: list[tuple[str, float, dict]],
    df,
    freq_map: dict[str, int],
    max_clauses_per_code: int = 2,
    max_chars_per_clause: int = 400,
    max_total_chars: int = 12_000,
) -> str:
    """Like v2's block but annotate each candidate with `(N projects)`."""
    lines = []
    used = 0
    for code, _score, _comp in fused:
        cnt = freq_map.get(code, 0)
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


class GraphRAGv25Baseline(Baseline):
    name = "graphrag-v25"

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
        self.fusion_cfg = fusion or FusionConfig(alpha=0.5, beta=0.5, gamma=0.0)
        self.model = model
        self.daemon_url = daemon_url
        self.top_k_after_fusion = top_k_after_fusion
        self.valid_codes = set(label_space())
        self.client = httpx.Client(timeout=DAEMON_TIMEOUT)

        # Popularity prior from val labels (paper-clean — no test leakage)
        try:
            splits = load_splits()
            self._freq_count = dict(Counter(p["gt"] for p in splits["val"]["pdds"]))
        except Exception as e:
            logger.warning("freq_count load failed: %s — defaulting to empty", e)
            self._freq_count = {}

    def _call_daemon(self, prompt: str) -> str:
        import time as _t
        last_exc = None
        for attempt in range(3):
            try:
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
            except httpx.HTTPStatusError as e:
                last_exc = e
                if e.response.status_code in (500, 502, 503) and attempt < 2:
                    _t.sleep(3 + attempt * 2)
                    continue
                raise
        raise last_exc

    def predict(self, pdd_text: str, top_k: int = 5) -> list[tuple[str, float]]:
        try:
            raw = self.R.query_with_raw(pdd_text)
        except Exception as e:
            logger.warning("retrieve failed: %s", e)
            return []

        try:
            feats = extract_features(pdd_text)
        except Exception as e:
            logger.warning("extract failed: %s", e)
            feats = PDDFeatures()

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

        block = _build_candidate_block_v25(fused, raw["df"], self._freq_count)
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
            logger.warning("daemon failed: %s — falling back to fusion", e)
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
            logger.info("v2.5: empty LLM output, falling back to fusion order")
            return [(c, s) for c, s, _ in fused[:top_k]]
        return out
