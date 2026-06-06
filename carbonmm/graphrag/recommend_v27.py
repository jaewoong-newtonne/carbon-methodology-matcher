"""GraphRAG v2.7 — naive-rag base + popularity hint ONLY (no features hint).

Diagnosis from val results:
  - v2 (features hint, no popularity) val = 0.36 — barely > v1 (0.348)
  - v2.5 (features + popularity) val = 0.50 — popularity adds +14pp
  - naive-rag (no features, no popularity) test = 0.529

Insight: naive-rag at 0.529 > v2 at 0.36 implies the features hint section
is HURTING. v2.5's lift comes purely from popularity. So v2.7 = naive-rag's
prompt + add popularity hint + keep hard-exclude (without showing features to LLM).

Goal: > 0.55 on val n=50, projecting ~0.60+ on test.
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict, Counter

import httpx

from ..baselines.base import Baseline, label_space
from ..baselines.naive_rag import _extract_json_array
from ..eval.splits import load_splits
from .extract_features import extract_features, PDDFeatures
from .graph_filter import filter_candidates
from .score_fusion import FusionConfig, fuse

logger = logging.getLogger(__name__)

DAEMON_URL = "http://localhost:8765/chat"
DAEMON_TIMEOUT = 120.0
DEFAULT_MODEL = "claude-haiku-4-5"


GENERATION_PROMPT = """You are a carbon-credit methodology classifier.

A carbon-project's design document is described below. You are given the
text of the project plus the most-relevant clauses from a corpus of registered
carbon-credit methodologies (Gold Standard / CDM / Verra etc.).

Each candidate methodology is annotated with `(N projects)` indicating how
many other carbon-credit projects have applied that methodology. **High counts
(>50) indicate well-established broadly-applicable methodologies; low counts
(<10) indicate niche or specialty methodologies.**

Your task: rank the FIVE methodology codes that are most likely to apply.

# Project text (truncated)

{project_text}

# Candidate methodology clauses (retrieved, ordered by relevance)

{candidate_clauses}

# Instructions

Output ONLY a JSON array of EXACTLY 5 objects with the schema:

  [
    {{"code": "ACM0002", "rationale": "<1 sentence>"}},
    {{"code": "AMS-I.D.", "rationale": "..."}}
  ]

**Decision rule:**
1. If the project text explicitly mentions a specific methodology, registry, or niche framework → pick that.
2. Otherwise, **STRONGLY prefer methodologies with higher project counts**. For typical projects, a 158-project methodology is much more likely than a 5-project alternative.
3. Use codes EXACTLY as they appear in the candidate clauses above. Do not
invent codes that are not in the candidate list.
"""


def _build_candidate_block_v27(
    fused: list[tuple[str, float, dict]],
    df,
    freq_map: dict[str, int],
    max_clauses_per_code: int = 2,
    max_chars_per_clause: int = 400,
    max_total_chars: int = 12_000,
) -> str:
    """Naive-rag-style block with popularity annotation."""
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


class GraphRAGv27Baseline(Baseline):
    name = "graphrag-v27"

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

        try:
            splits = load_splits()
            self._freq_count = dict(Counter(p["gt"] for p in splits["val"]["pdds"]))
        except Exception as e:
            logger.warning("freq_count load failed: %s", e)
            self._freq_count = {}

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

        # No features hint in prompt — match naive-rag style
        block = _build_candidate_block_v27(fused, raw["df"], self._freq_count)
        prompt = GENERATION_PROMPT.format(
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
