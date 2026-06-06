"""GraphRAG v2 — addresses LLM-rerank-demote bug found in test 535 eval.

v1 diagnosis (graphrag-test-n535-seed42.json + per-PDD comparison vs naive-rag):
  - 33.8% of test PDDs had GT in graphrag top-5 but NOT top-1 → LLM rerank demote
  - 138 PDDs naive-rag got top-1 but graphrag did not; ACM0002 = 60% of losses
  - top-5 also worse than naive-rag (0.686 vs 0.753) — LLM dropping correct GT from top-5

v2 changes (4):
  1. Drop graph_match additive score channel (γ=0). Hard-exclude rule retained.
  2. Expand candidate block to top-30 codes (matches naive-rag for recall).
  3. Strip score noise from prompt — candidates are listed plain with clauses,
     no `(bm25=X dense=Y graph_match=N/4)` annotations to distract the LLM.
  4. Keep extract_features → use as `# Project features` hint in prompt
     (structured signal that naive-rag lacks).
  5. Keep claude-haiku-4-5 for fair A/B with naive-rag (which also uses Haiku);
     Sonnet escalation reserved if Haiku v2 still ≮ goal.
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict

import httpx

from ..baselines.base import Baseline, label_space
from .extract_features import extract_features
from .graph_filter import filter_candidates
from .score_fusion import FusionConfig, fuse

logger = logging.getLogger(__name__)

DAEMON_URL = "http://localhost:8765/chat"
DAEMON_TIMEOUT = 120.0
DEFAULT_MODEL = "claude-haiku-4-5"  # parity with naive-rag for fair A/B; Sonnet escalation reserved if Haiku v2 ≮ goal


GENERATION_PROMPT = """You are a carbon-credit methodology classifier.

A carbon-project's design document is described below, along with structured features extracted from it and the most-relevant clauses retrieved from a corpus of registered methodologies (Gold Standard / CDM / Verra etc.).

Your task: rank the FIVE methodology codes that are most likely to apply to this project, ordered from most to least likely.

# Project features (extracted)

- GHG species: {ghg_species}
- Sectoral scope: {sectoral_scope}
- Technology: {technology}
- Country: {country_iso}
- Scale: {scale}

# Project text (truncated)

{project_text}

# Candidate methodology clauses (retrieved, ordered by relevance)

{candidate_clauses}

# Instructions

Pick the FIVE most likely methodology codes. The candidates are listed in approximate order of retrieval relevance — treat this as a strong prior and only deviate when the project text clearly contradicts a high-ranked candidate. Be careful with similar codes (e.g., ACM0002 vs AMS-I.D. — both are renewable-electricity but differ in scale).

Output ONLY a JSON array of EXACTLY 5 objects:

  [
    {{"code": "ACM0002", "rationale": "<1 sentence>"}},
    ...
  ]

Use codes EXACTLY as they appear in the candidate list. Do not invent codes outside the candidate list.
"""


def _extract_json_array(text: str) -> list[dict]:
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    start = text.find("[")
    if start < 0:
        return []
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except Exception:
                    return []
    return []


def _build_candidate_block_v2(
    fused: list[tuple[str, float, dict]],
    df,
    max_clauses_per_code: int = 2,
    max_chars_per_clause: int = 400,
    max_total_chars: int = 12_000,
) -> str:
    """Naive-rag-style clause listing — NO scores in prompt, ordered by fusion rank.

    Format per code:
      - [code] (clause_type) clause_text (up to 400 chars)
    """
    lines = []
    used = 0
    for code, _score, _comp in fused:
        rows = df[df["code"] == code]
        n_emitted = 0
        for row in rows.itertuples(index=False):
            txt = (row.clause_text or "").replace("\n", " ")[:max_chars_per_clause]
            line = f"- [{code}] ({row.clause_type}) {txt}"
            if used + len(line) > max_total_chars:
                return "\n".join(lines) + "\n- ... (truncated)"
            lines.append(line)
            used += len(line) + 1
            n_emitted += 1
            if n_emitted >= max_clauses_per_code:
                break
    return "\n".join(lines)


class GraphRAGv2Baseline(Baseline):
    name = "graphrag-v2"

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
        # v2 default: γ=0 (no additive graph signal — only hard-exclude retained)
        self.fusion_cfg = fusion or FusionConfig(alpha=0.5, beta=0.5, gamma=0.0)
        self.model = model
        self.daemon_url = daemon_url
        self.top_k_after_fusion = top_k_after_fusion
        self.valid_codes = set(label_space())
        self.client = httpx.Client(timeout=DAEMON_TIMEOUT)

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
            logger.warning("extract failed: %s — proceeding without features", e)
            from .extract_features import PDDFeatures
            feats = PDDFeatures()

        candidate_codes = list(
            {raw["idx_to_code"][i] for i, _ in raw["bm25_top"]} |
            {raw["idx_to_code"][i] for i, _ in raw["dense_top"]}
        )
        filter_results = filter_candidates(candidate_codes, feats)

        fused = fuse(
            raw["bm25_top"],
            raw["dense_top"],
            raw["idx_to_code"],
            filter_results,
            config=self.fusion_cfg,
            top_k=self.top_k_after_fusion,
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
            logger.warning("daemon recommend failed: %s — falling back to fusion order", e)
            return [(c, s) for c, s, _ in fused[:top_k]]

        parsed = _extract_json_array(text)
        out: list[tuple[str, float]] = []
        seen = set()
        for rank, item in enumerate(parsed, 1):
            code = (item or {}).get("code")
            if not code or code in seen:
                continue
            if code not in self.valid_codes:
                logger.debug("graphrag-v2 hallucinated code: %s", code)
                continue
            out.append((code, 1.0 / rank))
            seen.add(code)
            if len(out) >= top_k:
                break

        if not out:
            logger.info("graphrag-v2: empty LLM output, falling back to fusion order")
            return [(c, s) for c, s, _ in fused[:top_k]]
        return out
