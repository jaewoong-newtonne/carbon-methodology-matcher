"""GraphRAG full pipeline: retrieve → extract → filter → fuse → generate.

This is the system we compare to all 5 baselines in the paper. The architecture:

  1. HYBRID RETRIEVE (`retrieve.py`)        BM25 ∪ dense → top-30 clauses
  2. EXTRACT (`extract_features.py`)        self-hosted inference endpoint → structured features
  3. GRAPH FILTER (`graph_filter.py`)       hard-exclude + match-count rule-based
  4. SCORE FUSION (`score_fusion.py`)       α·BM25 + β·dense + γ·graph_match
  5. GENERATE (this file)                   Sonnet via daemon ranks top-5 with citations

Differs from `baselines/naive_rag.py` by 4 things:
  - structured feature extraction step
  - graph-filter hard-exclude (AMS-* + scale=large, etc.)
  - score fusion using graph_match_count as 3rd channel
  - generation prompt receives the filtered + reranked candidate list
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
DEFAULT_MODEL = "claude-haiku-4-5"


GENERATION_PROMPT = """You are a carbon-credit methodology recommender. The project below has been pre-filtered against a knowledge graph; the candidate methodologies provided are the ones that pass hard constraints (sectoral scope, scale, technology, gas).

# Extracted project features

- GHG species: {ghg_species}
- Sectoral scope: {sectoral_scope}
- Technology: {technology}
- Country: {country_iso}
- Scale: {scale}

# Project text (truncated)

{project_text}

# Ranked candidate methodologies (after graph filter + score fusion)

{candidate_block}

# Instructions

Choose the 5 methodologies from the candidate list above that best apply to this project, ordered from most to least likely. For each, cite at least one clause-id from the list as supporting evidence.

Output ONLY a JSON array of EXACTLY 5 objects:

  [
    {{"code": "ACM0002", "cited_clause_ids": ["ACM0002-app-3"], "rationale": "<1 sentence>"}},
    ...
  ]

Use codes EXACTLY as they appear above. Do not invent codes outside the candidate list.
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


def _build_candidate_block(
    fused: list[tuple[str, float, dict]],
    df,
    max_clauses_per_code: int = 2,
    max_chars: int = 8_000,
) -> str:
    """Render the fused candidate list + a few example clauses per code."""
    lines = []
    used = 0
    for code, score, comp in fused:
        rows = df[df["code"] == code]
        snippets = []
        for row in rows.itertuples(index=False):
            txt = (row.clause_text or "").replace("\n", " ")[:300]
            snippets.append(f"[{row.clause_id}] ({row.clause_type}) {txt}")
            if len(snippets) >= max_clauses_per_code:
                break
        line = (
            f"{code}  score={score:.3f}  "
            f"(bm25={comp['bm25_norm']} dense={comp['dense_norm']} "
            f"graph_match={comp['graph_match_raw']}/4)\n"
            + "\n".join("    " + s for s in snippets)
        )
        if used + len(line) > max_chars:
            return "\n\n".join(lines) + "\n\n... (truncated)"
        lines.append(line)
        used += len(line)
    return "\n\n".join(lines)


class GraphRAGBaseline(Baseline):
    name = "graphrag"

    def __init__(
        self,
        retriever=None,
        fusion: FusionConfig | None = None,
        model: str = DEFAULT_MODEL,
        daemon_url: str = DAEMON_URL,
        top_k_after_fusion: int = 10,
    ):
        if retriever is None:
            from .retrieve import HybridRetriever
            retriever = HybridRetriever.load_default()
        self.R = retriever
        self.fusion_cfg = fusion or FusionConfig()
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
        # 1-2. retrieve + extract (independent — extract can fail without blocking retrieve)
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

        # 3. graph filter — applies to all unique codes seen in raw retrieval
        candidate_codes = list({raw["idx_to_code"][i] for i, _ in raw["bm25_top"]} |
                              {raw["idx_to_code"][i] for i, _ in raw["dense_top"]})
        filter_results = filter_candidates(candidate_codes, feats)

        # 4. fusion
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

        # 5. generate
        block = _build_candidate_block(fused, raw["df"])
        prompt = GENERATION_PROMPT.format(
            ghg_species=", ".join(feats.ghg_species) or "unknown",
            sectoral_scope=feats.sectoral_scope or "unknown",
            technology=feats.technology or "unknown",
            country_iso=feats.country_iso or "unknown",
            scale=feats.scale,
            project_text=pdd_text[:5_000],
            candidate_block=block,
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
                logger.debug("graphrag hallucinated code: %s", code)
                continue
            out.append((code, 1.0 / rank))
            seen.add(code)
            if len(out) >= top_k:
                break

        # Fallback to fusion order if LLM output mangled
        if not out:
            logger.info("graphrag: empty LLM output, falling back to fusion order")
            return [(c, s) for c, s, _ in fused[:top_k]]
        return out


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdd-id", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    from pathlib import Path
    EVAL = Path(__file__).resolve().parents[1] / "data" / "eval-pdds"
    body = None
    for f in EVAL.rglob(f"{args.pdd_id}.body.json"):
        body = json.loads(f.read_text())
        break
    if body is None:
        raise SystemExit(f"PDD {args.pdd_id} not found")
    text = body.get("full_text") or ""
    gt = body.get("methodology_label")

    b = GraphRAGBaseline()
    preds = b.predict(text, top_k=args.top_k)
    rank = next((i + 1 for i, (c, _) in enumerate(preds) if c == gt), '>5')
    print(f"\nPDD {args.pdd_id} (gt={gt}) — top-{args.top_k}:")
    for i, (c, s) in enumerate(preds, 1):
        marker = " ✓" if c == gt else ""
        print(f"  {i}. {c:14s}  {s:.4f}{marker}")
    print(f"\nGT rank = {rank}")
