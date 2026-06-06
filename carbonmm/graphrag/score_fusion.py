"""Score fusion: rerank candidates by α·BM25 + β·dense + γ·graph_match.

The retrieval head (`retrieve.py`) returns top-30 clauses by RRF over
BM25 ∪ dense. The graph filter (`graph_filter.py`) produces per-methodology
match counts (0-4). This module rebuilds per-methodology scores from clauses,
normalizes each channel to [0, 1] by min-max, and computes the weighted sum.

Hard-excluded methodologies (from graph_filter) are dropped before fusion.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FusionConfig:
    # Default = "bm25_heavy" config selected by the validation sweep
    # (sweep: bm25_heavy top1=0.280 vs default 0.240 vs graph_heavy 0.240 vs graph_off 0.220)
    alpha: float = 0.5   # BM25 weight
    beta: float = 0.3    # dense weight
    gamma: float = 0.2   # graph_match weight


def _aggregate_by_code(
    bm25_top: list[tuple[int, float]],
    dense_top: list[tuple[int, float]],
    idx_to_code: dict[int, str],
) -> tuple[dict[str, float], dict[str, float]]:
    """Roll up per-clause BM25 / dense scores to per-methodology max."""
    bm25_by_code: dict[str, float] = defaultdict(float)
    dense_by_code: dict[str, float] = defaultdict(float)
    for i, s in bm25_top:
        c = idx_to_code.get(i)
        if c:
            bm25_by_code[c] = max(bm25_by_code[c], s)
    for i, s in dense_top:
        c = idx_to_code.get(i)
        if c:
            dense_by_code[c] = max(dense_by_code[c], s)
    return dict(bm25_by_code), dict(dense_by_code)


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    if hi - lo < 1e-9:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def fuse(
    bm25_top: list[tuple[int, float]],
    dense_top: list[tuple[int, float]],
    idx_to_code: dict[int, str],
    filter_results: dict,  # code → GraphFilterResult
    config: FusionConfig | None = None,
    top_k: int = 5,
    scale_multipliers: dict[str, float] | None = None,  # NEW: soft guidance down-weight (code→[0,1]); None=off
) -> list[tuple[str, float, dict]]:
    """Return top_k (code, fused_score, components) tuples.

    `scale_multipliers` (guidance-KG soft scale signal) multiplies each candidate's
    fused score; missing codes default to 1.0. None → byte-identical pre-guidance baseline.
    """
    cfg = config or FusionConfig()
    sm = scale_multipliers or {}

    bm25_raw, dense_raw = _aggregate_by_code(bm25_top, dense_top, idx_to_code)
    bm25_norm = _minmax(bm25_raw)
    dense_norm = _minmax(dense_raw)

    # Drop hard-excluded; collect graph match counts
    graph_match_raw = {}
    excluded = set()
    for code, r in filter_results.items():
        if r.excluded:
            excluded.add(code)
            continue
        graph_match_raw[code] = float(r.match_count)
    graph_norm = _minmax(graph_match_raw)

    # Union of candidates from all channels (less excluded)
    candidates = (set(bm25_norm) | set(dense_norm) | set(graph_norm)) - excluded

    fused: list[tuple[str, float, dict]] = []
    for c in candidates:
        b = bm25_norm.get(c, 0.0)
        d = dense_norm.get(c, 0.0)
        g = graph_norm.get(c, 0.0)
        mult = sm.get(c, 1.0)
        score = (cfg.alpha * b + cfg.beta * d + cfg.gamma * g) * mult
        fused.append(
            (
                c,
                score,
                {
                    "bm25_norm": round(b, 4),
                    "dense_norm": round(d, 4),
                    "graph_match_norm": round(g, 4),
                    "graph_match_raw": int(graph_match_raw.get(c, 0)),
                    "scale_mult": round(mult, 3),
                },
            )
        )
    fused.sort(key=lambda x: -x[1])
    return fused[:top_k]
