"""Verbose single-PDD trace through the full GraphRAG pipeline.

Print each intermediate stage's state — PDD excerpt, retrieved top-N codes,
extracted features, graph filter results, fusion scores, LLM-generated ranking,
GT comparison. Useful for paper qualitative examples and debugging.

Usage:
    python3 -m carbonmm.graphrag.trace_one --pdd-id VCS459
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .extract_features import extract_features
from .graph_filter import filter_candidates
from .recommend import (
    GraphRAGBaseline,
    GENERATION_PROMPT,
    _build_candidate_block,
    _extract_json_array,
)
from .retrieve import HybridRetriever
from .score_fusion import FusionConfig, fuse

EVAL = Path(__file__).resolve().parents[1] / "data" / "eval-pdds"


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(f" {title}")
    print("=" * 78)


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdd-id", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    body = None
    for f in EVAL.rglob(f"{args.pdd_id}.body.json"):
        body = json.loads(f.read_text())
        break
    if body is None:
        raise SystemExit(f"PDD {args.pdd_id} not found")
    text = body.get("full_text") or ""
    gt = body.get("methodology_label")

    hr(f"STAGE 0 — PDD INPUT  ({args.pdd_id}, gt={gt})")
    print(f"length: {len(text)} chars")
    print(f"\nExcerpt (first 500 chars):\n{text[:500]}")

    hr("STAGE 1 — HYBRID RETRIEVE (BM25 ∪ dense → top-30 clauses)")
    R = HybridRetriever.load_default()
    raw = R.query_with_raw(text)
    bm25_codes = [(raw["idx_to_code"][i], s) for i, s in raw["bm25_top"]]
    dense_codes = [(raw["idx_to_code"][i], s) for i, s in raw["dense_top"]]
    bm25_unique, dense_unique = [], []
    seen_b, seen_d = set(), set()
    for c, s in bm25_codes:
        if c not in seen_b:
            bm25_unique.append((c, s))
            seen_b.add(c)
    for c, s in dense_codes:
        if c not in seen_d:
            dense_unique.append((c, s))
            seen_d.add(c)
    print(f"\nBM25 top-10 unique codes (of {len(bm25_codes)} clauses):")
    for c, s in bm25_unique[:10]:
        marker = " ← GT" if c == gt else ""
        print(f"  {c:14s}  score={s:.4f}{marker}")
    print(f"\nDense top-10 unique codes (of {len(dense_codes)} clauses):")
    for c, s in dense_unique[:10]:
        marker = " ← GT" if c == gt else ""
        print(f"  {c:14s}  score={s:.4f}{marker}")

    hr("STAGE 2 — EXTRACT FEATURES (LLM call 1: claude-haiku-4-5)")
    feats = extract_features(text)
    print(json.dumps(feats.to_dict(), indent=2, ensure_ascii=False))

    hr("STAGE 3 — GRAPH FILTER (rule-based, 0-4 match count)")
    candidate_codes = list(
        {raw["idx_to_code"][i] for i, _ in raw["bm25_top"]} |
        {raw["idx_to_code"][i] for i, _ in raw["dense_top"]}
    )
    filter_results = filter_candidates(candidate_codes, feats)
    print(f"\n{len(candidate_codes)} unique candidate codes after retrieval")
    excluded = [c for c, r in filter_results.items() if r.excluded]
    print(f"Hard-excluded: {len(excluded)} ({excluded[:5]}{'...' if len(excluded) > 5 else ''})")
    print(f"\nTop-15 by match_count:")
    sorted_codes = sorted(filter_results.items(), key=lambda kv: -kv[1].match_count)[:15]
    print(f"  {'code':14s}  {'excl':6s}  {'match':5s}  scope/tech/ghg/scale")
    for c, r in sorted_codes:
        d = r.match_detail
        flags = (
            ("S" if d["scope"] else "-") +
            ("T" if d["technology"] else "-") +
            ("G" if d["ghg"] else "-") +
            ("L" if d["scale"] else "-")
        )
        marker = " ← GT" if c == gt else ""
        print(f"  {c:14s}  {('YES' if r.excluded else '-'):6s}  {r.match_count}/4    {flags}{marker}")

    hr("STAGE 4 — SCORE FUSION  (α·BM25 + β·dense + γ·graph_match)")
    cfg = FusionConfig()
    fused = fuse(
        raw["bm25_top"], raw["dense_top"], raw["idx_to_code"],
        filter_results, config=cfg, top_k=10,
    )
    print(f"\nFusion config: α(bm25)={cfg.alpha}  β(dense)={cfg.beta}  γ(graph)={cfg.gamma}")
    print(f"\nTop-10 fused:")
    print(f"  {'#':>2}  {'code':14s}  {'score':>7s}  bm25_n  dense_n  graph/4")
    for rank, (c, s, comp) in enumerate(fused, 1):
        marker = " ← GT" if c == gt else ""
        print(
            f"  {rank:>2}  {c:14s}  {s:>7.4f}  "
            f"{comp['bm25_norm']:.3f}   {comp['dense_norm']:.3f}    "
            f"{comp['graph_match_raw']}/4{marker}"
        )

    hr("STAGE 5 — GENERATE (LLM call 2: claude-haiku-4-5 ranks top-5 with citations)")
    block = _build_candidate_block(fused, raw["df"])
    print(f"\n[Candidate block sent to LLM, {len(block)} chars]")
    print(f"\n--- prompt (truncated to 600 chars after candidate block start) ---")
    full_prompt = GENERATION_PROMPT.format(
        ghg_species=", ".join(feats.ghg_species) or "unknown",
        sectoral_scope=feats.sectoral_scope or "unknown",
        technology=feats.technology or "unknown",
        country_iso=feats.country_iso or "unknown",
        scale=feats.scale,
        project_text=text[:5_000],
        candidate_block=block,
    )
    print(full_prompt[:1200] + "\n... (truncated)")

    b = GraphRAGBaseline()
    raw_text = b._call_daemon(full_prompt)
    print(f"\n--- raw LLM output ({len(raw_text)} chars) ---")
    print(raw_text[:1500])

    parsed = _extract_json_array(raw_text)
    print(f"\n--- parsed JSON ({len(parsed)} items) ---")
    for rank, item in enumerate(parsed, 1):
        code = (item or {}).get("code", "?")
        cites = (item or {}).get("cited_clause_ids", [])
        rationale = (item or {}).get("rationale", "")
        marker = " ✓ GT" if code == gt else ""
        print(f"  {rank}. {code:14s} cites={cites}{marker}")
        if rationale:
            print(f"     → {rationale}")

    hr("FINAL — GT comparison")
    final_preds = b.predict(text, top_k=args.top_k)
    gt_rank = next((i + 1 for i, (c, _) in enumerate(final_preds) if c == gt), ">5")
    print(f"\nGT methodology: {gt}")
    print(f"GraphRAG predicted top-{args.top_k}:")
    for i, (c, s) in enumerate(final_preds, 1):
        marker = " ✓" if c == gt else ""
        print(f"  {i}. {c:14s}  {s:.4f}{marker}")
    print(f"\nGT rank: {gt_rank}")


if __name__ == "__main__":
    main()
