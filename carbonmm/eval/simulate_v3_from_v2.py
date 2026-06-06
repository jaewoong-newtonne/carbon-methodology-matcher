"""Simulate v3's RRF-hybrid output from v2's per-PDD predictions + a fresh
fusion-only ranking. No LLM calls required — pure post-processing.

This lets us quickly estimate v3's top-1/top-5 without actually running v3
on the daemon (which would take ~30 min on val n=50). The key insight:

  - v2's `predicted_top5` is the LLM ranking
  - fusion-only's `predicted_top5` is the fusion ranking
  - v3 = RRF-combine of these two with w_F=0.6, w_L=0.4

This isn't perfectly accurate (v3's actual fused list shown to LLM differs
slightly from v2's because of the freq_prior pre-reorder) but gives a
directional estimate within ~2pp.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

RESULTS = Path(__file__).resolve().parents[1] / "results"

RRF_K = 60
W_F = 0.6
W_L = 0.4


def rrf_combine(fusion_ranking: list[str], llm_ranking: list[str], top_k: int = 5):
    scores: dict[str, float] = defaultdict(float)
    codes = set(fusion_ranking) | set(llm_ranking)
    for c in codes:
        rf = fusion_ranking.index(c) + 1 if c in fusion_ranking else len(fusion_ranking) + 1
        rl = llm_ranking.index(c) + 1 if c in llm_ranking else len(llm_ranking) + 1
        scores[c] = W_F / (rf + RRF_K) + W_L / (rl + RRF_K)
    return [c for c, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="graphrag-v2", help="LLM-side variant whose predictions are reranked via RRF")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    fname_v = RESULTS / f"{args.variant}-{args.split}-n{args.n}-seed{args.seed}.json"
    fname_fo = RESULTS / f"graphrag-fusion-only-test-n535-seed42.json"  # fusion-only test
    # If we don't have fusion-only on same split, fall back to test (gids likely differ)
    fname_fo_alt = RESULTS / f"graphrag-fusion-only-{args.split}-n{args.n}-seed{args.seed}.json"
    if fname_fo_alt.exists():
        fname_fo = fname_fo_alt

    if not fname_v.exists():
        raise SystemExit(f"missing {fname_v}")
    if not fname_fo.exists():
        raise SystemExit(f"missing {fname_fo}")

    v = json.loads(fname_v.read_text())
    fo = json.loads(fname_fo.read_text())

    fo_by_gid = {p["gid"]: p["predicted_top5"] for p in fo["per_pdd"]}

    sim_preds = []
    for p in v["per_pdd"]:
        gid = p["gid"]
        llm = p["predicted_top5"] or []
        fusion = fo_by_gid.get(gid, [])
        if not fusion:
            sim_top5 = llm[:5]
        else:
            sim_top5 = rrf_combine(fusion, llm, top_k=5)
        rank = next((i + 1 for i, c in enumerate(sim_top5) if c == p["gt_label"]), None)
        sim_preds.append({
            "gid": gid, "registry": p["registry"], "gt_label": p["gt_label"],
            "predicted_top5": sim_top5, "gt_rank": rank,
        })

    n = len(sim_preds)
    top1 = sum(1 for p in sim_preds if p["gt_rank"] == 1)
    top5 = sum(1 for p in sim_preds if isinstance(p["gt_rank"], int) and p["gt_rank"] <= 5)
    rr = [(1.0 / p["gt_rank"]) if isinstance(p["gt_rank"], int) and p["gt_rank"] <= 5 else 0 for p in sim_preds]
    mrr = mean(rr) if rr else 0
    print(f"\n=== v3 simulation (RRF on {args.variant}) ===")
    print(f"n={n}  top1={top1/n:.4f}  top5={top5/n:.4f}  MRR={mrr:.4f}")

    # Compare to v2 alone and fusion-only alone on same gids
    v_top1 = sum(1 for p in v["per_pdd"] if p["gt_rank"] == 1) / len(v["per_pdd"])
    fo_in_gids = [p for p in fo["per_pdd"] if p["gid"] in {q["gid"] for q in sim_preds}]
    fo_top1 = sum(1 for p in fo_in_gids if p["gt_rank"] == 1) / max(1, len(fo_in_gids))
    print(f"\nReference:")
    print(f"  {args.variant} alone     top1={v_top1:.4f}")
    print(f"  fusion-only on same gids top1={fo_top1:.4f}  (n={len(fo_in_gids)})")


if __name__ == "__main__":
    main()
