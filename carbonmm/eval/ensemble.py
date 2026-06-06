"""Per-PDD ensemble of multiple variants via RRF or majority voting.

Combines predicted top-5 lists from multiple systems on the SAME PDDs.
Goal: capture each system's strength while reducing individual errors.

Usage:
    python -m carbonmm.eval.ensemble \
        --variants graphrag-v25 naive-rag --split test --n 535 --seed 42 \
        --method rrf
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, Counter
from pathlib import Path
from statistics import mean

RESULTS = Path(__file__).resolve().parents[1] / "results"
RRF_K = 60


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0: return (0.0, 0.0)
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return ((center - margin) / denom, (center + margin) / denom)


def rrf_combine(rankings: list[list[str]], top_k: int = 5) -> list[str]:
    """Reciprocal Rank Fusion of multiple top-k lists."""
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, code in enumerate(ranking, 1):
            scores[code] += 1.0 / (RRF_K + rank)
    return [c for c, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]]


def vote_combine(rankings: list[list[str]], top_k: int = 5) -> list[str]:
    """Borda voting: each rank-i code gets (k - i + 1) points."""
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, code in enumerate(ranking, 1):
            scores[code] += max(0, 5 - rank + 1)
    return [c for c, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=535)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--method", choices=["rrf", "vote"], default="rrf")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    # Load each variant
    variants_data = {}
    for v in args.variants:
        path = RESULTS / f"{v}-{args.split}-n{args.n}-seed{args.seed}.json"
        if not path.exists():
            print(f"[skip] missing: {path}")
            continue
        variants_data[v] = json.load(path.open())

    if not variants_data:
        raise SystemExit("no variant results loaded")

    # Build per-PDD predictions table
    gid_to_preds = {}  # gid -> {variant: top5_list}
    gt_by_gid = {}
    reg_by_gid = {}
    first_variant = list(variants_data)[0]
    for p in variants_data[first_variant]["per_pdd"]:
        gid_to_preds[p["gid"]] = {first_variant: p["predicted_top5"]}
        gt_by_gid[p["gid"]] = p["gt_label"]
        reg_by_gid[p["gid"]] = p["registry"]

    for v in args.variants[1:]:
        if v not in variants_data:
            continue
        for p in variants_data[v]["per_pdd"]:
            if p["gid"] in gid_to_preds:
                gid_to_preds[p["gid"]][v] = p["predicted_top5"]

    # Apply ensemble
    ensemble_preds = []
    for gid, gt in gt_by_gid.items():
        rankings = [gid_to_preds[gid].get(v, []) for v in args.variants if v in variants_data]
        rankings = [r for r in rankings if r]
        if not rankings:
            continue
        if args.method == "rrf":
            top5 = rrf_combine(rankings, top_k=args.top_k)
        else:
            top5 = vote_combine(rankings, top_k=args.top_k)
        rank = next((i + 1 for i, c in enumerate(top5) if c == gt), None)
        ensemble_preds.append({
            "gid": gid,
            "registry": reg_by_gid[gid],
            "gt_label": gt,
            "predicted_top5": top5,
            "gt_rank": rank,
        })

    # Compute metrics
    n = len(ensemble_preds)
    top1 = sum(1 for p in ensemble_preds if p["gt_rank"] == 1) / max(1, n)
    top5 = sum(1 for p in ensemble_preds if isinstance(p["gt_rank"], int) and p["gt_rank"] <= 5) / max(1, n)
    rr = [1.0 / p["gt_rank"] if isinstance(p["gt_rank"], int) and p["gt_rank"] <= 5 else 0 for p in ensemble_preds]
    mrr = mean(rr) if rr else 0
    ci = wilson_ci(top1, n)

    print(f"\n{'='*60}")
    print(f" Ensemble: {' + '.join(args.variants)} via {args.method.upper()}")
    print(f"{'='*60}")
    print(f"n = {n}")
    print(f"top-1 = {top1:.4f}  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"top-5 = {top5:.4f}")
    print(f"MRR   = {mrr:.4f}")

    # Per-registry
    for reg in ["GS", "VCS"]:
        subset = [p for p in ensemble_preds if p["registry"] == reg]
        if not subset:
            continue
        sn = len(subset)
        s_top1 = sum(1 for p in subset if p["gt_rank"] == 1) / sn
        s_top5 = sum(1 for p in subset if isinstance(p["gt_rank"], int) and p["gt_rank"] <= 5) / sn
        print(f"  {reg}: n={sn}  top-1={s_top1:.4f}  top-5={s_top5:.4f}")

    # Compare each constituent's top-1 on common subset for context
    print(f"\nConstituent variants (for reference):")
    common_gids = {p["gid"] for p in ensemble_preds}
    for v in args.variants:
        if v not in variants_data:
            continue
        d = variants_data[v]
        sub = [p for p in d["per_pdd"] if p["gid"] in common_gids]
        if not sub: continue
        t1 = sum(1 for p in sub if p["gt_rank"] == 1) / len(sub)
        t5 = sum(1 for p in sub if isinstance(p["gt_rank"], int) and p["gt_rank"] <= 5) / len(sub)
        print(f"  {v:24s} n={len(sub)}  top-1={t1:.4f}  top-5={t5:.4f}")


if __name__ == "__main__":
    main()
