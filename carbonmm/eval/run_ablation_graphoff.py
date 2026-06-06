"""Ablation: GraphRAG with γ=0 (graph signal completely off).

Differs from `harness.py --baseline graphrag` only in the FusionConfig passed
to GraphRAGBaseline (gamma=0.0 instead of the default 0.2). Hard-exclude
rule (AMS-* on scale=large) IS still applied — we ablate only the additive
score signal, not the constraint enforcement. This isolates "how much of
the GraphRAG win is from the graph-match additive score vs the hard-exclude
filter?"
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from .harness import (
    BASELINE_REGISTRY,
    RESULTS_ROOT,
    compute_metrics,
    load_eval_pdds,
)

logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from ..graphrag.recommend import GraphRAGBaseline
    from ..graphrag.score_fusion import FusionConfig

    cfg = FusionConfig(alpha=0.5, beta=0.5, gamma=0.0)
    logger.info("loading GraphRAGBaseline with γ=0.0 (graph signal off) ...")
    t0 = time.time()
    b = GraphRAGBaseline(fusion=cfg)
    setup_s = time.time() - t0

    pdds = load_eval_pdds(args.n, args.seed, split=args.split)
    logger.info("loaded %d PDDs (split=%s)", len(pdds), args.split)

    per_pdd = []
    t_query = time.time()
    for gid, reg, label, text, _meta in pdds:
        try:
            preds = b.predict(text, top_k=args.top_k)
        except Exception as e:
            logger.warning("predict failed gid=%s: %s", gid, e)
            preds = []
        codes = [c for c, _ in preds]
        rank = next((i + 1 for i, c in enumerate(codes) if c == label), None)
        per_pdd.append(
            {
                "gid": gid,
                "registry": reg,
                "gt_label": label,
                "predicted_top5": codes,
                "gt_rank": rank,
            }
        )
    query_s = time.time() - t_query

    metrics = compute_metrics(per_pdd, top_k=args.top_k)
    out = {
        "baseline": "graphrag-graphoff",
        "fusion_config": {"alpha": 0.5, "beta": 0.5, "gamma": 0.0},
        "n_actual": len(pdds),
        "seed": args.seed,
        "split": args.split,
        "top_k": args.top_k,
        "setup_s": round(setup_s, 2),
        "query_s": round(query_s, 2),
        "per_query_s": round(query_s / max(1, len(pdds)), 2),
        "metrics": metrics,
        "per_pdd": per_pdd,
    }
    n_actual = len(pdds)
    out_path = (
        RESULTS_ROOT / f"graphrag-graphoff-{args.split}-n{n_actual}-seed{args.seed}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    m = metrics
    print(f"\n=== graphrag-graphoff ({n_actual}/{args.n if args.n else 'all'}) ===")
    print(f"  top1={m.get('top1')}  top5={m.get('top5')}  MRR={m.get('mrr')}")
    if "by_registry" in m:
        for reg, mb in m["by_registry"].items():
            print(f"    {reg:3s}: n={mb['n']:3d}  top1={mb['top1']}  top5={mb['top5']}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
