"""Parallel harness — processes multiple PDDs concurrently via ThreadPoolExecutor.

Daemon (the inference daemon) has MAX_CONCURRENT=2. Sequential harness uses 1 slot
at a time → underutilizes. With 2 worker threads here, both daemon slots stay
saturated → ~2x throughput on cold-rerank baselines (graphrag-v2, v25, v3).

Usage: same flags as harness.py + --workers N (default 2).

Note: baseline.predict() must be thread-safe — most current baselines have
read-only state after __init__ (retriever, valid_codes, freq_prior) and use
their own httpx.Client per instance (thread-safe). Verified for v2/v2.5/v3.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .harness import (
    BASELINE_REGISTRY,
    RESULTS_ROOT,
    compute_metrics,
    load_baseline,
    load_eval_pdds,
)

logger = logging.getLogger(__name__)


def predict_one(b, gid, reg, label, text, top_k, pdd_meta=None):
    try:
        try:
            preds = b.predict(text, top_k=top_k, pdd_meta=pdd_meta)
        except TypeError:
            # Baseline that doesn't accept pdd_meta yet
            preds = b.predict(text, top_k=top_k)
    except Exception as e:
        logger.warning("predict failed gid=%s: %s", gid, e)
        preds = []
    codes = [c for c, _ in preds]
    rank = next((i + 1 for i, c in enumerate(codes) if c == label), None)
    return {
        "gid": gid,
        "registry": reg,
        "gt_label": label,
        "predicted_top5": codes,
        "gt_rank": rank,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, choices=sorted(BASELINE_REGISTRY))
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default=None, choices=[None, "val", "test"])
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2, help="parallel PDD workers (default: matches daemon MAX_CONCURRENT)")
    ap.add_argument("--out-root", type=Path, default=RESULTS_ROOT,
                    help="output dir for the result JSON (default: results/); isolate (e.g. results/gpt55) to avoid clobbering another model's run")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("loading baseline %s ...", args.baseline)
    t0 = time.time()
    b = load_baseline(args.baseline)
    setup_s = time.time() - t0

    pdds = load_eval_pdds(args.n, args.seed, split=args.split)
    logger.info("loaded %d PDDs (split=%s) workers=%d", len(pdds), args.split, args.workers)

    per_pdd_by_gid: dict[str, dict] = {}
    t_query = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(predict_one, b, gid, reg, label, text, args.top_k, meta): gid
            for gid, reg, label, text, meta in pdds
        }
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            per_pdd_by_gid[r["gid"]] = r
            done += 1
            if done % 20 == 0 or done == len(pdds):
                logger.info("done %d/%d  elapsed=%.0fs", done, len(pdds), time.time() - t_query)
    query_s = time.time() - t_query

    # Reorder per_pdd to match original PDD order
    per_pdd = [per_pdd_by_gid[gid] for gid, _, _, _, _ in pdds if gid in per_pdd_by_gid]

    metrics = compute_metrics(per_pdd, top_k=args.top_k)
    out = {
        "baseline": args.baseline,
        "n_requested": args.n,
        "n_actual": len(per_pdd),
        "seed": args.seed,
        "split": args.split,
        "top_k": args.top_k,
        "workers": args.workers,
        "setup_s": round(setup_s, 2),
        "query_s": round(query_s, 2),
        "per_query_s": round(query_s / max(1, len(per_pdd)), 2),
        "metrics": metrics,
        "per_pdd": per_pdd,
    }

    out_path = args.out_root / f"{args.baseline}-{args.split or 'all'}-n{len(per_pdd)}-seed{args.seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    m = metrics
    print(f"\n=== {args.baseline} ({len(per_pdd)}/{args.n or 'all'}) workers={args.workers} ===")
    print(f"  top1={m.get('top1')}  top5={m.get('top5')}  MRR={m.get('mrr')}")
    if "by_registry" in m:
        for reg, mb in m["by_registry"].items():
            print(f"    {reg:3s}: n={mb['n']:3d}  top1={mb['top1']}  top5={mb['top5']}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
