"""Router v2 — date-aware ensemble.

Pipeline:
  1. For each PDD, get creditingPeriodStartDate.
  2. Filter both naive-rag top-5 and v2.5 top-5 by date validity (conservative).
  3. Apply router class-heuristic on FILTERED top-1.
  4. Fill ranks 2-5 with RRF over filtered union.

Saves ensemble-router-v2-test-n535-seed42.json.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, Counter
from datetime import datetime, date
from pathlib import Path
from statistics import mean

from ..graphrag.candidate_filter import (
    parse_date,
    is_invalid_at as _filter_is_invalid_at,
    load_catalog_indices,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
MANIFESTS = Path(__file__).resolve().parents[1] / "data" / "manifests"

V25_WIN_CODES = {
    "ACM0001", "AMS-II.G.", "ACM0010", "GS-EN-002",
    "AMS-III.G.", "AMS-III.H.",
}


def wilson_ci(p: float, n: int, z: float = 1.96):
    if n == 0: return (0.0, 0.0)
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return ((center - margin) / denom, (center + margin) / denom)


# Catalog indices reused from candidate_filter (single source of truth)
_ALL_STDS, CODE_VERSIONS, CODE_STATUS, _SCOPES = load_catalog_indices()


def is_invalid_at(code: str, target_date: date) -> bool:
    """Conservative date filter — thin wrapper around candidate_filter.is_invalid_at."""
    return _filter_is_invalid_at(code, target_date, CODE_VERSIONS, CODE_STATUS)


def filtered_top5(orig_top5, target_date):
    if not target_date:
        return list(orig_top5)
    return [c for c in orig_top5 if not is_invalid_at(c, target_date)]


def route_top1(nr_filt, v25_filt):
    """Use v2.5 filtered-top-1 if it's a V25-win code, else naive-rag filtered-top-1."""
    if v25_filt and v25_filt[0] in V25_WIN_CODES:
        return v25_filt[0]
    if nr_filt:
        return nr_filt[0]
    if v25_filt:
        return v25_filt[0]
    return None


def rrf_fill(nr_filt, v25_filt, exclude, k=60):
    scores = defaultdict(float)
    for r, c in enumerate(nr_filt, 1):
        if c not in exclude:
            scores[c] += 1.0 / (k + r)
    for r, c in enumerate(v25_filt, 1):
        if c not in exclude:
            scores[c] += 1.0 / (k + r)
    return [c for c, _ in sorted(scores.items(), key=lambda x: -x[1])[:4]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=535)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    v25 = json.load(open(RESULTS / f"graphrag-v25-{args.split}-n{args.n}-seed{args.seed}.json"))
    nr = json.load(open(RESULTS / f"naive-rag-{args.split}-n{args.n}-seed{args.seed}.json"))
    v25_by = {p["gid"]: p for p in v25["per_pdd"]}

    # Load PDD start dates
    EVAL = Path(__file__).resolve().parents[1] / 'data' / 'eval-pdds'
    test_gids = {p['gid'] for p in nr['per_pdd']}
    gid_to_start = {}
    for body_path in EVAL.rglob('*.body.json'):
        gid = body_path.stem.replace('.body', '')
        if gid not in test_gids:
            continue
        try:
            body = json.loads(body_path.read_text())
            d = parse_date(body.get('creditingPeriodStartDate'))
            if d:
                gid_to_start[gid] = d
        except Exception:
            pass

    preds = []
    for p in nr["per_pdd"]:
        gid = p["gid"]
        target_date = gid_to_start.get(gid)
        nr_top5 = p["predicted_top5"]
        v25_top5 = v25_by[gid]["predicted_top5"]
        # Date-filter both
        nr_filt = filtered_top5(nr_top5, target_date)
        v25_filt = filtered_top5(v25_top5, target_date)
        # Route top-1
        top1 = route_top1(nr_filt, v25_filt)
        # Fill ranks 2-5 by RRF over filtered union
        if top1:
            rest = rrf_fill(nr_filt, v25_filt, exclude={top1})
            top5 = [top1] + rest
        else:
            top5 = (nr_top5 or v25_top5)[:5]
        rank = next((i + 1 for i, c in enumerate(top5) if c == p["gt_label"]), None)
        preds.append({
            "gid": gid,
            "registry": p["registry"],
            "gt_label": p["gt_label"],
            "predicted_top5": top5,
            "gt_rank": rank,
        })

    n = len(preds)
    top1 = sum(1 for p in preds if p["gt_rank"] == 1) / n
    top5 = sum(1 for p in preds if isinstance(p["gt_rank"], int) and p["gt_rank"] <= 5) / n
    rr = [1.0 / p["gt_rank"] if isinstance(p["gt_rank"], int) and p["gt_rank"] <= 5 else 0 for p in preds]
    mrr = mean(rr)
    ci = wilson_ci(top1, n)

    print(f"\n=== Router v2 (date-aware ensemble) ===")
    print(f"n = {n}")
    print(f"top-1 = {top1:.4f}  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"top-5 = {top5:.4f}")
    print(f"MRR   = {mrr:.4f}")
    for reg in ["GS", "VCS"]:
        sub = [p for p in preds if p["registry"] == reg]
        sn = len(sub) or 1
        st1 = sum(1 for p in sub if p["gt_rank"] == 1) / sn
        st5 = sum(1 for p in sub if isinstance(p["gt_rank"], int) and p["gt_rank"] <= 5) / sn
        print(f"  {reg}: n={sn}  top-1={st1:.4f}  top-5={st5:.4f}")

    print(f"\nvs naive-rag (0.5290, 0.7533, 0.6100):")
    print(f"  Δtop-1 = {top1 - 0.529:+.4f} ({(top1 - 0.529)*100:+.1f}pp)")
    print(f"  Δtop-5 = {top5 - 0.7533:+.4f} ({(top5 - 0.7533)*100:+.1f}pp)")
    print(f"  ΔMRR   = {mrr - 0.6100:+.4f}")
    print(f"\nvs router v1 (0.5869, 0.7944):")
    print(f"  Δtop-1 = {top1 - 0.5869:+.4f}")
    print(f"  Δtop-5 = {top5 - 0.7944:+.4f}")

    out_path = RESULTS / f"ensemble-router-v2-{args.split}-n{n}-seed{args.seed}.json"
    out_path.write_text(json.dumps({
        "baseline": "ensemble-router-v2-date-aware",
        "n_actual": n,
        "split": args.split,
        "metrics": {"top1": top1, "top5": top5, "mrr": mrr,
                    "by_registry": {reg: {
                        "n": sum(1 for p in preds if p["registry"] == reg),
                        "top1": sum(1 for p in preds if p["registry"] == reg and p["gt_rank"] == 1) / max(1, sum(1 for p in preds if p["registry"] == reg)),
                        "top5": sum(1 for p in preds if p["registry"] == reg and isinstance(p["gt_rank"], int) and p["gt_rank"] <= 5) / max(1, sum(1 for p in preds if p["registry"] == reg)),
                    } for reg in ["GS", "VCS"]}},
        "per_pdd": preds,
    }, indent=2))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
