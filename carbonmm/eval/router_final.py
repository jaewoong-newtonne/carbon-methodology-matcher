"""Final ensemble router — combines naive-rag + GraphRAG-v2.5 with per-class
heuristic routing. Achieves top-1 = 0.587, top-5 = 0.793 on test n=535,
substantially above naive-rag (0.529, 0.753).

Routing rule (derived from val + visual confusion analysis):
  - If v2.5 predicts a code from V25_WIN_CODES (the rare-decent codes where
    v2.5's popularity hint outperforms naive-rag substantially) → trust v2.5
  - Else → trust naive-rag

V25_WIN_CODES discovered from per-class accuracy comparison:
  ACM0001 (+28pp), AMS-II.G. (+32pp), ACM0010 (+18pp), GS-EN-002 (+3pp),
  AMS-III.G. (modest), AMS-III.H. (modest)
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

RESULTS = Path(__file__).resolve().parents[1] / "results"

V25_WIN_CODES = {
    "ACM0001", "AMS-II.G.", "ACM0010", "GS-EN-002",
    "AMS-III.G.", "AMS-III.H.",
}


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0: return (0.0, 0.0)
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return ((center - margin) / denom, (center + margin) / denom)


def route_then_rrf(nr_top5: list[str], v25_top5: list[str], k: int = 60) -> list[str]:
    """Use v2.5 top-1 if it predicts a V25_WIN_CODE, else naive-rag top-1.
    Fill remaining ranks via RRF over union."""
    primary = nr_top5
    if v25_top5 and v25_top5[0] in V25_WIN_CODES:
        primary = v25_top5
    if not primary:
        primary = v25_top5 or nr_top5
    if not primary:
        return []
    top_1 = primary[0]
    # RRF over union (exclude top-1)
    scores = defaultdict(float)
    for r, c in enumerate(nr_top5, 1):
        if c != top_1: scores[c] += 1.0 / (k + r)
    for r, c in enumerate(v25_top5, 1):
        if c != top_1: scores[c] += 1.0 / (k + r)
    rest = [c for c, _ in sorted(scores.items(), key=lambda x: -x[1])[:4]]
    return [top_1] + rest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=535)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    v25 = json.load(open(RESULTS / f"graphrag-v25-{args.split}-n{args.n}-seed{args.seed}.json"))
    nr = json.load(open(RESULTS / f"naive-rag-{args.split}-n{args.n}-seed{args.seed}.json"))
    v25_by = {p["gid"]: p for p in v25["per_pdd"]}

    preds = []
    for p in nr["per_pdd"]:
        gid = p["gid"]
        v25_top5 = v25_by[gid]["predicted_top5"]
        top5 = route_then_rrf(p["predicted_top5"], v25_top5)
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

    print(f"\n=== Final ensemble router (v2.5 + naive-rag, class-heuristic) ===")
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

    # Compare to naive-rag
    print(f"\nvs naive-rag (0.5290, 0.7533, 0.6100):")
    print(f"  Δtop-1 = {top1 - 0.529:+.4f} ({(top1 - 0.529)*100:+.1f}pp)")
    print(f"  Δtop-5 = {top5 - 0.7533:+.4f} ({(top5 - 0.7533)*100:+.1f}pp)")
    print(f"  ΔMRR   = {mrr - 0.6100:+.4f}")

    # Save result file
    out = {
        "baseline": "ensemble-router-v25-nr",
        "n_actual": n,
        "split": args.split,
        "seed": args.seed,
        "top_k": 5,
        "metrics": {
            "top1": top1,
            "top5": top5,
            "mrr": mrr,
            "by_registry": {
                reg: {
                    "n": sum(1 for p in preds if p["registry"] == reg),
                    "top1": sum(1 for p in preds if p["registry"] == reg and p["gt_rank"] == 1) / max(1, sum(1 for p in preds if p["registry"] == reg)),
                    "top5": sum(1 for p in preds if p["registry"] == reg and isinstance(p["gt_rank"], int) and p["gt_rank"] <= 5) / max(1, sum(1 for p in preds if p["registry"] == reg)),
                }
                for reg in ["GS", "VCS"]
            },
        },
        "per_pdd": preds,
    }
    out_path = RESULTS / f"ensemble-router-{args.split}-n{n}-seed{args.seed}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
