"""Compare a variant's test 535 result to naive-rag baseline.

Usage: python -m carbonmm.eval.compare_to_naive --variant graphrag-v25

Outputs:
  - Main metrics table (variant vs naive-rag)
  - Wilson 95% CI on top-1
  - Per-class accuracy (top GT codes)
  - Confusion pair analysis
  - "decisively higher" verdict
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0: return (0.0, 0.0)
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return ((center - margin) / denom, (center + margin) / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="e.g., graphrag-v25")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=535)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    v = json.load(open(RESULTS / f"{args.variant}-{args.split}-n{args.n}-seed{args.seed}.json"))
    nr = json.load(open(RESULTS / f"naive-rag-{args.split}-n{args.n}-seed{args.seed}.json"))

    print(f"\n{'='*60}")
    print(f" {args.variant} vs naive-rag on {args.split} n={args.n}")
    print(f"{'='*60}")

    mv = v["metrics"]
    mn = nr["metrics"]
    n = v["n_actual"]

    cv = wilson_ci(mv["top1"], n)
    cn = wilson_ci(mn["top1"], n)

    print(f"\n{'system':24s} {'top-1':>7s} {'top-5':>7s} {'MRR':>7s} {'95% CI top-1':>20s}")
    print(f"{'-'*65}")
    print(f"{'naive-rag':24s} {mn['top1']:>7.4f} {mn['top5']:>7.4f} {mn['mrr']:>7.4f}  [{cn[0]:.3f}, {cn[1]:.3f}]")
    print(f"{args.variant:24s} {mv['top1']:>7.4f} {mv['top5']:>7.4f} {mv['mrr']:>7.4f}  [{cv[0]:.3f}, {cv[1]:.3f}]")

    delta_top1 = mv["top1"] - mn["top1"]
    delta_top5 = mv["top5"] - mn["top5"]
    delta_mrr = mv["mrr"] - mn["mrr"]
    print(f"\nDelta (variant - naive-rag):")
    print(f"  Δtop-1 = {delta_top1:+.4f} ({delta_top1*100:+.1f}pp)")
    print(f"  Δtop-5 = {delta_top5:+.4f} ({delta_top5*100:+.1f}pp)")
    print(f"  ΔMRR   = {delta_mrr:+.4f}")

    # 95% CI separation test
    if cv[0] > cn[1]:
        ci_status = "✓ Wilson 95% CI separated — significantly above naive-rag"
    elif cv[1] < cn[0]:
        ci_status = "✗ Wilson 95% CI separated — significantly BELOW naive-rag"
    else:
        ci_status = "⚠ Wilson 95% CI overlap — not statistically distinguishable"
    print(f"\n{ci_status}")

    # "decisively higher" verdict
    print(f"\n=== 'decisively higher' verdict ===")
    if delta_top1 >= 0.05 and delta_top5 >= 0.05 and cv[0] > cn[1]:
        verdict = "✓ GOAL MET — top-1 and top-5 both >5pp above naive-rag, CI separated"
    elif delta_top1 >= 0.02 and delta_top5 >= 0.02:
        verdict = "⚠ Marginally above naive-rag — not decisively"
    elif delta_top1 < 0:
        verdict = "✗ Below naive-rag — goal not met"
    else:
        verdict = "≈ Approximately equal to naive-rag — goal not met"
    print(f"  {verdict}")

    # Per-class accuracy on top GT classes
    print(f"\n=== Per-class top-1 accuracy (top 10 GT) ===")
    gt_counter = Counter(p["gt_label"] for p in v["per_pdd"])
    print(f"{'GT label':14s} {'n':>4s} {'naive-rag':>10s} {'variant':>10s} {'Δ':>8s}")
    for gt, total in gt_counter.most_common(10):
        nr_correct = sum(1 for p in nr["per_pdd"] if p["gt_label"] == gt and p["gt_rank"] == 1)
        v_correct = sum(1 for p in v["per_pdd"] if p["gt_label"] == gt and p["gt_rank"] == 1)
        d = (v_correct - nr_correct) / max(1, total)
        print(f"{gt:14s} {total:>4d} {nr_correct/total:>10.2f} {v_correct/total:>10.2f} {d:>+8.2f}")

    # Per-registry
    print(f"\n=== Per-registry top-1 ===")
    for reg in ["GS", "VCS"]:
        nr_reg = mn.get("by_registry", {}).get(reg, {})
        v_reg = mv.get("by_registry", {}).get(reg, {})
        if nr_reg and v_reg:
            d = v_reg.get("top1", 0) - nr_reg.get("top1", 0)
            print(f"  {reg}: naive-rag={nr_reg.get('top1', 0):.4f} variant={v_reg.get('top1', 0):.4f} Δ={d:+.4f}")


if __name__ == "__main__":
    main()
