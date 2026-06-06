"""Quick comparison: v2 (or v3) val n=50 vs naive-rag val n=50 vs v1.

Reports:
  - Aggregate top-1, top-5, MRR for each baseline on the same val subset
  - Per-baseline per-registry breakdown
  - ACM0002 / AMS-I.D. specific accuracy (the dominant GT classes)
  - Confusion pairs (predicted top-1 → GT) for the failing baselines
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

RESULTS = Path(__file__).resolve().parents[1] / "results"


def load_pdds(slug: str, split: str = "val", n: int | None = 50, seed: int = 42):
    """Load result JSON. Try with --n N first, else any val file."""
    candidates = []
    if n:
        candidates.append(RESULTS / f"{slug}-{split}-n{n}-seed{seed}.json")
    candidates.append(RESULTS / f"{slug}-{split}-n100-seed{seed}.json")
    for p in candidates:
        if p.exists():
            d = json.loads(p.read_text())
            return d, p
    return None, None


def metrics_subset(per_pdd, gids: set | None = None):
    """Compute top-1/top-5/MRR on given gid subset (or all)."""
    pdds = per_pdd if gids is None else [p for p in per_pdd if p["gid"] in gids]
    n = len(pdds)
    if n == 0:
        return {"n": 0, "top1": 0, "top5": 0, "mrr": 0}
    top1 = sum(1 for p in pdds if p["gt_rank"] == 1)
    top5 = sum(1 for p in pdds if isinstance(p["gt_rank"], int) and p["gt_rank"] <= 5)
    rr = [(1.0 / p["gt_rank"]) if isinstance(p["gt_rank"], int) and p["gt_rank"] <= 5 else 0 for p in pdds]
    return {
        "n": n,
        "top1": top1 / n,
        "top5": top5 / n,
        "mrr": mean(rr) if rr else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="e.g., graphrag-v2, graphrag-v3, graphrag-fusion-only")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    v2, p2 = load_pdds(args.variant, args.split, args.n, args.seed)
    nr, pnr = load_pdds("naive-rag", args.split, args.n, args.seed)
    gr, pgr = load_pdds("graphrag", args.split, args.n, args.seed)

    if v2 is None:
        raise SystemExit(f"{args.variant} {args.split} result missing")

    print(f"\n{'='*70}\n {args.variant} on {args.split}")
    print(f"{'='*70}")
    print(f"file: {p2}")
    m = v2["metrics"]
    print(f"top1={m['top1']:.4f}  top5={m['top5']:.4f}  MRR={m['mrr']:.4f}  (n={v2['n_actual']})")
    if "by_registry" in m:
        for reg, mb in m["by_registry"].items():
            print(f"  {reg:4s}: n={mb['n']:3d}  top1={mb['top1']:.4f}  top5={mb['top5']:.4f}")

    # Restrict comparisons to common subset
    v2_gids = {p["gid"] for p in v2["per_pdd"]}
    common = v2_gids & ({p["gid"] for p in nr["per_pdd"]} if nr else v2_gids)
    if gr:
        common = common & {p["gid"] for p in gr["per_pdd"]}
    print(f"\n--- Common subset (n={len(common)}) ---")
    print(f"{'baseline':28s} {'top1':>6s} {'top5':>6s} {'mrr':>6s}")
    for name, d in [(args.variant, v2), ("naive-rag", nr), ("graphrag (v1)", gr)]:
        if d is None:
            continue
        m = metrics_subset(d["per_pdd"], common)
        marker = " ✓" if m['top1'] > 0.529 else ""
        print(f"{name:28s} {m['top1']:>6.3f} {m['top5']:>6.3f} {m['mrr']:>6.3f}{marker}")

    # ACM0002 specific
    acm_gids = {p["gid"] for p in v2["per_pdd"] if p["gt_label"] == "ACM0002"}
    if acm_gids:
        print(f"\n--- ACM0002 cases (n={len(acm_gids)}) ---")
        for name, d in [(args.variant, v2), ("naive-rag", nr), ("graphrag (v1)", gr)]:
            if d is None: continue
            m = metrics_subset(d["per_pdd"], acm_gids)
            print(f"  {name:28s} top1={m['top1']:.3f}")

    # Confusion pairs in v2
    confusion = Counter()
    for p in v2["per_pdd"]:
        if p["gt_rank"] == 1: continue
        pred1 = p["predicted_top5"][0] if p["predicted_top5"] else "?"
        confusion[(pred1, p["gt_label"])] += 1
    print(f"\n--- {args.variant} confusion pairs (top-10) ---")
    for (pred, gt), c in confusion.most_common(10):
        print(f"  {pred:14s} -> {gt:14s}  {c}")


if __name__ == "__main__":
    main()
