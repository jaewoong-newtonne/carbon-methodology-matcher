"""Step 1 verification: GT pass-rate under Stage A + B filter (no LLM, ~30s).

For each test 535 PDD:
  1. Build allowlist from registry + creditingPeriodStartDate (+ extracted features if cached)
  2. Check whether GT code ∈ allowlist
  3. Report pass-rate by registry + diagnostic counts

Acceptance: GT pass rate ≥ 99% (target 100%).
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from statistics import mean, median

from ..baselines.base import load_clauses_df
from ..graphrag.candidate_filter import build_allowlist, parse_date

RESULTS = Path(__file__).resolve().parents[1] / "results"
EVAL = Path(__file__).resolve().parents[1] / "data" / "eval-pdds"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=535)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Load test set (gid, registry, gt, creditingPeriodStartDate)
    nr_path = RESULTS / f"naive-rag-{args.split}-n{args.n}-seed{args.seed}.json"
    nr = json.load(open(nr_path))
    pdds = []
    for p in nr["per_pdd"]:
        gid = p["gid"]
        reg = p["registry"]
        gt = p["gt_label"]
        body_path = EVAL / reg / gid / f"{gid}.body.json"
        if not body_path.exists():
            continue
        body = json.loads(body_path.read_text())
        d = parse_date(body.get("creditingPeriodStartDate"))
        pdds.append((gid, reg, gt, d))
    print(f"Loaded {len(pdds)} test PDDs")

    # Corpus codes
    df = load_clauses_df()
    corpus_codes = df["code"].unique().tolist()
    print(f"Corpus has {len(corpus_codes)} unique codes")

    # Pass-rate
    gt_pass = 0
    gt_fail = 0
    gt_fail_codes = Counter()
    allowlist_sizes = []
    allowlist_sizes_by_reg = {"GS": [], "VCS": []}
    diag_counts = {"after_stage_a": [], "after_date": [], "after_scale": []}

    for gid, reg, gt, d in pdds:
        # No extracted features at this stage (Stage A + date only)
        allowed, diag = build_allowlist(
            pdd_registry=reg,
            pdd_credit_start=d,
            pdd_scale=None,
            pdd_scope=None,
            corpus_codes=corpus_codes,
        )
        allowlist_sizes.append(len(allowed))
        allowlist_sizes_by_reg.setdefault(reg, []).append(len(allowed))
        diag_counts["after_stage_a"].append(diag.n_after_stage_a)
        diag_counts["after_date"].append(diag.n_after_date)
        diag_counts["after_scale"].append(diag.n_after_ams_scale)
        if gt in allowed:
            gt_pass += 1
        else:
            gt_fail += 1
            gt_fail_codes[gt] += 1

    n = len(pdds)
    print(f"\n=== GT pass-rate under Stage A + B filter ===")
    print(f"  pass     : {gt_pass}/{n} = {gt_pass/n*100:.1f}%")
    print(f"  fail     : {gt_fail}/{n} = {gt_fail/n*100:.1f}%")
    if gt_fail_codes:
        print(f"  failing GT codes:")
        for c, count in gt_fail_codes.most_common(10):
            print(f"    {c:14s}  {count}")

    print(f"\n=== Allowlist size (lower = more aggressive filter) ===")
    print(f"  Stage A only      : mean={mean(diag_counts['after_stage_a']):.0f}  median={median(diag_counts['after_stage_a']):.0f}  min={min(diag_counts['after_stage_a'])}")
    print(f"  Stage A + date    : mean={mean(diag_counts['after_date']):.0f}  median={median(diag_counts['after_date']):.0f}  min={min(diag_counts['after_date'])}")
    print(f"  Stage A+date+scale: mean={mean(diag_counts['after_scale']):.0f}  median={median(diag_counts['after_scale']):.0f}  min={min(diag_counts['after_scale'])}")
    print(f"\n  Total candidate pool: {len(corpus_codes)}")
    print(f"  Filter retention rate: {mean(allowlist_sizes)/len(corpus_codes)*100:.1f}%")

    print(f"\n=== Per-registry ===")
    for reg, sizes in allowlist_sizes_by_reg.items():
        if sizes:
            print(f"  {reg}: n={len(sizes)}  allowlist mean={mean(sizes):.0f}  median={median(sizes):.0f}")


if __name__ == "__main__":
    main()
