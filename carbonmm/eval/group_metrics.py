"""Group-level metrics + exact-vs-group gap decomposition (ICDM 2026).

Mirrors eval/harness.compute_metrics, but scores at the mitigation-activity
GROUP level using the frozen, prediction-blind map from build_method_groups.py.

For each PDD we recompute (self-contained, from predicted_top5 + gt_label):
  - EXACT  top1/top5/mrr  (== harness numbers; sanity cross-check)
  - GROUP  top1/top5/mrr  (gt_rank at group granularity)
  - per-registry breakdown (GS/VCS), as in harness

Gap decomposition — for PDDs where exact-top1 is WRONG but group-top1 is RIGHT,
the rank-1 prediction vs gt is bucketed into:
  - registry_variant : same group, different registry family (e.g. ACM0002 <-> GCCM001)
  - scale_tier       : same group, same family, different corpus scale (ACM0002 <-> AMS-I.D.)
  - fine_applicability: same group, same family, same scale (the genuine residual)
These three buckets sum EXACTLY to (group_top1 - exact_top1) count (closure invariant).

Usage (single file):
    python -m carbonmm.eval.group_metrics --result results/graphrag-v281-sonnet-test-n535-seed42.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ICDM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP = ICDM_ROOT / "data" / "manifests" / "method-groups.json"

# mirror graphrag/candidate_filter.code_family for codes absent from the map
_CDM_RE = re.compile(r"^(ACM|AM\d|AMS-|AR-|ARNM)")
_VCS_RE = re.compile(r"^(VM\d|VMR|VMD|GCCM)")
_GS_RE = re.compile(r"^(GS-|\d)")


def _family(code: str) -> str:
    if _CDM_RE.match(code):
        return "CDM"
    if _VCS_RE.match(code):
        return "VCS"
    if _GS_RE.match(code):
        return "GS"
    return "OTHER"


def load_group_map(path: Path = DEFAULT_MAP) -> dict:
    return json.loads(Path(path).read_text())


def make_group_of(gmap: dict):
    c2g = gmap["code_to_group"]

    def group_of(code: str) -> str:
        # Singleton fallback: any code not explicitly mapped is its own group,
        # so an unmapped distractor can never spuriously match a GT group.
        return c2g.get(code, f"SINGLETON::{code}")

    return group_of


def _first_rank(codes: list[str], pred_ok) -> int | None:
    for i, c in enumerate(codes):
        if pred_ok(c):
            return i + 1
    return None


def _metric_block(ranks: list[int | None]) -> dict:
    n = len(ranks)
    if n == 0:
        return {"n": 0, "top1": 0.0, "top5": 0.0, "mrr": 0.0}
    top1 = sum(1 for r in ranks if r == 1) / n
    top5 = sum(1 for r in ranks if r is not None and r <= 5) / n
    mrr = sum(1.0 / r for r in ranks if r is not None) / n
    return {"n": n, "top1": round(top1, 4), "top5": round(top5, 4), "mrr": round(mrr, 4)}


def score_result(result: dict, gmap: dict) -> dict:
    group_of = make_group_of(gmap)
    anchor = gmap.get("anchor_evidence", {})

    def fam(code: str) -> str:
        return anchor.get(code, {}).get("family") or _family(code)

    def cscale(code: str):
        return anchor.get(code, {}).get("corpus_scale")

    per = result.get("per_pdd", [])
    exact_ranks, group_ranks = [], []
    by_reg = {"GS": {"exact": [], "group": []}, "VCS": {"exact": [], "group": []}}
    decomp = Counter()
    decomp_examples = {"registry_variant": [], "scale_tier": [], "fine_applicability": []}

    for r in per:
        gt = r["gt_label"]
        codes = r.get("predicted_top5", [])
        reg = r.get("registry")
        e_rank = _first_rank(codes, lambda c: c == gt)
        g_rank = _first_rank(codes, lambda c: group_of(c) == group_of(gt))
        exact_ranks.append(e_rank)
        group_ranks.append(g_rank)
        if reg in by_reg:
            by_reg[reg]["exact"].append(e_rank)
            by_reg[reg]["group"].append(g_rank)
        # decomposition: exact-top1 wrong, group-top1 right
        if e_rank != 1 and g_rank == 1 and codes:
            p0 = codes[0]
            if fam(p0) != fam(gt):
                bucket = "registry_variant"
            elif cscale(p0) and cscale(gt) and cscale(p0) != cscale(gt):
                bucket = "scale_tier"
            else:
                bucket = "fine_applicability"
            decomp[bucket] += 1
            if len(decomp_examples[bucket]) < 8:
                decomp_examples[bucket].append({"gid": r.get("gid"), "pred0": p0, "gt": gt})

    exact = _metric_block(exact_ranks)
    group = _metric_block(group_ranks)

    # invariants
    assert group["top1"] >= exact["top1"] - 1e-9, "group_top1 < exact_top1 (map bug)"
    assert group["top5"] >= exact["top5"] - 1e-9, "group_top5 < exact_top5 (map bug)"
    n = len(per)
    gap_count = round(group["top1"] * n) - round(exact["top1"] * n)
    assert sum(decomp.values()) == gap_count, (
        f"decomposition {dict(decomp)} sums to {sum(decomp.values())} != gap {gap_count}")

    return {
        "baseline": result.get("baseline"),
        "n": n,
        "exact": exact,
        "group": group,
        "by_registry": {
            reg: {"exact": _metric_block(d["exact"]), "group": _metric_block(d["group"])}
            for reg, d in by_reg.items()
        },
        "gap_decomposition": {
            "gap_count": gap_count,
            "registry_variant": decomp["registry_variant"],
            "scale_tier": decomp["scale_tier"],
            "fine_applicability": decomp["fine_applicability"],
            "examples": decomp_examples,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path, required=True)
    ap.add_argument("--map", type=Path, default=DEFAULT_MAP)
    args = ap.parse_args()
    gmap = load_group_map(args.map)
    result = json.loads(args.result.read_text())
    s = score_result(result, gmap)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
