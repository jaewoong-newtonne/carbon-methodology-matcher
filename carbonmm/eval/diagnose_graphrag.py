"""GraphRAG diagnostic — answer 4 questions about WHY graphrag may or may
not beat naive-rag.

A. **Filter ceiling**: How often does the GT appear in graphrag's *fused*
   top-10 (the candidate set the LLM actually sees)? This is an upper
   bound on graphrag's top-K accuracy — the LLM cannot rank a code that
   was never given to it.

B. **Hard-exclude false negative**: How often does the AMS-on-scale=large
   rule wrongly remove an AMS-* GT? This isolates deterministic loss
   from the rule.

C. **Match-count saturation**: What is the distribution of graph_match
   counts (0-4) across all fused candidates? If most candidates score
   3 or 4, graph_match is effectively tied and γ has no tie-breaking
   effect — the "graph signal" is noise.

D. **LLM implicit-knowledge advantage**: How often does naive-rag's
   top-1 = GT, while graphrag's filter has *removed* the GT from the
   top-10 candidate set entirely? This counts cases where the LLM's
   internal knowledge of methodology applicability outperforms our
   rule-based filter.

Usage (after the eval chain finishes):
    python -m carbonmm.eval.diagnose_graphrag \\
        --split test --seed 42

Auto-detects results JSONs at results/{naive-rag,graphrag}-test-n*-seed42.json.
Outputs results/graphrag-diagnostics.json + console summary.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

ICDM_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ICDM_ROOT / "results"


def _auto_results(split: str, seed: int, baseline: str) -> Path | None:
    candidates = sorted(
        RESULTS_ROOT.glob(f"{baseline}-{split}-n*-seed{seed}.json"),
        key=lambda p: -int(p.name.split("-n")[1].split("-seed")[0]),
    )
    return candidates[0] if candidates else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--naive-rag-results", type=Path, default=None)
    ap.add_argument("--graphrag-results", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=RESULTS_ROOT / "graphrag-diagnostics.json")
    ap.add_argument("--top-k-fused", type=int, default=10)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from .harness import load_eval_pdds
    from ..graphrag.retrieve import HybridRetriever
    from ..graphrag.extract_features import extract_features
    from ..graphrag.graph_filter import filter_candidates
    from ..graphrag.score_fusion import FusionConfig, fuse

    naive_path = args.naive_rag_results or _auto_results(args.split, args.seed, "naive-rag")
    graphrag_path = args.graphrag_results or _auto_results(args.split, args.seed, "graphrag")
    if not naive_path or not graphrag_path:
        raise SystemExit(
            f"Required results not found at {RESULTS_ROOT}/ for split={args.split} seed={args.seed}. "
            "Run the eval chain first."
        )

    logger.info("naive-rag results: %s", naive_path)
    logger.info("graphrag results: %s", graphrag_path)
    naive_results = json.loads(naive_path.read_text())
    graphrag_results = json.loads(graphrag_path.read_text())

    naive_by_gid = {r["gid"]: r for r in naive_results["per_pdd"]}
    graphrag_by_gid = {r["gid"]: r for r in graphrag_results["per_pdd"]}

    logger.info("loading HybridRetriever (one-time setup) ...")
    R = HybridRetriever.load_default()
    fusion_cfg = FusionConfig()  # default = bm25_heavy (α=0.5/β=0.3/γ=0.2)

    pdds = load_eval_pdds(None, args.seed, split=args.split)
    logger.info("loaded %d PDDs from split=%s", len(pdds), args.split)

    diagnostics: list[dict] = []
    match_count_dist: Counter = Counter()
    n_ams_gt = 0
    n_ams_excluded = 0
    n_gt_in_fused = 0
    n_gt_in_raw_candidates = 0

    for i, (gid, reg, gt_label, text, _meta) in enumerate(pdds, 1):
        if i % 50 == 0:
            logger.info("processing %d / %d ...", i, len(pdds))
        try:
            raw = R.query_with_raw(text)
        except Exception as e:
            logger.warning("retrieve fail %s: %s", gid, e)
            continue
        try:
            feats = extract_features(text)
        except Exception as e:
            logger.warning("extract fail %s: %s — using empty features", gid, e)
            from ..graphrag.extract_features import PDDFeatures
            feats = PDDFeatures()

        # candidate codes from raw retrieval (pre-filter)
        candidate_codes = list(
            {raw["idx_to_code"][i_] for i_, _ in raw["bm25_top"]}
            | {raw["idx_to_code"][i_] for i_, _ in raw["dense_top"]}
        )

        filter_results = filter_candidates(candidate_codes, feats)
        fused = fuse(
            raw["bm25_top"],
            raw["dense_top"],
            raw["idx_to_code"],
            filter_results,
            config=fusion_cfg,
            top_k=args.top_k_fused,
        )
        fused_codes = [c for c, _, _ in fused]

        # ─── Metric collection ─────────────────────────────────────
        gt_in_fused = gt_label in fused_codes
        n_gt_in_fused += int(gt_in_fused)

        gt_in_candidates = gt_label in candidate_codes
        n_gt_in_raw_candidates += int(gt_in_candidates)

        is_ams_gt = gt_label.startswith("AMS-")
        gt_excluded = False
        if is_ams_gt:
            n_ams_gt += 1
            res = filter_results.get(gt_label)
            if res and res.excluded:
                gt_excluded = True
                n_ams_excluded += 1

        for code, _, comp in fused:
            match_count_dist[int(comp.get("graph_match_raw", 0))] += 1

        naive_rec = naive_by_gid.get(gid, {})
        naive_top1 = (naive_rec.get("predicted_top5") or [None])[0]
        naive_correct = (naive_top1 == gt_label)

        graphrag_rec = graphrag_by_gid.get(gid, {})
        graphrag_top1 = (graphrag_rec.get("predicted_top5") or [None])[0]
        graphrag_correct = (graphrag_top1 == gt_label)

        diagnostics.append(
            {
                "gid": gid,
                "registry": reg,
                "gt": gt_label,
                "is_ams_gt": is_ams_gt,
                "extracted_features": {
                    "scale": feats.scale,
                    "scope": feats.sectoral_scope,
                    "tech": feats.technology,
                    "ghg": feats.ghg_species,
                    "country": feats.country_iso,
                },
                "gt_in_raw_candidates": gt_in_candidates,
                "gt_excluded_by_filter": gt_excluded,
                "gt_in_fused_top10": gt_in_fused,
                "fused_top10_codes": fused_codes,
                "fused_top10_match_counts": [int(c.get("graph_match_raw", 0)) for _, _, c in fused],
                "naive_top1": naive_top1,
                "naive_correct": naive_correct,
                "graphrag_top1": graphrag_top1,
                "graphrag_correct": graphrag_correct,
                "case_naive_wins_filter_loses": naive_correct and not gt_in_fused,
            }
        )

    n = len(diagnostics)
    n_naive_correct = sum(1 for d in diagnostics if d["naive_correct"])
    n_graphrag_correct = sum(1 for d in diagnostics if d["graphrag_correct"])
    n_filter_loss = sum(1 for d in diagnostics if d["case_naive_wins_filter_loses"])

    summary = {
        "n": n,
        "fusion_config_used": {"alpha": fusion_cfg.alpha, "beta": fusion_cfg.beta, "gamma": fusion_cfg.gamma},
        "top_k_fused": args.top_k_fused,
        "diagnostic_A_filter_ceiling": {
            "n_gt_in_fused_top10": n_gt_in_fused,
            "frac": round(n_gt_in_fused / max(1, n), 4),
            "interpretation": (
                "Upper bound on graphrag top-K accuracy. If << naive-rag top-1, "
                "filter is dropping too many correct candidates."
            ),
        },
        "diagnostic_B_hard_exclude_fn": {
            "n_ams_gt": n_ams_gt,
            "n_ams_excluded_by_rule": n_ams_excluded,
            "frac_ams_excluded": round(n_ams_excluded / max(1, n_ams_gt), 4),
            "interpretation": (
                "Of AMS-* ground truths, the fraction wrongly removed by the "
                "AMS-on-scale=large hard-exclude rule (i.e., extracted scale=large "
                "for a project whose true methodology is AMS-*)."
            ),
        },
        "diagnostic_C_match_saturation": {
            "distribution": dict(sorted(match_count_dist.items())),
            "frac_4_of_4": round(
                match_count_dist.get(4, 0) / max(1, sum(match_count_dist.values())), 4
            ),
            "frac_3_or_4": round(
                (match_count_dist.get(3, 0) + match_count_dist.get(4, 0))
                / max(1, sum(match_count_dist.values())),
                4,
            ),
            "interpretation": (
                "If most candidates score 3-4/4, graph_match is saturated and "
                "γ·graph_match in fusion is effectively noise — γ has no tie-break."
            ),
        },
        "diagnostic_D_llm_implicit_knowledge": {
            "n_naive_correct_total": n_naive_correct,
            "n_naive_correct_but_gt_filtered": n_filter_loss,
            "frac_of_naive_correct": round(n_filter_loss / max(1, n_naive_correct), 4),
            "interpretation": (
                "PDDs where naive-rag's LLM correctly identified the GT at rank-1, "
                "but graphrag's filter removed the GT from the top-10 candidate set "
                "entirely. Direct evidence that the LLM's implicit knowledge of "
                "methodology applicability outperforms our rule-based heuristic."
            ),
        },
        "overall": {
            "gt_in_raw_candidates_frac": round(n_gt_in_raw_candidates / max(1, n), 4),
            "naive_top1": round(n_naive_correct / max(1, n), 4),
            "graphrag_top1": round(n_graphrag_correct / max(1, n), 4),
            "delta_graphrag_minus_naive": round(
                (n_graphrag_correct - n_naive_correct) / max(1, n), 4
            ),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "per_pdd": diagnostics}, indent=2))

    # Console summary
    print("\n" + "=" * 70)
    print("GRAPHRAG DIAGNOSTICS — paper §6 narrative inputs")
    print("=" * 70)
    print(f"n PDDs: {n}  ·  split: {args.split}  ·  fusion: α={fusion_cfg.alpha} β={fusion_cfg.beta} γ={fusion_cfg.gamma}")
    print(f"\nOverall top-1:")
    print(f"  naive-rag:   {summary['overall']['naive_top1']:.3f}  ({n_naive_correct}/{n})")
    print(f"  graphrag:    {summary['overall']['graphrag_top1']:.3f}  ({n_graphrag_correct}/{n})")
    print(f"  Δ:           {summary['overall']['delta_graphrag_minus_naive']:+.3f}")
    print(f"  GT in raw candidates (pre-filter): {summary['overall']['gt_in_raw_candidates_frac']:.3f}")

    A = summary["diagnostic_A_filter_ceiling"]
    print(f"\nA. FILTER CEILING (graphrag top-K upper bound)")
    print(f"   GT in fused top-{args.top_k_fused}: {A['frac']:.3f}  ({A['n_gt_in_fused_top10']}/{n})")
    if A["frac"] < summary["overall"]["naive_top1"]:
        print(f"   ⚠ ceiling < naive-rag top-1 → filter is dropping GTs the LLM would otherwise find")

    B = summary["diagnostic_B_hard_exclude_fn"]
    print(f"\nB. HARD-EXCLUDE FALSE NEGATIVE")
    print(f"   AMS-* GTs: {B['n_ams_gt']}  ·  excluded by rule: {B['n_ams_excluded_by_rule']}  ({B['frac_ams_excluded']:.3f})")

    C = summary["diagnostic_C_match_saturation"]
    print(f"\nC. MATCH-COUNT SATURATION (across fused top-{args.top_k_fused} of all PDDs)")
    for k, v in C["distribution"].items():
        bar = "█" * int(v / max(1, max(C["distribution"].values())) * 40)
        print(f"   match={k}/4: {v:5d} {bar}")
    print(f"   frac saturated (3 or 4): {C['frac_3_or_4']:.3f}  ·  frac perfect (4/4): {C['frac_4_of_4']:.3f}")
    if C["frac_3_or_4"] > 0.8:
        print(f"   ⚠ heavy saturation → γ·graph_match adds little discriminative signal")

    D = summary["diagnostic_D_llm_implicit_knowledge"]
    print(f"\nD. LLM IMPLICIT KNOWLEDGE vs FILTER")
    print(f"   naive correct AND GT filtered out: {D['n_naive_correct_but_gt_filtered']}  "
          f"({D['frac_of_naive_correct']:.3f} of naive-correct)")
    if D["frac_of_naive_correct"] > 0.05:
        print(f"   ⚠ non-trivial filter-loss rate → LLM's implicit knowledge > our rule-based heuristic for some cases")

    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
