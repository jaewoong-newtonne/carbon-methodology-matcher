"""Aggregate all per-baseline JSON results into a single markdown
table for the paper's results section. Also computes Wilson 95% CIs for top-1.

Usage:
    python -m carbonmm.eval.aggregate_results \\
        [--split test] [--out paper/sections/5-results-auto.md]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ICDM_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ICDM_ROOT / "results"

SYSTEM_DISPLAY_ORDER = [
    ("random", "Random"),
    ("freq-prior", "Frequency prior (per-registry)"),
    ("bm25-only", "BM25 (methodology-doc)"),
    ("dense-knn", "Dense kNN (mean-pooled)"),
    ("naive-rag", "Naive RAG"),
    ("graphrag-graphoff", "GraphRAG (γ=0 ablation)"),
    ("graphrag", "GraphRAG v1"),
    ("graphrag-fusion-only", "GraphRAG fusion-only (no LLM rerank)"),
    ("graphrag-v2", "**GraphRAG v2 (prompt fix)**"),
    ("graphrag-v3", "**GraphRAG v3 (RRF + freq prior)**"),
]


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return ((center - margin) / denom, (center + margin) / denom)


def _load_results(split: str) -> list[tuple[str, str, dict]]:
    """Return [(slug, display_name, result_dict), ...]."""
    out = []
    for slug, display in SYSTEM_DISPLAY_ORDER:
        candidates = sorted(RESULTS_ROOT.glob(f"{slug}-{split}-n*-seed*.json"))
        if not candidates:
            continue
        # Prefer the largest-n version
        candidates.sort(
            key=lambda p: -int(p.name.split("-n")[1].split("-seed")[0])
        )
        result = json.loads(candidates[0].read_text())
        out.append((slug, display, result))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument(
        "--out", type=Path, default=ICDM_ROOT / "paper" / "sections" / "5-results-auto.md"
    )
    args = ap.parse_args()

    rows = _load_results(args.split)
    if not rows:
        raise SystemExit(f"no results JSONs found in {RESULTS_ROOT} for split={args.split}")

    lines = []
    lines.append(f"# §5 Results table (auto-generated from {RESULTS_ROOT.relative_to(ICDM_ROOT)}/)\n")
    lines.append(f"Split: **{args.split}**\n")

    # Main table
    lines.append("## Main table — all systems")
    lines.append("| System | n | top-1 | top-5 | MRR | 95% CI (top-1) | s/query |")
    lines.append("|--------|---|------|------|-----|------|---------|")
    for slug, display, r in rows:
        n = r.get("n_actual", 0)
        m = r.get("metrics", {})
        top1 = m.get("top1", 0)
        top5 = m.get("top5", 0)
        mrr = m.get("mrr", 0)
        ci = wilson_ci(top1, n)
        s_query = r.get("per_query_s", 0)
        lines.append(
            f"| {display} | {n} | {top1:.3f} | {top5:.3f} | {mrr:.3f} | "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] | {s_query} |"
        )

    # Per-registry
    lines.append("\n## Per-registry breakdown")
    lines.append("| System | GS top-1 | GS n | VCS top-1 | VCS n | Δ (VCS - GS) |")
    lines.append("|--------|------|------|-------|------|------|")
    for slug, display, r in rows:
        by_reg = r.get("metrics", {}).get("by_registry", {})
        gs = by_reg.get("GS", {})
        vcs = by_reg.get("VCS", {})
        gs_top1 = gs.get("top1", 0.0)
        vcs_top1 = vcs.get("top1", 0.0)
        delta = vcs_top1 - gs_top1
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"| {display} | {gs_top1:.3f} | {gs.get('n', 0)} | "
            f"{vcs_top1:.3f} | {vcs.get('n', 0)} | {sign}{delta:.3f} |"
        )

    # GraphRAG ablation summary
    g_default = next((r for s, _, r in rows if s == "graphrag"), None)
    g_off = next((r for s, _, r in rows if s == "graphrag-graphoff"), None)
    if g_default and g_off:
        lines.append("\n## Graph filter ablation (γ=0.2 vs γ=0.0)")
        m1 = g_default["metrics"]
        m0 = g_off["metrics"]
        delta_top1 = m1["top1"] - m0["top1"]
        delta_top5 = m1["top5"] - m0["top5"]
        delta_mrr = m1["mrr"] - m0["mrr"]
        lines.append("| Variant | top-1 | top-5 | MRR |")
        lines.append("|---------|------|------|-----|")
        lines.append(f"| GraphRAG default (γ=0.2) | {m1['top1']:.3f} | {m1['top5']:.3f} | {m1['mrr']:.3f} |")
        lines.append(f"| GraphRAG graph-off (γ=0.0) | {m0['top1']:.3f} | {m0['top5']:.3f} | {m0['mrr']:.3f} |")
        sign1 = "+" if delta_top1 >= 0 else ""
        sign5 = "+" if delta_top5 >= 0 else ""
        signm = "+" if delta_mrr >= 0 else ""
        lines.append(
            f"| **Δ from graph signal** | **{sign1}{delta_top1:.3f}** | "
            f"**{sign5}{delta_top5:.3f}** | **{signm}{delta_mrr:.3f}** |"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n\nSaved: {args.out}")


if __name__ == "__main__":
    main()
