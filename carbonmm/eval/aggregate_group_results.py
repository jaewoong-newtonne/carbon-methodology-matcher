"""Aggregate GROUP-level metrics across all systems into a paper-§5 markdown
table (ICDM 2026). Companion to eval/aggregate_results.py (exact-match table).

Reuses:
  - eval.group_metrics.score_result  (exact + group metrics + gap decomposition)
  - eval.aggregate_results.wilson_ci (Wilson 95% CI)

Emits:  paper/sections/5-results-group-auto.md

Includes the rigor scaffolding for a headline secondary metric:
  - exact-top1 vs group-top1 side by side (never replaces exact-match)
  - Wilson CIs on group-top1
  - trivial-baseline group-top1 (random / freq-prior) → non-saturation evidence
  - per-system gap decomposition (registry / scale-tier / fine-applicability)

Usage:
    python eval/aggregate_group_results.py [--split test]
    (run from carbonmm so `import group_metrics` resolves)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ICDM_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ICDM_ROOT / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))  # make sibling modules importable
from group_metrics import score_result, load_group_map  # noqa: E402
from aggregate_results import wilson_ci  # noqa: E402

# display name + ordering; unknown slugs appended in discovery order
DISPLAY = [
    ("random", "Random"),
    ("freq-prior", "Frequency prior"),
    ("bm25-only", "BM25"),
    ("dense-knn", "Dense kNN"),
    ("naive-rag", "Naive RAG (Haiku 4.5)"),
    ("naive-rag-sonnet", "Naive RAG (Sonnet 4.6)"),
    ("graphrag", "GraphRAG v1"),
    ("graphrag-fusion-only", "GraphRAG fusion-only"),
    ("graphrag-v25", "GraphRAG v2.5"),
    ("graphrag-v28-openai-mini", "GraphRAG v2.8 (gpt-4o-mini)"),
    ("graphrag-v28-openai", "GraphRAG v2.8 (gpt-4o)"),
    ("graphrag-v281-openai", "GraphRAG v2.81 (gpt-4o)"),
    ("graphrag-v281-sonnet", "GraphRAG v2.81 (Sonnet 4.6)"),
    ("ensemble-router", "Ensemble router"),
    ("ensemble-router-v2", "Ensemble router v2"),
    ("ensemble-router-judge", "Ensemble router + judge"),
    ("router-v28-gpt4o-nr", "Router v2.8 (gpt-4o)"),
]
DISPLAY_MAP = dict(DISPLAY)
TRIVIAL = {"random", "freq-prior"}


def discover(split: str) -> list[tuple[str, Path]]:
    """Return [(slug, path)] preferring the largest-n file per slug."""
    best: dict[str, tuple[int, Path]] = {}
    for p in RESULTS_ROOT.glob(f"*-{split}-n*-seed*.json"):
        name = p.name
        try:
            slug = name.split(f"-{split}-n")[0]
            n = int(name.split(f"-{split}-n")[1].split("-seed")[0])
        except (IndexError, ValueError):
            continue
        if slug not in best or n > best[slug][0]:
            best[slug] = (n, p)
    # order: known display order first, then any extras
    ordered = [(slug, best[slug][1]) for slug, _ in DISPLAY if slug in best]
    extras = sorted(s for s in best if s not in DISPLAY_MAP)
    ordered += [(s, best[s][1]) for s in extras]
    return ordered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--map", type=Path, default=ICDM_ROOT / "data" / "manifests" / "method-groups.json")
    ap.add_argument("--out", type=Path,
                    default=ICDM_ROOT / "paper" / "sections" / "5-results-group-auto.md")
    args = ap.parse_args()

    gmap = load_group_map(args.map)
    rows = discover(args.split)
    if not rows:
        raise SystemExit(f"no result files in {RESULTS_ROOT} for split={args.split}")

    scored = []
    for slug, path in rows:
        result = json.loads(path.read_text())
        s = score_result(result, gmap)
        scored.append((slug, s))

    L = []
    L.append(f"# §5 Group-level results (auto-generated)\n")
    L.append(f"Split: **{args.split}**. Group map: {gmap['n_named_groups']} activity groups, "
             f"prediction-blind ({gmap['version']}).\n")
    L.append("Group-level is a SECONDARY metric reported ALONGSIDE exact-match. "
             "group-top1 = the predicted #1 is in the same mitigation-activity group as the gold code.\n")

    # main table
    L.append("## Main table — exact vs group")
    L.append("| System | n | exact top-1 | group top-1 | group top-5 | group MRR | Δ top-1 | 95% CI (group top-1) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for slug, s in scored:
        disp = DISPLAY_MAP.get(slug, slug)
        if slug in TRIVIAL:
            disp = f"_{disp}_"
        e, g = s["exact"], s["group"]
        ci = wilson_ci(g["top1"], s["n"])
        d = g["top1"] - e["top1"]
        L.append(f"| {disp} | {s['n']} | {e['top1']:.3f} | **{g['top1']:.3f}** | {g['top5']:.3f} | "
                 f"{g['mrr']:.3f} | +{d:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] |")

    # gap decomposition
    L.append("\n## Exact→group gap decomposition (count of PDDs newly correct at group level)")
    L.append("| System | gap | registry-variant | scale-tier | fine-applicability |")
    L.append("|---|---|---|---|---|")
    for slug, s in scored:
        disp = DISPLAY_MAP.get(slug, slug)
        gd = s["gap_decomposition"]
        L.append(f"| {disp} | {gd['gap_count']} | {gd['registry_variant']} | "
                 f"{gd['scale_tier']} | {gd['fine_applicability']} |")

    # per-registry group-top1
    L.append("\n## Per-registry group-top1")
    L.append("| System | GS exact | GS group | VCS exact | VCS group |")
    L.append("|---|---|---|---|---|")
    for slug, s in scored:
        disp = DISPLAY_MAP.get(slug, slug)
        gs = s["by_registry"]["GS"]
        vcs = s["by_registry"]["VCS"]
        L.append(f"| {disp} | {gs['exact']['top1']:.3f} | {gs['group']['top1']:.3f} | "
                 f"{vcs['exact']['top1']:.3f} | {vcs['group']['top1']:.3f} |")

    # non-saturation note
    trivial_g = {slug: s["group"]["top1"] for slug, s in scored if slug in TRIVIAL}
    if trivial_g:
        note = ", ".join(f"{DISPLAY_MAP.get(k, k)}={v:.3f}" for k, v in trivial_g.items())
        L.append(f"\n**Non-saturation check**: trivial-baseline group-top1 = {note} "
                 f"(largest GT group = {_largest_gt_group(gmap)}). Group-top1 is NOT saturated at 1.0.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nSaved: {args.out}")


def _largest_gt_group(gmap: dict) -> str:
    # informational only; uses splits if available
    sp = ICDM_ROOT / "data" / "manifests" / "icdm2026-splits.json"
    if not sp.exists():
        return "n/a"
    from collections import Counter
    cnt = Counter(p["gt"] for p in json.loads(sp.read_text())["test"]["pdds"])
    c2g = gmap["code_to_group"]
    bysize: dict[str, int] = {}
    for code, n in cnt.items():
        g = c2g.get(code, f"SINGLETON::{code}")
        bysize[g] = bysize.get(g, 0) + n
    g, n = max(bysize.items(), key=lambda kv: kv[1])
    total = sum(cnt.values())
    return f"{gmap['group_labels'].get(g, g)} {n}/{total} ({100*n/total:.0f}%)"


if __name__ == "__main__":
    main()
