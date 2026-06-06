"""Capability sweep → architecture-gain decomposition table (paper §5 / C1).

For each LLM, compares the *naive-RAG* arm (document-only retrieve + rerank) to
the *GraphRAG v2.8.1* arm (Stage A+B candidate filter + scale-aware rerank),
using the SAME model for both arms (model-controlled ablation). The "architecture
gain" is the per-metric delta v281 − naive.

The point of the table is the *decomposition*: graph structure reliably lifts
recall (top5 / MRR) for capable models, but whether that recall becomes a correct
top-1 answer is reranker-vendor-dependent (Claude converts it; OpenAI largely
does not). Run after the Claude results are available to fill the within-Claude row.

Usage:
    python3 -m carbonmm.eval.capability_decomposition
    python3 -m carbonmm.eval.capability_decomposition --md out.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"

# model label → (naive arm file, v281 arm file). Same model drives both arms.
MODELS: list[tuple[str, str, str]] = [
    ("gpt-4o-mini", "capability/gpt-4o-mini/naive-rag-openai-test-n535-seed42.json",
                    "capability/gpt-4o-mini/graphrag-v281-openai-test-n535-seed42.json"),
    ("gpt-4o",      "capability/gpt-4o/naive-rag-openai-test-n535-seed42.json",
                    "capability/gpt-4o/graphrag-v281-openai-test-n535-seed42.json"),
    ("haiku-4.5",   "capability/haiku/naive-rag-sonnet-test-n535-seed42.json",
                    "capability/haiku/graphrag-v281-sonnet-test-n535-seed42.json"),
    ("sonnet-4.6",  "naive-rag-sonnet-test-n535-seed42.json",
                    "graphrag-v281-sonnet-test-n535-seed42.json"),
    ("gpt-5.5",     "gpt55/naive-rag-openai-test-n535-seed42.json",
                    "gpt55/graphrag-v281-openai-test-n535-seed42.json"),
    ("opus-4.8",      "capability/opus/naive-rag-sonnet-test-n535-seed42.json",
                    "capability/opus/graphrag-v281-sonnet-test-n535-seed42.json"),
    ("gpt-5.4-nano", "clean/gpt-5.4-nano/naive-rag-openai-test-n535-seed42.json",
                     "clean/gpt-5.4-nano/graphrag-v281-openai-test-n535-seed42.json"),
    ("gpt-5.4",      "clean/gpt-5.4/naive-rag-openai-test-n535-seed42.json",
                     "clean/gpt-5.4/graphrag-v281-openai-test-n535-seed42.json"),
]


def _metrics(rel: str) -> dict | None:
    p = RESULTS / rel
    if not p.exists():
        return None
    m = json.loads(p.read_text()).get("metrics", {})
    return {"top1": m.get("top1"), "top5": m.get("top5"), "mrr": m.get("mrr"), "n": m.get("n")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", type=Path, default=None, help="also write a markdown table here")
    args = ap.parse_args()

    rows = []
    for label, naive_rel, v281_rel in MODELS:
        nv, gr = _metrics(naive_rel), _metrics(v281_rel)
        if nv is None or gr is None or nv["top1"] is None or gr["top1"] is None:
            rows.append((label, None))
            continue
        d = {k: round(gr[k] - nv[k], 4) for k in ("top1", "top5", "mrr")}
        rows.append((label, {"naive": nv, "v281": gr, "delta": d}))

    # console table
    hdr = f"{'model':12s} {'naive_top1':>10s} {'v281_top1':>9s} | {'Δtop1':>7s} {'Δtop5':>7s} {'Δmrr':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for label, r in rows:
        if r is None:
            print(f"{label:12s} {'—  (missing/pending)':>38s}")
            continue
        n, v, d = r["naive"], r["v281"], r["delta"]
        sign = lambda x: f"{x:+.3f}"
        print(f"{label:12s} {n['top1']:>10.3f} {v['top1']:>9.3f} | "
              f"{sign(d['top1']):>7s} {sign(d['top5']):>7s} {sign(d['mrr']):>7s}")

    if args.md:
        lines = ["| model | naive top1 | v281 top1 | Δtop1 | Δtop5 | Δmrr |",
                 "|---|---:|---:|---:|---:|---:|"]
        for label, r in rows:
            if r is None:
                lines.append(f"| {label} | — | — | _pending_ | | |")
                continue
            n, v, d = r["naive"], r["v281"], r["delta"]
            lines.append(f"| {label} | {n['top1']:.3f} | {v['top1']:.3f} | "
                         f"{d['top1']:+.3f} | {d['top5']:+.3f} | {d['mrr']:+.3f} |")
        args.md.write_text("\n".join(lines) + "\n")
        print(f"\nwrote {args.md}")


if __name__ == "__main__":
    main()
