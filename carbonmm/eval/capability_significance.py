"""Significance + 95% CIs for the capability-sweep architecture-gain (C1).

Per model: Wilson 95% CI on naive top1 and v281 top1 (marginal), and a
*paired* 95% CI on Δtop1 (v281 − naive) via per-PDD bootstrap (the arms are
paired on the same 535 PDDs, so the paired CI is the honest one), plus the
McNemar continuity-corrected test on top-1 correctness.

Deterministic: fixed bootstrap seed. Run after the clean sweep lands.

    python3 -m carbonmm.eval.capability_significance [--md out.md]
"""
from __future__ import annotations

import argparse
import json
import random
from math import sqrt, erfc
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"

MODELS = [
    ("gpt-4o-mini", "OpenAI", "capability/gpt-4o-mini/naive-rag-openai-test-n535-seed42.json",
                              "capability/gpt-4o-mini/graphrag-v281-openai-test-n535-seed42.json"),
    ("haiku-4.5",   "Claude", "capability/haiku/naive-rag-sonnet-test-n535-seed42.json",
                              "capability/haiku/graphrag-v281-sonnet-test-n535-seed42.json"),
    ("gpt-4o",      "OpenAI", "capability/gpt-4o/naive-rag-openai-test-n535-seed42.json",
                              "capability/gpt-4o/graphrag-v281-openai-test-n535-seed42.json"),
    ("sonnet-4.6",  "Claude", "naive-rag-sonnet-test-n535-seed42.json",
                              "graphrag-v281-sonnet-test-n535-seed42.json"),
    ("opus-4.8",      "Claude", "capability/opus/naive-rag-sonnet-test-n535-seed42.json",
                              "capability/opus/graphrag-v281-sonnet-test-n535-seed42.json"),
    ("gpt-5.5",     "OpenAI", "gpt55/naive-rag-openai-test-n535-seed42.json",
                              "gpt55/graphrag-v281-openai-test-n535-seed42.json"),
    ("gpt-5.4-nano", "OpenAI", "clean/gpt-5.4-nano/naive-rag-openai-test-n535-seed42.json",
                               "clean/gpt-5.4-nano/graphrag-v281-openai-test-n535-seed42.json"),
    ("gpt-5.4",      "OpenAI", "clean/gpt-5.4/naive-rag-openai-test-n535-seed42.json",
                               "clean/gpt-5.4/graphrag-v281-openai-test-n535-seed42.json"),
]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mcnemar(b: int, c: int) -> tuple[float, float]:
    if b + c == 0:
        return (0.0, 1.0)
    chi = (abs(b - c) - 1) ** 2 / (b + c)
    p = erfc(sqrt(chi / 2)) if chi > 0 else 1.0
    return (chi, p)


def correct_by_gid(path: str, metric: str = "top1") -> dict[str, int]:
    # top1: gt is the predicted #1. top5: gt appears anywhere in the predicted top-5.
    pp = json.loads((R / path).read_text())["per_pdd"]
    if metric == "top5":
        return {r["gid"]: int(r["gt_rank"] is not None) for r in pp}
    return {r["gid"]: int(r["gt_rank"] == 1) for r in pp}


def paired_boot_ci(nc: dict, vc: dict, B: int = 20000, seed: int = 42) -> tuple[float, float, float]:
    gids = sorted(set(nc) & set(vc))
    diffs = [vc[g] - nc[g] for g in gids]  # per-PDD paired difference in {-1,0,1}
    n = len(diffs)
    point = sum(diffs) / n
    rng = random.Random(seed)
    boots = []
    for _ in range(B):
        s = 0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    return point, boots[int(0.025 * B)], boots[int(0.975 * B)]


def run_metric(metric: str) -> list:
    out_rows = []
    label_m = metric.replace("top", "top-")
    print(f"\n### {label_m} ###")
    print(f"{'model':12s} {'vendor':7s} {'naive (95%CI)':>22s} {'v281 (95%CI)':>22s} "
          f"{f'Δ{metric} [paired 95%CI]':>26s} {'McNemar':>16s}")
    print("-" * 112)
    for label, vendor, nf, vf in MODELS:
        nc, vc = correct_by_gid(nf, metric), correct_by_gid(vf, metric)
        gids = sorted(set(nc) & set(vc))
        n = len(gids)
        kn, kv = sum(nc[g] for g in gids), sum(vc[g] for g in gids)
        nl, nh = wilson(kn, n)
        vl, vh = wilson(kv, n)
        b = sum(1 for g in gids if nc[g] and not vc[g])
        c = sum(1 for g in gids if not nc[g] and vc[g])
        chi, p = mcnemar(b, c)
        pt, dl, dh = paired_boot_ci(nc, vc)
        crosses0 = dl <= 0 <= dh
        print(f"{label:12s} {vendor:7s} "
              f"{kn/n:6.3f} [{nl:.3f},{nh:.3f}]  "
              f"{kv/n:6.3f} [{vl:.3f},{vh:.3f}]  "
              f"{pt:+.3f} [{dl:+.3f},{dh:+.3f}]  "
              f"χ²={chi:4.2f} p={p:.3f}{'  *' if p < 0.05 else ''}")
        out_rows.append((metric, label, vendor, kn / n, (nl, nh), kv / n, (vl, vh), pt, (dl, dh), p, crosses0))
    return out_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", type=Path, default=None)
    args = ap.parse_args()

    rows = run_metric("top1") + run_metric("top5")
    print("\nΔ paired-CI excludes 0  → that architecture gain is statistically real for that model.")
    if args.md:
        L = ["| metric | model | vendor | naive (95% CI) | v281 (95% CI) | Δ [paired 95% CI] | McNemar p | gain |",
             "|---|---|---|---|---|---|---|---|"]
        for metric, lab, ven, nt, (nl, nh), vt, (vl, vh), pt, (dl, dh), p, x0 in rows:
            verdict = "**real**" if not x0 else "n.s. (CI∋0)"
            L.append(f"| {metric} | {lab} | {ven} | {nt:.3f} [{nl:.3f}, {nh:.3f}] | {vt:.3f} [{vl:.3f}, {vh:.3f}] | "
                     f"{pt:+.3f} [{dl:+.3f}, {dh:+.3f}] | {p:.3f} | {verdict} |")
        args.md.write_text("\n".join(L) + "\n")
        print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
