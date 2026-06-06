"""Residual-ambiguity ceiling (ICDM 2026, Q2 headroom).

Before building richer within-group features, quantify whether they are even
needed: after conditioning the candidate pool on (group) -> (group + known
registry) -> (group + registry + extracted scale), how many candidates remain,
and what fraction of PDDs is already uniquely determined?

Levels (nested, monotone):
  A  group only                         (all retrievable codes in gt's group)
  B  group + known PDD registry         (registry_keep — a GS PDD may still use a CDM code)
  C  group + registry + extracted scale (scale_excluded — AMS-* are small-only)
     [C requires data/manifests/pdd-extracted-features-test.json; skipped if absent]

Reports, overall and per group:
  - residual candidate-count distribution
  - %(residual == 1)  := resolved by filtering alone (no fine features needed)
  - %(residual >= 2)  := genuine within-group ambiguity (Q2 target) + mean residual there

Self-contained (stdlib only): inlines the pure helpers from
graphrag/candidate_filter.py (code_family / registry_keep / scale_excluded) and
reads the catalog directly, so it runs under any Python without the retrieval stack.

Usage:
    python eval/residual_ambiguity.py [--features data/manifests/pdd-extracted-features-test.json]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ICDM_ROOT = Path(__file__).resolve().parents[1]
CATALOG = ICDM_ROOT / "data" / "manifests" / "genvision-methodology-catalog.json"
SPLITS = ICDM_ROOT / "data" / "manifests" / "icdm2026-splits.json"
GROUP_MAP = ICDM_ROOT / "data" / "manifests" / "method-groups.json"
CORPUS_ROOT = ICDM_ROOT / "data" / "corpus"

_CDM_RE = re.compile(r"^(ACM|AM\d|AMS-|AR-|ARNM)")
_VCS_RE = re.compile(r"^(VM\d|VMR|VMD|GCCM)")
_GS_RE = re.compile(r"^(GS-|\d)")
_AMS_RE = re.compile(r"^AMS-")


def code_family(code: str) -> str:
    if _CDM_RE.match(code):
        return "CDM"
    if _VCS_RE.match(code):
        return "VCS"
    if _GS_RE.match(code):
        return "GS"
    return "OTHER"


def registry_keep(pdd_registry: str, code: str, all_stds: dict[str, set]) -> bool:
    if not pdd_registry:
        return True
    fam = code_family(code)
    stds = all_stds.get(code, set())
    if pdd_registry == "GS":
        return fam in {"GS", "CDM"} or "GS" in stds
    if pdd_registry == "VCS":
        return fam in {"VCS", "CDM"} or "VCS" in stds
    return True


def scale_excluded(code: str, scale: str) -> bool:
    return scale == "large" and bool(_AMS_RE.match(code))


def corpus_codes() -> set[str]:
    return {p.parent.name for p in CORPUS_ROOT.glob("*/*/*.json")}


def all_standards_index() -> dict[str, set]:
    raw = json.loads(CATALOG.read_text())
    out: dict[str, set] = {}
    for e in raw.get("entries", []):
        c = e.get("code")
        if c:
            out[c] = set(e.get("all_standards") or [])
    return out


def _summary(counts: list[int]) -> dict:
    n = len(counts)
    if n == 0:
        return {"n": 0}
    one = sum(1 for c in counts if c == 1) / n
    multi = [c for c in counts if c >= 2]
    return {
        "n": n,
        "resolved_pct": round(one, 4),               # residual == 1
        "ambiguous_pct": round(len(multi) / n, 4),    # residual >= 2
        "mean_residual": round(statistics.mean(counts), 2),
        "median_residual": statistics.median(counts),
        "mean_residual_when_ambiguous": round(statistics.mean(multi), 2) if multi else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=ICDM_ROOT / "data" / "manifests" / "pdd-extracted-features-test.json",
                    help="optional {gid: {scale: ...}} cache enabling Level C")
    ap.add_argument("--map", type=Path, default=GROUP_MAP)
    ap.add_argument("--out", type=Path, default=ICDM_ROOT / "paper" / "sections" / "5-residual-ambiguity-auto.md")
    args = ap.parse_args()

    gmap = json.loads(args.map.read_text())
    c2g = gmap["code_to_group"]
    labels = gmap.get("group_labels", {})

    def group_of(code: str) -> str:
        return c2g.get(code, f"SINGLETON::{code}")

    universe = corpus_codes()
    all_stds = all_standards_index()
    # precompute group -> universe members
    group_members: dict[str, list[str]] = defaultdict(list)
    for c in universe:
        group_members[group_of(c)].append(c)

    feats = {}
    if args.features.exists():
        feats = json.loads(args.features.read_text())

    pdds = json.loads(SPLITS.read_text())["test"]["pdds"]
    levelA, levelB, levelC = [], [], []
    perA, perB = defaultdict(list), defaultdict(list)
    have_scale = 0

    for p in pdds:
        gid, reg, gt = p["gid"], p["registry"], p["gt"]
        g = group_of(gt)
        members = group_members.get(g, [])
        a = members  # Level A: whole group
        b = [c for c in a if registry_keep(reg, c, all_stds)]
        levelA.append(len(a))
        levelB.append(len(b))
        perA[g].append(len(a))
        perB[g].append(len(b))
        scale = (feats.get(gid) or {}).get("scale")
        if scale in {"small", "large"}:
            have_scale += 1
            c = [code for code in b if not scale_excluded(code, scale)]
            levelC.append(len(c))

    L = ["# §5 Residual-ambiguity ceiling (auto-generated)\n",
         f"Test PDDs: {len(pdds)}. Universe: {len(universe)} retrievable corpus codes. "
         f"Group map: {gmap['n_named_groups']} groups.\n",
         "How much within-group ambiguity remains after deterministic filtering? "
         "`resolved` = the group collapses to a single candidate (no fine features needed); "
         "`ambiguous` = >= 2 candidates remain (the genuine target of within-group disambiguation).\n",
         "| Level | conditioning | n | resolved (==1) | ambiguous (>=2) | mean residual | mean residual (when ambiguous) |",
         "|---|---|---|---|---|---|---|"]
    sA, sB = _summary(levelA), _summary(levelB)
    L.append(f"| A | group only | {sA['n']} | {sA['resolved_pct']:.3f} | {sA['ambiguous_pct']:.3f} | "
             f"{sA['mean_residual']} | {sA['mean_residual_when_ambiguous']} |")
    L.append(f"| B | + known registry | {sB['n']} | {sB['resolved_pct']:.3f} | {sB['ambiguous_pct']:.3f} | "
             f"{sB['mean_residual']} | {sB['mean_residual_when_ambiguous']} |")
    if levelC:
        sC = _summary(levelC)
        L.append(f"| C | + extracted scale | {sC['n']} | {sC['resolved_pct']:.3f} | {sC['ambiguous_pct']:.3f} | "
                 f"{sC['mean_residual']} | {sC['mean_residual_when_ambiguous']} |")
    else:
        L.append(f"\n_Level C skipped: extracted-features cache not found at {args.features.name} "
                 f"(run extract pass to enable scale conditioning)._")

    # nesting invariant
    assert all(b <= a for a, b in zip(levelA, levelB)), "Level B residual > Level A (nesting bug)"

    # per-group (Level B) — where does within-group ambiguity concentrate?
    L.append("\n## Per-group residual after group+registry (Level B), by GT frequency")
    L.append("| Group | label | PDDs | resolved (==1) | ambiguous (>=2) | mean residual |")
    L.append("|---|---|---|---|---|---|")
    for g in sorted(perB, key=lambda g: -len(perB[g])):
        if g.startswith("SINGLETON::"):
            continue
        s = _summary(perB[g])
        L.append(f"| {g} | {labels.get(g, g)} | {s['n']} | {s['resolved_pct']:.3f} | "
                 f"{s['ambiguous_pct']:.3f} | {s['mean_residual']} |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L))
    print("\n".join(L))
    if feats:
        print(f"\n[Level C used extracted scale for {have_scale}/{len(pdds)} PDDs]")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
