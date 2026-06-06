"""Apply inclusion + dedupe policy on top of the document-type classification.

Inputs:
  data/manifests/pdd-classification.json — output of classify_pdd_type.py

Policy:
  KEEP    iff semantic_class ∈ {single, component} AND doc_type ≠ SD-VISta
  EXCLUDE umbrella, multi, SD-VISta, unparseable, unknown

PoA dedupe (component class only):
  Group by poa_id (filled by the classifier). For each group, keep one
  component (earliest creditingPeriodStartDate, ties broken by lexical
  globalId). Components without a poa_id are kept individually (each forms
  its own singleton group).

Output:
  data/manifests/pdd-eval-set-restricted.json with:
    {
      globalIds: [...sorted...],
      by_class: { single: [...], component: [...] },
      by_doc_type: { ... counts ... },
      n_input: 1144,
      n_kept: ...,
      n_dropped_excluded: ...,
      n_dropped_dedupe: ...,
      dedupe_dropped_groups: [{poa_id, kept_globalId, dropped_globalIds[]}],
    }

Also surfaces a Genvision-supplementation hint to the user if len(globalIds) < 400.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_body_meta(eval_pdd_root: Path, registry: str, global_id: str) -> dict:
    """Read body.json for creditingPeriodStartDate (used as the dedupe sort key)."""
    p = eval_pdd_root / registry / global_id / f"{global_id}.body.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def main() -> None:
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    default_class = repo_root / "data" / "manifests" / "pdd-classification.json"
    default_eval = repo_root / "data" / "eval-pdds"
    default_out = repo_root / "data" / "manifests" / "pdd-eval-set-restricted.json"

    ap = argparse.ArgumentParser()
    ap.add_argument("--classification", type=Path, default=default_class)
    ap.add_argument("--eval-pdd-root", type=Path, default=default_eval)
    ap.add_argument("--out", type=Path, default=default_out)
    ap.add_argument("--gap-threshold", type=int, default=400,
                    help="if final eval set < this, surface a supplementation gap")
    args = ap.parse_args()

    if not args.classification.exists():
        sys.exit(f"ERROR: {args.classification} does not exist. "
                 f"Run classify_pdd_type.py first.")

    classification = json.loads(args.classification.read_text())
    items = classification["classifications"]
    n_input = len(items)

    # Stage 1: inclusion filter
    candidates = [c for c in items
                  if c["semantic_class"] in ("single", "component")
                  and c["doc_type"] != "SD-VISta"]
    n_dropped_excluded = n_input - len(candidates)

    # Stage 2: PoA dedupe (component only; single = always kept)
    singles = [c for c in candidates if c["semantic_class"] == "single"]
    components = [c for c in candidates if c["semantic_class"] == "component"]

    # Group components by poa_id (None → singleton group keyed by gid for dedupe purposes)
    poa_groups: dict[str, list[dict]] = defaultdict(list)
    for c in components:
        key = c.get("poa_id") or f"NOPOA:{c['globalId']}"
        poa_groups[key].append(c)

    # Sort each group by (creditingPeriodStartDate or '9999', globalId) ascending
    enriched: dict[str, list[tuple[str, dict]]] = {}
    for poa, members in poa_groups.items():
        with_keys = []
        for c in members:
            meta = load_body_meta(args.eval_pdd_root, c["registry"], c["globalId"])
            sort_key = (meta.get("creditingPeriodStartDate") or "9999-99-99",
                        c["globalId"])
            with_keys.append((sort_key, c))
        with_keys.sort()
        enriched[poa] = with_keys

    dedupe_dropped_groups = []
    kept_components: list[dict] = []
    n_dropped_dedupe = 0
    for poa, ordered in enriched.items():
        kept = ordered[0][1]
        kept_components.append(kept)
        if len(ordered) > 1:
            dropped = [c["globalId"] for _, c in ordered[1:]]
            n_dropped_dedupe += len(dropped)
            dedupe_dropped_groups.append({
                "poa_id": None if poa.startswith("NOPOA:") else poa,
                "kept_globalId": kept["globalId"],
                "dropped_globalIds": dropped,
            })

    final = singles + kept_components
    final_ids = sorted(c["globalId"] for c in final)

    # Tallies for the manifest
    by_class = {
        "single":    sorted(c["globalId"] for c in singles),
        "component": sorted(c["globalId"] for c in kept_components),
    }
    by_doc_type = Counter(c["doc_type"] for c in final)
    by_registry = Counter(c["registry"] for c in final)

    manifest = {
        "version": "1.0",
        "n_input": n_input,
        "n_kept": len(final_ids),
        "n_dropped_excluded": n_dropped_excluded,
        "n_dropped_dedupe": n_dropped_dedupe,
        "by_class": {k: len(v) for k, v in by_class.items()},
        "by_class_globalIds": by_class,
        "by_doc_type": dict(by_doc_type),
        "by_registry": dict(by_registry),
        "globalIds": final_ids,
        "dedupe_dropped_groups": dedupe_dropped_groups,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"=== Eval-set restriction — input {n_input} PDDs ===")
    print(f"  EXCLUDED (umbrella/multi/SD-VISta/unparseable/unknown):  {n_dropped_excluded}")
    print(f"  DEDUPE-DROPPED (extra components per PoA):               {n_dropped_dedupe}")
    print(f"  FINAL KEPT:                                              {len(final_ids)}")
    print()
    print(f"  By semantic class:")
    print(f"    single:    {len(by_class['single'])}")
    print(f"    component: {len(by_class['component'])}")
    print()
    print(f"  By doc_type:")
    for t, n in sorted(by_doc_type.items(), key=lambda x: -x[1]):
        print(f"    {t:<22} {n:>5}")
    print()
    print(f"  By registry: {dict(by_registry)}")
    print()
    n_dedupe_groups = len(dedupe_dropped_groups)
    if n_dedupe_groups > 0:
        avg_dropped = n_dropped_dedupe / n_dedupe_groups
        print(f"  Dedupe summary: {n_dedupe_groups} PoA groups had >1 component "
              f"(avg {avg_dropped:.1f} dropped per affected group)")
        # Show top-5 worst offenders
        worst = sorted(dedupe_dropped_groups,
                       key=lambda g: -len(g["dropped_globalIds"]))[:5]
        print(f"  Top-5 PoA groups by dedupe-dropped count:")
        for g in worst:
            print(f"    {g['poa_id'] or '(no poa_id)':<14} "
                  f"kept={g['kept_globalId']}  dropped={len(g['dropped_globalIds'])}")
    print()
    print(f"  → {args.out}")
    if len(final_ids) < args.gap_threshold:
        gap = args.gap_threshold - len(final_ids)
        print()
        print(f"  ⚠ GAP: eval set ({len(final_ids)}) is below threshold "
              f"({args.gap_threshold}); short by {gap} PDDs.")
        print(f"    Per user policy, supplement via Genvision API by sampling more single-project")
        print(f"    PDDs from underrepresented methodology families.")


if __name__ == "__main__":
    main()
