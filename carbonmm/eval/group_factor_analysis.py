"""Per-group differentiation-factor analysis (ICDM 2026, Part 1: structured/data-driven).

For each mitigation-activity group, discover WHICH factors actually differentiate its
member methodologies — because (as the data shows) different groups split on different
axes (G01 on scale, G03 on registry/region). Two views per group:

  (a) MEMBER-LEVEL heterogeneity — do the member methodologies differ on a factor?
      registries (catalog primary_standard), scale mix (corpus marker), version era,
      status, multi-registry adoption.
  (b) PDD-PREDICTIVE purity-gain — does knowing a PDD's observable factor concentrate
      the exact code? purity(F) = Σ_v max_code count(group,F=v,code)/N, gain vs the
      unconditional majority. Computed over registration records (descriptive, MI-style).

Output: paper/sections/group-factor-matrix-auto.md

Pure offline (stdlib). Run anywhere with the manifests present.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ICDM_ROOT = Path(__file__).resolve().parents[1]
M = ICDM_ROOT / "data" / "manifests"

REGION = {}
for codes, r in [
    (["IND", "PAK", "BGD", "NPL", "LKA", "AFG", "BTN"], "south-asia"),
    (["CHN", "MNG", "KOR", "JPN"], "east-asia"),
    (["VNM", "KHM", "LAO", "MMR", "IDN", "PHL", "THA", "MYS", "TLS"], "se-asia"),
    (["TUR", "EGY", "MAR", "JOR", "SAU", "ARE", "QAT", "IRQ", "IRN", "TUN", "DZA", "LBN", "OMN", "YEM", "KWT", "BHR"], "mena"),
    (["UGA", "KEN", "ETH", "TZA", "RWA", "MWI", "MOZ", "MDG", "ZMB", "ZWE", "BDI", "SOM", "SSD", "ERI"], "east-africa"),
    (["NGA", "GHA", "SEN", "CIV", "MLI", "BFA", "BEN", "TGO", "NER", "GIN", "SLE", "LBR", "GMB"], "west-africa"),
    (["ZAF", "BWA", "NAM", "LSO", "SWZ"], "southern-africa"),
    (["COD", "CMR", "CAF", "COG", "GAB", "TCD", "AGO"], "central-africa"),
    (["BRA", "MEX", "COL", "PER", "CHL", "ARG", "ECU", "BOL", "GTM", "HND", "NIC", "CRI", "PAN", "PRY", "URY", "DOM", "HTI"], "lac"),
    (["USA", "CAN"], "north-america"),
]:
    for c in codes:
        REGION[c] = r


def region(c):
    return REGION.get((c or "").upper(), "other")


def purity_gain(rows, idx):
    """rows: list of tuples (..., code); idx selects the factor column."""
    n = len(rows)
    if n == 0:
        return 0.0, 0.0
    uncond = max(Counter(r[-1] for r in rows).values()) / n
    byv = defaultdict(Counter)
    for r in rows:
        byv[r[idx]][r[-1]] += 1
    cond = sum(c.most_common(1)[0][1] for c in byv.values()) / n
    return cond, cond - uncond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ICDM_ROOT / "paper" / "sections" / "group-factor-matrix-auto.md")
    args = ap.parse_args()

    gmap = json.loads((M / "method-groups.json").read_text())
    c2g = gmap["code_to_group"]
    anchor = gmap["anchor_evidence"]
    labels = gmap["group_labels"]
    cat = {e["code"]: e for e in json.loads((M / "genvision-methodology-catalog.json").read_text())["entries"]}
    entries = json.loads((M / "genvision-pdd-eval-set.json").read_text())["entries"]
    splits = json.loads((M / "icdm2026-splits.json").read_text())
    gt_counts = Counter(p["gt"] for p in splits["test"]["pdds"])

    def group_of(code):
        return c2g.get(code, f"SINGLETON::{code}")

    # registration rows per group: (pdd_registry, country, region, code)
    group_rows = defaultdict(list)
    for e in entries:
        reg, c = e.get("registry"), e.get("country")
        for m in (e.get("methodologies") or []):
            group_rows[group_of(m)].append((reg, c, region(c), m))

    # only named groups that have GT mass, sorted by GT PDD count
    named = [g for g in labels]
    named.sort(key=lambda g: -sum(gt_counts[c] for c in gmap["groups"].get(g, [])))

    L = ["# Per-group differentiation-factor matrix (auto, Part 1: structured)\n",
         "Member-level heterogeneity + PDD-predictive purity-gain per factor. "
         "purity-gain = how much knowing the factor concentrates the exact code "
         "(over registration records). Higher = that factor differentiates the group.\n",
         "| Group | label | GT codes (PDDs) | registries (members) | scale mix | "
         "gain: registry | gain: country | gain: region | primary axis |",
         "|---|---|---|---|---|---|---|---|---|"]

    for g in named:
        members = gmap["groups"].get(g, [])
        gt_codes = [(c, gt_counts[c]) for c in members if gt_counts.get(c)]
        gt_codes.sort(key=lambda x: -x[1])
        n_gt_pdds = sum(n for _, n in gt_codes)
        if n_gt_pdds == 0:
            continue
        # member-level registries + scale mix (over GT-appearing members)
        regs = Counter(cat.get(c, {}).get("primary_standard") or "?" for c, _ in gt_codes)
        scales = Counter(anchor.get(c, {}).get("corpus_scale") or "unk" for c, _ in gt_codes)
        scale_mix = "+".join(sorted(k for k in scales if k != "unk")) or "unk"
        rows = group_rows.get(g, [])
        _, g_reg = purity_gain(rows, 0)
        _, g_cty = purity_gain(rows, 1)
        _, g_reg2 = purity_gain(rows, 2)
        # primary axis heuristic: scale if members span small+large, else max purity-gain factor
        gains = {"registry": g_reg, "country": g_cty, "region": g_reg2}
        if "small" in scales and "large" in scales:
            axis = "scale (members span small+large)"
        else:
            top = max(gains, key=gains.get)
            axis = f"{top} (+{gains[top]:.0%})" if gains[top] > 0.05 else "weak / near-deterministic"
        gt_str = ", ".join(f"{c}({n})" for c, n in gt_codes[:4]) + ("…" if len(gt_codes) > 4 else "")
        reg_str = ", ".join(f"{k}×{v}" for k, v in regs.most_common())
        L.append(f"| {g} | {labels[g][:26]} | {gt_str} ⟶ {n_gt_pdds} | {reg_str} | {scale_mix} | "
                 f"+{g_reg:.0%} | +{g_cty:.0%} | +{g_reg2:.0%} | **{axis}** |")

    out = "\n".join(L)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(out)
    print(out)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
