"""Build the frozen, prediction-blind methodology GROUP map (ICDM 2026).

A *group* is a mitigation-activity equivalence class: it collapses scale-variants
(ACM0002 large <-> AMS-I.D. small) and registry-variants (ACM0002 <-> GCCM001 <->
GS-EN-*) of the SAME abatement activity into one class, but keeps genuinely
different activities (grid-renewable vs safe-water-supply) apart.

Anchors are PREDICTION-BLIND and external:
  - official methodology TITLE (Genvision catalog `entries[].title`)
  - corpus PDF-header markers ("Large-scale"/"Small-scale", "Sectoral scope(s): NN")
    -- used only for anchor_evidence / scale-tier decomposition, NOT as the key.
The catalog `sectoral_scopes`/`scale` fields are NOT used: they are unreliable
(e.g. AMS-I.D. tagged "Large Scale"; GS-EN-002 tagged scope-2 "industrial/ANAB";
AR-ACM0003 tagged scope-14 "waste") -- a data-quality finding documented in the paper.

Output: data/manifests/method-groups.json  (frozen artifact, checked in)
        + an appendix-ready markdown table.

Usage:
    python -m carbonmm.eval.build_method_groups \\
        [--generated-at 2026-05-29] [--out data/manifests/method-groups.json]

This script reads ONLY the catalog + corpus. It never reads results/ -- assert
that in review; it is the anti-p-hacking guarantee.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ICDM_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ICDM_ROOT / "data" / "manifests" / "genvision-methodology-catalog.json"
CORPUS_ROOT = ICDM_ROOT / "data" / "corpus"
SPLITS_PATH = ICDM_ROOT / "data" / "manifests" / "icdm2026-splits.json"
DEFAULT_OUT = ICDM_ROOT / "data" / "manifests" / "method-groups.json"
DEFAULT_MD = ICDM_ROOT / "paper" / "sections" / "appendix-method-groups-auto.md"

# ─── Family prefix rules (mirror graphrag/candidate_filter.code_family) ──────
CDM_RE = re.compile(r"^(ACM|AM\d|AMS-|AR-|ARNM)")
VCS_RE = re.compile(r"^(VM\d|VMR|VMD|GCCM)")
GS_RE = re.compile(r"^(GS-|\d)")


def code_family(code: str) -> str:
    if CDM_RE.match(code):
        return "CDM"
    if VCS_RE.match(code):
        return "VCS"
    if GS_RE.match(code):
        return "GS"
    return "OTHER"


# ─── Group ruleset (ORDERED, first match wins; prediction-blind on TITLE) ────
# Each rule: (rule_id, group_id, label, kind, pattern)
#   kind="title" -> regex on lowercased official title
#   kind="code"  -> regex on the code (official AMS small-scale TYPE taxonomy;
#                   external, not derived from our predictions)
GROUP_LABELS = {
    "G01": "Grid-connected renewable electricity generation",
    "G02": "Landfill gas capture & flaring",
    "G03": "Thermal energy: cooking/heating, biomass fuel-switch & efficiency",
    "G04": "Afforestation & reforestation",
    "G05": "Forest conservation / REDD+ / IFM / avoided conversion",
    "G06": "Manure & livestock waste management",
    "G07": "Wastewater methane treatment",
    "G08": "Solid-waste treatment / composting / recycling / waste-gas-to-power",
    "G09": "Demand-side & appliance energy efficiency / lighting",
    "G10": "Transport / electric vehicles / modal shift",
    "G11": "Coal-mine / coal-bed / fugitive fuel methane",
    "G12": "Agriculture / rice / grassland land management",
    "G13": "Cement / concrete / pavement materials",
    "G14": "Fossil-fuel switch / fossil grid electricity",
    "G15": "Safe drinking water supply",
    "G16": "Industrial gas destruction (ODS / HFC / SF6)",
}

GROUP_RULES = [
    # specific waste / methane / fugitive domains BEFORE generic energy
    ("R01", "G02", "title", r"landfill"),
    ("R02", "G11", "title", r"coal\s*(mine|bed)|coal mines|leak detection|fugitive (methane|emission)"),
    ("R03", "G07", "title", r"waste\s*water"),
    ("R04", "G06", "title", r"manure|livestock"),
    ("R05", "G08", "title", r"waste treatment|composting|solid waste|waste gas|recycling of materials|recovery and recycling"),
    # land use
    ("R06", "G04", "title", r"afforest|reforest"),
    ("R07", "G05", "title", r"deforestation|\bredd|forest management|avoided ecosystem|conversion from logged"),
    ("R08", "G12", "title", r"agricultural land|\brice\b|grassland|grazing"),
    # water
    ("R09", "G15", "title", r"drinking water|water supply|safe water|water purification"),
    # transport
    ("R10", "G10", "title", r"vehicle|transport|modal shift|\bcharging|shore-side|off-shore electricity supply"),
    # industrial gas destruction
    ("R11", "G16", "title", r"ozone-depleting|hydrofluorocarbon|\bsf6\b|cover gas|\bods\b"),
    # cement / concrete / materials (word-boundaried: 'cement' must not match 'displacement')
    ("R12", "G13", "title", r"\b(cement|clinker|concrete|asphalt|pavement|quicklime)\b"),
    # lighting efficiency (LED/CFL) — before the fossil-fuel rule, which would
    # otherwise steal "fossil fuel based lighting" (AMS-III.AR.)
    ("R12L", "G09", "title", r"\blighting\b|\bled\b|\bcfl\b"),
    # grid-connected / renewable (incl. biomass) ELECTRICITY -- before thermal
    ("R13", "G01", "title",
     r"(grid[\- ]?connected|electrification|electricity generation|electricity and heat|power[\- ]only|renewable electricity)"
     r".*(renewable|wind|solar|hydro|geothermal|tidal|wave|biomass)"
     r"|(renewable|wind|solar|hydro|geothermal|tidal|wave|biomass).*(electric|grid|power generation|power[\- ]only)"),
    # official AMS small-scale TYPE taxonomy (external) -- electricity/mechanical → G01
    ("R14", "G01", "code", r"^AMS-I\.[ABFL]\.?$"),
    # thermal energy: cooking / heating / fuel-switch / thermal efficiency
    ("R15", "G03", "title", r"cooking|heating|thermal|cookstove|stove|space heating|district cooling"),
    ("R16", "G03", "code", r"^AMS-I\.[CE]\.?$"),
    # fossil-fuel switch / fossil grid electricity
    ("R17", "G14", "title", r"fuel switch|fossil fuel|natural gas|grid connected electricity"),
    # demand-side EE / lighting (generic energy efficiency, last)
    ("R18", "G09", "title", r"energy efficiency|lighting|\bled\b|\bcfl\b|demand-side|campus clean energy"),
]

_COMPILED = [(rid, gid, kind, re.compile(pat, re.I)) for rid, gid, kind, pat in GROUP_RULES]

# header markers in the corpus PDF text
_SCOPE_RE = re.compile(r"[Ss]ectoral scope\(s\):\s*0?(\d{1,2})")
_SMALL_RE = re.compile(r"small[\- ]scale", re.I)
_LARGE_RE = re.compile(r"large[\- ]scale", re.I)


def assign_group(code: str, title: str) -> tuple[str | None, str | None]:
    """Return (group_id, rule_id) or (None, None) if no rule matches."""
    t = (title or "").lower()
    for rid, gid, kind, rx in _COMPILED:
        target = code if kind == "code" else t
        if rx.search(target):
            return gid, rid
    return None, None


def corpus_header_anchors(code: str) -> tuple[str | None, str | None]:
    """Return (corpus_scale, cdm_scope) from the corpus PDF-header text, or (None, None).

    corpus_scale in {'small','large'} ; cdm_scope is the leading scope digits string.
    """
    matches = list(CORPUS_ROOT.glob(f"*/{code}/{code}.json"))
    if not matches:
        return None, None
    try:
        d = json.loads(matches[0].read_text())
    except Exception:
        return None, None
    header = " ".join(
        str(d.get(k) or "")
        for k in ("title", "typical_projects", "baseline_scenario", "mitigation_action")
    )
    appl = d.get("applicability") or []
    if appl:
        header += " " + str(appl[0])
    scale = None
    if _SMALL_RE.search(header):
        scale = "small"
    elif _LARGE_RE.search(header):
        scale = "large"
    m = _SCOPE_RE.search(header)
    scope = m.group(1) if m else None
    return scale, scope


def load_catalog() -> dict[str, dict]:
    raw = json.loads(CATALOG_PATH.read_text())
    return {e["code"]: e for e in raw.get("entries", []) if e.get("code")}


def load_gt_counts() -> Counter:
    if not SPLITS_PATH.exists():
        return Counter()
    s = json.loads(SPLITS_PATH.read_text())
    return Counter(p["gt"] for p in s.get("test", {}).get("pdds", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated-at", default="unset", help="timestamp string to stamp the artifact")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--md", type=Path, default=DEFAULT_MD)
    ap.add_argument("--read-corpus", action="store_true",
                    help="read corpus PDF-header anchors (scale/scope); slower, needs full corpus")
    args = ap.parse_args()

    catalog = load_catalog()
    gt_counts = load_gt_counts()

    code_to_group: dict[str, str] = {}
    code_to_rule: dict[str, str] = {}
    anchor_evidence: dict[str, dict] = {}
    groups: dict[str, list[str]] = defaultdict(list)

    for code, e in catalog.items():
        title = e.get("title") or ""
        gid, rid = assign_group(code, title)
        fam = code_family(code)
        if gid is None:
            # singleton fallback: own group, never accidentally matches a GT group
            gid = f"SINGLETON::{code}"
            rid = "R00-singleton"
        code_to_group[code] = gid
        code_to_rule[code] = rid
        groups[gid].append(code)
        ev = {"title": title, "family": fam}
        if args.read_corpus:
            cscale, cscope = corpus_header_anchors(code)
            ev["corpus_scale"] = cscale
            ev["cdm_scope"] = cscope
        anchor_evidence[code] = ev

    named_groups = {g: sorted(cs) for g, cs in groups.items() if not g.startswith("SINGLETON::")}

    out = {
        "version": "1.0",
        "generated_at": args.generated_at,
        "prediction_blind": True,
        "doc": ("Mitigation-activity equivalence classes. Keys derived ONLY from official "
                "catalog titles + corpus PDF-header markers; never from predictions. "
                "Collapses scale/registry variants of the same activity. See "
                "eval/build_method_groups.py GROUP_RULES (frozen) + appendix table."),
        "n_codes": len(code_to_group),
        "n_named_groups": len(named_groups),
        "group_labels": {g: GROUP_LABELS.get(g, g) for g in sorted(named_groups)},
        "rules": [{"rule_id": rid, "group_id": gid, "kind": kind, "pattern": pat}
                  for rid, gid, kind, pat in GROUP_RULES],
        "groups": {g: named_groups[g] for g in sorted(named_groups)},
        "code_to_group": code_to_group,
        "code_to_rule": code_to_rule,
        "anchor_evidence": anchor_evidence,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    # ── Diagnostics (printed; not part of the artifact) ──
    print(f"catalog codes: {len(catalog)}  named groups: {len(named_groups)}  "
          f"singletons: {sum(1 for g in groups if g.startswith('SINGLETON::'))}")
    print("\n── GT-code coverage by group (count = test PDDs) ──")
    gt_by_group: dict[str, list[tuple[str, int]]] = defaultdict(list)
    unmatched_gt = []
    for code, n in gt_counts.most_common():
        g = code_to_group.get(code, f"SINGLETON::{code}")
        if g.startswith("SINGLETON::"):
            unmatched_gt.append((code, n))
        gt_by_group[g].append((code, n))
    for g in sorted(gt_by_group, key=lambda g: -sum(n for _, n in gt_by_group[g])):
        if g.startswith("SINGLETON::"):
            continue
        members = gt_by_group[g]
        tot = sum(n for _, n in members)
        label = GROUP_LABELS.get(g, g)
        print(f"  {g} [{tot:3d}] {label}")
        for code, n in members:
            print(f"        {n:4d}  {code}")
    if unmatched_gt:
        print(f"\n── UNMATCHED GT codes → singletons ({len(unmatched_gt)}) ──")
        for code, n in unmatched_gt:
            print(f"  {n:4d}  {code}  | {catalog.get(code, {}).get('title', '?')[:60]}")
    # granularity guard: largest GT group
    gt_group_sizes = {g: sum(n for _, n in m) for g, m in gt_by_group.items()
                      if not g.startswith("SINGLETON::")}
    if gt_group_sizes:
        big = max(gt_group_sizes.items(), key=lambda kv: kv[1])
        total_gt = sum(gt_counts.values())
        print(f"\nlargest GT group: {big[0]} = {big[1]}/{total_gt} "
              f"({100*big[1]/total_gt:.1f}%)")

    # ── Appendix table ──
    md = ["# Appendix: Methodology group map (auto-generated)\n",
          f"Generated: {args.generated_at}. Prediction-blind. "
          f"{len(named_groups)} activity groups over {len(catalog)} catalog codes.\n",
          "| Group | Label | GT members (test count) |",
          "|---|---|---|"]
    for g in sorted(named_groups):
        members = gt_by_group.get(g, [])
        if not members:
            continue
        mtxt = ", ".join(f"{c} ({n})" for c, n in members)
        md.append(f"| {g} | {GROUP_LABELS.get(g, g)} | {mtxt} |")
    md.append("\n## Frozen rules (ordered, first-match)\n")
    md.append("| Rule | Group | Kind | Pattern |")
    md.append("|---|---|---|---|")
    for rid, gid, kind, pat in GROUP_RULES:
        md.append(f"| {rid} | {gid} | {kind} | `{pat}` |")
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text("\n".join(md))
    print(f"\nSaved: {args.out}\n       {args.md}")


if __name__ == "__main__":
    main()
