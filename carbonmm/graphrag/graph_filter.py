"""Rule-based graph constraint filter over methodology candidates.

Given extracted PDD features (from `extract_features.py`) and a candidate set
(from `retrieve.py`), compute:

  1. HARD-EXCLUDE methodologies that violate strict constraints. The only
     hard-exclude rule used here is the *scale* rule: AMS-prefixed CDM
     methodologies are small-scale-only by definition, so they cannot apply
     to projects classified as `scale="large"`.

  2. GRAPH-MATCH count per candidate (0-4). For each of the 4 features
     (scale, sectoral_scope, technology, ghg_species), check whether any
     clause text of the candidate mentions the feature. This count enters
     the score fusion (`score_fusion.py`) with weight γ.

Pure-Python over the corpus JSONs — no TypeDB dependency. This is the
"rule-based constraint filter" referenced in paper §3.3.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

EMBED_DIR = Path(__file__).resolve().parents[1] / "data" / "embeddings"

# Family / scope inference patterns (paper §3.3 footnote)
AMS_PREFIX_RE = re.compile(r"^AMS-")  # small-scale-only CDM methodologies
AR_PREFIX_RE = re.compile(r"^AR-")  # afforestation/reforestation CDM
SCOPE_KEYWORDS = {
    "01-energy-industries": [
        "electricity generation", "power plant", "renewable", "grid-connected",
        "wind", "solar", "hydro", "geothermal", "biomass power", "PV",
    ],
    "02-energy-distribution": ["transmission", "distribution loss", "grid"],
    "03-energy-demand": [
        "energy efficiency", "demand-side", "lighting", "cookstove", "appliance",
    ],
    "04-manufacturing-industries": ["cement", "steel", "ceramic", "glass", "pulp"],
    "05-chemical-industries": ["ammonia", "nitric acid", "adipic acid", "chemical"],
    "07-transport": ["transport", "vehicle", "fleet", "bus", "rail"],
    "08-mining-mineral": ["mining", "mineral"],
    "10-fugitive-fuel": ["fugitive", "coal mine methane", "venting"],
    "11-fugitive-halocarbon": ["halocarbon", "HFC", "PFC"],
    "12-solvent-use": ["solvent"],
    "13-waste-handling": [
        "landfill", "waste", "compost", "biogas", "anaerobic digestion", "manure",
    ],
    "14-afforestation-reforestation": [
        "afforestation", "reforestation", "forest", "tree planting", "REDD",
    ],
    "15-agriculture": [
        "agriculture", "soil carbon", "rice cultivation", "tillage", "fertilizer",
    ],
}

TECH_SYNONYMS = {
    "wind": ["wind", "turbine", "WPP"],
    "solar-PV": ["solar", "photovoltaic", "PV"],
    "hydro": ["hydro", "hydropower", "run-of-river"],
    "small-hydro": ["small hydro", "mini-hydro", "run-of-river"],
    "biomass": ["biomass", "bagasse", "agricultural residue"],
    "landfill-gas": ["landfill gas", "LFG"],
    "landfill-gas-flaring": ["landfill", "flaring", "LFG"],
    "cookstove": ["cookstove", "stove", "improved stove"],
    "clean-cookstove": ["cookstove", "stove", "improved", "efficient"],
    "improved-cookstove": ["cookstove", "stove", "improved"],
    "biomass-cookstove": ["biomass", "stove", "cookstove"],
    "afforestation": ["afforestation", "tree planting", "forest"],
    "REDD+": ["REDD", "deforestation", "forest"],
    "rice-cultivation": ["rice", "paddy"],
    "energy-efficiency": ["efficiency", "demand-side"],
    "energy-efficiency-lighting": ["lighting", "lamp", "LED", "CFL"],
    "geothermal": ["geothermal"],
    "transport": ["transport", "vehicle"],
}

GHG_NAMES = {
    "CO2": ["CO2", "carbon dioxide", "CO₂"],
    "CH4": ["CH4", "methane"],
    "N2O": ["N2O", "nitrous oxide"],
    "HFC": ["HFC", "hydrofluorocarbon", "refrigerant"],
    "PFC": ["PFC", "perfluorocarbon"],
    "SF6": ["SF6"],
}


@dataclass
class GraphFilterResult:
    excluded: bool
    reasons: list[str]
    match_count: int          # 0-4, used as γ feature in score fusion
    match_detail: dict        # which features matched


@lru_cache(maxsize=1)
def _code_to_concat_text() -> dict[str, str]:
    """One concatenated lowercased blob per methodology code (cached)."""
    df = pd.read_parquet(EMBED_DIR / "clauses-meta.parquet")
    out: dict[str, list[str]] = {}
    for row in df.itertuples(index=False):
        out.setdefault(row.code, []).append((row.clause_text or "").lower())
    return {c: " ".join(parts) for c, parts in out.items()}


def _scale_excluded(code: str, scale: str) -> bool:
    """AMS-* methodologies are small-scale-only by definition."""
    if scale == "large" and AMS_PREFIX_RE.match(code):
        return True
    return False


def _scope_match(text: str, scope: str) -> bool:
    if not scope:
        return False
    kws = SCOPE_KEYWORDS.get(scope, [])
    return any(kw.lower() in text for kw in kws)


def _tech_match(text: str, technology: str) -> bool:
    if not technology:
        return False
    keys = TECH_SYNONYMS.get(technology, [technology])
    return any(k.lower() in text for k in keys)


def _ghg_match(text: str, ghg_species: list[str]) -> bool:
    if not ghg_species:
        return False
    for sp in ghg_species:
        for name in GHG_NAMES.get(sp, [sp]):
            if name.lower() in text:
                return True
    return False


def _scale_match(text: str, scale: str, code: str) -> bool:
    """Match if the methodology's stated scale matches the project's."""
    if scale == "small":
        # AMS- prefix is small by convention; or explicit mention
        return AMS_PREFIX_RE.match(code) is not None or "small-scale" in text
    if scale == "large":
        # large-scale ACM/AM/VM/GS-EN families (heuristic: non-AMS)
        return AMS_PREFIX_RE.match(code) is None
    return False


def filter_one(code: str, features) -> GraphFilterResult:
    """Compute hard-exclude + match_count for one candidate methodology."""
    text = _code_to_concat_text().get(code, "")

    excluded = False
    reasons: list[str] = []
    if _scale_excluded(code, features.scale):
        excluded = True
        reasons.append(f"scale={features.scale} but code is AMS-* (small-only)")

    match = {
        "scale": _scale_match(text, features.scale, code),
        "scope": _scope_match(text, features.sectoral_scope),
        "technology": _tech_match(text, features.technology),
        "ghg": _ghg_match(text, features.ghg_species),
    }
    match_count = sum(1 for v in match.values() if v)

    return GraphFilterResult(
        excluded=excluded, reasons=reasons, match_count=match_count, match_detail=match
    )


def filter_candidates(
    candidates: list[str], features
) -> dict[str, GraphFilterResult]:
    """Apply filter_one to a list of candidate methodology codes."""
    return {c: filter_one(c, features) for c in candidates}


if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdd-id", required=True)
    ap.add_argument("--candidates", nargs="+", default=None,
                    help="explicit candidate codes; defaults to retrieve top-30")
    args = ap.parse_args()

    from .extract_features import extract_features
    EVAL = Path(__file__).resolve().parents[1] / "data" / "eval-pdds"
    body = None
    for f in EVAL.rglob(f"{args.pdd_id}.body.json"):
        body = json.loads(f.read_text())
        break
    if body is None:
        raise SystemExit(f"PDD {args.pdd_id} not found")
    text = body.get("full_text") or ""

    feats = extract_features(text)
    print(f"Features: {feats.to_dict()}\n")

    if args.candidates:
        codes = args.candidates
    else:
        from .retrieve import HybridRetriever
        R = HybridRetriever.load_default()
        hits = R.query(text, top_k=30)
        codes = list({row["code"] for row, _ in hits})

    results = filter_candidates(codes, feats)
    print(f"{'code':14s}  {'excl':6s}  {'matches':10s}  detail")
    for code, r in sorted(results.items(), key=lambda kv: -kv[1].match_count):
        print(
            f"{code:14s}  {('YES' if r.excluded else '-'):6s}  "
            f"{r.match_count}/4         {r.match_detail}"
        )
