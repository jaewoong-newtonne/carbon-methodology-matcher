"""Structured feature extraction from PDD text via the inference transport.

Pass: extract {ghg_species, sectoral_scope, technology, country_iso, scale}
from the project description. These structured features feed the graph filter
(`graph_filter.py`) which prunes methodology candidates that violate hard
constraints (e.g., scale mismatch, scope mismatch).

Routes LLM calls through the configured inference transport (self-hosted daemon,
default ``http://localhost:8765/chat``).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

DAEMON_URL = "http://localhost:8765/chat"
DAEMON_TIMEOUT = 90.0
DEFAULT_MODEL = "claude-haiku-4-5"

ALLOWED_GHG = {"CO2", "CH4", "N2O", "HFC", "PFC", "SF6"}

# CDM 15 sectoral scopes (canonical form: NN-kebab)
SECTORAL_SCOPES = {
    "01": "energy-industries",
    "02": "energy-distribution",
    "03": "energy-demand",
    "04": "manufacturing-industries",
    "05": "chemical-industries",
    "06": "construction",
    "07": "transport",
    "08": "mining-mineral",
    "09": "metal-production",
    "10": "fugitive-fuel",
    "11": "fugitive-halocarbon",
    "12": "solvent-use",
    "13": "waste-handling",
    "14": "afforestation-reforestation",
    "15": "agriculture",
}

ALLOWED_SCALE = {"small", "large", "unknown"}

PROMPT_TEMPLATE = """You are an environmental project classifier. Extract the structured features below from this carbon-project description. Return ONLY a JSON object with no prose.

# Project description (truncated)

{project_text}

# Output JSON schema (strict)

{{
  "ghg_species": [array of strings from: "CO2", "CH4", "N2O", "HFC", "PFC", "SF6"],
  "sectoral_scope": one of "01-energy-industries", "02-energy-distribution", "03-energy-demand", "04-manufacturing-industries", "05-chemical-industries", "06-construction", "07-transport", "08-mining-mineral", "09-metal-production", "10-fugitive-fuel", "11-fugitive-halocarbon", "12-solvent-use", "13-waste-handling", "14-afforestation-reforestation", "15-agriculture",
  "technology": short keyword phrase (e.g., "wind", "solar-PV", "small-hydro", "landfill-gas-flaring", "biomass-cookstove", "clean-cookstove", "improved-cookstove", "afforestation", "REDD+", "rice-cultivation", "energy-efficiency-lighting"),
  "country_iso": 2-letter ISO country code (e.g., "TR" for Turkey, "BR" for Brazil, "CN" for China; "" if unclear),
  "scale": one of "small", "large", "unknown" (small = capacity ≤15 MW OR ≤60 ktCO2e/yr reductions; else large)
}}

JSON only.
"""


@dataclass
class PDDFeatures:
    ghg_species: list[str] = field(default_factory=list)
    sectoral_scope: str = ""
    technology: str = ""
    country_iso: str = ""
    scale: str = "unknown"
    raw_response: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_json_object(text: str) -> dict:
    """Tolerate code fences + leading prose; return the first balanced JSON object."""
    # Try fenced
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Balanced-brace scan
    start = text.find("{")
    if start < 0:
        return {}
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except Exception:
                    return {}
    return {}


def _normalize_scope(raw: str) -> str:
    if not raw:
        return ""
    s = str(raw).strip().lower()
    # "01" or "1" → "01-energy-industries"
    m = re.match(r"^0?(\d{1,2})[-_\s]*(.*)$", s)
    if m and m.group(1).zfill(2) in SECTORAL_SCOPES:
        n = m.group(1).zfill(2)
        return f"{n}-{SECTORAL_SCOPES[n]}"
    # Already canonical?
    for n, name in SECTORAL_SCOPES.items():
        canon = f"{n}-{name}"
        if canon in s or name in s:
            return canon
    return ""


def _validate(d: dict) -> PDDFeatures:
    ghg_raw = d.get("ghg_species") or []
    if isinstance(ghg_raw, str):
        ghg_raw = [ghg_raw]
    ghg = [g.strip().upper() for g in ghg_raw if isinstance(g, str)]
    ghg = [g for g in ghg if g in ALLOWED_GHG]

    scope = _normalize_scope(d.get("sectoral_scope", ""))

    tech = (d.get("technology") or "").strip().lower()
    tech = re.sub(r"\s+", "-", tech)[:80]

    iso = (d.get("country_iso") or "").strip().upper()
    if not re.match(r"^[A-Z]{2,3}$", iso):
        iso = ""

    scale = (d.get("scale") or "").strip().lower()
    if scale not in ALLOWED_SCALE:
        scale = "unknown"

    return PDDFeatures(
        ghg_species=ghg,
        sectoral_scope=scope,
        technology=tech,
        country_iso=iso,
        scale=scale,
    )


def extract_features(
    pdd_text: str,
    *,
    daemon_url: str = DAEMON_URL,
    model: str = DEFAULT_MODEL,
    timeout: float = DAEMON_TIMEOUT,
    max_chars: int = 6_000,
) -> PDDFeatures:
    """POST PDD text to self-hosted inference endpoint, return validated PDDFeatures."""
    import time as _t
    prompt = PROMPT_TEMPLATE.format(project_text=pdd_text[:max_chars])
    last_exc = None
    for attempt in range(3):
        try:
            with httpx.Client(timeout=timeout) as c:
                resp = c.post(
                    daemon_url,
                    json={
                        "prompt": prompt,
                        "model": model,
                        "timeout_s": min(timeout - 5.0, 75.0),
                        "use_cache": True,
                    },
                )
            resp.raise_for_status()
            text = resp.json()["text"]
            parsed = _extract_json_object(text)
            feats = _validate(parsed)
            feats.raw_response = text[:500]
            return feats
        except httpx.HTTPStatusError as e:
            last_exc = e
            if e.response.status_code in (500, 502, 503) and attempt < 2:
                _t.sleep(3 + attempt * 2)
                continue
            raise
    raise last_exc


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdd-id", required=True)
    args = ap.parse_args()

    EVAL = Path(__file__).resolve().parents[1] / "data" / "eval-pdds"
    body = None
    for f in EVAL.rglob(f"{args.pdd_id}.body.json"):
        body = json.loads(f.read_text())
        break
    if body is None:
        raise SystemExit(f"PDD {args.pdd_id} not found")
    text = body.get("full_text") or ""
    feats = extract_features(text)
    print(json.dumps(feats.to_dict(), indent=2))
