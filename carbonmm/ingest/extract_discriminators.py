"""Extract a structured DISCRIMINATOR CARD per methodology in a group, from
the corpus eligibility text, via the configured inference transport.

The 5 PDD features are SHARED within a mitigation-activity group, so they cannot
pick the exact code. A discriminator card captures the axes that actually SEPARATE
members of a group (for G01 grid-renewable: scale tier, renewable sub-technology,
project configuration, grid connection, special eligibility like BESS/reservoir).

Source = the populated corpus JSON (`data/corpus/{reg}/{code}/{code}.json`),
which already holds the methodology PDF text. We clean PDF noise (page-break and
table-HTML artifacts) before extraction.

Output: data/manifests/discriminator-cards.json  = {code: card}

Usage:
    python ingest/extract_discriminators.py --group G01
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

ICDM_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = ICDM_ROOT / "data" / "corpus"
GROUP_MAP = ICDM_ROOT / "data" / "manifests" / "method-groups.json"
OUT = ICDM_ROOT / "data" / "manifests" / "discriminator-cards.json"

DAEMON_URL = "http://localhost:8765/chat"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("disc")

_PAGEBREAK = re.compile(r"<!--\s*PAGE BREAK\s*-->")
_TAGS = re.compile(r"</?(table|tr|td|th)[^>]*>")
_WS = re.compile(r"\s+")


def clean(text: str) -> str:
    text = _PAGEBREAK.sub(" ", text)
    text = _TAGS.sub(" ", text)
    text = _WS.sub(" ", text)
    return text.strip()


def corpus_eligibility_text(code: str, max_chars: int = 7000) -> str | None:
    matches = list(CORPUS_ROOT.glob(f"*/{code}/{code}.json"))
    if not matches:
        return None
    try:
        d = json.loads(matches[0].read_text())
    except Exception:
        return None
    parts = [str(d.get("typical_projects") or "")]
    for c in (d.get("applicability") or [])[:12]:
        parts.append(str(c))
    parts.append(str(d.get("baseline_scenario") or ""))
    return clean(" ".join(parts))[:max_chars]


CARD_PROMPT = """You read carbon-credit methodology eligibility text and extract a STRUCTURED CARD
capturing what makes THIS methodology distinct from sibling methodologies in the same activity.
Return ONLY a JSON object, no prose.

# Methodology code: {code}
# Official title: {title}
# Eligibility / applicability text (cleaned, truncated)

{elig}

# Output JSON schema (strict)
{{
  "scale_tier": one of "small", "large", "either", "unknown"  (small-scale CDM/AMS = small; large-scale ACM/AM = large),
  "renewable_technologies": [subset of "wind","solar-PV","hydro","geothermal","biomass","tidal","wave","mixed","none"],
  "project_configs": [subset of "greenfield","capacity-addition","retrofit","rehabilitation","replacement"],
  "grid_connection": one of "grid-connected","off-grid-or-captive","either","unknown",
  "special_conditions": [short distinctive eligibility phrases, e.g. "BESS integration","reservoir power density limit","biomass co-firing"],
  "excludes": [activity types explicitly NOT eligible, short phrases],
  "one_line": one sentence on the precise scope of this methodology
}}
JSON only."""


def _daemon_post(prompt: str, model: str, timeout: float = 120.0, retries: int = 4) -> str:
    """POST to the daemon with backoff on transient errors (500/502/503/429/timeout)."""
    last = None
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=timeout) as c:
                r = c.post(DAEMON_URL, json={"prompt": prompt, "model": model,
                                             "timeout_s": min(timeout - 5, 90.0), "use_cache": True})
            r.raise_for_status()
            return r.json()["text"]
        except httpx.HTTPStatusError as e:
            last = e
            if e.response.status_code in (429, 500, 502, 503) and attempt < retries - 1:
                time.sleep(2 + attempt * 3)
                continue
            raise
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last = e
            if attempt < retries - 1:
                time.sleep(2 + attempt * 3)
                continue
            raise
    raise last


def daemon_card(code: str, title: str, elig: str, model: str) -> dict:
    prompt = CARD_PROMPT.format(code=code, title=title, elig=elig)
    text = _daemon_post(prompt, model)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"error": "no-json", "raw": text[:200]}
    try:
        return json.loads(m.group(0))
    except Exception as e:
        return {"error": str(e)[:120], "raw": text[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="G01")
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--workers", type=int, default=2)  # match daemon max_concurrent
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    gmap = json.loads(GROUP_MAP.read_text())
    members = gmap["groups"].get(args.group, [])
    titles = {c: gmap["anchor_evidence"].get(c, {}).get("title", c) for c in members}
    logger.info("group %s: %d members", args.group, len(members))

    cards: dict[str, dict] = {}
    if args.out.exists():
        cards = json.loads(args.out.read_text())

    todo = []
    for code in members:
        if code in cards and "error" not in cards[code]:
            continue
        elig = corpus_eligibility_text(code)
        if not elig:
            cards[code] = {"error": "no-corpus"}
            continue
        todo.append((code, titles.get(code, code), elig))
    logger.info("extracting %d cards (model=%s)", len(todo), args.model)

    def work(item):
        code, title, elig = item
        try:
            return code, daemon_card(code, title, elig, args.model)
        except Exception as e:
            return code, {"error": str(e)[:160]}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(work, it) for it in todo]):
            code, card = fut.result()
            cards[code] = card
            logger.info("  %-12s scale=%s tech=%s", code,
                        card.get("scale_tier"), card.get("renewable_technologies"))

    args.out.write_text(json.dumps(cards, indent=2))
    ok = sum(1 for v in cards.values() if "error" not in v)
    logger.info("DONE: %d cards (%d ok) -> %s", len(cards), ok, args.out)


if __name__ == "__main__":
    main()
