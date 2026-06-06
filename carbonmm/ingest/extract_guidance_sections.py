"""Extract structured sections from REGISTRY FRAMEWORK RULEBOOK PDFs.

Companion to extract_gs_sections.py, but for the guidance harvest
(scrape_guidance.py) rather than methodology documents. Walks
  data/corpus/{REG}/_guidance/{doc_id}/*.pdf
reads the harvester manifest (guidance-review-{reg}.yaml) for doc metadata,
POSTs each PDF to the same DocLayout-YOLO layout service, splits sections, and
writes  {doc_id}.guidance.json  (definitions + applicability clauses, aligned
1:1 with the methodology clause sub-graph) next to the PDF.

For SSC-threshold-bearing docs (e.g. CDM-EB66-A23-GUID "General guidelines for
SSC CDM methodologies") the optional --llm pass extracts the Type I/II/III
numeric caps into  {doc_id}.thresholds.json  via OpenAI gpt-4o-mini
structured-output. Definitions/applicability need no LLM.

Reuses extract_gs_sections: call_layout_service, split_by_headers, parse_bullets.

Usage (run from repo root, once the _guidance PDFs are reachable by the layout service):
    # dry-run: list guidance PDFs that would be parsed
    python3 -m carbonmm.ingest.extract_guidance_sections --registry all --dry-run

    # full parse (sections only)
    python3 -m carbonmm.ingest.extract_guidance_sections --registry all --skip-existing

    # + SSC threshold extraction (needs OPENAI_API_KEY in the repo root .env)
    python3 -m carbonmm.ingest.extract_guidance_sections --registry cdm --llm
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

from .common import CORPUS_DIR, MANIFEST_DIR, RateLimiter, load_manifest, setup_logging, _utcnow
from .extract_gs_sections import (
    LAYOUT_URL_DEFAULT,
    call_layout_service,
    parse_bullets,
    split_by_headers,
)

logger = logging.getLogger(__name__)

# Docs whose SSC Type I/II/III caps we extract with the --llm pass. The
# authoritative source is the CDM PROJECT STANDARD (reg_stan03 PoA / reg_stan04
# PA) — its Type I/II/III definitions state the caps directly (15 MW, 45 MW(th),
# 60 GWh, 60 ktCO2e). EB66-A23-GUID is deliberately EXCLUDED: it only references
# the caps via a 1%-of-threshold example, which LLMs misread as the cap itself.
_SSC_DOC_RE = re.compile(r"reg_stan0[34]|cdm project standard", re.IGNORECASE)


def _manifest_for(registry: str) -> Path:
    return MANIFEST_DIR / f"guidance-review-{registry.lower()}.yaml"


def _load_doc_meta(registry: str) -> dict[str, object]:
    """Map doc_id (== manifest `code`) → MethodologyEntry for one registry."""
    mpath = _manifest_for(registry)
    if not mpath.exists():
        logger.warning("No manifest %s — metadata fields will be blank.", mpath)
        return {}
    return {e.code: e for e in load_manifest(mpath)}


def iter_guidance_pdfs(registry: str):
    """Yield (doc_id, pdf_path) under data/corpus/{REG}/_guidance/."""
    base = CORPUS_DIR / registry / "_guidance"
    if not base.exists():
        logger.warning("No _guidance dir for registry=%s at %s", registry, base)
        return
    for doc_dir in sorted(base.iterdir()):
        if not doc_dir.is_dir():
            continue
        pdfs = [p for p in doc_dir.glob("*.pdf") if p.stat().st_size > 1024]
        if not pdfs:
            continue
        pdfs.sort(key=lambda p: p.stat().st_size, reverse=True)
        yield doc_dir.name, pdfs[0]


def extract_ssc_thresholds(full_text: str, doc_id: str) -> Optional[dict]:
    """Extract SSC Type I/II/III caps from guidance text via gpt-4o-mini.

    Returns {guidance_doc_id, thresholds:[...]} or None if the LLM channel is
    unavailable. Uses OpenAI structured-output for this extraction.
    """
    api_key = _load_openai_key()
    if not api_key:
        logger.warning("--llm requested but OPENAI_API_KEY unavailable; skipping thresholds for %s", doc_id)
        return None
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed; skipping threshold extraction.")
        return None

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "thresholds": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "ssc_type": {"type": "string", "enum": ["I", "II", "III"]},
                        "dimension": {"type": "string"},
                        "cap_value": {"type": "number"},
                        "unit": {"type": "string"},
                        "applies_to_methodologies": {"type": "array", "items": {"type": "string"}},
                        "note": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["ssc_type", "dimension", "cap_value", "unit",
                                 "applies_to_methodologies", "note", "confidence"],
                },
            }
        },
        "required": ["thresholds"],
    }
    prompt = (
        "From the CDM regulatory text below, extract the PRIMARY small-scale (SSC) "
        "eligibility threshold caps from the Type I / II / III definitions. For each give: "
        "ssc_type (I|II|III), dimension (electrical-capacity | thermal-capacity | "
        "mechanical-capacity | annual-energy-savings | annual-emission-reductions), "
        "cap_value (number), unit (MW|MWth|GWh/yr|ktCO2e/yr), applies_to_methodologies "
        "(AMS codes if named, else []), a short note, and confidence 0-1.\n"
        "RULES: (1) For Type I capture BOTH the electrical/mechanical limit (e.g. 15 MW) "
        "AND, if stated, the thermal limit (e.g. 45 MW(th) → unit MWth, dimension "
        "thermal-capacity). (2) IGNORE debundling 'one per cent of the threshold' EXAMPLE "
        "values such as 150 kW / 600 MWh / 600 tCO2 — those are 1%% figures, NOT the caps. "
        "(3) IGNORE bundling sub-limits like 'up to 5 MW'. (4) Only the main Type-definition "
        "caps. Do not invent values.\n\n"
        f"TEXT:\n{full_text[:24000]}"
    )
    client = OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_schema",
                             "json_schema": {"name": "ssc_thresholds", "schema": schema, "strict": True}},
            temperature=0,
        )
        data = json.loads(resp.choices[0].message.content)
    except Exception as e:
        logger.error("threshold extraction failed for %s: %s", doc_id, str(e)[:200])
        return None
    data["guidance_doc_id"] = doc_id
    return data


def _load_openai_key() -> str:
    """OPENAI_API_KEY from env or the repo root .env (env takes precedence)."""
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    env = Path(__file__).resolve().parents[3] / ".env"  # → repo root .env
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    return ""


def extract_one(doc_id: str, pdf: Path, registry: str, meta, base_url: str, do_llm: bool) -> Optional[dict]:
    container_path = f"methodology-corpus/{registry}/_guidance/{doc_id}/{pdf.name}"
    logger.info("→ parsing %s (%s/_guidance)", pdf.name, registry)
    resp = call_layout_service(container_path, base_url)
    if resp is None:
        return None
    full_text = resp.get("full_text", "")
    sections = split_by_headers(full_text)

    out = {
        "guidance_doc_id": doc_id,
        "doc_title": getattr(meta, "title", "") if meta else "",
        "doc_version": getattr(meta, "version", "") if meta else "",
        "authority": getattr(meta, "authority", "") if meta else "",
        "registry": registry,
        "effective_date": getattr(meta, "effective_date", "") if meta else "",
        "source_url": getattr(meta, "pdf_url", "") if meta else "",
        "pdf_sha256": getattr(meta, "pdf_sha256", "") if meta else "",
        "page_count": resp.get("page_count", 0),
        "source": f"the layout service:{registry}/_guidance/{doc_id}/{pdf.name}",
        "definitions": parse_bullets(sections.get("definitions", "")),
        "applicability": parse_bullets(sections.get("applicability", "")),
        "extracted_at": _utcnow(),
    }
    out_path = pdf.parent / f"{doc_id}.guidance.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info("  ✓ %s | defs=%d applic=%d → %s",
                doc_id, len(out["definitions"]), len(out["applicability"]), out_path.name)

    if do_llm and _SSC_DOC_RE.search(doc_id + " " + out["doc_title"]):
        thr = extract_ssc_thresholds(full_text, doc_id)
        if thr:
            tp = pdf.parent / f"{doc_id}.thresholds.json"
            tp.write_text(json.dumps(thr, indent=2, ensure_ascii=False))
            logger.info("  ✓ %s | %d SSC thresholds → %s", doc_id, len(thr.get("thresholds", [])), tp.name)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--registry", default="all", help='Registry under data/corpus/ (GS|CDM|VCS) or "all".')
    p.add_argument("--dry-run", action="store_true", help="List guidance PDFs without calling the layout service.")
    p.add_argument("--skip-existing", action="store_true", help="Skip docs that already have a .guidance.json.")
    p.add_argument("--llm", action="store_true", help="Also extract SSC thresholds (gpt-4o-mini) for SSC docs.")
    p.add_argument("--max-n", type=int, default=None)
    p.add_argument("--url", default=os.environ.get("LAYOUT_SERVICE_URL", LAYOUT_URL_DEFAULT),
                   help="the layout service base URL (default: $LAYOUT_SERVICE_URL or %(default)s)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    setup_logging(args.verbose)

    if args.registry == "all":
        registries = [d.name for d in sorted(CORPUS_DIR.iterdir())
                      if d.is_dir() and (d / "_guidance").exists()]
    else:
        registries = [args.registry]
    logger.info("Registries with _guidance: %s", registries)

    jobs: list[tuple[str, str, Path]] = []
    meta_by_reg: dict[str, dict] = {}
    for reg in registries:
        meta_by_reg[reg] = _load_doc_meta(reg)
        for doc_id, pdf in iter_guidance_pdfs(reg):
            jobs.append((reg, doc_id, pdf))
    if args.skip_existing:
        before = len(jobs)
        jobs = [j for j in jobs if not (j[2].parent / f"{j[1]}.guidance.json").exists()]
        logger.info("--skip-existing: %d → %d jobs", before, len(jobs))
    if args.max_n:
        jobs = jobs[: args.max_n]

    logger.info("Plan: %d guidance PDFs across %d registries", len(jobs), len(registries))
    for r, d, pth in jobs[:8]:
        logger.info("  %s/_guidance/%s → %s (%d KB)", r, d, pth.name, pth.stat().st_size // 1024)
    if len(jobs) > 8:
        logger.info("  ... (%d more)", len(jobs) - 8)

    if args.dry_run:
        logger.info("--dry-run: not calling the layout service. Target URL would be: %s", args.url)
        return 0

    rl = RateLimiter(requests_per_second=0.5)
    n_ok = 0
    for reg, doc_id, pdf in jobs:
        rl.wait()
        if extract_one(doc_id, pdf, reg, meta_by_reg[reg].get(doc_id), args.url, args.llm) is not None:
            n_ok += 1
    logger.info("Done. %d/%d guidance PDFs section-extracted.", n_ok, len(jobs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
