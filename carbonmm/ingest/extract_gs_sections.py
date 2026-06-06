"""Extract structured sections from GS methodology PDFs using a layout service.

A small FastAPI HTTP service wraps the public DocLayout-YOLO model
(HuggingFace weights juliozhao/DocLayout-YOLO-DocStructBench; code
https://github.com/opendatalab/DocLayout-YOLO ; pip install doclayout-yolo)
and listens on port 8080. This script:
  1. Iterates data/corpus/GS/{code}/*.pdf
  2. POSTs each PDF to the service's /parse/extract endpoint
  3. Parses the response (per-region layout + structured text)
  4. Filters for the same target sections as the CDM booklet:
       Definitions / Applicability / Eligibility Criteria / Baseline /
       Project boundary / Monitoring
  5. Writes JSON to data/corpus/GS/{code}/{code}.json matching the CDM schema
     so the unified KG-ingest script can treat both registries uniformly.

Set the service endpoint with the LAYOUT_SERVICE_URL environment variable, e.g.:
    export LAYOUT_SERVICE_URL="http://<layout-service-host>:8080"
    python -m carbonmm.ingest.extract_gs_sections

When an LLM is used to identify section boundaries that the layout model alone
cannot resolve, the call routes via the configured inference transport.

Usage:
    # Dry-run: list PDFs that would be processed
    python -m carbonmm.ingest.extract_gs_sections --dry-run

    # Single code smoke test
    python -m carbonmm.ingest.extract_gs_sections --code 402

    # Full run (against the layout service)
    python -m carbonmm.ingest.extract_gs_sections
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .common import CORPUS_DIR, RateLimiter, setup_logging, _utcnow
from .load_cdm_booklet import BookletMethodology  # reuse the structured schema

logger = logging.getLogger(__name__)

LAYOUT_URL_DEFAULT = "http://localhost:8082"  # the layout service base URL

# GS methodology PDFs use 5 distinct heading styles (observed across a sample
# of codes 402, 402-1, 411, 414, 415, 417, 421, 443, 408):
#   A) Arabic top:     "3. Applicability"                           — 402, 423, 444
#   B) Pipe:           "2.2 | Applicability" / "3| SCOPE, APPLIC."  — 408, 443
#   C) Roman:          "III. Applicability" / "I. SOURCE AND APPL." — 414, 417, 421
#   D) Section prefix: "Section I: Source and Applicability"        — 411, 415
#   E) Bare label:     "Applicability A project applying..."        — 402-1..6 Modules
#
# TOC entries are stripped FIRST via `strip_toc_entries()` so heading regexes
# never accidentally anchor on TOC dots. Heading detection is 2-stage:
#   1. NUMBERED_HEADING matches any line starting with one of the 4 numbering
#      styles + heading text. BARE_HEADING matches `Label TITLECASEWORD` line
#      start (Activity Module style).
#   2. classify_heading(text) maps the heading text → category. Priority order
#      matters so "Baseline scenario monitoring" classifies as `baseline`
#      (not `monitoring`).

NUMBERED_HEADING = re.compile(
    r"^[ \t]*"
    r"(?:"
    r"Section\s+(?:\d{1,2}|[IVX]{1,4})\s*[:.\)]"   # "Section I:" / "Section 1."
    r"|"
    r"\d{1,2}(?:\.\d{1,2}){0,2}\s*\|"               # "3|" / "3.2 |"
    r"|"
    r"\d{1,2}(?:\.\d{1,2}){0,2}\."                   # "3." / "3.1." / "3.1.2."
    r"|"
    r"[IVX]{1,4}(?:\.\d{1,2})?\."                    # "III." / "I.3."
    r")"
    r"\s+([^\n]{2,200})",
    re.MULTILINE,
)

# Bare label = Activity-Module style heading where the label sits at line start
# followed by space + capital letter (start of body).
BARE_HEADING = re.compile(
    r"^("
    r"Applicability|Eligibility|"
    r"Baseline\s+(?:scenario|emissions?|methodology|determination)|"
    r"Monitoring|"
    r"Project\s+boundar(?:y|ies)|Boundar(?:y|ies)|"
    r"Definitions?"
    r")\s+[A-Z]",
    re.MULTILINE | re.IGNORECASE,
)


# Keyword regexes for each section category. Order in classify_heading()
# matters: applicability is always preferred when present (it's the GraphRAG
# retrieval target), then ties resolved by earliest keyword position.
_KW_APPLICABILITY = re.compile(r"\b(applicability|eligibility|eligible\s+activit)", re.IGNORECASE)
_KW_DEFINITIONS = re.compile(r"\bdefinition", re.IGNORECASE)
_KW_BASELINE = re.compile(r"\bbaseline\b|calculation\s+of\s+(?:baseline\s+)?emission\s+reduction", re.IGNORECASE)
_KW_BOUNDARY = re.compile(r"\b(project\s+boundar|boundar(?:y|ies)|project\s+scenario)\b", re.IGNORECASE)
_KW_MONITORING = re.compile(r"\bmonitoring\b|\bparameters\s+(?:to\s+be\s+)?monitored\b", re.IGNORECASE)
_KW_SCOPE = re.compile(r"\bscope\b", re.IGNORECASE)


def classify_heading(heading_text: str) -> Optional[str]:
    """Map a heading line's text to a section category, or None if irrelevant.

    Applicability wins if present (highest-priority retrieval target). Ties
    among other categories resolve by which keyword appears EARLIEST in the
    heading text — so "BASELINE AND PROJECT SCENARIO METHODOLOGY 5.1 | Project
    Boundary..." (heading text that spans two parser-joined logical lines)
    classifies as baseline (offset 0), not project_scenario (offset ~30).
    """
    if _KW_APPLICABILITY.search(heading_text):
        return "applicability"
    # Definitions skip "definition of monitoring" false positives.
    candidates = []
    if (m := _KW_DEFINITIONS.search(heading_text)) and "monitor" not in heading_text.lower():
        candidates.append((m.start(), "definitions"))
    if m := _KW_BASELINE.search(heading_text):
        candidates.append((m.start(), "baseline_scenario"))
    if m := _KW_BOUNDARY.search(heading_text):
        candidates.append((m.start(), "project_scenario"))
    if m := _KW_MONITORING.search(heading_text):
        candidates.append((m.start(), "parameters_monitored"))
    if m := _KW_SCOPE.search(heading_text):
        candidates.append((m.start(), "typical_projects"))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def call_layout_service(container_path: str, base_url: str, *, timeout: float = 300.0) -> Optional[dict]:
    """POST {file_path, doc_type} to the layout service /parse/extract.

    `container_path` is relative to DOCUMENT_STORAGE_DIR (= /downloads in
    the layout service container, mapped to $DATA_ROOT/downloads/ on the host
    that runs the service).

    Returns the full JSON response with `full_text`, `all_tables`,
    `page_count`, `source_file`, `doc_type`, or None on error.
    """
    import httpx

    url = f"{base_url.rstrip('/')}/parse/extract"
    payload = {"file_path": container_path, "doc_type": "UNKNOWN"}
    try:
        r = httpx.post(url, json=payload, timeout=timeout)
    except Exception as e:
        logger.error("the layout service request failed (%s): %s", container_path, e)
        return None
    if r.status_code != 200:
        logger.error("the layout service %s returned HTTP %d: %.300s", container_path, r.status_code, r.text)
        return None
    return r.json()


# A TOC entry line ends in a dot-leader followed by a page number. We strip
# these line-by-line BEFORE running heading detection so accidental matches on
# TOC content don't anchor at the top of the document. This handles both the
# "Table of Contents header present" and "no header, just dotted entries" cases.
TOC_LINE = re.compile(r"^[^\n]*?\.{8,}\s*\d+\s*$\n?", re.MULTILINE)


def strip_toc_entries(text: str) -> str:
    return TOC_LINE.sub("", text)


def _find_headings(text: str) -> list[tuple[int, str, str]]:
    """Locate all heading candidates in text. Returns sorted (offset, category, heading_text)."""
    hits = []
    for m in NUMBERED_HEADING.finditer(text):
        heading_text = m.group(0)
        cat = classify_heading(m.group(1))
        if cat:
            hits.append((m.start(), cat, heading_text))
    for m in BARE_HEADING.finditer(text):
        # Bare-label first capture group IS the label itself.
        cat = classify_heading(m.group(1))
        if cat:
            hits.append((m.start(), cat, m.group(0)))
    # Deduplicate by (start, category) — same line might match both NUMBERED and BARE.
    seen = set()
    uniq = []
    for h in sorted(hits):
        key = (h[0], h[1])
        if key not in seen:
            seen.add(key)
            uniq.append(h)
    return uniq


def split_by_headers(text: str) -> dict[str, str]:
    """Walk text once, split by detected section headers.

    Strip TOC entries first, then locate numbered + bare headings, classify
    each, accumulate body until next heading (regardless of category).
    """
    text = strip_toc_entries(text)
    hits = _find_headings(text)

    sections: dict[str, str] = {}
    for i, (start, key, header_text) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        line_end = text.find("\n", start)
        body_start = line_end + 1 if line_end > 0 else start + len(header_text)
        body = text[body_start:end].strip()
        # Per-hit body cap to bound TypeQL attribute values. Multiple hits per
        # category accumulate (e.g. baseline has 3 sub-sections in 443).
        body = body[:8000]
        if key in sections:
            sections[key] += "\n\n" + body
        else:
            sections[key] = body
    return sections


def parse_bullets(text: str) -> list[str]:
    """Extract bullets from a section body.

    Two cases: (1) bullet-form text uses •/-/* or (a)/(1)/a./1. markers at line
    start — split into one item per bullet. (2) paragraph-form text has no
    bullet markers — split on blank-line boundaries so the section text is
    still chunked into retrievable units.

    Codes like 402 / 402-1 / 444 use bullet-form. 411 / 414 / 417 / 421 use
    paragraph-form (sentences flow as paragraphs, no bullets). Both need to
    yield non-empty lists for KG ingest to create clause entities.
    """
    bullet_starts = re.compile(r"^\s*(?:[•\-\*]|\(?[a-z\d]\)?[.\)])\s+", re.MULTILINE)
    items = []
    current = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        if bullet_starts.match(line):
            if current:
                items.append(" ".join(current).strip())
            stripped = bullet_starts.sub("", line).strip()
            current = [stripped]
        else:
            if current:
                current.append(line.strip())
    if current:
        items.append(" ".join(current).strip())
    items = [it for it in items if it]
    if items:
        return items

    # Fallback: paragraph-split. Caps each paragraph at 4000 chars and drops
    # entries shorter than ~80 chars (page-break tags, single-word artifacts).
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    return [p[:4000] for p in paragraphs if len(p) >= 80]


def extract_methodology(pdf_path: Path, code: str, name_hint: str, registry: str, base_url: str) -> Optional[BookletMethodology]:
    """Call the layout service on one PDF, parse out target sections.

    pdf_path is the local path
    (`carbonmm/data/corpus/{registry}/{code}/{name}.pdf`).
    The parser reads from its own mounted volume; we translate the local path
    to the container-relative form
    (`methodology-corpus/{registry}/{code}/{name}.pdf`) — assumes the PDF has
    already been copied under the service's `$DATA_ROOT/downloads/...` mount.
    """
    container_path = f"methodology-corpus/{registry}/{code}/{pdf_path.name}"
    logger.info("→ parsing %s (%s/%s)", pdf_path.name, registry, code)
    resp = call_layout_service(container_path, base_url)
    if resp is None:
        return None
    full_text = resp.get("full_text", "")
    page_count = resp.get("page_count", 0)
    sections = split_by_headers(full_text)

    return BookletMethodology(
        code=code,
        title=name_hint,
        page=page_count,              # whole-doc page count
        registry=registry,
        booklet_version="",
        source=f"genvision-cdn:{registry}/{pdf_path.name}",
        typical_projects=sections.get("typical_projects", "")[:4000],
        mitigation_action="",         # not a separate section in most registries
        applicability=parse_bullets(sections.get("applicability", "")),
        parameters_at_validation=[],  # not split val/monitored as cleanly
        parameters_monitored=parse_bullets(sections.get("parameters_monitored", "")),
        baseline_scenario=sections.get("baseline_scenario", "")[:4000],
        project_scenario=sections.get("project_scenario", "")[:4000],
        extracted_at=_utcnow(),
    )


def iter_registry_pdfs(registry: str):
    """Yield (code, pdf_path) tuples from data/corpus/{registry}/.

    Skips dirs whose only PDF is a Rule-Update (RU_*) or status-update document
    — those aren't methodology specs and produce empty extractions (originally
    observed on GS codes 404/405/406/412/432/440). For each code dir, prefers
    the largest non-RU PDF.
    """
    base = CORPUS_DIR / registry
    if not base.exists():
        logger.error("No corpus dir for registry=%s at %s", registry, base)
        return
    for code_dir in sorted(base.iterdir()):
        if not code_dir.is_dir():
            continue
        pdfs = list(code_dir.glob("*.pdf"))
        if not pdfs:
            continue
        non_ru = [p for p in pdfs if not p.name.startswith("RU_")]
        if non_ru:
            non_ru.sort(key=lambda p: p.stat().st_size, reverse=True)
            yield code_dir.name, non_ru[0]
        else:
            logger.info("skipping %s/%s — only Rule-Update PDF present",
                        registry, code_dir.name)
            continue


# Backwards-compat shim (a few existing tools may import iter_gs_pdfs).
def iter_gs_pdfs():
    return iter_registry_pdfs("GS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="GS",
                        help="Registry directory under data/corpus/ to process. "
                             "Use 'all' to iterate every subdirectory.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List PDFs without calling the layout service.")
    parser.add_argument("--code", default=None,
                        help="Process only this code (smoke test).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip codes that already have a per-code JSON output.")
    parser.add_argument("--max-n", type=int, default=None)
    parser.add_argument("--url", default=os.environ.get("LAYOUT_SERVICE_URL", LAYOUT_URL_DEFAULT),
                        help="the layout service base URL (default: $LAYOUT_SERVICE_URL or %(default)s)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.registry == "all":
        registries = sorted(d.name for d in CORPUS_DIR.iterdir() if d.is_dir())
        logger.info("Registries detected under %s: %s", CORPUS_DIR, registries)
    else:
        registries = [args.registry]

    jobs: list[tuple[str, str, Path]] = []
    for reg in registries:
        for code, pdf in iter_registry_pdfs(reg):
            jobs.append((reg, code, pdf))
    if args.code:
        jobs = [j for j in jobs if j[1] == args.code]
    if args.skip_existing:
        before = len(jobs)
        jobs = [j for j in jobs if not (j[2].parent / f"{j[1]}.json").exists()]
        logger.info("--skip-existing: %d → %d jobs after dropping done ones",
                    before, len(jobs))
    if args.max_n:
        jobs = jobs[: args.max_n]

    logger.info("Plan: %d PDFs across %d registries", len(jobs), len(registries))
    for r, c, p in jobs[:5]:
        logger.info("  %s/%s → %s (%d KB)", r, c, p.name, p.stat().st_size // 1024)
    if len(jobs) > 5:
        logger.info("  ... (%d more)", len(jobs) - 5)

    if args.dry_run:
        logger.info("--dry-run: not calling the layout service. Target URL would be: %s", args.url)
        return 0

    rl = RateLimiter(requests_per_second=0.5)  # 2s gap between PDFs (parser is GPU-bound)
    n_ok = 0
    for reg, code, pdf in jobs:
        rl.wait()
        m = extract_methodology(pdf, code=code, name_hint=code, registry=reg, base_url=args.url)
        if m is None:
            continue
        out = pdf.parent / f"{code}.json"
        out.write_text(json.dumps(asdict(m), indent=2, ensure_ascii=False))
        logger.info(
            "  ✓ %s/%s | applicability=%d params=%d → %s",
            reg, code, len(m.applicability), len(m.parameters_monitored), out.name,
        )
        n_ok += 1

    logger.info("Done. %d/%d PDFs section-extracted.", n_ok, len(jobs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
