"""CDM Methodology Booklet → structured per-methodology JSON.

Input: the CDM Methodology Booklet PDF (286 pages, BSR/UNFCCC publication,
       December 2022, "14th edition up to EB 116"). Point --pdf at a local copy.

The booklet renders **one methodology per page** in a consistent 6-row table
(checked against AM0001, AMS-I.A., AMS-I.B., AR-AMS0003 pages):

    | Header line                                              | Top-right: CODE |
    | (big H1) CODE Title text ...                                              |
    +----------------------+--------------------------------------------+
    | Typical project(s)   | (project description)                      |
    +----------------------+--------------------------------------------+
    | Type of GHG          | (mitigation action bullets)                |
    | emissions mitigation |                                            |
    | action               |                                            |
    +----------------------+--------------------------------------------+
    | Important conditions | (eligibility criteria bullets)             |
    | under which the      |  → the GraphRAG retrieval target           |
    | methodology is       |                                            |
    | applicable           |                                            |
    +----------------------+--------------------------------------------+
    | Important parameters | At validation:                             |
    |                      |   • ...                                    |
    |                      | Monitored:                                 |
    |                      |   • ...                                    |
    +----------------------+--------------------------------------------+
    | BASELINE SCENARIO    | (text + diagram)                           |
    +----------------------+--------------------------------------------+
    | PROJECT SCENARIO     | (text + diagram)                           |
    +----------------------+--------------------------------------------+

This bypasses cdm.unfccc.int (Incapsula-blocked) — the booklet PDF was
distributed by BSR via UNFCCC and is hand-checked to match the website's
methodology summary sheets at the time of December 2022 publication.

Usage:
    python -m carbonmm.ingest.load_cdm_booklet
    python -m carbonmm.ingest.load_cdm_booklet --max-n 5    # smoke
    python -m carbonmm.ingest.load_cdm_booklet --code AM0001
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import yaml

from .common import (
    CORPUS_DIR,
    MANIFEST_DIR,
    MethodologyEntry,
    save_manifest,
    setup_logging,
    _utcnow,
)

logger = logging.getLogger(__name__)

BOOKLET_PATH = (
    Path(__file__).resolve().parents[3]
    / "data" / "reference"
    / "cdm_methodology_booklet_v04.pdf"
)
BOOKLET_SOURCE_TAG = "booklet:cdm_methodology_booklet_v04.pdf"
BOOKLET_VERSION = "December 2022 (up to EB 116)"

# Code patterns for top-right header detection
CODE_PATTERNS = [
    re.compile(r'^(ACM\d{4})$'),
    re.compile(r'^(AM\d{4})$'),
    re.compile(r'^(AMS-[IVX]+\.[A-Z]{1,2}\.?)$'),
    re.compile(r'^(AR-AM\d{4})$'),
    re.compile(r'^(AR-AMS\d{4})$'),
    re.compile(r'^(AR-AMS-[IVX]+\.[A-Z]{1,2}\.?)$'),
]

# Table anchors — must have ≥2 to confirm methodology page
TABLE_ANCHORS = [
    "Typical project(s)",
    "Important conditions",
    "BASELINE SCENARIO",
    "PROJECT SCENARIO",
]

# Section split regex — multi-line tolerant
SECTION_REGEXES = {
    "typical_projects":     re.compile(r"Typical project\(s\)", re.IGNORECASE),
    "mitigation_action":    re.compile(r"Type of GHG emissions\s+mitigation action", re.IGNORECASE),
    "applicability":        re.compile(r"Important conditions under\s+which the methodology is\s+applicable", re.IGNORECASE),
    "parameters":           re.compile(r"Important parameters", re.IGNORECASE),
    "baseline_scenario":    re.compile(r"BASELINE SCENARIO"),
    "project_scenario":     re.compile(r"PROJECT SCENARIO"),
}

SECTION_ORDER = [
    "typical_projects", "mitigation_action", "applicability",
    "parameters", "baseline_scenario", "project_scenario",
]


@dataclass
class BookletMethodology:
    """Per-methodology structured record from the CDM booklet."""

    code: str
    title: str
    page: int                          # 1-indexed
    registry: str = "CDM"
    booklet_version: str = BOOKLET_VERSION
    source: str = BOOKLET_SOURCE_TAG
    typical_projects: str = ""
    mitigation_action: str = ""
    applicability: list[str] = field(default_factory=list)  # bullet list
    parameters_at_validation: list[str] = field(default_factory=list)
    parameters_monitored: list[str] = field(default_factory=list)
    baseline_scenario: str = ""
    project_scenario: str = ""
    icons: list[str] = field(default_factory=list)  # e.g. "Women and children", "Suppressed demand"
    extracted_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def detect_code(page) -> Optional[str]:
    """Find the methodology code in the page's top-right header block.

    Strategy: scan blocks with x0 > 350 and y0 < 60 (top-right). The code
    appears as a standalone line ("AM0001", "AMS-I.A.", "AR-AM0014", etc.).
    """
    blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, type)
    candidates = []
    for x0, y0, x1, y1, btext, _bno, _btype in blocks:
        if x0 < 350 or y0 > 60:
            continue
        # Stripped lines from this block
        for line in btext.split("\n"):
            s = line.strip().rstrip(":")
            for pat in CODE_PATTERNS:
                if pat.match(s):
                    candidates.append((y0, x0, s))
                    break
    if not candidates:
        return None
    # Prefer top-most, then right-most
    candidates.sort(key=lambda t: (t[0], -t[1]))
    return candidates[0][2]


def is_methodology_page(text: str) -> bool:
    """A methodology page must have ≥2 of the canonical table anchors."""
    hits = sum(1 for a in TABLE_ANCHORS if a in text)
    return hits >= 2


def detect_title(page, code: str) -> str:
    """Title block sits at y in [70, 130], x near left edge.

    Block text usually reads "{CODE} {TITLE}" (e.g., "AM0001 Decomposition of …").
    """
    blocks = page.get_text("blocks")
    candidates = []
    for x0, y0, x1, y1, btext, _bno, _btype in blocks:
        if x0 > 100 or y0 < 60 or y0 > 130:
            continue
        candidates.append((y0, btext.strip()))
    if not candidates:
        return ""
    candidates.sort()  # top-most first
    raw = candidates[0][1]
    # Strip leading code from title text
    raw = re.sub(rf"^{re.escape(code)}\s+", "", raw).strip()
    # Strip trailing newlines / footer noise
    raw = raw.split("\n")[0].strip()
    return raw


def split_sections(text: str) -> dict[str, str]:
    """Split the page text into named sections by anchor labels."""
    # Find positions of all known section anchors
    matches: list[tuple[int, str]] = []
    for key, regex in SECTION_REGEXES.items():
        for m in regex.finditer(text):
            matches.append((m.start(), key))
    matches.sort()
    if not matches:
        return {}
    sections: dict[str, str] = {}
    for i, (start, key) in enumerate(matches):
        end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        chunk = text[start:end]
        # Strip the anchor itself from the start
        m = SECTION_REGEXES[key].match(chunk)
        body = chunk[m.end():] if m else chunk
        sections[key] = body.strip()
    return sections


def parse_bullets(text: str) -> list[str]:
    """Convert a bullet-list text chunk into a list of items.

    Bullets in the booklet are encoded as standalone "•" lines followed by
    the item text on subsequent lines, ended by ";" or "." or newline.
    """
    # Replace " • " runs with newlines for easier splitting
    normalized = re.sub(r"\n\s*•\s*\n", "\n• ", text)
    normalized = re.sub(r"\n\s*•\s+", "\n• ", normalized)
    items = []
    current = []
    for line in normalized.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("•"):
            if current:
                items.append(" ".join(current).strip().rstrip(";.,"))
            current = [line.lstrip("•").strip()]
        else:
            if current:
                current.append(line)
            else:
                current = [line]
    if current:
        items.append(" ".join(current).strip().rstrip(";.,"))
    return [it for it in items if it]


def split_parameters(params_text: str) -> tuple[list[str], list[str]]:
    """Parameters section has 'At validation:' and 'Monitored:' sub-sections."""
    at_val_match = re.search(r"At validation:", params_text)
    monitored_match = re.search(r"Monitored:", params_text)

    at_val: list[str] = []
    monitored: list[str] = []

    if at_val_match and monitored_match:
        at_val_text = params_text[at_val_match.end():monitored_match.start()]
        monitored_text = params_text[monitored_match.end():]
        at_val = parse_bullets(at_val_text)
        monitored = parse_bullets(monitored_text)
    elif monitored_match:
        monitored_text = params_text[monitored_match.end():]
        monitored = parse_bullets(monitored_text)
    elif at_val_match:
        at_val_text = params_text[at_val_match.end():]
        at_val = parse_bullets(at_val_text)
    else:
        # No labels — treat the whole thing as a single bullet list
        monitored = parse_bullets(params_text)
    return at_val, monitored


def parse_page(page_idx: int, page) -> Optional[BookletMethodology]:
    """Parse a single PDF page into a BookletMethodology, or None if not a methodology page."""
    text = page.get_text()
    if not is_methodology_page(text):
        return None

    code = detect_code(page)
    if not code:
        return None

    title = detect_title(page, code)
    sections = split_sections(text)

    at_val, monitored = split_parameters(sections.get("parameters", ""))

    # Clean mitigation_action: it's a small bullet list, render as one string
    mit_raw = sections.get("mitigation_action", "")
    mit_items = parse_bullets(mit_raw) or [mit_raw.strip()]
    mit_clean = " | ".join(it for it in mit_items if it)

    # Clean baseline/project scenarios: collapse whitespace and stop at any
    # repeated code+title line or post-scenario diagram labels (they show up
    # as a tail after PROJECT SCENARIO due to PDF reading order).
    def _clean_scenario(s: str) -> str:
        s = re.sub(r"\s+", " ", s).strip()
        # Cut at the repeated "{CODE}  {title}" tail (e.g., "AM0001  Decomposition…")
        cut = re.search(rf"\s+{re.escape(code)}\s+\w+", s)
        if cut:
            s = s[:cut.start()].strip()
        return s

    return BookletMethodology(
        code=code,
        title=title,
        page=page_idx + 1,
        typical_projects=re.sub(r"\s+", " ", sections.get("typical_projects", "")).strip(),
        mitigation_action=mit_clean,
        applicability=parse_bullets(sections.get("applicability", "")),
        parameters_at_validation=at_val,
        parameters_monitored=monitored,
        baseline_scenario=_clean_scenario(sections.get("baseline_scenario", "")),
        project_scenario=_clean_scenario(sections.get("project_scenario", "")),
        extracted_at=_utcnow(),
    )


def iter_methodologies(pdf_path: Path, *, max_n: Optional[int] = None) -> Iterator[BookletMethodology]:
    import fitz

    doc = fitz.open(str(pdf_path))
    count = 0
    for i, page in enumerate(doc):
        m = parse_page(i, page)
        if m is None:
            continue
        yield m
        count += 1
        if max_n and count >= max_n:
            break


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf", type=Path, default=BOOKLET_PATH,
        help="Booklet PDF path (default: %(default)s)",
    )
    parser.add_argument(
        "--manifest", type=Path, default=MANIFEST_DIR / "cdm-booklet.yaml",
        help="Manifest YAML path (default: data/manifests/cdm-booklet.yaml)",
    )
    parser.add_argument(
        "--corpus-dir", type=Path, default=CORPUS_DIR / "CDM",
        help="Per-methodology JSON output dir (default: data/corpus/CDM/)",
    )
    parser.add_argument("--max-n", type=int, default=None, help="Cap for smoke test")
    parser.add_argument("--code", default=None, help="Extract only the given code (for spot check)")
    parser.add_argument("--dry-run", action="store_true", help="Parse but do not write")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    if not args.pdf.exists():
        logger.error("Booklet PDF not found at %s. Provide a local copy via --pdf.", args.pdf)
        return 1

    entries: list[MethodologyEntry] = []
    n_parsed = 0
    for m in iter_methodologies(args.pdf, max_n=args.max_n):
        if args.code and m.code != args.code:
            continue
        n_parsed += 1
        logger.info(
            "[p.%3d] %-12s applicability=%d params(val/mon)=%d/%d | %s",
            m.page, m.code,
            len(m.applicability),
            len(m.parameters_at_validation), len(m.parameters_monitored),
            m.title[:55],
        )

        if not args.dry_run:
            out_dir = args.corpus_dir / m.code
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{m.code}.json").write_text(
                json.dumps(m.to_dict(), indent=2, ensure_ascii=False)
            )

        # Build flat manifest entry mirroring GS scraper schema
        entries.append(MethodologyEntry(
            registry="CDM",
            code=m.code,
            title=m.title,
            detail_url=f"booklet://page-{m.page}",
            pdf_url="",
            pdf_local_path=str(args.pdf),  # all CDM share the single booklet PDF
            fetched_at=m.extracted_at,
            notes=(
                f"page={m.page}; applicability_items={len(m.applicability)}; "
                f"params_val={len(m.parameters_at_validation)}; "
                f"params_mon={len(m.parameters_monitored)}; "
                f"source={BOOKLET_SOURCE_TAG}"
            ),
        ))
        if args.code:
            break

    logger.info("Parsed %d methodology pages.", n_parsed)

    if not args.dry_run and entries:
        save_manifest(entries, args.manifest)
        logger.info("Wrote manifest: %s", args.manifest)
        logger.info("Per-methodology JSONs under: %s/", args.corpus_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
