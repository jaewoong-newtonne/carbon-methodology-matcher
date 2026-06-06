"""Section-aware 3-tier rich-text assembler for the leakage-aware eval pilot.

Input: a PDD full_text (preferably the Pass1-redacted body) + body_meta.
Output: {"rich_text", "tier", "template", "sections_kept"} — full Section A plus
the fact-bearing Section B subsections (project boundary, monitoring/parameter
tables), with methodology-narrative subsections (applicability, baseline,
emission calculations) excluded.
"""
from __future__ import annotations
import re
import sys
from importlib import import_module
from pathlib import Path

# Ensure repo root is on sys.path so import_module("carbonmm...") works
# when this module is loaded via spec_from_file_location in tests.
_REPO_ROOT = str(Path(__file__).resolve().parents[4])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_sa = import_module("carbonmm.ingest.extract_pdd_section_a")

# Subsections to KEEP from Section B: project boundary and monitoring/parameters.
KEEP_B = re.compile(
    r"(?im)^[ \t]*(?:B\.\d+\.?\s+)?(?:"
    r"Project\s+Boundary"
    r"|Monitoring(?:\s+Plan)?"
    r"|Data\s+and\s+[Pp]arameters?(?:\s+Monitored)?"
    r"|Parameters?\s+Monitored"
    r")\b"
)

# Subsection start markers used to BOUND the end of a kept span.
# Note: word boundary (\b) is placed on keyword alternatives only; the
# B\.\d+ structural pattern does not need it (and \b would fail after \S
# when the next char is also a word char, e.g. "B.4 Baseline").
NEXT_SUBSECTION = re.compile(
    r"(?im)^[ \t]*(?:"
    r"B\.\d+\.?\s+\S"
    r"|SECTION\s+[BCD]"
    r"|Title\s+and\s+Reference\s+of\s+Methodology\b"
    r"|Applicability\s+of\s+Methodology\b"
    r"|Baseline\s+Scenario\b"
    r"|Baseline\s+Emissions\b"
    r"|Estimation\s+of\s+Emission\s+Reductions\b"
    r")"
)
TABLE_NEAR_MONITOR = re.compile(r"(?is)(?:Monitor|Parameter)[^\n]{0,200}?(<table>.*?</table>)")
CAPACITY = re.compile(r"\d+(?:\.\d+)?\s*[MkG]W(?:e|th|p)?\b", re.I)


def _b_fact_spans(stripped: str, b_start: int) -> list[tuple[int, int]]:
    spans = []
    for m in KEEP_B.finditer(stripped, b_start):
        nxt = NEXT_SUBSECTION.search(stripped, m.end())
        end = nxt.start() if nxt else min(m.start() + 4000, len(stripped))
        spans.append((m.start(), end))
    return spans


def build_rich_text(full_text: str, body_meta: dict) -> dict:
    toc = _sa.strip_toc_entries(full_text)
    template = _sa.detect_template(toc)
    stripped = _sa.strip_recurring_headers(toc)
    res = _sa.extract_section_a(full_text, body_meta)
    sections_kept = []

    if res.extraction_status == "ok" and res.section_a_text:
        parts = [res.section_a_text]
        sections_kept.append("section_a")
        b_start = res.source_char_end if res.source_char_end > 0 else 0
        spans = _b_fact_spans(stripped, b_start)
        for s, e in spans:
            parts.append(stripped[s:e].strip())
        if spans:
            sections_kept.append(f"b_facts:{len(spans)}")
            return {"rich_text": "\n\n".join(p for p in parts if p),
                    "tier": 1, "template": template, "sections_kept": sections_kept}
        tables = [m.group(1) for m in TABLE_NEAR_MONITOR.finditer(stripped)]
        if tables:
            sections_kept.append(f"tables:{len(tables)}")
            return {"rich_text": "\n\n".join(parts + tables),
                    "tier": 2, "template": template, "sections_kept": sections_kept}
        cap = CAPACITY.search(stripped)
        if cap:
            w = stripped[max(0, cap.start() - 600): cap.end() + 600].strip()
            parts.append(w); sections_kept.append("capacity_window")
        return {"rich_text": "\n\n".join(parts), "tier": 3,
                "template": template, "sections_kept": sections_kept}

    cap = CAPACITY.search(stripped)
    head = stripped[:6000]
    if cap:
        head = head + "\n\n" + stripped[max(0, cap.start() - 600): cap.end() + 600]
        sections_kept.append("capacity_window")
    return {"rich_text": head.strip(), "tier": 3,
            "template": template, "sections_kept": sections_kept or ["head"]}
