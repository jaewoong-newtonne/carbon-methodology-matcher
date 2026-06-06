"""Pass 1 redaction patterns — single source of truth.

Extracted from `pipeline.py` so both the redaction pipeline and the κ-audit
pool builder can load the same patterns. This module has no project-internal
imports so it is safe to load via `importlib.util.spec_from_file_location`
across packages whose path contains a hyphen (`carbonmm/`).
"""

from __future__ import annotations

import re


REPLACEMENT = "[REDACTED:METHODOLOGY_REF]"

# Patterns ordered by specificity, plus extensions for codes observed in the
# methodology catalog (VMR0006, GS-EN-001 native, etc.).
# Sentence-boundary detector. Four kinds of boundary:
#   1. standard sentence-ending punctuation followed by whitespace + capital
#      letter (the lookahead avoids matching on abbreviations like "AMS-II.G,")
#   2. blank-line paragraph break
#   3. HTML table row close `</tr>` — treats each `<tr>...</tr>` row as a unit,
#      preventing whole-table over-redaction when a single cell mentions a
#      methodology code (e.g., a project-info table that includes a
#      "methodology" row also containing "PDD form version", "GS ID", etc.).
#   4. HTML table close `</table>` — terminates the redaction span at a table
#      end so cross-table collapse cannot happen.
_SENTENCE_BOUNDARY = re.compile(
    r"(?:"
    r"[.!?](?:\s|[)\"”])+(?=[A-Z\"“(\[])"
    r"|"
    r"\n\s*\n"
    r"|"
    r"</tr>"
    r"|"
    r"</table>"
    r")"
)


def expand_match_to_sentence(text: str, start: int, end: int) -> tuple[int, int]:
    """Given a match span (start, end), return the enclosing sentence span.

    Three boundary categories:
      - `[.!?]\\s+(?=[A-Z])` : sentence INCLUDES the punctuation; the next
        sentence starts right after the whitespace.
      - `\\n\\s*\\n` (paragraph break) and `</tr>`/`</table>` (HTML row/table
        boundaries): sentence excludes the boundary itself so the structural
        marker is preserved verbatim in the surrounding text.
    """
    sentence_start = 0
    for m in _SENTENCE_BOUNDARY.finditer(text, 0, start):
        sentence_start = m.end()
    m = _SENTENCE_BOUNDARY.search(text, end)
    if m:
        matched = m.group(0)
        if matched.startswith("\n") or matched.startswith("<"):
            sentence_end = m.start()
        else:
            sentence_end = m.start() + 1
            while sentence_end < len(text) and text[sentence_end] in "\"”)]":
                sentence_end += 1
    else:
        sentence_end = len(text)
    return sentence_start, sentence_end


def apply_pass1_sentence_level(text: str) -> tuple[str, list[dict]]:
    """Apply the 15 Pass-1 redaction patterns at *sentence* granularity.

    For each match, the entire enclosing sentence is replaced by the single
    REPLACEMENT sentinel. Overlapping sentence spans are merged so multiple
    matches in the same sentence collapse to one redaction.

    This matches the original specification ("Replace match + enclosing
    sentence with [REDACTED:METHODOLOGY_REF]") and addresses the partial-catch
    failure mode where Pass 1 captured the methodology code but left version
    suffixes, quoted long titles, or methodology-narrative residue intact.

    Returns (redacted_text, marks_list). Marks contain pattern_name, match
    string, and offset in the ORIGINAL text — same schema as the legacy
    word-level apply_pass1, so downstream code is unaffected.
    """
    raw_matches: list[tuple[int, int, str, str]] = []
    for pname, pat in REDACTION_PATTERNS:
        for m in pat.finditer(text):
            raw_matches.append((m.start(), m.end(), pname, m.group(0)))
    if not raw_matches:
        return text, []

    # Marks log (for audit / per-PDD mark counts) keeps the original word-level offsets
    marks = [{"pattern": pname, "match": match_str, "offset": start}
             for start, _, pname, match_str in raw_matches]

    # Expand each match to its sentence span, then merge overlapping spans
    spans = [expand_match_to_sentence(text, s, e) for s, e, _, _ in raw_matches]
    spans.sort()
    merged: list[tuple[int, int]] = [spans[0]]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    # Substitute back-to-front so earlier offsets stay valid
    out = text
    for s, e in reversed(merged):
        out = out[:s] + REPLACEMENT + out[e:]
    return out, marks


REDACTION_PATTERNS = [
    # CDM
    ("AR-AMS-Roman", re.compile(r"\bAR[-\s]?AMS[-\s]*[IVX]{1,5}\b", re.IGNORECASE)),
    ("AR-AM####",    re.compile(r"\bAR[-\s]?AM\d{4}\b", re.IGNORECASE)),
    ("AR-ACM####",   re.compile(r"\bAR[-\s]?ACM\d{4}\b", re.IGNORECASE)),
    ("ACM####",      re.compile(r"\bACM\d{4}\b")),
    ("AM####",       re.compile(r"\bAM\d{4}\b")),
    ("AMS-Roman",    re.compile(r"\bAMS[-\s]*[IVX]{1,5}(?:\.[A-Z]{1,3})?\.?", re.IGNORECASE)),
    # Verra
    ("VMR####",      re.compile(r"\bVMR\d{4}\b")),
    ("VMD####",      re.compile(r"\bVMD\d{4}\b")),
    ("VM####",       re.compile(r"\bVM\d{4}\b")),
    ("VT####",       re.compile(r"\bVT\d{4}\b")),
    # Gold Standard
    ("TPDDTEC",      re.compile(r"\bTPDDTEC\b", re.IGNORECASE)),
    ("GS-VER",       re.compile(r"\bGS[-\s]*VER\d*\b", re.IGNORECASE)),
    ("GS-XX-###",    re.compile(r"\bGS-(?:EN|AG|WA|FW|LU|FS|HI|EE|RE|WM|WMH|OTH|FO|TR|CS|BCFW|FM|ICS|ESG|MS)-\d{2,4}\b", re.IGNORECASE)),
    ("GS-Passport-#", re.compile(r"\bGS[-\s]?Passport\b", re.IGNORECASE)),
    # CDM verbose-form catcher (single very long line)
    ("verbose-METHODOLOGY-FOR-...-V #", re.compile(
        r"\bMethodology\s+(?:for|to)\s+[^.\n]{10,200}\b(?:V|version)\s*\d+(?:\.\d+)?\b",
        re.IGNORECASE,
    )),
]
