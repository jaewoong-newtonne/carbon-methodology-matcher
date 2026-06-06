"""Extract the project-description section (GS "Section A" or VCS equivalent)
from PDD body.json files into a separate per-PDD JSON.

Scope: the methodology recommender's input is restricted to this section only.
Sections B/2/3 onward (where the chosen
methodology is declared) are dropped from the recommender input entirely; Pass
2 LLM verification is also dropped because Section A is structurally clean of
methodology references.

Template auto-detection:
  * GS               — `SECTION A. DESCRIPTION OF PROJECT` → `SECTION B. APPLICATION ...`
  * VCS-ProjectDesc  — first `1.NN <Title>` subsection → first `2.NN`/`3.NN` or
                       methodology-keyword start
  * VCS-CDM-SSC      — `SECTION A. General description ...` → `A.4` subsection
                       (we keep A.1-A.3 only — A.4+ contains project-boundary
                       and baseline material that declares methodology choice)
  * unknown          — record `extraction_status: skipped` and continue

Usage:
    python3 -m carbonmm.ingest.extract_pdd_section_a
    python3 -m carbonmm.ingest.extract_pdd_section_a --code GS11044
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Literal, Optional


# ─── Shared helpers ───────────────────────────────────────────────────────

# TOC entries end in dot-leader + page number; strip them before boundary
# detection so accidental matches on TOC don't anchor at the document head.
# Same regex as ingest/extract_gs_sections.py:171.
TOC_LINE = re.compile(r"^[^\n]*?\.{8,}\s*\d+\s*$\n?", re.MULTILINE)


def strip_toc_entries(text: str) -> str:
    return TOC_LINE.sub("", text)


# Recurring per-page header injected by the layout service on every VCS page — must be
# stripped before boundary detection so it doesn't confuse subsection regex.
RECURRING_PAGE_HEADERS = [
    re.compile(r"Project\s+Description:?\s+VCS\s+Version\s+[\d.]+", re.IGNORECASE),
    re.compile(r"<!--\s*PAGE\s+BREAK\s*-->", re.IGNORECASE),
]


def strip_recurring_headers(text: str) -> str:
    for pat in RECURRING_PAGE_HEADERS:
        text = pat.sub("", text)
    return text


TemplateType = Literal[
    "GS", "GS-CDM-CPA-DD",
    "VCS-ProjectDesc", "VCS-CDM-SSC",
    "unknown",
]


# ─── Template detection ───────────────────────────────────────────────────

# Separators between "SECTION A" and the title text: period, hyphen, en-dash
# (U+2013), em-dash (U+2014), colon, or whitespace.
_SEP = r"[\s\.\-–—:]+"

GS_SIGNATURE = re.compile(
    r"(?im)"
    r"SECTION\s+A" + _SEP + r"DESCRIPTION\s+OF\s+PROJECT"
    r"|VPA\s+DESIGN\s+DOCUMENT"
    r"|KEY\s+PROJECT\s+INFORMATION\s*[&\sAND]+\s*VPA"
    r"|Gold\s+Standard\s+for\s+the\s+Global\s+Goals"
)
# CDM-CPA-DD (Component Project Activity Design Document) shows up under
# `registry=GS` in our eval set (GS PoA component projects use the CDM form).
GS_CDM_CPA_DD_SIGNATURE = re.compile(
    r"(?im)CDM[-\s]?CPA[-\s]?DD"
    r"|Component\s+project\s+activity\s+design\s+document"
)
VCS_CDM_SSC_SIGNATURE = re.compile(
    r"(?im)SECTION\s+A" + _SEP + r"General\s+description"
    r"|CDM[-\s]?SSC[-\s]?PDD"
    r"|Small[-\s]?scale\s+(?:CDM\s+)?project\s+activity"
)
VCS_PROJECTDESC_SIGNATURE = re.compile(
    r"(?im)PROJECT\s+DESCRIPTION:?\s+VCS\s+Version"
    r"|PROJECT\s+DETAILS"
)
# 1st-generation CDM-PDD (different from CDM-CPA-DD and CDM-SSC-PDD).
CDM_PDD_SIGNATURE = re.compile(
    r"(?im)PROJECT\s+DESIGN\s+DOCUMENT\s+FORM\s*\(CDM[-\s]?PDD\)"
    r"|CLEAN\s+DEVELOPMENT\s+MECHANISM\s+PROJECT\s+DESIGN"
)
# GS-standalone (non-VPA) project design document.
GS_STANDALONE_SIGNATURE = re.compile(
    r"(?im)KEY\s+PROJECT\s+INFORMATION\s*[&\s]+\s*PROJECT\s+DESIGN\s+DOCUMENT"
)
# Body-level heuristic — used when cover signatures fail.
BODY_HAS_SECTION_A_B = re.compile(
    r"(?im)^[ \t]*SECTION\s+A[\s\.\-–—:]"
)
BODY_HAS_SECTION_B = re.compile(
    r"(?im)^[ \t]*SECTION\s+B[\s\.\-–—:]"
)
BODY_HAS_VCS_NUMBERED = re.compile(r"(?m)^[ \t]*1\.\d{1,2}\s+[A-Z][a-z]")


def _has_section_AB(text: str) -> bool:
    return (BODY_HAS_SECTION_A_B.search(text) is not None
            and BODY_HAS_SECTION_B.search(text) is not None)


def detect_template(text: str) -> TemplateType:
    """Identify the PDD template via tiered detection.

    Tier 1 — strong cover signatures over the first 20k chars.
    Tier 2 — body-level structural heuristics (any 'SECTION A.' + 'SECTION B.'
    pair anywhere → treat as GS-style; '1.X' subsection markers → VCS-style).
    """
    head = text[:20000]
    # Tier 1: strong cover signatures, ordered most specific first
    if CDM_PDD_SIGNATURE.search(head):
        return "GS-CDM-CPA-DD"  # same A→B extractor works for both CDM-PDD and CDM-CPA-DD
    if GS_CDM_CPA_DD_SIGNATURE.search(head):
        return "GS-CDM-CPA-DD"
    if VCS_CDM_SSC_SIGNATURE.search(head):
        return "VCS-CDM-SSC"
    if VCS_PROJECTDESC_SIGNATURE.search(head):
        return "VCS-ProjectDesc"
    if GS_SIGNATURE.search(head) or GS_STANDALONE_SIGNATURE.search(head):
        return "GS"
    # Tier 2: body-structural fallback
    if _has_section_AB(text):
        return "GS"  # any SECTION A→B body → use GS max-gap extractor
    if BODY_HAS_VCS_NUMBERED.search(text):
        return "VCS-ProjectDesc"
    return "unknown"


# ─── Section A extraction per template ────────────────────────────────────


# GS body headers — match either the inline form (SECTION A. Description of...)
# or the standalone-line form (SECTION A. on its own line). The structural
# requirement is a SECTION A/B header at line start with any reasonable
# separator following — title text is optional because cross-references like
# "Section A.5 of this PDD" are filtered by the line-start anchor (they appear
# mid-paragraph in body, not at line start).
GS_SECTION_A = re.compile(r"(?im)^[ \t]*SECTION\s+A[\s\.\-–—:](?!\d)")
GS_SECTION_B = re.compile(r"(?im)^[ \t]*SECTION\s+B[\s\.\-–—:](?!\d)")
# GS-CDM-CPA-DD uses naked 'SECTION A.' on its own line, then the title on the
# next line ('Description of component project activity (CPA)'). Match either
# the inline form or the bare 'SECTION A.' standalone form.
GS_CDM_CPA_SECTION_A = re.compile(r"(?im)^[ \t]*SECTION\s+A\.?\s*$")
GS_CDM_CPA_SECTION_B = re.compile(r"(?im)^[ \t]*SECTION\s+B\.?\s*$")

VCS_CDM_SECTION_A = re.compile(r"(?im)^[ \t]*SECTION\s+A" + _SEP + r"General\s+description")
VCS_CDM_A4 = re.compile(r"(?im)^[ \t]*A\.4\.?\s+")
VCS_CDM_SECTION_C = re.compile(r"(?im)^[ \t]*SECTION\s+[CD]\.?\s+")

# VCS-ProjectDesc subsection markers — title-cased word required to avoid
# matching mid-text decimals like "(version 3.0)" or list items like "3.36".
VCS_SUBSECTION_1 = re.compile(r"(?m)^[ \t]*1\.(\d{1,2})\s+([A-Z][A-Za-z][^\n]{4,80})")
VCS_SUBSECTION_2_OR_3 = re.compile(r"(?m)^[ \t]*[23]\.(\d{1,2})\s+([A-Z][A-Za-z][^\n]{4,80})")
# Fallback end marker if no Section 2/3 subsection found — methodology-narrative
# keywords that overwhelmingly indicate Section 3 territory.
VCS_METHODOLOGY_KW = re.compile(
    r"(?im)\b(?:APPLICATION\s+OF\s+METHODOLOGY"
    r"|Title\s+and\s+Reference\s+of\s+Methodology"
    r"|Applicability\s+of\s+Methodology"
    r"|Project\s+Boundary"
    r"|Baseline\s+Scenario"
    r"|Baseline\s+Emissions"
    r")\b"
)


@dataclass
class ExtractionResult:
    globalId: str
    registry: str
    template_type: TemplateType
    section_a_text: str
    source_char_offset: int
    source_char_end: int
    extraction_method: str
    extraction_status: str  # "ok" | "skipped:<reason>"
    char_count: int


def _max_gap_AB(stripped: str,
                a_pat: re.Pattern, b_pat: re.Pattern,
                method_label: str) -> Optional[tuple[int, int, str]]:
    a_hits = list(a_pat.finditer(stripped))
    b_hits = list(b_pat.finditer(stripped))
    if not a_hits or not b_hits:
        return None
    best = None
    for a in a_hits:
        for b in b_hits:
            if b.start() <= a.start():
                continue
            span = b.start() - a.start()
            if best is None or span > best[2]:
                best = (a.start(), b.start(), span)
    if best is None:
        return None
    return best[0], best[1], method_label


GS_FALLBACK_END_KEYWORDS = re.compile(
    r"(?im)^[ \t]*(?:"
    r"SECTION\s+[BCD]"
    r"|B\.\s+Application"
    r"|Application\s+of\s+(?:approved\s+)?(?:Gold\s+Standard\s+)?Methodology"
    r"|APPROVED\s+(?:GOLD\s+STANDARD\s+)?METHODOLOGY"
    r")"
)


def _extract_gs(stripped: str) -> Optional[tuple[int, int, str]]:
    """Find the A→B pair with the largest span (i.e. the body, not the TOC).

    If a SECTION B header isn't found, try a fallback keyword set (e.g.,
    'Application of Methodology' on its own line) so PDDs with non-standard
    section labels still yield a usable boundary.
    """
    span = _max_gap_AB(stripped, GS_SECTION_A, GS_SECTION_B, "regex_GS_max_gap_AB")
    if span is not None:
        return span
    # Section A found but no Section B — find first methodology-keyword line
    a_hits = list(GS_SECTION_A.finditer(stripped))
    if not a_hits:
        return None
    a_start = a_hits[0].start()
    kw = GS_FALLBACK_END_KEYWORDS.search(stripped, pos=a_start + 50)  # skip the SECTION A line itself
    if kw is not None:
        return a_start, kw.start(), "regex_GS_section_A_to_method_keyword"
    return None


CDM_PDD_LEGACY_A = re.compile(
    r"(?im)^[ \t]*A\.\s+General\s+description\s+of\s+project\s+activity"
)
CDM_PDD_LEGACY_B = re.compile(
    r"(?im)^[ \t]*B\.\s+Application\s+of\s+(?:a\s+)?baseline\s+(?:and\s+)?monitoring"
)


def _extract_gs_cdm_cpa(stripped: str) -> Optional[tuple[int, int, str]]:
    """GS-CDM-CPA-DD: 'SECTION A.' / 'A.' → 'SECTION B.' / 'B.'.

    Tries in order:
      1. Standalone 'SECTION A.' lines (CDM-CPA-DD layout)
      2. Inline 'SECTION A. Description ...' (some variants)
      3. Bare 'A. General description of project activity' (legacy CDM-PDD)
    """
    span = _max_gap_AB(stripped, GS_CDM_CPA_SECTION_A, GS_CDM_CPA_SECTION_B,
                       "regex_GS_CDM_CPA_max_gap_AB")
    if span is not None:
        return span
    span = _max_gap_AB(stripped, GS_SECTION_A, GS_SECTION_B,
                       "regex_GS_CDM_CPA_inline_AB")
    if span is not None:
        return span
    span = _max_gap_AB(stripped, CDM_PDD_LEGACY_A, CDM_PDD_LEGACY_B,
                       "regex_CDM_PDD_legacy_AB")
    if span is not None:
        return span
    # Final fallback: any SECTION A / A. start + methodology-keyword end
    for a_pat, method in [
        (GS_SECTION_A, "regex_GS_CDM_CPA_section_A_to_method_kw"),
        (CDM_PDD_LEGACY_A, "regex_CDM_PDD_legacy_A_to_method_kw"),
        (GS_CDM_CPA_SECTION_A, "regex_GS_CDM_CPA_standalone_A_to_method_kw"),
    ]:
        a_hits = list(a_pat.finditer(stripped))
        if not a_hits:
            continue
        a_start = a_hits[0].start()
        kw = GS_FALLBACK_END_KEYWORDS.search(stripped, pos=a_start + 50)
        if kw is not None:
            return a_start, kw.start(), method
    return None


def _extract_vcs_cdm_ssc(stripped: str) -> Optional[tuple[int, int, str]]:
    """SECTION A → A.4 boundary, keeping only A.1-A.3 (project description)."""
    a_hits = list(VCS_CDM_SECTION_A.finditer(stripped))
    if not a_hits:
        return None
    start = a_hits[0].start()
    # End at A.4 if present; else fall back to SECTION C; else 10k chars
    a4 = VCS_CDM_A4.search(stripped, pos=start)
    sec_c = VCS_CDM_SECTION_C.search(stripped, pos=start)
    if a4:
        return start, a4.start(), "regex_VCS_CDM_SSC_A1_to_A3"
    if sec_c:
        return start, sec_c.start(), "regex_VCS_CDM_SSC_to_section_C"
    return start, min(start + 10000, len(stripped)), "regex_VCS_CDM_SSC_fallback_10k"


def _extract_vcs_projectdesc(stripped: str) -> Optional[tuple[int, int, str]]:
    """First 1.NN subsection → first 2.NN/3.NN subsection (or methodology kw)."""
    sec1 = VCS_SUBSECTION_1.search(stripped)
    if sec1 is None:
        return None
    start = sec1.start()
    sec23 = VCS_SUBSECTION_2_OR_3.search(stripped, pos=start)
    if sec23 is not None:
        return start, sec23.start(), "regex_VCS_ProjectDesc_1_to_2or3"
    kw = VCS_METHODOLOGY_KW.search(stripped, pos=start)
    if kw is not None:
        return start, kw.start(), "regex_VCS_ProjectDesc_1_to_methodology_kw"
    return start, len(stripped), "regex_VCS_ProjectDesc_1_to_EOF"


def extract_section_a(full_text: str, body_meta: dict) -> ExtractionResult:
    # TOC strip first, then template detect on text that still contains the
    # recurring page-header signature ("Project Description: VCS Version 4.0").
    # Only after detection do we strip recurring headers — otherwise the
    # VCS-ProjectDesc signature gets removed before it can be matched.
    toc_stripped = strip_toc_entries(full_text)
    template = detect_template(toc_stripped)
    stripped = strip_recurring_headers(toc_stripped)

    pdd_id = body_meta["globalId"]
    registry = body_meta["registry"]

    base = dict(
        globalId=pdd_id,
        registry=registry,
        template_type=template,
        section_a_text="",
        source_char_offset=-1,
        source_char_end=-1,
        extraction_method="",
        extraction_status="",
        char_count=0,
    )

    if template == "GS":
        span = _extract_gs(stripped)
    elif template == "GS-CDM-CPA-DD":
        span = _extract_gs_cdm_cpa(stripped)
    elif template == "VCS-CDM-SSC":
        span = _extract_vcs_cdm_ssc(stripped)
    elif template == "VCS-ProjectDesc":
        span = _extract_vcs_projectdesc(stripped)
    else:
        base["extraction_status"] = "skipped:unknown_template"
        return ExtractionResult(**base)

    if span is None:
        base["extraction_status"] = "skipped:no_boundary_found"
        return ExtractionResult(**base)

    start, end, method = span
    section_a = stripped[start:end].strip()
    base.update(
        section_a_text=section_a,
        source_char_offset=start,
        source_char_end=end,
        extraction_method=method,
        extraction_status="ok",
        char_count=len(section_a),
    )
    return ExtractionResult(**base)


# ─── CLI ──────────────────────────────────────────────────────────────────


def iter_body_files(eval_pdd_root: Path) -> Iterator[Path]:
    for reg in ("GS", "VCS"):
        reg_dir = eval_pdd_root / reg
        if not reg_dir.exists():
            continue
        for f in sorted(reg_dir.glob("*/*.body.json")):
            yield f


def main() -> None:
    here = Path(__file__).resolve().parent
    repo_root = here.parent  # carbonmm/
    default_eval = repo_root / "data" / "eval-pdds"
    default_out = repo_root / "data" / "section-a"

    default_restricted = repo_root / "data" / "manifests" / "pdd-eval-set-restricted.json"

    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-pdd-root", type=Path, default=default_eval)
    ap.add_argument("--out-root", type=Path, default=default_out)
    ap.add_argument("--code", type=str, default=None,
                    help="run only on a single PDD by globalId (smoke test)")
    ap.add_argument("--restricted-manifest", type=Path, default=None,
                    help="if set, only process globalIds listed in this restricted "
                    "eval set manifest (e.g., data/manifests/pdd-eval-set-restricted.json). "
                    "Pass an empty path to disable. Default: None (process all).")
    args = ap.parse_args()

    allowed_ids: Optional[set[str]] = None
    if args.restricted_manifest is not None and args.restricted_manifest.exists():
        manifest = json.loads(args.restricted_manifest.read_text())
        allowed_ids = set(manifest["globalIds"])
        print(f"[restricted-manifest] gating to {len(allowed_ids)} PDDs")
    elif default_restricted.exists() and args.code is None:
        # Default behavior: if the restricted manifest exists, gate by it.
        manifest = json.loads(default_restricted.read_text())
        allowed_ids = set(manifest["globalIds"])
        print(f"[restricted-manifest auto] gating to {len(allowed_ids)} PDDs "
              f"(use --restricted-manifest=/dev/null to disable)")

    if not args.eval_pdd_root.exists():
        sys.exit(f"ERROR: {args.eval_pdd_root} does not exist")
    args.out_root.mkdir(parents=True, exist_ok=True)

    stats: dict[str, int] = {
        "total": 0,
        "GS": 0,
        "VCS-ProjectDesc": 0,
        "VCS-CDM-SSC": 0,
        "unknown": 0,
        "skipped": 0,
        "ok": 0,
        "empty_full_text": 0,
    }
    char_lengths: list[int] = []
    by_template_status: dict[str, dict[str, int]] = {}

    for body_path in iter_body_files(args.eval_pdd_root):
        body = json.loads(body_path.read_text())
        pdd_id = body["globalId"]
        if args.code and pdd_id != args.code:
            continue
        if allowed_ids is not None and pdd_id not in allowed_ids:
            continue
        stats["total"] += 1
        registry = body["registry"]

        full_text = body.get("full_text") or ""
        if not full_text:
            stats["empty_full_text"] += 1
            stats["skipped"] += 1
            continue

        result = extract_section_a(full_text, body)
        stats[result.template_type] = stats.get(result.template_type, 0) + 1

        if result.extraction_status == "ok":
            stats["ok"] += 1
            char_lengths.append(result.char_count)
        else:
            stats["skipped"] += 1

        # Stratified status counter for the report
        bucket = by_template_status.setdefault(result.template_type, {})
        bucket[result.extraction_status] = bucket.get(result.extraction_status, 0) + 1

        out_dir = args.out_root / registry / pdd_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{pdd_id}.section-a.json"
        out_path.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False))

    print(f"=== Section A extraction — {stats['total']} PDDs processed ===")
    print(f"  Templates:")
    for t in ("GS", "GS-CDM-CPA-DD", "VCS-ProjectDesc", "VCS-CDM-SSC", "unknown"):
        n = stats.get(t, 0)
        print(f"    {t:<20} {n}")
    print(f"  ok:               {stats['ok']}")
    print(f"  skipped:          {stats['skipped']}")
    print(f"    empty full_text:  {stats['empty_full_text']}")
    if char_lengths:
        char_lengths.sort()
        n = len(char_lengths)
        print(f"  Section A char count (ok PDDs):")
        print(f"    min={char_lengths[0]}  median={char_lengths[n//2]}  "
              f"p90={char_lengths[int(0.9*n)]}  max={char_lengths[-1]}")
    print()
    print(f"  Per-template status breakdown:")
    for t, statuses in sorted(by_template_status.items()):
        print(f"    {t}:")
        for status, count in sorted(statuses.items()):
            print(f"      {status:<40} {count}")


if __name__ == "__main__":
    main()
