"""Apply Pass 1 regex redaction to each PDD's Section A text.

Inputs from `data/section-a/{REG}/{PID}/{PID}.section-a.json`.
Outputs to `data/section-a-redacted/{REG}/{PID}/{PID}.section-a.redacted.json`.

Section A is structurally a project-description section that precedes the
methodology-declaration sections, so a single regex pass over the 15
high-precision methodology-code patterns is sufficient — no Pass 2 LLM verifier.

Reuses `redact/patterns.py` (single source of truth) and `pipeline.pass1_regex`
via importlib (the project path contains a hyphen that Python's normal import
syntax cannot parse).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator


# Paragraph splitter — mirror of build_kappa_pool.split_paragraphs (same filter,
# preserves [REDACTED]-only paragraphs so per-paragraph processing keeps an
# exact 1:1 mapping between pre and post paragraph_idx).
PARA_SPLIT = re.compile(r"\n\s*\n+")


def split_paragraphs(text: str) -> list[tuple[int, int, str]]:
    """Return list of (start_offset, end_offset, content) for each paragraph.

    Same filter as audit/build_kappa_pool.split_paragraphs — drops <30-char
    fragments unless they contain the [REDACTED:METHODOLOGY_REF] marker.
    """
    out = []
    cursor = 0
    for m in PARA_SPLIT.finditer(text):
        chunk = text[cursor:m.start()]
        stripped = chunk.strip()
        if stripped and (len(stripped) >= 30 or "[REDACTED:METHODOLOGY_REF]" in stripped):
            out.append((cursor, m.start(), stripped))
        cursor = m.end()
    if cursor < len(text):
        chunk = text[cursor:]
        stripped = chunk.strip()
        if stripped and (len(stripped) >= 30 or "[REDACTED:METHODOLOGY_REF]" in stripped):
            out.append((cursor, len(text), stripped))
    return out


def _load_module(rel_path: str):
    p = Path(__file__).resolve().parent.parent / rel_path
    spec = importlib.util.spec_from_file_location(f"_loaded_{p.stem}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_PATTERNS = _load_module("redact/patterns.py")
REDACTION_PATTERNS = _PATTERNS.REDACTION_PATTERNS
REPLACEMENT = _PATTERNS.REPLACEMENT
# Sentence-level Pass 1 — V15 plan's original spec ("Replace match + enclosing
# sentence"). Catches methodology version suffixes, quoted long titles, and
# narrative residue that bare word-level substitution leaves intact.
apply_pass1_sentence_level = _PATTERNS.apply_pass1_sentence_level


def pass1_regex(text: str) -> tuple[str, list[dict]]:
    """Apply Pass 1 sentence-level, processed paragraph-by-paragraph.

    Per-paragraph processing keeps the post-Pass-1 paragraph count exactly equal
    to the pre-Pass-1 paragraph count, which preserves paragraph_idx alignment
    in the audit pool. The text substitution within each paragraph is still
    sentence-level (matches expanded to their enclosing sentence).

    Returns (redacted_text, marks) with marks offsets in the ORIGINAL text.
    Output text is the original paragraphs joined by '\\n\\n', each paragraph
    independently passed through apply_pass1_sentence_level.
    """
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        # No qualifying paragraphs — fall back to whole-text sentence-level
        return apply_pass1_sentence_level(text)

    redacted_chunks: list[str] = []
    all_marks: list[dict] = []
    for p_start, _p_end, p_content in paragraphs:
        redacted_p, marks = apply_pass1_sentence_level(p_content)
        redacted_chunks.append(redacted_p)
        for mark in marks:
            shifted = dict(mark)
            shifted["offset"] = mark["offset"] + p_start
            all_marks.append(shifted)
    return "\n\n".join(redacted_chunks), all_marks


@dataclass
class RedactedRecord:
    globalId: str
    registry: str
    template_type: str
    extraction_method: str
    char_count_section_a: int
    char_count_section_a_pass1: int
    n_pass1_marks: int
    pass1_marks: list[dict] = field(default_factory=list)
    section_a_text_pass1: str = ""


def iter_section_a_files(root: Path) -> Iterator[Path]:
    for reg in ("GS", "VCS"):
        reg_dir = root / reg
        if not reg_dir.exists():
            continue
        for f in sorted(reg_dir.glob("*/*.section-a.json")):
            yield f


def main() -> None:
    here = Path(__file__).resolve().parent
    repo_root = here.parent  # carbonmm/
    default_in = repo_root / "data" / "section-a"
    default_out = repo_root / "data" / "section-a-redacted"

    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", type=Path, default=default_in)
    ap.add_argument("--out-root", type=Path, default=default_out)
    ap.add_argument("--code", type=str, default=None)
    args = ap.parse_args()

    if not args.in_root.exists():
        sys.exit(f"ERROR: {args.in_root} does not exist")
    args.out_root.mkdir(parents=True, exist_ok=True)

    n_processed = n_skipped = 0
    mark_counts: list[int] = []
    pattern_counter: dict[str, int] = {}

    for sa_path in iter_section_a_files(args.in_root):
        sa = json.loads(sa_path.read_text())
        if args.code and sa["globalId"] != args.code:
            continue
        if sa["extraction_status"] != "ok" or not sa["section_a_text"]:
            n_skipped += 1
            continue

        redacted, marks = pass1_regex(sa["section_a_text"])
        rec = RedactedRecord(
            globalId=sa["globalId"],
            registry=sa["registry"],
            template_type=sa["template_type"],
            extraction_method=sa["extraction_method"],
            char_count_section_a=len(sa["section_a_text"]),
            char_count_section_a_pass1=len(redacted),
            n_pass1_marks=len(marks),
            pass1_marks=marks,
            section_a_text_pass1=redacted,
        )

        out_dir = args.out_root / rec.registry / rec.globalId
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{rec.globalId}.section-a.redacted.json").write_text(
            json.dumps(asdict(rec), indent=2, ensure_ascii=False)
        )

        n_processed += 1
        mark_counts.append(len(marks))
        for m in marks:
            pattern_counter[m["pattern"]] = pattern_counter.get(m["pattern"], 0) + 1

    print(f"=== Section A Pass 1 redaction — {n_processed} PDDs processed "
          f"({n_skipped} skipped because Section A extraction failed) ===")
    if mark_counts:
        mark_counts.sort()
        n = len(mark_counts)
        print(f"  Pass 1 marks per PDD:")
        print(f"    min={mark_counts[0]}  mean={sum(mark_counts)/n:.2f}  "
              f"median={mark_counts[n//2]}  p90={mark_counts[int(0.9*n)]}  "
              f"max={mark_counts[-1]}")
        print(f"    PDDs with zero marks: {sum(1 for c in mark_counts if c == 0)} "
              f"({sum(1 for c in mark_counts if c == 0) / n:.1%})")
        print(f"  Top patterns triggered:")
        for pname, count in sorted(pattern_counter.items(), key=lambda x: -x[1])[:8]:
            print(f"    {pname:<35} {count}")


if __name__ == "__main__":
    main()
