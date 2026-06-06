"""One-off inspector — call the layout service on selected GS PDFs, dump raw text to disk.

Helps diagnose why `extract_gs_sections.py` produces empty applicability lists
on most GS methodologies. Run AFTER the bulk extraction batch finishes (to
avoid GPU contention).

Usage:
    python3 -m carbonmm.ingest._inspect_gs_raw 405 411 414 417 418

Writes:
    carbonmm/data/_raw_inspect/{code}.txt
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

from .common import CORPUS_DIR
from .extract_gs_sections import (
    call_layout_service,
    LAYOUT_URL_DEFAULT,
    NUMBERED_HEADING,
    BARE_HEADING,
    classify_heading,
    strip_toc_entries,
)


def main() -> int:
    codes = sys.argv[1:] or ["405", "411", "414", "417", "418", "402"]
    out_root = CORPUS_DIR.parent / "_raw_inspect"
    out_root.mkdir(parents=True, exist_ok=True)
    base_url = os.environ.get("LAYOUT_SERVICE_URL", LAYOUT_URL_DEFAULT)

    for code in codes:
        pdf_dir = CORPUS_DIR / "GS" / code
        pdfs = list(pdf_dir.glob("*.pdf"))
        if not pdfs:
            print(f"{code}: no PDF in {pdf_dir}")
            continue
        pdf = max(pdfs, key=lambda p: p.stat().st_size)
        container_path = f"methodology-corpus/GS/{code}/{pdf.name}"
        print(f"\n=== {code} ({pdf.name}) ===")
        resp = call_layout_service(container_path, base_url)
        if resp is None:
            print(f"  ERR: the layout service returned None")
            continue
        full_text = resp.get("full_text", "")
        n_pages = resp.get("page_count", 0)
        out = out_root / f"{code}.txt"
        out.write_text(full_text)
        print(f"  pages={n_pages} text_len={len(full_text):,} → {out}")

        # Quick diagnostics
        stripped = strip_toc_entries(full_text)
        print(f"  After TOC strip: {len(full_text):,} → {len(stripped):,} chars")

        # Heading classification
        hits = []
        for m in NUMBERED_HEADING.finditer(stripped):
            cat = classify_heading(m.group(1)) or "_"
            hits.append((m.start(), cat, m.group(0)[:80]))
        for m in BARE_HEADING.finditer(stripped):
            cat = "BARE:" + (classify_heading(m.group(1)) or "_")
            hits.append((m.start(), cat, m.group(0)[:80]))
        hits.sort()
        print(f"  {len(hits)} heading candidates (first 6):")
        for h in hits[:6]:
            print(f"    @{h[0]:6d} [{h[1]}] {h[2]!r}")

        # Search for likely section labels in raw text (case-insensitive)
        import re as _re
        for needle in ["APPLICABILITY", "Applicability", "ELIGIBILITY", "Eligibility",
                       "SCOPE", "Scope", "BASELINE", "Baseline", "MONITORING", "Monitoring",
                       "DEFINITION", "Definition", "BOUNDARIES", "Boundary"]:
            ms = list(_re.finditer(rf"\b{needle}\b", full_text))
            if ms:
                # Show first 3 with context line
                samples = []
                for m in ms[:3]:
                    # find line containing this
                    line_start = full_text.rfind("\n", 0, m.start()) + 1
                    line_end = full_text.find("\n", m.end())
                    line = full_text[line_start:line_end if line_end > 0 else m.end()+80].strip()[:120]
                    samples.append(f"@{m.start():6d}: {line!r}")
                print(f"  '{needle}': {len(ms)} hits — {samples}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
