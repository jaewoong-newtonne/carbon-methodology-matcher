"""Extract PDD body text via the DocLayout-YOLO layout service.

Reads the Genvision PDD eval set (data/manifests/genvision-pdd-eval-set.json)
and the actual PDFs already downloaded by download_pdd_eval_set.py to
data/eval-pdds/{REGISTRY}/{globalId}/. For each PDD, calls the layout service
/parse/extract and writes a per-project JSON to the same directory:

    data/eval-pdds/{REGISTRY}/{globalId}/{globalId}.body.json
    {
      "globalId": "VCS10",
      "registry": "VCS",
      "name": "BAESA Project",
      "country": "...",
      "methodology_label": "ACM0002",        # ground-truth label from manifest
      "secondary_labels": [...],
      "creditingPeriodStartDate": "2020-04-06",
      "totalCreditsIssued": 6726799,
      "source_pdf": "VCS_PD_V04.pdf",
      "page_count": 42,
      "full_text": "...",                     # the layout service /parse/extract output
      "extracted_at": "2026-05-21T..."
    }

This output is the input to the redaction pipeline: the regex+LLM passes
operate on `full_text` paragraph-by-paragraph.

Sequential by the layout service's GPU-bound contract (the DocLayout-YOLO
model runs one PDF at a time). ~50s/PDF average → ~15-16h for 1,160 PDDs.
Designed to run as a long-lived background job.

Usage:
    python -m carbonmm.ingest.extract_pdd_body
    python -m carbonmm.ingest.extract_pdd_body --skip-existing
    python -m carbonmm.ingest.extract_pdd_body --max-n 5     # smoke
    python -m carbonmm.ingest.extract_pdd_body --registry VCS
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from .common import DATA_ROOT, MANIFEST_DIR, RateLimiter, setup_logging, _utcnow
from .extract_gs_sections import call_layout_service, LAYOUT_URL_DEFAULT

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST = MANIFEST_DIR / "genvision-pdd-eval-set.json"
PDD_ROOT = DATA_ROOT / "eval-pdds"


def find_pdd_pdf(registry: str, global_id: str, expected_filename: str) -> Optional[Path]:
    """Find the downloaded PDF for a project. Prefers exact filename match,
    falls back to the only PDF in the project dir."""
    proj_dir = PDD_ROOT / registry / global_id
    if not proj_dir.exists():
        return None
    # exact match
    exact = proj_dir / expected_filename
    if exact.exists():
        return exact
    # URL-encoded variant might be on disk; try unquoted
    from urllib.parse import unquote
    cand = proj_dir / unquote(expected_filename)
    if cand.exists():
        return cand
    # Otherwise, take the largest PDF in the dir (should be unique since we
    # only downloaded primary_pdd per project)
    pdfs = sorted(proj_dir.glob("*.pdf"), key=lambda p: p.stat().st_size, reverse=True)
    return pdfs[0] if pdfs else None


def extract_one(entry: dict, base_url: str) -> Optional[dict]:
    """Call the layout service on one PDD and assemble the body record."""
    registry = entry["registry"]
    global_id = entry["globalId"]
    fname = entry.get("primary_pdd_filename") or ""
    pdf = find_pdd_pdf(registry, global_id, fname)
    if pdf is None:
        logger.warning("[%s/%s] no PDF on disk", registry, global_id)
        return None

    # Container path within the layout service's mounted volume. The container's
    # DOCUMENT_STORAGE_DIR maps $DATA_ROOT/downloads/ → /downloads/, so the
    # eval-pdds tree must be reachable under that mount (e.g. mirrored there) for
    # the service to read it.
    container_path = f"eval-pdds/{registry}/{global_id}/{pdf.name}"
    logger.info("→ %s/%s (%s, %d KB)", registry, global_id, pdf.name, pdf.stat().st_size // 1024)
    resp = call_layout_service(container_path, base_url, timeout=300.0)
    if resp is None:
        return None

    methods = entry.get("methodologies") or []
    primary_label = methods[0] if methods else None
    secondary = methods[1:] if len(methods) > 1 else []

    return {
        "globalId": global_id,
        "registry": registry,
        "name": entry.get("name"),
        "country": entry.get("country"),
        "methodology_label": primary_label,
        "secondary_labels": secondary,
        "creditingPeriodStartDate": entry.get("creditingPeriodStartDate"),
        "totalCreditsIssued": entry.get("totalCreditsIssued"),
        "source_pdf": pdf.name,
        "page_count": resp.get("page_count", 0),
        "full_text": resp.get("full_text", ""),
        "extracted_at": _utcnow(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--registry", default=None,
                        help="Filter to a single registry (GS / VCS).")
    parser.add_argument("--code", default=None,
                        help="Process only this globalId (smoke test).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip projects whose *.body.json already exists.")
    parser.add_argument("--max-n", type=int, default=None,
                        help="Stop after processing this many entries.")
    parser.add_argument("--url", default=os.environ.get("LAYOUT_SERVICE_URL", LAYOUT_URL_DEFAULT))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    manifest = json.loads(args.manifest.read_text())
    entries = manifest["entries"]
    if args.registry:
        entries = [e for e in entries if e["registry"] == args.registry]
    if args.code:
        entries = [e for e in entries if e["globalId"] == args.code]

    if args.skip_existing:
        before = len(entries)
        kept = []
        for e in entries:
            out = PDD_ROOT / e["registry"] / e["globalId"] / f"{e['globalId']}.body.json"
            if not out.exists():
                kept.append(e)
        entries = kept
        logger.info("--skip-existing: %d → %d entries", before, len(entries))

    if args.max_n:
        entries = entries[: args.max_n]

    logger.info("Planned %d PDDs (the layout service at %s)", len(entries), args.url)

    rl = RateLimiter(requests_per_second=0.5)  # 2s gap, GPU-bound
    n_ok = 0
    n_fail = 0
    for i, e in enumerate(entries, 1):
        rl.wait()
        body = extract_one(e, base_url=args.url)
        if body is None:
            n_fail += 1
            continue
        out = PDD_ROOT / e["registry"] / e["globalId"] / f"{e['globalId']}.body.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(body, indent=2, ensure_ascii=False))
        n_ok += 1
        logger.info("  ✓ %s/%s | %d pages, %d chars → %s",
                    e["registry"], e["globalId"], body["page_count"], len(body["full_text"]), out.name)
        if i % 25 == 0:
            logger.info("  …%d/%d (ok=%d fail=%d)", i, len(entries), n_ok, n_fail)

    logger.info("Done. ok=%d fail=%d / total=%d", n_ok, n_fail, len(entries))
    return 0


if __name__ == "__main__":
    sys.exit(main())
