"""Genvision catalog dump — paginate /methodologies, emit a manifest JSON.

The Genvision API mirrors UNFCCC/Verra/GS methodology PDFs on its public CDN
(`assets.genvision.com`). The CDN URL is deterministic:

    https://assets.genvision.com/methodologies/{STD}/methodologies/{CODE}/{CODE}_{FILE_ID}.pdf

where FILE_ID = last URL segment of `versions[].pdfUrl` (an UNFCCC/Verra file
storage token). We dump the catalog once, synthesize CDN URLs, and feed those
to a downloader.

Output schema (manifest):
    {
      "version": "0.1",
      "generated_at": "...",
      "totalCount": 594,
      "entries": [
        {
          "code": "ACM0013",
          "title": "...",
          "primary_standard": "CDM",
          "all_standards": ["CDM", "VCS"],
          "is_cdm": true,
          "sectoral_scopes": [{"code": "1", "name": "Energy industries..."}],
          "latest_version": "5.0.0",
          "latest_pdf_url_source": "https://cdm.unfccc.int/...FileStorage/2BMR...",
          "latest_pdf_url_cdn": "https://assets.genvision.com/methodologies/CDM/methodologies/ACM0013/ACM0013_2BMR....pdf",
          "all_versions": [{"version": "5.0.0", "cdn_url": "...", "source_url": "..."}],
          "tools": [{"code": "AM-TOOL-01", "version": "7.0.0", "cdn_url": "...", "source_url": "..."}]
        },
        ...
      ]
    }

Usage:
    python -m carbonmm.ingest.dump_genvision_catalog
    python -m carbonmm.ingest.dump_genvision_catalog --page-size 100
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from urllib.parse import quote

import httpx

from .common import MANIFEST_DIR, _utcnow, setup_logging

logger = logging.getLogger(__name__)

API_BASE = "https://api.registry.genvision.com/v1"
CDN_BASE = "https://assets.genvision.com"


def synth_methodology_cdn_url(std_code: str, code: str, file_id: str) -> str:
    """Synthesize the deterministic CDN URL for a methodology PDF."""
    return f"{CDN_BASE}/methodologies/{quote(std_code)}/methodologies/{quote(code)}/{quote(code)}_{quote(file_id)}.pdf"


def synth_tool_cdn_url(std_code: str, tool_code: str, source_url: str) -> str:
    """Synthesize CDN URL for a tool PDF.

    Verified for CDM AM-TOOL-XX where source_url is like:
        https://cdm.unfccc.int/methodologies/PAmethodologies/tools/am-tool-02-v7.0.pdf
    Pattern: take basename of source_url. Other standards (Verra/GS) may not be
    mirrored under this path — caller should HEAD-check before relying on it.
    """
    basename = source_url.rsplit("/", 1)[-1] if source_url else ""
    return f"{CDN_BASE}/methodologies/{quote(std_code)}/tools/{quote(tool_code)}/{quote(basename)}"


def file_id_from_pdf_url(pdf_url: str) -> str:
    """Extract the file_storage_id (last URL segment) from a versions[].pdfUrl.

    Examples:
        https://cdm.unfccc.int/UserManagement/FileStorage/2BMR6X7ZP3TY89NAWUOI4EGHDK1QFS
            → 2BMR6X7ZP3TY89NAWUOI4EGHDK1QFS
        https://registry.verra.org/.../FileID=12345&IDKEY=abc...
            → "" (Verra storage uses query params, not path segment — caller
              must use API documents[].fileUrl in that case, not synth)
    """
    if not pdf_url:
        return ""
    if "?" in pdf_url:
        # Verra-style: storage ID in query params, can't synth deterministically
        return ""
    return pdf_url.rstrip("/").rsplit("/", 1)[-1]


def fetch_page(client: httpx.Client, api_key: str, page: int, page_size: int) -> dict:
    """One paginated /methodologies call. Returns the parsed JSON body."""
    url = (
        f"{API_BASE}/methodologies?"
        f"pagination%5BpageSize%5D={page_size}&pagination%5Bpage%5D={page}"
    )
    r = client.get(url, headers={"x-api-key": api_key}, timeout=30.0)
    r.raise_for_status()
    return r.json()


def normalize_entry(m: dict) -> dict:
    """Flatten a /methodologies item into a portable manifest row."""
    code = m.get("code") or ""
    standards = m.get("standards") or []
    primary = next((s["standardCode"] for s in standards if s.get("isPrimary")), None) or (
        standards[0]["standardCode"] if standards else None
    )
    all_std = sorted({s["standardCode"] for s in standards if s.get("standardCode")})

    sect_scopes = [
        {"code": s.get("code"), "name": s.get("name"), "standard": s.get("standardCode")}
        for s in (m.get("sectoralScopes") or [])
    ]

    versions = m.get("versions") or []
    # Sort by effective_from desc to pick latest
    versions_sorted = sorted(
        versions,
        key=lambda v: v.get("effectiveFrom") or "",
        reverse=True,
    )

    all_versions = []
    latest_cdn = None
    latest_src = None
    latest_ver_no = None
    for v in versions_sorted:
        src = v.get("pdfUrl") or ""
        fid = file_id_from_pdf_url(src)
        cdn = synth_methodology_cdn_url(primary, code, fid) if (primary and code and fid) else None
        all_versions.append({
            "version": v.get("versionNumber"),
            "effective_from": v.get("effectiveFrom"),
            "effective_to": v.get("effectiveTo"),
            "source_url": src,
            "cdn_url": cdn,
        })
        if latest_cdn is None and cdn:
            latest_cdn = cdn
            latest_src = src
            latest_ver_no = v.get("versionNumber")

    # Tools — only attach if any version has tools[]
    tools_flat = []
    for v in versions_sorted:
        for t in (v.get("tools") or []):
            src = t.get("currentUrl") or ""
            cdn = synth_tool_cdn_url(primary, t.get("code") or "", src) if (primary and t.get("code") and src) else None
            tools_flat.append({
                "code": t.get("code"),
                "title": t.get("title"),
                "version": t.get("currentVersion"),
                "source_url": src,
                "cdn_url": cdn,
            })

    # Dedupe tools by (code, version)
    seen = set()
    tools_unique = []
    for t in tools_flat:
        key = (t.get("code"), t.get("version"))
        if key in seen:
            continue
        seen.add(key)
        tools_unique.append(t)

    return {
        "code": code,
        "title": m.get("title"),
        "scale": m.get("scale"),
        "category": m.get("category"),
        "status": m.get("status"),
        "primary_standard": primary,
        "all_standards": all_std,
        "is_cdm": bool(m.get("isCdm")),
        "sectoral_scopes": sect_scopes,
        "latest_version": latest_ver_no,
        "latest_pdf_url_source": latest_src,
        "latest_pdf_url_cdn": latest_cdn,
        "all_versions": all_versions,
        "tools": tools_unique,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--output", type=Path,
                        default=MANIFEST_DIR / "genvision-methodology-catalog.json")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Stop after this many pages (smoke test).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    api_key = os.environ.get("GENVISION_API_KEY")
    if not api_key:
        logger.error("GENVISION_API_KEY not set. source .env first.")
        return 1

    entries = []
    seen_codes = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client() as client:
        # First call to learn total
        first = fetch_page(client, api_key, page=1, page_size=args.page_size)
        meta = first.get("meta", {}).get("pagination", {})
        total = meta.get("totalCount", 0)
        total_pages = meta.get("totalPages", 0)
        logger.info("Genvision catalog: %d methodologies in %d pages of %d",
                    total, total_pages, args.page_size)

        def consume_page(body):
            for m in body.get("data", []):
                e = normalize_entry(m)
                if not e["code"]:
                    continue
                if e["code"] in seen_codes:
                    continue
                seen_codes.add(e["code"])
                entries.append(e)

        consume_page(first)
        logger.info("  page 1: cumulative %d entries", len(entries))

        last_page = total_pages
        if args.max_pages:
            last_page = min(last_page, args.max_pages)

        for page in range(2, last_page + 1):
            body = fetch_page(client, api_key, page=page, page_size=args.page_size)
            consume_page(body)
            if page % 2 == 0 or page == last_page:
                logger.info("  page %d/%d: cumulative %d entries", page, last_page, len(entries))
            time.sleep(0.1)  # polite

    # Stats
    by_std = {}
    for e in entries:
        s = e.get("primary_standard") or "_UNKNOWN"
        by_std[s] = by_std.get(s, 0) + 1
    with_cdn = sum(1 for e in entries if e.get("latest_pdf_url_cdn"))
    with_tools = sum(1 for e in entries if e.get("tools"))

    out = {
        "version": "0.1",
        "generated_at": _utcnow(),
        "total_count_from_api": total,
        "entries_collected": len(entries),
        "stats": {
            "by_primary_standard": by_std,
            "with_synthesized_cdn_url": with_cdn,
            "with_tool_references": with_tools,
        },
        "entries": entries,
    }
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    logger.info("\n=== Catalog dump complete ===")
    logger.info("Entries: %d", len(entries))
    logger.info("By primary standard: %s", by_std)
    logger.info("With synthesized CDN URL (latest version): %d", with_cdn)
    logger.info("With tool references: %d", with_tools)
    logger.info("Written: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
