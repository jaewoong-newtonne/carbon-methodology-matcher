"""Enrich the Genvision catalog with API-accurate documents[].fileUrl.

The listing endpoint (`/methodologies?pagination[pageSize]=100`) returns
versions[].pdfUrl (UNFCCC/Verra source URLs) but NOT documents[].fileUrl.
The single-doc endpoint (`/methodologies?filters[code][$eq]=XXX&pagination[pageSize]=1`)
DOES return documents[] with explicit fileUrl pointing at the CDN mirror.

Our first download pass (synthesizing CDN URLs from versions[].pdfUrl) only
worked for CDM. For Verra/ACR/BCR/GS/ACCU/etc. the synthesis pattern was
wrong (.pdf.pdf double extension, query-string-based storage tokens, etc.)
and 265/527 downloads got 403.

This enrich pass does one async API call per methodology code to fetch the
authoritative documents[].fileUrl list. Concurrency 8, ~3-5 minutes for 594.

Input:  data/manifests/genvision-methodology-catalog.json
Output: data/manifests/genvision-methodology-catalog.json (same file; in-place)
        adds `documents` field per entry with [{version, fileUrl, fileSize}]

Usage:
    python -m carbonmm.ingest.enrich_genvision_catalog
    python -m carbonmm.ingest.enrich_genvision_catalog --concurrency 16
    python -m carbonmm.ingest.enrich_genvision_catalog --only-missing
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import httpx

from .common import MANIFEST_DIR, _utcnow, setup_logging

logger = logging.getLogger(__name__)

API_BASE = "https://api.registry.genvision.com/v1"
DEFAULT_MANIFEST = MANIFEST_DIR / "genvision-methodology-catalog.json"


async def fetch_one(client: httpx.AsyncClient, api_key: str, code: str) -> tuple[str, dict | None]:
    from urllib.parse import quote
    url = (
        f"{API_BASE}/methodologies?"
        f"filters%5Bcode%5D%5B%24eq%5D={quote(code, safe='')}"
        f"&pagination%5BpageSize%5D=1"
    )
    try:
        r = await client.get(url, headers={"x-api-key": api_key}, timeout=30.0)
        r.raise_for_status()
        body = r.json()
        items = body.get("data", [])
        if not items:
            return code, None
        return code, items[0]
    except Exception as e:
        logger.error("fetch failed for %s: %s", code, str(e)[:120])
        return code, None


def extract_documents(item: dict) -> list[dict]:
    """Pull documents[] from a single-doc API response. Returns list of
    {version, fileUrl, source_url, fileSize} dicts ordered as in API."""
    out = []
    # Build version→document map by matching url (source) to documents[].url
    docs = item.get("documents") or []
    versions = item.get("versions") or []

    # documents[].url is the UNFCCC/Verra source; map back to version
    src_to_version = {}
    for v in versions:
        if v.get("pdfUrl"):
            src_to_version[v["pdfUrl"]] = v.get("versionNumber")

    for d in docs:
        out.append({
            "version": src_to_version.get(d.get("url")),
            "source_url": d.get("url"),
            "fileUrl": d.get("fileUrl"),
            "fileSize": d.get("fileSizeBytes"),
        })
    return out


async def run(catalog: dict, api_key: str, concurrency: int, only_missing: bool) -> dict:
    sem = asyncio.Semaphore(concurrency)
    entries = catalog["entries"]
    targets = entries if not only_missing else [
        e for e in entries
        if not e.get("documents") or all(not (d.get("fileUrl") or "").strip()
                                          for d in e.get("documents", []))
    ]
    logger.info("Enriching %d / %d entries (only_missing=%s)",
                len(targets), len(entries), only_missing)

    code_to_entry = {e["code"]: e for e in entries}
    completed = 0
    ok = 0
    no_data = 0

    async with httpx.AsyncClient() as client:
        async def worker(code):
            async with sem:
                return await fetch_one(client, api_key, code)

        tasks = [asyncio.create_task(worker(e["code"])) for e in targets]
        for coro in asyncio.as_completed(tasks):
            code, item = await coro
            if item is None:
                no_data += 1
            else:
                docs = extract_documents(item)
                code_to_entry[code]["documents"] = docs
                if docs:
                    ok += 1
                else:
                    no_data += 1
            completed += 1
            if completed % 100 == 0:
                logger.info("  …%d/%d (ok=%d no_data=%d)",
                            completed, len(targets), ok, no_data)

    return {"enriched": ok, "no_data": no_data}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--only-missing", action="store_true",
                        help="Only enrich entries that lack documents[].fileUrl")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    api_key = os.environ.get("GENVISION_API_KEY")
    if not api_key:
        logger.error("GENVISION_API_KEY not set. source .env first.")
        return 1

    catalog = json.loads(args.manifest.read_text())
    t0 = time.time()
    report = asyncio.run(run(catalog, api_key, args.concurrency, args.only_missing))
    dt = time.time() - t0

    catalog["enriched_at"] = _utcnow()
    catalog["enrich_stats"] = report
    args.manifest.write_text(json.dumps(catalog, indent=2, ensure_ascii=False))
    logger.info("Done in %.1fs. Wrote enriched manifest. Stats: %s", dt, report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
