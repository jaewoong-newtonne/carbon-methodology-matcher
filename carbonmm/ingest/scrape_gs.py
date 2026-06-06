"""Gold Standard methodology scraper — index-driven via xlsx hyperlink list.

Input: `carbonmm/configs/gs_methodology_index.yaml`
       (45 entries extracted from the eligible-CDM-GS-methodologies xlsx,
        sheet 'Gold Standard meths', via openpyxl hyperlink extraction)

Per-methodology detail page (e.g. https://globalgoals.goldstandard.org/{slug}/)
contains:
  - <h1> methodology title
  - <time> release date (datetime attr in ISO 8601)
  - category links (".../documents/methodology/{n}-{name}/")
  - methodology-type links ("framework-methodology" / "activity-module" / "methodology")
  - PDF download link: <a href="https://globalgoals.goldstandard.org/standards/*.pdf">

PDF URL pattern: /standards/{code}_V{version}_{CATEGORY-ABBR}_{slug-with-dashes}.pdf
  e.g. 440_V2.0_CCS_Biomass-Fermentation-with-Carbon-Capture-and-Geologic-Storage.pdf

Reachability: detail pages probed without a WAF block. A soft Cloudflare layer
permits headless browsers.

Usage:
    # 1. Resolve all detail pages → PDF URLs + metadata (per-entry browser fetch)
    python -m carbonmm.ingest.scrape_gs --resolve

    # 2. Then download all PDFs (separate step to allow review of manifest first)
    python -m carbonmm.ingest.scrape_gs --download

    # 3. Both at once
    python -m carbonmm.ingest.scrape_gs --resolve --download
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import yaml

from .common import (
    CORPUS_DIR,
    MANIFEST_DIR,
    MethodologyEntry,
    RateLimiter,
    build_client,
    save_manifest,
    setup_logging,
    _utcnow,
)

logger = logging.getLogger(__name__)

INDEX_YAML = Path(__file__).resolve().parents[1] / "configs" / "gs_methodology_index.yaml"

# Selectors verified against two real detail pages (2026-05-20):
#   - 402 Soil Organic Carbon Framework Methodology
#   - 440 Biomass Fermentation with Carbon Capture and Geologic storage
PDF_SELECTOR = 'a[href*="/standards/"][href$=".pdf"]'
TITLE_SELECTOR = "article h1"
TIME_SELECTOR = "article time, article [datetime]"
CATEGORY_LINK_SELECTOR = 'article a[href*="/documents/methodology/"]'
TYPE_LINK_SELECTOR = 'article a[href*="/documents/framework-methodology/"], article a[href*="/documents/activity-module/"], article a[href*="/documents/methodology/"]:not([href*="/15-"]):not([href*="/16-"]):not([href*="/14-"])'


def load_index() -> list[dict]:
    """Load 45 GS entries from configs/gs_methodology_index.yaml."""
    if not INDEX_YAML.exists():
        raise FileNotFoundError(
            f"Run xlsx extraction first to create {INDEX_YAML} "
            f"(see build_gs_index.py)."
        )
    return yaml.safe_load(INDEX_YAML.read_text())["entries"]


def _parse_pdf_filename(pdf_url: str) -> dict:
    """Parse the standards/*.pdf filename for version + category metadata.

    Format: {code}_V{version}_{CATEGORY}_{slug-with-dashes}.pdf
    """
    name = pdf_url.rsplit("/", 1)[-1]
    # Code may use - or . as sub-separator: 402, 402-4, 402.4 all valid
    m = re.match(r"(\d+(?:[-.]\d+)*)_V([\d.]+)_([A-Z_]+)_(.+?)\.pdf$", name)
    if not m:
        return {"pdf_filename": name}
    # Normalize code: 402.4 → 402-4 so it matches xlsx-derived code_prefix
    code_normalized = m.group(1).replace(".", "-")
    return {
        "pdf_filename": name,
        "code_from_pdf": code_normalized,
        "version": m.group(2),
        "category_abbr": m.group(3),  # AGR / CCS / LUF / RE / etc.
    }


def resolve_detail(page, entry: dict) -> MethodologyEntry:
    """Navigate to a GS methodology detail page and extract metadata."""
    url = entry["url"]
    page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    time.sleep(1.5)  # let dynamic content (sidebar) settle

    # Title
    title_el = page.query_selector(TITLE_SELECTOR)
    title = title_el.inner_text().strip() if title_el else entry.get("name", "")

    # Release date — try datetime attribute first
    date_el = page.query_selector(TIME_SELECTOR)
    release_date = ""
    if date_el:
        release_date = (date_el.get_attribute("datetime") or date_el.inner_text()).strip()

    # PDF link
    pdf_el = page.query_selector(PDF_SELECTOR)
    pdf_url = pdf_el.get_attribute("href") if pdf_el else ""

    # Category + type (collect text of all "/documents/methodology/..." links)
    category_links = page.query_selector_all(CATEGORY_LINK_SELECTOR)
    categories = [el.inner_text().strip().rstrip(",").strip() for el in category_links]

    pdf_meta = _parse_pdf_filename(pdf_url) if pdf_url else {}

    notes_parts = []
    if categories:
        notes_parts.append("categories=" + ", ".join(categories))
    if release_date:
        notes_parts.append(f"released={release_date}")
    if pdf_meta.get("version"):
        notes_parts.append(f"version={pdf_meta['version']}")
    if pdf_meta.get("category_abbr"):
        notes_parts.append(f"abbr={pdf_meta['category_abbr']}")

    return MethodologyEntry(
        registry="GS",
        code=entry.get("code_prefix") or pdf_meta.get("code_from_pdf", ""),
        title=title,
        detail_url=url,
        pdf_url=pdf_url,
        fetched_at=_utcnow(),
        notes="; ".join(notes_parts),
    )


def resolve_all(*, max_n: int | None = None) -> list[MethodologyEntry]:
    """Visit each GS detail page via Playwright and extract PDF URLs + metadata."""
    from .playwright_fetcher import browser_context

    index = load_index()
    if max_n:
        index = index[:max_n]
    logger.info("Resolving %d GS methodology detail pages…", len(index))

    entries: list[MethodologyEntry] = []
    with browser_context() as ctx:
        page = ctx.new_page()
        rl = RateLimiter(requests_per_second=0.5)  # 2s between pages — polite
        for i, idx_entry in enumerate(index, 1):
            rl.wait()
            try:
                e = resolve_detail(page, idx_entry)
                entries.append(e)
                logger.info(
                    "  [%2d/%d] code=%s | %s | pdf=%s",
                    i, len(index),
                    e.code or "?",
                    e.title[:55],
                    "OK" if e.pdf_url else "MISSING",
                )
            except Exception as ex:
                logger.error("  [%2d/%d] %s → %s", i, len(index), idx_entry["url"], ex)
                entries.append(MethodologyEntry(
                    registry="GS",
                    code=idx_entry.get("code_prefix", ""),
                    title=idx_entry.get("name", ""),
                    detail_url=idx_entry["url"],
                    fetched_at=_utcnow(),
                    notes=f"resolve-error: {ex}",
                ))
    return entries


def download_pdfs(entries: list[MethodologyEntry], *, max_n: int | None = None) -> list[MethodologyEntry]:
    """Download each entry's PDF to data/corpus/GS/{code}/{filename}.pdf."""
    import hashlib

    target_entries = [e for e in entries if e.pdf_url][:max_n] if max_n else [e for e in entries if e.pdf_url]
    logger.info("Downloading %d PDFs…", len(target_entries))
    rl = RateLimiter(requests_per_second=1.0)
    with build_client(timeout=60.0) as client:
        for i, e in enumerate(target_entries, 1):
            rl.wait()
            out_dir = CORPUS_DIR / "GS" / (e.code or "uncoded")
            out_dir.mkdir(parents=True, exist_ok=True)
            filename = e.pdf_url.rsplit("/", 1)[-1]
            out_path = out_dir / filename
            if out_path.exists() and out_path.stat().st_size > 1024:
                logger.info("  [%2d/%d] cached: %s", i, len(target_entries), out_path.name)
                e.pdf_local_path = str(out_path)
                continue
            try:
                r = client.get(e.pdf_url)
                r.raise_for_status()
                if r.headers.get("content-type", "").startswith("application/pdf") or len(r.content) > 10_000:
                    out_path.write_bytes(r.content)
                    e.pdf_local_path = str(out_path)
                    e.pdf_bytes = len(r.content)
                    e.pdf_sha256 = hashlib.sha256(r.content).hexdigest()
                    logger.info(
                        "  [%2d/%d] saved: %s (%d KB)",
                        i, len(target_entries), out_path.name, e.pdf_bytes // 1024,
                    )
                else:
                    logger.warning("  [%2d/%d] non-PDF or tiny response: %s", i, len(target_entries), e.pdf_url)
            except Exception as ex:
                logger.error("  [%2d/%d] %s → %s", i, len(target_entries), e.pdf_url, ex)
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolve", action="store_true", help="Visit detail pages, extract PDF URLs + metadata")
    parser.add_argument("--download", action="store_true", help="Download PDFs from manifest")
    parser.add_argument("--max-n", type=int, default=None, help="Cap entries (for smoke tests)")
    parser.add_argument(
        "--manifest", type=Path,
        default=MANIFEST_DIR / "gs-all.yaml",
        help="Manifest path (default: data/manifests/gs-all.yaml)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    if not (args.resolve or args.download):
        parser.error("Pass --resolve and/or --download")

    if args.resolve:
        entries = resolve_all(max_n=args.max_n)
        save_manifest(entries, args.manifest)
        logger.info("Manifest written: %s", args.manifest)
    else:
        # Load existing manifest
        from .common import load_manifest
        entries = load_manifest(args.manifest)
        logger.info("Loaded %d entries from %s", len(entries), args.manifest)

    if args.download:
        entries = download_pdfs(entries, max_n=args.max_n)
        save_manifest(entries, args.manifest)
        logger.info("Manifest updated with download metadata: %s", args.manifest)

    n_with_pdf = sum(1 for e in entries if e.pdf_url)
    n_downloaded = sum(1 for e in entries if e.pdf_local_path)
    logger.info(
        "Done. %d entries | %d with PDF URL | %d downloaded.",
        len(entries), n_with_pdf, n_downloaded,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
