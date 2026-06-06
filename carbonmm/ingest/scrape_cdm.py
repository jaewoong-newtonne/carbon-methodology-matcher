"""UNFCCC CDM methodology scraper.

Discovers (code, hash, title, detail_url) tuples from the four CDM index pages:
    - Large-scale baseline+monitoring (AM, ACM)
    - Small-scale (AMS)
    - A/R large-scale (AR-AM)
    - A/R small-scale (AR-AMS)

Verified HTML structure (2026-05-19):
    Each methodology row in the index page is a <tr> containing:
        <th nowrap="nowrap">{CODE}</th>
        <td>
            <a href="https://cdm.unfccc.int/methodologies/DB/{HASH}">{TITLE} --- Version X.Y</a>
        </td>
        <td>
            <a href="/methodologies/documentation/meth_booklet.pdf#{CODE}">
                <img src="/CommonImages/pdf.gif"/> Ref {CODE}
            </a>
        </td>

The booklet PDF (`meth_booklet.pdf`) is a single document covering all CDM
methodologies — easier to download once than per-methodology. We capture both
the booklet ref and the detail URL so the granularity can be chosen later.

Usage:
    python -m carbonmm.ingest.scrape_cdm --dry-run
    python -m carbonmm.ingest.scrape_cdm --category PA --output data/manifests/cdm-PA.yaml

WAF NOTE:
    UNFCCC sits behind an Incapsula/Imperva WAF that fingerprints TLS clients
    (JA3/JA4). httpx and stock curl both get returned an 846-byte block page
    (HTTP 200 but no methodology content). The parse_index() regex below IS
    correct (verified against a successful 133 KB HTML snapshot) — only the
    fetcher needs swapping. Three options:

      1. Playwright (chromium headless) — most reliable, already used by Verra
         scraper. Pattern: a separate crawler service.
      2. curl_cffi (pip install curl_cffi) — drop-in httpx-compat with browser
         TLS impersonation (impersonate="chrome120"). Fastest fix.
      3. Cached snapshot — download the index pages once via real browser and
         feed parse_index() the saved HTML.

    The httpx code path is left in place for documentation; swap in one of the
    options above for live crawling.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

from .common import (
    MANIFEST_DIR,
    MethodologyEntry,
    RateLimiter,
    build_client,
    save_manifest,
    setup_logging,
    _utcnow,
)

logger = logging.getLogger(__name__)

BASE = "https://cdm.unfccc.int"

# Category → (URL path, expected code prefix regex)
CATEGORIES = {
    "PA": ("/methodologies/PAmethodologies/approved", r"^(ACM|AM)\d{4}$"),
    "SSC": ("/methodologies/SSCmethodologies/approved", r"^AMS-[IVX]+\.[A-Z]{1,2}$"),
    "AR": ("/methodologies/ARmethodologies/approved", r"^AR-AM\d{4}$"),
    "SSCAR": ("/methodologies/SSCARmethodologies/approved", r"^AR-AMS-[IVX]+\.[A-Z]{1,2}$"),
}


# Row pattern — matches the <tr> structure verified on 2026-05-19
# <th nowrap="nowrap">CODE</th> ... <a href="DETAIL_URL">TITLE</a> ... Ref CODE
ROW_PATTERN = re.compile(
    r'<th[^>]*>(?P<code>[A-Z][A-Z0-9\-\.]+)</th>'
    r'.*?'
    r'<a\s+href="(?P<detail_url>https?://cdm\.unfccc\.int/methodologies/DB/[A-Z0-9]+)">'
    r'(?P<title>[^<]+?)</a>',
    re.DOTALL,
)


def parse_index(html: str, code_pattern: re.Pattern) -> list[MethodologyEntry]:
    """Extract methodology entries from a CDM index page."""
    entries = []
    for m in ROW_PATTERN.finditer(html):
        code = m.group("code").strip()
        if not code_pattern.match(code):
            continue
        title = re.sub(r"\s+", " ", m.group("title")).strip()
        title = re.sub(r"\s*---\s*Version\s*[\d\.]+\s*$", "", title, flags=re.I).strip()
        entries.append(
            MethodologyEntry(
                registry="CDM",
                code=code,
                title=title,
                detail_url=m.group("detail_url"),
                fetched_at=_utcnow(),
                notes=f"booklet:{BASE}/methodologies/documentation/meth_booklet.pdf#{code}",
            )
        )
    return entries


def scrape_category(
    category: str,
    *,
    dry_run: bool = False,
    html_snapshot: Path | None = None,
    engine: str = "httpx",
) -> list[MethodologyEntry]:
    """Scrape one CDM category index page.

    `html_snapshot`: optional path to a previously-saved index HTML. Used to
    bypass WAF during dev (see the module docstring WAF NOTE).
    `engine`: "httpx" (default, blocked by Incapsula WAF) | "playwright"
    """
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category: {category} (valid: {list(CATEGORIES)})")
    path, code_re = CATEGORIES[category]
    code_pattern = re.compile(code_re)
    url = BASE + path

    if html_snapshot:
        logger.info("Reading snapshot %s (category=%s)", html_snapshot, category)
        html = Path(html_snapshot).read_text()
    elif engine == "playwright":
        from .playwright_fetcher import fetch_html
        # Wait for the methodology table — selector based on observed HTML
        html = fetch_html(url, wait_selector="th[nowrap]", wait_seconds=2)
    else:
        logger.info("Fetching %s via httpx (may hit WAF)", url)
        rl = RateLimiter(requests_per_second=1.0)
        with build_client() as client:
            rl.wait()
            r = client.get(url)
            r.raise_for_status()
            html = r.text
        if len(html) < 5000:
            logger.warning(
                "Response is %d bytes — WAF block likely. Re-run with --engine playwright.",
                len(html),
            )

    entries = parse_index(html, code_pattern)
    logger.info("  category=%s → %d methodology entries", category, len(entries))
    if dry_run:
        for e in entries[:3]:
            logger.info("  sample: %s | %s | %s", e.code, e.title[:60], e.detail_url)
        if len(entries) > 3:
            logger.info("  ... (%d more)", len(entries) - 3)
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--category",
        choices=list(CATEGORIES) + ["all"],
        default="all",
        help="Which CDM category to scrape (default: all 4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and parse but do not save manifest",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Manifest output path (default: data/manifests/cdm-{category}.yaml)",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Path to a saved index HTML snapshot (bypass WAF during dev)",
    )
    parser.add_argument(
        "--engine",
        choices=["httpx", "playwright"],
        default="httpx",
        help="HTTP fetcher (default: httpx, blocked by WAF; use playwright)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    cats = list(CATEGORIES) if args.category == "all" else [args.category]
    all_entries: list[MethodologyEntry] = []
    for cat in cats:
        try:
            all_entries.extend(scrape_category(
                cat, dry_run=args.dry_run, html_snapshot=args.snapshot, engine=args.engine,
            ))
        except Exception as e:
            logger.error("category=%s failed: %s", cat, e)

    logger.info("Total CDM methodology entries: %d", len(all_entries))

    if args.dry_run:
        return 0

    out = args.output or (MANIFEST_DIR / f"cdm-{args.category}.yaml")
    save_manifest(all_entries, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
