"""Verra VCS methodology scraper.

Source URL: https://verra.org/methodologies-main/  (redirects to the program
methodology overview). The actual methodology table is JavaScript-rendered
behind Cloudflare, so the reliable harvest path is:

    1. Manual snapshot: download the rendered HTML in a real Chrome browser
       (right-click → Save Page As → "Webpage, Complete") and save to
       data/snapshots/verra-methodologies-YYYYMMDD.html
    2. Run: python -m carbonmm.ingest.scrape_verra
            --snapshot data/snapshots/verra-methodologies-YYYYMMDD.html

    Playwright is offered as a best-effort secondary path (--engine playwright)
    but Cloudflare bot-checks may fail.

Parser status:
    - Pattern below is the working hypothesis based on Verra's WordPress-based
      catalog; expect to refine after first snapshot is available.
    - Verra codes follow patterns: VM\\d{4} (methodologies), VMD\\d{4} (modules),
      and increasingly named methodologies without numeric codes.
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
    build_client,
    save_manifest,
    setup_logging,
    _utcnow,
)

logger = logging.getLogger(__name__)

VERRA_METHODOLOGY_URL = "https://verra.org/methodologies-main/"

# Hypothesis pattern — refine after first real HTML snapshot. Looks for
# {VM/VMD/VT}#### codes nearby a methodology name/link.
ROW_PATTERN_HYPOTHESIS = re.compile(
    r'(?P<code>V(?:M|MD|T)\d{4})'
    r'.{1,500}?'
    r'<a[^>]+href="(?P<detail_url>https?://[^"]+)"[^>]*>(?P<title>[^<]+)</a>',
    re.DOTALL,
)


def parse_index(html: str) -> list[MethodologyEntry]:
    """Extract Verra methodology entries from index HTML."""
    entries = []
    seen = set()
    for m in ROW_PATTERN_HYPOTHESIS.finditer(html):
        code = m.group("code").strip()
        if code in seen:
            continue
        seen.add(code)
        title = re.sub(r"\s+", " ", m.group("title")).strip()
        entries.append(
            MethodologyEntry(
                registry="Verra",
                code=code,
                title=title,
                detail_url=m.group("detail_url"),
                fetched_at=_utcnow(),
            )
        )
    return entries


def scrape(
    *,
    dry_run: bool = False,
    html_snapshot: Path | None = None,
    engine: str = "snapshot",
) -> list[MethodologyEntry]:
    """Scrape the Verra methodology index. Defaults to snapshot mode."""
    if html_snapshot:
        logger.info("Reading snapshot %s", html_snapshot)
        html = Path(html_snapshot).read_text()
    elif engine == "playwright":
        from .playwright_fetcher import fetch_html
        # Verra methodology list might have selector `.methodology-row` or table.tbody — refine after real fetch
        html = fetch_html(
            VERRA_METHODOLOGY_URL,
            wait_selector=None,  # TBD: discover correct selector
            wait_seconds=5.0,
        )
    elif engine == "httpx":
        with build_client() as client:
            r = client.get(VERRA_METHODOLOGY_URL)
            r.raise_for_status()
            html = r.text
        if len(html) < 5000:
            logger.warning("Tiny response (%d bytes) — likely WAF block.", len(html))
    else:
        raise ValueError(f"Unknown engine: {engine}")

    entries = parse_index(html)
    logger.info("Verra: %d methodology entries parsed", len(entries))
    if dry_run:
        for e in entries[:3]:
            logger.info("  sample: %s | %s | %s", e.code, e.title[:60], e.detail_url[:80])
        if len(entries) > 3:
            logger.info("  ... (%d more)", len(entries) - 3)
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--snapshot", type=Path, default=None,
        help="HTML snapshot path (preferred — bypass Cloudflare)",
    )
    parser.add_argument(
        "--engine", choices=["snapshot", "httpx", "playwright"], default="snapshot",
        help="Fetch engine (default: snapshot — most reliable)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Manifest output path (default: data/manifests/verra-all.yaml)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.engine == "snapshot" and not args.snapshot:
        logger.error(
            "engine=snapshot requires --snapshot path. "
            "Download the methodology index from %s in a real browser first.",
            VERRA_METHODOLOGY_URL,
        )
        return 1

    entries = scrape(dry_run=args.dry_run, html_snapshot=args.snapshot, engine=args.engine)
    if args.dry_run:
        return 0

    out = args.output or (MANIFEST_DIR / "verra-all.yaml")
    save_manifest(entries, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
