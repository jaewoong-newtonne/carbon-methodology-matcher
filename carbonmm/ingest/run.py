"""Methodology corpus ingestion orchestrator.

Runs all 3 registry scrapers in sequence (CDM functional; Verra/GS skeleton),
consolidates manifests, and reports counts.

Usage:
    python -m carbonmm.ingest.run --dry-run
    python -m carbonmm.ingest.run --registry all --output data/manifests/
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import scrape_cdm, scrape_gs, scrape_verra
from .common import MANIFEST_DIR, save_manifest, setup_logging

logger = logging.getLogger(__name__)

REGISTRIES = {
    "cdm": scrape_cdm,
    "verra": scrape_verra,
    "gs": scrape_gs,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        choices=list(REGISTRIES) + ["all"],
        default="all",
        help="Which registry to scrape (default: all)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    targets = list(REGISTRIES) if args.registry == "all" else [args.registry]
    total = 0
    for reg in targets:
        logger.info("=== Scraping registry: %s ===", reg.upper())
        try:
            if reg == "cdm":
                # CDM has 4 sub-categories; scrape all
                from .scrape_cdm import scrape_category, CATEGORIES
                entries = []
                for cat in CATEGORIES:
                    entries.extend(scrape_category(cat, dry_run=args.dry_run))
                if not args.dry_run:
                    save_manifest(entries, MANIFEST_DIR / "cdm-all.yaml")
            elif reg == "verra":
                entries = scrape_verra.scrape_verra(dry_run=args.dry_run)
                if not args.dry_run:
                    save_manifest(entries, MANIFEST_DIR / "verra-all.yaml")
            elif reg == "gs":
                entries = scrape_gs.scrape_gs(dry_run=args.dry_run)
                if not args.dry_run:
                    save_manifest(entries, MANIFEST_DIR / "gs-all.yaml")
            total += len(entries)
            logger.info("  → %d entries from %s", len(entries), reg.upper())
        except NotImplementedError as e:
            logger.warning("  → SKIPPED (%s)", e)

    logger.info("Total entries across all registries: %d", total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
