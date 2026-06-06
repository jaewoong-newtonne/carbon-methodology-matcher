"""Shared utilities for methodology corpus scrapers.

Provides:
    - polite HTTP client (httpx + tenacity backoff, 1 req/s default rate)
    - canonical output paths (local dev / shared corpus)
    - manifest read/write
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ─── Paths ─────────────────────────────────────────────────────────────────
# Local development output lives under the package's data/ directory. A shared
# corpus location can be selected with --remote (e.g. for a bulk harvest host),
# overridable via the SHARED_CORPUS_DIR environment variable.
ICDM_ROOT = Path(__file__).resolve().parents[1]              # carbonmm/
DATA_ROOT = ICDM_ROOT / "data"                                # gitignored
MANIFEST_DIR = DATA_ROOT / "manifests"
CORPUS_DIR = DATA_ROOT / "corpus"                             # PDFs land here locally

# Shared corpus location, used when --remote is passed.
SHARED_CORPUS_DIR = Path(os.environ.get("SHARED_CORPUS_DIR", "shared-corpus/methodology-corpus"))


def output_dirs(registry: str, code: str, remote: bool = False) -> tuple[Path, Path]:
    """Return (manifest_dir, pdf_dir) for a given (registry, code) tuple.

    If `remote` is True, paths target the shared corpus folder. Otherwise,
    the local dev path under carbonmm/data/.
    """
    base = SHARED_CORPUS_DIR if remote else CORPUS_DIR
    pdf_dir = base / registry / code
    manifest = MANIFEST_DIR if not remote else SHARED_CORPUS_DIR / "_manifests"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    manifest.mkdir(parents=True, exist_ok=True)
    return manifest, pdf_dir


def guidance_dirs(registry: str, doc_id: str, remote: bool = False) -> tuple[Path, Path]:
    """Like output_dirs() but for registry framework rulebooks.

    Namespaces guidance under `{REGISTRY}/_guidance/{doc_id}/` (leading-underscore
    matches the existing `_manifests` convention; can never collide with a
    methodology `code`). Returns (manifest_dir, pdf_dir).
    """
    return output_dirs(registry, f"_guidance/{doc_id}", remote=remote)


# ─── HTTP ──────────────────────────────────────────────────────────────────
def build_client(timeout: float = 30.0, user_agent: str | None = None):
    """Return an httpx.Client with sensible defaults for registry scraping.

    Caller is responsible for closing (use as context manager).
    """
    import httpx  # lazy import so the module imports without httpx in env

    # UNFCCC firewall rejects User-Agents containing the substring "bot",
    # so we present as a regular browser. Contact info goes in From: header
    # per RFC 7231 §5.5.1.
    ua = user_agent or (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    return httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={
            "User-Agent": ua,
            "Accept-Language": "en-US,en;q=0.9",
            "From": "carbon-methodology-research@example.org",  # polite-research contact (RFC 7231 §5.5.1)
        },
    )


class RateLimiter:
    """Simple monotonic-clock-based polite-poll limiter.

    Usage:
        rl = RateLimiter(requests_per_second=1.0)
        for url in urls:
            rl.wait()
            response = client.get(url)
    """

    def __init__(self, requests_per_second: float = 1.0):
        self.min_interval = 1.0 / max(requests_per_second, 1e-6)
        self._last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        delta = now - self._last
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last = time.monotonic()


# ─── Manifest dataclass ────────────────────────────────────────────────────
@dataclass
class MethodologyEntry:
    """One row of the per-registry manifest."""

    registry: str                  # "CDM" | "Verra" | "GS"
    code: str                      # canonical methodology code
    title: str = ""                # human-readable name
    detail_url: str = ""           # registry detail page URL
    pdf_url: str = ""              # direct PDF URL (may be empty until resolved)
    pdf_local_path: str = ""       # filesystem path after download
    pdf_sha256: str = ""           # hash for integrity check
    pdf_bytes: int = 0
    fetched_at: str = ""           # ISO 8601 UTC
    notes: str = ""
    # ── Guidance-harvest fields (optional; "" for methodology rows) ──────────
    # Appended with defaults so existing manifests round-trip unchanged via
    # load_manifest()'s MethodologyEntry(**e). For guidance rows the `code`
    # field holds the doc_id (e.g. "GS-101"), so output_dirs()/skip-exists work
    # without change. See seeds/typeql/schema-patch-guidance-corpus.tql.
    doc_type: str = ""             # "core" | "guideline" | "optional" | "procedure" | "definitions"
    version: str = ""              # e.g. "2.1" (GS) | "25" (CDM-EB66-A23-GUID)
    authority: str = ""            # issuing body: "Gold Standard" | "UNFCCC CDM EB" | "Verra"
    effective_date: str = ""       # ISO date or version-effective string, if catalog exposes it
    review_status: str = ""        # "" | "pending" | "approved" | "rejected" (discover→review gate)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def save_manifest(entries: list[MethodologyEntry], path: Path) -> None:
    """Write a list of MethodologyEntry to YAML."""
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": "0.1",
        "generated_at": _utcnow(),
        "count": len(entries),
        "entries": [e.as_dict() for e in entries],
    }
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
    logger.info("Wrote manifest: %s (%d entries)", path, len(entries))


def load_manifest(path: Path) -> list[MethodologyEntry]:
    import yaml

    with open(path) as f:
        doc = yaml.safe_load(f)
    return [MethodologyEntry(**e) for e in doc.get("entries", [])]


def _utcnow() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ─── Logging setup ─────────────────────────────────────────────────────────
def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)-25s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
