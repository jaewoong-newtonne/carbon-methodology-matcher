"""Download methodology PDFs from Genvision CDN per the catalog manifest.

Reads `data/manifests/genvision-methodology-catalog.json` (produced by
`dump_genvision_catalog.py`) and downloads the latest-version PDF for each
entry to `data/corpus/{REGISTRY}/{code}/{code}_{file_id}.pdf`.

The CDN (`assets.genvision.com`) is public S3 + CloudFront — no API key
required, and these downloads do NOT count against any rate limit. We use
asyncio for parallel downloads (default concurrency 8).

Skips files that already exist on disk. Logs failures to a sidecar JSON.

Usage:
    python -m carbonmm.ingest.download_genvision_pdfs
    python -m carbonmm.ingest.download_genvision_pdfs --concurrency 16
    python -m carbonmm.ingest.download_genvision_pdfs --standards CDM,VCS,GS
    python -m carbonmm.ingest.download_genvision_pdfs --all-versions
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import httpx

from .common import CORPUS_DIR, MANIFEST_DIR, setup_logging, _utcnow

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST = MANIFEST_DIR / "genvision-methodology-catalog.json"


@dataclass
class DownloadJob:
    code: str
    registry: str
    version: str | None
    cdn_url: str
    out_path: Path


def build_jobs(entries: list[dict], standards_filter: set[str] | None, all_versions: bool) -> list[DownloadJob]:
    """Build download jobs. Prefers API-provided `documents[].fileUrl` (added by
    enrich_genvision_catalog.py) over the synthesized `latest_pdf_url_cdn` —
    synthesis only works for CDM URL patterns; Verra/ACR/BCR/GS/ACCU use
    different filename conventions that only the API knows.
    """
    jobs = []
    for e in entries:
        code = e["code"]
        registry = e.get("primary_standard") or "_UNKNOWN"
        if standards_filter and registry not in standards_filter:
            continue

        # Source 1: documents[] from API (preferred — exact CDN URLs)
        documents = [d for d in (e.get("documents") or []) if d.get("fileUrl")]

        # Source 2: synthesized latest CDN URL (fallback for entries the enrich
        # pass couldn't fetch — should be rare).
        synth_cdn = e.get("latest_pdf_url_cdn")
        synth_ver = e.get("latest_version")

        version_records = []
        if documents:
            if all_versions:
                version_records = [{"version": d.get("version"),
                                     "fileUrl": d.get("fileUrl")} for d in documents]
            else:
                # Latest = first (API returns documents in chronological order
                # matching versions[] which is desc-by-effectiveFrom in our normalize)
                d0 = documents[0]
                version_records = [{"version": d0.get("version"), "fileUrl": d0["fileUrl"]}]
        elif synth_cdn:
            version_records = [{"version": synth_ver, "fileUrl": synth_cdn}]

        for v in version_records:
            url = v.get("fileUrl")
            if not url:
                continue
            fname = url.rsplit("/", 1)[-1]
            # Strip URL encoding from filename for filesystem safety
            from urllib.parse import unquote
            fname = unquote(fname)
            out = CORPUS_DIR / registry / code / fname
            jobs.append(DownloadJob(
                code=code, registry=registry,
                version=v.get("version"),
                cdn_url=url,
                out_path=out,
            ))
    return jobs


async def download_one(client: httpx.AsyncClient, job: DownloadJob) -> tuple[DownloadJob, str, str | None]:
    """Return (job, status, error). status ∈ {ok, skip_exists, fail_404, fail_other}."""
    try:
        if job.out_path.exists() and job.out_path.stat().st_size > 0:
            return job, "skip_exists", None
        job.out_path.parent.mkdir(parents=True, exist_ok=True)
        r = await client.get(job.cdn_url, timeout=60.0, follow_redirects=True)
        if r.status_code == 200:
            job.out_path.write_bytes(r.content)
            return job, "ok", None
        if r.status_code in (403, 404):
            return job, f"fail_{r.status_code}", None
        return job, "fail_other", f"HTTP {r.status_code}"
    except Exception as e:
        return job, "fail_other", str(e)[:120]


async def run(jobs: list[DownloadJob], concurrency: int) -> dict:
    sem = asyncio.Semaphore(concurrency)
    results = {"ok": 0, "skip_exists": 0, "fail_404": 0, "fail_403": 0, "fail_other": 0}
    failures = []

    async with httpx.AsyncClient(timeout=60.0) as client:
        async def worker(job):
            async with sem:
                return await download_one(client, job)

        tasks = [asyncio.create_task(worker(j)) for j in jobs]
        completed = 0
        for coro in asyncio.as_completed(tasks):
            job, status, err = await coro
            results[status] = results.get(status, 0) + 1
            if status.startswith("fail"):
                failures.append({"code": job.code, "registry": job.registry,
                                 "url": job.cdn_url, "status": status, "error": err})
            completed += 1
            if completed % 50 == 0:
                logger.info("  …%d/%d (ok=%d skip=%d fail=%d)",
                            completed, len(jobs), results["ok"], results["skip_exists"],
                            sum(v for k, v in results.items() if k.startswith("fail")))

    return {"results": results, "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--standards", type=str, default=None,
                        help="Comma-separated standardCodes to download (default: all)")
    parser.add_argument("--all-versions", action="store_true",
                        help="Download every version (default: latest only)")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true",
                        help="List jobs without downloading")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    manifest = json.loads(args.manifest.read_text())
    entries = manifest["entries"]
    std_filter = set(args.standards.split(",")) if args.standards else None
    jobs = build_jobs(entries, std_filter, args.all_versions)

    by_std = {}
    for j in jobs:
        by_std[j.registry] = by_std.get(j.registry, 0) + 1
    logger.info("Planned %d downloads. By registry: %s", len(jobs), by_std)

    if args.dry_run:
        for j in jobs[:5]:
            logger.info("  %s/%s v%s → %s", j.registry, j.code, j.version, j.out_path)
        if len(jobs) > 5:
            logger.info("  ... (%d more)", len(jobs) - 5)
        return 0

    t0 = time.time()
    report = asyncio.run(run(jobs, concurrency=args.concurrency))
    dt = time.time() - t0

    logger.info("Done in %.1fs. Results: %s", dt, report["results"])
    if report["failures"]:
        fail_log = args.manifest.with_name("genvision-download-failures.json")
        fail_log.write_text(json.dumps({
            "generated_at": _utcnow(),
            "total_failures": len(report["failures"]),
            "failures": report["failures"],
        }, indent=2))
        logger.info("Wrote %d failure records to %s", len(report["failures"]), fail_log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
