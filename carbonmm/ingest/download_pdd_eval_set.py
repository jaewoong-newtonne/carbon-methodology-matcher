"""Download the ICDM 2026 PDD evaluation set per the manifest from
fetch_genvision_pdds.py.

The eval set is GS 510 + VCS 990 = 1,500 PDDs served from the Genvision CDN
(public S3+CloudFront), so the PDF downloads themselves are free; only the
manifest fetch goes through the metered API.

Output path: data/eval-pdds/{registry}/{globalId}/{filename}.pdf

This is intentionally separate from data/corpus/ (methodology corpus) so the
two artefacts don't collide. The eval-pdds tree is made reachable by the layout
service the same way as the methodology corpus.

Usage:
    python -m carbonmm.ingest.download_pdd_eval_set
    python -m carbonmm.ingest.download_pdd_eval_set --concurrency 16
    python -m carbonmm.ingest.download_pdd_eval_set --supporting-docs
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
from urllib.parse import unquote

import httpx

from .common import DATA_ROOT, MANIFEST_DIR, setup_logging, _utcnow

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST = MANIFEST_DIR / "genvision-pdd-eval-set.json"
PDD_ROOT = DATA_ROOT / "eval-pdds"


@dataclass
class Job:
    global_id: str
    registry: str
    url: str
    out_path: Path
    is_primary: bool


def build_jobs(entries: list[dict], include_supporting: bool) -> list[Job]:
    jobs = []
    for e in entries:
        gid = e["globalId"]
        reg = e["registry"]
        dest_dir = PDD_ROOT / reg / gid
        url = e.get("primary_pdd_url")
        if url:
            fname = unquote(url.rsplit("/", 1)[-1]) or f"{gid}.pdf"
            jobs.append(Job(gid, reg, url, dest_dir / fname, is_primary=True))
        if include_supporting:
            for url in e.get("supporting_doc_urls") or []:
                fname = unquote(url.rsplit("/", 1)[-1]) or "supp.pdf"
                jobs.append(Job(gid, reg, url, dest_dir / "supporting" / fname, is_primary=False))
    return jobs


async def download_one(client: httpx.AsyncClient, job: Job) -> tuple[Job, str, str | None]:
    try:
        if job.out_path.exists() and job.out_path.stat().st_size > 0:
            return job, "skip_exists", None
        job.out_path.parent.mkdir(parents=True, exist_ok=True)
        r = await client.get(job.url, timeout=120.0, follow_redirects=True)
        if r.status_code == 200:
            job.out_path.write_bytes(r.content)
            return job, "ok", None
        if r.status_code in (403, 404):
            return job, f"fail_{r.status_code}", None
        return job, "fail_other", f"HTTP {r.status_code}"
    except Exception as e:
        return job, "fail_other", str(e)[:120]


async def run(jobs: list[Job], concurrency: int) -> dict:
    sem = asyncio.Semaphore(concurrency)
    results = {"ok": 0, "skip_exists": 0, "fail_403": 0, "fail_404": 0, "fail_other": 0}
    failures = []

    async with httpx.AsyncClient(timeout=120.0) as client:
        async def worker(j):
            async with sem:
                return await download_one(client, j)

        tasks = [asyncio.create_task(worker(j)) for j in jobs]
        completed = 0
        for coro in asyncio.as_completed(tasks):
            j, status, err = await coro
            results[status] = results.get(status, 0) + 1
            if status.startswith("fail"):
                failures.append({"globalId": j.global_id, "registry": j.registry,
                                 "url": j.url, "status": status, "error": err})
            completed += 1
            if completed % 100 == 0 or completed == len(jobs):
                fails = sum(v for k, v in results.items() if k.startswith("fail"))
                logger.info("  …%d/%d ok=%d skip=%d fail=%d",
                            completed, len(jobs), results["ok"], results["skip_exists"], fails)

    return {"results": results, "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--supporting-docs", action="store_true",
                        help="Also download supporting_doc_urls (validation/monitoring reports).")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    manifest = json.loads(args.manifest.read_text())
    entries = manifest["entries"]
    jobs = build_jobs(entries, include_supporting=args.supporting_docs)

    by_reg = {}
    for j in jobs:
        by_reg[j.registry] = by_reg.get(j.registry, 0) + 1
    logger.info("Planned %d downloads (supporting_docs=%s). By registry: %s",
                len(jobs), args.supporting_docs, by_reg)

    if args.dry_run:
        for j in jobs[:5]:
            logger.info("  %s/%s primary=%s → %s", j.registry, j.global_id, j.is_primary, j.out_path)
        if len(jobs) > 5:
            logger.info("  ... (%d more)", len(jobs) - 5)
        return 0

    t0 = time.time()
    report = asyncio.run(run(jobs, args.concurrency))
    dt = time.time() - t0
    logger.info("Done in %.1fs. Results: %s", dt, report["results"])
    if report["failures"]:
        fail_log = args.manifest.with_name("genvision-pdd-download-failures.json")
        fail_log.write_text(json.dumps({
            "generated_at": _utcnow(),
            "total_failures": len(report["failures"]),
            "failures": report["failures"],
        }, indent=2))
        logger.info("Wrote %d failure records to %s", len(report["failures"]), fail_log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
