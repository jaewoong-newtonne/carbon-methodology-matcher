"""Build a PDD evaluation-set manifest from Genvision /projects + documents[].

Evaluation-set construction policy:
  - Filter: globalId startsWith {GS|VCS} AND creditingPeriodStartDate >= 2020-01-01 AND totalCreditsIssued > 0
  - CDM excluded (Verra+GS PDDs carry CDM methodology code labels)
  - Proportional sample to a target N (default 1500):
      GS  = N * 799/(799+1552) ≈ 509
      VCS = N * 1552/(799+1552) ≈ 991

The fetched listing response includes documents[] inline. We identify the PDD
file from documents[] by filename pattern (case-insensitive):
  - "PDD", "PD_", "Project Description", "Project Design Document"
  - fallback: largest PDF in documents[] that has isLatest=true

Output manifest schema:
{
  "version": "0.1",
  "generated_at": "...",
  "filter": {...},
  "sample_target": 1500,
  "stats": {"GS_pool": 799, "VCS_pool": 1552, ...},
  "entries": [
    {
      "globalId": "VCS10",
      "registry": "VCS",
      "name": "BAESA Project",
      "methodologies": ["ACM0002"],
      "creditingPeriodStartDate": "2020-04-06",
      "totalCreditsIssued": 6726799,
      "country": "...",
      "primary_pdd_url": "https://assets.genvision.com/projects/VCS/VCS10/BAESA VCS PD_...pdf",
      "primary_pdd_filename": "BAESA VCS PD_CP Renewal_V03.pdf",
      "primary_pdd_size": 1059021,
      "primary_pdd_version": 3,
      "supporting_doc_urls": [ ... ]  # all other docs (validation/monitoring etc)
    },
    ...
  ]
}

Usage:
    python -m carbonmm.ingest.fetch_genvision_pdds
    python -m carbonmm.ingest.fetch_genvision_pdds --target 1500
    python -m carbonmm.ingest.fetch_genvision_pdds --gs-sample 509 --vcs-sample 991
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from urllib.parse import quote

import httpx

from .common import MANIFEST_DIR, _utcnow, setup_logging

logger = logging.getLogger(__name__)

API_BASE = "https://api.registry.genvision.com/v1"
DEFAULT_OUT = MANIFEST_DIR / "genvision-pdd-eval-set.json"

# PDD identification — patterns based on 2026-05-21 sample inspection of
# GS11564 / GS11568 / VCS10 / VCS1001 documents[] arrays.
#
# Key insight: NEVER fall back to "largest non-excluded PDF" — large docs are
# overwhelmingly Issuance/Registration Representations (legal docs), FVRs,
# or Monitoring Reports. Better to SKIP a project entirely than to mis-label
# a non-PDD as the PDD (would corrupt redaction target downstream).

# Word boundary that treats `_` and `-` as separators (Python's \b treats `_`
# as a word char, so `_PDD_` is not a `\bPDD\b` match. Use lookarounds.)
_NL = r"(?<![A-Za-z])"   # left boundary — no preceding letter
_NR = r"(?![A-Za-z])"    # right boundary — no following letter

PDD_FILENAME_PATTERNS = [
    # Generic PDD acronym (most reliable; matches `_PDD_`, ` PDD `, `PDD-`, etc.)
    re.compile(_NL + r"PDD" + _NR, re.IGNORECASE),
    # Explicit document title (most authoritative)
    re.compile(r"Project[\s_\-]*Activity[\s_\-]*Design[\s_\-]*Document", re.IGNORECASE),
    re.compile(r"Project[\s_\-]*Design[\s_\-]*Document", re.IGNORECASE),
    re.compile(r"Project[\s_\-]*Description", re.IGNORECASE),
    # VCS native — `VCS PD` / `VCS_PD` / `VCS-PD`
    re.compile(r"VCS[\s_\-]+PD" + _NR, re.IGNORECASE),
    # GS framework — VPA-DD / CPA-DD / PoA-DD
    re.compile(_NL + r"VPA[\s_\-]*DD" + _NR, re.IGNORECASE),
    re.compile(_NL + r"CPA[\s_\-]*DD" + _NR, re.IGNORECASE),
    re.compile(_NL + r"PoA[\s_\-]*DD" + _NR, re.IGNORECASE),
    # GS Passport (PDD-equivalent in some GS frameworks)
    re.compile(r"GS[\s_\-]+Passport", re.IGNORECASE),
    # CDM PDD (VCS projects sometimes carry CDM PDD too)
    re.compile(r"CDM[\s_]+PDD", re.IGNORECASE),
]

# Strong negative patterns — never PDD even if some PDD-keyword appears
EXCLUDE_PATTERNS = [
    # Validation / Verification reports
    re.compile(r"\b(?:Final[\s_]+)?Val(?:idation)?[\s_]+(?:Report|Statement|Opinion|Representation)", re.IGNORECASE),
    re.compile(r"\b(?:Final[\s_]+)?Ver(?:ification)?[\s_]+(?:Report|Statement|Opinion|Representation)", re.IGNORECASE),
    re.compile(r"\bFVR[\s_\-]", re.IGNORECASE),
    re.compile(r"\bFVerRep", re.IGNORECASE),
    re.compile(r"\bVerification[\s_]+Statement", re.IGNORECASE),
    # Monitoring
    re.compile(r"\bMonitoring[\s_]+Report", re.IGNORECASE),
    re.compile(r"\bMR[\s_\-]\d", re.IGNORECASE),
    re.compile(r"\bMP\d+[\s_]+MR", re.IGNORECASE),
    re.compile(r"^MR[\s_\-]", re.IGNORECASE),
    re.compile(r"^VR[\s_\-]", re.IGNORECASE),
    re.compile(r"MONIT_?REP", re.IGNORECASE),
    # Legal / admin
    re.compile(r"\bIssuance[\s_]+(?:Deed|Representation|Notice|of)", re.IGNORECASE),
    re.compile(r"\bRegistration[\s_]+(?:Deed|Representation|Certificate|Notice)", re.IGNORECASE),
    re.compile(r"\bAudit[\s_]+Report", re.IGNORECASE),
    re.compile(r"\bDeviation[\s_]+Request", re.IGNORECASE),
    # Test reports
    re.compile(r"\bWBT\b", re.IGNORECASE),
    re.compile(r"\bKPT\b", re.IGNORECASE),
    re.compile(r"\bBBT\b", re.IGNORECASE),
    re.compile(r"Water[\s_]+Boiling[\s_]+Test", re.IGNORECASE),
    re.compile(r"Kitchen[\s_]+Performance[\s_]+Test", re.IGNORECASE),
    # Stakeholder / community
    re.compile(r"Stakeholder[\s_]+Consultation", re.IGNORECASE),
    re.compile(r"\bLSC\b", re.IGNORECASE),
    re.compile(r"\bSCR\b", re.IGNORECASE),
    re.compile(r"Comm[\w]*[\s_]+Agreement", re.IGNORECASE),
    # Annual / quarterly / certification
    re.compile(r"Annual[\s_]+Report", re.IGNORECASE),
    re.compile(r"\bPerfCert", re.IGNORECASE),
    re.compile(r"Performance[\s_]+Certificate", re.IGNORECASE),
    # Generic non-PDD admin
    re.compile(r"\bAlteracao", re.IGNORECASE),
    re.compile(r"\bContrato\b", re.IGNORECASE),
    re.compile(r"Comfort[\s_]+Letter", re.IGNORECASE),
    re.compile(r"Management[\s_]+(?:Letter|Comment)", re.IGNORECASE),
]


def identify_primary_pdd(documents: list[dict]) -> tuple[dict | None, list[dict]]:
    """Return (primary_pdd_doc, supporting_docs).

    Strict — returns None if no document matches a positive PDD pattern. This
    is intentional: better to drop a project from the eval set than to mis-label
    a Monitoring Report or Issuance Deed as the PDD (which would corrupt the
    redaction target downstream). Empirically, ~10-20% of projects have no
    clearly-labeled PDD in their documents[] tree.
    """
    if not documents:
        return None, []

    # Only consider PDFs with a CDN URL
    candidates = [d for d in documents if d.get("fileUrl") and (d.get("originalFilename") or "").lower().endswith(".pdf")]
    if not candidates:
        return None, documents

    # Prefer latest version (isLatest=true) — but if none, fall back to all
    latest = [d for d in candidates if d.get("isLatest", True)]
    if latest:
        candidates = latest

    # Find PDD candidates: must match a positive pattern AND no negative
    pdd_candidates = []
    for idx, d in enumerate(candidates):
        fn = d.get("originalFilename") or ""
        if any(ex.search(fn) for ex in EXCLUDE_PATTERNS):
            continue
        for i, pat in enumerate(PDD_FILENAME_PATTERNS):
            if pat.search(fn):
                ver = d.get("version") if isinstance(d.get("version"), (int, float)) else 0
                # Rank by (pattern priority, version desc, size desc). idx as
                # final tiebreaker so the tuple is always sortable (avoids
                # TypeError on dict comparison).
                pdd_candidates.append((i, -ver, -(d.get("fileSize") or 0), idx, d))
                break

    if not pdd_candidates:
        return None, documents

    pdd_candidates.sort(key=lambda t: t[:4])
    primary = pdd_candidates[0][4]
    supporting = [x for x in documents if x is not primary]
    return primary, supporting


async def fetch_page(client: httpx.AsyncClient, api_key: str, prefix: str, page: int, page_size: int) -> dict | None:
    """Fetch one page with retry. Returns None if all retries fail (caller skips
    that page rather than aborting the whole batch — strict PDD filtering means
    we need 50+ pages to get 1000 entries, so persistent 500s on a single page
    shouldn't kill the run.)"""
    url = (
        f"{API_BASE}/projects"
        f"?filters%5BglobalId%5D%5B%24startsWith%5D={prefix}"
        f"&filters%5BcreditingPeriodStartDate%5D%5B%24gte%5D=2020-01-01"
        f"&filters%5BtotalCreditsIssued%5D%5B%24gt%5D=0"
        f"&pagination%5BpageSize%5D={page_size}"
        f"&pagination%5Bpage%5D={page}"
    )
    for attempt in range(6):
        try:
            r = await client.get(url, headers={"x-api-key": api_key}, timeout=120.0)
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPStatusError, httpx.ReadTimeout, httpx.ConnectError) as e:
            wait = min(2 ** attempt, 30)
            if attempt < 5:
                logger.warning("[%s page=%d] %s; retry %d in %ds",
                               prefix, page, str(e)[:80], attempt + 1, wait)
                await asyncio.sleep(wait)
    logger.error("[%s page=%d] giving up after 6 retries — skipping this page", prefix, page)
    return None


def normalize_entry(p: dict, registry: str) -> dict | None:
    """Convert one /projects item to a manifest row. Returns None if no PDD found."""
    documents = p.get("documents") or []
    primary, supporting = identify_primary_pdd(documents)
    if primary is None:
        return None
    methods = [m.get("code") if isinstance(m, dict) else m for m in (p.get("methodologies") or [])]
    return {
        "globalId": p.get("globalId"),
        "registry": registry,
        "name": p.get("name"),
        "country": p.get("country"),
        "methodologies": [m for m in methods if m],
        "creditingPeriodStartDate": (p.get("creditingPeriodStartDate") or "")[:10],
        "creditingPeriodEndDate": (p.get("creditingPeriodEndDate") or "")[:10],
        "totalCreditsIssued": p.get("totalCreditsIssued"),
        "estimatedAnnualCredits": p.get("estimatedAnnualCredits"),
        "primary_pdd_url": primary.get("fileUrl"),
        "primary_pdd_filename": primary.get("originalFilename"),
        "primary_pdd_size": primary.get("fileSize"),
        "primary_pdd_version": primary.get("version"),
        "supporting_doc_urls": [d.get("fileUrl") for d in supporting if d.get("fileUrl")][:30],
    }


async def fetch_all(api_key: str, prefix: str, target_sample: int, page_size: int = 20) -> tuple[list[dict], int]:
    """Fetch all matching projects (sample size capped to target). Returns (entries, pool_total).
    Skips pages that fail after all retries."""
    async with httpx.AsyncClient() as client:
        first = await fetch_page(client, api_key, prefix, page=1, page_size=page_size)
        if first is None:
            logger.error("[%s] first page failed — aborting fetch_all for this prefix", prefix)
            return [], 0
        meta = first.get("meta", {}).get("pagination", {})
        pool = meta.get("totalCount", 0)
        total_pages = meta.get("totalPages", 0)
        logger.info("[%s] pool=%d totalPages=%d target_sample=%d", prefix, pool, total_pages, target_sample)

        def consume(body, into: list, max_n: int):
            for p in body.get("data", []):
                if len(into) >= max_n:
                    return False
                row = normalize_entry(p, prefix)
                if row:
                    into.append(row)
            return True

        out = []
        if not consume(first, out, target_sample):
            return out[:target_sample], pool
        skipped_pages = 0
        for page in range(2, total_pages + 1):
            if len(out) >= target_sample:
                break
            body = await fetch_page(client, api_key, prefix, page=page, page_size=page_size)
            if body is None:
                skipped_pages += 1
                continue  # skip transient-fail page, keep going
            if not consume(body, out, target_sample):
                break
        if skipped_pages:
            logger.warning("[%s] skipped %d pages due to repeated 500s", prefix, skipped_pages)
        return out[:target_sample], pool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=1500,
                        help="Total cross-registry sample target (default 1500)")
    parser.add_argument("--gs-sample", type=int, default=None,
                        help="Override GS sample size (default proportional)")
    parser.add_argument("--vcs-sample", type=int, default=None,
                        help="Override VCS sample size (default proportional)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    api_key = os.environ.get("GENVISION_API_KEY")
    if not api_key:
        logger.error("GENVISION_API_KEY not set. source .env first.")
        return 1

    # Known pool sizes from 2026-05-21 sweep:
    KNOWN_POOLS = {"GS": 799, "VCS": 1552}
    total_pool = sum(KNOWN_POOLS.values())
    samples = {
        "GS": args.gs_sample or round(args.target * KNOWN_POOLS["GS"] / total_pool),
        "VCS": args.vcs_sample or round(args.target * KNOWN_POOLS["VCS"] / total_pool),
    }
    logger.info("Sampling plan: %s (total target=%d)", samples, sum(samples.values()))

    all_entries = []
    actual_pools = {}
    for prefix in ["GS", "VCS"]:
        logger.info("Fetching %s up to %d...", prefix, samples[prefix])
        entries, pool = asyncio.run(fetch_all(api_key, prefix, samples[prefix]))
        actual_pools[prefix] = pool
        logger.info("  → got %d entries (pool live=%d)", len(entries), pool)
        all_entries.extend(entries)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": "0.1",
        "generated_at": _utcnow(),
        "filter": {
            "globalId_startsWith": ["GS", "VCS"],
            "creditingPeriodStartDate_gte": "2020-01-01",
            "totalCreditsIssued_gt": 0,
        },
        "target_sample": args.target,
        "live_pools": actual_pools,
        "sample_per_registry": {k: sum(1 for e in all_entries if e["registry"] == k) for k in ["GS", "VCS"]},
        "total_entries": len(all_entries),
        "entries": all_entries,
    }
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info("Wrote %s (%d entries)", args.output, len(all_entries))

    # Methodology distribution preview
    from collections import Counter
    method_counts = Counter()
    for e in all_entries:
        for m in (e.get("methodologies") or []):
            method_counts[m] += 1
    logger.info("Distinct methodology codes: %d", len(method_counts))
    logger.info("Top 10 most frequent labels:")
    for code, n in method_counts.most_common(10):
        logger.info("  %s: %d", code, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
