"""Query a harvested-data Postgres for the full GS-project methodology label set.

A small set of curated QA pairs covers only a subset of projects with a
methodology_applied label. The full field-extraction table typically covers
many more of the scraped GS PDDs. This script queries that table directly to
build the comprehensive eligible-project list.

Connection (choose one):
    1. DATABASE_URL env var (preferred), e.g.
         export DATABASE_URL="postgresql://user:pass@localhost:5432/carbonmethod"
    2. PGHOST / PGUSER / PGPASSWORD / PGDATABASE env vars
    3. If the database is only reachable over SSH, open a local port-forward
       first, then point DATABASE_URL at the forwarded localhost port.

Connection details (host, db name, user, password) are kept outside this
repository and supplied via environment variables.

Usage:
    python -m carbonmm.ingest.query_postgres
    python -m carbonmm.ingest.query_postgres --output data/postgres_methodologies.json
    python -m carbonmm.ingest.query_postgres --dry-run    # show SQL, do not connect
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .common import DATA_ROOT, setup_logging

logger = logging.getLogger(__name__)

# Query: every PDD with a non-empty methodology_applied field.
# Filter: keep entries the field extractor produced with reasonable confidence
#         (extraction_rate >= 0.6 eligibility filter).
SQL = """
SELECT
    project_gs_id,
    methodology_applied,
    extraction_rate
FROM pdd_v15_key_info
WHERE methodology_applied IS NOT NULL
  AND methodology_applied <> ''
  AND extraction_rate >= 0.6
ORDER BY project_gs_id
"""

# Optional: cross-reference with lifecycle_events (issued projects only)
SQL_ISSUED_FILTER = """
SELECT k.project_gs_id, k.methodology_applied, k.extraction_rate
FROM pdd_v15_key_info k
JOIN lifecycle_events e ON e.project_gs_id = k.project_gs_id
WHERE k.methodology_applied IS NOT NULL
  AND k.methodology_applied <> ''
  AND k.extraction_rate >= 0.6
  AND e.event_type = 'ISSUANCE'
GROUP BY k.project_gs_id, k.methodology_applied, k.extraction_rate
ORDER BY k.project_gs_id
"""


def normalize_code(raw: str) -> str:
    """AMSI.D → AMS-I.D; collapse whitespace; uppercase code segments."""
    s = re.sub(r"\s+", "", raw).upper()
    s = re.sub(r"^AMS([IVX]+)", r"AMS-\1", s)
    s = re.sub(r"^ARAMS([IVX]+)", r"AR-AMS-\1", s)
    s = re.sub(r"^ARAM(\d)", r"AR-AM\1", s)
    return s


CODE_PATTERNS = [
    (r"\bACM\d{4}\b", "CDM"),
    (r"\bAM\d{4}\b", "CDM"),
    (r"\bAMS[-\s]*[IVX]{1,4}(?:\.[A-Z]{1,2})?\b", "CDM"),
    (r"\bAR-?AM\d{4}\b", "CDM"),
    (r"\bAR-?AMS[-\s]*[IVX]+\b", "CDM"),
    (r"\bVM\d{4}\b", "Verra"),
    (r"\bVMD\d{4}\b", "Verra"),
    (r"\bTPDDTEC\b", "GS"),
    (r"\bGS[-\s]*VER\d+\b", "GS"),
]


def canonicalize(raw_label: str) -> tuple[str, str, str]:
    """Return (registry, code_or_title, label_type).

    label_type: "coded" (matched code regex) | "verbose-title" (free text)
    """
    if not raw_label:
        return ("Unknown", "", "empty")
    for pattern, registry in CODE_PATTERNS:
        m = re.search(pattern, raw_label)
        if m:
            return (registry, normalize_code(m.group(0)), "coded")
    # No code → treat as verbose GS-native title (will be canonicalized via
    # LLM-assisted title→code mapping)
    clean = re.sub(r"^\s*[|\-\s]+", "", raw_label)
    clean = re.sub(r"\s*\|\s*", " ", clean).strip()
    clean = re.sub(r"(?i)\s+(?:version|v\.?\s*)\s*\d+(?:\.\d+)?\s*$", "", clean).strip()
    return ("GS-native", clean.lower(), "verbose-title")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_ROOT / "postgres_methodologies.json",
        help="Output JSON path (default: data/postgres_methodologies.json)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show the SQL and connection info; do not connect.",
    )
    parser.add_argument(
        "--issued-only", action="store_true",
        help="Filter to projects with at least one ISSUANCE lifecycle event.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    sql = SQL_ISSUED_FILTER if args.issued_only else SQL
    logger.info("SQL:\n%s", sql.strip())

    # Connection
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        host = os.environ.get("PGHOST", "localhost")
        port = os.environ.get("PGPORT", "5432")
        user = os.environ.get("PGUSER", "postgres")
        pw = os.environ.get("PGPASSWORD", "")
        db = os.environ.get("PGDATABASE", "carbonmethod")
        dsn = f"postgresql://{user}:{pw}@{host}:{port}/{db}"

    sanitized_dsn = re.sub(r"://([^:]+):[^@]+@", r"://\1:***@", dsn)
    logger.info("Connecting to: %s", sanitized_dsn)

    if args.dry_run:
        logger.info("--dry-run: no actual connection. Exit.")
        return 0

    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        logger.error("psycopg2 not installed. Run: pip install psycopg2-binary")
        return 1

    try:
        conn = psycopg2.connect(dsn)
    except Exception as e:
        logger.error("Connection failed: %s", e)
        logger.error("Tips: if the database is only reachable over SSH, open a local port-forward to it first.")
        return 1

    rows = []
    with conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    conn.close()
    logger.info("Fetched %d rows from pdd_v15_key_info", len(rows))

    # Canonicalize + count
    project_label = {}
    freq = Counter()
    for r in rows:
        gs_id = r["project_gs_id"]
        registry, code, label_type = canonicalize(r["methodology_applied"])
        project_label[gs_id] = {
            "raw": r["methodology_applied"],
            "registry": registry,
            "canonical": code,
            "label_type": label_type,
            "extraction_rate": float(r["extraction_rate"]) if r.get("extraction_rate") is not None else None,
        }
        freq[(registry, code)] += 1

    logger.info("Unique projects: %d | distinct canonical labels: %d", len(project_label), len(freq))
    logger.info("Top 10 labels:")
    for (reg, code), n in freq.most_common(10):
        logger.info("  %s | %-40s  %5d", reg, code[:40], n)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "source": "field-extraction table (harvested-data Postgres)",
            "issued_only": args.issued_only,
            "row_count": len(rows),
            "project_count": len(project_label),
            "distinct_labels": len(freq),
        },
        "projects": project_label,
        "label_frequencies": [
            {"registry": reg, "code_or_title": code, "count": n}
            for (reg, code), n in freq.most_common()
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info("Saved → %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
