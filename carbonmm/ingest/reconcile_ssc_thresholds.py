"""Reconcile cdm-ssc-thresholds.json against the harvested EB66-A23-GUID caps.

The SSC threshold map (data/manifests/cdm-ssc-thresholds.json) carries two
`"verify": true` entries — AMS-I.C. and AMS-I.E. (thermal 45 MWth) — pending
confirmation against the source rulebook. The authoritative source is the CDM
PROJECT STANDARD (reg_stan04), whose Type I definition states "15 MW(e) is
equivalent to a 45 MW thermal output ... the limit of 45 MW(th)". (The manifest
originally pointed at EB66-A23-GUID, but that doc only references the caps via a
1%-of-threshold example — it does not state them.) The harvest fetches the
Project Standard; --llm extraction → {doc_id}.thresholds.json.

This step matches each verify:true entry to a harvested threshold by
(ssc_type, dimension):
  - cap matches   → set verify:false + record a `provenance` sub-object.
  - cap differs   → set verify:"conflict" + log both (no silent overwrite).
  - no harvested match → leave verify:true (logs "unconfirmed").
Top-level `prediction_blind` stays true (public regulatory thresholds); a
confirmation note is appended to the top-level `source` string.

Usage (after extract_guidance_sections.py --llm produced the thresholds JSON):
    python3 -m carbonmm.ingest.reconcile_ssc_thresholds --dry-run
    python3 -m carbonmm.ingest.reconcile_ssc_thresholds
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .common import CORPUS_DIR, MANIFEST_DIR, load_manifest, setup_logging, _utcnow

logger = logging.getLogger(__name__)

SSC_MAP = MANIFEST_DIR / "cdm-ssc-thresholds.json"
# Authoritative SSC-threshold source = the CDM PROJECT STANDARD (Type I/II/III
# definitions: 15 MW electrical, 45 MW(th) thermal, 60 GWh/yr, 60 ktCO2e/yr).
# NOTE: EB66-A23-GUID ("General guidelines for SSC methodologies") does NOT state
# the caps directly — it only references them (via a 1%-of-threshold example),
# so the original manifest pointer to it was incorrect.
SSC_SOURCE_DOC_ID = "CDM-reg_stan04_v03.0"
HARVESTED = CORPUS_DIR / "CDM" / "_guidance" / SSC_SOURCE_DOC_ID / f"{SSC_SOURCE_DOC_ID}.thresholds.json"
GUIDANCE_JSON = CORPUS_DIR / "CDM" / "_guidance" / SSC_SOURCE_DOC_ID / f"{SSC_SOURCE_DOC_ID}.guidance.json"


def _dim_match(a: str, b: str) -> bool:
    a, b = (a or "").lower().strip(), (b or "").lower().strip()
    return a == b or (a in b) or (b in a)


def reconcile(dry_run: bool) -> int:
    if not SSC_MAP.exists():
        logger.error("SSC map not found: %s", SSC_MAP)
        return 1
    if not HARVESTED.exists():
        logger.error("Harvested thresholds not found: %s\n  Run extract_guidance_sections.py --registry cdm --llm first.", HARVESTED)
        return 1

    ssc = json.loads(SSC_MAP.read_text())
    harvested = json.loads(HARVESTED.read_text()).get("thresholds", [])
    doc_version = ""
    if GUIDANCE_JSON.exists():
        doc_version = json.loads(GUIDANCE_JSON.read_text()).get("doc_version", "")
    if not doc_version:  # fallback: harvester manifest carries the version
        mpath = MANIFEST_DIR / "guidance-review-cdm.yaml"
        if mpath.exists():
            doc_version = next((e.version for e in load_manifest(mpath) if e.code == SSC_SOURCE_DOC_ID), "")
    today = _utcnow()[:10]

    logger.info("Harvested %d thresholds from %s (v%s)", len(harvested), SSC_SOURCE_DOC_ID, doc_version or "?")

    confirmed = conflicts = unconfirmed = 0
    for code, entry in ssc.get("thresholds", {}).items():
        if entry.get("verify") is not True:
            continue
        ssc_type = entry.get("type", "")
        dim = entry.get("dimension", "")
        cap = entry.get("cap")
        match = next((h for h in harvested
                      if h.get("ssc_type") == ssc_type and _dim_match(h.get("dimension", ""), dim)), None)
        if not match:
            unconfirmed += 1
            logger.warning("  %s: no harvested Type-%s %s cap → left verify:true", code, ssc_type, dim)
            continue
        h_cap = float(match.get("cap_value", 0) or 0)
        if abs(h_cap - float(cap)) < 1e-6:
            entry["verify"] = False
            entry["provenance"] = {
                "guidance_doc_id": SSC_SOURCE_DOC_ID, "doc_version": doc_version,
                "confirmed_at": today, "confirmed_cap": h_cap, "unit": match.get("unit", entry.get("unit")),
            }
            confirmed += 1
            logger.info("  %s: CONFIRMED %.1f %s by %s", code, h_cap, match.get("unit"), SSC_SOURCE_DOC_ID)
        else:
            entry["verify"] = "conflict"
            entry["provenance"] = {
                "guidance_doc_id": SSC_SOURCE_DOC_ID, "doc_version": doc_version, "confirmed_at": today,
                "manifest_cap": cap, "harvested_cap": h_cap, "unit": match.get("unit"),
            }
            conflicts += 1
            logger.warning("  %s: CONFLICT manifest=%.1f vs harvested=%.1f %s → verify:\"conflict\" (review)",
                           code, float(cap), h_cap, match.get("unit"))

    note = f" SSC caps reconciled {today} vs {SSC_SOURCE_DOC_ID} v{doc_version}: {confirmed} confirmed, {conflicts} conflict."
    if note.strip() not in ssc.get("source", ""):
        ssc["source"] = ssc.get("source", "") + note

    logger.info("Reconcile summary: %d confirmed, %d conflict, %d unconfirmed.", confirmed, conflicts, unconfirmed)
    if dry_run:
        logger.info("--dry-run: %s NOT written.", SSC_MAP.name)
        return 0
    SSC_MAP.write_text(json.dumps(ssc, indent=2, ensure_ascii=False) + "\n")
    logger.info("Wrote %s", SSC_MAP)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="Show reconciliation without writing the manifest.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    setup_logging(args.verbose)
    return reconcile(args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
