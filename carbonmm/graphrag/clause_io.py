"""Clause extraction from methodology-corpus JSONs.

Mirrors the clause taxonomy in `tools/load_methodology_corpus.py` so the
graphrag index is row-aligned with `methodology-clause` entities in TypeDB.

Clause types (7):
    applicability                — one per bullet
    mitigation-action            — one per methodology
    parameters-at-validation     — one (combined bullets)
    parameters-monitored         — one (combined bullets)
    baseline-scenario            — one per methodology
    project-scenario             — one per methodology
    typical-projects             — one per methodology (NEW for embeddings;
                                   not loaded into TypeDB but useful as a
                                   retrieval target since it summarizes
                                   project archetypes)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

CORPUS_ROOT = Path(__file__).resolve().parents[1] / "data" / "corpus"


@dataclass
class ClauseRecord:
    clause_id: str
    code: str
    registry: str
    clause_type: str
    clause_text: str

    def to_dict(self) -> dict:
        return asdict(self)


def _infer_registry(code: str, json_registry: str | None) -> str:
    if json_registry:
        return json_registry
    if any(code.startswith(p) for p in ("AM", "ACM", "AR")):
        return "CDM"
    if code.startswith(("VM", "VMD", "VMR")):
        return "VCS"
    if code.startswith(("TPDDTEC", "GS")) or code.isdigit() or "-" in code:
        return "GS"
    return ""


def iter_clauses_from_json(path: Path) -> Iterator[ClauseRecord]:
    """Yield ClauseRecord per non-empty section of one methodology JSON."""
    try:
        d = json.loads(path.read_text())
    except Exception as e:
        logger.error("skip %s: %s", path, e)
        return
    code = d.get("code")
    if not code:
        return
    registry = _infer_registry(code, d.get("registry"))

    # applicability — one clause per bullet
    for i, bullet in enumerate(d.get("applicability") or [], 1):
        text = (bullet or "").strip()
        if text:
            yield ClauseRecord(f"{code}-app-{i}", code, registry, "applicability", text)

    # mitigation-action — one combined
    mit = (d.get("mitigation_action") or "").strip()
    if mit:
        yield ClauseRecord(f"{code}-mit-1", code, registry, "mitigation-action", mit)

    # typical-projects — one combined (embed only; not in TypeDB)
    tp = (d.get("typical_projects") or "").strip()
    if tp:
        yield ClauseRecord(f"{code}-typical-projects", code, registry, "typical-projects", tp)

    # parameters-at-validation, parameters-monitored — combined bullets
    for sub_key, label in [
        ("parameters_at_validation", "parameters-at-validation"),
        ("parameters_monitored", "parameters-monitored"),
    ]:
        bullets = d.get(sub_key) or []
        joined = " | ".join(b for b in bullets if b).strip()
        if joined:
            yield ClauseRecord(
                f"{code}-{sub_key.replace('_', '-')}-1", code, registry, label, joined
            )

    # baseline-scenario, project-scenario
    for sub_key, label in [
        ("baseline_scenario", "baseline-scenario"),
        ("project_scenario", "project-scenario"),
    ]:
        text = (d.get(sub_key) or "").strip()
        if text:
            yield ClauseRecord(f"{code}-{label}", code, registry, label, text)


def iter_all_clauses(corpus_root: Path = CORPUS_ROOT) -> Iterator[ClauseRecord]:
    """Walk corpus_root for every methodology JSON, yield ClauseRecords."""
    for f in sorted(corpus_root.glob("*/*/*.json")):
        yield from iter_clauses_from_json(f)


def dump_clauses_parquet(out_path: Path, corpus_root: Path = CORPUS_ROOT) -> int:
    """Write all clauses to a parquet file. Returns row count."""
    import pandas as pd

    rows = [c.to_dict() for c in iter_all_clauses(corpus_root)]
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return len(df)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="carbonmm/data/embeddings/clauses-meta.parquet")
    args = ap.parse_args()
    n = dump_clauses_parquet(Path(args.out))
    print(f"wrote {n} clauses to {args.out}")
