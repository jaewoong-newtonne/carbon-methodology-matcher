"""Guidance-KG seam for the GraphRAG connection (Approach C, hybrid).

Single boundary between the matcher (candidate_filter / retrieve / rerank) and
the guidance knowledge. Consumers go through these functions and never touch the
pack format or TypeDB directly:

  load_pack(path)                         -> GuidancePack (cached)
  scale_signal(code, pdd_value, conf)     -> ScaleSignal | None   (None = no signal/ceiling)
  guidance_clauses_for(text, registry, k) -> list[ClauseHit]      (semantic, registry-filtered)
  kg_fallback(...)                        -> live TypeDB read (pack-miss / live_kg=True)

Design notes:
- Scale signal is the SAFE, high-precision direction of cdm-ssc-thresholds.json
  `scale_fit_logic`: a small-scale (AMS) candidate is *soft-down-weighted* when the
  PDD's stated value EXCEEDS its cap ("C>T ⇒ AMS ineligible ⇒ favor large"). Large
  candidates get no penalty (their relative boost comes from AMS down-weighting).
- SOFT, never hard-exclude. None when the PDD value is absent (the information
  ceiling — no penalty). Low-confidence values soften the penalty.
- The numeric cap is deterministic (from the KG-derived pack); only the prose
  clauses are left for the LLM — encodes the 60→600 LLM-misread lesson.

Clause retrieval embeds via OpenAI text-embedding-3-large. The TypeDB fallback
is read-only.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_ICDM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK = _ICDM_ROOT / "data" / "manifests" / "guidance-pack.json"
DEFAULT_EMB = _ICDM_ROOT / "data" / "manifests" / "guidance-clause-embeddings.npz"

# Soft scale penalties (multipliers in [0,1]; 1.0 = no penalty).
EXCEED_PENALTY = 0.5          # high-confidence value exceeds the AMS cap
EXCEED_PENALTY_LOWCONF = 0.75  # low-confidence value exceeds the cap (softer)
LOWCONF_THRESHOLD = 0.7
EMB_MODEL = "text-embedding-3-large"


@dataclass
class ScaleSignal:
    code: str
    multiplier: float          # blend into candidate score
    fits: bool                 # True if PDD value within cap
    cap: float
    unit: str
    pdd_value: float
    rationale: str             # human/LLM-readable, includes provenance
    provenance: dict           # {guidance_doc_id, doc_title, threshold_id}


@dataclass
class ClauseHit:
    clause_id: str
    clause_type: str
    registry: str
    text: str
    source_doc_id: str
    score: float


@dataclass
class GuidancePack:
    thresholds_by_methodology: dict
    guidance_clauses: list
    meta: dict


@lru_cache(maxsize=4)
def load_pack(path: str | os.PathLike = DEFAULT_PACK) -> Optional[GuidancePack]:
    """Load the KG-derived guidance pack; None (warn) if absent → consumers degrade."""
    p = Path(path)
    if not p.exists():
        logger.warning("guidance pack absent (%s) — guidance signals disabled; baseline behavior.", p)
        return None
    d = json.loads(p.read_text())
    return GuidancePack(
        thresholds_by_methodology=d.get("thresholds_by_methodology", {}),
        guidance_clauses=d.get("guidance_clauses", []),
        meta=d.get("meta", {}),
    )


def scale_signal(code: str, pdd_value: Optional[float], pdd_value_conf: float = 1.0,
                 *, pack: Optional[GuidancePack] = None) -> Optional[ScaleSignal]:
    """Soft SSC scale fit for one candidate. None when no signal applies."""
    pack = pack or load_pack()
    if pack is None:
        return None
    thr = pack.thresholds_by_methodology.get(code)
    if thr is None:
        return None  # large/uncapped candidate: no penalty (relative boost via AMS down-weight)
    if pdd_value is None:
        return None  # information ceiling — value absent, no penalty
    cap = float(thr.get("cap") or 0.0)
    unit = thr.get("unit", "")
    prov = thr.get("provenance", {})
    doc = prov.get("guidance_doc_id", "guidance")
    if pdd_value <= cap:
        return ScaleSignal(code, 1.0, True, cap, unit, pdd_value,
                           f"within SSC cap (≤{cap:g} {unit}) per {doc}", prov)
    mult = EXCEED_PENALTY if pdd_value_conf >= LOWCONF_THRESHOLD else EXCEED_PENALTY_LOWCONF
    return ScaleSignal(
        code, mult, False, cap, unit, pdd_value,
        f"PDD states {pdd_value:g} {unit} > SSC cap {cap:g} {unit} → favor large variant, per {doc}",
        prov,
    )


def scale_multipliers(candidate_codes, pdd_value: Optional[float], pdd_value_conf: float = 1.0,
                      *, pack: Optional[GuidancePack] = None) -> dict[str, float]:
    """code -> soft multiplier for every candidate with a (non-neutral) scale signal.

    Returns only codes that get a penalty (<1.0); callers treat missing codes as 1.0.
    Empty dict when value absent or pack missing (ceiling / degrade-to-baseline).
    """
    pack = pack or load_pack()
    if pack is None or pdd_value is None:
        return {}
    out = {}
    for code in candidate_codes:
        s = scale_signal(code, pdd_value, pdd_value_conf, pack=pack)
        if s is not None and s.multiplier < 1.0:
            out[code] = s.multiplier
    return out


# ─── Clause retrieval (semantic, registry-filtered) ─────────────────────────
def _load_openai_key() -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    env = _ICDM_ROOT.parents[1] / ".env"  # the project root/.env
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    return ""


@lru_cache(maxsize=2)
def _load_clause_embeddings(path: str | os.PathLike = DEFAULT_EMB):
    """Return (clause_ids: list[str], matrix: np.ndarray) or None if absent."""
    p = Path(path)
    if not p.exists():
        logger.warning("guidance-clause embeddings absent (%s) — clause signal disabled.", p)
        return None
    import numpy as np
    # Self-generated artifact (tools/embed_guidance_clauses.py); clause_ids stored
    # as a numpy unicode-string array and vectors as float32 — both load WITHOUT
    # allow_pickle (no object arrays), so no arbitrary-code-execution surface.
    d = np.load(p, allow_pickle=False)
    return [str(x) for x in d["clause_ids"]], d["vectors"]


def guidance_clauses_for(pdd_text: str, registry: str, k: int = 5,
                         *, pack: Optional[GuidancePack] = None,
                         openai_client=None) -> list[ClauseHit]:
    """Top-K guidance clauses by similarity to the PDD, filtered to `registry`."""
    pack = pack or load_pack()
    if pack is None:
        return []
    emb = _load_clause_embeddings()
    if emb is None:
        return []
    import numpy as np
    clause_ids, mat = emb
    by_id = {c["clause_id"]: c for c in pack.guidance_clauses}
    # registry mask
    keep = [i for i, cid in enumerate(clause_ids)
            if (by_id.get(cid, {}).get("registry", "") == registry or not registry)]
    if not keep:
        return []
    # embed the PDD query
    try:
        if openai_client is None:
            from openai import OpenAI
            openai_client = OpenAI(api_key=_load_openai_key())
        q = openai_client.embeddings.create(model=EMB_MODEL, input=pdd_text[:8000]).data[0].embedding
        qv = np.asarray(q, dtype=np.float32)
    except Exception as e:
        logger.warning("clause query embedding failed: %s", e)
        return []
    sub = mat[keep]
    sims = sub @ qv / (np.linalg.norm(sub, axis=1) * np.linalg.norm(qv) + 1e-9)
    order = np.argsort(-sims)[:k]
    hits = []
    for j in order:
        i = keep[int(j)]
        c = by_id.get(clause_ids[i], {})
        hits.append(ClauseHit(clause_ids[i], c.get("clause_type", ""), c.get("registry", ""),
                              c.get("text", ""), c.get("source_doc_id", ""), float(sims[int(j)])))
    return hits


def rerank_block(candidate_codes, pdd_value: Optional[float], pdd_value_conf: float = 1.0,
                 pdd_text: str = "", registry: str = "", *, want_scale: bool = True,
                 want_clauses: bool = True, k_clauses: int = 4,
                 pack: Optional[GuidancePack] = None, openai_client=None) -> str:
    """Injectable rerank-prompt text: per-candidate SSC scale facts (with provenance)
    + retrieved registry clauses. Returns "" if nothing to add (→ baseline prompt)."""
    pack = pack or load_pack()
    if pack is None:
        return ""
    parts: list[str] = []
    if want_scale and pdd_value is not None:
        lines = []
        for code in candidate_codes:
            s = scale_signal(code, pdd_value, pdd_value_conf, pack=pack)
            if s is not None:
                lines.append(f"- {code}: {s.rationale}")
        if lines:
            parts.append("## SSC scale thresholds (authoritative — apply with priority, cite source)\n"
                         + "\n".join(lines))
    if want_clauses:
        hits = guidance_clauses_for(pdd_text, registry, k_clauses, pack=pack, openai_client=openai_client)
        if hits:
            cl = [f"- [{h.source_doc_id}] ({h.clause_type}) {h.text[:300]}" for h in hits]
            parts.append("## Relevant registry rules (retrieved)\n" + "\n".join(cl))
    if not parts:
        return ""
    return ("# Guidance facts (registry rules + scale thresholds, with provenance)\n"
            + "\n\n".join(parts))


# ─── Live-KG fallback (the "C" hybrid half) ─────────────────────────────────
def kg_fallback(code: str, *, host: str = "localhost:1729", db: str = "climate_kg") -> Optional[dict]:
    """Live read of a code's SSC threshold from TypeDB (pack-miss / deployment).

    Read-only; returns the same shape as a pack threshold entry, or None.
    """
    try:
        from typedb.driver import TypeDB, Credentials, DriverOptions, TransactionType
    except ImportError:
        return None
    pw = os.environ.get("TYPEDB_PASSWORD")
    if not pw:
        return None
    creds = Credentials(os.environ.get("TYPEDB_USER", "admin"), pw)
    q = (f'match $m isa carbon-methodology, has code "{code}"; '
         '(threshold: $t, subject: $m) isa threshold-applies-to; '
         '$t has ssc-type $ty, has threshold-dimension $dim, has cap-value $cap, has cap-unit $u; '
         '(document: $d, threshold: $t) isa guidance-source-of; $d has guidance-doc-id $did; '
         'select $ty, $dim, $cap, $u, $did;')
    try:
        with TypeDB.driver(f"typedb://{host}", creds, DriverOptions(is_tls_enabled=False)) as drv:
            with drv.transaction(db, TransactionType.READ) as tx:
                for r in tx.query(q).resolve():
                    return {
                        "ssc_type": r.get("ty").get_value(),
                        "dimension": r.get("dim").get_value(),
                        "cap": r.get("cap").get_value(),
                        "unit": r.get("u").get_value(),
                        "provenance": {"guidance_doc_id": r.get("did").get_value()},
                    }
    except Exception as e:
        logger.warning("kg_fallback(%s) failed: %s", code, e)
    return None
