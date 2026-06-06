"""Two-stage retrieval-side candidate filter.

Stage A — registry + family pre-filter (deterministic, lossless verified on val+test).
Stage B — date validity (conservative) + AMS-large hard-exclude (existing rule).
         Sectoral-scope soft-boost is exposed as a side-channel for the retriever
         to weight BM25 scores (not used to exclude).

Designed for post-retrieval masking: builds an `allow_clause_idx` set for the
retriever instead of rebuilding indices per query.

Data sources (all known at PDD-registration time, no test leakage):
  - PDD body: registry, creditingPeriodStartDate
  - Genvision catalog: all_standards, status, all_versions[].effective_to,
                       sectoral_scopes[].code
  - Static prefix conventions (CDM/VCS/GS family)

See ../paper/sections/6-discussion-oracle-ceiling.md for the motivation
(Oracle ceiling 0.632; 87/535 retrieval failures from off-registry distractors).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, date
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


# ─── Family prefix rules ───────────────────────────────────────────────────

CDM_RE = re.compile(r"^(ACM|AM\d|AMS-|AR-|ARNM)")
VCS_RE = re.compile(r"^(VM\d|VMR|VMD|GCCM)")
GS_RE = re.compile(r"^(GS-|\d)")  # numeric 4xx are GS


def code_family(code: str) -> str:
    """Return one of {'CDM', 'VCS', 'GS', 'OTHER'} by prefix."""
    if CDM_RE.match(code):
        return "CDM"
    if VCS_RE.match(code):
        return "VCS"
    if GS_RE.match(code):
        return "GS"
    return "OTHER"


def registry_keep(pdd_registry: str, code: str, catalog_all_stds: dict[str, set[str]]) -> bool:
    """Stage A: keep code IFF it's in the PDD's eligible family OR catalog lists
    the PDD's registry in `all_standards` (safety net for prefix-rule misses).

    Unknown registries fall back to permissive (keep everything).
    """
    if not pdd_registry:
        return True
    fam = code_family(code)
    stds = catalog_all_stds.get(code, set())
    if pdd_registry == "GS":
        return fam in {"GS", "CDM"} or "GS" in stds
    if pdd_registry == "VCS":
        return fam in {"VCS", "CDM"} or "VCS" in stds
    return True


# ─── Date validity (conservative, from router_v2_dated) ────────────────────

def parse_date(s) -> date | None:
    if s is None or s == "":
        return None
    if isinstance(s, str):
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
        except Exception:
            try:
                return datetime.strptime(s, "%Y-%m-%d").date()
            except Exception:
                return None
    if isinstance(s, date):
        return s
    return None


def is_invalid_at(code: str, target_date: date | None,
                  versions_index: dict[str, list[tuple[date | None, date | None]]],
                  status_index: dict[str, str]) -> bool:
    """Conservative: only exclude clearly-invalid (Withdrawn / fully deprecated)."""
    if target_date is None:
        return False
    if status_index.get(code, "Unknown") == "Withdrawn":
        return True
    versions = versions_index.get(code, [])
    if not versions:
        return False
    all_ended = True
    has_end_date = False
    for ef, et in versions:
        if et is None:
            all_ended = False
            break
        has_end_date = True
        if et >= target_date:
            all_ended = False
            break
    return all_ended and has_end_date


# ─── AMS-large hard-exclude (reuse semantics from graph_filter._scale_excluded) ──

_AMS_RE = re.compile(r"^AMS-")


def scale_excluded(code: str, scale: str) -> bool:
    """AMS-* methodologies are small-scale-only; exclude when project scale=large."""
    return scale == "large" and bool(_AMS_RE.match(code))


# ─── Catalog loader (cached) ───────────────────────────────────────────────

CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "manifests" / "genvision-methodology-catalog.json"


@lru_cache(maxsize=1)
def load_catalog_indices() -> tuple[
    dict[str, set[str]],
    dict[str, list[tuple[date | None, date | None]]],
    dict[str, str],
    dict[str, set[str]],
]:
    """Returns: (all_standards_idx, versions_idx, status_idx, sectoral_scope_idx)."""
    raw = json.loads(CATALOG_PATH.read_text())
    entries = raw.get("entries", [])
    all_stds: dict[str, set[str]] = {}
    versions: dict[str, list[tuple[date | None, date | None]]] = {}
    status: dict[str, str] = {}
    scopes: dict[str, set[str]] = {}
    for e in entries:
        c = e.get("code")
        if not c:
            continue
        all_stds[c] = set(e.get("all_standards") or [])
        status[c] = e.get("status", "Unknown")
        vlist = []
        for v in (e.get("all_versions") or []):
            vlist.append((parse_date(v.get("effective_from")), parse_date(v.get("effective_to"))))
        versions[c] = vlist
        ss = e.get("sectoral_scopes") or []
        scopes[c] = {str(s.get("code")) for s in ss if isinstance(s, dict) and s.get("code")}
    logger.info("loaded catalog: %d codes (%d with versions, %d with scopes)",
                len(all_stds), sum(1 for v in versions.values() if v),
                sum(1 for s in scopes.values() if s))
    return all_stds, versions, status, scopes


# ─── Composed Stage A + B filter ───────────────────────────────────────────

@dataclass
class FilterDiag:
    n_total_codes: int = 0
    n_after_stage_a: int = 0
    n_after_date: int = 0
    n_after_ams_scale: int = 0
    scope_boosted_codes: set[str] = field(default_factory=set)


def build_allowlist(
    pdd_registry: str | None,
    pdd_credit_start: date | None,
    pdd_scale: str | None,
    pdd_scope: str | None,
    corpus_codes: Iterable[str],
    apply_date_filter: bool = False,
) -> tuple[set[str], FilterDiag]:
    """Compose Stage A + B into a single allowlist + diagnostic.

    `pdd_scale` / `pdd_scope` come from extract_features (may be empty).
    `pdd_scope` format e.g. "01-energy-industries"; we use the leading digits.

    `apply_date_filter` is OFF by default — measured 13/535 (2.4%) GT
    false-negative rate against our test set (PDDs registered when methodology
    was still valid but later withdrawn). Enable only when the date constraint
    is strict regulatory requirement, not a lossy retrieval signal.

    Returns (allowed_codes, diagnostic_counts).
    Empty allowlist guarantees retrieval falls back to full corpus upstream.
    """
    all_stds, versions, status, scopes = load_catalog_indices()
    diag = FilterDiag()
    codes = list(corpus_codes)
    diag.n_total_codes = len(codes)

    # Stage A
    after_a = {c for c in codes if registry_keep(pdd_registry or "", c, all_stds)}
    diag.n_after_stage_a = len(after_a)

    # Stage B.1 — date validity (opt-in, default OFF)
    if apply_date_filter:
        after_date = {c for c in after_a if not is_invalid_at(c, pdd_credit_start, versions, status)}
    else:
        after_date = after_a
    diag.n_after_date = len(after_date)

    # Stage B.2 — AMS-large hard-exclude
    scale_norm = (pdd_scale or "unknown").lower()
    after_scale = {c for c in after_date if not scale_excluded(c, scale_norm)}
    diag.n_after_ams_scale = len(after_scale)

    # Stage B.3 — sectoral scope side-channel (not exclusion)
    scope_code = ""
    if pdd_scope:
        m = re.match(r"0?(\d{1,2})", pdd_scope.strip())
        if m:
            scope_code = m.group(1)
    if scope_code:
        diag.scope_boosted_codes = {c for c in after_scale if scope_code in scopes.get(c, set())}

    return after_scale, diag


# ─── Adapter: codes → clause indices for HybridRetriever ───────────────────

def codes_to_clause_idx(allowed_codes: set[str], meta_df: pd.DataFrame) -> set[int]:
    """Map allowed methodology codes to the row indices in clauses-meta.parquet
    that the HybridRetriever uses (the same int indices it uses for BM25 / HNSW)."""
    if not allowed_codes:
        return set()
    mask = meta_df["code"].isin(allowed_codes)
    return set(int(i) for i in meta_df.index[mask].tolist())
