"""Leakage gate for rich eval text: GT-code re-scan (must be 0) + optional BM25
rank-1 sanity (must be > 1, i.e. the correct methodology is NOT retrieved at rank 1)."""
from __future__ import annotations
import importlib.util
import pathlib
import re
import sys

# sys.path guard: ensure repo root is importable so sibling packages resolve.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]  # .../the project root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Load redact.patterns via spec_from_file_location to bypass the hyphen in
# the package directory name ("carbonmm/"), which makes
# importlib.import_module("carbonmm.redact.patterns") invalid.
_REDACT_PAT_PATH = pathlib.Path(__file__).resolve().parents[1] / "redact" / "patterns.py"
_spec = importlib.util.spec_from_file_location("_redact_patterns", _REDACT_PAT_PATH)
_pat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pat)

# REDACTION_PATTERNS is imported here to confirm redact.patterns resolves and
# to make the patterns available for future BM25/pattern reuse in this module.
_REDACTION_PATTERNS = _pat.REDACTION_PATTERNS

_CODE = re.compile(
    r"AR-A?CM\d{3,4}|ACM\d{3,4}|AM\d{3,4}|AMS-[IVX]+\.?[A-Z]?\.?"
    r"|VMR\d{3,4}|VMD\d{3,4}|VM\d{3,4}|GS-[A-Z]{2,}-?\d{2,3}|TPDDTEC", re.I)


def _gt_code(gt_label: str) -> str:
    m = _CODE.search(gt_label or "")
    return m.group(0) if m else (gt_label or "").strip()


def code_leak_count(text: str, gt_label: str) -> int:
    code = _gt_code(gt_label)
    if not code or len(code) < 4:
        return 0
    return len(re.findall(re.escape(code), text, re.I))


def any_code_count(text: str) -> int:
    return len(_CODE.findall(text))


def gate_text(text: str, gt_label: str, bm25_rank1: int | None,
              bm25_rank1_threshold: int = 1) -> dict:
    """passes = (no literal GT code) and (if a bm25 rank is supplied, it is > threshold)."""
    leaks = code_leak_count(text, gt_label)
    rank_ok = True if bm25_rank1 is None else (bm25_rank1 > bm25_rank1_threshold)
    return {"code_leaks": leaks, "any_codes": any_code_count(text),
            "bm25_rank1": bm25_rank1, "passes": leaks == 0 and rank_ok}
