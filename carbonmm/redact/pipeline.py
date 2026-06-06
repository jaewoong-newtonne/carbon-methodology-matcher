"""ICDM 2026 redaction pipeline.

Reads PDD body JSONs (data/eval-pdds/{REG}/{globalId}/{globalId}.body.json) and
emits redacted-spec JSONs (data/redacted/{REG}/{globalId}/{globalId}.redacted.json)
suitable as input to the GraphRAG retriever's frozen-test evaluation.

Three passes:
  Pass 1  REGEX     — explicit methodology-code masking (~13 patterns)
                      Applied to ALL projects in the manifest.
  Pass 2  LLM       — paragraph-level structured-output verification via
                      the configured inference transport (/verify_redaction).
                      Default: only on paragraphs that Pass 1 already marked
                      (cheaper); --pass2-all for every paragraph.
                      --pass2-sample N for a random N-PDD subset.
  Pass 3  BM25      — sanity score-drop test (separate module: bm25_sanity.py)

Output per project:
  data/redacted/{REG}/{globalId}/{globalId}.redacted.json
    {
      "globalId": "VCS10",
      "registry": "VCS",
      "methodology_label": "ACM0002",
      "secondary_labels": [...],
      "country": "...",
      "source_pdf": "VCS_PD_V04.pdf",
      "page_count": 42,
      "char_count_original": 50312,
      "char_count_redacted": 49890,
      "pass1_n_marks": 5,
      "pass1_marks": [{"pattern": "ACM####", "match": "ACM0002", "offset": 1234}, ...],
      "pass2_applied": true,
      "pass2_n_paragraphs": 47,
      "pass2_n_leaks_found": 3,
      "pass2_marks": [{"paragraph_idx": 12, "evidence": "...", "before": "...", "after": "..."}, ...],
      "redacted_text": "...",
      "redacted_at": "..."
    }

Usage:
    # Default: Pass1 all + Pass2 on regex-marked paragraphs only
    python -m carbonmm.redact.pipeline

    # Pass2 on a random 20-PDD sample (full paragraph verification)
    python -m carbonmm.redact.pipeline --pass2-sample 20

    # Maximum thoroughness: Pass2 on every paragraph of every PDD (~60K calls)
    python -m carbonmm.redact.pipeline --pass2-all

    # Smoke
    python -m carbonmm.redact.pipeline --max-n 5 --pass2-sample 5
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from ..ingest.common import DATA_ROOT, setup_logging, _utcnow

logger = logging.getLogger(__name__)

PDD_ROOT = DATA_ROOT / "eval-pdds"
REDACT_ROOT = DATA_ROOT / "redacted"


# ─── Pass 1: regex patterns ───────────────────────────────────────────────
# Patterns live in `patterns.py` (single source of truth — also loaded by
# audit/build_kappa_pool.py via importlib for full BEFORE reconstruction).

from .patterns import REPLACEMENT, REDACTION_PATTERNS  # noqa: E402,F401


@dataclass
class Pass1Mark:
    pattern_name: str
    match: str
    offset: int


def pass1_regex(text: str) -> tuple[str, list[Pass1Mark]]:
    """Apply regex masking. Returns (redacted_text, marks_found).

    Marks are collected before any substitution so offsets refer to the
    ORIGINAL text. Substitution happens in pattern order (most-specific first).
    """
    marks: list[Pass1Mark] = []
    for pname, pat in REDACTION_PATTERNS:
        for m in pat.finditer(text):
            marks.append(Pass1Mark(pname, m.group(0), m.start()))
    # Now substitute. Same patterns, applied left-to-right per pattern.
    redacted = text
    for _, pat in REDACTION_PATTERNS:
        redacted = pat.sub(REPLACEMENT, redacted)
    return redacted, marks


# ─── Pass 2: LLM verifier via the configured inference transport ─────────────

@dataclass
class Pass2Mark:
    paragraph_idx: int
    evidence: str
    before: str
    after: str


def split_paragraphs(text: str) -> list[str]:
    """Naive paragraph split on blank lines. Drops trivial fragments (<30 chars)
    which are typically page-break artifacts or single-token noise."""
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    return [p for p in paras if len(p) >= 30]


def pass2_llm(text: str, daemon_url: str, only_marked_paragraphs: bool,
               pass1_marks: list[Pass1Mark], timeout_s: float = 60.0
               ) -> tuple[str, list[Pass2Mark]]:
    """Call /verify_redaction per paragraph. Returns (redacted_text, marks).

    If only_marked_paragraphs=True, skip paragraphs that had no Pass 1 marks
    (efficient default — those are unlikely to contain methodology refs).
    """
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return text, []

    # Build set of paragraph indices that overlap with Pass 1 marks (by offset)
    marked_para_idx: set[int] = set()
    if only_marked_paragraphs and pass1_marks:
        offsets = sorted(m.offset for m in pass1_marks)
        # Walk paragraphs and check if any pass1 mark offset falls within
        cursor = 0
        for i, p in enumerate(paragraphs):
            try:
                start = text.index(p, cursor)
            except ValueError:
                continue
            end = start + len(p)
            for o in offsets:
                if start <= o <= end:
                    marked_para_idx.add(i)
            cursor = end

    out_paragraphs = list(paragraphs)  # mutable copy
    marks: list[Pass2Mark] = []
    with httpx.Client(timeout=timeout_s) as client:
        for i, p in enumerate(paragraphs):
            if only_marked_paragraphs and i not in marked_para_idx:
                continue
            try:
                r = client.post(f"{daemon_url}/verify_redaction",
                                json={"paragraph": p})
                r.raise_for_status()
                resp = r.json()
            except Exception as e:
                logger.warning("[para=%d] daemon call failed: %s", i, str(e)[:120])
                continue
            if resp.get("leaks") and resp.get("evidence"):
                marks.append(Pass2Mark(
                    paragraph_idx=i,
                    evidence=resp["evidence"],
                    before=p[:200],
                    after=resp["redacted_paragraph"][:200],
                ))
                out_paragraphs[i] = resp["redacted_paragraph"]

    # Reassemble: split original by `\n\n`, join with redacted versions
    # back into the same position. Use paragraphs list (subset of all blocks)
    # — anything that was filtered (< 30 chars) stays unchanged.
    # Simpler: walk original text, replace each occurrence of paragraphs[i]
    # by out_paragraphs[i] in order.
    redacted_text = text
    cursor = 0
    for orig, new in zip(paragraphs, out_paragraphs):
        if orig == new:
            continue
        idx = redacted_text.find(orig, cursor)
        if idx == -1:
            continue
        redacted_text = redacted_text[:idx] + new + redacted_text[idx + len(orig):]
        cursor = idx + len(new)
    return redacted_text, marks


# ─── Orchestrator ─────────────────────────────────────────────────────────

def find_body_jsons(registry: str | None = None) -> list[Path]:
    base = PDD_ROOT
    if registry:
        base = base / registry
    if not base.exists():
        return []
    return sorted(base.rglob("*.body.json"))


def redact_one(body_path: Path, daemon_url: str | None,
                pass2_mode: str, sample_paragraph_only: bool) -> dict | None:
    body = json.loads(body_path.read_text())
    text = body.get("full_text", "")
    if not text:
        return None

    n_orig = len(text)
    p1_text, p1_marks = pass1_regex(text)

    do_pass2 = pass2_mode in ("on", "sample", "full")
    p2_marks: list[Pass2Mark] = []
    p2_text = p1_text
    n_paragraphs = 0
    if do_pass2 and daemon_url:
        n_paragraphs = len(split_paragraphs(p1_text))
        p2_text, p2_marks = pass2_llm(
            p1_text, daemon_url,
            only_marked_paragraphs=sample_paragraph_only,
            pass1_marks=p1_marks,
        )

    return {
        "globalId": body.get("globalId"),
        "registry": body.get("registry"),
        "methodology_label": body.get("methodology_label"),
        "secondary_labels": body.get("secondary_labels") or [],
        "country": body.get("country"),
        "source_pdf": body.get("source_pdf"),
        "page_count": body.get("page_count"),
        "char_count_original": n_orig,
        "char_count_redacted": len(p2_text),
        "pass1_n_marks": len(p1_marks),
        "pass1_marks": [
            {"pattern": m.pattern_name, "match": m.match, "offset": m.offset}
            for m in p1_marks[:200]  # cap to keep JSON size sane
        ],
        "pass2_applied": do_pass2,
        "pass2_paragraph_only_marked": sample_paragraph_only,
        "pass2_n_paragraphs": n_paragraphs,
        "pass2_n_leaks_found": len(p2_marks),
        "pass2_marks": [
            {"paragraph_idx": m.paragraph_idx, "evidence": m.evidence,
             "before": m.before, "after": m.after}
            for m in p2_marks
        ],
        "redacted_text": p2_text,
        "redacted_at": _utcnow(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=None,
                        help="Restrict to one registry (GS / VCS).")
    parser.add_argument("--max-n", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--daemon-url", default=os.environ.get(
        "CLAUDE_DAEMON_URL", "http://localhost:8765"
    ))
    # Pass 2 modes
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--pass2-off", action="store_true",
                   help="Regex only; skip LLM verify entirely.")
    g.add_argument("--pass2-marked-only", action="store_true",
                   help="LLM verify only paragraphs that had Pass 1 marks (default).")
    g.add_argument("--pass2-all", action="store_true",
                   help="LLM verify EVERY paragraph (slow but most thorough).")
    g.add_argument("--pass2-sample", type=int, default=None,
                   help="Apply --pass2-all but only on a random N-PDD subset (e.g. N=20).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    # Resolve pass2 mode
    if args.pass2_off:
        pass2_mode = "off"
        sample_only = False
    elif args.pass2_all:
        pass2_mode = "on"
        sample_only = False
    elif args.pass2_sample:
        pass2_mode = "sample"
        sample_only = False
    else:
        pass2_mode = "on"
        sample_only = True  # default

    body_paths = find_body_jsons(args.registry)
    if args.max_n:
        body_paths = body_paths[: args.max_n]
    logger.info("Planned %d PDDs (pass2_mode=%s, marked_only=%s, daemon=%s)",
                len(body_paths), pass2_mode, sample_only, args.daemon_url)

    # Drop PDDs whose body.json has empty full_text (layout service failure cases).
    # Otherwise sample mode would pick these as "fail" and waste sample slots.
    before_valid = len(body_paths)
    valid_paths = []
    for p in body_paths:
        try:
            b = json.loads(p.read_text())
            if (b.get("full_text") or "").strip():
                valid_paths.append(p)
        except Exception:
            pass
    if len(valid_paths) < before_valid:
        logger.info("filter empty-body: %d → %d (drop %d empty-full_text PDDs)",
                    before_valid, len(valid_paths), before_valid - len(valid_paths))
    body_paths = valid_paths

    if args.skip_existing:
        before = len(body_paths)
        body_paths = [
            p for p in body_paths
            if not (REDACT_ROOT / p.parent.parent.name / p.parent.name /
                    f"{p.stem.replace('.body','')}.redacted.json").exists()
        ]
        logger.info("--skip-existing: %d → %d", before, len(body_paths))

    # Sample mode: pick N at random
    if pass2_mode == "sample":
        n = args.pass2_sample
        if n < len(body_paths):
            sampled = set(random.Random(42).sample(range(len(body_paths)), n))
        else:
            sampled = set(range(len(body_paths)))
        logger.info("Sample mode: %d PDDs get full Pass 2", len(sampled))
    else:
        sampled = None

    n_ok = 0
    n_fail = 0
    for i, body_path in enumerate(body_paths):
        try:
            this_pass2 = pass2_mode
            this_sample_only = sample_only
            if pass2_mode == "sample":
                if i in sampled:
                    this_pass2 = "on"
                    this_sample_only = False  # full Pass 2 on this PDD
                else:
                    this_pass2 = "off"

            t0 = time.time()
            result = redact_one(body_path, args.daemon_url, this_pass2, this_sample_only)
            if result is None:
                n_fail += 1
                continue
            out_dir = REDACT_ROOT / result["registry"] / result["globalId"]
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{result['globalId']}.redacted.json"
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
            n_ok += 1
            dt = time.time() - t0
            logger.info(
                "  ✓ %s/%s | pass1=%d marks  pass2=%s/%d para/%d leaks  %.1fs",
                result["registry"], result["globalId"],
                result["pass1_n_marks"],
                this_pass2,
                result["pass2_n_paragraphs"],
                result["pass2_n_leaks_found"],
                dt,
            )
        except Exception as e:
            logger.error("[%s] %s", body_path, e)
            n_fail += 1

    logger.info("Done. ok=%d fail=%d total=%d", n_ok, n_fail, len(body_paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
