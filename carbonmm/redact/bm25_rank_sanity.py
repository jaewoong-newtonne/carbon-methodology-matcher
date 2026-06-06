"""BM25 top-K retrieval rank sanity (alternative to the score-drop test).

Measures how far the ground-truth methodology label SLIDES DOWN in BM25
ranking after redaction. This is the metric the paper's main task actually
cares about — if redaction works, the gt label should rank lower in the
redacted query than in the original.

For each redacted PDD:
  query_orig    = tokens(full_text)
  query_redact  = tokens(redacted_text)

  For each methodology code in corpus (610 codes):
    score_orig(code)   = BM25(query_orig,    methodology_doc[code])
    score_redact(code) = BM25(query_redact,  methodology_doc[code])

  rank_orig    = rank position of the ground-truth label in score_orig sort
  rank_redact  = rank position of the ground-truth label in score_redact sort

  "Top-K leak" = ground-truth still in top K after redaction (e.g., K=1 / 10)

Aggregates:
  n_top1_leak    — rank_redact == 1 (ground-truth still ranked #1 → severe leak)
  n_top10_leak   — rank_redact <= 10
  n_safe         — rank_redact > K_threshold (default 10)
  median_rank_drop, avg_rank_drop

Output: data/manifests/bm25-rank-sanity-results.json

Usage:
  python -m carbonmm.redact.bm25_rank_sanity
  python -m carbonmm.redact.bm25_rank_sanity --top-k 10
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from ..ingest.common import DATA_ROOT, MANIFEST_DIR, setup_logging, _utcnow
from .bm25_sanity import BM25, build_methodology_corpus, tokenize, PDD_ROOT, REDACT_ROOT

logger = logging.getLogger(__name__)
RESULTS_OUT_FULL = MANIFEST_DIR / "bm25-rank-sanity-results.json"
RESULTS_OUT_SECTION_A = MANIFEST_DIR / "bm25-rank-sanity-section-a-results.json"
SECTION_A_ROOT = DATA_ROOT / "section-a"
SECTION_A_REDACTED_ROOT = DATA_ROOT / "section-a-redacted"


def rank_of(bm25: BM25, query: list[str], target_idx: int, all_codes_len: int) -> int:
    """Score every doc in the BM25 index against the query, return 1-based rank of target_idx."""
    scores = [(bm25.score(query, i), i) for i in range(all_codes_len)]
    scores.sort(key=lambda x: (-x[0], x[1]))
    for r, (_, i) in enumerate(scores, 1):
        if i == target_idx:
            return r
    return all_codes_len + 1  # not found (shouldn't happen)


def find_redacted_jsons() -> list[Path]:
    if not REDACT_ROOT.exists():
        return []
    return sorted(REDACT_ROOT.rglob("*.redacted.json"))


def find_section_a_redacted_jsons() -> list[Path]:
    if not SECTION_A_REDACTED_ROOT.exists():
        return []
    return sorted(SECTION_A_REDACTED_ROOT.rglob("*.section-a.redacted.json"))


def _load_query_pair(p: Path, input_source: str,
                     label_lookup: dict[str, str]) -> tuple[str, str, str, str] | None:
    """Return (gid, registry, query_orig, query_redact) or None to skip.

    For `redacted_full`: query_orig = full_text from body.json, query_redact =
    redacted_text from `<gid>.redacted.json`.

    For `section_a_pass1`: query_orig = section_a_text (pre Pass-1), query_redact
    = section_a_text_pass1 (post Pass-1). Methodology label is taken from the
    label_lookup dict built from body.json files, since section-a.json doesn't
    carry it.
    """
    r = json.loads(p.read_text())
    gid = r["globalId"]
    reg = r["registry"]
    if input_source == "redacted_full":
        label = r.get("methodology_label")
        if not label:
            return None
        body_path = PDD_ROOT / reg / gid / f"{gid}.body.json"
        if not body_path.exists():
            return None
        body = json.loads(body_path.read_text())
        orig = body.get("full_text", "")
        red = r.get("redacted_text", "")
        if not orig or not red:
            return None
        return gid, reg, orig, red, label
    elif input_source == "section_a_pass1":
        label = label_lookup.get(gid)
        if not label:
            return None
        sa_path = SECTION_A_ROOT / reg / gid / f"{gid}.section-a.json"
        if not sa_path.exists():
            return None
        sa = json.loads(sa_path.read_text())
        orig = sa.get("section_a_text", "")
        red = r.get("section_a_text_pass1", "")
        if not orig or not red:
            return None
        return gid, reg, orig, red, label
    return None


def _build_label_lookup() -> dict[str, str]:
    """Read body.json files for the methodology_label → gid mapping."""
    lookup: dict[str, str] = {}
    if not PDD_ROOT.exists():
        return lookup
    for body_path in PDD_ROOT.rglob("*.body.json"):
        try:
            body = json.loads(body_path.read_text())
            gid = body["globalId"]
            label = body.get("methodology_label")
            if label:
                lookup[gid] = label
        except Exception:
            continue
    return lookup


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=10,
                        help="K for 'top-K leak' detection (default 10).")
    parser.add_argument("--rebuild-corpus", action="store_true")
    parser.add_argument("--max-n", type=int, default=None)
    parser.add_argument("--input-source", choices=["redacted_full", "section_a_pass1"],
                        default="redacted_full",
                        help="redacted_full: legacy full-body Pass-2 redacted PDDs; "
                        "section_a_pass1: Section A only, Pass-1-redacted (new scope).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    # Load + index methodology corpus
    corpus = build_methodology_corpus(use_cache=not args.rebuild_corpus)
    if not corpus:
        logger.error("methodology corpus empty")
        return 1
    codes = sorted(corpus.keys())
    code_idx = {c: i for i, c in enumerate(codes)}
    docs_tokens = [tokenize(corpus[c]) for c in codes]
    bm25 = BM25(docs_tokens)
    logger.info("BM25 index: %d codes, avgdl=%.1f", bm25.N, bm25.avgdl)

    # Source-specific input loaders
    if args.input_source == "section_a_pass1":
        paths = find_section_a_redacted_jsons()
        label_lookup = _build_label_lookup()
        out_path = RESULTS_OUT_SECTION_A
    else:
        paths = find_redacted_jsons()
        label_lookup = {}
        out_path = RESULTS_OUT_FULL

    if args.max_n:
        paths = paths[: args.max_n]
    logger.info("ranking %d %s PDDs", len(paths), args.input_source)

    per_pdd = []
    n_top1_leak = 0
    n_top10_leak = 0
    n_safe = 0
    n_skip = 0
    rank_drops = []

    for p in paths:
        loaded = _load_query_pair(p, args.input_source, label_lookup)
        if loaded is None:
            n_skip += 1
            continue
        gid, reg, orig_text, red_text, label = loaded
        if label not in code_idx:
            n_skip += 1
            continue

        q_orig = tokenize(orig_text)
        q_red = tokenize(red_text)
        target = code_idx[label]
        rank_o = rank_of(bm25, q_orig, target, bm25.N)
        rank_r = rank_of(bm25, q_red, target, bm25.N)
        drop = rank_r - rank_o
        rank_drops.append(drop)

        if rank_r == 1:
            n_top1_leak += 1
        if rank_r <= 10:
            n_top10_leak += 1
        if rank_r > args.top_k:
            n_safe += 1

        per_pdd.append({
            "globalId": gid,
            "registry": reg,
            "label": label,
            "rank_orig": rank_o,
            "rank_redacted": rank_r,
            "rank_drop_positions": drop,
            "in_top1_orig": rank_o == 1,
            "in_top1_redacted": rank_r == 1,
            "in_topK_redacted": rank_r <= args.top_k,
        })

    n_scored = len(per_pdd)
    avg_drop = sum(rank_drops) / max(1, n_scored)
    median_drop = sorted(rank_drops)[len(rank_drops) // 2] if rank_drops else 0
    top1_pct = n_top1_leak / max(1, n_scored)
    top10_pct = n_top10_leak / max(1, n_scored)
    safe_pct = n_safe / max(1, n_scored)

    out = {
        "generated_at": _utcnow(),
        "input_source": args.input_source,
        "top_k_threshold": args.top_k,
        "n_scored": n_scored,
        "n_top1_leak": n_top1_leak,
        "n_top10_leak": n_top10_leak,
        "n_safe": n_safe,
        "n_skipped": n_skip,
        "top1_leak_rate": round(top1_pct, 4),
        "top10_leak_rate": round(top10_pct, 4),
        "safe_rate": round(safe_pct, 4),
        "avg_rank_drop": round(avg_drop, 2),
        "median_rank_drop": median_drop,
        "n_methodologies_in_corpus": bm25.N,
        "per_pdd": per_pdd,
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    logger.info("=== BM25 rank sanity ===")
    logger.info("  scored:           %d", n_scored)
    logger.info("  top-1 leak count: %d (%.1f%%)   ← gt still ranked #1 after redaction",
                n_top1_leak, top1_pct * 100)
    logger.info("  top-10 leak count: %d (%.1f%%)",
                n_top10_leak, top10_pct * 100)
    logger.info("  safe (rank > %d): %d (%.1f%%)",
                args.top_k, n_safe, safe_pct * 100)
    logger.info("  median rank drop: %d positions", median_drop)
    logger.info("  avg rank drop:    %.1f positions", avg_drop)
    logger.info("  corpus size:      %d methodologies", bm25.N)
    logger.info("  → %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
