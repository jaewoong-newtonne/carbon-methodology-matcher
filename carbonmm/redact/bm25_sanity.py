"""Pass 3 of redaction — BM25 sanity-check.

For each PDD that has been redacted, measure the BM25 score drop between
the ORIGINAL full_text and the REDACTED text, scored against a corpus of
methodology-clause documents (one per carbon-methodology code in the KG).

Target: average score drop ≥ 50% on the ground-truth label's methodology
document. This proves the redaction actually masks the methodology-identifying
signal, not just the literal code.

Input:
  data/eval-pdds/{REG}/{globalId}/{globalId}.body.json   (original body text)
  data/redacted/{REG}/{globalId}/{globalId}.redacted.json (redacted output)

Methodology corpus:
  Queried from climate_kg TypeDB (carbon-methodology + methodology-clause)
  Cached to data/manifests/bm25-methodology-corpus.json on first run.

Output:
  data/manifests/bm25-sanity-results.json
  {
    "pass_count": 873,
    "fail_count": 287,
    "avg_drop": 0.62,
    "median_drop": 0.71,
    "per_pdd": [
      {"globalId": "VCS10", "label": "ACM0002", "score_orig": 12.4, "score_redacted": 3.1, "drop": 0.75, "passed": true},
      ...
    ]
  }

Usage:
  python -m carbonmm.redact.bm25_sanity
  python -m carbonmm.redact.bm25_sanity --drop-threshold 0.5
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

from ..ingest.common import DATA_ROOT, MANIFEST_DIR, setup_logging, _utcnow

logger = logging.getLogger(__name__)

PDD_ROOT = DATA_ROOT / "eval-pdds"
REDACT_ROOT = DATA_ROOT / "redacted"
CORPUS_CACHE = MANIFEST_DIR / "bm25-methodology-corpus.json"
RESULTS_OUT = MANIFEST_DIR / "bm25-sanity-results.json"


# ─── Tokenizer ────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]{2,}")
# Very common English words that add noise to BM25
_STOPWORDS = {
    "the", "and", "for", "are", "with", "that", "this", "from", "have", "has",
    "been", "shall", "will", "may", "can", "must", "should", "would", "could",
    "any", "all", "such", "these", "those", "their", "there", "they", "them",
    "which", "where", "when", "what", "who", "whose", "while",
    "into", "onto", "upon", "over", "under", "between", "through", "during",
    "above", "below", "without", "within", "against", "before", "after",
    "but", "not", "nor", "only", "also", "however", "therefore", "thus",
    "more", "most", "less", "least", "much", "some", "each", "every", "both",
    "other", "another", "either", "neither", "same", "different", "new", "old",
    "one", "two", "three", "four", "five",
    "year", "years", "day", "days", "month", "months",
    "table", "figure", "section", "annex", "appendix", "page",
}


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if t.lower() not in _STOPWORDS]


# ─── Minimal BM25 implementation (no external dep) ────────────────────────

class BM25:
    """Plain BM25 over a small static corpus (Robertson/Sparck-Jones, k1=1.5, b=0.75).
    No external dependency — rank_bm25 would work but we want offline guarantee."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.doc_lens = [len(d) for d in docs]
        self.avgdl = sum(self.doc_lens) / max(1, self.N)
        self.df: Counter = Counter()
        for d in docs:
            for term in set(d):
                self.df[term] += 1
        self.idf: dict[str, float] = {}
        for term, df in self.df.items():
            # IDF with +1 smoothing to avoid negatives
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
        # Pre-compute term-frequency tables per doc
        self.tf: list[Counter] = [Counter(d) for d in docs]

    def score(self, query: list[str], doc_idx: int) -> float:
        if doc_idx >= self.N:
            return 0.0
        tf = self.tf[doc_idx]
        dl = self.doc_lens[doc_idx]
        s = 0.0
        for q in query:
            if q not in tf:
                continue
            idf = self.idf.get(q, 0.0)
            f = tf[q]
            num = f * (self.k1 + 1)
            denom = f + self.k1 * (1 - self.b + self.b * dl / max(1, self.avgdl))
            s += idf * num / denom
        return s


# ─── Methodology corpus loader ────────────────────────────────────────────

def build_methodology_corpus(use_cache: bool = True) -> dict[str, str]:
    """Return {methodology_code: concatenated_clause_text}. Queries TypeDB
    climate_kg for all carbon-methodology + their methodology-clause relations
    on first call; cached to JSON afterwards."""
    if use_cache and CORPUS_CACHE.exists():
        logger.info("loading cached methodology corpus: %s", CORPUS_CACHE)
        return json.loads(CORPUS_CACHE.read_text())

    logger.info("querying climate_kg for methodology corpus...")
    try:
        from typedb.driver import TypeDB, Credentials, DriverOptions, TransactionType
    except ImportError:
        raise RuntimeError("typedb-driver not installed")

    pw = os.environ.get("TYPEDB_PASSWORD") or _try_read_env_file()
    if not pw:
        raise RuntimeError("TYPEDB_PASSWORD not available (set the env var or an ENV_FILE)")

    host = os.environ.get("TYPEDB_HOST", "localhost:1729")
    creds = Credentials(os.environ.get("TYPEDB_USER", "admin"), pw)
    opts = DriverOptions(is_tls_enabled=False)
    db = os.environ.get("TYPEDB_DB", "climate_kg")

    corpus: dict[str, list[str]] = {}
    with TypeDB.driver(f"typedb://{host}", creds, opts) as d:
        with d.transaction(db, TransactionType.READ) as tx:
            q = (
                'match $m isa carbon-methodology, has code $c; '
                '(method: $m, clause: $cl) isa clause-of; '
                '$cl has clause-text $t; '
                'select $c, $t;'
            )
            answers = list(tx.query(q).resolve())
            logger.info("got %d (code, clause) pairs", len(answers))
            for row in answers:
                try:
                    code = row.get("c").as_attribute().get_value()
                    text = row.get("t").as_attribute().get_value()
                    corpus.setdefault(code, []).append(text)
                except Exception as e:
                    logger.debug("row parse failed: %s", e)

    out = {code: " ".join(texts) for code, texts in corpus.items()}
    CORPUS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_CACHE.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info("cached %d methodology docs to %s", len(out), CORPUS_CACHE)
    return out


def _try_read_env_file() -> str | None:
    path = Path(os.environ.get("ENV_FILE", ".env"))
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.startswith("TYPEDB_PASSWORD="):
            return line.split("=", 1)[1].strip()
    return None


# ─── Main: score per PDD ──────────────────────────────────────────────────

def find_redacted_jsons() -> list[Path]:
    if not REDACT_ROOT.exists():
        return []
    return sorted(REDACT_ROOT.rglob("*.redacted.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drop-threshold", type=float, default=0.5,
                        help="Pass threshold for score-drop (default: 0.5 = 50%%)")
    parser.add_argument("--rebuild-corpus", action="store_true",
                        help="Re-query TypeDB instead of using cached corpus.")
    parser.add_argument("--max-n", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    # 1. Load methodology corpus
    corpus_text = build_methodology_corpus(use_cache=not args.rebuild_corpus)
    if not corpus_text:
        logger.error("methodology corpus is empty — climate_kg query returned nothing")
        return 1

    # 2. Build BM25 index over methodology docs (one doc = one code)
    codes = sorted(corpus_text.keys())
    docs_tokens = [tokenize(corpus_text[c]) for c in codes]
    bm25 = BM25(docs_tokens)
    code_idx = {c: i for i, c in enumerate(codes)}
    logger.info("BM25 index built: %d methodology docs, avgdl=%.1f", bm25.N, bm25.avgdl)

    # 3. Walk redacted JSONs + score
    redacted_paths = find_redacted_jsons()
    if args.max_n:
        redacted_paths = redacted_paths[: args.max_n]
    logger.info("scoring %d redacted PDDs", len(redacted_paths))

    per_pdd = []
    n_pass = n_fail = n_skip = 0
    drops_pass = []
    for p in redacted_paths:
        r = json.loads(p.read_text())
        label = r.get("methodology_label")
        if not label or label not in code_idx:
            n_skip += 1
            continue
        # Find original body.json
        gid = r["globalId"]
        reg = r["registry"]
        body_path = PDD_ROOT / reg / gid / f"{gid}.body.json"
        if not body_path.exists():
            n_skip += 1
            continue
        body = json.loads(body_path.read_text())
        full_text = body.get("full_text", "")
        redacted_text = r.get("redacted_text", "")
        if not full_text or not redacted_text:
            n_skip += 1
            continue

        q_orig = tokenize(full_text)
        q_red = tokenize(redacted_text)
        s_orig = bm25.score(q_orig, code_idx[label])
        s_red = bm25.score(q_red, code_idx[label])
        drop = (s_orig - s_red) / s_orig if s_orig > 0 else 0.0
        passed = drop >= args.drop_threshold

        per_pdd.append({
            "globalId": gid,
            "registry": reg,
            "methodology_label": label,
            "score_orig": round(s_orig, 3),
            "score_redacted": round(s_red, 3),
            "drop": round(drop, 4),
            "passed": passed,
        })
        if passed:
            n_pass += 1
            drops_pass.append(drop)
        else:
            n_fail += 1

    drops_all = [r["drop"] for r in per_pdd if r["score_orig"] > 0]
    avg = sum(drops_all) / len(drops_all) if drops_all else 0.0
    median = sorted(drops_all)[len(drops_all) // 2] if drops_all else 0.0

    out = {
        "generated_at": _utcnow(),
        "drop_threshold": args.drop_threshold,
        "n_scored": len(per_pdd),
        "n_passed": n_pass,
        "n_failed": n_fail,
        "n_skipped": n_skip,
        "avg_drop": round(avg, 4),
        "median_drop": round(median, 4),
        "pass_rate": round(n_pass / max(1, len(per_pdd)), 4),
        "per_pdd": per_pdd,
    }
    RESULTS_OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info("=== BM25 sanity test ===")
    logger.info("  scored:  %d", out["n_scored"])
    logger.info("  passed:  %d  (drop ≥ %.0f%%)", n_pass, args.drop_threshold * 100)
    logger.info("  failed:  %d", n_fail)
    logger.info("  skipped: %d (no label / no body)", n_skip)
    logger.info("  avg drop: %.1f%%   median: %.1f%%", avg * 100, median * 100)
    logger.info("  pass rate: %.1f%%", out["pass_rate"] * 100)
    logger.info("  → %s", RESULTS_OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
