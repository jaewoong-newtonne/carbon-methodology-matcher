"""Batch-extract the 5 structured PDD features for the test split via the
configured inference transport, caching to data/manifests/pdd-extracted-features-test.json.

Feeds: residual_ambiguity Level C (scale conditioning) + the within-group
disambiguator. Resumable: re-running skips gids already in the cache.

Example (long-running, backgrounded):
    nohup python eval/dump_features.py \\
        --split test --workers 3 > /tmp/dump-features-test.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ICDM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ICDM_ROOT / "graphrag"))
from extract_features import extract_features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("dump_features")

SPLITS = ICDM_ROOT / "data" / "manifests" / "icdm2026-splits.json"
EVAL_PDD_ROOT = ICDM_ROOT / "data" / "eval-pdds"


def load_pdds(split: str) -> list[tuple[str, str]]:
    """[(gid, full_text)] for the split."""
    pdds = json.loads(SPLITS.read_text())[split]["pdds"]
    out = []
    for p in pdds:
        gid, reg = p["gid"], p["registry"]
        body = EVAL_PDD_ROOT / reg / gid / f"{gid}.body.json"
        if not body.exists():
            continue
        try:
            text = json.loads(body.read_text()).get("full_text") or ""
        except Exception:
            text = ""
        if text:
            out.append((gid, text))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--out", type=Path,
                    default=ICDM_ROOT / "data" / "manifests" / "pdd-extracted-features-test.json")
    args = ap.parse_args()

    cache: dict[str, dict] = {}
    if args.out.exists():
        cache = json.loads(args.out.read_text())
        logger.info("resuming: %d cached", len(cache))

    pdds = [(g, t) for g, t in load_pdds(args.split) if g not in cache]
    logger.info("to extract: %d (split=%s, workers=%d, model=%s)",
                len(pdds), args.split, args.workers, args.model)

    done = 0

    def work(item):
        gid, text = item
        try:
            f = extract_features(text, model=args.model)
            d = f.to_dict()
            d.pop("raw_response", None)
            return gid, d
        except Exception as e:
            return gid, {"error": str(e)[:200]}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, it) for it in pdds]
        for fut in as_completed(futs):
            gid, d = fut.result()
            cache[gid] = d
            done += 1
            if done % 25 == 0:
                args.out.write_text(json.dumps(cache, indent=2))
                logger.info("checkpoint %d/%d (last %s scale=%s)",
                            done, len(pdds), gid, d.get("scale"))

    args.out.write_text(json.dumps(cache, indent=2))
    n_err = sum(1 for v in cache.values() if "error" in v)
    n_scale = sum(1 for v in cache.values() if v.get("scale") in {"small", "large"})
    logger.info("DONE: %d cached, %d errors, %d with concrete scale -> %s",
                len(cache), n_err, n_scale, args.out)


if __name__ == "__main__":
    main()
