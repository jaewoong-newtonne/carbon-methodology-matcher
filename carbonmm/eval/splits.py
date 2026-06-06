"""Project-isolated train/val/test split.

Stratified split over BM25-scored PDDs (the cleanest pool),
preserving GS:VCS registry ratio. Saved to
`data/manifests/icdm2026-splits.json` so the assignment is frozen and
reproducible across all eval runs.

Split policy:
    val = 100 PDDs (hyperparameter sweep + mini-sweep)
    test = remaining (≈ 535) PDDs
    Stratified by registry (GS / VCS) per restricted manifest's distribution.

For ICDM v1 there is no train split (zero-shot inference; the freq-prior
baseline uses the full retained set's empirical distribution — this is a
mild leak but reasonable since the prior approximates registry-level
issuance statistics and the paper discloses it).
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

ICDM_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ICDM_ROOT / "data" / "manifests"
EVAL_PDD_ROOT = ICDM_ROOT / "data" / "eval-pdds"
RESTRICTED_MANIFEST = MANIFEST_DIR / "pdd-eval-set-restricted.json"
BM25_RESULTS = MANIFEST_DIR / "bm25-rank-sanity-section-a-results.json"
SPLITS_OUT = MANIFEST_DIR / "icdm2026-splits.json"


def _bm25_scored_pdds() -> set[str]:
    """Read the BM25 rank-sanity output to find which PDDs have a usable
    full_text body (the input requirement for our eval). Returns gid set."""
    if not BM25_RESULTS.exists():
        return set()
    d = json.loads(BM25_RESULTS.read_text())
    scored = set()
    for r in d.get("per_pdd", []):
        gid = r.get("globalId") or r.get("gid")
        if gid:
            scored.add(gid)
    return scored


def _retained_pdds_with_body() -> list[tuple[str, str, str]]:
    """[(gid, registry, methodology_label), ...] for PDDs that have body.json
    and a methodology_label, restricted to the BM25-scored set."""
    m = json.loads(RESTRICTED_MANIFEST.read_text())
    gids = []
    if "globalIds" in m and isinstance(m["globalIds"], list):
        gids = m["globalIds"]
    else:
        for cls_ids in m.get("by_class_globalIds", {}).values():
            gids.extend(cls_ids)

    bm25_scored = _bm25_scored_pdds()

    out = []
    for gid in gids:
        reg = "GS" if gid.startswith("GS") else "VCS" if gid.startswith("VCS") else None
        if reg is None:
            continue
        body_path = EVAL_PDD_ROOT / reg / gid / f"{gid}.body.json"
        if not body_path.exists():
            continue
        try:
            d = json.loads(body_path.read_text())
        except Exception:
            continue
        label = d.get("methodology_label")
        text = d.get("full_text") or ""
        if not label or not text:
            continue
        if bm25_scored and gid not in bm25_scored:
            continue  # restrict to BM25-scored if that pool exists
        out.append((gid, reg, label))
    return out


def build_splits(val_size: int, seed: int) -> dict:
    """Stratified val/test split preserving GS:VCS ratio."""
    pdds = _retained_pdds_with_body()
    if not pdds:
        raise RuntimeError("no PDDs available — check retained set + bodies")

    gs = [r for r in pdds if r[1] == "GS"]
    vcs = [r for r in pdds if r[1] == "VCS"]
    n_total = len(pdds)
    logger.info("pool: %d total · GS=%d · VCS=%d", n_total, len(gs), len(vcs))

    gs_ratio = len(gs) / n_total
    n_gs_val = round(val_size * gs_ratio)
    n_vcs_val = val_size - n_gs_val

    rng = random.Random(seed)
    rng.shuffle(gs)
    rng.shuffle(vcs)

    val = gs[:n_gs_val] + vcs[:n_vcs_val]
    test = gs[n_gs_val:] + vcs[n_vcs_val:]
    rng.shuffle(val)
    rng.shuffle(test)

    return {
        "seed": seed,
        "n_total_pool": n_total,
        "val": {
            "n": len(val),
            "by_registry": {
                "GS": sum(1 for _, r, _ in val if r == "GS"),
                "VCS": sum(1 for _, r, _ in val if r == "VCS"),
            },
            "pdds": [
                {"gid": gid, "registry": reg, "gt": label}
                for gid, reg, label in val
            ],
        },
        "test": {
            "n": len(test),
            "by_registry": {
                "GS": sum(1 for _, r, _ in test if r == "GS"),
                "VCS": sum(1 for _, r, _ in test if r == "VCS"),
            },
            "pdds": [
                {"gid": gid, "registry": reg, "gt": label}
                for gid, reg, label in test
            ],
        },
    }


def load_splits(splits_path: Path = SPLITS_OUT) -> dict:
    if not splits_path.exists():
        raise FileNotFoundError(
            f"{splits_path} missing — run `python -m carbonmm.eval.splits` first"
        )
    return json.loads(splits_path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=SPLITS_OUT)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    splits = build_splits(args.val_size, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(splits, indent=2))

    print(f"\nSplits saved: {args.out}")
    print(f"  pool: {splits['n_total_pool']}")
    print(f"  val:  n={splits['val']['n']}  GS={splits['val']['by_registry']['GS']}  "
          f"VCS={splits['val']['by_registry']['VCS']}")
    print(f"  test: n={splits['test']['n']}  GS={splits['test']['by_registry']['GS']}  "
          f"VCS={splits['test']['by_registry']['VCS']}")


if __name__ == "__main__":
    main()
