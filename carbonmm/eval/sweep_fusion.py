"""GraphRAG fusion-weight hyperparameter sweep on val split.

Sweeps over (alpha, beta, gamma) fusion config and selects best by top-1.
50 val PDDs stratified GS / VCS (15 / 35).

The inference-transport response cache ensures the extract_features call is shared
across configs (same prompt per PDD), so only the final generation prompt
varies — keeping wall-time tractable.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

ICDM_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ICDM_ROOT / "results"

# 4 fusion configs for the sweep
FUSION_CONFIGS = [
    {"name": "default",     "alpha": 0.3, "beta": 0.5, "gamma": 0.2},
    {"name": "bm25_heavy",  "alpha": 0.5, "beta": 0.3, "gamma": 0.2},
    {"name": "graph_heavy", "alpha": 0.3, "beta": 0.3, "gamma": 0.4},
    {"name": "graph_off",   "alpha": 0.5, "beta": 0.5, "gamma": 0.0},
]


def _stratified_val_pdds(n_gs: int, n_vcs: int, seed: int):
    """Pull stratified val PDDs from the restricted manifest."""
    from .harness import load_eval_pdds, EVAL_PDD_ROOT, RESTRICTED_MANIFEST
    import random

    m = json.loads(RESTRICTED_MANIFEST.read_text())
    gids = []
    if "globalIds" in m and isinstance(m["globalIds"], list):
        gids = m["globalIds"]
    else:
        for cls_ids in m.get("by_class_globalIds", {}).values():
            gids.extend(cls_ids)

    rng = random.Random(seed)
    rng.shuffle(gids)

    gs_pdds, vcs_pdds = [], []
    for gid in gids:
        reg = "GS" if gid.startswith("GS") else "VCS" if gid.startswith("VCS") else None
        if reg is None:
            continue
        path = EVAL_PDD_ROOT / reg / gid / f"{gid}.body.json"
        if not path.exists():
            continue
        try:
            d = json.loads(path.read_text())
        except Exception:
            continue
        if not d.get("methodology_label") or not d.get("full_text"):
            continue
        rec = (gid, reg, d["methodology_label"], d["full_text"])
        if reg == "GS" and len(gs_pdds) < n_gs:
            gs_pdds.append(rec)
        elif reg == "VCS" and len(vcs_pdds) < n_vcs:
            vcs_pdds.append(rec)
        if len(gs_pdds) >= n_gs and len(vcs_pdds) >= n_vcs:
            break
    return gs_pdds + vcs_pdds


def _eval_config(retriever, pdds, fusion_cfg) -> dict:
    """Run GraphRAG with a specific fusion config on `pdds`."""
    from ..graphrag.recommend import GraphRAGBaseline
    from ..graphrag.score_fusion import FusionConfig

    cfg = FusionConfig(
        alpha=fusion_cfg["alpha"],
        beta=fusion_cfg["beta"],
        gamma=fusion_cfg["gamma"],
    )
    b = GraphRAGBaseline(retriever=retriever, fusion=cfg)

    per_pdd = []
    t0 = time.time()
    for gid, reg, label, text, _meta in pdds:
        try:
            preds = b.predict(text, top_k=5)
        except Exception as e:
            logger.warning("predict failed gid=%s err=%s", gid, e)
            preds = []
        codes = [c for c, _ in preds]
        rank = next((i + 1 for i, c in enumerate(codes) if c == label), None)
        per_pdd.append(
            {
                "gid": gid,
                "registry": reg,
                "gt": label,
                "top5": codes,
                "gt_rank": rank,
            }
        )
    elapsed = time.time() - t0

    n = len(per_pdd)
    top1 = sum(1 for r in per_pdd if r["gt_rank"] == 1) / n if n else 0
    top5 = sum(1 for r in per_pdd if r["gt_rank"] is not None and r["gt_rank"] <= 5) / n if n else 0
    mrr = sum(1.0 / r["gt_rank"] for r in per_pdd if r["gt_rank"] is not None) / n if n else 0

    by_reg = {}
    for reg_name in ("GS", "VCS"):
        items = [r for r in per_pdd if r["registry"] == reg_name]
        if not items:
            continue
        nr = len(items)
        by_reg[reg_name] = {
            "n": nr,
            "top1": round(sum(1 for r in items if r["gt_rank"] == 1) / nr, 4),
            "top5": round(sum(1 for r in items if r["gt_rank"] is not None and r["gt_rank"] <= 5) / nr, 4),
            "mrr": round(sum(1.0 / r["gt_rank"] for r in items if r["gt_rank"] is not None) / nr, 4),
        }

    return {
        "config": fusion_cfg,
        "n": n,
        "top1": round(top1, 4),
        "top5": round(top5, 4),
        "mrr": round(mrr, 4),
        "by_registry": by_reg,
        "elapsed_s": round(elapsed, 1),
        "per_pdd": per_pdd,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-gs", type=int, default=15)
    ap.add_argument("--n-vcs", type=int, default=35)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=RESULTS_ROOT / "graphrag-val-sweep.json")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Single retriever instance shared across configs (HNSW + BM25 build is ~2s)
    from ..graphrag.retrieve import HybridRetriever
    logger.info("loading HybridRetriever (shared across configs) ...")
    R = HybridRetriever.load_default()

    logger.info("loading %d GS + %d VCS val PDDs ...", args.n_gs, args.n_vcs)
    pdds = _stratified_val_pdds(args.n_gs, args.n_vcs, args.seed)
    logger.info("loaded %d PDDs", len(pdds))

    all_results = []
    for i, fc in enumerate(FUSION_CONFIGS, 1):
        logger.info("=== sweep %d/%d: %s ===", i, len(FUSION_CONFIGS), fc["name"])
        result = _eval_config(R, pdds, fc)
        all_results.append({"name": fc["name"], **result})
        logger.info(
            "  %s · n=%d · top1=%.3f · top5=%.3f · MRR=%.3f · %.1fs (%.1fs/query)",
            fc["name"], result["n"], result["top1"], result["top5"], result["mrr"],
            result["elapsed_s"], result["elapsed_s"] / max(1, result["n"]),
        )

    # Pick best by top1 (tie-break by MRR)
    best = max(all_results, key=lambda r: (r["top1"], r["mrr"]))
    summary = {
        "n": pdds and len(pdds),
        "seed": args.seed,
        "n_gs": args.n_gs,
        "n_vcs": args.n_vcs,
        "configs": all_results,
        "best": {
            "name": best["name"],
            "config": best["config"],
            "top1": best["top1"],
            "top5": best["top5"],
            "mrr": best["mrr"],
        },
    }
    args.out.write_text(json.dumps(summary, indent=2))

    print("\n=== Sweep summary ===")
    print(f"{'config':14s}  {'top1':>6s}  {'top5':>6s}  {'MRR':>6s}  {'s/q':>6s}")
    for r in all_results:
        marker = " ←" if r["name"] == best["name"] else ""
        print(
            f"{r['name']:14s}  {r['top1']:>6.3f}  {r['top5']:>6.3f}  "
            f"{r['mrr']:>6.3f}  {r['elapsed_s']/max(1,r['n']):>6.1f}{marker}"
        )
    print(f"\nbest: {best['name']}  α={best['config']['alpha']}  β={best['config']['beta']}  γ={best['config']['gamma']}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
