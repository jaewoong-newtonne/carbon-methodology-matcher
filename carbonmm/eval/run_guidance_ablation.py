"""Ablation: guidance-KG signals on a v281 reranker, over test535.

Conditions (env-gated via graphrag.recommend_v281_*._guidance_flags):
  baseline  : all OFF (must reproduce the published v281 number)
  filter    : V281_GUIDANCE_FILTER=1            (soft scale down-weight in fusion)
  scale     : V281_GUIDANCE_RERANK_SCALE=1      (scale facts + provenance in prompt)
  clauses   : V281_GUIDANCE_RERANK_CLAUSES=1    (retrieved registry clauses in prompt)
  all       : all three

Augments each PDD's meta with extracted capacity (g01-capacity-extracted.json) so
the scale signal has an input; absent capacity → null signal (information ceiling).
Reports overall top1/top5/MRR + two diagnostic splits: capacity-stated-vs-absent and
G01/G02 (scale-sensitive). The C1↔C3 hypothesis: does guidance let OpenAI rerankers
convert recall→top1 (closing the vendor-gated gap)?

Resource note: the graphrag-v281-sonnet baseline issues Claude (Anthropic) API calls;
the OpenAI baselines do not.

Run from repo root:
  python3 -m carbonmm.eval.run_guidance_ablation \
      --baseline graphrag-v281-openai --conditions baseline,all --n 5 --workers 2
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .harness import load_baseline, load_eval_pdds, compute_metrics

logger = logging.getLogger(__name__)

ICDM = Path(__file__).resolve().parents[1]
MANIFEST = ICDM / "data" / "manifests"
RESULTS = ICDM / "results" / "guidance-ablation"

CONDITIONS = {
    "baseline": {},
    "filter": {"V281_GUIDANCE_FILTER": "1"},
    "scale": {"V281_GUIDANCE_RERANK_SCALE": "1"},
    "clauses": {"V281_GUIDANCE_RERANK_CLAUSES": "1"},
    "all": {"V281_GUIDANCE_FILTER": "1", "V281_GUIDANCE_RERANK_SCALE": "1",
            "V281_GUIDANCE_RERANK_CLAUSES": "1"},
}
_FLAG_KEYS = ["V281_GUIDANCE_FILTER", "V281_GUIDANCE_RERANK_SCALE", "V281_GUIDANCE_RERANK_CLAUSES"]


def _load_capacity() -> dict:
    p = MANIFEST / "g01-capacity-extracted.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    rows = d.get("results", d if isinstance(d, list) else [])
    return {r["gid"]: r for r in rows if isinstance(r, dict) and "gid" in r}


def _load_groups() -> dict:
    p = MANIFEST / "method-groups.json"
    return json.loads(p.read_text()).get("code_to_group", {}) if p.exists() else {}


def _set_flags(flags: dict):
    for k in _FLAG_KEYS:
        os.environ.pop(k, None)
    for k, v in flags.items():
        os.environ[k] = v


def _predict_one(b, gid, reg, label, text, top_k, cap_row):
    meta = {"registry": reg}
    if cap_row:
        meta["capacity_value"] = cap_row.get("capacity_mw")
        meta["capacity_conf"] = 1.0 if cap_row.get("stated") else 0.4
    try:
        preds = b.predict(text, top_k=top_k, pdd_meta=meta)
    except Exception as e:
        logger.warning("predict failed gid=%s: %s", gid, e)
        preds = []
    codes = [c for c, _ in preds]
    rank = next((i + 1 for i, c in enumerate(codes) if c == label), None)
    return {"gid": gid, "registry": reg, "gt_label": label,
            "predicted_top5": codes, "gt_rank": rank}


def _split_metrics(per_pdd, cap_map, c2g, top_k):
    stated = [r for r in per_pdd if cap_map.get(r["gid"], {}).get("capacity_mw") is not None]
    absent = [r for r in per_pdd if cap_map.get(r["gid"], {}).get("capacity_mw") is None]
    g0102 = [r for r in per_pdd if c2g.get(r["gt_label"]) in ("G01", "G02")]
    out = {}
    if stated:
        out["capacity_stated"] = compute_metrics(stated, top_k)
    if absent:
        out["capacity_absent"] = compute_metrics(absent, top_k)  # the information ceiling
    if g0102:
        out["groups_G01_G02"] = compute_metrics(g0102, top_k)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", required=True, choices=["graphrag-v281-openai", "graphrag-v281-sonnet"])
    p.add_argument("--conditions", default="baseline,all", help="comma list from: " + ",".join(CONDITIONS))
    p.add_argument("--n", type=int, default=None, help="cap PDDs (smoke)")
    p.add_argument("--split", default="test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out-subdir", default="",
                   help="per-model subdir under results/guidance-ablation/ "
                        "(avoids clobber across models that share a --baseline, e.g. the "
                        "clean 6-model sweep where 4 vendors share graphrag-v281-sonnet)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    bad = [c for c in conds if c not in CONDITIONS]
    if bad:
        raise SystemExit(f"unknown conditions: {bad}; choose from {list(CONDITIONS)}")

    cap_map = _load_capacity()
    c2g = _load_groups()
    pdds = load_eval_pdds(args.n, args.seed, split=args.split)
    logger.info("Loaded %d PDDs (%d with capacity) | conditions=%s | baseline=%s",
                len(pdds), sum(1 for g, *_ in pdds if cap_map.get(g, {}).get("capacity_mw") is not None),
                conds, args.baseline)

    outdir = RESULTS / (args.out_subdir or args.baseline)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for cond in conds:
        _set_flags(CONDITIONS[cond])
        logger.info("── condition=%s flags=%s ──", cond, {k: os.environ.get(k) for k in _FLAG_KEYS})
        b = load_baseline(args.baseline)  # re-instantiate per condition (clean state)
        per = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_predict_one, b, g, r, l, t, args.top_k, cap_map.get(g)): g
                    for g, r, l, t, _m in pdds}
            for f in as_completed(futs):
                rec = f.result()
                per[rec["gid"]] = rec
        per_pdd = [per[g] for g, *_ in pdds if g in per]
        metrics = compute_metrics(per_pdd, args.top_k)
        splits = _split_metrics(per_pdd, cap_map, c2g, args.top_k)
        res = {"baseline": args.baseline, "condition": cond, "flags": CONDITIONS[cond],
               "n_actual": len(per_pdd), "seed": args.seed, "split": args.split,
               "metrics": metrics, "splits": splits, "per_pdd": per_pdd}
        (outdir / f"{cond}.json").write_text(json.dumps(res, indent=2))
        summary[cond] = {"top1": metrics["top1"], "top5": metrics["top5"], "mrr": metrics["mrr"],
                         **{f"{k}.top1": v["top1"] for k, v in splits.items()}}
        logger.info("  %s: top1=%.3f top5=%.3f mrr=%.3f", cond, metrics["top1"], metrics["top5"], metrics["mrr"])
    _set_flags({})  # restore baseline env
    print("\n=== ablation summary ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
