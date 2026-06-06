"""Frozen-test-style eval harness for ICDM 2026 baselines (and later GraphRAG).

During development this runs on a VALIDATION subset only — the test split
is reserved for the frozen final evaluation. Smoke usage:

    python -m carbonmm.eval.harness \\
        --baseline naive-rag --n 20 --seed 42

Outputs results JSON at:
    carbonmm/results/{baseline}-{split}-n{N}-seed{seed}.json

Metrics:
    top1, top5, MRR (mean reciprocal rank, capped at top_k)
    per-registry breakdown (GS / VCS)
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import random
import time
from pathlib import Path

logger = logging.getLogger(__name__)

ICDM_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ICDM_ROOT / "results"
EVAL_PDD_ROOT = ICDM_ROOT / "data" / "eval-pdds"
RESTRICTED_MANIFEST = ICDM_ROOT / "data" / "manifests" / "pdd-eval-set-restricted.json"

BASELINE_REGISTRY = {
    "random": ("carbonmm.baselines.random_baseline", "RandomBaseline"),
    "freq-prior": ("carbonmm.baselines.freq_prior", "FreqPriorBaseline"),
    "bm25-only": ("carbonmm.baselines.bm25_only", "BM25OnlyBaseline"),
    "dense-knn": ("carbonmm.baselines.dense_knn", "DenseKNNBaseline"),
    "naive-rag": ("carbonmm.baselines.naive_rag", "NaiveRAGBaseline"),
    "naive-rag-sonnet": ("carbonmm.baselines.naive_rag_sonnet", "NaiveRAGSonnetBaseline"),
    "naive-rag-openai": ("carbonmm.baselines.naive_rag_openai", "NaiveRAGOpenAIBaseline"),
    "graphrag": ("carbonmm.graphrag.recommend", "GraphRAGBaseline"),
    "graphrag-v2": ("carbonmm.graphrag.recommend_v2", "GraphRAGv2Baseline"),
    "graphrag-v3": ("carbonmm.graphrag.recommend_v3", "GraphRAGv3Baseline"),
    "graphrag-fusion-only": ("carbonmm.graphrag.recommend_fusion_only", "GraphRAGFusionOnlyBaseline"),
    "graphrag-v25": ("carbonmm.graphrag.recommend_v25", "GraphRAGv25Baseline"),
    "graphrag-v26": ("carbonmm.graphrag.recommend_v26", "GraphRAGv26Baseline"),
    "graphrag-v27": ("carbonmm.graphrag.recommend_v27", "GraphRAGv27Baseline"),
    "graphrag-v25-sonnet": ("carbonmm.graphrag.recommend_v25_sonnet", "GraphRAGv25SonnetBaseline"),
    "graphrag-v28": ("carbonmm.graphrag.recommend_v28", "GraphRAGv28Baseline"),
    "graphrag-v28-openai": ("carbonmm.graphrag.recommend_v28_openai", "GraphRAGv28OpenAIBaseline"),
    "graphrag-v281-openai": ("carbonmm.graphrag.recommend_v281_openai", "GraphRAGv281OpenAIBaseline"),
    "graphrag-v281-sonnet": ("carbonmm.graphrag.recommend_v281_sonnet", "GraphRAGv281SonnetBaseline"),
}


def load_baseline(name: str):
    if name not in BASELINE_REGISTRY:
        raise SystemExit(f"unknown baseline: {name}. choices: {sorted(BASELINE_REGISTRY)}")
    module_path, class_name = BASELINE_REGISTRY[name]
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)()


def load_eval_pdds(
    n: int | None,
    seed: int,
    split: str | None = None,
) -> list[tuple[str, str, str, str, dict]]:
    """Return [(gid, registry, methodology_label, full_text, pdd_meta), ...].

    `pdd_meta` carries the reliable-at-registration metadata used by the
    candidate filter (graphrag-v28): {"registry", "creditingPeriodStartDate"}.

    If `split` is "val" or "test", uses the frozen splits manifest at
    data/manifests/icdm2026-splits.json (built via eval.splits) — returns
    ALL PDDs in that split unless `n` is also given (then truncates to first n).

    If `split` is None, falls back to random shuffle over the restricted set
    (legacy behavior for the original 20-PDD smoke).
    """
    if split in {"val", "test"}:
        from .splits import load_splits
        splits = load_splits()
        gids = [(p["gid"], p["registry"], p["gt"]) for p in splits[split]["pdds"]]
        out = []
        for gid, reg, label in gids:
            if n is not None and len(out) >= n:
                break
            path = EVAL_PDD_ROOT / reg / gid / f"{gid}.body.json"
            if not path.exists():
                continue
            try:
                d = json.loads(path.read_text())
            except Exception:
                continue
            text = d.get("full_text") or ""
            _src = os.environ.get("EVAL_TEXT_SOURCE")
            if _src == "rich":
                rich = ICDM_ROOT / "data" / "rich-eval" / reg / f"{gid}.rich.json"
                if not rich.exists():
                    continue  # pilot mode: only PDDs with rich text are evaluated
                text = json.loads(rich.read_text()).get("rich_text") or ""
            elif _src == "clean":
                clean = ICDM_ROOT / "data" / "clean-eval" / reg / f"{gid}.clean.json"
                if not clean.exists():
                    continue  # only PDDs with a built clean (section-bounded) input
                text = json.loads(clean.read_text()).get("clean_text") or ""
            if not text:
                continue
            meta = {"registry": reg, "creditingPeriodStartDate": d.get("creditingPeriodStartDate")}
            out.append((gid, reg, label, text, meta))
        return out

    # Legacy random-shuffle path
    m = json.loads(RESTRICTED_MANIFEST.read_text())
    gids: list[str] = []
    if "globalIds" in m and isinstance(m["globalIds"], list):
        gids = m["globalIds"]
    else:
        for cls_ids in m.get("by_class_globalIds", {}).values():
            gids.extend(cls_ids)
    rng = random.Random(seed)
    rng.shuffle(gids)

    out = []
    for gid in gids:
        if n is not None and len(out) >= n:
            break
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
        label = d.get("methodology_label")
        text = d.get("full_text") or ""
        if not label or not text:
            continue
        meta = {"registry": reg, "creditingPeriodStartDate": d.get("creditingPeriodStartDate")}
        out.append((gid, reg, label, text, meta))
    return out


def compute_metrics(per_pdd: list[dict], top_k: int = 5) -> dict:
    """Per-baseline aggregate + per-registry breakdown."""
    n = len(per_pdd)
    if n == 0:
        return {"n": 0}

    def metric_block(items: list[dict]) -> dict:
        if not items:
            return {"n": 0, "top1": 0.0, "top5": 0.0, "mrr": 0.0}
        nb = len(items)
        top1 = sum(1 for r in items if r["gt_rank"] == 1) / nb
        top5 = sum(1 for r in items if r["gt_rank"] is not None and r["gt_rank"] <= 5) / nb
        mrr = (
            sum(1.0 / r["gt_rank"] for r in items if r["gt_rank"] is not None) / nb
        )
        return {"n": nb, "top1": round(top1, 4), "top5": round(top5, 4), "mrr": round(mrr, 4)}

    all_block = metric_block(per_pdd)
    by_reg = {
        reg: metric_block([r for r in per_pdd if r["registry"] == reg])
        for reg in ("GS", "VCS")
    }
    return {**all_block, "by_registry": by_reg}


def evaluate(
    baseline_name: str,
    n: int | None,
    seed: int,
    top_k: int = 5,
    split: str | None = None,
) -> dict:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger.info("loading baseline %s …", baseline_name)
    t0 = time.time()
    b = load_baseline(baseline_name)
    setup_s = time.time() - t0

    pdds = load_eval_pdds(n, seed, split=split)
    logger.info(
        "loaded %d PDDs (target=%s, split=%s) seed=%d",
        len(pdds), n if n else "all", split or "random", seed,
    )

    per_pdd = []
    t_query = time.time()
    for gid, reg, label, text, meta in pdds:
        try:
            preds = b.predict(text, top_k=top_k, pdd_meta=meta)
        except TypeError:
            # Baseline that doesn't accept pdd_meta yet
            preds = b.predict(text, top_k=top_k)
        except Exception as e:
            logger.warning("predict failed gid=%s err=%s", gid, e)
            preds = []
        codes = [c for c, _ in preds]
        rank = next((i + 1 for i, c in enumerate(codes) if c == label), None)
        per_pdd.append(
            {
                "gid": gid,
                "registry": reg,
                "gt_label": label,
                "predicted_top5": codes,
                "gt_rank": rank,
            }
        )
    query_s = time.time() - t_query
    metrics = compute_metrics(per_pdd, top_k=top_k)
    return {
        "baseline": baseline_name,
        "n_requested": n,
        "n_actual": len(pdds),
        "seed": seed,
        "split": split or "random",
        "top_k": top_k,
        "setup_s": round(setup_s, 2),
        "query_s": round(query_s, 2),
        "per_query_s": round(query_s / max(1, len(pdds)), 2),
        "metrics": metrics,
        "per_pdd": per_pdd,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseline",
        required=True,
        choices=sorted(BASELINE_REGISTRY) + ["all"],
    )
    ap.add_argument("--n", type=int, default=None, help="cap n; default = all in split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--split", default=None, choices=[None, "val", "test"],
                    help="use frozen split (val/test) or random shuffle (None)")
    ap.add_argument("--out-root", type=Path, default=RESULTS_ROOT)
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)

    if args.baseline == "all":
        names = list(BASELINE_REGISTRY.keys())
    else:
        names = [args.baseline]

    summary = []
    split_tag = args.split or "rand"
    for name in names:
        result = evaluate(name, args.n, args.seed, args.top_k, split=args.split)
        n_actual = result["n_actual"]
        out_path = (
            args.out_root / f"{name}-{split_tag}-n{n_actual}-seed{args.seed}.json"
        )
        out_path.write_text(json.dumps(result, indent=2))
        m = result["metrics"]
        summary.append(
            (
                name,
                result["n_actual"],
                m.get("top1"),
                m.get("top5"),
                m.get("mrr"),
                result["per_query_s"],
            )
        )
        print(
            f"\n=== {name} ({result['n_actual']}/{args.n}) ===\n"
            f"  top1={m.get('top1')}  top5={m.get('top5')}  MRR={m.get('mrr')}"
            f"  · {result['per_query_s']}s/query"
        )
        if "by_registry" in m:
            for reg, mb in m["by_registry"].items():
                print(
                    f"    {reg:3s}: n={mb['n']:3d}  top1={mb['top1']}  "
                    f"top5={mb['top5']}  MRR={mb['mrr']}"
                )

    if len(names) > 1:
        print("\n=== summary ===")
        print(f"{'baseline':14s} {'n':>4s}  {'top1':>6s}  {'top5':>6s}  {'MRR':>6s}  {'s/q':>6s}")
        for row in summary:
            print(f"{row[0]:14s} {row[1]:>4d}  {row[2]:>6}  {row[3]:>6}  {row[4]:>6}  {row[5]:>6}")


if __name__ == "__main__":
    main()
