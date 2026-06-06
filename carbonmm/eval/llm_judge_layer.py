"""LLM-as-judge layer for disagreement cases.

For PDDs where naive-rag top-1 ≠ v2.5 top-1, ask Haiku to choose between the
two candidates given the project text. This targets the 119 disagreement
cases (22% of test) where the heuristic router has ambiguity.

Cost: ~120 LLM calls (one per disagreement) × ~30s daemon = ~60 min wall (workers=2).

Saves judged predictions → ensemble-router-judge-test-n535-seed42.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date
from pathlib import Path
from statistics import mean

import httpx

RESULTS = Path(__file__).resolve().parents[1] / "results"
EVAL = Path(__file__).resolve().parents[1] / "data" / "eval-pdds"

DAEMON_URL = "http://localhost:8765/chat"
MODEL = "claude-haiku-4-5"
TIMEOUT = 120.0

JUDGE_PROMPT = """You are a carbon-credit methodology classifier deciding between two candidate methodologies for a project.

# Project text (truncated)

{project_text}

# Candidate A: {code_a}

{clauses_a}

# Candidate B: {code_b}

{clauses_b}

# Instructions

Pick the SINGLE methodology code that best applies to this project. Reply with EXACTLY one line:

CHOICE: <code_a or code_b>

Then on the next line a 1-sentence rationale. Use the exact code string."""


def call_judge(prompt: str, client: httpx.Client, model: str = MODEL):
    """Call daemon judge with retry on 500/502/503."""
    last_exc = None
    for attempt in range(3):
        try:
            resp = client.post(
                DAEMON_URL,
                json={"prompt": prompt, "model": model, "timeout_s": 90.0, "use_cache": True},
            )
            resp.raise_for_status()
            return resp.json()["text"]
        except httpx.HTTPStatusError as e:
            last_exc = e
            if e.response.status_code in (500, 502, 503) and attempt < 2:
                time.sleep(3 + attempt * 2)
                continue
            raise
    raise last_exc


def parse_choice(text: str, code_a: str, code_b: str) -> str | None:
    m = re.search(r"CHOICE\s*:\s*(\S+)", text, re.IGNORECASE)
    if m:
        choice = m.group(1).strip().strip(".,;:")
        if choice == code_a or choice == code_b:
            return choice
    # Fallback: scan for either code in response
    if code_a in text and code_b not in text:
        return code_a
    if code_b in text and code_a not in text:
        return code_b
    return None


def build_clauses_excerpt(code: str, df, max_chars: int = 1500) -> str:
    rows = df[df["code"] == code].head(3)
    out = []
    used = 0
    for row in rows.itertuples(index=False):
        txt = (row.clause_text or "").replace("\n", " ")[:600]
        line = f"- ({row.clause_type}) {txt}"
        if used + len(line) > max_chars:
            break
        out.append(line)
        used += len(line)
    return "\n".join(out) if out else "(no clauses)"


def wilson_ci(p: float, n: int, z: float = 1.96):
    if n == 0: return (0.0, 0.0)
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return ((center - margin) / denom, (center + margin) / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=535)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    v25 = json.load(open(RESULTS / f"graphrag-v25-{args.split}-n{args.n}-seed{args.seed}.json"))
    nr = json.load(open(RESULTS / f"naive-rag-{args.split}-n{args.n}-seed{args.seed}.json"))
    v25_by = {p["gid"]: p for p in v25["per_pdd"]}

    # Find disagreement cases
    disagreements = []
    agreements = []
    for nrp in nr["per_pdd"]:
        gid = nrp["gid"]
        v25p = v25_by[gid]
        nr_t1 = nrp["predicted_top5"][0] if nrp["predicted_top5"] else None
        v25_t1 = v25p["predicted_top5"][0] if v25p["predicted_top5"] else None
        if nr_t1 and v25_t1 and nr_t1 != v25_t1:
            disagreements.append((nrp, v25p, nr_t1, v25_t1))
        else:
            agreements.append((nrp, v25p, nr_t1, v25_t1))

    logger.info(f"Disagreements: {len(disagreements)}/{len(nr['per_pdd'])}")
    logger.info(f"Agreements   : {len(agreements)}")

    # Load PDD bodies for project text
    test_gids = {nrp["gid"] for nrp in nr["per_pdd"]}
    gid_to_text = {}
    for body_path in EVAL.rglob("*.body.json"):
        gid = body_path.stem.replace(".body", "")
        if gid in test_gids:
            try:
                body = json.loads(body_path.read_text())
                gid_to_text[gid] = (body.get("full_text") or "")[:6000]
            except Exception:
                pass
    logger.info(f"Loaded {len(gid_to_text)} PDD texts")

    # Load clause corpus for excerpts
    import pandas as pd
    df = pd.read_parquet("carbonmm/data/embeddings/clauses-meta.parquet")
    logger.info(f"Loaded {len(df)} clauses")

    # Call judge in parallel
    client = httpx.Client(timeout=TIMEOUT)
    judged_preds = {}

    def judge_one(nrp, v25p, nr_t1, v25_t1):
        gid = nrp["gid"]
        text = gid_to_text.get(gid, "")
        if not text:
            return gid, nr_t1  # fallback to naive-rag
        prompt = JUDGE_PROMPT.format(
            project_text=text,
            code_a=nr_t1,
            clauses_a=build_clauses_excerpt(nr_t1, df),
            code_b=v25_t1,
            clauses_b=build_clauses_excerpt(v25_t1, df),
        )
        try:
            raw = call_judge(prompt, client)
            choice = parse_choice(raw, nr_t1, v25_t1)
            return gid, choice or nr_t1
        except Exception as e:
            logger.warning(f"judge failed gid={gid}: {e}")
            return gid, nr_t1

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(judge_one, *d): d[0]["gid"] for d in disagreements}
        done = 0
        for fut in as_completed(futs):
            gid, choice = fut.result()
            judged_preds[gid] = choice
            done += 1
            if done % 20 == 0:
                logger.info(f"judged {done}/{len(disagreements)} elapsed={time.time()-t0:.0f}s")
    elapsed = time.time() - t0
    logger.info(f"Total judge time: {elapsed:.0f}s")

    # Combine: agreements keep nr_t1 (== v25_t1); disagreements use judged choice
    final_preds = []
    for nrp, v25p, nr_t1, v25_t1 in agreements:
        gid = nrp["gid"]
        # Agreements: top-1 is nr_t1 (==v25_t1 if both exist, else whichever)
        top1 = nr_t1 or v25_t1
        # Fill 2-5 from RRF union
        scores = defaultdict(float)
        for r, c in enumerate(nrp["predicted_top5"], 1):
            if c != top1:
                scores[c] += 1.0 / (60 + r)
        for r, c in enumerate(v25p["predicted_top5"], 1):
            if c != top1:
                scores[c] += 1.0 / (60 + r)
        rest = [c for c, _ in sorted(scores.items(), key=lambda x: -x[1])[:4]]
        top5 = [top1] + rest if top1 else (nrp["predicted_top5"] or v25p["predicted_top5"])[:5]
        rank = next((i + 1 for i, c in enumerate(top5) if c == nrp["gt_label"]), None)
        final_preds.append({"gid": gid, "registry": nrp["registry"], "gt_label": nrp["gt_label"],
                            "predicted_top5": top5, "gt_rank": rank})

    for nrp, v25p, nr_t1, v25_t1 in disagreements:
        gid = nrp["gid"]
        top1 = judged_preds.get(gid, nr_t1)
        # Fill 2-5 from RRF union
        scores = defaultdict(float)
        for r, c in enumerate(nrp["predicted_top5"], 1):
            if c != top1:
                scores[c] += 1.0 / (60 + r)
        for r, c in enumerate(v25p["predicted_top5"], 1):
            if c != top1:
                scores[c] += 1.0 / (60 + r)
        rest = [c for c, _ in sorted(scores.items(), key=lambda x: -x[1])[:4]]
        top5 = [top1] + rest
        rank = next((i + 1 for i, c in enumerate(top5) if c == nrp["gt_label"]), None)
        final_preds.append({"gid": gid, "registry": nrp["registry"], "gt_label": nrp["gt_label"],
                            "predicted_top5": top5, "gt_rank": rank})

    n = len(final_preds)
    t1 = sum(1 for p in final_preds if p["gt_rank"] == 1) / n
    t5 = sum(1 for p in final_preds if isinstance(p["gt_rank"], int) and p["gt_rank"] <= 5) / n
    rr = [1.0 / p["gt_rank"] if isinstance(p["gt_rank"], int) and p["gt_rank"] <= 5 else 0 for p in final_preds]
    mrr = mean(rr)
    ci = wilson_ci(t1, n)

    print(f"\n=== Router + LLM judge layer ===")
    print(f"n = {n}")
    print(f"disagreements judged: {len(judged_preds)}")
    print(f"top-1 = {t1:.4f}  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"top-5 = {t5:.4f}")
    print(f"MRR   = {mrr:.4f}")
    for reg in ["GS", "VCS"]:
        sub = [p for p in final_preds if p["registry"] == reg]
        sn = len(sub) or 1
        st1 = sum(1 for p in sub if p["gt_rank"] == 1) / sn
        st5 = sum(1 for p in sub if isinstance(p["gt_rank"], int) and p["gt_rank"] <= 5) / sn
        print(f"  {reg}: n={sn}  top-1={st1:.4f}  top-5={st5:.4f}")

    print(f"\nvs naive-rag (0.5290, 0.7533):")
    print(f"  Δtop-1 = {t1 - 0.529:+.4f} ({(t1 - 0.529)*100:+.1f}pp)")
    print(f"  Δtop-5 = {t5 - 0.7533:+.4f}")
    print(f"vs router v1 (0.5869, 0.7944):")
    print(f"  Δtop-1 = {t1 - 0.5869:+.4f}")
    print(f"  Δtop-5 = {t5 - 0.7944:+.4f}")
    print(f"vs Oracle ceiling (0.6318):")
    print(f"  Δtop-1 = {t1 - 0.6318:+.4f}")

    out = {
        "baseline": "ensemble-router-judge",
        "n_actual": n,
        "metrics": {"top1": t1, "top5": t5, "mrr": mrr,
                    "by_registry": {reg: {
                        "n": sum(1 for p in final_preds if p["registry"] == reg),
                        "top1": sum(1 for p in final_preds if p["registry"] == reg and p["gt_rank"] == 1) / max(1, sum(1 for p in final_preds if p["registry"] == reg)),
                        "top5": sum(1 for p in final_preds if p["registry"] == reg and isinstance(p["gt_rank"], int) and p["gt_rank"] <= 5) / max(1, sum(1 for p in final_preds if p["registry"] == reg)),
                    } for reg in ["GS", "VCS"]}},
        "per_pdd": final_preds,
    }
    out_path = RESULTS / f"ensemble-router-judge-{args.split}-n{n}-seed{args.seed}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
