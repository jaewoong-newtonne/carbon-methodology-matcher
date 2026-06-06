"""Within-group disambiguation PoC on G01 (grid-connected renewable).

Takes the best system's frozen predictions (graphrag-v281-sonnet), and for each
G01 PDD re-ranks ONLY the in-group candidates among the slots they already occupy,
using an LLM judge over the structured discriminator cards + the PDD's extracted
scale. Monotone-safe: never introduces an unseen code, so group-top5 is unchanged
and exact-top5 is non-decreasing; only exact-top1 (the within-group pick) can move.

Reports exact-top1 on G01 PDDs: baseline vs disambiguated, with fixed/broken examples.

Example (long-running, backgrounded):
    nohup python eval/run_disambig_poc.py --group G01 \\
        --judge-model claude-sonnet-4-6 > /tmp/disambig-poc.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import httpx

ICDM_ROOT = Path(__file__).resolve().parents[1]
RESULTS = ICDM_ROOT / "results"
MANIFESTS = ICDM_ROOT / "data" / "manifests"
EVAL_PDD_ROOT = ICDM_ROOT / "data" / "eval-pdds"
DAEMON_URL = "http://localhost:8765/chat"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("poc")


def load(name: str) -> dict:
    return json.loads((MANIFESTS / name).read_text())


def pdd_text(gid: str, reg: str, max_chars: int = 5000) -> str:
    p = EVAL_PDD_ROOT / reg / gid / f"{gid}.body.json"
    if not p.exists():
        return ""
    try:
        return (json.loads(p.read_text()).get("full_text") or "")[:max_chars]
    except Exception:
        return ""


JUDGE_PROMPT = """You are matching a carbon project to the SINGLE most appropriate methodology.
All candidates govern the same activity (grid-connected renewable electricity); they differ
mainly in SCALE (small-scale AMS-* vs large-scale ACM/AM), renewable sub-technology, and
project configuration. Use the project text + the candidate cards to rank them best-first.

# Project (truncated)
{ptext}

# Extracted project scale hint: {scale}

# Candidate methodologies (with distinguishing cards)
{cards}

Return ONLY JSON: {{"ranking": ["CODE_best", ...all candidate codes in order...], "why": "one line"}}"""


def _daemon_post(prompt: str, model: str, timeout: float = 120.0, retries: int = 4) -> str:
    last = None
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=timeout) as cl:
                r = cl.post(DAEMON_URL, json={"prompt": prompt, "model": model,
                                              "timeout_s": min(timeout - 5, 90.0), "use_cache": True})
            r.raise_for_status()
            return r.json()["text"]
        except httpx.HTTPStatusError as e:
            last = e
            if e.response.status_code in (429, 500, 502, 503) and attempt < retries - 1:
                time.sleep(2 + attempt * 3)
                continue
            raise
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last = e
            if attempt < retries - 1:
                time.sleep(2 + attempt * 3)
                continue
            raise
    raise last


def judge(ptext: str, scale: str, cand_cards: dict[str, dict], model: str) -> list[str]:
    cards_txt = "\n".join(
        f"- {c}: {json.dumps(card, ensure_ascii=False)}" for c, card in cand_cards.items()
    )
    prompt = JUDGE_PROMPT.format(ptext=ptext, scale=scale or "unknown", cards=cards_txt)
    text = _daemon_post(prompt, model)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return []
    try:
        return [str(c) for c in (json.loads(m.group(0)).get("ranking") or [])]
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="G01")
    ap.add_argument("--system", default="graphrag-v281-sonnet")
    ap.add_argument("--judge-model", default="claude-sonnet-4-6")
    ap.add_argument("--out", type=Path, default=ICDM_ROOT / "paper" / "sections" / "6-disambiguation-poc-auto.md")
    args = ap.parse_args()

    gmap = load("method-groups.json")
    c2g = gmap["code_to_group"]
    cards = load("discriminator-cards.json")
    feats = load("pdd-extracted-features-test.json") if (MANIFESTS / "pdd-extracted-features-test.json").exists() else {}

    def group_of(code: str) -> str:
        return c2g.get(code, f"SINGLETON::{code}")

    res_files = sorted(RESULTS.glob(f"{args.system}-test-n*-seed*.json"))
    if not res_files:
        raise SystemExit(f"no result file for {args.system}")
    result = json.loads(res_files[-1].read_text())

    g_pdds = [r for r in result["per_pdd"] if group_of(r["gt_label"]) == args.group]
    logger.info("%s: %d test PDDs in group %s", args.system, len(g_pdds), args.group)

    base_correct = 0
    new_correct = 0
    n_rerank = 0
    fixed, broken = [], []

    for r in g_pdds:
        gt = r["gt_label"]
        codes = list(r.get("predicted_top5", []))
        base_top1 = bool(codes) and codes[0] == gt
        base_correct += base_top1

        ingroup_idx = [i for i, c in enumerate(codes) if group_of(c) == args.group]
        ingroup_codes = [codes[i] for i in ingroup_idx]
        new_codes = codes

        if len(ingroup_codes) >= 2:
            cand_cards = {c: cards.get(c, {"note": "no-card"}) for c in ingroup_codes}
            scale = (feats.get(r["gid"]) or {}).get("scale", "unknown")
            try:
                ranked = judge(pdd_text(r["gid"], r["registry"]), scale, cand_cards, args.judge_model)
            except Exception as e:
                logger.warning("judge failed gid=%s: %s", r["gid"], e)
                ranked = []
            new_ingroup = [c for c in ranked if c in ingroup_codes] + \
                          [c for c in ingroup_codes if c not in ranked]
            if new_ingroup != ingroup_codes:
                n_rerank += 1
                new_codes = list(codes)
                for slot, c in zip(ingroup_idx, new_ingroup):
                    new_codes[slot] = c

        new_top1 = bool(new_codes) and new_codes[0] == gt
        new_correct += new_top1
        if new_top1 and not base_top1:
            fixed.append({"gid": r["gid"], "gt": gt, "was": codes[0], "now": new_codes[0]})
        elif base_top1 and not new_top1:
            broken.append({"gid": r["gid"], "gt": gt, "was": codes[0], "now": new_codes[0]})

    n = len(g_pdds)
    base_t1 = base_correct / n if n else 0
    new_t1 = new_correct / n if n else 0
    L = [f"# §6 Within-group disambiguation PoC — group {args.group} (auto)\n",
         f"System: **{args.system}** · judge: {args.judge_model} · group PDDs: {n}\n",
         f"Re-ranked (>=2 in-group candidates, order changed): {n_rerank}\n",
         "| Metric | baseline | + disambiguator | Δ |",
         "|---|---|---|---|",
         f"| exact top-1 on {args.group} | {base_t1:.3f} | **{new_t1:.3f}** | +{new_t1-base_t1:.3f} |",
         f"| fixed | | {len(fixed)} | |",
         f"| broken (monotone-safety check) | | {len(broken)} | |",
         "\n## Fixed examples", "```", json.dumps(fixed[:12], indent=2), "```",
         "\n## Broken examples (should be few; tie-break noise)", "```", json.dumps(broken[:12], indent=2), "```"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L))
    print("\n".join(L))
    logger.info("DONE: G01 exact-top1 %.3f -> %.3f (fixed %d, broken %d, reranked %d)",
                base_t1, new_t1, len(fixed), len(broken), n_rerank)


if __name__ == "__main__":
    main()
