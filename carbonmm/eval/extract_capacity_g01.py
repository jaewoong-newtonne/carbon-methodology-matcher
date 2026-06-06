"""Extract numeric ELECTRICAL capacity (MW) from each G01 PDD's redacted Section A.

Combined with data/manifests/cdm-ssc-thresholds.json (AMS-I.D. small-scale cap = 15 MW)
this enables the scale-fit *information-availability* test:
  (1) of the G01 PDDs (and the ACM0002<->AMS-I.D. flip subset), how many even STATE a
      capacity in the redacted Section A?  -> if absent, the scale signal is information-bounded.
  (2) when stated, does the 15 MW cap correctly separate large (ACM/AM) from small (AMS) GT?
  (3) feed the controlled model-dependence test (does an explicit threshold help weak models
      but not the frontier, which already infers scale?).

Prediction-blind: reads ONLY the PDD input text (never predictions/labels for extraction).
Channel: OpenAI. A `quote` field (verbatim) guards against hallucinated figures.

Run:  unset OPENAI_API_KEY; python3 -m carbonmm.eval.extract_capacity_g01
Out:  data/manifests/g01-capacity-extracted.json
"""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .harness import load_eval_pdds
from ..graphrag.recommend_v28_openai import _make_client

ICDM_ROOT = Path(__file__).resolve().parents[1]
GMAP = json.loads((ICDM_ROOT / "data" / "manifests" / "method-groups.json").read_text())
C2G = GMAP["code_to_group"]
MODEL = os.environ.get("EXTRACT_MODEL", "gpt-4o-mini")
OUT = ICDM_ROOT / "data" / "manifests" / "g01-capacity-extracted.json"
WORKERS = int(os.environ.get("EXTRACT_WORKERS", "8"))

PROMPT = """You extract the installed/nameplate ELECTRICAL capacity of a carbon-credit project from its (redacted) project-description text.

Return STRICT JSON only (no prose):
{{"capacity_mw": <number or null>, "stated": <true|false>, "quote": "<verbatim snippet containing the capacity figure, <=120 chars, or empty>", "unit_raw": "<unit exactly as written, or empty>"}}

Rules:
- capacity_mw = the project's TOTAL/aggregate power capacity converted to MW (kW/1000; GW*1000; MWp/MWac treated as MW).
- stated = true ONLY if an explicit capacity figure appears in the text. The "quote" MUST be copied verbatim from the text.
- If no capacity is stated, return capacity_mw=null, stated=false, quote="".

TEXT:
{text}
"""


def _extract_one(client, gid: str, gt: str, text: str) -> dict:
    rec = {"gid": gid, "gt": gt}
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT.format(text=text[:6000])}],
            timeout=60,
        )
        raw = r.choices[0].message.content or ""
        m = re.search(r"\{.*\}", raw, re.S)
        d = json.loads(m.group(0)) if m else {}
        cap = d.get("capacity_mw")
        try:
            cap = float(cap) if cap is not None else None
        except (TypeError, ValueError):
            cap = None
        rec.update({
            "capacity_mw": cap,
            "stated": bool(d.get("stated")) and bool(str(d.get("quote") or "").strip()),
            "quote": (d.get("quote") or "")[:200],
            "unit_raw": (d.get("unit_raw") or "")[:40],
        })
    except Exception as e:
        rec.update({"capacity_mw": None, "stated": False, "quote": "", "unit_raw": "", "error": str(e)[:140]})
    return rec


def main():
    pdds = load_eval_pdds(None, 42, "test")
    g01 = [(gid, label, text) for (gid, _reg, label, text, _meta) in pdds if C2G.get(label) == "G01"]
    print(f"G01 test PDDs: {len(g01)}  (model={MODEL}, workers={WORKERS})")
    client = _make_client()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(_extract_one, client, gid, gt, text): gid for gid, gt, text in g01}
        done = 0
        for f in as_completed(futs):
            results.append(f.result())
            done += 1
            if done % 50 == 0:
                print(f"  done {done}/{len(g01)}")
    results.sort(key=lambda r: r["gid"])
    OUT.write_text(json.dumps({"model": MODEL, "n": len(results), "results": results}, indent=2))

    stated = [r for r in results if r.get("stated")]
    errs = [r for r in results if r.get("error")]
    print(f"\n=== G01 capacity extraction ===")
    print(f"  n={len(results)}  stated={len(stated)} ({len(stated)/len(results):.1%})  errors={len(errs)}")
    # scale-fit sanity vs 15 MW cap, by GT scale class (AMS=small expects <=15, ACM/AM=large expects >15)
    def gt_small(gt): return gt.startswith("AMS-")
    sm = [r for r in stated if gt_small(r["gt"]) and r["capacity_mw"] is not None]
    lg = [r for r in stated if not gt_small(r["gt"]) and r["capacity_mw"] is not None]
    if sm:
        ok = sum(1 for r in sm if r["capacity_mw"] <= 15.0)
        print(f"  GT=small(AMS), stated: {len(sm)}  | capacity<=15MW (consistent): {ok} ({ok/len(sm):.1%})")
    if lg:
        ok = sum(1 for r in lg if r["capacity_mw"] > 15.0)
        print(f"  GT=large(ACM/AM), stated: {len(lg)} | capacity>15MW (consistent): {ok} ({ok/len(lg):.1%})")
    print(f"  saved: {OUT}")


if __name__ == "__main__":
    main()
