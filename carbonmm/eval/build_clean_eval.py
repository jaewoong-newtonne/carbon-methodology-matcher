"""Build per-PDD CLEAN eval input: the full project-description Section A (up to the
Section-B / methodology-declaration boundary), code-redacted — the principled replacement
for the uniform `full_text[:6000]` truncation (which both leaked GT codes and starved late
capacity facts). Writes `data/clean-eval/{REG}/{gid}.clean.json`. The recommender caps the
text at runtime via `EVAL_TEXT_MAXCHARS` (set 16000 for the clean runs).

Reuses the existing section extractor + redaction + leakage gate so the construction matches
the §3.3 protocol the paper describes.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
from importlib import import_module

ICDM = Path(__file__).resolve().parents[1]
_REPO = str(ICDM.parents[1])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_sa = import_module("carbonmm.ingest.extract_pdd_section_a")
_pat = import_module("carbonmm.redact.patterns")
_lg = import_module("carbonmm.eval.leakage_gate")

_CAP = re.compile(r"\d+(?:\.\d+)?\s*[MkG]W(?:e|th|p)?\b", re.I)


def _body(reg: str, gid: str) -> str:
    p = ICDM / "data/eval-pdds" / reg / gid / f"{gid}.body.json"
    return json.loads(p.read_text()).get("full_text", "") if p.exists() else ""


def build_one(reg: str, gid: str, gt: str) -> dict:
    full = _body(reg, gid)
    res = _sa.extract_section_a(full, {"globalId": gid, "registry": reg})
    if res.extraction_status == "ok" and res.section_a_text:
        text, status = res.section_a_text, "section_a"
    else:
        # unparseable template → conservative fallback: head of the body (still redacted below).
        text, status = full[:16000], f"fallback_head:{res.extraction_status}"
    text, _ = _pat.apply_pass1_sentence_level(text)  # strip residual methodology codes
    gate = _lg.gate_text(text, gt, bm25_rank1=None)
    if not gate["passes"]:  # sentence-level pass missed a variant -> force-redact the literal GT code
        code = _lg._gt_code(gt)
        if code and len(code) >= 4:
            text = re.sub(re.escape(code), "[REDACTED:METHODOLOGY_REF]", text, flags=re.I)
            gate = _lg.gate_text(text, gt, bm25_rank1=None)
    return {"globalId": gid, "registry": reg, "gt": gt, "clean_text": text,
            "status": status, "section_a_end": res.source_char_end,
            "char_count": len(text), "capacity_present": bool(_CAP.search(text)),
            "gate": gate}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    splits = json.loads((ICDM / "data/manifests/icdm2026-splits.json").read_text())
    pdds = splits[a.split]["pdds"]
    if a.limit:
        pdds = pdds[: a.limit]
    out_root = ICDM / "data/clean-eval"
    n_fail = n_fb = cap = 0
    chars: list[int] = []
    by_reg: dict[str, list[int]] = {"GS": [], "VCS": []}
    for r in pdds:
        rec = build_one(r["registry"], r["gid"], r["gt"])
        chars.append(rec["char_count"])
        by_reg.setdefault(r["registry"], []).append(rec["char_count"])
        if rec["status"].startswith("fallback"):
            n_fb += 1
        if not rec["gate"]["passes"]:
            n_fail += 1
            print(f"  GATE FAIL {r['gid']} ({r['registry']}) leaks={rec['gate']['code_leaks']} status={rec['status']}")
        if rec["capacity_present"]:
            cap += 1
        d = out_root / r["registry"]
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{r['gid']}.clean.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2))
    chars.sort()
    med = chars[len(chars) // 2] if chars else 0
    n = len(pdds)
    print(f"built {n} | fallback_head={n_fb} | gate_fail={n_fail} | "
          f"median_chars={med} | capacity_present={cap} ({cap / n:.1%})")
    for reg, cs in by_reg.items():
        if cs:
            cs2 = sorted(cs)
            print(f"  {reg}: n={len(cs)} median_chars={cs2[len(cs2) // 2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
