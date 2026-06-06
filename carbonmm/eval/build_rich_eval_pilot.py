"""Build rich, leakage-safe eval text for the pilot PDDs:
  redacted body -> build_rich_text (3-tier) -> Pass1 sentence-level (codes)
  -> Pass2 daemon paraphrase verify (optional) -> leakage gate
  -> data/rich-eval/{REG}/{gid}.rich.json

Pass2 is OFF by default (--pass2 to enable); the daemon URL is read from env
V281_DAEMON_URL (default http://localhost:8765/chat) per project policy.
Gate-failing PDDs are re-emitted forced to Tier 3 (Section A + capacity window).
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from importlib import import_module

ICDM = Path(__file__).resolve().parents[1]
_REPO_ROOT = str(ICDM.parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_bret = import_module("carbonmm.ingest.build_rich_eval_text")
_pat = import_module("carbonmm.redact.patterns")
_pipe = import_module("carbonmm.redact.pipeline")
_lg = import_module("carbonmm.eval.leakage_gate")


def _source_text(reg: str, gid: str) -> str:
    """Prefer the Pass1-redacted body (codes already removed); fall back to raw body."""
    red = ICDM / "data/redacted" / reg / gid / f"{gid}.redacted.json"
    if red.exists():
        return json.loads(red.read_text()).get("redacted_text") or ""
    body = ICDM / "data/eval-pdds" / reg / gid / f"{gid}.body.json"
    return json.loads(body.read_text()).get("full_text", "") if body.exists() else ""


def build_one(reg: str, gid: str, gt: str, pass2: bool) -> dict:
    src = _source_text(reg, gid)
    built = _bret.build_rich_text(src, {"globalId": gid, "registry": reg})
    text, _m1 = _pat.apply_pass1_sentence_level(built["rich_text"])  # codes -> sentinel
    if pass2 and text:
        url = os.environ.get("V281_DAEMON_URL", "http://localhost:8765/chat")
        text, _m2 = _pipe.pass2_llm(text, daemon_url=url, only_marked_paragraphs=False)
    gate = _lg.gate_text(text, gt, bm25_rank1=None)

    # Gate failure (residual literal code) -> force Tier 3 (Section A + capacity window).
    if not gate["passes"]:
        forced = _bret.build_rich_text("", {"globalId": gid, "registry": reg})  # placeholder
        # Re-derive a conservative Tier-3 text directly from the source and re-redact.
        cap = _bret.CAPACITY.search(src)
        head = src[:6000]
        if cap:
            head = head + "\n\n" + src[max(0, cap.start() - 600): cap.end() + 600]
        text, _ = _pat.apply_pass1_sentence_level(head)
        built = {"tier": 3, "template": built["template"], "sections_kept": ["forced_tier3"]}
        gate = _lg.gate_text(text, gt, bm25_rank1=None)

    return {"globalId": gid, "registry": reg, "gt": gt, "rich_text": text,
            "tier": built["tier"], "template": built["template"],
            "sections_kept": built["sections_kept"], "char_count": len(text),
            "gate": gate}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ICDM / "data/manifests/pilot-150.json")
    ap.add_argument("--limit", type=int, default=None, help="cap for smoke")
    ap.add_argument("--pass2", action="store_true")
    a = ap.parse_args()
    pdds = json.loads(a.manifest.read_text())["pdds"]
    if a.limit:
        pdds = pdds[: a.limit]
    out_root = ICDM / "data/rich-eval"
    n_fail = 0
    tiers: dict[int, int] = {}
    chars: list[int] = []
    for r in pdds:
        rec = build_one(r["registry"], r["gid"], r["gt"], a.pass2)
        tiers[rec["tier"]] = tiers.get(rec["tier"], 0) + 1
        chars.append(rec["char_count"])
        if not rec["gate"]["passes"]:
            n_fail += 1
            print(f"  GATE FAIL {r['gid']} leaks={rec['gate']['code_leaks']} tier={rec['tier']}")
        d = out_root / r["registry"]
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{r['gid']}.rich.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2))
    chars.sort()
    med = chars[len(chars) // 2] if chars else 0
    print(f"built {len(pdds)} | tiers={tiers} | gate_fail={n_fail} | median_chars={med}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
