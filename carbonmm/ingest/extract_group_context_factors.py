"""Part 2: per-group NON-TECHNICAL differentiator extraction via LLM document comparison.

Part 1 (group_factor_analysis) showed WHICH structured factor differentiates each group.
Part 2 reads the actual methodology DOCUMENTS together and extracts the non-technical
decision factors NOT in structured metadata: target carbon market, host-country/regulatory
conditions, registry strategy & regional presence, additionality/baseline stringency,
buyer/program targeting, suppressed-demand / development framing.

For each group: one LLM (Sonnet) call comparing all GT-appearing member methodologies
(catalog metadata + cleaned eligibility excerpt + real-world registration footprint).

Output: data/manifests/group-context-factors.json + paper/sections/group-context-factors-auto.md

Usage:
    python ingest/extract_group_context_factors.py --groups G03 G05 G06 G15 G09
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import httpx

ICDM_ROOT = Path(__file__).resolve().parents[1]
M = ICDM_ROOT / "data" / "manifests"
CORPUS = ICDM_ROOT / "data" / "corpus"
DAEMON_URL = "http://localhost:8765/chat"

REGION = {}
for codes, r in [
    (["IND", "PAK", "BGD", "NPL", "LKA", "AFG", "BTN"], "south-asia"),
    (["CHN", "MNG", "KOR", "JPN"], "east-asia"),
    (["VNM", "KHM", "LAO", "MMR", "IDN", "PHL", "THA", "MYS"], "se-asia"),
    (["TUR", "EGY", "MAR", "JOR", "SAU", "ARE", "QAT", "IRQ", "IRN", "TUN", "DZA", "OMN", "KWT", "BHR"], "mena"),
    (["UGA", "KEN", "ETH", "TZA", "RWA", "MWI", "MOZ", "MDG", "ZMB", "ZWE", "BDI"], "east-africa"),
    (["NGA", "GHA", "SEN", "CIV", "MLI", "BFA", "BEN", "TGO", "NER", "GIN", "SLE", "LBR", "GMB"], "west-africa"),
    (["ZAF", "BWA", "NAM", "LSO", "SWZ"], "southern-africa"),
    (["COD", "CMR", "CAF", "COG", "GAB", "TCD", "AGO"], "central-africa"),
    (["BRA", "MEX", "COL", "PER", "CHL", "ARG", "ECU", "BOL", "GTM", "HND", "CRI", "PAN", "PRY", "URY"], "lac"),
    (["USA", "CAN"], "north-america"),
]:
    for c in codes:
        REGION[c] = r


def region(c):
    return REGION.get((c or "").upper(), "other")


_PAGEBREAK = re.compile(r"<!--\s*PAGE BREAK\s*-->")
_TAGS = re.compile(r"</?(table|tr|td|th)[^>]*>")
_WS = re.compile(r"\s+")


def clean(t):
    return _WS.sub(" ", _TAGS.sub(" ", _PAGEBREAK.sub(" ", t or ""))).strip()


def elig_excerpt(code, n=800):
    ms = list(CORPUS.glob(f"*/{code}/{code}.json"))
    if not ms:
        return ""
    try:
        d = json.loads(ms[0].read_text())
    except Exception:
        return ""
    parts = [str(d.get("typical_projects") or "")] + [str(x) for x in (d.get("applicability") or [])[:6]]
    return clean(" ".join(parts))[:n]


def daemon_post(prompt, model, timeout=150.0, retries=4):
    last = None
    for a in range(retries):
        try:
            with httpx.Client(timeout=timeout) as c:
                r = c.post(DAEMON_URL, json={"prompt": prompt, "model": model,
                                             "timeout_s": min(timeout - 5, 120.0), "use_cache": True})
            r.raise_for_status()
            return r.json()["text"]
        except httpx.HTTPStatusError as e:
            last = e
            if e.response.status_code in (429, 500, 502, 503, 504) and a < retries - 1:
                time.sleep(2 + a * 3); continue
            raise
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last = e
            if a < retries - 1:
                time.sleep(2 + a * 3); continue
            raise
    raise last


PROMPT = """These carbon-credit methodologies all govern the SAME mitigation activity: "{label}".
A project developer must choose ONE. They are issued by different registries and used in different
regions. Using each methodology's eligibility text AND its real-world registration footprint,
identify the NON-TECHNICAL factors that decide which one a project chooses — i.e. factors BEYOND
the technical/eligibility match (which is similar across them).

Consider: target carbon market (voluntary vs compliance; CORSIA / Article 6 eligibility),
host-country & regulatory conditions, registry strategy and regional presence, additionality /
baseline stringency, buyer or program targeting, suppressed-demand / development-context framing.

# Methodologies
{members}

Return ONLY a JSON object:
{{
  "differentiators": [{{"factor": "<short name>", "splits_how": "<which members differ and the driver>"}}],
  "member_profiles": {{"<code>": {{"target_market": "...", "region_rationale": "...", "registry_rationale": "...", "one_line": "..."}}}},
  "summary": "<one paragraph: what really decides the choice within this group>"
}}"""


def build_members_block(codes, cat, foot, anchor):
    lines = []
    for c in codes:
        e = cat.get(c, {})
        regs = "/".join(e.get("all_standards") or [e.get("primary_standard") or "?"])
        topr = ", ".join(f"{k}({v})" for k, v in foot.get(c, Counter()).most_common(4)) or "n/a"
        title = e.get("title") or anchor.get(c, {}).get("title") or c
        lines.append(f"## {c} — {title}\n"
                     f"registry={e.get('primary_standard')} adopted_by=[{regs}] "
                     f"scale={anchor.get(c, {}).get('corpus_scale') or '?'} status={e.get('status')}\n"
                     f"registration regions: {topr}\n"
                     f"eligibility excerpt: {elig_excerpt(c)}")
    return "\n\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["G03", "G05", "G06", "G15", "G09"])
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--out", type=Path, default=M / "group-context-factors.json")
    ap.add_argument("--md", type=Path, default=ICDM_ROOT / "paper" / "sections" / "group-context-factors-auto.md")
    args = ap.parse_args()

    gmap = json.loads((M / "method-groups.json").read_text())
    anchor = gmap["anchor_evidence"]
    labels = gmap["group_labels"]
    cat = {e["code"]: e for e in json.loads((M / "genvision-methodology-catalog.json").read_text())["entries"]}
    entries = json.loads((M / "genvision-pdd-eval-set.json").read_text())["entries"]
    gt = Counter(p["gt"] for p in json.loads((M / "icdm2026-splits.json").read_text())["test"]["pdds"])

    # registration footprint (region) per code
    foot = defaultdict(Counter)
    for e in entries:
        c = e.get("country")
        for m in (e.get("methodologies") or []):
            foot[m][region(c)] += 1

    out = {}
    if args.out.exists():
        out = json.loads(args.out.read_text())

    for g in args.groups:
        members = [c for c in gmap["groups"].get(g, []) if gt.get(c)]
        members.sort(key=lambda c: -gt[c])
        if not members:
            print(f"[skip] {g}: no GT members"); continue
        block = build_members_block(members, cat, foot, anchor)
        prompt = PROMPT.format(label=labels.get(g, g), members=block)
        print(f"[{g}] {len(members)} members -> daemon ({args.model})")
        try:
            text = daemon_post(prompt, args.model)
            mm = re.search(r"\{.*\}", text, re.DOTALL)
            parsed = json.loads(mm.group(0)) if mm else {"error": "no-json", "raw": text[:300]}
        except Exception as e:
            parsed = {"error": str(e)[:200]}
        out[g] = {"label": labels.get(g, g), "members": members, **parsed}
        args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # render md from ALL groups present in the json (not just this run's --groups)
    L = ["# Per-group NON-technical context factors (auto, Part 2: LLM doc comparison)\n"]
    for g in sorted(out):
        d = out.get(g)
        if not d:
            continue
        L.append(f"## {g} — {d.get('label')}  (members: {', '.join(d.get('members', []))})\n")
        if "error" in d:
            L.append(f"_error: {d['error']}_\n"); continue
        L.append(f"**Summary**: {d.get('summary','')}\n")
        L.append("**Differentiators:**")
        for f in d.get("differentiators", []):
            L.append(f"- **{f.get('factor')}** — {f.get('splits_how')}")
        L.append("\n**Member profiles:**")
        L.append("| code | target market | region rationale | registry rationale |")
        L.append("|---|---|---|---|")
        for c, p in (d.get("member_profiles") or {}).items():
            L.append(f"| {c} | {p.get('target_market','')} | {p.get('region_rationale','')} | {p.get('registry_rationale','')} |")
        L.append("")
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text("\n".join(L))
    print(f"\nDONE -> {args.out}\n        {args.md}")


if __name__ == "__main__":
    main()
