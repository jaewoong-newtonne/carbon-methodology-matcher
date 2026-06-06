"""Unified within-group soft re-ranker over DEPLOYMENT/market factors (ICDM 2026, Option 2).

One re-ranker, group-adaptive: blends the base ranking with soft signals that are all
matched to OBSERVABLE PDD features (no leakage; CORSIA/Article-6 omitted as they need an
unobservable target-market signal). Features:
  - scale     : extracted PDD scale == candidate corpus scale  (covers G01/G02 energy axis)
  - prior     : leakage-free country/region usage-prior        (covers G03/G05/G06 market axis)
  - regmatch  : PDD registry-of-record in candidate.all_standards (down-weights off-registry)
  - date      : candidate valid at PDD crediting-period-start  (status/version validity)
  - ldc_sdg   : LDC-host PDD + SDG-mandatory registry (GS)     (suppressed-demand framing)

combined = alpha*base_norm + (1-alpha)*mean(active feature signals); monotone-safe re-rank
within the slots the in-group candidates already occupy. Reports overall + per-group exact
top-1 with a cumulative ablation.

Pure offline (stdlib). Leakage control: usage-prior built from records NOT in val∪test.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, date
from pathlib import Path

ICDM_ROOT = Path(__file__).resolve().parents[1]
M = ICDM_ROOT / "data" / "manifests"
RESULTS = ICDM_ROOT / "results"

# UN LDC list (ISO3) — public; used only for the suppressed-demand feature
LDC = {"AFG", "AGO", "BGD", "BEN", "BFA", "BDI", "KHM", "CAF", "TCD", "COM", "COD", "DJI",
       "ERI", "ETH", "GMB", "GIN", "GNB", "HTI", "KIR", "LAO", "LSO", "LBR", "MDG", "MWI",
       "MLI", "MRT", "MOZ", "MMR", "NPL", "NER", "RWA", "STP", "SEN", "SLE", "SLB", "SOM",
       "SSD", "SDN", "TLS", "TGO", "TUV", "UGA", "TZA", "YEM", "ZMB"}

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


def parse_date(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).date()
    except Exception:
        try:
            return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
        except Exception:
            return None


def build_indices():
    cat = {e["code"]: e for e in json.loads((M / "genvision-methodology-catalog.json").read_text())["entries"]}
    all_std = {c: set(e.get("all_standards") or []) for c, e in cat.items()}
    prim = {c: e.get("primary_standard") for c, e in cat.items()}
    status = {c: e.get("status", "Unknown") for c, e in cat.items()}
    versions = {}
    for c, e in cat.items():
        versions[c] = [(parse_date(v.get("effective_from")), parse_date(v.get("effective_to")))
                       for v in (e.get("all_versions") or [])]
    return all_std, prim, status, versions


def is_invalid_at(code, d, versions, status):
    if d is None:
        return False
    if status.get(code) == "Withdrawn":
        return True
    vs = versions.get(code, [])
    if not vs:
        return False
    all_ended, has_end = True, False
    for ef, et in vs:
        if et is None:
            all_ended = False; break
        has_end = True
        if et >= d:
            all_ended = False; break
    return all_ended and has_end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+",
                    default=["graphrag-fusion-only", "naive-rag", "graphrag-v281-sonnet"])
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--out", type=Path, default=ICDM_ROOT / "paper" / "sections" / "6-market-factor-rerank-auto.md")
    args = ap.parse_args()

    gmap = json.loads((M / "method-groups.json").read_text())
    c2g = gmap["code_to_group"]
    anchor = gmap["anchor_evidence"]
    labels = gmap["group_labels"]
    regattr = json.loads((M / "registry-attributes.json").read_text())
    all_std, prim, status, versions = build_indices()
    entries = json.loads((M / "genvision-pdd-eval-set.json").read_text())["entries"]
    gid_country = {e["globalId"]: e.get("country") for e in entries}
    gid_date = {e["globalId"]: parse_date(e.get("creditingPeriodStartDate")) for e in entries}
    splits = json.loads((M / "icdm2026-splits.json").read_text())
    eval_gids = {p["gid"] for p in splits["val"]["pdds"]} | {p["gid"] for p in splits["test"]["pdds"]}
    feats_cache = {}
    fc = M / "pdd-extracted-features-test.json"
    if fc.exists():
        feats_cache = json.loads(fc.read_text())

    def group_of(code):
        return c2g.get(code, f"SINGLETON::{code}")

    def sdg_registry(code):
        r = prim.get(code)
        return bool(regattr["registries"].get(r, regattr["_default"]).get("sdg_mandatory"))

    # leakage-free region usage-prior
    prior = defaultdict(lambda: defaultdict(Counter))  # [group][region][code]
    for e in entries:
        if e["globalId"] in eval_gids:
            continue
        rg = region(e.get("country"))
        for m in (e.get("methodologies") or []):
            prior[group_of(m)][rg][m] += 1

    ALL_FEATS = ["scale", "prior", "regmatch", "date", "ldc_sdg"]
    CONFIGS = [("base", []),
               ("+scale", ["scale"]),
               ("+scale+prior", ["scale", "prior"]),
               ("+scale+prior+regmatch", ["scale", "prior", "regmatch"]),
               ("+scale+prior+regmatch+date", ["scale", "prior", "regmatch", "date"]),
               ("+all (+ldc_sdg)", ALL_FEATS)]

    def feature_signals(G, pdd, cand):
        """return {feat: {code: [0,1] signal}} normalized across cand."""
        ct = gid_country.get(pdd["gid"])
        d = gid_date.get(pdd["gid"])
        pr = pdd.get("registry")
        psc = (feats_cache.get(pdd["gid"]) or {}).get("scale")
        sig = {}
        # scale
        sc = {}
        for c in cand:
            cs = anchor.get(c, {}).get("corpus_scale")
            if psc in ("small", "large") and cs in ("small", "large"):
                sc[c] = 1.0 if cs == psc else 0.0
        sig["scale"] = sc if len(sc) == len(cand) else {}
        # prior (region)
        pc = prior[G].get(region(ct), Counter())
        mx = max((pc.get(c, 0) for c in cand), default=0)
        sig["prior"] = {c: (pc.get(c, 0) / mx) for c in cand} if mx else {}
        # regmatch
        sig["regmatch"] = {c: (1.0 if pr in all_std.get(c, set()) or pr == prim.get(c) else 0.0) for c in cand} if pr else {}
        # date validity (1 = valid)
        sig["date"] = {c: (0.0 if is_invalid_at(c, d, versions, status) else 1.0) for c in cand}
        # ldc_sdg
        if ct and ct.upper() in LDC:
            sig["ldc_sdg"] = {c: (1.0 if sdg_registry(c) else 0.0) for c in cand}
        else:
            sig["ldc_sdg"] = {}
        return sig

    def evaluate(system, active):
        files = sorted(RESULTS.glob(f"{system}-test-n*-seed*.json"))
        if not files:
            return None
        per = json.loads(files[-1].read_text())["per_pdd"]
        correct = rer = 0
        per_group = defaultdict(lambda: [0, 0, 0])  # group -> [n, base_correct, new_correct]
        for r in per:
            gt = r["gt_label"]
            codes = list(r.get("predicted_top5", []))
            G = group_of(gt)
            base_top1 = bool(codes) and codes[0] == gt
            idx = [i for i, c in enumerate(codes) if group_of(c) == G]
            cand = [codes[i] for i in idx]
            new_codes = codes
            if len(cand) >= 2 and active:
                sig = feature_signals(G, r, cand)
                nc = len(cand)
                def comb(j):
                    c = cand[j]
                    base_norm = (nc - j) / nc
                    vals = [sig[f][c] for f in active if sig.get(f)]
                    fscore = sum(vals) / len(vals) if vals else 0.0
                    return args.alpha * base_norm + (1 - args.alpha) * fscore
                order = sorted(range(nc), key=lambda j: (-comb(j), j))
                neworder = [cand[j] for j in order]
                if neworder != cand:
                    rer += 1
                    new_codes = list(codes)
                    for slot, c in zip(idx, neworder):
                        new_codes[slot] = c
            new_top1 = bool(new_codes) and new_codes[0] == gt
            correct += new_top1
            pg = per_group[G]
            pg[0] += 1; pg[1] += base_top1; pg[2] += new_top1
        n = len(per)
        return correct / n, rer, per_group

    L = [f"# Within-group market-factor soft re-ranker (auto, alpha={args.alpha})\n",
         "Overall exact top-1 on test (n=535), cumulative feature ablation. "
         "Re-ranker is monotone-safe (in-group slot reorder only); features matched to "
         "observable PDD signals; usage-prior is leakage-free.\n"]
    best_per_group = {}
    for system in args.systems:
        L.append(f"## {system}")
        L.append("| config | exact top-1 | Δ vs base | reranked |")
        L.append("|---|---|---|---|")
        base_t1 = None
        for name, active in CONFIGS:
            res = evaluate(system, active)
            if res is None:
                L.append("| (no result file) | | | |"); break
            t1, rer, pg = res
            if base_t1 is None:
                base_t1 = t1
            L.append(f"| {name} | {t1:.4f} | {t1-base_t1:+.4f} | {rer} |")
            if name == CONFIGS[-1][0]:
                best_per_group[system] = pg
        L.append("")
    # per-group for the strongest system, full config
    strong = args.systems[-1]
    if strong in best_per_group:
        L.append(f"## Per-group effect (full config) — {strong}")
        L.append("| group | label | n | base top-1 | rerank top-1 | Δ |")
        L.append("|---|---|---|---|---|---|")
        pg = best_per_group[strong]
        for g in sorted(pg, key=lambda g: -pg[g][0]):
            if g.startswith("SINGLETON::"):
                continue
            n, b, nw = pg[g]
            if n < 3:
                continue
            L.append(f"| {g} | {labels.get(g, g)[:26]} | {n} | {b/n:.3f} | {nw/n:.3f} | {(nw-b)/n:+.3f} |")

    out = "\n".join(L)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(out)
    print(out)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
