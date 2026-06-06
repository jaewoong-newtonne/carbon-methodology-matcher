"""Within-group disambiguation via a registry/region USAGE-PRIOR re-ranker (ICDM 2026).

Tests the hypothesis that for groups whose siblings are the SAME activity under
different registries/programs (e.g. G03 cookstove/thermal: GS-EN-002 vs AMS-II.G. vs
VMR0006), the discriminating signal is the MARKET/REGION choice, not technical eligibility.

Re-ranker (deterministic, NO LLM): within the base system's in-group top-k candidates,
re-rank by P(methodology | country, group), with country -> region -> group-global backoff.
Monotone-safe: re-orders only within the slots the in-group candidates already occupy
(never introduces an unseen code).

LEAKAGE CONTROL: the prior is built ONLY from registration records whose globalId is NOT
in val ∪ test. The test PDD's own `country` is an observable input feature (not the label),
so using it at query time is legitimate.

Usage:
    python eval/run_usage_prior_poc.py --group G03
    python eval/run_usage_prior_poc.py --group G01   # contrast: should NOT help
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ICDM_ROOT = Path(__file__).resolve().parents[1]
M = ICDM_ROOT / "data" / "manifests"
RESULTS = ICDM_ROOT / "results"

# coarse ISO3 -> region (backoff only; country is the primary key)
REGION = {}
for codes, r in [
    (["IND", "PAK", "BGD", "NPL", "LKA", "AFG", "BTN"], "south-asia"),
    (["CHN", "MNG", "KOR", "JPN"], "east-asia"),
    (["VNM", "KHM", "LAO", "MMR", "IDN", "PHL", "THA", "MYS", "TLS"], "se-asia"),
    (["TUR", "EGY", "MAR", "JOR", "SAU", "ARE", "QAT", "IRQ", "IRN", "TUN", "DZA", "LBN", "OMN", "YEM", "KWT", "BHR"], "mena"),
    (["UGA", "KEN", "ETH", "TZA", "RWA", "MWI", "MOZ", "MDG", "ZMB", "ZWE", "BDI", "SOM", "SSD", "ERI"], "east-africa"),
    (["NGA", "GHA", "SEN", "CIV", "MLI", "BFA", "BEN", "TGO", "NER", "GIN", "SLE", "LBR", "GMB"], "west-africa"),
    (["ZAF", "BWA", "NAM", "LSO", "SWZ"], "southern-africa"),
    (["COD", "CMR", "CAF", "COG", "GAB", "TCD", "AGO"], "central-africa"),
    (["BRA", "MEX", "COL", "PER", "CHL", "ARG", "ECU", "BOL", "GTM", "HND", "NIC", "CRI", "PAN", "PRY", "URY", "DOM", "HTI"], "lac"),
    (["USA", "CAN"], "north-america"),
]:
    for c in codes:
        REGION[c] = r


def region(c: str) -> str:
    return REGION.get((c or "").upper(), "other")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="G03")
    ap.add_argument("--systems", nargs="+",
                    default=["graphrag-fusion-only", "naive-rag", "graphrag-v28-openai", "graphrag-v281-sonnet"])
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="blend weight on BASE order (1=base/no-op, 0=pure prior re-rank)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    gmap = json.loads((M / "method-groups.json").read_text())
    c2g = gmap["code_to_group"]
    label = gmap["group_labels"].get(args.group, args.group)

    def group_of(code: str) -> str:
        return c2g.get(code, f"SINGLETON::{code}")

    entries = json.loads((M / "genvision-pdd-eval-set.json").read_text())["entries"]
    splits = json.loads((M / "icdm2026-splits.json").read_text())
    eval_gids = {p["gid"] for p in splits["val"]["pdds"]} | {p["gid"] for p in splits["test"]["pdds"]}
    gid2country = {e["globalId"]: e.get("country") for e in entries}

    # ── build leakage-free usage prior ──
    prior_c = defaultdict(lambda: defaultdict(Counter))   # [group][country][method]
    prior_r = defaultdict(lambda: defaultdict(Counter))   # [group][region][method]
    prior_g = defaultdict(Counter)                        # [group][method]
    n_train = 0
    for e in entries:
        if e["globalId"] in eval_gids:
            continue
        n_train += 1
        c = e.get("country")
        for m in (e.get("methodologies") or []):
            g = group_of(m)
            prior_g[g][m] += 1
            if c:
                prior_c[g][c][m] += 1
                prior_r[g][region(c)][m] += 1

    def prior_scores(G, country, cands):
        pc = prior_c[G].get(country, Counter())
        if any(pc.get(m, 0) for m in cands):
            return {m: pc.get(m, 0) for m in cands}, "country"
        pr = prior_r[G].get(region(country), Counter())
        if any(pr.get(m, 0) for m in cands):
            return {m: pr.get(m, 0) for m in cands}, "region"
        pg = prior_g[G]
        return {m: pg.get(m, 0) for m in cands}, "global"

    L = [f"# Within-group usage-prior re-ranker — {args.group} ({label})\n",
         f"Leakage-free prior from {n_train} non-eval registration records. "
         f"Group-{args.group} training methodology-occurrences: {sum(prior_g[args.group].values())}.\n",
         "| System | group PDDs | reranked | exact top-1 base | + usage-prior | Δ | fixed/broken |",
         "|---|---|---|---|---|---|---|"]

    for system in args.systems:
        files = sorted(RESULTS.glob(f"{system}-test-n*-seed*.json"))
        if not files:
            L.append(f"| {system} | (no result file) | | | | | |")
            continue
        result = json.loads(files[-1].read_text())
        g_pdds = [r for r in result["per_pdd"] if group_of(r["gt_label"]) == args.group]
        base_c = new_c = rer = fixed = broken = 0
        for r in g_pdds:
            gt = r["gt_label"]
            codes = list(r.get("predicted_top5", []))
            country = gid2country.get(r["gid"])
            base_top1 = bool(codes) and codes[0] == gt
            base_c += base_top1
            idx = [i for i, c in enumerate(codes) if group_of(c) == args.group]
            cand = [codes[i] for i in idx]
            new_codes = codes
            if len(cand) >= 2:
                pscores, _lvl = prior_scores(args.group, country, cand)
                pmax = max(pscores.values()) or 1
                nc = len(cand)
                # combined = alpha*base_norm + (1-alpha)*prior_norm; ties keep base order
                def combined(j):
                    base_norm = (nc - j) / nc
                    prior_norm = pscores[cand[j]] / pmax
                    return args.alpha * base_norm + (1 - args.alpha) * prior_norm
                order = sorted(range(nc), key=lambda j: (-combined(j), j))
                new_cand = [cand[j] for j in order]
                if new_cand != cand:
                    rer += 1
                    new_codes = list(codes)
                    for slot, c in zip(idx, new_cand):
                        new_codes[slot] = c
            new_top1 = bool(new_codes) and new_codes[0] == gt
            new_c += new_top1
            fixed += (new_top1 and not base_top1)
            broken += (base_top1 and not new_top1)
        n = len(g_pdds)
        b = base_c / n if n else 0
        a = new_c / n if n else 0
        L.append(f"| {system} | {n} | {rer} | {b:.3f} | **{a:.3f}** | {a-b:+.3f} | {fixed}/{broken} |")

    out = "\n".join(L)
    print(out)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(out)
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
