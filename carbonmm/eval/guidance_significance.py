"""Significance + table for the guidance-KG ablation (C3).

Compares guidance CONDITIONS within one model's ablation dir against the
`baseline` condition, on the SAME paired PDDs. The C3 claim is whether injecting
registry guidance converts recall→top1 (closing C1's weak-conversion gap): we
report Δtop1(cond − baseline) with the McNemar continuity-corrected test, a paired
bootstrap CI, and Wilson marginals — identical machinery to the C1 capability
significance (`capability_significance`), just across conditions rather than arms.

Also breaks out the two diagnostic splits the C2/C3 story rests on:
  - capacity_stated vs capacity_absent  (the information ceiling)
  - G01_G02                              (scale-sensitive energy activities)

Validation: run against the backed-up gpt-4o ablation — it must reproduce
the reference figures "all Δ+0.035 (McNemar p=.009); filter +0.030 (p=.024); scale +0.000
(p=.88); clauses +0.022 (p=.090); cap-absent 0.404→0.446 (+0.042)".

Runs locally (the ablation result directory must be present):
    python3 -m carbonmm.eval.guidance_significance \
        --dir graphrag-v281-openai-gpt-4o --md out.md
    python3 -m carbonmm.eval.guidance_significance \
        --dir graphrag-v281-sonnet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .capability_significance import wilson, mcnemar, paired_boot_ci

ICDM = Path(__file__).resolve().parents[1]
ABL = ICDM / "results" / "guidance-ablation"
MANIFEST = ICDM / "data" / "manifests"

ALL_CONDITIONS = ["baseline", "filter", "scale", "clauses", "all"]
SPLITS = ["overall", "capacity_stated", "capacity_absent", "G01_G02"]


def _load_cond(dirpath: Path, cond: str) -> dict | None:
    p = dirpath / f"{cond}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _correct(per_pdd: list, metric: str = "top1") -> dict[str, int]:
    if metric == "top5":
        return {r["gid"]: int(r["gt_rank"] is not None) for r in per_pdd}
    return {r["gid"]: int(r["gt_rank"] == 1) for r in per_pdd}


def _n_empty(per_pdd: list) -> int:
    return sum(1 for r in per_pdd if not r.get("predicted_top5"))


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


def _subset_gids(per_pdd: list, cap_map: dict, c2g: dict, which: str) -> set[str]:
    if which == "capacity_stated":
        return {r["gid"] for r in per_pdd if cap_map.get(r["gid"], {}).get("capacity_mw") is not None}
    if which == "capacity_absent":
        return {r["gid"] for r in per_pdd if cap_map.get(r["gid"], {}).get("capacity_mw") is None}
    if which == "G01_G02":
        return {r["gid"] for r in per_pdd if c2g.get(r["gt_label"]) in ("G01", "G02")}
    return {r["gid"] for r in per_pdd}  # overall


def _compare(base_c: dict, cond_c: dict, gids: set, boot: int) -> dict | None:
    gids = sorted(gids & set(base_c) & set(cond_c))
    n = len(gids)
    if n == 0:
        return None
    kb = sum(base_c[g] for g in gids)
    kc = sum(cond_c[g] for g in gids)
    b = sum(1 for g in gids if base_c[g] and not cond_c[g])
    c = sum(1 for g in gids if not base_c[g] and cond_c[g])
    chi, p = mcnemar(b, c)
    nc = {g: base_c[g] for g in gids}
    vc = {g: cond_c[g] for g in gids}
    pt, dl, dh = paired_boot_ci(nc, vc, B=boot)
    bl, bh = wilson(kb, n)
    cl, ch = wilson(kc, n)
    return {"n": n, "base": kb / n, "base_ci": (bl, bh), "cond": kc / n, "cond_ci": (cl, ch),
            "delta": kc / n - kb / n, "delta_ci": (dl, dh), "mcnemar_p": p, "b": b, "c": c}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="model dir under results/guidance-ablation/")
    ap.add_argument("--conditions", default="filter,scale,clauses,all")
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--md", type=Path, default=None)
    args = ap.parse_args()

    dirpath = ABL / args.dir
    if not dirpath.exists():
        raise SystemExit(f"no such dir: {dirpath}")
    base = _load_cond(dirpath, "baseline")
    if base is None:
        raise SystemExit(f"no baseline.json in {dirpath} (run incomplete?)")

    cap_map, c2g = _load_capacity(), _load_groups()
    base_pp = base["per_pdd"]
    n_base = base.get("n_actual", len(base_pp))
    base_top1 = _correct(base_pp, "top1")
    base_top5 = _correct(base_pp, "top5")

    conds = [c.strip() for c in args.conditions.split(",") if c.strip() and c != "baseline"]
    loaded = {c: _load_cond(dirpath, c) for c in conds}
    present = [c for c in conds if loaded[c] is not None]
    missing = [c for c in conds if loaded[c] is None]

    print(f"# guidance C3 significance — {args.dir}")
    print(f"baseline: top1={base['metrics']['top1']:.3f} top5={base['metrics']['top5']:.3f} "
          f"mrr={base['metrics']['mrr']:.3f}  n={n_base}  empty={_n_empty(base_pp)}")
    if missing:
        print(f"(pending conditions: {missing})")
    print()

    md_rows = [
        "| condition | top1 | Δtop1 [95% CI] | McNemar p | top5 | Δtop5 | "
        "cap-absent Δtop1 (p) | G01/G02 Δtop1 (p) | empty |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| baseline | {base['metrics']['top1']:.3f} | — | — | "
        f"{base['metrics']['top5']:.3f} | — | — | — | {_n_empty(base_pp)} |",
    ]

    for cond in present:
        cpp = loaded[cond]["per_pdd"]
        cond_top1 = _correct(cpp, "top1")
        cond_top5 = _correct(cpp, "top5")
        ov = _compare(base_top1, cond_top1, _subset_gids(base_pp, cap_map, c2g, "overall"), args.boot)
        ov5 = _compare(base_top5, cond_top5, _subset_gids(base_pp, cap_map, c2g, "overall"), args.boot)
        ab = _compare(base_top1, cond_top1, _subset_gids(base_pp, cap_map, c2g, "capacity_absent"), args.boot)
        st = _compare(base_top1, cond_top1, _subset_gids(base_pp, cap_map, c2g, "capacity_stated"), args.boot)
        gg = _compare(base_top1, cond_top1, _subset_gids(base_pp, cap_map, c2g, "G01_G02"), args.boot)

        sig = "  *" if ov and ov["mcnemar_p"] < 0.05 else ""
        print(f"## {cond}")
        print(f"  top1 {ov['base']:.3f}→{ov['cond']:.3f}  Δ{ov['delta']:+.3f} "
              f"[{ov['delta_ci'][0]:+.3f},{ov['delta_ci'][1]:+.3f}]  "
              f"McNemar b={ov['b']} c={ov['c']} p={ov['mcnemar_p']:.3f}{sig}")
        print(f"  top5 {ov5['base']:.3f}→{ov5['cond']:.3f}  Δ{ov5['delta']:+.3f} p={ov5['mcnemar_p']:.3f}")
        if st:
            print(f"  cap-stated  {st['base']:.3f}→{st['cond']:.3f}  Δ{st['delta']:+.3f} "
                  f"p={st['mcnemar_p']:.3f}  (n={st['n']})")
        if ab:
            print(f"  cap-absent  {ab['base']:.3f}→{ab['cond']:.3f}  Δ{ab['delta']:+.3f} "
                  f"p={ab['mcnemar_p']:.3f}  (n={ab['n']})  <- information ceiling")
        if gg:
            print(f"  G01/G02     {gg['base']:.3f}→{gg['cond']:.3f}  Δ{gg['delta']:+.3f} "
                  f"p={gg['mcnemar_p']:.3f}  (n={gg['n']})")
        print()

        ab_cell = f"{ab['delta']:+.3f} ({ab['mcnemar_p']:.3f})" if ab else "—"
        gg_cell = f"{gg['delta']:+.3f} ({gg['mcnemar_p']:.3f})" if gg else "—"
        md_rows.append(
            f"| {cond} | {ov['cond']:.3f} | {ov['delta']:+.3f} "
            f"[{ov['delta_ci'][0]:+.3f},{ov['delta_ci'][1]:+.3f}] | {ov['mcnemar_p']:.3f}{' *' if ov['mcnemar_p']<0.05 else ''} | "
            f"{ov5['cond']:.3f} | {ov5['delta']:+.3f} | {ab_cell} | {gg_cell} | {_n_empty(cpp)} |"
        )

    if args.md:
        args.md.write_text("\n".join(md_rows) + "\n")
        print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
