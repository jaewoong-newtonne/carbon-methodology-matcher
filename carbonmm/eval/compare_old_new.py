"""OLD (full_text[:6000]) vs NEW (rich) guidance Δtop1 on the pilot gids, per model.
OLD = existing 535 runs subset to the pilot; NEW = rich-text runs."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from importlib import import_module

# Ensure repo root is on sys.path so "carbonmm.eval.*" imports work
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

ICDM = Path(__file__).resolve().parents[1]
ABL = ICDM / "results" / "guidance-ablation"
_gs = import_module("carbonmm.eval.guidance_significance")


def _correct(dir_name: str, cond: str, gids: set) -> dict:
    pp = json.loads((ABL / dir_name / f"{cond}.json").read_text())["per_pdd"]
    return {r["gid"]: int(r["gt_rank"] == 1) for r in pp if r["gid"] in gids}


def _mcnemar_p(b: int, c: int) -> float:
    res = _gs.mcnemar(b, c)
    return res[1] if isinstance(res, (tuple, list)) else res   # mcnemar returns (chi2, p)


def _delta(dir_name: str, gids: set):
    b = _correct(dir_name, "baseline", gids)
    a = _correct(dir_name, "all", gids)
    common = sorted(set(b) & set(a))
    n = len(common)
    if n == 0:
        return None
    kb = sum(b[g] for g in common) / n
    ka = sum(a[g] for g in common) / n
    bb = sum(1 for g in common if b[g] and not a[g])
    cc = sum(1 for g in common if not b[g] and a[g])
    return {"n": n, "base": kb, "all": ka, "delta": ka - kb, "mcnemar_p": _mcnemar_p(bb, cc)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ICDM / "data/manifests/pilot-150.json")
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="model=OLDDIR,NEWDIR, e.g. gpt-4o=graphrag-v281-openai-gpt-4o,graphrag-v281-openai-gpt-4o-rich150")
    a = ap.parse_args()
    gids = {r["gid"] for r in json.loads(a.manifest.read_text())["pdds"]}
    print(f"{'model':9s} {'arm':4s} {'n':>4s} {'base':>6s} {'all':>6s} {'delta':>7s} {'p':>7s}")
    print("-" * 50)
    for spec in a.pairs:
        model, dirs = spec.split("=", 1)
        old_dir, new_dir = dirs.split(",", 1)
        for arm, d in (("OLD", old_dir), ("NEW", new_dir)):
            r = _delta(d, gids)
            if r:
                print(f"{model:9s} {arm:4s} {r['n']:4d} {r['base']:6.3f} {r['all']:6.3f} "
                      f"{r['delta']:+7.3f} {r['mcnemar_p']:7.3f}")
    print("\nGO if NEW delta clearly exceeds OLD (spec 3.6) with leakage ~0.")


if __name__ == "__main__":
    main()
