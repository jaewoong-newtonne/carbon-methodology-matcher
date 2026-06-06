"""Select a ~150-PDD stratified pilot: all leaked PDDs + non-leaked stratified by
(template x capacity). Emits data/manifests/pilot-150.json."""
from __future__ import annotations
import argparse, glob, json, random, sys
from collections import defaultdict
from pathlib import Path
from importlib import import_module

ICDM = Path(__file__).resolve().parents[1]

# sys.path guard: ensure repo root is importable so sibling packages resolve.
_REPO_ROOT = Path(__file__).resolve().parents[3]  # .../the project root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_lg = import_module("carbonmm.eval.leakage_gate")
_sa = import_module("carbonmm.ingest.extract_pdd_section_a")


def select_pilot(rows: list[dict], target: int, seed: int) -> list[dict]:
    """Return a pilot set:
    - All leaked PDDs are always included.
    - Non-leaked PDDs are stratified by (template, capacity), round-robin
      filled until `target` is reached (or all cells exhausted).
    - If leaked count >= target, all leaked are still returned (len > target).
    """
    rng = random.Random(seed)
    leaked = [r for r in rows if r["leaked"]]
    rest = [r for r in rows if not r["leaked"]]
    cells: dict[tuple, list] = defaultdict(list)
    for r in rest:
        cells[(r["template"], r["capacity"])].append(r)
    for v in cells.values():
        rng.shuffle(v)
    picked = list(leaked)
    keys = sorted(cells.keys())
    i = 0
    while len(picked) < target and any(cells[k] for k in keys):
        k = keys[i % len(keys)]
        if cells[k]:
            picked.append(cells[k].pop())
        i += 1
    return picked[:max(target, len(leaked))]


def _rows_from_repo() -> list[dict]:
    splits = json.loads((ICDM / "data/manifests/icdm2026-splits.json").read_text())
    cap_raw = json.loads((ICDM / "data/manifests/g01-capacity-extracted.json").read_text())
    cap_list = cap_raw.get("results", cap_raw) if isinstance(cap_raw, dict) else cap_raw
    capmap = {r["gid"]: r for r in cap_list if isinstance(r, dict) and "gid" in r}
    bj = {Path(p).parent.name: p for p in glob.glob(str(ICDM / "data/eval-pdds/*/*/*.body.json"))}
    rows = []
    for e in splits["test"]["pdds"]:
        g, gt, reg = e["gid"], e["gt"], e["registry"]
        d = json.loads(Path(bj[g]).read_text())
        full = d.get("full_text", "")
        leaked = _lg.code_leak_count(full[:6000], gt) > 0
        template = _sa.detect_template(_sa.strip_toc_entries(full))
        c = capmap.get(g, {})
        capacity = "stated" if c.get("capacity_mw") is not None else "absent"
        rows.append({"gid": g, "registry": reg, "gt": gt, "template": template,
                     "leaked": leaked, "capacity": capacity})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=ICDM / "data/manifests/pilot-150.json")
    a = ap.parse_args()
    rows = _rows_from_repo()
    sel = select_pilot(rows, a.target, a.seed)
    a.out.write_text(json.dumps({"n": len(sel), "seed": a.seed, "pdds": sel}, indent=2))
    from collections import Counter
    print("selected", len(sel), "| leaked", sum(r["leaked"] for r in sel))
    print("by template:", dict(Counter(r["template"] for r in sel)))
    print("by capacity:", dict(Counter(r["capacity"] for r in sel)))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
