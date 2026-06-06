"""Conservative date filter — only excludes methodologies that are clearly
INVALID at PDD's registration:
  1. status == "Withdrawn" → never valid (corpus deletion is regulatory final).
  2. effective_to set AND PDD date > effective_to → already deprecated when PDD registered.
  (no effective_from check — Genvision catalog is incomplete on early versions)

Apply to v2.5 and naive-rag predictions. Measure top-1 change.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, date
from pathlib import Path
from statistics import mean

RESULTS = Path('carbonmm/results')
MANIFESTS = Path('carbonmm/data/manifests')


def parse_date(s):
    if s is None or s == '': return None
    if isinstance(s, str):
        try:
            return datetime.fromisoformat(s.replace('Z', '+00:00')).date()
        except Exception:
            try:
                return datetime.strptime(s, "%Y-%m-%d").date()
            except Exception:
                return None
    return None


# Load catalog
gv = json.load(open(MANIFESTS / 'genvision-methodology-catalog.json'))
entries = gv['entries']

code_to_versions = {}
code_to_status = {}
for e in entries:
    code = e.get('code')
    if not code:
        continue
    code_to_status[code] = e.get('status', 'Unknown')
    versions = []
    for v in (e.get('all_versions') or []):
        ef = parse_date(v.get('effective_from'))
        et = parse_date(v.get('effective_to'))
        versions.append((ef, et))
    code_to_versions[code] = versions


def is_invalid_at(code: str, target_date: date) -> bool:
    """Return True iff methodology is CLEARLY invalid at target_date.

    Permissive: missing data = valid.
    Strict only on Withdrawn status OR effective_to-already-past.
    """
    status = code_to_status.get(code, 'Unknown')
    if status == 'Withdrawn':
        return True

    versions = code_to_versions.get(code, [])
    if not versions:
        return False  # no data → permissive

    # If ALL versions have effective_to set AND all of them < target_date,
    # then methodology is fully deprecated at PDD registration time.
    all_ended = True
    has_any_end_date = False
    for ef, et in versions:
        if et is None:
            all_ended = False
            break
        has_any_end_date = True
        if et >= target_date:
            all_ended = False
            break
    if all_ended and has_any_end_date:
        return True

    return False


# Load test PDDs
EVAL = Path('carbonmm/data/eval-pdds')
nr = json.load(open(RESULTS / 'naive-rag-test-n535-seed42.json'))
v25 = json.load(open(RESULTS / 'graphrag-v25-test-n535-seed42.json'))
test_gids = {p['gid'] for p in nr['per_pdd']}

gid_to_start = {}
for body_path in EVAL.rglob('*.body.json'):
    gid = body_path.stem.replace('.body', '')
    if gid not in test_gids:
        continue
    try:
        body = json.loads(body_path.read_text())
        d = parse_date(body.get('creditingPeriodStartDate'))
        if d:
            gid_to_start[gid] = d
    except Exception:
        pass

# 1. GT pass-rate
gt_invalid = 0
gt_invalid_codes = Counter()
for p in nr['per_pdd']:
    gid = p['gid']
    if gid not in gid_to_start:
        continue
    if is_invalid_at(p['gt_label'], gid_to_start[gid]):
        gt_invalid += 1
        gt_invalid_codes[p['gt_label']] += 1
print(f"=== Conservative filter: GT invalidity ===")
print(f"  GT marked invalid (false-negative): {gt_invalid}/535 = {gt_invalid/535*100:.1f}%")
print(f"  Top false-negatives:")
for code, c in gt_invalid_codes.most_common(8):
    print(f"    {code:14s}  {c}")

# 2. How many candidate codes get filtered out per PDD on average?
print(f"\n=== Filter aggressiveness ===")
v25_by = {p['gid']: p for p in v25['per_pdd']}
all_excluded = []
for p in nr['per_pdd']:
    gid = p['gid']
    if gid not in gid_to_start:
        continue
    d = gid_to_start[gid]
    # Count how many of top-5 get excluded
    nr_excluded = sum(1 for c in p['predicted_top5'] if is_invalid_at(c, d))
    v25_excluded = sum(1 for c in v25_by[gid]['predicted_top5'] if is_invalid_at(c, d))
    all_excluded.append((nr_excluded, v25_excluded))
nr_avg = mean(x[0] for x in all_excluded)
v25_avg = mean(x[1] for x in all_excluded)
print(f"  Average # of top-5 excluded by filter:")
print(f"    naive-rag: {nr_avg:.2f} / 5")
print(f"    v2.5     : {v25_avg:.2f} / 5")

# 3. Apply filter — take filtered top-1
def filtered_top1(orig_top5, gid):
    if gid not in gid_to_start:
        return orig_top5[0] if orig_top5 else None
    d = gid_to_start[gid]
    for c in orig_top5:
        if not is_invalid_at(c, d):
            return c
    return orig_top5[0] if orig_top5 else None

# Combined: keep filtered top-5 in order
def filtered_top5(orig_top5, gid, fill=None):
    if gid not in gid_to_start:
        return orig_top5
    d = gid_to_start[gid]
    valid = [c for c in orig_top5 if not is_invalid_at(c, d)]
    return valid

print(f"\n=== Filtered top-1 lift ===")
for name, source in [('naive-rag', nr), ('v2.5', v25)]:
    orig_correct = sum(1 for p in source['per_pdd'] if p['gt_rank'] == 1)
    filt_correct = 0
    for p in source['per_pdd']:
        gid = p['gid']
        if not p['predicted_top5']:
            continue
        ft1 = filtered_top1(p['predicted_top5'], gid)
        if ft1 == p['gt_label']:
            filt_correct += 1
    n = len(source['per_pdd'])
    print(f"  {name:12s}: orig top-1 = {orig_correct}/{n} = {orig_correct/n:.4f}")
    print(f"             filtered = {filt_correct}/{n} = {filt_correct/n:.4f}  (Δ={(filt_correct-orig_correct)/n*100:+.1f}pp)")

# 4. Also apply to router_final output
ROUTER_PATH = RESULTS / 'ensemble-router-test-n535-seed42.json'
if ROUTER_PATH.exists():
    router = json.load(open(ROUTER_PATH))
    orig = sum(1 for p in router['per_pdd'] if p['gt_rank'] == 1)
    filt = 0
    for p in router['per_pdd']:
        ft1 = filtered_top1(p['predicted_top5'], p['gid'])
        if ft1 == p['gt_label']:
            filt += 1
    n = len(router['per_pdd'])
    print(f"  router      : orig top-1 = {orig}/{n} = {orig/n:.4f}")
    print(f"             filtered = {filt}/{n} = {filt/n:.4f}  (Δ={(filt-orig)/n*100:+.1f}pp)")
