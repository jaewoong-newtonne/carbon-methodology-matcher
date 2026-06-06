"""Analyze whether methodology version dates (from Genvision catalog) +
PDD creditingPeriodStartDate enables a high-precision hard-filter that
prunes invalid candidates and improves top-1.

Hypothesis: a PDD registered with creditingPeriodStartDate=D cannot legally
use a methodology that has no version effective at D. This is a HARD
CONSTRAINT — methodology lifecycle is regulatory, not statistical.

Output:
  1. GT pass-rate under date filter (should be ~100% if filter is correct)
  2. Effective candidate space reduction per PDD (mean / median / min)
  3. Counterfactual lift if router predictions are filtered by date validity
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, date
from pathlib import Path

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


# 1. Load catalog → code → [(effective_from, effective_to), ...]
print("=== Loading Genvision catalog ===")
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

print(f"  Catalog has {len(code_to_versions)} codes")
n_with_versions = sum(1 for v in code_to_versions.values() if v)
print(f"  {n_with_versions} have at least 1 version with date metadata")

# Sample stats
status_counts = Counter(code_to_status.values())
print(f"  Status distribution: {dict(status_counts.most_common(10))}")


def is_valid_at(code: str, target_date: date) -> bool:
    """True if methodology has at least one version covering target_date."""
    versions = code_to_versions.get(code, [])
    if not versions:
        # No version metadata = unknown; default permissive
        return True
    for ef, et in versions:
        if ef is None:
            # Treat as always-effective from inception
            ef = date(1900, 1, 1)
        if et is None:
            # Still active
            et = date(2099, 12, 31)
        if ef <= target_date <= et:
            return True
    return False


# 2. Load test PDDs (extract creditingPeriodStartDate from manifest)
manifest = json.load(open(MANIFESTS / 'pdd-eval-set-restricted.json'))
print(f"\n=== Loading test PDDs ===")
print(f"Manifest type: {type(manifest).__name__}, keys: {list(manifest.keys())[:8] if isinstance(manifest, dict) else 'list'}")

# Need to map gid → creditingPeriodStartDate. Load from PDD body files.
print("  Loading PDD body files for creditingPeriodStartDate ...")
EVAL = Path('carbonmm/data/eval-pdds')
nr = json.load(open(RESULTS / 'naive-rag-test-n535-seed42.json'))
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

print(f"  {len(gid_to_start)}/{len(test_gids)} test PDDs have creditingPeriodStartDate")

# 3. GT pass-rate under date filter (sanity)
gt_pass = 0
gt_fail = 0
gt_fail_codes = Counter()
gt_unknown = 0
for p in nr['per_pdd']:
    gid = p['gid']
    gt = p['gt_label']
    if gid not in gid_to_start:
        gt_unknown += 1
        continue
    if is_valid_at(gt, gid_to_start[gid]):
        gt_pass += 1
    else:
        gt_fail += 1
        gt_fail_codes[gt] += 1

print(f"\n=== GT validity under date filter ===")
print(f"  GT date-valid : {gt_pass} ({gt_pass/535*100:.1f}%)")
print(f"  GT date-INVALID: {gt_fail} ({gt_fail/535*100:.1f}%)")
print(f"  GT/PDD missing date: {gt_unknown}")
print(f"\n  Top GTs that fail date filter (would be hard-excluded but shouldn't):")
for code, c in gt_fail_codes.most_common(10):
    versions = code_to_versions.get(code, [])
    print(f"    {code:14s} {c:3d} fails | versions: {versions[:3]} | status: {code_to_status.get(code, '?')}")


# 4. Candidate-space reduction
codes_in_test_gt = set(p['gt_label'] for p in nr['per_pdd'])
all_codes = set(code_to_versions.keys())
print(f"\n=== Candidate space ===")
print(f"  All catalog codes: {len(all_codes)}")
print(f"  Test GT codes: {len(codes_in_test_gt)}")

reductions = []
for gid, d in gid_to_start.items():
    valid = [c for c in all_codes if is_valid_at(c, d)]
    reductions.append(len(valid))
if reductions:
    print(f"  Valid codes per PDD (mean): {sum(reductions)/len(reductions):.0f}")
    print(f"  Valid codes per PDD (min):  {min(reductions)}")
    print(f"  Valid codes per PDD (max):  {max(reductions)}")


# 5. Counterfactual: apply filter to v2.5 + naive-rag predictions
v25 = json.load(open(RESULTS / 'graphrag-v25-test-n535-seed42.json'))
v25_by = {p['gid']: p for p in v25['per_pdd']}

def filtered_top1(orig_top5, gid):
    if gid not in gid_to_start:
        return orig_top5[0] if orig_top5 else None
    d = gid_to_start[gid]
    for c in orig_top5:
        if is_valid_at(c, d):
            return c
    return orig_top5[0] if orig_top5 else None

filtered_results = {}
for name, source in [('naive-rag', nr), ('v2.5', v25)]:
    correct = 0
    for p in source['per_pdd']:
        gid = p['gid']
        if not p['predicted_top5']:
            continue
        ft1 = filtered_top1(p['predicted_top5'], gid)
        if ft1 == p['gt_label']:
            correct += 1
    filtered_results[name] = correct
    n = sum(1 for p in source['per_pdd'] if p['predicted_top5'])
    orig_top1 = sum(1 for p in source['per_pdd'] if p['gt_rank'] == 1)
    print(f"\n{name}: orig top-1 = {orig_top1}/{n} = {orig_top1/n:.4f}, filtered top-1 = {correct}/{n} = {correct/n:.4f}  (Δ={(correct-orig_top1)/n*100:+.1f}pp)")
