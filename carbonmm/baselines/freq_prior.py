"""Baseline 2: per-registry methodology frequency prior.

Approximates an empirical issuance distribution. This baseline uses the
restricted-PDD manifest's `methodology_label` field as a *train-set* proxy
(no formal train/test split yet — project-isolated splits are introduced
later).

The classifier predicts the top-K most frequent codes within the inferred
PDD registry (GS/VCS). If the registry can't be inferred from the PDD text,
fall back to the global frequency distribution.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .base import Baseline, code_to_registry, load_clauses_df

RESTRICTED_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "manifests"
    / "pdd-eval-set-restricted.json"
)

EVAL_PDD_ROOT = Path(__file__).resolve().parents[1] / "data" / "eval-pdds"


def _empirical_labels_by_registry() -> dict[str, Counter]:
    """Read methodology_label out of each retained PDD body.json."""
    out: dict[str, Counter] = {"GS": Counter(), "VCS": Counter()}
    if not RESTRICTED_MANIFEST.exists():
        return out
    m = json.loads(RESTRICTED_MANIFEST.read_text())
    kept = m.get("globalIds") or m.get("by_class_globalIds", {}).get("single", [])
    if not kept:
        # different schema flavor
        kept = []
        for cls_ids in m.get("by_class_globalIds", {}).values():
            kept.extend(cls_ids)
    for gid in kept:
        reg = "GS" if gid.startswith("GS") else "VCS" if gid.startswith("VCS") else None
        if reg is None:
            continue
        body_path = EVAL_PDD_ROOT / reg / gid / f"{gid}.body.json"
        if not body_path.exists():
            continue
        try:
            d = json.loads(body_path.read_text())
            label = d.get("methodology_label")
            if label:
                out[reg][label] += 1
        except Exception:
            pass
    return out


class FreqPriorBaseline(Baseline):
    name = "freq-prior"

    def __init__(self):
        self.by_registry = _empirical_labels_by_registry()
        # Global fallback distribution
        self.global_counts = Counter()
        for c in self.by_registry.values():
            self.global_counts.update(c)
        self.code_registry = code_to_registry()

    def _infer_registry(self, pdd_text: str) -> str | None:
        # Crude: scan the first 4 KB of text for registry markers.
        head = pdd_text[:4000]
        if "Gold Standard" in head or "GoldStandard" in head:
            return "GS"
        if "Verra" in head or "VCS" in head or "Verified Carbon Standard" in head:
            return "VCS"
        return None

    def predict(self, pdd_text: str, top_k: int = 5) -> list[tuple[str, float]]:
        reg = self._infer_registry(pdd_text)
        if reg and self.by_registry.get(reg):
            counter = self.by_registry[reg]
        else:
            counter = self.global_counts
        total = sum(counter.values()) or 1
        top = counter.most_common(top_k)
        return [(code, count / total) for code, count in top]
