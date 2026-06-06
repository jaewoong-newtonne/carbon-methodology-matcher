"""Baseline 1: uniform random over the methodology label space (lower bound)."""
from __future__ import annotations

import random

from .base import Baseline, label_space


class RandomBaseline(Baseline):
    name = "random"

    def __init__(self, seed: int = 42):
        self.codes = label_space()
        self.rng = random.Random(seed)

    def predict(self, pdd_text: str, top_k: int = 5) -> list[tuple[str, float]]:
        # Sample WITHOUT replacement; ties broken by sample order.
        picks = self.rng.sample(self.codes, k=min(top_k, len(self.codes)))
        # Score = inverse rank for compatibility with score-based metrics.
        return [(c, 1.0 / (i + 1)) for i, c in enumerate(picks)]
