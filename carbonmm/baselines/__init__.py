"""Baselines 1-5 for ICDM 2026 methodology matcher.

Unified interface: each module exports a class implementing
    predict(pdd_text: str) -> list[tuple[code, score]]   # top-5
"""
