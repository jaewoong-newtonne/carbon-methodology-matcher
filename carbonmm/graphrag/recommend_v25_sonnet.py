"""v2.5 with claude-sonnet-4-6 for the rerank LLM. Quick Sonnet escalation.

Inherits v2.5's retry logic on 500/502/503.
"""
from __future__ import annotations

from .recommend_v25 import GraphRAGv25Baseline


class GraphRAGv25SonnetBaseline(GraphRAGv25Baseline):
    name = "graphrag-v25-sonnet"

    def __init__(self, **kw):
        kw.setdefault("model", "claude-sonnet-4-6")
        super().__init__(**kw)
