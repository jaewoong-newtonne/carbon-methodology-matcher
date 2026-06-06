"""GraphRAG fusion-only — no LLM rerank, just fusion order.

Sanity check + cheap competitor. Uses the same retrieve + features + hard-exclude
+ fusion as v2, then returns the fusion top-5 directly (no LLM rerank step).

If this beats LLM-reranked graphrag, that's strong evidence that the LLM rerank
is HURTING rather than helping — paper §6 narrative.
"""
from __future__ import annotations

import logging

from ..baselines.base import Baseline, label_space
from .extract_features import extract_features, PDDFeatures
from .graph_filter import filter_candidates
from .score_fusion import FusionConfig, fuse

logger = logging.getLogger(__name__)


class GraphRAGFusionOnlyBaseline(Baseline):
    name = "graphrag-fusion-only"

    def __init__(
        self,
        retriever=None,
        fusion: FusionConfig | None = None,
        top_k_after_fusion: int = 10,
    ):
        if retriever is None:
            from .retrieve import HybridRetriever
            retriever = HybridRetriever.load_default()
        self.R = retriever
        self.fusion_cfg = fusion or FusionConfig(alpha=0.5, beta=0.5, gamma=0.0)
        self.top_k_after_fusion = top_k_after_fusion
        self.valid_codes = set(label_space())

    def predict(self, pdd_text: str, top_k: int = 5) -> list[tuple[str, float]]:
        try:
            raw = self.R.query_with_raw(pdd_text)
        except Exception as e:
            logger.warning("retrieve failed: %s", e)
            return []

        try:
            feats = extract_features(pdd_text)
        except Exception as e:
            logger.warning("extract failed: %s", e)
            feats = PDDFeatures()

        candidate_codes = list(
            {raw["idx_to_code"][i] for i, _ in raw["bm25_top"]} |
            {raw["idx_to_code"][i] for i, _ in raw["dense_top"]}
        )
        filter_results = filter_candidates(candidate_codes, feats)

        fused = fuse(
            raw["bm25_top"], raw["dense_top"], raw["idx_to_code"],
            filter_results, config=self.fusion_cfg, top_k=self.top_k_after_fusion,
        )
        return [(c, s) for c, s, _ in fused[:top_k]]
