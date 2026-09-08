"""Stage A1 dictionary readout, with no model-runtime dependencies."""

from .core import (
    FeatureTopK,
    ScoreComparison,
    compare_score_vectors,
    compose_decoder,
    feature_scores,
    iter_feature_topk,
    reference_feature_scores,
)

__all__ = [
    "FeatureTopK",
    "ScoreComparison",
    "compare_score_vectors",
    "compose_decoder",
    "feature_scores",
    "iter_feature_topk",
    "reference_feature_scores",
]
