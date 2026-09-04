"""Training-time backbone patches (SPEC-001 slices installed only while a stage trains).

Importing this package imports MLX and the pinned ``mlx_lm`` backbone modules, so callers
import it lazily, inside the stage that needs it, the way ``pipeline.cli`` imports the
trainer entry point.
"""

from __future__ import annotations

from local_llm_lab.training.gated_delta_chunked import (
    gated_delta_chunked_ops,
    install_chunked_gated_delta,
    training_state_bytes,
)
from local_llm_lab.training.gated_delta_chunkwise import (
    fallback_counts,
    gated_delta_chunkwise_ops,
    install_chunkwise_gated_delta,
)

__all__ = [
    "fallback_counts",
    "gated_delta_chunked_ops",
    "gated_delta_chunkwise_ops",
    "install_chunked_gated_delta",
    "install_chunkwise_gated_delta",
    "training_state_bytes",
]
