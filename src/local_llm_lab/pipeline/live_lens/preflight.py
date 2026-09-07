"""Pure launch checks; safe to import before the trainer or Metal runtime."""

from __future__ import annotations

import math


def training_preflight(train, spec, *, iters=None):
    """Check only models with an evidence-backed launch envelope in their registry.

    This checks recipe declarations; it does not prove that data fit the tokenizer's prompt
    cap or grant a model-use window. The existing rendered-split loader still checks each row.
    """
    envelope = spec.train.get("launch_preflight")
    if envelope is None:
        return None
    if train.get("gated_delta_mode") != "chunkwise":
        raise ValueError("train.gated_delta_mode must explicitly name chunkwise")

    def integer(name, value):
        if type(value) is not int or value <= 0:
            raise ValueError(f"train.{name} must be a positive integer")
        return value

    chunk = integer("gated_delta_chunk", train.get("gated_delta_chunk"))
    cap = integer("max_seq_length", train.get("max_seq_length"))
    batch = integer("batch_size", train.get("batch_size"))
    if (
        chunk != envelope["chunk_size"]
        or not chunk <= cap <= envelope["max_row_tokens"]
        or batch > envelope["max_batch_size"]
    ):
        raise ValueError(
            "chunk, row cap or batch size is outside the verified envelope; "
            "a new measured envelope is required before this launch"
        )
    if train.get("iters_unit") != "batches":
        raise ValueError(
            "train.iters_unit must explicitly be batches; iters is not optimizer steps"
        )
    batches = integer("iters", train.get("iters") if iters is None else iters)
    accumulation = integer("grad_accumulation_steps", train.get("grad_accumulation_steps"))
    # mlx-lm 0.31.3 calls optimizer.update only when it % accumulation == 0.
    if batches % accumulation:
        raise ValueError("iteration count leaves unapplied accumulated gradients")
    gib = train.get("metal_cache_gib", 2.0)
    if (
        isinstance(gib, bool)
        or not isinstance(gib, (float, int))
        or not math.isfinite(gib)
        or not 0 < gib <= envelope["max_metal_cache_gib"]
    ):
        raise ValueError(
            "train.metal_cache_gib must be finite, positive and inside the cache envelope"
        )
    return {
        "batches": batches,
        "rows": batches * batch,
        "optimizer_updates": batches // accumulation,
        "iterations_unit": "batches",
        "chunk_size": chunk,
        "max_row_tokens": cap,
        "metal_cache_bytes": int(gib * 2**30),
        "envelope": dict(envelope),
    }
