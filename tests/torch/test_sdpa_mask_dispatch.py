"""Gemma 3 runs two different attention kernels in one forward under `sdpa`.

Measured on CPU, torch 2.14.0, transformers 5.16.1, on the machine that wrote this file.
**Not verified on CUDA.** This is the measured basis for the `eager` ruling
(CUDA-MIGRATION-RESEARCH-BRIEF-ANSWERS-2026-09-09 §2).

`sdpa_attention.py` gates the fast causal path on `q_length > 1 and attention_mask is None`.
Gemma 3's sliding layers always receive an explicit 4D mask, so they cannot take it and dispatch
to a different backend from the full-attention layers, which receive `None`. Two kernels in one
forward is a hazard for both determinism and for any eager-versus-sdpa numerical comparison.
`eager` removes the split.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig  # noqa: E402
from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel  # noqa: E402


def test_sliding_layers_receive_a_mask_and_full_layers_do_not() -> None:
    torch.manual_seed(0)
    config = Gemma3TextConfig(
        num_hidden_layers=4,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        intermediate_size=64,
        vocab_size=128,
        sliding_window=8,
        sliding_window_pattern=2,
        attn_implementation="sdpa",
    )
    model = Gemma3TextModel(config).eval()
    assert config.layer_types == [
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
    ]

    received: dict[int, object] = {}

    def recorder(index: int):
        def hook(_module, args, kwargs, _output):
            mask = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
            received[index] = None if mask is None else tuple(mask.shape)

        return hook

    handles = [
        model.layers[index].self_attn.register_forward_hook(recorder(index), with_kwargs=True)
        for index in range(4)
    ]
    try:
        with torch.no_grad():
            model(input_ids=torch.randint(0, 128, (1, 12)), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    for index, layer_type in enumerate(config.layer_types):
        if layer_type == "sliding_attention":
            assert received[index] is not None, (
                f"layer {index} is sliding but received no mask; the two-kernel dispatch this "
                "test records may have changed, and with it the basis for the eager ruling"
            )
            assert received[index] == (1, 1, 12, 12)
        else:
            assert received[index] is None, (
                f"layer {index} is full attention but received a mask {received[index]}; it can "
                "no longer take the fast causal path"
            )
