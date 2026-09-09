"""Batching the Jacobian backward is about 1.5x faster under both attention implementations.

Measured on CPU, torch 2.14.0, transformers 5.16.1, on the machine that wrote this file.
**Not verified on CUDA.** See CUDA-MIGRATION-RESEARCH-BRIEF-ANSWERS-2026-09-09 §1.

`torch.autograd.grad(..., is_grads_batched=True)` batches the backward over cotangents while
leaving `ActivationRecorder`'s `requires_grad_` semantics intact, so the existing hooks work
unchanged. It uses the vmap backend, and vmap falls back to a per-sample loop for any operation
without a batching rule. The scaled-dot-product attention backward has no batching rule
(pytorch#117016, open; fix PR#176265 unmerged as of 2026-09). Measured here, that fallback costs a
**one-time first-call penalty** under `sdpa` (22.9 ms against a 9.2 ms steady state) and nothing
per call thereafter. Steady-state the batched form is ~1.5x faster under both implementations.
A single-shot measurement reads that first call as a 0.62x regression, which is why this test
takes the best of several runs.

The assertion is on the timing ratio rather than on the warning, because the warning was observed
on the `vmap` path and not on the `is_grads_batched` path, so a warning-based assertion would not
fire for the form actually recommended. Both implementations are asserted, since both are faster
in steady state.
"""

from __future__ import annotations

import time

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig  # noqa: E402
from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel  # noqa: E402

D_MODEL, N_LAYERS, SEQ, N_COTANGENTS = 64, 6, 48, 16
SOURCE_LAYER, TARGET_LAYER = 1, 4
REPEATS = 3


def _model(attn_implementation: str) -> Gemma3TextModel:
    config = Gemma3TextConfig(
        num_hidden_layers=N_LAYERS,
        hidden_size=D_MODEL,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        intermediate_size=128,
        vocab_size=128,
        sliding_window=16,
        sliding_window_pattern=2,
        attn_implementation=attn_implementation,
    )
    model = Gemma3TextModel(config).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _best_of(fn, repeats: int = REPEATS) -> tuple[float, object]:
    best, result = float("inf"), None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return best, result


@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
def test_batched_backward_ratio(attn_implementation: str, capsys) -> None:
    torch.manual_seed(0)
    model = _model(attn_implementation)
    activations: dict[int, torch.Tensor] = {}

    def recorder(index: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if index == SOURCE_LAYER:
                # As `ActivationRecorder` does: root the graph at the earliest source layer.
                hidden = hidden.detach().requires_grad_(True)
            activations[index] = hidden
            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

        return hook

    handles = [
        model.layers[index].register_forward_hook(recorder(index))
        for index in (SOURCE_LAYER, TARGET_LAYER)
    ]
    try:
        input_ids = torch.randint(0, 128, (1, SEQ))
        with torch.enable_grad():
            model(input_ids=input_ids, use_cache=False)
            target, source = activations[TARGET_LAYER], activations[SOURCE_LAYER]

            cotangents = torch.zeros(N_COTANGENTS, 1, SEQ, D_MODEL)
            for row in range(N_COTANGENTS):
                cotangents[row, 0, :, row % D_MODEL] = 1.0

            batched_seconds, batched = _best_of(
                lambda: torch.autograd.grad(
                    target, source, cotangents, retain_graph=True, is_grads_batched=True
                )[0]
            )
            sequential_seconds, sequential = _best_of(
                lambda: torch.stack(
                    [
                        torch.autograd.grad(target, source, cotangents[row], retain_graph=True)[0]
                        for row in range(N_COTANGENTS)
                    ]
                )
            )
    finally:
        for handle in handles:
            handle.remove()

    # Correctness holds either way, and is the part that must never regress.
    assert torch.allclose(batched, sequential, atol=1e-5)

    ratio = sequential_seconds / batched_seconds
    with capsys.disabled():
        print(f"\n  {attn_implementation}: batched {batched_seconds * 1000:.1f} ms, "
              f"sequential {sequential_seconds * 1000:.1f} ms, ratio {ratio:.2f}x")

    assert ratio > 1.0, (
        f"batching the backward is no longer faster than the sequential loop under "
        f"{attn_implementation} (ratio {ratio:.2f}x); the Jacobian path's speedup has regressed"
    )
