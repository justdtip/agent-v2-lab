from __future__ import annotations

import mlx.core as mx

from local_llm_lab.arch import ArchitectureView


class _HybridEmbedding:
    num_embeddings = 128
    dims = 3

    def __call__(self, ids: mx.array) -> mx.array:
        values = ids.astype(mx.float32)[..., None] * mx.array([0.1001, 0.2003, 0.3007])
        return values.astype(mx.bfloat16)


class _HybridBlock:
    def __init__(self, index: int, *, is_linear: bool, calls: list[tuple[object, ...]]) -> None:
        self.index = index
        self.is_linear = is_linear
        self.calls = calls

    def __call__(self, h: mx.array, *, mask: object, cache: object) -> mx.array:
        output = h * mx.array(1.001 + self.index * 0.0007).astype(h.dtype)
        output = output + mx.array(0.0003 * (self.index + 1)).astype(h.dtype)
        self.calls.append(("block", self.index, self.is_linear, mask, cache, h.dtype, output.dtype))
        return output


class _HybridTextModule:
    def __init__(self, calls: list[tuple[object, ...]]) -> None:
        self.calls = calls
        self.embed_tokens = _HybridEmbedding()
        self.layers = [
            _HybridBlock(0, is_linear=True, calls=calls),
            _HybridBlock(1, is_linear=True, calls=calls),
            _HybridBlock(2, is_linear=True, calls=calls),
            _HybridBlock(3, is_linear=False, calls=calls),
        ]

    def create_attention_mask(self, h: mx.array, cache: object) -> str:
        self.calls.append(("attention_mask", cache, h.dtype))
        return "attention-sentinel"

    def create_ssm_mask(self, h: mx.array, cache: object) -> str:
        self.calls.append(("ssm_mask", cache, h.dtype))
        return "ssm-sentinel"

    def norm(self, h: mx.array) -> mx.array:
        output = h * mx.array(1.0009).astype(h.dtype) + mx.array(0.0001).astype(h.dtype)
        self.calls.append(("norm", 4, h.dtype, output.dtype))
        return output

    def __call__(
        self,
        ids: mx.array,
        cache: list[object] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        embedding_dtype = None if input_embeddings is None else input_embeddings.dtype
        self.calls.append(("native", cache, embedding_dtype))
        h = self.embed_tokens(ids) if input_embeddings is None else input_embeddings
        if cache is None:
            cache = [None] * len(self.layers)
        attention_mask = self.create_attention_mask(h, cache[3])
        ssm_mask = self.create_ssm_mask(h, cache[0])
        for layer, cache_i in zip(self.layers, cache, strict=True):
            h = layer(h, mask=ssm_mask if layer.is_linear else attention_mask, cache=cache_i)
        return self.norm(h)


class _Head:
    weight = mx.zeros((128, 3))


class _HybridModel:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.model = _HybridTextModule(self.calls)
        self.lm_head = _Head()


def _fp32_residual(view: ArchitectureView, ids: mx.array) -> mx.array:
    h = view.embed(ids)
    masks = view.masks(h, None)
    for index in range(view.num_layers):
        h = view.run_block(index, h, masks, None)
    return view.final_norm(h)


def _assert_uncached_hybrid_traversal(calls: list[tuple[object, ...]], dtype: mx.Dtype) -> None:
    assert calls[:2] == [
        ("attention_mask", None, dtype),
        ("ssm_mask", None, dtype),
    ]
    assert [entry[1:5] for entry in calls if entry[0] == "block"] == [
        (0, True, "ssm-sentinel", None),
        (1, True, "ssm-sentinel", None),
        (2, True, "ssm-sentinel", None),
        (3, False, "attention-sentinel", None),
    ]
    assert [(entry[5], entry[6]) for entry in calls if entry[0] == "block"] == [
        (dtype, dtype),
        (dtype, dtype),
        (dtype, dtype),
        (dtype, dtype),
    ]
    assert calls[-1] == ("norm", 4, dtype, dtype)


def test_hybrid_native_diagnostic_preserves_bfloat16_and_matches_reference() -> None:
    """The diagnostic must mirror the native BF16 loop without using the FP32 view helpers."""
    model = _HybridModel()
    view = ArchitectureView.from_model(model)
    ids = mx.arange(64, dtype=mx.int32)[None, :]

    fp32_residual = _fp32_residual(view, ids)
    _assert_uncached_hybrid_traversal(model.calls, mx.float32)
    model.calls.clear()

    native_reference = model.model(ids)
    assert model.calls[0] == ("native", None, None)
    _assert_uncached_hybrid_traversal(model.calls[1:], mx.bfloat16)
    model.calls.clear()

    native_manual = view.diagnostic_native_final_residual(ids)
    _assert_uncached_hybrid_traversal(model.calls, mx.bfloat16)

    assert view.embed(ids).dtype == mx.float32
    assert fp32_residual.dtype == mx.float32
    assert native_reference.dtype == mx.bfloat16
    assert native_manual.dtype == mx.bfloat16
    assert float(mx.max(mx.abs(fp32_residual - native_reference)).item()) > 0.0
    assert float(mx.max(mx.abs(native_manual - native_reference)).item()) == 0.0
