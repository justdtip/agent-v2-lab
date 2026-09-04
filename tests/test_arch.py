from __future__ import annotations

import mlx.core as mx
import pytest

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


# --------------------------------------------------------------------------- LoRA wrappers


def _module_at(block: object, path: str) -> object:
    """The module a block holds at one of its own ``named_modules`` paths."""
    module = block
    for name in path.split("."):
        module = getattr(module, name)
    return module


def test_lora_targets_see_through_adapter_wrappers_on_dense_and_hybrid_fakes() -> None:
    """Loading an adapter wraps projections; the view must report the same paths regardless.

    Regression for the live P6 failure: ``spec.resolve`` on an adapter-loaded policy raised
    ``LoRA policy 'attention+mlp' matched no linear modules`` because the wrapper is not an
    ``nn.Linear`` and its ``.linear`` child carries the wrong path suffix.
    """
    from mlx_lm.tuner.lora import LoRALinear
    from test_probes import make_dense_fake, make_fake, make_hybrid_fake

    for make, policy in ((make_dense_fake, "attention+mlp"), (make_hybrid_fake, "auto")):
        plain, _ = make()
        expected = ArchitectureView.from_model(plain).lora_targets(policy)
        expected_count = ArchitectureView.from_model(plain).lora_parameter_count(expected, 16)

        wrapped_model, _ = make_fake(make, "wrapped")
        view = ArchitectureView.from_model(wrapped_model)
        assert any(
            isinstance(module, LoRALinear)
            for block in view.blocks
            for _, module in block.named_modules()
        ), "the fake must actually carry LoRA wrappers for this test to mean anything"

        targets = view.lora_targets(policy)
        assert targets == expected
        assert not any(path.endswith(".linear") for path in targets)
        assert view.lora_parameter_count(targets, 16) == expected_count


def test_model_spec_resolve_succeeds_on_an_adapter_wrapped_policy() -> None:
    """The exact call chain that crashed: load_policy -> spec.resolve -> lora_targets."""
    from test_probes import make_dense_fake, make_fake

    from local_llm_lab.models import load_model_spec

    plain, _ = make_dense_fake()
    wrapped, _ = make_fake(make_dense_fake, "wrapped")
    spec = load_model_spec("qwen25-coder-3b")
    tokenizer = type("Tokenizer", (), {"snapshot_revision": "fake-dense-revision"})()

    expected = spec.resolve(plain, tokenizer)
    resolved = spec.resolve(wrapped, tokenizer)

    assert resolved.lora_keys == expected.lora_keys
    assert resolved.trainable_parameters == expected.trainable_parameters


def test_lora_wrapper_over_a_quantized_base_reports_dequantized_input_width() -> None:
    """The real 3B is 4-bit: the wrapper's base is a ``QuantizedLinear`` and dims come from it."""
    import mlx.nn as nn
    from mlx_lm.tuner.lora import LoRALinear

    from local_llm_lab.arch import _linear_dimensions

    base = nn.QuantizedLinear(64, 8, bias=False, group_size=64, bits=4)
    wrapper = LoRALinear.from_base(base, r=2)
    assert _linear_dimensions(base) == (8, 64)

    class _Attn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = wrapper

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = _Attn()

    block = _Block()
    found = dict((path, module) for path, module in block.named_modules() if path)
    # The shape mlx-lm produces: wrapper at the projection path, base one level down.
    assert list(found) == [
        "self_attn",
        "self_attn.q_proj",
        "self_attn.q_proj.dropout",
        "self_attn.q_proj.linear",
    ]
    assert isinstance(found["self_attn.q_proj"], LoRALinear)
    assert isinstance(found["self_attn.q_proj.linear"], nn.QuantizedLinear)
    # The wrapper owns no ``weight``: dimension readers must be handed the base, which is what
    # the view's discovery now does. Handing them the wrapper is the pre-fix failure mode.
    try:
        _linear_dimensions(wrapper)
    except ValueError:
        pass
    else:
        raise AssertionError("the LoRA wrapper must not satisfy the linear-dimension reader")


@pytest.mark.parametrize(
    "factory_name",
    ["make_dense_fake", "make_hybrid_fake", "make_split_hybrid_fake", "make_quantized_dense_fake"],
)
def test_linear_modules_report_the_base_under_every_wrapper_path(factory_name: str) -> None:
    """Wrapped and bare enumerate the same (block, path) pairs, and the module is the base.

    This is the assertion the standard fixtures now carry for every arch test: the wrapped
    variant must see through the wrappers exactly as the bare variant sees the bare modules.
    Where identity matters, the wrapper sits at the reported path and holds the reported module
    at ``.linear`` -- so path-keyed consumers (adapter comparison, provenance) are unaffected and
    dimension readers get the module that owns ``weight``/``bits``.
    """
    import test_probes
    from mlx_lm.tuner.lora import LoRALinear

    from local_llm_lab.arch import _linear_dimensions

    factory = getattr(test_probes, factory_name)
    bare, _ = test_probes.make_fake(factory, "bare")
    wrapped, _ = test_probes.make_fake(factory, "wrapped")
    bare_view = ArchitectureView.from_model(bare)
    wrapped_view = ArchitectureView.from_model(wrapped)

    bare_entries = bare_view._linear_modules()
    wrapped_entries = wrapped_view._linear_modules()
    assert bare_entries, "the fake must expose linear projections for this test to mean anything"
    assert [(index, path) for index, path, _ in wrapped_entries] == [
        (index, path) for index, path, _ in bare_entries
    ]
    assert not any(path.endswith(".linear") for _, path, _ in wrapped_entries)

    targets = set(wrapped_view.lora_targets("auto"))
    seen_wrappers = 0
    for (index, path, module), (_, _, bare_module) in zip(
        wrapped_entries, bare_entries, strict=True
    ):
        held = _module_at(wrapped_view.blocks[index], path)
        if path in targets:
            assert isinstance(held, LoRALinear)
            assert held.linear is module
            seen_wrappers += 1
        else:
            assert held is module
        assert _linear_dimensions(module) == _linear_dimensions(bare_module)
    assert seen_wrappers == len(wrapped_entries)

    assert wrapped_view.lora_targets("auto") == bare_view.lora_targets("auto")
    assert wrapped_view.lora_parameter_count(
        wrapped_view.lora_targets("auto"), 16
    ) == bare_view.lora_parameter_count(bare_view.lora_targets("auto"), 16)


def test_adapter_wrapped_quantized_fixture_keeps_every_structural_answer() -> None:
    """The 4-bit case as a standard fixture: wrappers over ``QuantizedLinear``, same answers.

    ``89dfb56``'s quantized test pins the shape mlx-lm produces for one hand-built module; this
    drives the same case through a whole fake, so the quantized base is exercised by the view's
    block walk, its dimension reader and its residual capture rather than in isolation.
    """
    import mlx.nn as nn
    from mlx_lm.tuner.lora import LoRALinear
    from test_probes import make_fake, make_quantized_dense_fake

    from local_llm_lab.arch import _linear_dimensions

    bare, ids = make_fake(make_quantized_dense_fake, "bare")
    wrapped, _ = make_fake(make_quantized_dense_fake, "wrapped")
    bare_view = ArchitectureView.from_model(bare)
    view = ArchitectureView.from_model(wrapped)

    entries = view._linear_modules()
    assert entries and all(isinstance(module, nn.QuantizedLinear) for _, _, module in entries)
    assert all(
        isinstance(_module_at(view.blocks[index], path), LoRALinear) for index, path, _ in entries
    )
    # 4-bit packing: the base's stored width is a quarter of the true input width, and the
    # dequantized width is what the view reports -- read off the base, never the wrapper.
    q_proj = next(module for _, path, module in entries if path == "self_attn.q_proj")
    assert int(q_proj.weight.shape[1]) == 8
    assert _linear_dimensions(q_proj) == (64, 64)

    layers = (0, view.num_layers)
    residuals = view.residuals(ids, layers)
    expected = bare_view.residuals(ids, layers)
    assert view.num_layers == bare_view.num_layers
    for layer in layers:
        assert residuals[layer].shape == expected[layer].shape
        assert float(mx.max(mx.abs(residuals[layer] - expected[layer])).item()) == 0.0
