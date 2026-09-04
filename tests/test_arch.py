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


# --------------------------------------------------------------------------- LoRA wrappers


def _wrap_dense_projections(model: object, *, rank: int = 2) -> int:
    """Replace every dense LoRA-target projection with a real mlx-lm ``LoRALinear`` wrapper.

    This is what ``mlx_lm.load(..., adapter_path=...)`` does to a policy: each projection
    becomes a plain ``nn.Module`` holding the original at ``.linear``. Returns the wrap count.
    """
    import mlx.nn as nn
    from mlx_lm.tuner.lora import LoRALinear

    suffixes = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    wrapped = 0
    for block in ArchitectureView.from_model(model).blocks:
        for path, module in list(block.named_modules()):
            if not path or not isinstance(module, nn.Linear):
                continue
            if path.rsplit(".", 1)[-1] not in suffixes:
                continue
            parent = block
            *parents, leaf = path.split(".")
            for name in parents:
                parent = getattr(parent, name)
            setattr(parent, leaf, LoRALinear.from_base(module, r=rank))
            wrapped += 1
    return wrapped


def test_lora_targets_see_through_adapter_wrappers_on_dense_and_hybrid_fakes() -> None:
    """Loading an adapter wraps projections; the view must report the same paths regardless.

    Regression for the live P6 failure: ``spec.resolve`` on an adapter-loaded policy raised
    ``LoRA policy 'attention+mlp' matched no linear modules`` because the wrapper is not an
    ``nn.Linear`` and its ``.linear`` child carries the wrong path suffix.
    """
    from mlx_lm.tuner.lora import LoRALinear

    from test_probes import make_dense_fake, make_hybrid_fake

    for make, policy in ((make_dense_fake, "attention+mlp"), (make_hybrid_fake, "auto")):
        plain, _ = make()
        expected = ArchitectureView.from_model(plain).lora_targets(policy)
        expected_count = ArchitectureView.from_model(plain).lora_parameter_count(expected, 16)

        wrapped_model, _ = make()
        assert _wrap_dense_projections(wrapped_model) > 0
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
    from local_llm_lab.models import load_model_spec

    from test_probes import make_dense_fake

    plain, _ = make_dense_fake()
    wrapped, _ = make_dense_fake()
    _wrap_dense_projections(wrapped)
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
