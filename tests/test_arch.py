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


def _mask_of(view, masks: dict, kind: str):
    """The mask the model hands the first block of ``kind``.

    `view.masks` is keyed by **block index** since the architecture port, because Gemma builds two
    masks and dispatches on the index, so two blocks of one kind can receive different masks and a
    kind-keyed mapping cannot say which. On this hybrid every block of a kind still gets the same
    mask, so the assertions below are unchanged in meaning; this is the one place that translates.
    """
    for index in range(view.num_layers):
        if view.layer_kind(index) == kind:
            return masks[index]
    raise AssertionError(f"this decoder has no {kind} block")


def _fp32_residual(view: ArchitectureView, ids: mx.array) -> mx.array:
    h = view.embed(ids)
    masks = view.masks(h, None)
    for index in range(view.num_layers):
        h = view.run_block(index, h, masks, None)
    return view.final_norm(h)


def _assert_uncached_hybrid_traversal(
    calls: list[tuple[object, ...]], dtype: mx.Dtype, *, mask_dtype: mx.Dtype | None = None
) -> None:
    """The block traversal, and the mask construction that now happens inside the model's forward.

    Since the architecture port, `view.masks` observes the model's own forward instead of calling
    the mask constructors itself, so the log opens with a `native` entry and the constructors see
    the **model's** embedding dtype rather than the caller's promoted residual. That is the point:
    they are the model's constructors and it builds its masks the way it builds them. The view
    casts an array mask to the residual's dtype before returning it, so a float32 loop still adds
    a float32 mask; these fakes return sentinels, so the cast is not observable here.
    """
    mask_calls = [entry for entry in calls if entry[0] in ("attention_mask", "ssm_mask")]
    expected_pair = [
        ("attention_mask", None, mask_dtype or dtype),
        ("ssm_mask", None, mask_dtype or dtype),
    ]
    # **Two pairs, not one, and that is a real cost of the port.** `embed` and `masks` are each
    # one observing forward, and a forward builds masks whether or not the caller wanted them, so
    # `embed`'s pair is built and discarded. Correct but not free: on a long sequence the
    # discarded attention mask is an N-squared allocation. The Chief's ruling was to pay the
    # forward and never cache it, and to hoist at the call site if measurement ever says
    # otherwise; this assertion is where that measurement would first show up as a change.
    assert mask_calls == expected_pair * (len(mask_calls) // 2)
    assert len(mask_calls) in (2, 4), "one pair per observing forward"
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
    _assert_uncached_hybrid_traversal(model.calls, mx.float32, mask_dtype=mx.bfloat16)
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


# ------------------------------------------- EXP-002 S1: the attention-only span mask
#
# R31/R38: the seam under test is mlx-lm's own mask contract, so these fixtures drive the
# library's real classes on every side that decides an outcome -- ``ArraysCache`` and
# ``KVCache`` for the caches, ``qwen3_5.DecoderLayer``/``GatedDeltaNet``/``Qwen3NextAttention``
# for the blocks, and the library's ``create_attention_mask`` for the mask itself. Only the
# weights and the dimensions are ours. A hand-written double cannot show any of what follows:
# that ``ArraysCache`` is untrimmable is inherited from ``_BaseCache`` and absent from the
# subclass, and that ``make_mask`` returns ``None`` at a single step is a fact about the
# library's function, not about a shape a fake could be given.

TINY_HYBRID_SEED = 96


@pytest.fixture
def cpu_stream():
    """Keep these slices off the GPU: they are toy weights and nothing here needs a device."""
    with mx.stream(mx.cpu):
        yield


def make_tiny_hybrid_model(num_hidden_layers: int = 4):
    """A real ``mlx_lm`` Qwen3.5 text model at toy dimensions with random weights.

    Same block layout as the 4B in miniature -- ``full_attention_interval`` puts recurrent
    ``GatedDeltaNet`` blocks everywhere except the last of each group of four -- so
    ``make_cache`` returns the production mix of ``ArraysCache(size=2)`` and ``KVCache`` that
    EXP-002 runs over. No checkpoint is read and nothing is loaded; the weights are random.

    ``num_hidden_layers`` defaults to one group of four, which has exactly one attention block
    and it is the last. That is enough for every mask-shape question S1 and S2 ask, and it is
    *not* enough for a question about information reaching the decision through attention: with
    no attention block before the last, nothing an earlier position read can propagate. S3 asks
    that question and passes eight.
    """
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

    mx.random.seed(TINY_HYBRID_SEED)
    args = TextModelArgs(
        model_type="qwen3_5_text",
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=24,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        tie_word_embeddings=True,
        full_attention_interval=4,
        max_position_embeddings=256,
    )
    model = TextModel(args)
    mx.eval(model.parameters())
    return model


def _tiny_ids(values: tuple[int, ...]) -> mx.array:
    return mx.array([list(values)], dtype=mx.int32)


def _run_cached(view: ArchitectureView, ids: mx.array, cache: list[object], **kwargs) -> mx.array:
    """One forward over a live cache through the view's own primitives, returning logits."""
    h = view.embed(ids)
    masks = view.masks(h, cache, **kwargs)
    for index in range(view.num_layers):
        h = view.run_block(index, h, masks, cache[index])
    return view.unembed(view.final_norm(h))


def test_tiny_hybrid_fixture_is_the_real_cache_mix(cpu_stream) -> None:
    """The fixture is only worth anything if it really is the library's classes."""
    from mlx_lm.models.cache import ArraysCache, KVCache

    view = ArchitectureView.from_model(make_tiny_hybrid_model())
    cache = view.make_cache()

    assert [type(entry) for entry in cache] == [ArraysCache, ArraysCache, ArraysCache, KVCache]
    assert [view.layer_kind(index) for index in range(view.num_layers)] == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "attention",
    ]
    # The premise EXP-002's regime rests on, taken from the real class rather than asserted:
    # ``ArraysCache`` defines neither name and inherits ``False``, so a persistent cache
    # cannot be rewound and each probe point needs its own prefill.
    assert "is_trimmable" not in vars(ArraysCache) and "trim" not in vars(ArraysCache)
    assert cache[0].is_trimmable() is False and cache[3].is_trimmable() is True
    assert view.cache_trimmable is False


def test_hidden_spans_match_the_library_route_on_the_full_sequence_path(cpu_stream) -> None:
    """S1 acceptance, part one: forcing the array must not move the logits by itself.

    With a cache present the default route returns the ``'causal'`` sentinel, which MLX's
    attention kernel handles itself. Arm A needs an explicit array instead, so every arm would
    be measured through a different attention path than EXP-001's baseline unless the two agree
    exactly on a prompt that hides nothing. ``hidden_spans=()`` is that condition: the array is
    built and nothing is ANDed to False.
    """
    view = ArchitectureView.from_model(make_tiny_hybrid_model())
    prefix = _tiny_ids((3, 1, 4, 6, 5, 2, 7, 8, 1, 9))
    scored = _tiny_ids((2, 5, 6))
    sentinel_cache = view.make_cache()
    array_cache = view.make_cache()
    _run_cached(view, prefix, sentinel_cache)
    _run_cached(view, prefix, array_cache)

    offset = sentinel_cache[3].offset
    h = view.embed(scored)
    default = view.masks(h, sentinel_cache)
    forced = view.masks(h, sentinel_cache, hidden_spans=())

    assert offset == prefix.shape[1]
    assert _mask_of(view, default, "attention") == "causal"
    assert isinstance(_mask_of(view, forced, "attention"), mx.array)
    assert _mask_of(view, forced, "attention").dtype == mx.bool_
    assert _mask_of(view, forced, "attention").shape == (
        scored.shape[1],
        offset + scored.shape[1],
    )

    sentinel_logits = _run_cached(view, scored, sentinel_cache)
    array_logits = _run_cached(view, scored, array_cache, hidden_spans=())
    assert float(mx.max(mx.abs(sentinel_logits - array_logits)).item()) == 0.0


def test_hidden_spans_match_the_library_route_at_a_cached_single_step(cpu_stream) -> None:
    """S1 acceptance, part two: the scored step is one token, where the library gives nothing.

    Measured, not read off ``base.py``: ``KVCache.make_mask(1, ...)`` returns ``None`` with
    ``return_array=True`` as well, so no flag forces an array out of the library here and the
    whole ``(1, offset + 1)`` row is the view's to build. This is the step at which an unmasked
    read would be invisible, so the built row must agree with the library's ``None`` exactly.
    """
    view = ArchitectureView.from_model(make_tiny_hybrid_model())
    prefix = _tiny_ids((3, 1, 4, 6, 5, 2, 7, 8, 1, 9))
    step = _tiny_ids((4,))
    sentinel_cache = view.make_cache()
    array_cache = view.make_cache()
    _run_cached(view, prefix, sentinel_cache)
    _run_cached(view, prefix, array_cache)

    offset = sentinel_cache[3].offset
    h = view.embed(step)
    assert sentinel_cache[3].make_mask(1, return_array=True, window_size=None) is None
    assert _mask_of(view, view.masks(h, sentinel_cache), "attention") is None

    forced = _mask_of(view, view.masks(h, sentinel_cache, hidden_spans=()), "attention")
    assert isinstance(forced, mx.array)
    assert forced.shape == (1, offset + 1)
    assert bool(mx.all(forced).item())

    sentinel_logits = _run_cached(view, step, sentinel_cache)
    array_logits = _run_cached(view, step, array_cache, hidden_spans=())
    assert float(mx.max(mx.abs(sentinel_logits - array_logits)).item()) == 0.0


def test_hidden_spans_hide_exactly_the_named_columns_and_keep_causality(cpu_stream) -> None:
    """The mask must cover the named span and nothing else, on both mask routes."""
    view = ArchitectureView.from_model(make_tiny_hybrid_model())
    prefix = _tiny_ids((3, 1, 4, 6, 5, 2, 7, 8, 1, 9))
    cache = view.make_cache()
    _run_cached(view, prefix, cache)
    offset = cache[3].offset

    for queries, spans in (((2, 5, 6), ((2, 5), (7, 8))), ((4,), ((2, 5),))):
        ids = _tiny_ids(queries)
        length = ids.shape[1]
        mask = _mask_of(
            view, view.masks(view.embed(ids), cache, hidden_spans=spans), "attention"
        )
        rows = mx.arange(offset, offset + length)[:, None]
        columns = mx.arange(offset + length)[None]
        hidden = mx.zeros(columns.shape, dtype=mx.bool_)
        for start, end in spans:
            hidden = hidden | ((columns >= start) & (columns < end))
        # Masking never deletes, so every later position keeps its index and RoPE is untouched;
        # what changes is only which columns a query may attend to.
        assert bool(mx.array_equal(mask, (rows >= columns) & ~hidden).item())


def test_hidden_spans_move_the_scored_distribution(cpu_stream) -> None:
    """A mask that changed nothing would make every equivalence check above vacuous."""
    view = ArchitectureView.from_model(make_tiny_hybrid_model())
    prefix = _tiny_ids((3, 1, 4, 6, 5, 2, 7, 8, 1, 9))
    step = _tiny_ids((4,))
    open_cache = view.make_cache()
    masked_cache = view.make_cache()
    _run_cached(view, prefix, open_cache)
    _run_cached(view, prefix, masked_cache)

    open_logits = _run_cached(view, step, open_cache, hidden_spans=())
    masked_logits = _run_cached(view, step, masked_cache, hidden_spans=((2, 6),))
    assert float(mx.max(mx.abs(open_logits - masked_logits)).item()) > 0.0


def test_hidden_spans_leave_the_recurrent_mask_untouched(cpu_stream) -> None:
    """The recurrent blocks take no span mask; that asymmetry is the whole experiment."""
    view = ArchitectureView.from_model(make_tiny_hybrid_model())
    cache = view.make_cache()
    ids = _tiny_ids((3, 1, 4, 6))
    h = view.embed(ids)
    # ``ArraysCache.make_mask`` answers ``None`` until a batch length is prepared; prepare one
    # so the comparison below is between two real arrays rather than between two ``None``s.
    for entry in cache[:3]:
        entry.prepare(lengths=[3])

    default = _mask_of(view, view.masks(h, cache), "linear_attention")
    with_spans = _mask_of(
        view, view.masks(h, cache, hidden_spans=((1, 3),)), "linear_attention"
    )
    assert isinstance(default, mx.array)
    assert bool(mx.array_equal(default, with_spans).item())


@pytest.mark.parametrize(
    "spans",
    [((1, 1),), ((3, 2),), ((-1, 2),), ((0, 99),), ((True, 2),), ((1, 2, 3),)],
)
def test_hidden_spans_reject_a_span_that_cannot_name_hidden_text(cpu_stream, spans) -> None:
    """An empty, inverted, out-of-range or malformed span hides nothing while running cleanly."""
    view = ArchitectureView.from_model(make_tiny_hybrid_model())
    cache = view.make_cache()
    h = view.embed(_tiny_ids((3, 1, 4, 6)))

    with pytest.raises(ValueError):
        view.masks(h, cache, hidden_spans=spans)


# ------------------------------------ EXP-002 S2: a cached forward that scores one position
#
# ``residuals`` runs with no cache and ``tail`` takes none, so nothing in the probe path could
# score a decision under a persistent cache. These tests read the same real tiny hybrid as S1's,
# so the cache the forward advances is the production mix of ``ArraysCache`` and ``KVCache``.


def test_cached_logits_match_the_models_own_forward_over_the_same_cache(cpu_stream) -> None:
    """The view's hand-run loop and the model's own loop must agree over one live cache.

    The model builds its masks from the first attention entry and the first recurrent entry;
    the view picks the same two structurally, so a disagreement here would mean the probe path
    scores a decision through different masks than inference uses.
    """
    model = make_tiny_hybrid_model()
    view = ArchitectureView.from_model(model)
    prefix = _tiny_ids((3, 1, 4, 6, 5, 2, 7, 8, 1, 9))
    step = _tiny_ids((4,))

    native_cache = model.make_cache()
    model(prefix, cache=native_cache)
    native = model(step, cache=native_cache)

    view_cache = view.make_cache()
    view.cached_logits(prefix, view_cache)
    scored = view.cached_logits(step, view_cache)

    assert scored.shape == (1, view.vocab_size)
    assert view_cache[3].offset == native_cache[3].offset
    assert float(mx.max(mx.abs(scored - native[:, -1, :])).item()) == 0.0


def test_cached_logits_score_the_chosen_position_only(cpu_stream) -> None:
    """A decision is scored at one position, so the readout is taken at that row alone.

    Measured, and the reason this row is not asserted bit-identical: the block loop and the
    final norm agree with the model exactly, and the whole difference comes from unembedding
    one row instead of the whole sequence, which is a differently shaped matmul. It is one
    float32 epsilon on a logit, and it exists because projecting every position to the
    vocabulary would dominate the cost of a probe point that reads one distribution.
    """
    model = make_tiny_hybrid_model()
    view = ArchitectureView.from_model(model)
    ids = _tiny_ids((3, 1, 4, 6, 5))
    native = model(ids)

    h = view.embed(ids)
    masks = view.masks(h, None)
    for index in range(view.num_layers):
        h = view.run_block(index, h, masks, None)
    assert float(mx.max(mx.abs(view.unembed(view.final_norm(h)) - native)).item()) == 0.0

    for position in (0, 2, 4, -1, -5):
        scored = view.cached_logits(ids, None, position=position)
        assert scored.shape == (1, view.vocab_size)
        difference = float(mx.max(mx.abs(scored - native[:, position, :])).item())
        assert difference <= float(mx.finfo(mx.float32).eps)


def test_cached_logits_advance_the_persistent_cache(cpu_stream) -> None:
    """Two calls over one cache must continue the sequence, not restart it."""
    model = make_tiny_hybrid_model()
    view = ArchitectureView.from_model(model)
    whole = _tiny_ids((3, 1, 4, 6, 5, 2))
    head, tail = _tiny_ids((3, 1, 4, 6)), _tiny_ids((5, 2))

    cache = view.make_cache()
    view.cached_logits(head, cache)
    assert cache[3].offset == head.shape[1]
    continued = view.cached_logits(tail, cache)
    assert cache[3].offset == whole.shape[1]

    assert float(mx.max(mx.abs(continued - view.cached_logits(whole, None))).item()) == 0.0


def test_cached_logits_carry_the_span_mask_to_the_attention_blocks(cpu_stream) -> None:
    """S1's mask has to reach the blocks through this forward, since S3 calls only this one.

    ``hidden_spans=()`` forces the explicit array and hides nothing, so it must leave the
    scored distribution exactly where the default route leaves it; a real span must move it.
    """
    view = ArchitectureView.from_model(make_tiny_hybrid_model())
    prefix = _tiny_ids((3, 1, 4, 6, 5, 2, 7, 8, 1, 9))
    step = _tiny_ids((4,))
    caches = [view.make_cache() for _ in range(3)]
    for cache in caches:
        view.cached_logits(prefix, cache)

    default = view.cached_logits(step, caches[0])
    forced = view.cached_logits(step, caches[1], hidden_spans=())
    masked = view.cached_logits(step, caches[2], hidden_spans=((2, 6),))

    assert float(mx.max(mx.abs(default - forced)).item()) == 0.0
    assert float(mx.max(mx.abs(default - masked)).item()) > 0.0


def test_cached_logits_reject_a_position_outside_the_scored_window(cpu_stream) -> None:
    """Scoring the wrong row would answer about a token the arm never asked about."""
    view = ArchitectureView.from_model(make_tiny_hybrid_model())
    ids = _tiny_ids((3, 1, 4, 6))

    for position in (4, -5, True, 1.0):
        with pytest.raises(ValueError):
            view.cached_logits(ids, None, position=position)


def test_cached_logits_reject_a_cache_that_does_not_match_the_blocks(cpu_stream) -> None:
    """A short cache would silently run later blocks uncached against an advancing prefix."""
    view = ArchitectureView.from_model(make_tiny_hybrid_model())
    ids = _tiny_ids((3, 1, 4, 6))

    with pytest.raises(ValueError):
        view.cached_logits(ids, view.make_cache()[:-1])


def test_cache_offset_refuses_to_default_a_missing_offset() -> None:
    """A silent zero would put the span's columns at the wrong absolute positions.

    Every attention cache mlx-lm builds carries ``offset`` -- ``KVCache`` above is checked
    against the real class -- so this covers only the guard, whose input is deliberately not
    a cache: the point is that an object that cannot answer is rejected rather than assumed.
    """
    from local_llm_lab.arch import _cache_offset

    assert _cache_offset(None) == 0
    for entry in (object(), type("Entry", (), {"offset": -1})(), type("E", (), {"offset": 1.0})()):
        with pytest.raises(ValueError):
            _cache_offset(entry)


# ------------------------------- EXP-002 S3: the artifact records the route that actually ran
#
# Two mask forms exist by design and either can be correct at the scored step: the library's
# own boolean array with the hidden columns ANDed to False, and the row built here when the
# library answers ``None`` at a single token. They are different code paths with different
# failure modes, so a reader of an arm-A result has to be told which one ran rather than left
# to re-derive it. The same principle governs the spans: the artifact records the resolved
# absolute column ranges the mask indexed, not the message boundaries they came from, because
# those come apart exactly when something is wrong.


def test_masks_record_the_route_and_the_resolved_spans_that_ran(cpu_stream) -> None:
    """Both routes must name themselves, with the columns the mask actually used."""
    view = ArchitectureView.from_model(make_tiny_hybrid_model())
    prefix = _tiny_ids((3, 1, 4, 6, 5, 2, 7, 8, 1, 9))
    cache = view.make_cache()
    _run_cached(view, prefix, cache)
    offset = cache[3].offset

    multi: dict[str, object] = {}
    view.masks(view.embed(_tiny_ids((2, 5, 6))), cache, hidden_spans=((2, 5),), record=multi)
    assert multi["attention_mask_route"] == "library array, hidden columns ANDed"
    assert multi["hidden_spans"] == ((2, 5),)
    assert multi["query_tokens"] == 3
    assert multi["cache_offset"] == offset
    assert multi["attention_mask_shape"] == (3, offset + 3)

    single: dict[str, object] = {}
    view.masks(view.embed(_tiny_ids((4,))), cache, hidden_spans=((2, 5), (7, 8)), record=single)
    assert single["attention_mask_route"] == "row constructed at a single step"
    assert single["hidden_spans"] == ((2, 5), (7, 8))
    assert single["attention_mask_shape"] == (1, offset + 1)


def test_masks_record_the_default_route_when_no_span_is_hidden(cpu_stream) -> None:
    """A record that only ever fills in on the span route could not report arm B's forward."""
    view = ArchitectureView.from_model(make_tiny_hybrid_model())
    cache = view.make_cache()
    record: dict[str, object] = {}

    view.masks(view.embed(_tiny_ids((3, 1, 4, 6))), cache, record=record)
    assert record["attention_mask_route"] == "model helper, unmodified"
    assert record["hidden_spans"] == ()


def test_cached_logits_carry_the_record_out_of_the_scored_forward(cpu_stream) -> None:
    """S3 scores through ``cached_logits`` alone, so the record has to survive that call."""
    view = ArchitectureView.from_model(make_tiny_hybrid_model())
    cache = view.make_cache()
    view.cached_logits(_tiny_ids((3, 1, 4, 6, 5, 2, 7, 8, 1, 9)), cache)
    record: dict[str, object] = {}

    view.cached_logits(_tiny_ids((4,)), cache, hidden_spans=((2, 6),), record=record)
    assert record["attention_mask_route"] == "row constructed at a single step"
    assert record["hidden_spans"] == ((2, 6),)
