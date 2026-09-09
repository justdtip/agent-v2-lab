"""CPU fp32 seam tests on real HF blocks; these are not checkpoint acceptance."""

import inspect

import pytest
import torch
from jlens.fitting import _check_layer_indices
from transformers import Gemma3ForCausalLM, Gemma3TextConfig


def make_model():
    torch.manual_seed(17)
    torch.set_num_threads(1)
    config = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        sliding_window=1024,
        max_position_embeddings=2048,
        layer_types=["sliding_attention", "full_attention", "sliding_attention"],
        attn_implementation="eager",
        pad_token_id=0,
    )
    model = Gemma3ForCausalLM(config).float().eval()
    model.requires_grad_(False)
    return model


@pytest.fixture
def view():
    from local_llm_lab.arch_torch import TorchArchitectureView

    return TorchArchitectureView.from_model(make_model())


def test_discovery_calls_upstream(monkeypatch):
    import jlens.hf

    from local_llm_lab.arch_torch import TorchArchitectureView

    calls = []
    original = jlens.hf._find_layout

    def discover(model):
        calls.append(model)
        return original(model)

    monkeypatch.setattr(jlens.hf, "_find_layout", discover)
    model = make_model()
    view = TorchArchitectureView.from_model(model)
    assert calls and all(value is model for value in calls)
    assert (view.num_layers, view.hidden_size, view.vocab_size) == (3, 16, 64)
    assert [view.attention_span(i) for i in range(3)] == ["sliding", "global", "sliding"]
    assert view.layer_kind(1) == "attention"
    assert view.tie_word_embeddings


def test_public_call_signatures_are_unchanged():
    from local_llm_lab.arch import ArchitectureView
    from local_llm_lab.arch_torch import TorchArchitectureView

    names = (
        "embed",
        "masks",
        "run_block",
        "final_norm",
        "unembed",
        "native_readout",
        "make_cache",
        "residuals",
        "native_residuals",
        "residual_source_agreement",
        "cached_logits",
        "tail",
        "layer_kind",
        "attention_span",
        "lora_targets",
    )
    for name in names:

        def signature(cls, name=name):
            return [
                (p.name, p.kind, p.default)
                for p in inspect.signature(getattr(cls, name)).parameters.values()
            ]

        assert signature(ArchitectureView) == signature(TorchArchitectureView), name


def test_residual_convention_is_upstreams_block_output(view):
    from jlens.hooks import ActivationRecorder

    sources, target = _check_layer_indices(None, None, view.num_layers)
    ids = torch.tensor([[1, 4, 9, 2]])
    with ActivationRecorder(view.blocks, at=[*sources, target]) as recorder:
        view.model(input_ids=ids, use_cache=False)
    residuals = view.native_residuals(ids, range(1, view.num_layers + 1))
    for index, value in recorder.activations.items():
        assert torch.equal(value, residuals[index + 1])
    assert torch.equal(
        view.native_readout(residuals[view.num_layers]),
        view.model(input_ids=ids, use_cache=False).logits,
    )


@pytest.mark.parametrize("length", [64, 1400])
def test_sources_agree_and_mask_negative_control(view, length, monkeypatch):
    ids = torch.arange(length) % view.vocab_size
    assert set(view.residual_source_agreement(ids, range(4)).values()) == {0.0}
    observe = view._observe_forward

    def broken(ids, cache=None):
        entry, bundles = observe(ids, cache)
        for index in bundles:
            bundles[index] = {**bundles[index], "attention_mask": bundles[1]["attention_mask"]}
        return entry, bundles

    monkeypatch.setattr(view, "_observe_forward", broken)
    errors = view.residual_source_agreement(ids, range(4))
    assert (max(errors.values()) > 0) == (length > 1024)


def test_cached_loop_matches_native_and_observation_does_not_advance_cache(view):
    native_cache = view.make_cache()
    loop_cache = view.make_cache()
    for ids in ([1, 3, 5, 7], [9], [11]):
        ids = torch.tensor([ids])
        native = view.model(
            input_ids=ids, past_key_values=view._unwrap_cache(native_cache), use_cache=True
        ).logits[:, -1, :]
        actual = view.cached_logits(ids, loop_cache)
        torch.testing.assert_close(actual, native, atol=2e-6, rtol=2e-6)
        before = [entry.offset for entry in loop_cache]
        view.masks(torch.zeros(1, 1, view.hidden_size), loop_cache)
        assert [entry.offset for entry in loop_cache] == before
    assert before == [6, 6, 6]
    with pytest.raises(ValueError, match="cache"):
        view._unwrap_cache([loop_cache[0], *native_cache[1:]])


def test_tail_has_same_endpoint_and_supports_gradient(view):
    ids = [1, 4, 8, 10]
    residuals = view.residuals(ids, range(4))
    for layer in range(4):
        h = residuals[layer].detach().requires_grad_()
        tail = view.tail(layer)(h)
        assert torch.equal(tail, view.final_norm(residuals[3]))
        assert torch.isfinite(torch.autograd.grad(tail.square().sum(), h)[0]).all()


def test_embed_observes_transform_outside_embedding(view):
    old = view.blocks[0].register_forward_pre_hook(
        lambda module, args: (args[0] + 7, *args[1:]), prepend=True
    )
    try:
        expected = view.text_module.embed_tokens(torch.tensor([[1, 2]])) + 7
        assert torch.equal(view.embed([1, 2]), expected)
    finally:
        old.remove()


def test_boolean_masks_keep_their_meaning(view):
    masks = view.masks(torch.zeros(1, 4, view.hidden_size), None)
    masks[0]["attention_mask"] = torch.ones(1, 1, 4, 4, dtype=torch.bool).tril()
    before = masks[0]["attention_mask"].clone()
    # SDPA accepts Boolean masks; the view must not reinterpret them as additive weights.
    view.model.config._attn_implementation = "sdpa"
    result = view.run_block(0, view.embed([1, 2, 3, 4]), masks, None)
    assert torch.isfinite(result).all()
    assert masks[0]["attention_mask"].dtype == torch.bool
    assert torch.equal(before, masks[0]["attention_mask"])


def test_dense_hidden_spans_fail_and_bad_indices_fail(view):
    with pytest.raises(ValueError, match="deferred|dense"):
        view.masks(torch.zeros(1, 1, view.hidden_size), None, hidden_spans=[])
    for value in (-1, True, view.num_layers):
        with pytest.raises(ValueError):
            view.layer_kind(value)
    with pytest.raises(ValueError):
        view.native_residuals([1, 2], [])
    with pytest.raises(ValueError):
        view.tail(view.num_layers + 1)


def test_lora_uses_existing_policy(view):
    targets = view.lora_targets("auto")
    assert targets == view.lora_targets("attention+mlp")
    assert set(targets) == {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    }


def test_constructor_rejects_conflicting_text_module():
    from local_llm_lab.arch_torch import TorchArchitectureView

    first, second = make_model(), make_model()
    with pytest.raises(ValueError, match="disagrees"):
        TorchArchitectureView(first, second.model)


def test_diagnostic_promotion_on_bfloat16_weights_preserves_model_dtype():
    from local_llm_lab.arch_torch import TorchArchitectureView

    model = make_model().to(torch.bfloat16)
    view = TorchArchitectureView.from_model(model)
    native = view.native_residuals([1, 3, 5], range(4))
    loop = view.residuals([1, 3, 5], range(4))
    assert all(h.dtype == torch.bfloat16 for h in native.values())
    assert all(h.dtype == torch.float32 and torch.isfinite(h).all() for h in loop.values())
    assert view.unembed(view.final_norm(loop[3])).dtype == torch.float32
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    assert torch.equal(
        view.native_readout(native[3]),
        model(input_ids=torch.tensor([[1, 3, 5]]), use_cache=False).logits,
    )


def test_cached_entry_observes_absolute_position_transform(view, monkeypatch):
    # An entry transform outside embed_tokens must see the same cache offset as native.
    old_forward = view.text_module.forward

    def positioned(input_ids=None, past_key_values=None, **kwargs):
        offset = 0 if past_key_values is None else past_key_values.get_seq_length()
        h = view._embed_tokens(input_ids) + float(offset)
        kwargs.pop("inputs_embeds", None)
        return old_forward(inputs_embeds=h, past_key_values=past_key_values, **kwargs)

    monkeypatch.setattr(view.text_module, "forward", positioned)
    native_cache, loop_cache = view.make_cache(), view.make_cache()
    for values in ([1, 2, 3], [4]):
        ids = torch.tensor([values])
        expected = view.model(
            input_ids=ids, past_key_values=view._unwrap_cache(native_cache), use_cache=True
        ).logits[:, -1, :]
        torch.testing.assert_close(
            view.cached_logits(ids, loop_cache), expected, atol=2e-6, rtol=2e-6
        )


def test_readout_probe_preserves_missing_and_zero_and_detects_controls(view):
    from research.acceptance.torch_seam import decode_readout_probe

    ids = [1, 5, 9, 3]
    baseline = decode_readout_probe(view, ids, steps=3)
    assert len(baseline["emitted_ids"]) == 3
    assert baseline["identity_failures"] == []
    tolerance = max(baseline["readout_errors"])
    assert tolerance is not None
    assert decode_readout_probe(view, ids, steps=3, readout=False)["readout_errors"] == [None] * 3
    skew = decode_readout_probe(view, ids, steps=3, control="skew_after_prefill")
    assert skew["readout_errors"][0] <= tolerance
    assert all(error > tolerance for error in skew["readout_errors"][1:])
    corrupt = decode_readout_probe(view, ids, steps=3, control="corrupt_residual")
    assert all(error > tolerance for error in corrupt["readout_errors"])


def test_observation_cleans_hooks_after_failure(view, monkeypatch):
    counts = [len(block._forward_pre_hooks) for block in view.blocks]

    def fail(*args, **kwargs):
        raise RuntimeError("native failure")

    monkeypatch.setattr(view.text_module, "forward", fail)
    with pytest.raises(RuntimeError, match="native failure"):
        view.embed([1, 3])
    assert [len(block._forward_pre_hooks) for block in view.blocks] == counts
    assert not view._observing


def test_observation_handles_nonleaf_cache_without_mutating_caller(view):
    view.model.requires_grad_(True)
    cache = view.make_cache()
    view.cached_logits([1, 3, 5], cache)
    native = view._unwrap_cache(cache)
    original_keys = [layer.keys for layer in native.layers]
    assert any(value.grad_fn is not None for value in original_keys)
    value = view.cached_logits([7], cache)
    value.square().sum().backward()
    assert [entry.offset for entry in cache] == [4, 4, 4]
    assert all(
        layer.keys is not previous
        for layer, previous in zip(native.layers, original_keys, strict=True)
    )
    assert view.blocks[0].self_attn.q_proj.weight.grad is not None


def test_cache_crop_preserves_upstream_remove_count_semantics(view):
    # A global-only DynamicCache supports crop; sliding retention cannot promise rollback.
    from transformers.cache_utils import DynamicCache

    from local_llm_lab.arch_torch import _CacheEntry

    cache = DynamicCache()
    keys = torch.zeros(1, 1, 3, 2)
    cache.update(keys, keys.clone(), 0)
    entry = _CacheEntry(cache, 0)
    entry.crop(tokens_to_remove=-3)
    assert entry.offset == 0
    assert not view.cache_trimmable
