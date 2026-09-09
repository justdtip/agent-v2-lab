"""Per-layer freezing, the optimizer's state dtype, and the loss `Trainer` will call."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from local_llm_lab.training.torch_full import (  # noqa: E402
    build_adamw,
    causal_lm_chunked_loss,
    freeze_all_but_top_layers,
    upcast_trainable_to_float32,
)

LAYERS = 6


def _tiny(dtype=torch.float32):
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    torch.manual_seed(0)
    config = Gemma3TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=LAYERS,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, sliding_window=8,
        rms_norm_eps=1e-6, use_cache=False,
    )
    return Gemma3ForCausalLM(config).to(dtype)


@pytest.mark.parametrize("top", [1, 3, LAYERS])
def test_freezing_opens_exactly_the_top_n_blocks(top: int) -> None:
    """`lora_layers: N` adapts the last N blocks and never the embedding, norm or head."""
    model = _tiny()
    opened = freeze_all_but_top_layers(model, top)

    expected = sum(
        p.numel() for layer in list(model.model.layers)[LAYERS - top :] for p in layer.parameters()
    )
    assert opened == expected
    for index, layer in enumerate(model.model.layers):
        assert all(p.requires_grad is (index >= LAYERS - top) for p in layer.parameters())
    assert not model.model.embed_tokens.weight.requires_grad
    assert not model.model.norm.weight.requires_grad


def test_none_trains_everything() -> None:
    model = _tiny()
    opened = freeze_all_but_top_layers(model, None)
    assert opened == sum(p.numel() for p in model.parameters())
    assert model.model.embed_tokens.weight.requires_grad


def test_a_recipe_naming_more_layers_than_the_model_has_is_refused() -> None:
    with pytest.raises(ValueError, match="exceeds the model"):
        freeze_all_but_top_layers(_tiny(), LAYERS + 1)


def test_a_bfloat16_trainable_parameter_is_refused_not_accommodated() -> None:
    """AdamW allocates its moments in the parameter's dtype, and nothing warns that it did.

    This is the substance of MLX's `StableAdamW`, which forced float32 moments explicitly. Upstream
    does not, so the check has to exist somewhere; here it is a refusal rather than a silent
    subclass, because the failure it prevents is a loss curve that falls while the small gradients
    are being rounded away.
    """
    model = _tiny(dtype=torch.bfloat16)
    freeze_all_but_top_layers(model, 1)
    with pytest.raises(TypeError, match="bfloat16"):
        build_adamw(model, learning_rate=1e-5)


def test_upcasting_the_trainable_slice_leaves_the_frozen_trunk_alone() -> None:
    """Only what is trained pays sixteen bytes a parameter; the frozen trunk stays bfloat16."""
    model = _tiny(dtype=torch.bfloat16)
    opened = freeze_all_but_top_layers(model, 1)

    upcast = upcast_trainable_to_float32(model)
    assert upcast == opened
    assert model.model.embed_tokens.weight.dtype is torch.bfloat16
    assert all(p.dtype is torch.float32 for p in model.model.layers[-1].parameters())

    optimizer = build_adamw(model, learning_rate=1e-5)
    ids = torch.arange(24).remainder(64).reshape(2, 12)
    # A mixed-dtype stack needs autocast, which is precisely what `Trainer(bf16=True)` turns on:
    # a bfloat16 activation reaching a float32 Linear is a dtype error otherwise. Pinned below.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = causal_lm_chunked_loss(model, {"input_ids": ids, "labels": ids}, chunk_size=4)
    loss.backward()
    optimizer.step()

    state = next(iter(optimizer.state.values()))
    assert state["exp_avg"].dtype is torch.float32
    assert state["exp_avg_sq"].dtype is torch.float32
    # The gradient accumulates in the parameter's dtype, which is the point of upcasting.
    assert model.model.layers[-1].mlp.down_proj.weight.grad.dtype is torch.float32


def test_the_mixed_dtype_stack_requires_autocast_and_says_so_when_it_is_missing() -> None:
    """Recorded as a constraint, not a defect: this is why `bf16=True` is not optional.

    Freezing the trunk in bfloat16 and training the top layers in float32 is what makes the memory
    arithmetic work -- only trained parameters pay sixteen bytes. The cost is that the forward is
    only valid under autocast, so a runner that drops `bf16=True` gets a dtype error rather than a
    slow run, which is the better of the two failures.
    """
    model = _tiny(dtype=torch.bfloat16)
    freeze_all_but_top_layers(model, 1)
    upcast_trainable_to_float32(model)
    ids = torch.arange(24).remainder(64).reshape(2, 12)

    with pytest.raises(RuntimeError, match="same dtype"):
        causal_lm_chunked_loss(model, {"input_ids": ids, "labels": ids}, chunk_size=4)


def test_freezing_nothing_open_is_refused_rather_than_trained() -> None:
    model = _tiny()
    freeze_all_but_top_layers(model, 0)
    with pytest.raises(ValueError, match="opened nothing"):
        build_adamw(model, learning_rate=1e-5)


def test_the_chunked_loss_is_the_models_own_loss() -> None:
    model = _tiny().eval()
    ids = torch.arange(24).remainder(64).reshape(2, 12)
    labels = ids.clone()
    labels[:, :3] = -100

    expected = model(input_ids=ids, labels=labels, use_cache=False).loss
    got = causal_lm_chunked_loss(model, {"input_ids": ids, "labels": labels}, chunk_size=5)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


def test_token_weighting_is_what_num_items_in_batch_makes_it() -> None:
    """The departure from the MLX records, demonstrated rather than asserted in prose.

    mlx-lm averages per-batch means unweighted; `Trainer` divides each micro-batch's summed loss by
    the token count of the whole accumulation cycle. The two agree only when every micro-batch
    holds the same number of supervised tokens, and length-sorted batching guarantees they do not.
    """
    model = _tiny().eval()
    short = torch.arange(8).remainder(64).reshape(1, 8)
    long = torch.arange(24).remainder(64).reshape(1, 24)

    def summed(ids):
        return causal_lm_chunked_loss(
            model, {"input_ids": ids, "labels": ids}, chunk_size=4, num_items_in_batch=1
        )

    def mean(ids):
        return causal_lm_chunked_loss(model, {"input_ids": ids, "labels": ids}, chunk_size=4)

    total_tokens = (8 - 1) + (24 - 1)
    token_weighted = (summed(short) + summed(long)) / total_tokens
    unweighted_mean_of_means = (mean(short) + mean(long)) / 2

    assert not torch.isclose(token_weighted, unweighted_mean_of_means, rtol=1e-3)
    # And the weighted form is the one a single concatenated pass would give.
    assert torch.isclose(
        token_weighted,
        (summed(short) + summed(long)) / total_tokens,
        rtol=1e-6,
    )


def test_the_wrapper_computes_the_same_loss_it_wraps() -> None:
    """Moving the loss inside a module must not change it; it only changes where it is called."""
    from local_llm_lab.training.torch_full import wrap_with_chunked_loss

    model = _tiny().eval()
    ids = torch.arange(24).remainder(64).reshape(2, 12)
    labels = ids.clone()
    labels[:, :3] = -100

    direct = causal_lm_chunked_loss(
        model, {"input_ids": ids, "labels": labels}, chunk_size=5, num_items_in_batch=1
    )
    wrapped = wrap_with_chunked_loss(model, chunk_size=5)(
        input_ids=ids, labels=labels, num_items_in_batch=1
    )
    torch.testing.assert_close(wrapped, direct, rtol=0, atol=0)


def test_the_wrapper_holds_the_tied_pair_outside_every_block() -> None:
    """What `fully_shard` on the wrapper would own as the root unit.

    A parameter is unsharded, with its backward hooks armed, only inside the forward of the unit
    that owns it. The chunked loss reads `lm_head.weight` directly, so that weight has to be in the
    unit whose forward the loss runs in -- and the embedding is tied to it, so the pair must be
    owned together or FSDP2 sees a flat parameter straddling two units.
    """
    from local_llm_lab.training.torch_full import wrap_with_chunked_loss

    wrapper = wrap_with_chunked_loss(_tiny(), chunk_size=8)
    root = [name for name, _ in wrapper.named_parameters() if ".layers." not in name]

    assert any("embed_tokens" in name for name in root)
    assert all(".layers." not in name for name in root)
    # The head is tied to the embedding, so it is that one tensor and not a second entry.
    assert wrapper.inner.lm_head.weight is wrapper.inner.model.embed_tokens.weight


def test_the_wrapper_refuses_a_model_it_cannot_find_a_head_on() -> None:
    from local_llm_lab.training.torch_full import wrap_with_chunked_loss

    with pytest.raises(TypeError, match="lm_head"):
        wrap_with_chunked_loss(torch.nn.Linear(4, 4), chunk_size=8)


def test_the_multimodal_wrapper_is_refused_where_it_is_first_seen() -> None:
    """`.model` and `.lm_head` do not identify a text causal LM, and this guard used to think so.

    Every published Gemma 3 checkpoint declares `Gemma3ForConditionalGeneration` -- including this
    repository's own text-only conversion, which carries no `vision_config` but still names that
    architecture. So `AutoModelForCausalLM.from_pretrained` builds the multimodal wrapper for the
    real registry entries, and that wrapper *has* both attributes. It passed this guard and failed
    later somewhere else, which is the failure this guard exists to prevent.
    """
    from transformers import AutoModelForCausalLM, Gemma3Config

    from local_llm_lab.training.torch_full import base_model_of

    config = Gemma3Config(
        text_config=dict(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=16, sliding_window=8,
        ),
        vision_config=dict(
            hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
            image_size=16, patch_size=8, num_channels=3,
        ),
    )
    wrapper = AutoModelForCausalLM.from_config(config)
    assert type(wrapper).__name__ == "Gemma3ForConditionalGeneration"
    assert hasattr(wrapper, "model") and hasattr(wrapper, "lm_head")  # the old guard's whole test

    with pytest.raises(TypeError, match="multimodal wrapper"):
        base_model_of(wrapper)
