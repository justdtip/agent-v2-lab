"""Adapters by in-place swap: identity at birth, trainable, mergeable, and nothing else touched."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from local_llm_lab.lora_torch import (  # noqa: E402
    DEFAULT_TARGETS, LoRALinear, apply_lora, count_lora_parameters, load_lora_state_dict,
    lora_parameters, lora_state_dict, merge_lora,
)


def _model():
    from transformers import AutoModelForCausalLM, LlamaConfig
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
        tie_word_embeddings=True)).eval()


IDS = torch.tensor([[1, 5, 9, 12, 30, 7]])


def test_the_adapter_is_the_identity_at_initialisation():
    """lora_B starts at zero, so the first step's loss is the base model's, exactly."""
    model = _model()
    before = float(model(input_ids=IDS, labels=IDS).loss)
    swapped = apply_lora(model, r=4, alpha=8)
    after = float(model(input_ids=IDS, labels=IDS).loss)
    assert swapped == 3 * len(DEFAULT_TARGETS)
    assert after == pytest.approx(before, abs=1e-9)


def test_a_non_zero_adapter_changes_the_output():
    """Otherwise the identity test above would pass on an adapter that does nothing at all."""
    model = _model()
    apply_lora(model, r=4, alpha=8)
    before = float(model(input_ids=IDS, labels=IDS).loss)
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                module.lora_B.normal_(0, 0.1)
    assert abs(float(model(input_ids=IDS, labels=IDS).loss) - before) > 1e-4


def test_only_the_adapters_are_trainable_after_a_frozen_load():
    """`load_text_causal_lm` returns model.eval().requires_grad_(False); adapters come after."""
    model = _model()
    model.requires_grad_(False)
    apply_lora(model, r=4, alpha=8)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable, "nothing is trainable — adapters must be created after the frozen load"
    assert all(n.endswith(("lora_A", "lora_B")) for n in trainable), trainable
    assert count_lora_parameters(model) == sum(p.numel() for p in lora_parameters(model))


def test_the_head_is_never_adapted():
    """The chunked loss applies lm_head itself, so an adapter there is bypassed and never trained."""
    model = _model()
    apply_lora(model, r=4, alpha=8)
    assert not isinstance(model.lm_head, LoRALinear)


def test_gradients_reach_the_adapters_and_not_the_base():
    model = _model()
    model.requires_grad_(False)
    apply_lora(model, r=4, alpha=8)
    with torch.no_grad():
        for _n, m in model.named_modules():
            if isinstance(m, LoRALinear):
                m.lora_B.normal_(0, 0.05)
    model(input_ids=IDS, labels=IDS).loss.backward()
    got = [p.grad is not None and float(p.grad.abs().sum()) > 0 for p in lora_parameters(model)]
    assert any(got)
    base = [p.grad for n, p in model.named_parameters() if not n.endswith(("lora_A", "lora_B"))]
    assert all(g is None for g in base)


def test_a_state_dict_round_trips():
    model = _model()
    apply_lora(model, r=4, alpha=8)
    with torch.no_grad():
        for _n, m in model.named_modules():
            if isinstance(m, LoRALinear):
                m.lora_B.normal_(0, 0.1)
    saved = lora_state_dict(model)
    reference = float(model(input_ids=IDS, labels=IDS).loss)

    fresh = _model()
    apply_lora(fresh, r=4, alpha=8)
    assert load_lora_state_dict(fresh, saved) == len(saved) // 2
    assert float(fresh(input_ids=IDS, labels=IDS).loss) == pytest.approx(reference, abs=1e-9)
    assert all(k.endswith(("lora_A", "lora_B")) for k in saved)


def test_merging_reproduces_the_adapted_output_and_restores_plain_linears():
    model = _model()
    apply_lora(model, r=4, alpha=8)
    with torch.no_grad():
        for _n, m in model.named_modules():
            if isinstance(m, LoRALinear):
                m.lora_B.normal_(0, 0.1)
    before = float(model(input_ids=IDS, labels=IDS).loss)
    assert merge_lora(model) == 3 * len(DEFAULT_TARGETS)
    assert float(model(input_ids=IDS, labels=IDS).loss) == pytest.approx(before, abs=1e-5)
    assert not any(isinstance(m, LoRALinear) for m in model.modules())


def test_a_target_that_is_not_a_linear_is_refused_rather_than_skipped():
    model = _model()
    model.model.layers[0].self_attn.q_proj = torch.nn.Identity()
    with pytest.raises(TypeError):
        apply_lora(model, r=4, alpha=8)
