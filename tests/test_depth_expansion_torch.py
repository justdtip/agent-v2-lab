"""The torch path of ``depth_expansion``, checked on the property the mechanism exists to have.

Kept apart from ``test_depth_expansion.py`` so that file stays importable without torch: it
covers ``ExpansionSpec``, which is backend-free, and the suite's import closure is a rule rather
than an accident.

Every model here is a randomly initialised tiny Gemma 3. Nothing loads a checkpoint, so these run
anywhere the package imports and take no lock.
"""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from local_llm_lab.depth_expansion import (  # noqa: E402
    ExpansionSpec,
    expand_model_torch,
    set_full_extension_training_torch,
    set_output_warmup_torch,
    trainable_parameter_count_torch,
)

#: Six layers puts exactly one globally-attending layer (index 5) in the stack, so the tests can
#: insert onto a sliding layer and onto a global one and tell the two cases apart.
LAYERS = 6


def _tiny(seed: int = 0):
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    torch.manual_seed(seed)
    config = Gemma3TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=LAYERS,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, sliding_window=8,
        rms_norm_eps=1e-6, use_cache=False,
    )
    model = Gemma3ForCausalLM(config).eval()
    # Gemma's RMSNorm scales by ``(1 + weight)`` and initialises ``weight`` to zero, so at default
    # initialisation every norm is the identity: a block copied *without* its norms would pass the
    # identity check below. Randomising them is what gives that check its teeth.
    for parameter in model.parameters():
        if parameter.dim() == 1:
            torch.nn.init.normal_(parameter, std=0.3)
    return model


def _logits(model) -> torch.Tensor:
    ids = torch.arange(24, dtype=torch.long).remainder(64).reshape(2, 12)
    with torch.no_grad():
        return model(input_ids=ids, use_cache=False).logits


@pytest.mark.parametrize("after", list(range(LAYERS)))
def test_expanded_model_is_an_exact_identity_at_initialisation(after: int) -> None:
    """The whole point of ``copied_identity``: at step 0 the expanded model *is* the base model.

    Checked at every insertion point rather than one, because the failure this guards against --
    a block copied field-by-field against a different architecture's part list -- shows up only
    where the missing part carries a non-default value.
    """
    base = _tiny()
    expected = _logits(base)

    expanded = copy.deepcopy(base)
    expand_model_torch(expanded, ExpansionSpec((after,)))
    expanded.eval()

    assert torch.equal(_logits(expanded), expected)


def test_inserted_block_inherits_its_source_attention_type() -> None:
    """Gemma 3 selects each layer's mask by position, so an insertion rewrites the pattern.

    The copy takes the type of the block it copies and every original layer keeps its own. The
    regression this pins: indexing the type list *after* an earlier insertion has already shifted
    it hands the copy a neighbour's type. A zeroed block outputs zero whichever way it attends, so
    the identity test above passes either way and the recipe is wrong from the first warmed step.
    """
    base = _tiny()
    original = list(base.config.layer_types)
    assert original.count("full_attention") == 1, original
    global_index = original.index("full_attention")

    expanded = copy.deepcopy(base)
    # Two insertions, the first strictly before the global layer, so the type list has already
    # shifted by the time the second one reads it.
    added = expand_model_torch(expanded, ExpansionSpec((0, global_index)))
    types = list(expanded.config.layer_types)

    assert len(types) == LAYERS + 2
    assert expanded.config.num_hidden_layers == LAYERS + 2
    # The original global layer moved by one insertion, and its copy sits directly after it.
    assert types[global_index + 1] == "full_attention"
    assert types[global_index + 2] == "full_attention"
    assert [index for index, kind in enumerate(types) if kind == "full_attention"] == [
        global_index + 1,
        global_index + 2,
    ]
    # Every layer is bound to the position it now occupies; a stale ``layer_idx`` addresses the
    # wrong KV cache entry during generation while training never notices.
    assert [layer.self_attn.layer_idx for layer in expanded.model.layers] == list(range(LAYERS + 2))
    assert [layer.self_attn.layer_type for layer in expanded.model.layers] == types
    assert added == (1, global_index + 2)


def test_expansion_freezes_everything_inherited_and_opens_only_the_added_blocks() -> None:
    model = _tiny()
    added = expand_model_torch(model, ExpansionSpec((1, 3)))

    added_parameters = sum(
        p.numel() for index in added for p in model.model.layers[index].parameters()
    )
    assert trainable_parameter_count_torch(model) == added_parameters
    for index, layer in enumerate(model.model.layers):
        expected = index in added
        assert all(p.requires_grad is expected for p in layer.parameters())


def test_warmup_opens_only_the_two_zeroed_residual_outputs() -> None:
    """Stage 1 trains the projections that were zeroed to make the identity, and nothing else."""
    model = _tiny()
    added = expand_model_torch(model, ExpansionSpec((1, 3)))

    warmup = set_output_warmup_torch(model, added)
    expected = sum(
        getattr(getattr(model.model.layers[index], parent), child).weight.numel()
        for index in added
        for parent, child in (("self_attn", "o_proj"), ("mlp", "down_proj"))
    )
    assert warmup == expected
    # Every open parameter is one of those projections, in one of the added blocks.
    open_names = {name for name, p in model.named_parameters() if p.requires_grad}
    assert open_names == {
        f"model.layers.{index}.{parent}.{child}.weight"
        for index in added
        for parent, child in (("self_attn", "o_proj"), ("mlp", "down_proj"))
    }

    # Stage 2 opens strictly more: the whole of each added block.
    full = set_full_extension_training_torch(model, added)
    assert full > warmup


def test_expansion_refuses_an_index_the_stack_does_not_have() -> None:
    model = _tiny()
    with pytest.raises(ValueError, match="but model has"):
        expand_model_torch(model, ExpansionSpec((LAYERS,)))


def test_expansion_refuses_a_block_whose_residual_outputs_it_cannot_find() -> None:
    """An architecture whose residual outputs are named differently must stop, not train."""
    model = _tiny()
    del model.model.layers[1].mlp.down_proj
    with pytest.raises(TypeError, match="down_proj"):
        expand_model_torch(model, ExpansionSpec((1,)))
