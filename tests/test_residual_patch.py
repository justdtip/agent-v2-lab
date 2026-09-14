"""The training-time injection, and the four ways it could be silently wrong.

Every fixture here injects at a NON-ZERO site, strictly inside the sequence. At site zero a
single-index write and a write-to-the-end produce the same tensor, so a fixture that injects from
position 0 passes under both and establishes nothing — method entry 35, applied literally.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from local_llm_lab.residual_patch import (  # noqa: E402
    PatchPlan, PlannedPatch, build_masks, decoder_blocks,
)

SITE = 3
WIDTH = 8


def _model():
    """A Gemma 4 text model small enough to run, with the per-layer embedding path ON.

    PLE matters: with it off, a pre-hook that returns a bare tensor passes silently, and the test
    that is supposed to catch that defect goes green.
    """
    from transformers import AutoModelForCausalLM
    cfg = transformers.AutoConfig.for_model(
        "gemma4_text", vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        max_position_embeddings=64, hidden_size_per_layer_input=16, tie_word_embeddings=True)
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg).train()


@pytest.fixture
def model():
    try:
        return _model()
    except Exception as exc:                       # pragma: no cover
        pytest.skip(f"no gemma4_text config in this transformers: {exc}")


def _plan(hidden: int, seed: int = 0, layer: int = 2):
    g = torch.Generator().manual_seed(seed)
    return PatchPlan(layer=[layer, layer], site=[SITE, SITE], scale=[4.0, 4.0],
                     vector=[torch.randn(hidden, generator=g),
                             torch.randn(hidden, generator=g)])


def _loss(model, plan, *, checkpointing=None, hidden=32):
    ids = torch.arange(2 * WIDTH).reshape(2, WIDTH) % 60
    if checkpointing is not None:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": checkpointing})
        model.config.use_cache = False
    try:
        patches = []
        if plan is not None:
            blocks = decoder_blocks(model)
            masks = build_masks(plan, width=WIDTH, hidden=hidden,
                                device=ids.device, dtype=torch.float32)
            patches = [PlannedPatch(blocks[layer], mask=m, delta=d, layer=layer)
                       for layer, (m, d) in masks.items()]
        for p in patches:
            p.__enter__()
        try:
            loss = model(input_ids=ids, labels=ids).loss
            model.zero_grad(set_to_none=True)
            loss.backward()
            grads = torch.cat([p.grad.flatten() for p in model.parameters()
                               if p.grad is not None])
            return float(loss), grads, patches
        finally:
            for p in patches:
                p.__exit__()
    finally:
        if checkpointing is not None:
            model.gradient_checkpointing_disable()


def test_checkpointing_does_not_change_the_injected_forward(model):
    """(a) no checkpointing, injected  ==  (b) checkpointed, injected — under either mode."""
    plan = _plan(32)
    base_loss, base_grads, _ = _loss(model, plan)
    for reentrant in (True, False):
        loss, grads, _ = _loss(_model(), _plan(32), checkpointing=reentrant)
        assert loss == pytest.approx(base_loss, rel=1e-6), reentrant
        worst = (grads - base_grads).abs().max() / base_grads.abs().max().clamp(min=1e-12)
        assert float(worst) < 1e-5, f"use_reentrant={reentrant} worst relative {float(worst):.3e}"


def test_the_fixture_can_tell_an_injection_from_no_injection(model):
    """Otherwise the agreement above would be evidence of nothing."""
    injected, _, _ = _loss(model, _plan(32))
    clean, _, _ = _loss(_model(), None)
    assert abs(injected - clean) > 1e-3, (injected, clean)


def test_the_fixture_can_tell_one_vector_from_another(model):
    a, _, _ = _loss(model, _plan(32, seed=0))
    b, _, _ = _loss(_model(), _plan(32, seed=7))
    assert abs(a - b) > 1e-4, (a, b)


def test_a_single_index_write_differs_from_a_write_to_the_end(model):
    """The defect that would have shipped: `sustain=False` cannot mean 'one token' here.

    In the bench, prefill-only works because decoding is a series of one-token forwards and the
    hook short-circuits them. In training there is one forward, so that short-circuit is dead code
    and the delta lands on the site AND every supervised answer token after it — which also makes
    the task trivial, since the answer being scored carries the delta.
    """
    hidden = 32
    plan = _plan(hidden)
    one, _, _ = _loss(model, plan)

    wide = _model()
    masks = build_masks(plan, width=WIDTH, hidden=hidden, device=torch.device("cpu"),
                        dtype=torch.float32)
    layer, (mask, delta) = next(iter(masks.items()))
    mask[:, SITE:, 0] = 1.0                       # the buggy form: site to the end
    ids = torch.arange(2 * WIDTH).reshape(2, WIDTH) % 60
    with PlannedPatch(decoder_blocks(wide)[layer], mask=mask, delta=delta, layer=layer):
        many = float(wide(input_ids=ids, labels=ids).loss)
    assert abs(one - many) > 1e-4, (one, many)


def test_a_bare_tensor_return_mutilates_the_forward_when_ple_is_on(model):
    """Why the hook returns the whole tuple. With PLE on this raises; with PLE off it would pass
    silently, which is the case the arity check below covers instead."""
    blocks = decoder_blocks(model)
    handle = blocks[2].register_forward_pre_hook(lambda _m, args: args[0] + 1.0)
    ids = torch.arange(2 * WIDTH).reshape(2, WIDTH) % 60
    try:
        with pytest.raises(Exception):
            model(input_ids=ids, labels=ids)
    finally:
        handle.remove()


def test_a_none_second_argument_is_not_treated_as_a_fault():
    """A checkpoint with the per-layer embedding disabled passes None there legitimately.

    The first version of this hook asserted otherwise and stopped a working run on the real 31B,
    whose hidden_size_per_layer_input is 0.
    """
    class Block(torch.nn.Module):
        def forward(self, hidden, extra=None):
            return hidden

    block = Block()
    mask = torch.zeros(1, WIDTH, 1)
    mask[0, SITE, 0] = 1.0
    with PlannedPatch(block, mask=mask, delta=torch.ones(1, 4), layer=0) as patch:
        block(torch.zeros(1, WIDTH, 4), None)
    assert patch.fires == 1


def test_a_changing_argument_count_is_refused():
    """The invariant that actually matters, and it holds whether or not PLE is on."""
    class Block(torch.nn.Module):
        def forward(self, hidden, *extra):
            return hidden

    block = Block()
    mask = torch.zeros(1, WIDTH, 1)
    mask[0, SITE, 0] = 1.0
    with PlannedPatch(block, mask=mask, delta=torch.ones(1, 4), layer=0):
        block(torch.zeros(1, WIDTH, 4), None)
        with pytest.raises(AssertionError, match="positional arguments"):
            block(torch.zeros(1, WIDTH, 4))


def test_the_hook_reports_a_consistent_fingerprint(model):
    _value, _grads, patches = _loss(model, _plan(32), checkpointing=False)
    for p in patches:
        assert p.fires >= 1 and p.consistent(), (p.layer, p.fires, p.fingerprints)


def test_a_site_outside_the_real_tokens_is_refused():
    plan = PatchPlan(layer=[1], site=[9], scale=[1.0], vector=[torch.ones(32)])
    with pytest.raises(IndexError):
        build_masks(plan, width=WIDTH, hidden=32, device=torch.device("cpu"),
                    dtype=torch.float32)


def test_a_row_the_layer_does_not_own_gets_a_zero_mask():
    plan = PatchPlan(layer=[1, 2], site=[SITE, SITE], scale=[1.0, 1.0],
                     vector=[torch.ones(32), torch.ones(32)])
    masks = build_masks(plan, width=WIDTH, hidden=32, device=torch.device("cpu"),
                        dtype=torch.float32)
    assert set(masks) == {1, 2}
    assert float(masks[1][0][1].sum()) == 0.0 and float(masks[1][0][0].sum()) == 1.0
    assert float(masks[2][0][0].sum()) == 0.0 and float(masks[2][0][1].sum()) == 1.0


# -- the span branch ---------------------------------------------------------------------------
# Shipped with no test at all. An adversarial review mutated build_masks nine ways and ran the whole
# suite against each: nine survived, and three of them make --scope prompt silently identical to
# --scope final, which is the entire claim the steering result rests on.

def _sp_plan(**kw):
    from local_llm_lab.residual_patch import PatchPlan
    base = dict(layer=[0], site=[3], scale=[2.0], vector=[torch.ones(4)])
    base.update(kw)
    return PatchPlan(**base)


def _sp_masks(plan, *, width=8, lengths=None):
    from local_llm_lab.residual_patch import build_masks
    return build_masks(plan, width=width, hidden=4, device=torch.device("cpu"),
                       dtype=torch.float32, lengths=lengths)


def _sp_marked(mask, row=0):
    return [int(i) for i in torch.nonzero(mask[row, :, 0]).flatten().tolist()]


def test_a_plan_with_no_span_marks_the_site_and_only_the_site():
    """The separating control. Without it every test below passes on a build_masks that spans
    unconditionally, which is precisely the mutation that makes the two scopes identical."""
    mask, _delta = _sp_masks(_sp_plan())[0]
    assert _sp_marked(mask) == [3], _sp_marked(mask)


def test_a_span_marks_every_position_in_the_half_open_range():
    mask, _delta = _sp_masks(_sp_plan(span=[(1, 4)]))[0]
    assert _sp_marked(mask) == [1, 2, 3], "half-open: start in, end out"


def test_a_span_is_not_quietly_ignored():
    plain = _sp_masks(_sp_plan())[0][0]
    spanned = _sp_masks(_sp_plan(span=[(0, 6)]))[0][0]
    assert not torch.equal(plain, spanned), "a span that changes nothing is a scope that lied"


def test_a_span_may_not_reach_into_the_padding():
    with pytest.raises(IndexError, match="row 0"):
        _sp_masks(_sp_plan(span=[(2, 7)]), width=8, lengths=[5])


def test_an_empty_or_inverted_or_negative_span_is_refused():
    for bad in ((3, 3), (4, 2), (-1, 3)):
        with pytest.raises(IndexError, match="row 0"):
            _sp_masks(_sp_plan(span=[bad]), width=8, lengths=[8])


def test_a_span_that_is_not_a_pair_of_ints_is_refused_with_its_row():
    for bad in (5, (True, 2), (1, 2, 3), "ab"):
        with pytest.raises(IndexError, match="row 0"):
            _sp_masks(_sp_plan(span=[bad]), width=8, lengths=[8])


def test_each_row_gets_its_own_span():
    plan = _sp_plan(layer=[0, 0], site=[1, 1], scale=[1.0, 1.0],
                    vector=[torch.ones(4), torch.ones(4)], span=[(0, 2), (5, 7)])
    mask, _delta = _sp_masks(plan, width=8, lengths=[8, 8])[0]
    assert _sp_marked(mask, 0) == [0, 1] and _sp_marked(mask, 1) == [5, 6]


def test_the_span_list_must_have_one_entry_per_row():
    with pytest.raises(ValueError, match="span"):
        _sp_plan(layer=[0, 0], site=[1, 1], scale=[1.0, 1.0],
                 vector=[torch.ones(4), torch.ones(4)], span=[(0, 2)])


def test_a_span_on_one_row_does_not_disable_the_site_check_on_another():
    plan = _sp_plan(layer=[0, 0], site=[1, 99], scale=[1.0, 1.0],
                    vector=[torch.ones(4), torch.ones(4)], span=[(0, 2), None])
    with pytest.raises(IndexError, match="row 1"):
        _sp_masks(plan, width=8, lengths=[8, 8])
