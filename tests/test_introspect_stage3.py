"""The renderer, the vector generators and the damage meter, on a real Gemma 4 text model."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from local_llm_lab.introspect import meter as meter_mod  # noqa: E402
from local_llm_lab.introspect import vectors as vec  # noqa: E402
from local_llm_lab.introspect.render import (  # noqa: E402
    final_prompt_position, render_prompt, render_supervised,
)
from local_llm_lab.residual_patch import decoder_blocks  # noqa: E402


@pytest.fixture(scope="module")
def pair():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    tok.pad_token = tok.eos_token
    try:
        cfg = transformers.AutoConfig.for_model(
            # The embedding has to cover the tokenizer, or a real prompt indexes past it.
            "gemma4_text", vocab_size=len(tok), hidden_size=32, intermediate_size=64,
            num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            max_position_embeddings=256, hidden_size_per_layer_input=16,
            tie_word_embeddings=True)
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(cfg).eval()
    except Exception as exc:                                   # pragma: no cover
        pytest.skip(f"no gemma4_text config: {exc}")
    return model, tok


# -- the renderer ------------------------------------------------------------------------------

def test_the_supervised_boundary_is_exact(pair):
    _model, tok = pair
    ids, prompt_length = render_supervised(tok, "hello there", "YES. It is about bread.")
    assert ids[:prompt_length] == render_prompt(tok, "hello there")
    assert len(ids) > prompt_length
    assert final_prompt_position(ids[:prompt_length]) == prompt_length - 1


def test_the_site_is_never_inside_the_supervised_answer(pair):
    _model, tok = pair
    ids, prompt_length = render_supervised(tok, "a longer question here", "NO.")
    site = final_prompt_position(ids[:prompt_length])
    assert site < prompt_length, "the site must be in the prompt, never in what is scored"


def test_an_empty_target_is_refused(pair):
    _model, tok = pair
    with pytest.raises(ValueError):
        render_supervised(tok, "hello", "")


# -- contamination gates -----------------------------------------------------------------------

def test_no_baseline_word_is_a_training_concept():
    """A concept whose own baseline contains it has a direction pointing at nothing."""
    assert len(set(w.lower() for w in vec.BASELINE_POOL)) == len(vec.BASELINE_POOL)
    with pytest.raises(ValueError):
        vec.concept_vector(None, None, None, "table", 1, device="cpu")


def test_the_legacy_concepts_are_not_in_the_baseline_pool():
    """The five the base-model grids used; a shared word would join two records by accident."""
    pool = {w.lower() for w in vec.BASELINE_POOL}
    for word in ("bread", "the ocean", "paris", "betrayal", "butt holes", "lighthouses"):
        assert word not in pool, word


# -- the vector generators ---------------------------------------------------------------------

def test_a_concept_vector_is_finite_non_zero_and_the_right_width(pair):
    model, tok = pair
    blocks = decoder_blocks(model)
    v = vec.concept_vector(model, tok, blocks, "bread", 2, device=torch.device("cpu"),
                           pool=vec.BASELINE_POOL[:6])
    assert v.shape == (32,) and v.dtype == torch.float32
    assert torch.isfinite(v).all() and float(v.norm()) > 0


def test_two_concepts_give_different_directions(pair):
    model, tok = pair
    blocks = decoder_blocks(model)
    a = vec.concept_vector(model, tok, blocks, "bread", 2, device=torch.device("cpu"),
                           pool=vec.BASELINE_POOL[:6])
    b = vec.concept_vector(model, tok, blocks, "harbourmaster", 2, device=torch.device("cpu"),
                           pool=vec.BASELINE_POOL[:6])
    cosine = float(torch.nn.functional.cosine_similarity(a[None], b[None]))
    assert cosine < 0.999, f"the two concepts are nearly the same direction ({cosine:.4f})"


def test_spectrum_matched_noise_keeps_the_bank_norm_and_carries_no_concept():
    torch.manual_seed(0)
    bank = torch.randn(40, 32) * torch.linspace(3.0, 0.2, 32)
    g = torch.Generator().manual_seed(1)
    drawn = vec.spectrum_matched(bank, generator=g)
    assert drawn.shape == (32,)
    ratio = float(drawn.norm()) / float(bank.norm(dim=-1).mean())
    assert 0.8 < ratio < 1.25, ratio
    best = max(float(torch.nn.functional.cosine_similarity(drawn[None], row[None]))
               for row in bank)
    assert best < 0.8, f"the draw landed on a bank direction (cos {best:.3f})"


def test_a_permuted_vector_keeps_its_norm_and_loses_its_direction():
    torch.manual_seed(0)
    bank = torch.randn(40, 32)
    v = bank[3].clone()
    g = torch.Generator().manual_seed(2)
    out = vec.permuted_in_basis(v, bank, generator=g)
    assert float(out.norm()) == pytest.approx(float(v.norm()), rel=1e-4)
    assert abs(float(torch.nn.functional.cosine_similarity(out[None], v[None]))) < 0.8


# -- the meter ---------------------------------------------------------------------------------

def test_the_battery_is_not_binary_and_not_about_the_model():
    for prompt in meter_mod.BATTERY:
        low = prompt.lower()
        assert not low.startswith(("is ", "are ", "do ", "does ", "can ", "did ")), prompt
        for word in ("you", "your", "activation", "inject", "detect", "unusual"):
            assert word not in low.split(), prompt
    assert len(meter_mod.BATTERY) == len(set(meter_mod.BATTERY)) == 24


def test_damage_is_zero_at_zero_scale_and_negative_beyond(pair):
    model, tok = pair
    m = meter_mod.DamageMeter(model, tok, device=torch.device("cpu"), dtype=torch.float32,
                              battery=meter_mod.BATTERY[:4])
    torch.manual_seed(0)
    v = torch.randn(32)
    assert m.damage(v, 2, 0.0).damage == pytest.approx(0.0, abs=1e-6)
    hard = m.damage(v, 2, 40.0).damage
    assert hard < -0.05, hard


def test_damage_is_monotone_enough_to_interpolate(pair):
    model, tok = pair
    m = meter_mod.DamageMeter(model, tok, device=torch.device("cpu"), dtype=torch.float32,
                              battery=meter_mod.BATTERY[:4])
    torch.manual_seed(0)
    v = torch.randn(32)
    curve = m.curve(v, 2, [0.0, 1.0, 4.0, 16.0, 64.0])
    damages = [d for _s, d in curve]
    assert damages[0] > damages[-1]
    assert damages == sorted(damages, reverse=True), damages


def test_the_clean_baseline_can_be_invalidated(pair):
    """LoRA changes the forward pass; a stale baseline would report the adapter as damage."""
    model, tok = pair
    m = meter_mod.DamageMeter(model, tok, device=torch.device("cpu"), dtype=torch.float32,
                              battery=meter_mod.BATTERY[:3])
    first = m.clean()
    assert m.clean() is first
    m.reset_clean()
    assert m.clean() is not first


def test_the_batched_curve_equals_the_one_at_a_time_one(pair):
    """The sweep is seventy-two rows in one forward; it must measure what nine forwards measured."""
    model, tok = pair
    m = meter_mod.DamageMeter(model, tok, device=torch.device("cpu"), dtype=torch.float32,
                              battery=meter_mod.BATTERY[:4])
    torch.manual_seed(0)
    v = torch.randn(32)
    scales = [0.0, 2.0, 8.0, 32.0]
    batched = m.curve(v, 2, scales)
    singly = [(s, m.damage(v, 2, s).damage) for s in scales]
    for (s_a, d_a), (s_b, d_b) in zip(batched, singly):
        assert s_a == s_b
        assert d_a == pytest.approx(d_b, abs=1e-4), (s_a, d_a, d_b)


def test_a_rung_below_the_whole_curve_takes_the_smallest_scale(pair):
    """Asked for less damage than even the gentlest sampled scale produces, the answer is that
    gentlest scale -- not the strongest, which is what the first version returned and which turned
    a request for -0.002 nats into fifteen."""
    model, tok = pair
    m = meter_mod.DamageMeter(model, tok, device=torch.device("cpu"), dtype=torch.float32,
                              battery=meter_mod.BATTERY[:3])
    torch.manual_seed(0)
    v = torch.randn(32)
    rungs = m.scales_for_ladder(v, 2, (-1e-9, -0.05, -1e6), residual_norm=50.0, verify=False)
    smallest = min(s for s, _d in m.curve(v, 2, [0.0]))
    assert rungs[-1e-9][0] <= rungs[-0.05][0], "a gentler rung must not get a stronger scale"
    assert rungs[-1e6][0] >= rungs[-0.05][0], "an unreachable rung takes the strongest scale"
    assert smallest is not None
