"""E2's transport rule under Amendment 1: nested, deterministic, and identity-outside-the-rank."""

from __future__ import annotations

import numpy as np
import pytest

from local_llm_lab.pipeline.state_programme.transport import fit, nearer_the_successor


def moving(rng, n=300, p=40, rank=6, ratio=0.056, noise=0.02):
    """Sources, and successors displaced by a low-rank function of the source.

    `ratio` is the displacement's norm as a fraction of the state's, normalised explicitly. The
    default is the corpus's own regime, ~1,548 against ~27,767 at the headline depth: the state
    barely moves, which is exactly when the identity is expensive to represent and cheap to keep.
    """
    x = rng.normal(size=(n, p))
    raw = (x @ rng.normal(size=(p, rank))) @ rng.normal(size=(rank, p))
    raw = raw / np.linalg.norm(raw, axis=1).mean() * np.linalg.norm(x, axis=1).mean()
    displacement = raw * ratio + rng.normal(size=(n, p)) * noise
    return x, x + displacement


def test_the_ladder_is_one_decomposition_sliced(rng=np.random.default_rng(0)):
    """Amendment 1 §2 asserts this; NIPALS is greedy with deflation, so it must hold."""
    x, y = moving(rng)
    full = fit(x, y - x, 16)
    for rank in (1, 2, 4, 8):
        short = fit(x, y - x, rank)
        assert np.allclose(full.weights[:, :rank], short.weights, atol=1e-8)
        assert np.allclose(full.coefficients(rank), short.coefficients(rank), atol=1e-6)


def test_the_power_iteration_restarts_from_the_current_cross_product(rng=np.random.default_rng(8)):
    """Restarting from the previous component's direction stalls: it lies in the part just deflated."""
    x, y = moving(rng)
    assert fit(x, y - x, 8).max_rank == 8


def test_the_fit_is_deterministic(rng=np.random.default_rng(1)):
    """No random number enters it, so two fits of one input agree bit for bit."""
    x, y = moving(rng)
    a, b = fit(x, y - x, 8), fit(x, y - x, 8)
    assert np.array_equal(a.weights, b.weights)
    assert a.iterations == b.iterations


def test_more_rank_transports_further(rng=np.random.default_rng(2)):
    """The displacement here is rank 6, so the gate should climb with rank and then hold."""
    x, y = moving(rng)
    fitted = fit(x[:200], y[:200] - x[:200], 16)
    scores = []
    for rank in (1, 2, 4, 8):
        step = fitted.target_mean + ((x[200:] - fitted.mean) / fitted.scale) @ fitted.coefficients(rank)
        scores.append(nearer_the_successor(x[200:] + step, x[200:], y[200:]).mean())
    assert scores[-1] > scores[0], scores


def test_the_identity_is_outside_the_rank(rng=np.random.default_rng(3)):
    """§2's reason, and it only bites in the corpus's regime, where the state barely moves.

    With a displacement as large as the state the two forms are comparable; as it shrinks, the map
    straight to the successor degrades because its rank goes on reproducing the state, while the
    correction form is unaffected. The test is run at the corpus's ratio.
    """
    x, y = moving(rng)
    to_successor = fit(x[:200], y[:200], 4)          # the form the amendment rejects
    to_displacement = fit(x[:200], y[:200] - x[:200], 4)  # the form it adopts
    direct = to_successor.target_mean + ((x[200:] - to_successor.mean) / to_successor.scale) @ to_successor.coefficients(4)
    step = to_displacement.target_mean + ((x[200:] - to_displacement.mean) / to_displacement.scale) @ to_displacement.coefficients(4)
    assert nearer_the_successor(x[200:] + step, x[200:], y[200:]).mean() > \
           nearer_the_successor(direct, x[200:], y[200:]).mean()


def test_rank_zero_of_the_correction_is_the_withdrawn_constant_displacement(rng=np.random.default_rng(4)):
    """§2: the withdrawn rule is the floor of this ladder, since the intercept is the mean displacement."""
    x, y = moving(rng)
    fitted = fit(x, y - x, 4)
    assert np.allclose(fitted.target_mean, (y - x).mean(axis=0), atol=1e-8)


def test_cost_m_applies_the_one_step_rule_m_times(rng=np.random.default_rng(5)):
    x, y = moving(rng)
    fitted = fit(x, y - x, 4)
    once = fitted.apply(x[:5], rank=4, times=1)
    twice = fitted.apply(x[:5], rank=4, times=2)
    assert np.allclose(twice, fitted.apply(once, rank=4, times=1))
    assert not np.allclose(once, twice)


def test_a_rank_beyond_the_components_refuses(rng=np.random.default_rng(6)):
    x, y = moving(rng)
    fitted = fit(x, y - x, 4)
    with pytest.raises(ValueError, match="exceeds"):
        fitted.coefficients(8)


def test_a_transition_of_cost_below_one_refuses(rng=np.random.default_rng(7)):
    x, y = moving(rng)
    with pytest.raises(ValueError, match="cost below one"):
        fit(x, y - x, 4).apply(x[:2], rank=2, times=0)


def test_the_gate_is_a_two_point_comparison_not_a_retrieval():
    transported = np.array([[0.0, 0.0]])
    source = np.array([[1.0, 0.0]])
    successor = np.array([[0.5, 0.0]])
    assert nearer_the_successor(transported, source, successor).tolist() == [True]
    assert nearer_the_successor(source, source, successor).tolist() == [False]
