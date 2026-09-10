"""E2's transport rule under Amendment 1: a rank-r supervised map, nested and deterministic."""

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
    full = fit(x, y, 16)
    for rank in (1, 2, 4, 8):
        short = fit(x, y, rank)
        assert np.allclose(full.weights[:, :rank], short.weights, atol=1e-8)
        assert np.allclose(full.coefficients(rank), short.coefficients(rank), atol=1e-6)


def test_the_power_iteration_restarts_from_the_current_cross_product(rng=np.random.default_rng(8)):
    """Restarting from the previous component's direction stalls: it lies in the part just deflated."""
    x, y = moving(rng)
    assert fit(x, y, 8).max_rank == 8


def test_the_fit_is_deterministic(rng=np.random.default_rng(1)):
    """No random number enters it, so two fits of one input agree bit for bit."""
    x, y = moving(rng)
    a, b = fit(x, y, 8), fit(x, y, 8)
    assert np.array_equal(a.weights, b.weights)
    assert a.iterations == b.iterations


def test_more_rank_transports_further(rng=np.random.default_rng(2)):
    """The displacement here is rank 6, so the gate should climb with rank and then hold."""
    x, y = moving(rng)
    fitted = fit(x[:200], y[:200], 16)
    scores = []
    for rank in (1, 2, 4, 8):
        moved = fitted.apply(x[200:], rank=rank)
        scores.append(nearer_the_successor(moved, x[200:], y[200:]).mean())
    assert scores[-1] > scores[0], scores


def test_the_intercept_is_the_mean_target(rng=np.random.default_rng(4)):
    """The fit's intercept is the fitting rows' mean successor, so rank zero is the mean predictor."""
    x, y = moving(rng)
    assert np.allclose(fit(x, y, 4).target_mean, y.mean(axis=0), atol=1e-8)


def test_cost_m_applies_the_one_step_rule_m_times(rng=np.random.default_rng(5)):
    x, y = moving(rng)
    fitted = fit(x, y, 4)
    once = fitted.apply(x[:5], rank=4, times=1)
    twice = fitted.apply(x[:5], rank=4, times=2)
    assert np.allclose(twice, fitted.apply(once, rank=4, times=1))
    assert not np.allclose(once, twice)


def test_a_rank_beyond_the_components_refuses(rng=np.random.default_rng(6)):
    x, y = moving(rng)
    fitted = fit(x, y, 4)
    with pytest.raises(ValueError, match="exceeds"):
        fitted.coefficients(8)


def test_a_transition_of_cost_below_one_refuses(rng=np.random.default_rng(7)):
    x, y = moving(rng)
    with pytest.raises(ValueError, match="cost below one"):
        fit(x, y, 4).apply(x[:2], rank=2, times=0)


def test_the_gate_is_a_two_point_comparison_not_a_retrieval():
    transported = np.array([[0.0, 0.0]])
    source = np.array([[1.0, 0.0]])
    successor = np.array([[0.5, 0.0]])
    assert nearer_the_successor(transported, source, successor).tolist() == [True]
    assert nearer_the_successor(source, source, successor).tolist() == [False]
