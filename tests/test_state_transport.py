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


ORTHOGONAL_ROWS = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])


def targets_for_cross(cross):
    """Successors such that XᵀY equals `cross` exactly for the orthogonal rows (XᵀX = 4I)."""
    return ORTHOGONAL_ROWS @ (np.asarray(cross, dtype=float) / 4.0)


def test_the_restart_witness_yields_two_components(codex_witness=(2, -1)):
    """Codex F2's discriminating fixture: successors (2x0+x1, 2x0-x1, 0), rank 2 requested. The old
    fitter, restarting from the previous component's direction, returned rank 1; the fixed one two."""
    x = ORTHOGONAL_ROWS
    y = np.stack([2 * x[:, 0] + x[:, 1], 2 * x[:, 0] - x[:, 1], np.zeros(4)], axis=1)
    fitted = fit(x, y, 2)
    assert fitted.max_rank == 2
    assert np.allclose(fitted.apply(x, rank=2), y, atol=1e-8)


def test_a_cross_product_with_zero_row_sums_and_zero_first_column_stays_finite():
    """Codex F1's zero-start witness: the old start `cross @ 1` was zero and its fallback, the first
    column, zero too, so the old fitter divided zero by zero and kept NaNs while reporting rank 1."""
    cross = np.array([[0.0, 1.0, -1.0], [0.0, -1.0, 1.0], [0.0, 0.0, 0.0]])
    x, y = ORTHOGONAL_ROWS, targets_for_cross(cross)
    fitted = fit(x, y, 2)
    assert np.isfinite(fitted.weights).all() and np.isfinite(fitted.coefficients(fitted.max_rank)).all()
    reference = np.linalg.svd(cross, full_matrices=False)[0][:, 0]
    assert abs(float(fitted.weights[:, 0] @ reference)) == pytest.approx(1.0, abs=1e-9)


def test_the_leading_direction_is_the_leading_one_not_a_fixed_point():
    """Codex F1's non-leading witness: XᵀY = [[8,-8,0],[0,0,4],[0,0,0]]; the leading left singular
    direction is (1,0,0) with singular value sqrt(128); the old start (0,4,0) converged to (0,1,0),
    singular value 4, alignment exactly zero, and was certified. The SVD form returns the leading one."""
    cross = np.array([[8.0, -8.0, 0.0], [0.0, 0.0, 4.0], [0.0, 0.0, 0.0]])
    x, y = ORTHOGONAL_ROWS, targets_for_cross(cross)
    fitted = fit(x, y, 1)
    assert abs(float(fitted.weights[:, 0] @ np.array([1.0, 0.0, 0.0]))) == pytest.approx(1.0, abs=1e-9)
    assert fitted.leading_singular_values[0] == pytest.approx(np.sqrt(128.0), abs=1e-9)


def test_thirty_two_components_exist_and_the_ladder_is_nested_to_the_top(rng=np.random.default_rng(8)):
    """§10's claim: a 32-component fit reaches 32, and its first r components are the r-component fit."""
    x, y = moving(rng)
    full = fit(x, y, 32)
    assert full.max_rank == 32
    for rank in (1, 2, 4, 8, 16):
        assert np.allclose(full.coefficients(rank), fit(x, y, rank).coefficients(rank), atol=1e-6)


def test_the_full_rank_rule_is_least_squares(rng=np.random.default_rng(9)):
    """Known answer: with as many components as predictors (and targets at least as wide, since the
    component count is bounded by the rank of XᵀY) the PLS rule is the least-squares map, so its
    prediction equals the projection of Y onto the column space of the centred X. The old fitter
    stopped at 11 of 12 components on this fixture."""
    x = rng.normal(size=(120, 12))
    y = x @ rng.normal(size=(12, 12)) + 0.1 * rng.normal(size=(120, 12))
    fitted = fit(x, y, 12)
    assert fitted.max_rank == 12
    xc = x - x.mean(axis=0)
    ols = xc @ np.linalg.lstsq(xc, y - y.mean(axis=0), rcond=None)[0] + y.mean(axis=0)
    assert np.max(np.abs(fitted.apply(x, rank=12) - ols)) < 1e-6


def test_targets_narrower_than_sources_are_held(rng=np.random.default_rng(10)):
    """The target loadings are allocated at the target width; the old allocation used the source
    width and could not hold a narrower target (a broadcast error at the first component)."""
    x = rng.normal(size=(80, 12))
    y = x @ rng.normal(size=(12, 6))
    fitted = fit(x, y, 4)
    assert fitted.max_rank == 4
    assert fitted.target_loadings.shape == (6, 4)
    assert np.isfinite(fitted.coefficients(4)).all()
    assert np.max(np.abs(fitted.apply(x, rank=4) - y)) < np.max(np.abs(y))


def test_a_non_finite_input_is_refused_not_carried():
    x, y = ORTHOGONAL_ROWS.copy(), targets_for_cross(np.eye(3))
    y[0, 0] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        fit(x, y, 1)


def test_the_fit_is_deterministic(rng=np.random.default_rng(1)):
    """No random number enters it, so two fits of one input agree bit for bit."""
    x, y = moving(rng)
    a, b = fit(x, y, 8), fit(x, y, 8)
    assert np.array_equal(a.weights, b.weights)
    assert a.leading_singular_values == b.leading_singular_values


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
