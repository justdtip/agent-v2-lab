"""E1's estimator: that it can learn, that its nulls are nulls, and that it cannot cheat."""

from __future__ import annotations

import numpy as np
import pytest

from local_llm_lab.pipeline.state_programme.decodability import (
    HEADLINE_FRACTION,
    HEADLINE_RANK,
    LADDER,
    episode_scores,
    decompose,
    fit_at_rank,
    layers_for,
    mean_absolute_error,
    permute_within,
    r_squared,
)


def test_the_depths_are_the_ones_the_document_names():
    # §5 names these explicitly; a change to the fractions must break this test, not pass quietly.
    assert list(layers_for(34).values()) == [6, 11, 17, 23, 28, 34]
    assert list(layers_for(48).values()) == [8, 16, 24, 32, 40, 48]


def test_the_headline_is_fixed_in_advance():
    assert HEADLINE_RANK == 8 and HEADLINE_FRACTION == 0.5
    assert HEADLINE_RANK in LADDER and HEADLINE_FRACTION in layers_for(34)


def synthetic(rng, n=400, d=40, noise=0.0):
    """A target that is a linear function of one direction, so a rank-1 probe should find it."""
    x = rng.normal(size=(n, d))
    y = np.rint(3.0 * x[:, 0] + 5.0 + noise * rng.normal(size=n)).astype(np.int64)
    return x, y


def test_a_probe_that_can_learn_does(rng=np.random.default_rng(0)):
    x, y = synthetic(rng)
    fit = fit_at_rank(decompose(x[:300], y[:300], 32), 8)
    accuracy = (fit.predict(x[300:]) == y[300:]).mean()
    assert accuracy > 0.5, accuracy


def test_permuted_labels_destroy_it(rng=np.random.default_rng(1)):
    x, y = synthetic(rng)
    families = np.array(["a"] * 200 + ["b"] * 200)
    real = fit_at_rank(decompose(x[:300], y[:300], 32), 8)
    permuted = permute_within(y[:300], families[:300], rng)
    null = fit_at_rank(decompose(x[:300], permuted, 32), 8)
    assert (real.predict(x[300:]) == y[300:]).mean() > (null.predict(x[300:]) == y[300:]).mean()


def test_the_permutation_keeps_each_family_composition_exactly(rng=np.random.default_rng(2)):
    labels = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    families = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
    permuted = permute_within(labels, families, rng)
    for family in ("a", "b"):
        rows = families == family
        assert sorted(permuted[rows]) == sorted(labels[rows])


def test_a_probe_cannot_name_a_value_it_was_never_shown(rng=np.random.default_rng(3)):
    x, y = synthetic(rng)
    y = np.clip(y, 2, 8)
    fit = fit_at_rank(decompose(x[:300], y[:300], 32), 16)
    predicted = fit.predict(x[300:] * 50.0)  # far outside the training range
    assert predicted.min() >= y[:300].min() and predicted.max() <= y[:300].max()


def test_the_ladder_is_one_decomposition_sliced_not_six_fits(rng=np.random.default_rng(4)):
    """PLS1 is greedy, so the first r components of a 32-component fit ARE the r-component fit."""
    x, y = synthetic(rng)
    full = decompose(x, y, 32)
    for rank in [r for r in LADDER if r <= full.max_rank]:
        short = decompose(x, y, rank)
        assert np.allclose(full.weights[:, :rank], short.weights, atol=1e-10)
        assert np.allclose(fit_at_rank(full, rank).beta, fit_at_rank(short, rank).beta, atol=1e-8)


def test_a_decomposition_stops_when_nothing_is_left_to_explain(rng=np.random.default_rng(7)):
    """Features spanning five directions cannot yield thirty-two; the rank is then honestly short."""
    latent = rng.normal(size=(200, 5))
    x = latent @ rng.normal(size=(5, 40))  # forty columns, five directions
    y = np.rint(3.0 * latent[:, 0] + 5.0).astype(np.int64)
    components = decompose(x, y, 32)
    assert components.max_rank <= 5, components.max_rank


def test_a_rank_beyond_the_computed_directions_refuses(rng=np.random.default_rng(5)):
    x, y = synthetic(rng, d=10)
    with pytest.raises(ValueError, match="exceeds"):
        fit_at_rank(decompose(x, y, 8), 16)


def test_one_bounded_observation_per_episode():
    predicted = np.array([1, 1, 2, 3, 3, 3])
    actual = np.array([1, 2, 2, 3, 3, 4])
    episodes = np.array(["e1", "e1", "e1", "e2", "e2", "e2"])
    names, scores = episode_scores(predicted, actual, episodes)
    assert list(names) == ["e1", "e2"]
    assert scores.tolist() == [2 / 3, 2 / 3]
    assert scores.min() >= 0.0 and scores.max() <= 1.0


def test_the_descriptive_numbers_behave():
    actual = np.array([1, 2, 3, 4])
    assert mean_absolute_error(actual, actual) == 0.0
    assert r_squared(actual, actual) == 1.0
    assert mean_absolute_error(np.array([2, 3, 4, 5]), actual) == 1.0


def test_r_squared_says_nothing_when_there_is_no_variance():
    constant = np.array([3, 3, 3])
    assert np.isnan(r_squared(constant, constant))


def test_the_fit_never_sees_an_evaluation_label(rng=np.random.default_rng(6)):
    """Structural: the fit takes training rows and training targets and nothing else."""
    import inspect

    assert set(inspect.signature(decompose).parameters) == {"train", "targets", "max_rank"}
    assert set(inspect.signature(fit_at_rank).parameters) == {"components", "rank"}
