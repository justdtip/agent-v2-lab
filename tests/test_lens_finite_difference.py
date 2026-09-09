"""The finite-difference operand, checked against the estimator it is meant to differ from.

The whole workstream's number is the difference between two estimators, so the one thing this file
must establish is that the difference is **truncation and not a bug**. It does that three ways: the
residual against upstream's exact fit is small, the transposed orientation is worse by more than two
orders of magnitude, and halving the step divides the residual by four, which is what a central
difference does and what a wrong derivative does not.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from test_lens_upstream import NUM_LAYERS, SEQ_LEN, TinyLensModel, make_rows

from local_llm_lab.pipeline.lens_fitting import golden
from local_llm_lab.pipeline.lens_fitting import upstream as adapter
from local_llm_lab.pipeline.lens_fitting.finite_difference import (
    EPSILON_RULE,
    fit_finite_difference_jacobian,
)

D_MODEL = 6


@pytest.fixture(scope="module")
def upstream():
    try:
        return adapter.load_upstream()
    except adapter.UpstreamUnavailable as error:  # pragma: no cover - box without the clone
        pytest.skip(str(error))


@pytest.fixture(scope="module")
def exact(upstream):
    return adapter.fit_upstream_jacobian(
        TinyLensModel(), make_rows(), dim_batch=3, max_seq_len=SEQ_LEN, upstream=upstream
    )


def _fd(upstream, **kwargs):
    return fit_finite_difference_jacobian(
        TinyLensModel(),
        kwargs.pop("rows", None) or make_rows(),
        max_seq_len=SEQ_LEN,
        upstream=upstream,
        **{"direction_batch": D_MODEL, **kwargs},
    )


def _worst(exact_fit, other) -> float:
    return max(
        float(
            np.linalg.norm(np.asarray(other[layer], np.float64) - np.asarray(a := np.asarray(
                exact_fit.jacobians[layer], np.float64), np.float64))
            / np.linalg.norm(a)
        )
        for layer in exact_fit.jacobians
    )


# --------------------------------------------------------- the difference is truncation, not a bug


def test_the_finite_difference_fit_reproduces_the_exact_one_to_the_step_squared(exact, upstream):
    fit = _fd(upstream)

    assert sorted(fit.jacobians) == sorted(exact.jacobians)
    assert fit.target_layer == exact.target_layer and fit.n_prompts == exact.n_prompts
    assert _worst(exact, fit.jacobians) < 1e-2


def test_the_orientation_is_pinned_by_being_two_orders_worse_when_transposed(exact, upstream):
    """The map is square, so no shape, dtype or identity check can see a transpose. This can."""
    fit = _fd(upstream)
    upright = _worst(exact, fit.jacobians)
    transposed = _worst(exact, {layer: m.T for layer, m in fit.jacobians.items()})

    assert transposed > upright * 100


def test_halving_the_step_divides_the_residual_by_four(exact, upstream):
    """A central difference has error O(eps^2). A derivative computed wrongly does not improve on
    a schedule, so this is the check that separates truncation from a defect."""
    coarse = _worst(exact, _fd(upstream, epsilon_scale=0.01).jacobians)
    fine = _worst(exact, _fd(upstream, epsilon_scale=0.005).jacobians)

    assert 3.5 < coarse / fine < 4.5


def test_the_direction_batch_is_numerically_inert(exact, upstream) -> None:
    """It is a memory knob. If it moved a number, every fit would depend on the box it ran on."""
    one = _fd(upstream, direction_batch=1)
    several = _fd(upstream, direction_batch=4)

    for layer in one.jacobians:
        np.testing.assert_array_equal(one.jacobians[layer], several.jacobians[layer])


# ------------------------------------------------------------------------ the positions are tied


def test_the_target_sum_takes_the_mask_and_not_every_position(exact, upstream) -> None:
    """Upstream's default mask drops the last position, and both reductions take the same mask.

    Summing the target over every position instead would be a different estimator that no ν field
    records, so the two fits would still declare themselves comparable.
    """
    everything = _fd(upstream, position_selector=lambda n: torch.ones(n, dtype=torch.bool))
    default = _fd(upstream)

    assert _worst(exact, everything.jacobians) > _worst(exact, default.jacobians) * 10


def test_a_selector_that_chooses_nothing_is_counted_as_a_skip_not_a_smaller_fit(upstream) -> None:
    with pytest.raises(ValueError, match="contributed nothing"):
        _fd(upstream, position_selector=lambda n: torch.zeros(n, dtype=torch.bool))


def test_a_declared_precision_that_is_not_the_observed_one_is_refused(upstream) -> None:
    with pytest.raises(ValueError, match="must be a measurement"):
        _fd(upstream, dtype="bfloat16")


# ------------------------------------------------------------------------------ what it declares


def test_the_result_carries_the_step_the_residual_is_a_function_of(upstream) -> None:
    fit = _fd(upstream)
    provenance = fit.provenance

    assert provenance["estimator"] == adapter.ESTIMATOR_FINITE_DIFFERENCE
    assert provenance["epsilon_rule"] == EPSILON_RULE
    assert provenance["epsilon_scale"] == 0.01
    assert provenance["difference"] == "central"
    assert provenance["direction_basis"].startswith("full")
    # Measured per layer, not one number for the model: the step scales with each layer's norm.
    per_layer = provenance["epsilon_per_layer"]
    assert sorted(per_layer) == [str(layer) for layer in sorted(fit.jacobians)]
    assert all(0.0 < entry["min"] <= entry["mean"] <= entry["max"] for entry in per_layer.values())
    # The estimator says it exists to be retired, in the artefact rather than only in a plan.
    assert "once" in provenance["retired_when"]
    # No graph is built, and the precision block says what its accumulation dtype describes.
    assert fit.precision["backward_accumulation_dtype"] == "float64"
    assert "no autograd graph" in fit.precision["note"]


# ---------------------------------------------------------------- the golden comparison, for real


def test_the_two_estimators_differ_in_exactly_one_nu_field(exact, upstream) -> None:
    """The condition the harness enforces, met by construction rather than by discipline: both
    fits are the same result type over the same rows on the same model."""
    fd = _fd(upstream)
    corpus = {"synthetic": True}
    exact_nu = adapter.declare_nu(exact, num_layers=NUM_LAYERS, corpus=corpus)
    fd_nu = adapter.declare_nu(
        fd, num_layers=NUM_LAYERS, corpus=corpus, estimator=adapter.ESTIMATOR_FINITE_DIFFERENCE
    )

    golden.assert_estimator_is_the_only_difference(exact_nu, fd_nu)

    # Everything that decides the fitted quantity agrees, key by key.
    for field, keys in golden.COMPARABLE_KEYS.items():
        for key in keys:
            assert exact_nu[field].get(key) == fd_nu[field].get(key), f"{field}.{key}"
    assert exact_nu["endpoint"] == fd_nu["endpoint"] and exact_nu["corpus"] == fd_nu["corpus"]
    # And the blocks are *not* equal whole, which is the reason the comparison is key by key: each
    # estimator records its own knob there, and comparing the blocks would be a gate that no two
    # correct fits of one corpus could ever pass.
    assert exact_nu["position_weighting"] != fd_nu["position_weighting"]
    assert exact_nu["position_weighting"]["dim_batch"] == 3
    assert fd_nu["position_weighting"]["direction_batch"] == D_MODEL


def test_the_golden_report_runs_end_to_end_with_a_real_second_operand(exact, upstream) -> None:
    """The whole thing at fixture scale: exact against finite difference, the controls gating it.

    This is the run the device repeats with two arguments changed — a real model and real rows.
    """
    fd = _fd(upstream)
    corpus = {"synthetic": True}
    reference = {
        adapter.repo_layer_of_upstream(i): np.asarray(m, np.float32)
        for i, m in exact.jacobians.items()
    }
    candidate = {
        adapter.repo_layer_of_upstream(i): np.asarray(m, np.float32)
        for i, m in fd.jacobians.items()
    }
    report = golden.golden_report(
        reference=reference,
        candidate=candidate,
        reference_nu=adapter.declare_nu(exact, num_layers=NUM_LAYERS, corpus=corpus),
        candidate_nu=adapter.declare_nu(
            fd, num_layers=NUM_LAYERS, corpus=corpus,
            estimator=adapter.ESTIMATOR_FINITE_DIFFERENCE,
        ),
        storage_dtypes=["float32"],
        finite_difference_epsilon=fd.provenance["epsilon_per_layer"],
        reproduction=reference,
        controls={
            "transposed": golden.transposed_control(reference),
            "layer_shifted": golden.layer_shifted_control(reference),
        },
        fixture_residual=None,
    )
    finding = report["finding"]["worst_relative_difference"]

    assert report["gates"]["exact_reproduces_itself"]["measured"] == 0.0
    assert all(row["separates"] for row in report["gates"]["controls_separate"].values())
    assert report["finding"]["threshold"] is None
    # The fixture's own residual, which is the magnitude the device's reading is compared against.
    # Above the float16 storage floor, so at this scale the estimator difference is resolvable
    # rather than hidden under the precision the hosted lenses are stored at.
    assert 1e-3 < finding < 1e-2
    assert finding > golden.storage_floor("float16")
    assert report["finding"]["at_or_below_storage_floor"] is False
