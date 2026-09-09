"""The golden harness against two synthetic lenses, so the device run is a parameter change.

The operands here are real fits of the tiny synthetic decoder, not arrays typed into the test. The
candidate is the reference stored through `float16` and back, which makes the fixture's residual a
*real* residual of a real kind — the storage floor itself — and so exercises the one reading the
checklist fixes before the numbers exist: a difference at or below the floor is indistinguishable at
storage precision, which is not agreement.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_lens_upstream import NUM_LAYERS, SEQ_LEN, TinyLensModel, make_rows

from local_llm_lab.pipeline.lens_fitting import golden
from local_llm_lab.pipeline.lens_fitting import upstream as adapter

EPSILON = "0.01 * norm(full sequence primal) / norm(one tangent)"


@pytest.fixture(scope="module")
def upstream():
    try:
        return adapter.load_upstream()
    except adapter.UpstreamUnavailable as error:  # pragma: no cover - box without the clone
        pytest.skip(str(error))


def _fit(upstream, rows=None):
    fit = adapter.fit_upstream_jacobian(
        TinyLensModel(), rows or make_rows(), dim_batch=3, max_seq_len=SEQ_LEN, upstream=upstream
    )
    return {
        adapter.repo_layer_of_upstream(index): np.asarray(matrix, dtype=np.float32)
        for index, matrix in fit.jacobians.items()
    }, adapter.declare_nu(fit, num_layers=NUM_LAYERS, corpus={"synthetic": True})


def _through_float16(maps):
    return {layer: matrix.astype(np.float16).astype(np.float32) for layer, matrix in maps.items()}


def _as_finite_difference(nu):
    return {**nu, "estimator": adapter.ESTIMATOR_FINITE_DIFFERENCE}


# ------------------------------------------------------------------------------- the floor


def test_the_floor_is_the_coarsest_operand_not_the_finest() -> None:
    """A comparison is only as resolved as its blunter side; taking the finer one would report a
    floor the data cannot support."""
    assert golden.storage_floor("float16") == pytest.approx(4.8828125e-4)
    assert golden.storage_floor(["float32", "float16"]) == golden.storage_floor("float16")
    assert golden.storage_floor(["float32"]) < golden.storage_floor("float16")


def test_an_undeclared_storage_dtype_is_refused_rather_than_defaulted() -> None:
    with pytest.raises(ValueError, match="rather than guess"):
        golden.storage_floor(["float8_e4m3"])
    with pytest.raises(ValueError, match="not assumable"):
        golden.storage_floor([])


# --------------------------------------------------------------- comparability before comparison


def test_two_fits_differing_in_more_than_the_estimator_are_refused(upstream) -> None:
    """The residual would be a sum of causes, and nothing downstream could take it apart."""
    _, nu = _fit(upstream)
    other_corpus = {**_as_finite_difference(nu), "corpus": {"synthetic": False}}
    with pytest.raises(golden.NotComparable, match="sum of causes"):
        golden.assert_estimator_is_the_only_difference(nu, other_corpus)


def test_comparing_an_estimator_with_itself_is_sent_to_the_exactness_gate(upstream) -> None:
    _, nu = _fit(upstream)
    with pytest.raises(golden.NotComparable, match="exactness gate"):
        golden.assert_estimator_is_the_only_difference(nu, dict(nu))


# ------------------------------------------------------------------------ gate 1: exactness at zero


def test_the_exact_estimator_reproduces_itself_at_exactly_zero(upstream) -> None:
    """Not "within tolerance": a nonzero value here is nondeterminism, and a tolerance would hide
    the one thing this gate exists to see."""
    first, _ = _fit(upstream)
    second, _ = _fit(upstream)
    rows = golden.assert_exact_reproduction(first, second)

    assert [row["relative_difference"] for row in rows] == [0.0] * len(rows)
    assert len(rows) == NUM_LAYERS - 1


def test_a_reproduction_that_is_not_exact_fails_the_gate_by_name(upstream) -> None:
    first, _ = _fit(upstream)
    drifted = {layer: matrix.copy() for layer, matrix in first.items()}
    drifted[2][0, 0] += 1e-6
    with pytest.raises(golden.ControlFailed, match="nondeterminism, not a tolerance"):
        golden.assert_exact_reproduction(first, drifted)


# ------------------------------------------------------------------------------- the whole report


def test_the_report_carries_the_gates_the_finding_and_what_the_finding_rests_on(upstream) -> None:
    exact, nu = _fit(upstream)
    candidate = _through_float16(exact)
    # A different corpus, not a reordering: equal per-prompt weighting makes order inert.
    wrong_corpus, _ = _fit(upstream, rows=make_rows(count=5))

    report = golden.golden_report(
        reference=exact,
        candidate=candidate,
        reference_nu=nu,
        candidate_nu=_as_finite_difference(nu),
        storage_dtypes=["float32", "float16"],
        finite_difference_epsilon=EPSILON,
        reproduction=_fit(upstream)[0],
        controls={
            "transposed": golden.transposed_control(exact),
            "layer_shifted": golden.layer_shifted_control(exact),
            "wrong_corpus": wrong_corpus,
        },
        fixture_residual=None,
    )

    assert report["gates"]["exact_reproduces_itself"] == {
        "expected": 0.0, "measured": 0.0, "state": "passed"
    }
    assert all(row["separates"] for row in report["gates"]["controls_separate"].values())
    assert set(report["gates"]["controls_separate"]) == {
        "transposed", "layer_shifted", "wrong_corpus"
    }
    # The finding has no threshold, and the report says so rather than leaving it absent.
    assert report["finding"]["threshold"] is None
    assert "prediction" in report["finding"]["threshold_note"]
    assert len(report["finding"]["per_layer"]) == NUM_LAYERS - 1
    # Everything the residual is a function of, declared beside it.
    assert report["declared"]["finite_difference_epsilon"] == EPSILON
    assert report["declared"]["storage_floor_relative"] == golden.storage_floor("float16")
    assert report["declared"]["reference_estimator"] == adapter.ESTIMATOR_EXACT_AUTOGRAD
    assert report["declared"]["candidate_estimator"] == adapter.ESTIMATOR_FINITE_DIFFERENCE


def test_a_residual_at_the_floor_is_reported_as_indistinguishable_and_never_as_agreement(upstream):
    """The fixture's own residual is the storage floor, which is exactly the reading the checklist
    fixes in advance: a float16 round trip of an unchanged map lands here."""
    exact, nu = _fit(upstream)
    report = golden.golden_report(
        reference=exact,
        candidate=_through_float16(exact),
        reference_nu=nu,
        candidate_nu=_as_finite_difference(nu),
        storage_dtypes=["float32", "float16"],
        finite_difference_epsilon=EPSILON,
        controls={"transposed": golden.transposed_control(exact)},
    )
    floor = report["declared"]["storage_floor_relative"]

    assert 0.0 < report["finding"]["worst_relative_difference"] <= floor
    assert report["finding"]["at_or_below_storage_floor"] is True
    assert all(row["at_or_below_storage_floor"] for row in report["finding"]["per_layer"])
    assert "not agreement" in report["finding"]["reading_if_at_floor"]
    # And the word does not appear as a verdict anywhere in the report.
    assert "agreement" not in report["gates"]


def test_an_absent_reproduction_reads_as_not_run_rather_than_as_a_pass(upstream) -> None:
    """A missing figure must never read as a passing one."""
    exact, nu = _fit(upstream)
    report = golden.golden_report(
        reference=exact,
        candidate=_through_float16(exact),
        reference_nu=nu,
        candidate_nu=_as_finite_difference(nu),
        storage_dtypes=["float32", "float16"],
        finite_difference_epsilon=EPSILON,
        controls={"transposed": golden.transposed_control(exact)},
    )
    gate = report["gates"]["exact_reproduces_itself"]

    assert gate["measured"] is None and gate["state"] == "not run"


# ----------------------------------------------------------------------------------- the controls


def test_each_control_is_further_from_the_reference_than_a_floor_level_candidate_is(upstream):
    exact, _ = _fit(upstream)
    at_floor = max(
        row["relative_difference"] for row in golden.compare_lenses(exact, _through_float16(exact))
    )
    for control in (golden.transposed_control(exact), golden.layer_shifted_control(exact)):
        worst = max(row["relative_difference"] for row in golden.compare_lenses(exact, control))
        assert worst > at_floor * 100, "a control that barely separates is not a control"


def test_a_control_that_does_not_separate_fails_the_report_rather_than_being_reported(upstream):
    """The gate that makes the finding a measurement.

    A deliberately wrong lens no further from the reference than the candidate means the comparison
    cannot tell the two apart, and then the residual means nothing.
    """
    exact, nu = _fit(upstream)
    with pytest.raises(golden.ControlFailed, match="has not measured the candidate"):
        golden.golden_report(
            reference=exact,
            candidate=golden.transposed_control(exact),
            reference_nu=nu,
            candidate_nu=_as_finite_difference(nu),
            storage_dtypes=["float32"],
            finite_difference_epsilon=EPSILON,
            # The control is the candidate: it cannot be further away than itself.
            controls={"transposed": golden.transposed_control(exact)},
        )


def test_the_layer_shift_is_the_off_by_one_a_conversion_makes(upstream) -> None:
    """Upstream indexes blocks from zero and this repository names layer L the output of block L-1,
    so the shift is the real mistake, and every shape, dtype and identity check passes it."""
    exact, _ = _fit(upstream)
    shifted = golden.layer_shifted_control(exact)

    assert set(shifted) == set(exact)
    assert all(shifted[layer].shape == exact[layer].shape for layer in exact)
    assert not any(np.array_equal(shifted[layer], exact[layer]) for layer in exact)


def test_a_partial_layer_intersection_is_refused_rather_than_compared(upstream) -> None:
    exact, _ = _fit(upstream)
    partial = {layer: matrix for layer, matrix in exact.items() if layer != min(exact)}
    with pytest.raises(ValueError, match="present in one lens only"):
        golden.compare_lenses(exact, partial)
