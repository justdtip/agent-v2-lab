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


def _matched_pair(upstream, *, width=D_MODEL, anchor=None, exact_width=None):
    """An exact fit and a finite-difference fit that share their batch schedule.

    Matching is not automatic and is the point of this helper. The exact path forwards the prompt
    replicated `dim_batch` times and differentiates *that* forward; the finite-difference path
    forwards at `direction_batch` and anchors on a capture taken at `anchor_batch`. The bf16 forward
    is not batch-invariant, so a pair that disagrees on either width is a pair of estimates of two
    different functions, and the gate refuses it. Every argument here is explicit for that reason.
    """
    corpus = {"synthetic": True}
    exact_fit = adapter.fit_upstream_jacobian(
        TinyLensModel(), make_rows(), dim_batch=exact_width or width,
        max_seq_len=SEQ_LEN, upstream=upstream,
    )
    fd_fit = _fd(upstream, direction_batch=width, anchor_batch=width if anchor is None else anchor)
    return (
        adapter.declare_nu(exact_fit, num_layers=NUM_LAYERS, corpus=corpus),
        adapter.declare_nu(fd_fit, num_layers=NUM_LAYERS, corpus=corpus,
                           estimator=adapter.ESTIMATOR_FINITE_DIFFERENCE),
    )


def test_a_pair_whose_batch_schedule_differs_is_refused(upstream) -> None:
    """The gate's newest field, and the one the golden run needed and did not have.

    The bf16 forward is not batch-invariant: measured on the card with no hook present, not one of
    Gemma 3 4B's 34 blocks is bitwise identical between batch 1 and batch 64, and the divergence
    compounds from the first block to a mean absolute difference of 76.4 at the target
    (`research/records/WSD-FD-CALIBRATION-2026-09-10`). So the batch width is part of the arithmetic
    path, and two fits at two widths estimate two functions. The golden run compared an exact fit at
    `dim_batch=64` with a finite-difference fit at `direction_batch=256` anchored at width 1, and the
    gate passed, because it was never given the field. It is given it now.
    """
    exact_nu, fd_nu = _matched_pair(upstream, exact_width=D_MODEL - 3)
    with pytest.raises(golden.NotComparable, match="sum of causes"):
        golden.assert_estimator_is_the_only_difference(exact_nu, fd_nu)

    # The anchor is a second, independent width, and a mismatch there is refused on its own.
    exact_nu, fd_nu = _matched_pair(upstream, anchor=1)
    assert fd_nu["precision"]["forward_batch"] == exact_nu["precision"]["forward_batch"]
    assert fd_nu["precision"]["anchor_batch"] != exact_nu["precision"]["anchor_batch"]
    with pytest.raises(golden.NotComparable, match="sum of causes"):
        golden.assert_estimator_is_the_only_difference(exact_nu, fd_nu)


def test_a_fit_that_declares_no_batch_schedule_is_refused_by_name(upstream) -> None:
    """Absent must not compare equal to absent.

    `_compared` drops a key missing from a block, so two fits that both say nothing about their
    batch schedule would agree about it. That is the failure this whole gate exists to prevent, so
    the requirement is presence and not merely equality, and the message says which fit is silent.
    """
    exact_nu, fd_nu = _matched_pair(upstream)
    golden.assert_estimator_is_the_only_difference(exact_nu, fd_nu)  # the pair is good as it stands

    for silenced in ("reference", "candidate"):
        a, b = dict(exact_nu), dict(fd_nu)
        target = a if silenced == "reference" else b
        target["precision"] = {k: v for k, v in target["precision"].items()
                               if k != "forward_batch"}
        with pytest.raises(golden.NotComparable, match=f"the {silenced} fit declares no"):
            golden.assert_estimator_is_the_only_difference(a, b)

    # Both silent at once still refuses, which is the case a presence-blind gate would have passed.
    a = {**exact_nu, "precision": {k: v for k, v in exact_nu["precision"].items()
                                   if k not in ("forward_batch", "anchor_batch")}}
    b = {**fd_nu, "precision": {k: v for k, v in fd_nu["precision"].items()
                                if k not in ("forward_batch", "anchor_batch")}}
    with pytest.raises(golden.NotComparable, match="declares no"):
        golden.assert_estimator_is_the_only_difference(a, b)


def test_the_two_estimators_differ_in_exactly_one_nu_field(exact, upstream) -> None:
    """The condition the harness enforces, met by construction rather than by discipline: both
    fits are the same result type over the same rows on the same model.

    The pair is now built at a **matched** batch schedule. Before 2026-09-09 this test passed on a
    pair whose widths were 3 and 6, because the gate could not see them.
    """
    exact_nu, fd_nu = _matched_pair(upstream)

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
    assert exact_nu["position_weighting"]["dim_batch"] == D_MODEL
    assert fd_nu["position_weighting"]["direction_batch"] == D_MODEL
    # The knob appears twice on purpose and means two different things. In `position_weighting` it
    # is each estimator's own batching detail and is not compared, because it does not change which
    # positions were selected. In `precision` it is the width of the forward being differentiated,
    # and it *is* compared, because the forward is not batch-invariant. Narrowing the first was
    # right; concluding from that narrowing that the width did not matter anywhere was not.
    assert exact_nu["precision"]["forward_batch"] == fd_nu["precision"]["forward_batch"] == D_MODEL
    assert exact_nu["precision"]["anchor_batch"] == fd_nu["precision"]["anchor_batch"] == D_MODEL


def test_the_golden_report_runs_end_to_end_with_a_real_second_operand(upstream) -> None:
    """The whole thing at fixture scale: exact against finite difference, the controls gating it.

    This is the run the device repeats with two arguments changed — a real model and real rows. Both
    operands are fitted at the same batch schedule, which the device run did not do and could not
    have known to do: see `research/records/WSD-FD-CALIBRATION-2026-09-10`.
    """
    exact = adapter.fit_upstream_jacobian(
        TinyLensModel(), make_rows(), dim_batch=D_MODEL, max_seq_len=SEQ_LEN, upstream=upstream
    )
    fd = _fd(upstream, direction_batch=D_MODEL, anchor_batch=D_MODEL)
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


# ------------------------------------------------------- the arithmetic path, declared and gated


def _bf16_model():
    model = TinyLensModel().to(torch.bfloat16)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval()


def test_the_perturbation_keeps_the_block_dtype_under_native_and_promotes_under_the_other() -> None:
    """The claim itself, model-independent: which arithmetic the blocks above the source run.

    Two fits of one checkpoint can declare the same weight dtype and not share this, which is why
    the golden gate refuses a cross-path comparison. The 69.4% figure that motivated the gate is
    `residual_precision_probe`'s `promoted_fp32_loop` — whole-block float32 through
    `arch_torch.run_block` and `_call_promoted` — and **not** this option, which writes float32 into
    a block that is not, and which a real bf16 Gemma refuses inside its own matmul. The fit now says
    that by name; this test holds the property the gate rests on, which is that the two paths hand
    the block different dtypes.
    """
    from local_llm_lab.pipeline.lens_fitting.finite_difference import _perturbation

    base_bf16 = torch.zeros(1, 4, 3, dtype=torch.bfloat16)
    columns = torch.eye(3)[:2]
    native = _perturbation(base_bf16, 1, 0.5, columns)(None, None, torch.zeros(2, 4, 3))
    promoted = _perturbation(base_bf16.float(), 1, 0.5, columns)(None, None, torch.zeros(2, 4, 3))

    assert native.dtype is torch.bfloat16, "native hands the block its own dtype back"
    assert promoted.dtype is torch.float32, "the promoted path makes every block above run float32"
    assert native.shape == promoted.shape == (2, 4, 3)


def test_a_bf16_fit_runs_native_and_says_so(upstream) -> None:
    fit = fit_finite_difference_jacobian(
        _bf16_model(), make_rows(), max_seq_len=SEQ_LEN, direction_batch=D_MODEL,
        dtype="bfloat16", upstream=upstream,
    )

    assert fit.precision["dtype"] == "bfloat16"
    assert fit.precision["capture_dtype"] == "native"
    assert fit.provenance["capture_dtype"] == "native"
    assert all(np.isfinite(m).all() for m in fit.jacobians.values())


def test_an_unknown_capture_path_is_refused_by_name(upstream) -> None:
    with pytest.raises(ValueError, match="capture_dtype must be one of"):
        _fd(upstream, capture_dtype="float64")


def test_the_golden_gate_refuses_two_fits_that_ran_different_arithmetic(upstream) -> None:
    """The gate that makes the finding a measurement of the estimator and not of the path.

    Both operands are fitted at the exact fit's batch schedule, so the only thing left for the gate
    to catch is the arithmetic path. Isolating one difference at a time is the whole discipline.
    """
    corpus = {"synthetic": True}
    matched = dict(direction_batch=D_MODEL, anchor_batch=D_MODEL)
    exact_fit = adapter.fit_upstream_jacobian(
        TinyLensModel(), make_rows(), dim_batch=D_MODEL, max_seq_len=SEQ_LEN, upstream=upstream
    )
    native = _fd(upstream, capture_dtype="native", **matched)
    promoted = _fd(upstream, capture_dtype="promoted-float32", **matched)
    exact_nu = adapter.declare_nu(exact_fit, num_layers=NUM_LAYERS, corpus=corpus)
    kw = dict(num_layers=NUM_LAYERS, corpus=corpus,
              estimator=adapter.ESTIMATOR_FINITE_DIFFERENCE)

    # Native against the exact estimator: the estimator is the only difference, so it compares.
    golden.assert_estimator_is_the_only_difference(exact_nu, adapter.declare_nu(native, **kw))
    # Promoted against it: two causes, and the gate says which field carries the second one.
    with pytest.raises(golden.NotComparable, match="precision"):
        golden.assert_estimator_is_the_only_difference(exact_nu, adapter.declare_nu(promoted, **kw))
    assert "capture_dtype" in golden.COMPARABLE_KEYS["precision"]


# ------------------------------------------------ combining fits, so a declared subset extends


def _fd_rows(upstream, rows, **kwargs):
    return fit_finite_difference_jacobian(
        TinyLensModel(), rows, max_seq_len=SEQ_LEN, upstream=upstream,
        **{"direction_batch": D_MODEL, **kwargs},
    )


def test_three_rows_in_one_call_and_in_three_calls_are_the_same_fit(upstream) -> None:
    """The property the Chief's extension rule rests on: the estimator means over prompts, so a
    subset met by extension is the subset met in one run."""
    from local_llm_lab.pipeline.lens_fitting.finite_difference import combine_fits

    rows = make_rows(count=3)
    together = _fd_rows(upstream, rows)
    apart = combine_fits([_fd_rows(upstream, [row]) for row in rows])

    assert apart.n_prompts == together.n_prompts == 3
    assert [r["index"] for r in apart.per_prompt] == [r["index"] for r in together.per_prompt]
    for layer in together.jacobians:
        np.testing.assert_allclose(apart.jacobians[layer], together.jacobians[layer],
                                   rtol=1e-6, atol=1e-7)


def test_the_weighting_is_by_prompt_count_and_not_by_fit(upstream) -> None:
    from local_llm_lab.pipeline.lens_fitting.finite_difference import combine_fits

    rows = make_rows(count=3)
    two = _fd_rows(upstream, rows[:2])
    one = _fd_rows(upstream, rows[2:])
    combined = combine_fits([two, one])

    assert combined.n_prompts == 3
    assert combined.provenance["combined_from"] == [
        {"n_prompts": 2, "rows": [0, 1], "elapsed_s": two.elapsed_s},
        {"n_prompts": 1, "rows": [2], "elapsed_s": one.elapsed_s},
    ]
    expected = {
        layer: (two.jacobians[layer].astype(np.float64) * 2
                + one.jacobians[layer].astype(np.float64)) / 3
        for layer in two.jacobians
    }
    for layer, value in expected.items():
        np.testing.assert_allclose(combined.jacobians[layer], value.astype(np.float32),
                                   rtol=1e-6, atol=1e-7)


def test_fits_that_are_not_the_same_measurement_are_refused_by_the_field_that_differs(upstream):
    """The golden gate's rule one level down: a mean over two different quantities is neither."""
    from local_llm_lab.pipeline.lens_fitting.finite_difference import combine_fits

    rows = make_rows(count=2)
    base = _fd_rows(upstream, rows[:1])
    with pytest.raises(ValueError, match="epsilon_scale"):
        combine_fits([base, _fd_rows(upstream, rows[1:], epsilon_scale=0.02)])
    with pytest.raises(ValueError, match="source layers"):
        combine_fits([base, _fd_rows(upstream, rows[1:], source_layers=[0])])
    with pytest.raises(ValueError, match="precision"):
        combine_fits([base, _fd_rows(upstream, rows[1:], capture_dtype="promoted-float32")])


def test_a_repeated_row_is_refused_because_a_repeat_is_not_more_data(upstream) -> None:
    from local_llm_lab.pipeline.lens_fitting.finite_difference import combine_fits

    one = _fd_rows(upstream, make_rows(count=1))
    with pytest.raises(ValueError, match="row 0 appears in fit 0 and fit 1"):
        combine_fits([one, _fd_rows(upstream, make_rows(count=1))])
    assert combine_fits([one]) is one, "one fit combines to itself without copying"


def test_promoting_a_bf16_model_is_refused_by_name_and_points_at_the_probe(upstream) -> None:
    """A named refusal beats `expected mat1 and mat2 to have the same dtype` three frames down.

    Measured on the card: asking this estimator for the promoted path on a bf16 Gemma raises inside
    the block's own matmul, at every step and both layers tried. The refusal now names what the
    option is for, where whole-block promotion actually lives, and why this estimator cannot offer
    it — it hooks the model's own forward rather than running the blocks itself.
    """
    with pytest.raises(ValueError, match="residual_precision_probe"):
        fit_finite_difference_jacobian(
            _bf16_model(), make_rows(), max_seq_len=SEQ_LEN, direction_batch=D_MODEL,
            dtype="bfloat16", capture_dtype="promoted-float32", upstream=upstream,
        )
    # On a float32 model the option is what it was built to be: a path to measure against native.
    fit = _fd(upstream, capture_dtype="promoted-float32")
    assert fit.precision["capture_dtype"] == "promoted-float32"


def test_repeating_the_residual_batch_does_not_change_the_per_row_step(upstream) -> None:
    """The step is a property of one row's residual, not of how many copies are in the batch.

    A capture at width w is `[w, seq, hidden]` with every row the same prompt, so a norm taken over
    the whole tensor is √w times the norm over one row and the step silently inherits the factor.
    The calibration ladder did exactly that and ran its width-64 rungs at eight times the fitter's
    step, which shifted its whole axis by three rungs and made "the interval moves with width" a
    reading of the bug. The fitter takes `activations[layer][:1]` before the norm; this pins that,
    because the two paths agreed in intent and differed in arithmetic and nothing compared them.
    """
    fit_one = _fd(upstream, direction_batch=D_MODEL, anchor_batch=1)
    fit_wide = _fd(upstream, direction_batch=D_MODEL, anchor_batch=4)

    per_layer_one = fit_one.provenance["epsilon_per_layer"]
    per_layer_wide = fit_wide.provenance["epsilon_per_layer"]
    assert sorted(per_layer_one) == sorted(per_layer_wide)
    for layer, entry in per_layer_one.items():
        assert entry["mean"] == pytest.approx(per_layer_wide[layer]["mean"], rel=1e-6), (
            f"layer {layer}: the step moved with the anchor width, so it is a function of the "
            "batch and not of the residual"
        )

    # And the direct arithmetic the mistake rests on, so the reason is pinned beside the behaviour.
    row = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    repeated = row.expand(4, -1, -1)
    assert torch.linalg.vector_norm(repeated) == pytest.approx(
        2.0 * torch.linalg.vector_norm(row), rel=1e-6
    )
    assert torch.linalg.vector_norm(repeated[:1]) == pytest.approx(
        torch.linalg.vector_norm(row), rel=1e-6
    )


def test_the_progress_callback_is_exercised_and_emits_a_layer_event_per_layer(upstream) -> None:
    """The per-layer emitter, actually called. Nothing called it until now.

    This callback exists because a fit that reports only at the end lost twenty-seven paid minutes
    when its logger raised after the work was done. It was added to fix that, and then no test ever
    passed a callback through it — the same shape as the defect it was written to prevent, and the
    same shape as the capture pass's `progress`, which referred to a name that did not exist and
    would have died on the device after its first shard.

    A seam nothing calls is a seam nothing checks.
    """
    seen: list[dict] = []
    fit = _fd(upstream, direction_batch=D_MODEL, anchor_batch=D_MODEL, progress=seen.append)

    layer_events = [e for e in seen if e["event"] == "layer"]
    row_events = [e for e in seen if e["event"] == "row"]
    assert layer_events, "no layer event was emitted, so the per-layer reporting is not reporting"
    assert row_events, "no row event was emitted"

    sources = sorted(fit.jacobians)
    for event in layer_events:
        assert event["estimator"] == adapter.ESTIMATOR_FINITE_DIFFERENCE
        assert event["upstream_layer"] in sources
        assert 1 <= event["layers_done"] <= event["layers_total"] == len(sources)
        assert event["epsilon"] > 0
    # Every layer reports, once per row, which is what "per layer, not per row" means. Counted, not
    # compared as a sorted sequence: sorting groups the layers and the repetition is per row.
    from collections import Counter

    assert Counter(e["upstream_layer"] for e in layer_events) == {
        layer: len(row_events) for layer in sources
    }
