"""Requirements §15: position-local steps and two-point plateau selection."""

import numpy as np
import pytest

from local_llm_lab.pipeline.lens_fitting import position_step_diagnostic as diagnostic
from local_llm_lab.pipeline.lens_fitting.jacobian import PositionState, finite_difference_steps


def state(full):
    return PositionState(None, 1, 0, full, full[:, :1], [], 1.0, float(np.linalg.norm(full)))


def test_position_step_ignores_unperturbed_tokens():
    # A return to whole-sequence scaling must fail this suffix-invariance check.
    a = state(np.array([[[3.0, 4.0], [0.0, 0.0]]], dtype=np.float32))
    b = state(np.array([[[3.0, 4.0], [600.0, 800.0]]], dtype=np.float32))
    for source in (a, b):
        corrected = diagnostic.position_local_state(source)
        assert corrected.primal_norm == 5.0
        assert corrected.full_primal is source.full_primal
        assert finite_difference_steps(corrected.primal_norm, [2.0], 0.1)[0] == pytest.approx(
            0.0025
        )
    with pytest.raises(ValueError, match="positive"):
        diagnostic.position_local_state(state(np.zeros((1, 2, 2), dtype=np.float32)))


def points(failures):
    return [
        dict(coefficient=c, failed_coordinates=f)
        for c, f in zip(diagnostic.COEFFICIENTS, failures, strict=True)
    ]


def test_plateau_needs_adjacent_passes_and_chooses_largest():
    assert diagnostic.plateaus(points([0, 1, 0, 1, 0, 1])) == []
    assert diagnostic.plateaus(points([0, 0, 1, 0, 0, 1])) == [3e-4, 1e-2]
    assert diagnostic.plateaus(points([0, 0, 0, 0, 0, 0]))[-1] == 3e-2
    with pytest.raises(ValueError, match="complete"):
        diagnostic.plateaus(points([0, 0, 0, 0, 0, 0])[:-1])


def test_sweep_records_all_36_reference_measurements_and_six_cache_checks(tmp_path):
    import hashlib
    import json
    import time
    from types import SimpleNamespace

    from local_llm_lab.pipeline.lens_fitting.jacobian import digest

    rows = [dict(ids=list(range(439)))]
    plan = dict(
        self_row=0,
        self_position=438,
        self_layers=[1, 16, 31],
        seeds=dict(self_directions=20260904),
        rows_sha256=digest(rows),
        hidden_size=3,
        plan_sha256="p",
        self_bounds=dict(atol=0.0003, rtol=0.003),
        stability_bounds=dict(atol=0.003, rtol=0.03),
    )
    calls = []

    def prepare(view, ids, layer, position, guard):
        full = np.ones((1, 439, 3), dtype=np.float32)
        return PositionState(view, layer, position, full, full[:, -1:], [], 1.0, 36.0)

    def response(state, directions, *, epsilon_scale, guard, mode="reference", batch_size=None):
        calls.append((state.layer, epsilon_scale, mode))
        assert state.primal_norm == pytest.approx(np.sqrt(3))
        state.view.run_block(state.layer, np.zeros((1, 439 if mode == "reference" else 1, 3)))
        guard("materialized")
        return directions.copy()  # Analytic identity tail: every coefficient is stable.

    result = diagnostic.run_sweep(
        SimpleNamespace(run_block=lambda *a: a[1]),
        rows,
        plan,
        tmp_path,
        started=time.monotonic(),
        guard=lambda *a, **k: None,
        provenance={},
        prepare=prepare,
        reference=response,
        cached=response,
    )
    assert result["status"] == "self_check_passed"
    assert result["selected_common_coefficient"] == 0.03
    assert len(result["stability"]) == 18
    assert len(calls) == 48  # 36 grid refs + 6 fresh refs + 6 cache comparisons.
    assert len(result["self_check"]) == 6
    assert result["block_accounting"]["pending_block_calls"] == 0
    assert result["block_accounting"]["materialized_block_tokens"] == 42 * 439 + 6
    names = [m["file"] for m in result["measurements"]]
    assert len(set(names)) == len(names)  # Self-check must never overwrite sweep responses.
    for m in result["measurements"]:
        assert hashlib.sha256((tmp_path / m["file"]).read_bytes()).hexdigest() == m["file_sha256"]
    assert json.loads((tmp_path / "diagnostic.json").read_text())["status"] == result["status"]


def test_production_position_rule_preserves_primals_and_propagates_coefficient():
    from local_llm_lab.pipeline.lens_fitting import jacobian as j

    original = state(np.array([[[3.0, 4.0], [600.0, 800.0]]], dtype=np.float32))
    corrected = j.with_position_step(original, 0.03)
    assert corrected.primal_norm == 5.0
    assert corrected.epsilon == 0.15
    assert corrected.step_coefficient == 0.03
    assert corrected.full_primal is original.full_primal
    assert corrected.primal is original.primal
    assert corrected.prefix_cache is original.prefix_cache
    assert j.finite_difference_steps(
        corrected.primal_norm, [2.0], 0.5, coefficient=corrected.step_coefficient
    )[0] == pytest.approx(0.0375)
    for c in (0.0, float("nan"), -0.01):
        with pytest.raises(ValueError):
            j.with_position_step(original, c)


def test_revised_plan_round_trip_binds_step_without_changing_samples(tmp_path):
    from local_llm_lab.pipeline.lens_fitting import jacobian as j

    rows = [
        dict(
            index=i,
            source=f"source-{i}",
            split=split,
            ids=[1, 2, 3],
            spans=["task", "task", "call"],
        )
        for i, split in enumerate(("fit", "held"))
    ]
    kwargs = dict(
        layers=[1, 2, 3],
        hidden_size=4,
        corpus_sha256="corpus",
        snapshot_sha256="snapshot",
        seed=9,
        self_bounds=dict(atol=0.0003, rtol=0.003),
        response_bounds=dict(atol=0.01, rtol=0.1),
        stability_bounds=dict(atol=0.003, rtol=0.03),
        held_count=150,
        working_set_bytes=10000,
        initial_peak_bytes=1000,
    )
    old = j.make_plan(rows, **kwargs)
    revised = j.make_plan(rows, **kwargs, step_coefficient=0.03)
    assert old["fit_positions"] == revised["fit_positions"]
    assert old["held_positions"] == revised["held_positions"]
    assert old["self_bounds"] == revised["self_bounds"]
    assert j.digest(old) != j.digest(revised)
    path = tmp_path / "plan.json"
    frozen = j.freeze_plan(path, revised)
    assert j.read_plan(path, rows) == frozen
    assert j.position_step_kwargs(frozen) == dict(step_coefficient=0.03)
    assert j.position_step_kwargs(old) == {}
