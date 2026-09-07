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
