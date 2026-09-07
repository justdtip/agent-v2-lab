import json
from types import SimpleNamespace

import numpy as np
import pytest

from local_llm_lab.pipeline.lens_fitting.epsilon_diagnostic import (
    BlockLedger,
    DeadlineExceeded,
    compare,
    run_diagnostic,
)
from local_llm_lab.pipeline.lens_fitting.jacobian import digest


def test_denominator_is_larger_scale_reference():
    # .003 + .03*100=3.003; using 97 instead incorrectly rejects error=3.
    result = compare([[100.0]], [[97.0]], atol=0.003, rtol=0.03)
    assert result["failed_coordinates"] == 0
    assert result["total_coordinates"] == 1
    assert result["max_error"] == 3
    assert result["rms_error"] == 3
    assert compare([[97.0]], [[100.0]], atol=0.003, rtol=0.03)["failed_coordinates"] == 1


def test_nonfinite_is_failure_and_json_null():
    result = compare([[np.nan, 1], [2, np.inf]], [[0, 1], [3, np.inf]], atol=0, rtol=0)
    assert result["failed_coordinates"] == 3
    assert result["nonfinite_coordinates"] == 2
    assert result["total_coordinates"] == 4
    assert result["max_error"] is None
    assert result["per_direction"][0]["rms_error"] is None
    json.dumps(result, allow_nan=False)


class View:
    hidden_size = 3

    def run_block(self, index, h, *args):
        return h


def test_ledger_deadline_counts_and_materialization():
    now = [0.0]
    events = []
    view = BlockLedger(View(), events.append, started=0, clock=lambda: now[0])
    view.phase = {"operation": "broadcast"}
    view.run_block(16, np.zeros((8, 1, 3)))
    view.run_block(17, np.zeros((1, 439, 3)))
    assert view.summary()["scheduled_block_tokens"] == 447
    assert view.summary()["materialized_block_tokens"] == 0
    view.confirm("after_eval")
    assert view.summary()["materialized_block_tokens"] == 447
    assert view.summary()["pending_block_calls"] == 0
    previous = "0" * 64
    for event in events:
        assert event["previous_sha256"] == previous
        previous = digest({k: v for k, v in event.items() if k != "sha256"})
        assert previous == event["sha256"]
    now[0] = 480
    with pytest.raises(DeadlineExceeded):
        view.run_block(18, np.zeros((8, 1, 3)))
    assert view.attempts == 2


def inputs():
    rows = [dict(ids=list(range(439)))]
    plan = dict(
        self_row=0,
        self_position=438,
        self_layers=[1, 16, 31],
        seeds=dict(self_directions=20260904),
        rows_sha256=digest(rows),
        hidden_size=3,
        plan_sha256="plan",
        snapshot_sha256="snapshot",
        self_bounds=dict(atol=0.0003, rtol=0.003),
        stability_bounds=dict(atol=0.003, rtol=0.03),
    )
    return rows, plan


def test_exact_schedule_and_independent_cache_checks(tmp_path):
    rows, plan = inputs()
    schedule = []

    def prepare(view, ids, layer, position, guard):
        assert position == 438 and len(ids) == 439
        return SimpleNamespace(view=view, layer=layer, primal_norm=100.0)

    def response(state, directions, *, epsilon_scale, guard, mode="reference", batch_size=None):
        schedule.append((state.layer, epsilon_scale, mode))
        assert directions.shape == (16, 3)
        if mode != "reference":
            assert batch_size == 8
        state.view.run_block(state.layer, np.zeros((1, 439 if mode == "reference" else 1, 3)))
        guard("fake_after_eval")
        return directions * epsilon_scale  # Deliberately unstable across scales; cache matches.

    result = run_diagnostic(
        View(),
        rows,
        plan,
        tmp_path,
        started=0,
        clock=lambda: 0,
        prepare=prepare,
        reference=response,
        cached=response,
        guard=lambda *a, **k: None,
        provenance={},
    )
    expected_layer = [
        (s, m)
        for s, modes in [
            (0.125, ["reference", "restore", "broadcast"]),
            (0.25, ["reference"]),
            (0.5, ["reference"]),
            (1.0, ["reference", "restore", "broadcast"]),
        ]
        for m in modes
    ]
    assert schedule == [(layer, s, m) for layer in [1, 16, 31] for s, m in expected_layer]
    assert result["status"] == "completed"
    assert len(result["measurements"]) == 24
    assert len(result["comparisons"]) == 21
    assert all(
        c["failed_coordinates"] == 0
        for c in result["comparisons"]
        if c["kind"] == "cache_reference"
    )
    assert any(
        c["failed_coordinates"] > 0
        for c in result["comparisons"]
        if c["kind"] == "adjacent_stability"
    )
    assert result["block_accounting"]["materialized_block_tokens"] == 3 * (4 * 439 + 4)
    assert result["block_accounting"]["pending_block_calls"] == 0
    assert len(result["measurements"][0]["epsilon"]) == 16
    np.testing.assert_allclose(result["measurements"][0]["epsilon"], 0.125, atol=1e-7)
    assert len(list(tmp_path.glob("layer-*.npy"))) == 24
    assert not list(tmp_path.glob("*.npz"))


def test_timeout_preserves_completed_scale_and_partial_ledger(tmp_path):
    rows, plan = inputs()
    now = [0.0]

    def prepare(view, *args, **kwargs):
        return SimpleNamespace(view=view, layer=1, primal_norm=1.0)

    def reference(state, directions, **kwargs):
        now[0] = 480
        return directions

    result = run_diagnostic(
        View(),
        rows,
        plan,
        tmp_path,
        started=0,
        clock=lambda: now[0],
        prepare=prepare,
        reference=reference,
        guard=lambda *a, **k: None,
        provenance={},
    )
    assert result["status"] == "incomplete"
    assert result["error_type"] == "DeadlineExceeded"
    assert len(result["measurements"]) == 1
    assert json.loads((tmp_path / "diagnostic.json").read_text()) == result


def test_failed_native_call_is_not_materialized():
    class Broken(View):
        def run_block(self, *args):
            raise RuntimeError("native failure")

    events = []
    view = BlockLedger(Broken(), events.append, started=0, clock=lambda: 0)
    with pytest.raises(RuntimeError, match="native failure"):
        view.run_block(1, np.zeros((8, 1, 3)))
    assert view.summary()["attempted_block_calls"] == 1
    assert view.summary()["scheduled_block_tokens"] == 0
    assert view.summary()["materialized_block_tokens"] == 0
    assert events[-1]["event"] == "block_failed"


def test_memory_stop_is_partial_with_durable_reason(tmp_path):
    rows, plan = inputs()

    def prepare(*args, **kwargs):
        raise ValueError("measured peak exceeds bound")

    result = run_diagnostic(
        View(),
        rows,
        plan,
        tmp_path,
        started=0,
        clock=lambda: 0,
        prepare=prepare,
        guard=lambda *a, **k: None,
        provenance={},
    )
    assert result["status"] == "incomplete"
    assert result["measurements"] == []
    assert result["error"] == "measured peak exceeds bound"
    assert json.loads((tmp_path / "diagnostic.json").read_text()) == result
