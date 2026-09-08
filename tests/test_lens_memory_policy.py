"""Pure resource qualification checks: no model, MLX import, or device access."""

import copy
import json
from types import SimpleNamespace

import pytest

from local_llm_lab.pipeline.lens_fitting.memory_policy import (
    binding,
    check_projection,
    device_working_set,
    validate_evidence,
)


def test_threshold_is_device_fraction_not_registry_budget():
    working = 17.76 * 2**30
    assert check_projection(10.656 * 2**30, working) == pytest.approx(10.656 * 2**30)
    for peak in (14.5, 16, 21, 22):
        with pytest.raises(ValueError, match="0.6"):
            check_projection(peak * 2**30, working)


@pytest.mark.parametrize("value", [None, True, 0, -1, float("nan"), float("inf")])
def test_missing_or_invalid_device_fails_closed(value):
    with pytest.raises(ValueError):
        check_projection(1, value)


def test_owner_refusal_precedes_device_query():
    called = []

    def refuse():
        raise ValueError("foreign window")

    with pytest.raises(ValueError, match="foreign"):
        device_working_set(require_window=refuse, device_info=lambda: called.append("device"))
    assert called == []


def test_device_query_follows_owner_check():
    called = []

    def query():
        called.append("device")
        return {"max_recommended_working_set_size": 100}

    assert (
        device_working_set(require_window=lambda: called.append("owner"), device_info=query) == 100
    )
    assert called == ["owner", "device"]


def evidence(tmp_path):
    expected = {
        "workload": [{"input": 1024, "scored": 512, "split": "fit"}],
        "geometry": {"hidden_size": 1, "num_layers": 3},
        "statistics_storage": "memory",
    }
    events = [
        {
            "event": "begin",
            "qualification_binding": expected,
            "initial_bound_bytes": 50,
            "mode": "qualification",
            "lengths": [256, 512, 1024],
        },
        {"event": "loaded", "load_peak_bytes": 20, "resident_bytes": 20, "statistics_bytes": 32},
        *[
            {
                "event": "measured",
                "tokens": n,
                "peak_bytes": 45,
                "projected_peak_bytes": 50 if n < 1024 else 58.5,
                "scored_positions_per_sequence": n // 2,
            }
            for n in (256, 512, 1024)
        ],
        {
            "event": "full_fit_envelope",
            "dense_matrix_bytes": 4,
            "nonfinal_layers": 2,
            "active_floor_bytes": 35,
            "host_maps_and_serialization_bytes": 16,
            "solver_workspace_bytes": 64,
            "projected_peak_bytes": 115,
        },
        {"event": "measured_solve", "peak_bytes": 50, "projected_peak_bytes": 115},
        {"event": "end", "status": "measured", "fit_started": False},
    ]
    path = tmp_path / "preflight.jsonl"

    def write():
        path.write_text("\n".join(json.dumps(e) for e in events))

    write()
    return expected, events, path, write


def test_complete_evidence_qualifies_under_current_device(tmp_path):
    expected, _, path, _ = evidence(tmp_path)
    assert validate_evidence(path, expected, 200)["cap_bytes"] == 120
    with pytest.raises(ValueError, match="0.6"):
        validate_evidence(path, expected, 80)


@pytest.mark.parametrize(
    "mutation",
    [
        "failed",
        "partial",
        "binding",
        "load",
        "projection",
        "dense",
        "long",
        "refusal",
        "diagnostic",
        "reserve",
    ],
)
def test_invalid_evidence_refuses(tmp_path, mutation):
    expected, events, path, write = evidence(tmp_path)
    if mutation == "failed":
        events[-1]["status"] = "error"
    elif mutation == "partial":
        del events[3]
    elif mutation == "binding":
        expected = copy.deepcopy(expected)
        expected["statistics_storage"] = "split_spill"
    elif mutation == "load":
        events[1]["load_peak_bytes"] = 121
    elif mutation == "projection":
        events[-2]["projected_peak_bytes"] = 121
    elif mutation == "dense":
        expected["workload"][0]["scored"] = 513
    elif mutation == "long":
        expected["workload"][0]["input"] = 1025
    elif mutation == "diagnostic":
        events[0]["mode"] = "diagnostic"
    elif mutation == "reserve":
        del events[-3]
    else:
        events.insert(-1, {"event": "stopped"})
    write()
    with pytest.raises(ValueError):
        validate_evidence(path, expected, 200)


def test_binding_changes_with_producer_storage_snapshot_corpus_and_workload(tmp_path, monkeypatch):
    from local_llm_lab.pipeline.lens_fitting import memory_policy

    monkeypatch.setattr(memory_policy, "runtime_fingerprint", lambda: {"test": "runtime"})
    (tmp_path / "config.json").write_text(json.dumps({"hidden_size": 1, "num_hidden_layers": 3}))
    prepared = SimpleNamespace(
        snapshot={"snapshot_sha256": "one", "snapshot_path": str(tmp_path)},
        corpus_manifest_sha256="corpus",
        manifest={"sequences": {"sha256": "rows"}},
        rows=[{"ids": [1, 2], "score_positions": [1], "split": "fit"}],
    )
    original = binding(prepared, "native", "memory")
    assert binding(prepared, "hand_run", "memory") != original
    assert binding(prepared, "native", "split_spill") != original
    for field, value in [
        ("snapshot", {"snapshot_sha256": "two", "snapshot_path": str(tmp_path)}),
        ("corpus_manifest_sha256", "changed"),
        ("rows", [{"ids": [1, 2], "split": "fit"}]),
    ]:
        changed = copy.deepcopy(prepared)
        setattr(changed, field, value)
        assert binding(changed, "native", "memory") != original


def test_bare_full_fit_refuses_before_prepare(monkeypatch, tmp_path):
    import importlib.util
    from pathlib import Path

    from local_llm_lab.pipeline.lens_fitting import runtime

    path = Path(__file__).parents[1] / "scripts/lens_fit.py"
    spec = importlib.util.spec_from_file_location("memory_fit_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(runtime, "prepare_fit", lambda *_: pytest.fail("preparation reached"))
    with pytest.raises(SystemExit):
        module.main(
            [
                "--kind",
                "regression",
                "--model",
                "gemma3-4b",
                "--corpus",
                str(tmp_path / "corpus"),
                "--out",
                str(tmp_path / "lens"),
            ]
        )


def test_invalid_full_fit_evidence_never_reaches_loader(monkeypatch, tmp_path):
    import importlib.util
    from pathlib import Path

    from local_llm_lab.pipeline.lens_fitting import memory_policy, runtime

    expected, events, report, write = evidence(tmp_path)
    events[-1]["status"] = "diagnostic_complete"
    write()
    path = Path(__file__).parents[1] / "scripts/lens_fit.py"
    spec = importlib.util.spec_from_file_location("memory_fit_bad_evidence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(runtime, "prepare_fit", lambda *a, **k: object())
    monkeypatch.setattr(runtime, "load_runtime", lambda *_: pytest.fail("loader reached"))
    monkeypatch.setattr(memory_policy, "binding", lambda *a: expected)
    monkeypatch.setattr(memory_policy, "device_working_set", lambda: 100)
    with pytest.raises(ValueError, match="complete"):
        module.main(
            [
                "--kind",
                "regression",
                "--model",
                "gemma3-4b",
                "--corpus",
                str(tmp_path / "corpus"),
                "--out",
                str(tmp_path / "lens"),
                "--memory-preflight",
                str(report),
            ]
        )


def test_runtime_measured_breach_prevents_next_forward_and_artifact():
    from local_llm_lab.pipeline.lens_fitting.memory_policy import guard_runtime_event

    qualification = {"cap_bytes": 60, "qualified_peak_bytes": 55}
    calls = []
    with pytest.raises(ValueError, match="peak_bytes=56, qualified_peak_bytes=55, cap_bytes=60"):
        guard_runtime_event(
            {"phase": "after_statistics_eval", "peak_memory_bytes": 56}, qualification
        )
        calls.append("next forward")
        calls.append("artifact")
    assert calls == []


def test_runtime_before_forward_rechecks_owner():
    from local_llm_lab.pipeline.lens_fitting.memory_policy import guard_runtime_event

    def refuse():
        raise ValueError("window expired")

    with pytest.raises(ValueError, match="window expired"):
        guard_runtime_event(
            {"phase": "before_forward", "peak_memory_bytes": 1},
            {"cap_bytes": 60, "qualified_peak_bytes": 55},
            require_window=refuse,
        )


def test_late_observed_peak_under_cap_is_part_of_qualified_ceiling(tmp_path):
    expected, events, path, write = evidence(tmp_path)
    measured = [event for event in events if event["event"] == "measured"]
    measured[-1]["peak_bytes"] = 119
    write()
    qualified = validate_evidence(path, expected, 200)
    assert qualified["qualified_peak_bytes"] == 119
    from local_llm_lab.pipeline.lens_fitting.memory_policy import guard_runtime_event

    guard_runtime_event({"phase": "after_statistics_eval", "peak_memory_bytes": 119}, qualified)
