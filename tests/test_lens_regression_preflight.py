"""Requirements §3.2 and R47(b): resource estimates precede larger native forwards."""

import importlib.util
from pathlib import Path

import pytest


def api():
    path = Path(__file__).resolve().parents[1] / "scripts/lens_regression_preflight.py"
    spec = importlib.util.spec_from_file_location("lens_resource_preflight", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_projection_preserves_fixed_storage_and_at_least_quadratic_variable_cost():
    """R47(b): project next size from prior peaks before allocating its forward."""
    rows = [{"tokens": 2, "peak_bytes": 104}, {"tokens": 4, "peak_bytes": 116}]
    assert api().project_peak(rows, 8, fixed_bytes=100) == pytest.approx(164)
    rows[-1]["peak_bytes"] = 132
    assert api().project_peak(rows, 8, fixed_bytes=100) == pytest.approx(356)


def test_projection_rejects_invalid_or_extrapolation_free_rows():
    """R47(b): missing/nonfinite measurements cannot certify a next size."""
    a = api()
    with pytest.raises(ValueError):
        a.project_peak([{"tokens": 2, "peak_bytes": 104}], 8, fixed_bytes=100)
    with pytest.raises(ValueError):
        a.project_peak(
            [{"tokens": 2, "peak_bytes": 104}, {"tokens": 4, "peak_bytes": float("nan")}],
            8,
            fixed_bytes=100,
        )
    with pytest.raises(ValueError):
        a.project_peak(
            [{"tokens": 2, "peak_bytes": 104}, {"tokens": 4, "peak_bytes": 116}], 3, fixed_bytes=100
        )


def test_ladder_uses_actual_maximum_and_never_extends_the_corpus():
    """§3.2: resource probing does not authorise longer prompts than the frozen corpus."""
    a = api()
    assert a.ladder(900) == [256, 512, 900]
    assert a.ladder(1024) == [256, 512, 1024]
    assert a.ladder(2044) == [256, 512, 1024, 2044]
    with pytest.raises(ValueError, match="short"):
        a.ladder(100)


def test_refused_projection_never_calls_larger_measurement():
    """R47(b): a post-measurement stop would fail this acceptance test."""
    a = api()
    called, emitted = [], []

    def measure(n):
        called.append(n)
        return {"tokens": n, "peak_bytes": {2: 104, 4: 116, 8: 164}[n]}

    rows, stop = a.run_ladder(
        [2, 4, 8],
        measure,
        fixed_bytes=100,
        initial_bound_bytes=120,
        cap_bytes=150,
        emit=emitted.append,
    )
    assert called == [2, 4]
    assert len(rows) == 2
    assert stop["projected_peak_bytes"] == pytest.approx(164)
    assert stop["next_tokens"] == 8
    assert emitted[-1] == stop


def test_initial_bound_and_measured_breach_stop_without_another_call():
    """R47: the initial declared bound is checked first; measured breach is labelled."""
    a = api()
    called = []

    def measure(n):
        called.append(n)
        return {"tokens": n, "peak_bytes": 151}

    rows, stop = a.run_ladder(
        [2, 4],
        measure,
        fixed_bytes=100,
        initial_bound_bytes=151,
        cap_bytes=150,
        emit=lambda x: None,
    )
    assert not called and not rows and stop["reason"] == "initial bound exceeds cap"
    rows, stop = a.run_ladder(
        [2, 4],
        measure,
        fixed_bytes=100,
        initial_bound_bytes=149,
        cap_bytes=150,
        emit=lambda x: None,
    )
    assert called == [2] and len(rows) == 1
    assert stop["reason"] == "measured peak exceeded cap; breach, not pre-launch protection"


def test_timing_projection_carries_all_sequences_and_all_fixed_grid_solves():
    """§3.2: projected total includes the full frozen corpus and every alpha/layer."""
    a = api()
    report = a.project_time(
        [{"tokens": 2, "seconds_per_sequence": 1}, {"tokens": 4, "seconds_per_sequence": 2}],
        [2, 2, 4],
        one_layer_solve_s=3,
        nonfinal_layers=7,
    )
    assert report["forward_seconds"] == pytest.approx(4)
    assert report["solve_seconds"] == pytest.approx(21)
    assert report["projected_fit_seconds"] == pytest.approx(25)


def test_registration_failure_precedes_preparation_or_loading(tmp_path, monkeypatch):
    """§4/7: no checkpoint import is reached without a durable pre-registration."""
    from local_llm_lab.pipeline.lens_fitting import runtime

    monkeypatch.setattr(
        runtime, "prepare_fit", lambda *a, **k: pytest.fail("prepared before registration")
    )
    with pytest.raises(FileNotFoundError):
        api().main(
            [
                "--model",
                "qwen35-4b",
                "--corpus",
                str(tmp_path / "manifest.json"),
                "--planned-lens",
                str(tmp_path / "qwen35-4b-agentic-regression.npz"),
                "--registration",
                str(tmp_path / "missing.md"),
                "--report",
                str(tmp_path / "report.jsonl"),
                "--initial-bound-gib",
                "1",
            ]
        )
    assert not (tmp_path / "report.jsonl").exists()


def test_existing_report_is_preserved_before_loading(tmp_path, monkeypatch):
    """§4/7: resource evidence is exclusive; replaying a command cannot overwrite it."""
    from types import SimpleNamespace

    from local_llm_lab.pipeline.lens_fitting import runtime

    monkeypatch.setattr(
        runtime,
        "prepare_fit",
        lambda *a, **k: SimpleNamespace(rows=[{"ids": [1] * 900, "split": "fit"}]),
    )
    monkeypatch.setattr(
        runtime, "load_runtime", lambda *a, **k: pytest.fail("loaded despite existing report")
    )
    report = tmp_path / "report.jsonl"
    report.write_text("prior evidence")
    registration = tmp_path / "registration.md"
    registration.write_text("Bound declared before launch")
    with pytest.raises(FileExistsError):
        api().main(
            [
                "--model",
                "qwen35-4b",
                "--corpus",
                str(tmp_path / "manifest.json"),
                "--planned-lens",
                str(tmp_path / "qwen35-4b-agentic-regression.npz"),
                "--registration",
                str(registration),
                "--report",
                str(report),
                "--initial-bound-gib",
                "1",
            ]
        )
    assert report.read_text() == "prior evidence"
