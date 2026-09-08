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


def test_falsified_initial_bound_stops_before_second_calibration():
    """R47(b)/review: measured140 below cap150 still falsifies initial bound100."""
    calls = []

    def measure(tokens):
        calls.append(tokens)
        return {"tokens": tokens, "peak_bytes": 140}

    rows, stop = api().run_ladder(
        [256, 512],
        measure,
        fixed_bytes=10,
        initial_bound_bytes=100,
        cap_bytes=150,
        emit=lambda e: None,
    )
    assert calls == [256]
    assert len(rows) == 1
    assert stop["reason"] == "measured peak falsified initial bound"
    assert stop["projected_peak_bytes"] == 100
    assert stop["peak_bytes"] == 140


@pytest.mark.parametrize("residual_source", ["hand_run", "native"])
def test_solve_resets_peak_and_reports_successful_measurement(
    tmp_path, monkeypatch, residual_source
):
    """R47(b)/review: solve peak starts after forward and records measured+projected bytes."""
    import json
    import sys
    from types import ModuleType, SimpleNamespace

    from local_llm_lab.pipeline.lens_fitting import regression, runtime

    calls = []
    backend = ModuleType("mlx.core")
    backend.set_cache_limit = lambda n: 1024
    backend.clear_cache = lambda: None
    backend.get_active_memory = lambda: 10
    backend.device_info = lambda: {"max_recommended_working_set_size": 1000}
    backend.reset_peak_memory = lambda: calls.append("reset")
    backend.get_peak_memory = lambda: 50 if calls[-1] == "solve" else 100
    mlx = ModuleType("mlx")
    mlx.core = backend
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", backend)
    prepared = SimpleNamespace(
        rows=[dict(index=0, ids=[1] * 900, split="fit", score_positions=[0, 300, 899])],
        snapshot={},
        corpus_manifest_sha256="corpus",
    )
    monkeypatch.setattr(runtime, "prepare_fit", lambda *a, **k: prepared)
    monkeypatch.setattr(
        runtime,
        "load_runtime",
        lambda *a: SimpleNamespace(
            model=SimpleNamespace(eval=lambda: None),
            lock_path=tmp_path / "lock",
            view=SimpleNamespace(hidden_size=1, num_layers=2),
        ),
    )

    def accumulate(*args, **kwargs):
        assert kwargs["residual_source"] == residual_source
        for row in args[1]:
            start = 900 - len(row["ids"])
            assert row["score_positions"] == [p - start for p in [0, 300, 899] if p >= start]
        calls.append("forward")
        return {"fit": {1: None}, "held": {1: None}}, {}

    def solve(*args):
        assert calls[-1] == "reset"
        calls.append("solve")
        return None

    monkeypatch.setattr(regression, "accumulate", accumulate)
    monkeypatch.setattr(regression, "solve_layer", solve)
    registration = tmp_path / "registration.md"
    registration.write_text(json.dumps({"residual_source": residual_source}))
    report = tmp_path / "report.jsonl"
    assert (
        api().main(
            [
                "--model",
                "qwen35-4b",
                "--corpus",
                str(tmp_path / "corpus.json"),
                "--planned-lens",
                str(tmp_path / "lens.npz"),
                "--registration",
                str(registration),
                "--report",
                str(report),
                "--initial-bound-gib",
                str(200 / 2**30),
                "--residual-source",
                residual_source,
            ]
        )
        == 0
    )
    event = next(
        e
        for e in map(json.loads, report.read_text().splitlines())
        if e["event"] == "measured_solve"
    )
    begin = json.loads(report.read_text().splitlines()[0])
    assert begin["residual_source"] == residual_source
    assert begin["source_input_positions"] == 900
    assert begin["source_scored_positions"] == 3
    assert begin["calibration_window_rule"].startswith("suffix ending at registered source end")
    measured = [
        e for e in map(json.loads, report.read_text().splitlines()) if e["event"] == "measured"
    ]
    assert measured[0]["source_window_start"] == 644
    assert measured[0]["source_window_end"] == 900
    assert measured[0]["scored_positions_per_sequence"] == 1
    assert measured[-1]["scored_positions_per_sequence"] == 3
    assert event["peak_bytes"] == 50
    assert event["projected_peak_bytes"] == 74


def test_calibration_rows_preserve_and_truncate_score_selection():
    a = api()
    source = dict(ids=[1, 2, 3, 4], score_positions=[1, 3], index=5, split="fit")
    rows = a.calibration_rows(source, 3)
    assert rows == [dict(ids=[2, 3, 4], score_positions=[0, 2], split=s) for s in ("fit", "held")]
    assert source["score_positions"] == [1, 3]
    with pytest.raises(ValueError, match="scor"):
        a.calibration_rows(dict(ids=[1, 2, 3, 4], score_positions=[0]), 1)
    with pytest.raises(ValueError):
        a.calibration_rows(source, 5)
    assert a.calibration_rows(dict(ids=[1, 2]), 1) == [
        dict(ids=[1], split=s) for s in ("fit", "held")
    ]


@pytest.mark.parametrize(
    "record,source,valid",
    [
        ('{"residual_source":"native"}', "native", True),
        ('{"residual_source":"hand_run"}', "native", False),
        ('{"residual_source":"native"}', "hand_run", False),
        ("{}", "native", False),
        ("Legacy bound", "native", False),
        ("Legacy bound", "hand_run", True),
    ],
)
def test_registration_residual_source_is_explicit_for_native(tmp_path, record, source, valid):
    path = tmp_path / "registration.json"
    path.write_text(record)
    if valid:
        api().validate_registration_source(path, source)
    else:
        with pytest.raises(ValueError, match="residual_source"):
            api().validate_registration_source(path, source)


@pytest.mark.parametrize("tokens", [256, 512, 1024, 2048])
def test_tail_scoring_calibration_preserves_registered_token_ownership(tokens):
    source = dict(ids=list(range(2048)), score_positions=list(range(1024, 2048)))
    rows = api().calibration_rows(source, tokens)
    start = 2048 - tokens
    expected_scores = list(range(max(0, 1024 - start), tokens))
    for row in rows:
        assert row["ids"] == list(range(start, 2048))
        assert row["score_positions"] == expected_scores
        assert [row["ids"][p] for p in row["score_positions"]] == [
            p for p in source["score_positions"] if p >= start
        ]
    assert len(rows[0]["score_positions"]) == min(tokens, 1024)
    assert source == dict(ids=list(range(2048)), score_positions=list(range(1024, 2048)))
