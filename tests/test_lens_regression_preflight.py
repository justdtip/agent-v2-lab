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
    assert a.ladder(100) == [25, 50, 100]
    assert a.ladder(128) == [32, 64, 128]
    assert a.ladder(2) == [1, 2]
    with pytest.raises(ValueError, match="short"):
        a.ladder(1)


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
@pytest.mark.parametrize("storage", ["memory", "split_spill"])
def test_solve_resets_peak_and_reports_successful_measurement(
    tmp_path, monkeypatch, residual_source, storage
):
    """R47(b)/review: solve peak starts after forward and records measured+projected bytes."""
    import json
    import sys
    from types import ModuleType, SimpleNamespace

    from local_llm_lab.pipeline.lens_fitting import memory_policy, regression, runtime

    monkeypatch.setattr(memory_policy, "require_owned_window", lambda: None)
    monkeypatch.setattr(memory_policy, "binding", lambda *a: {})
    calls = []
    backend = ModuleType("mlx.core")
    backend.set_cache_limit = lambda n: 1024
    backend.clear_cache = lambda: None
    backend.get_active_memory = lambda: 10
    backend.device_info = lambda: {"max_recommended_working_set_size": 1000}
    backend.reset_peak_memory = lambda: calls.append("reset")
    backend.get_peak_memory = lambda: 50 if calls and calls[-1] == "solve" else 100
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

    point_directories = []

    def accumulate(*args, **kwargs):
        assert kwargs["residual_source"] == residual_source
        assert kwargs["statistics_storage"] == storage
        point = kwargs["scratch_dir"]
        assert point.is_dir()
        assert all(not prior.exists() for prior in point_directories)
        point_directories.append(point)
        (point / "retained-stats").write_text("fake statistics")
        for row in args[1]:
            start = 900 - len(row["ids"])
            assert row["score_positions"] == [p - start for p in [0, 300, 899] if p >= start]
        calls.append("forward")
        return {"fit": {1: None}, "held": {1: None}}, {}

    def solve(*args):
        assert (point_directories[-1] / "retained-stats").is_file()
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
                "--statistics-storage",
                storage,
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
    assert event["projected_peak_bytes"] == 82
    assert len(point_directories) == 3
    assert all(not point.exists() for point in point_directories)
    assert not point_directories[-1].parent.exists()


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


def test_initial_over_cap_refuses_before_preparation_or_loading(monkeypatch, tmp_path):
    from local_llm_lab.pipeline.lens_fitting import memory_policy, runtime

    (tmp_path / "registration.json").write_text('{"residual_source": "hand_run"}')
    monkeypatch.setattr(memory_policy, "device_working_set", lambda: 17.76 * 2**30)
    monkeypatch.setattr(runtime, "prepare_fit", lambda *_: pytest.fail("preparation reached"))
    monkeypatch.setattr(runtime, "load_runtime", lambda *_: pytest.fail("loading reached"))
    with pytest.raises(ValueError, match="0.6"):
        api().main(
            [
                "--model",
                "gemma3-4b",
                "--corpus",
                str(tmp_path / "corpus.json"),
                "--planned-lens",
                str(tmp_path / "lens.npz"),
                "--registration",
                str(tmp_path / "registration.json"),
                "--report",
                str(tmp_path / "report.jsonl"),
                "--initial-bound-gib",
                "21",
            ]
        )


@pytest.mark.parametrize("breach_peak", [None, 55, 65])
def test_short_diagnostic_repeats_eight_rows_and_never_solves(monkeypatch, tmp_path, breach_peak):
    import json
    import sys
    from types import SimpleNamespace

    from local_llm_lab.pipeline.lens_fitting import memory_policy, regression, runtime

    mx = SimpleNamespace(
        set_cache_limit=lambda _: 0,
        clear_cache=lambda: None,
        reset_peak_memory=lambda: None,
        get_peak_memory=lambda: 30,
        get_active_memory=lambda: 20,
        device_info=lambda: {"max_recommended_working_set_size": 100},
    )
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=mx))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setattr(memory_policy, "device_working_set", lambda: 100)
    monkeypatch.setattr(memory_policy, "require_owned_window", lambda: None)
    monkeypatch.setattr(memory_policy, "binding", lambda *a: {})
    prepared = SimpleNamespace(
        rows=[{"index": 0, "ids": [1] * 128, "split": "fit"}],
        snapshot={},
        corpus_manifest_sha256="test",
    )
    monkeypatch.setattr(runtime, "prepare_fit", lambda *a: prepared)
    monkeypatch.setattr(
        runtime,
        "load_runtime",
        lambda *a: SimpleNamespace(
            model=SimpleNamespace(eval=lambda: None),
            view=SimpleNamespace(hidden_size=1, num_layers=2),
            lock_path=tmp_path / "lock",
        ),
    )
    calls = []

    def accumulate(view, rows, **kwargs):
        calls.append((rows, kwargs["statistics_storage"]))
        assert kwargs["scratch_dir"].is_dir()
        kwargs["memory_progress"]({"phase": "before_forward", "peak_memory_bytes": 30})
        kwargs["memory_progress"](
            {"phase": "after_statistics_eval", "peak_memory_bytes": breach_peak or 30}
        )
        calls.append("subsequent forward permitted")
        return {}, {"fit": {"sequences": 4}, "held": {"sequences": 4}}

    monkeypatch.setattr(regression, "accumulate", accumulate)
    monkeypatch.setattr(regression, "solve_layer", lambda *a: pytest.fail("solver reached"))
    registration = tmp_path / "registration.json"
    registration.write_text(json.dumps({"residual_source": "native"}))
    report = tmp_path / "report.jsonl"

    def invoke():
        return api().main(
            [
                "--model",
                "gemma3-4b",
                "--corpus",
                str(tmp_path / "corpus"),
                "--planned-lens",
                str(tmp_path / "lens"),
                "--registration",
                str(registration),
                "--report",
                str(report),
                "--initial-bound-gib",
                str(50 / 2**30),
                "--residual-source",
                "native",
                "--statistics-storage",
                "split_spill",
                "--diagnostic-tokens",
                "128",
            ]
        )

    if breach_peak is not None:
        with pytest.raises(ValueError, match="measured memory breach"):
            invoke()
        assert len(calls) == 1  # no subsequent row/step after the observed breach
        events = [json.loads(line) for line in report.read_text().splitlines()]
        assert events[-1]["status"] == "error"
        assert events[-2]["event"] == "memory_phase"
        assert events[-2]["peak_memory_bytes"] == breach_peak
        return
    assert invoke() == 0
    assert len(calls) == 2 and len(calls[0][0]) == 8 and calls[0][1] == "split_spill"
    assert all(len(row["ids"]) == 128 for row in calls[0][0])
    events = [json.loads(line) for line in report.read_text().splitlines()]
    assert events[-1]["status"] == "diagnostic_complete"
    assert not any(event["event"] == "measured_solve" for event in events)
