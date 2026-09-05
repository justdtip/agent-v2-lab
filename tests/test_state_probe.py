from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from local_llm_lab.models import load_model_spec
from local_llm_lab.pipeline.tasks import GENERATOR_VERSION, make_tasks
from local_llm_lab.probes import state_probe


def _events(output: Path) -> list[dict[str, Any]]:
    """Every event the run log wrote to ``output/events.jsonl``, in order."""
    lines = (output / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _flat(event: dict[str, Any]) -> dict[str, Any]:
    """One event's top-level keys with its ``fields`` object merged in."""
    flat = {key: value for key, value in event.items() if key != "fields"}
    flat.update(event.get("fields") or {})
    return flat


def test_explicit_registered_specs_preserve_legacy_bytes_and_qwen35_template_policy() -> None:
    from local_llm_lab.pipeline.protocol import build_prompt, generation_suffix

    class Tokenizer:
        def __init__(self) -> None:
            self.kwargs = []

        def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, **kwargs):
            assert not tokenize
            self.kwargs.append(kwargs)
            rendered = "".join(f"{message['role']}:{message['content']}\n" for message in messages)
            if not add_generation_prompt:
                return rendered
            suffix = "<|im_start|>assistant\n"
            if kwargs.get("enable_thinking") is False:
                suffix += "<think>\n\n</think>\n\n"
            return rendered + suffix

    tokenizer = Tokenizer()
    legacy = load_model_spec("qwen25-coder-3b")
    qwen35 = load_model_spec("qwen35-4b")
    messages = [{"role": "user", "content": "hello"}]

    assert build_prompt(tokenizer, messages) == build_prompt(tokenizer, messages, spec=legacy)
    assert build_prompt(tokenizer, messages, spec=qwen35).endswith(generation_suffix(qwen35))
    assert tokenizer.kwargs[-1] == qwen35.chat.template_kwargs


def test_build_probe_dataset_forwards_the_selected_spec_to_prompt_rendering(
    monkeypatch, tmp_path
) -> None:
    selected = load_model_spec("qwen35-4b")
    task = SimpleNamespace(task_id="fake", family="read")
    seen = []

    contexts = []
    monkeypatch.setattr(
        state_probe,
        "_checkpoint_signature",
        lambda *_args, context, **_kwargs: contexts.append(context) or "fake",
    )
    monkeypatch.setattr(
        state_probe,
        "build_rows",
        lambda *_args, **_kwargs: [
            {
                "messages": [
                    {"role": "user", "content": "x"},
                    {"role": "assistant", "content": "y"},
                ],
                "metadata": {"step": 0},
            }
        ],
    )
    monkeypatch.setattr(
        state_probe,
        "build_prompt",
        lambda *_args, spec=None, **_kwargs: seen.append(spec) or "prompt",
    )
    monkeypatch.setattr(
        state_probe,
        "capture_residuals",
        lambda *_args, **_kwargs: {0: np.array([1.0])},
    )
    monkeypatch.setattr(state_probe, "_materialize_residuals", lambda captured, *_args: captured)
    monkeypatch.setattr(
        state_probe, "row_labels", lambda *_args: {name: 0.0 for name in state_probe.TARGETS}
    )
    monkeypatch.setattr(state_probe, "mlx_memory_snapshot", lambda *_args: {})

    runtime = SimpleNamespace(reset_peak_memory=lambda: None, clear_cache=lambda: None)
    tokenizer = SimpleNamespace(encode=lambda *_args, **_kwargs: [1])
    dataset = state_probe.build_probe_dataset(
        None,
        tokenizer,
        [task],
        [0],
        spec=selected,
        mlx_runtime=runtime,
        checkpoint_dir=tmp_path / "checkpoints",
    )

    assert seen == [selected]
    assert dataset.meta["generator_version"] == GENERATOR_VERSION
    assert contexts == [{"generator_version": GENERATOR_VERSION}]
    shard = state_probe.load_dataset(tmp_path / "checkpoints" / "0001.npz")
    assert shard.meta["generator_version"] == GENERATOR_VERSION


def test_state_probe_default_layers_use_actual_depth_and_persist_selection(
    monkeypatch, tmp_path
) -> None:
    """Catches qwen25 defaults, spec-free policy lookup, or missing capture metadata."""
    import sys

    from local_llm_lab import models
    from local_llm_lab.pipeline import evaluate, tasks
    from local_llm_lab.probes import guard, policies

    selected = SimpleNamespace(
        name="qwen35-4b",
        hf_id="fake/hf",
        policies={},
        probe_layer_fractions=(0.167, 0.333, 0.5, 0.667, 0.833, 1.0),
    )
    model = object()
    seen = []
    monkeypatch.setattr(
        models,
        "load_model_spec",
        lambda model: seen.append(("load", model)) or selected,
    )
    monkeypatch.setattr(
        tasks,
        "make_tasks",
        lambda *_args, **_kwargs: [SimpleNamespace(task_id="t", difficulty=0)],
    )
    monkeypatch.setattr(state_probe, "task_difficulties", lambda *_args: {"t": 0})
    monkeypatch.setattr(guard, "require_idle_gpu", lambda *_args: None)
    monkeypatch.setattr(
        policies,
        "resolve_policy",
        lambda name, spec: seen.append(("policy", name, spec)) or None,
    )
    monkeypatch.setattr(
        evaluate,
        "load_policy",
        lambda *_args: (model, object(), SimpleNamespace(num_layers=32), object()),
    )
    monkeypatch.setattr(state_probe, "artifact_identity", lambda *_args: {})
    monkeypatch.setattr(state_probe, "set_mlx_cache_limit", lambda *_args: 0)
    fake_mlx = SimpleNamespace(clear_cache=lambda: None, set_cache_limit=lambda _value: None)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mlx)

    expected = {
        "source": "registry-default",
        "requested": ["0.167", "0.333", "0.5", "0.667", "0.833", "1.0"],
        "fractions": [0.167, 0.333, 0.5, 0.667, 0.833, 1.0],
        "indices": [5, 11, 16, 21, 27, 32],
        "num_layers": 32,
    }

    def capture(_model, _tokenizer, _tasks, layers, *_args, **kwargs):
        assert layers == [5, 11, 16, 21, 27, 32]
        assert kwargs["layer_selection"] == expected
        assert kwargs["checkpoint_context"]["layer_selection"] == expected
        seen.append(("dispatch", kwargs["spec"]))
        return state_probe.ProbeDataset(
            layers=layers,
            features={layer: np.empty((0, 1), dtype=np.float32) for layer in layers},
            labels={},
            task_ids=np.array([], dtype=str),
            meta={"layers": layers, "layer_selection": expected},
        )

    monkeypatch.setattr(state_probe, "build_probe_dataset", capture)
    monkeypatch.setattr(state_probe, "save_dataset", lambda _dataset, path: path)

    def fit(dataset, _targets, layers, *_args, **_kwargs):
        assert dataset.meta["layer_selection"] == expected
        assert layers == expected["indices"]
        return {"meta": dataset.meta, "targets": {}}

    monkeypatch.setattr(state_probe, "fit_probes", fit)
    monkeypatch.setattr(state_probe, "render_markdown", lambda *_args: "# fake")
    monkeypatch.setattr(
        "sys.argv",
        ["state-probe", "--model", "qwen35-4b", "--output", str(tmp_path), "--limit", "1"],
    )

    state_probe.main()

    assert seen == [
        ("load", "qwen35-4b"),
        ("policy", "base", selected),
        ("dispatch", selected),
    ]
    payload = json.loads((tmp_path / "state-base.json").read_text(encoding="utf-8"))
    assert payload["meta"]["layer_selection"] == expected


def test_state_probe_capture_writes_a_run_log_with_identity_and_progress(
    monkeypatch, tmp_path, capsys
) -> None:
    """R26 (issue #35, clauses e and g): the capture CLI leaves run.log and events.jsonl,
    the start event names what ran, and one progress line fires per captured task."""
    import sys

    from local_llm_lab import models
    from local_llm_lab.pipeline import evaluate, tasks
    from local_llm_lab.probes import guard, policies

    selected = SimpleNamespace(
        name="qwen35-4b",
        hf_id="fake/hf",
        policies={},
        probe_layer_fractions=(0.5, 1.0),
    )
    monkeypatch.setattr(models, "load_model_spec", lambda _name: selected)
    monkeypatch.setattr(
        tasks,
        "make_tasks",
        lambda *_args, **_kwargs: [SimpleNamespace(task_id="t", difficulty=0)],
    )
    monkeypatch.setattr(state_probe, "task_difficulties", lambda *_args: {"t": 0})
    monkeypatch.setattr(guard, "require_idle_gpu", lambda *_args: None)
    monkeypatch.setattr(policies, "resolve_policy", lambda _name, _spec: None)
    monkeypatch.setattr(
        evaluate,
        "load_policy",
        lambda *_args: (object(), object(), SimpleNamespace(num_layers=4), object()),
    )
    monkeypatch.setattr(state_probe, "artifact_identity", lambda *_args: {})
    monkeypatch.setattr(state_probe, "set_mlx_cache_limit", lambda *_args: 0)
    monkeypatch.setitem(
        sys.modules,
        "mlx.core",
        SimpleNamespace(clear_cache=lambda: None, set_cache_limit=lambda _value: None),
    )

    mib = 2**20

    def capture(_model, _tokenizer, _tasks, layers, *_args, **kwargs):
        kwargs["memory_progress"](
            {
                "resumed": False,
                "active_bytes": 3 * mib,
                "cache_bytes": 5 * mib,
                "peak_bytes": 7 * mib,
            }
        )
        kwargs["progress"](1, 2, 4)
        kwargs["memory_progress"](
            {
                "resumed": True,
                "active_bytes": 11 * mib,
                "cache_bytes": 13 * mib,
                "peak_bytes": None,
            }
        )
        kwargs["progress"](2, 2, 9)
        return state_probe.ProbeDataset(
            layers=list(layers),
            features={layer: np.empty((0, 1), dtype=np.float32) for layer in layers},
            labels={},
            task_ids=np.array([], dtype=str),
            meta={"layers": list(layers)},
        )

    monkeypatch.setattr(state_probe, "build_probe_dataset", capture)
    monkeypatch.setattr(state_probe, "save_dataset", lambda _dataset, path: path)
    monkeypatch.setattr(
        state_probe,
        "fit_probes",
        lambda dataset, *_args, **_kwargs: {"meta": dataset.meta, "targets": {}},
    )
    monkeypatch.setattr(state_probe, "render_markdown", lambda *_args: "# fake")
    monkeypatch.setattr(
        "sys.argv",
        ["state-probe", "--model", "qwen35-4b", "--output", str(tmp_path), "--limit", "1"],
    )

    state_probe.main()

    out = capsys.readouterr().out
    assert (tmp_path / "run.log").is_file()
    events = _events(tmp_path)
    start = _flat(events[0])
    assert events[0]["kind"] == "start"
    assert events[0]["run"] == "state-probe"
    assert start["model"] == "qwen35-4b"
    assert start["hf_id"] == "fake/hf"
    assert start["policy"] == "base"
    assert start["adapter"] is None
    assert start["splits"] == ["train"]
    assert start["command"] == list(sys.argv)
    assert isinstance(start["git_commit"], str)

    progress = [_flat(event) for event in events if event["kind"] == "progress"]
    assert [(item["step"], item["total"], item["label"]) for item in progress] == [
        (1, 2, "capture"),
        (2, 2, "capture"),
    ]
    assert [item["rows"] for item in progress] == [4, 9]
    assert [item["source"] for item in progress] == ["captured", "resumed"]
    assert [item["active_mib"] for item in progress] == [3, 11]
    assert [item["cache_mib"] for item in progress] == [5, 13]
    assert [item["task_peak_mib"] for item in progress] == [7, None]

    npz_path = tmp_path / "state-base.npz"
    wrote = [
        item
        for item in map(_flat, events)
        if item["kind"] == "info" and item.get("path") == str(npz_path)
    ]
    assert [item["rows"] for item in wrote] == [0]

    # The information the CLI used to print reaches stdout through the log lines.
    assert "tasks=1" in out
    assert 'splits=["train"]' in out
    assert f"mlx_cache_limit_mib={state_probe.DEFAULT_MLX_CACHE_LIMIT_MIB}" in out
    assert f"checkpoint_dir={tmp_path / 'state-base.checkpoints'}" in out
    assert str(npz_path) in out
    assert str(tmp_path / "state-base.json") in out
    assert "[capture 1/2 50%]" in out
    assert "# fake" in out.splitlines()  # print(markdown) is still the CLI's own contract


def test_state_probe_rejects_malformed_layers_before_gpu_or_loader(monkeypatch, tmp_path) -> None:
    """Catches empty layer cells being discarded before the model-loading boundary."""
    from local_llm_lab import models
    from local_llm_lab.pipeline import evaluate, tasks
    from local_llm_lab.probes import guard

    monkeypatch.setattr(models, "load_model_spec", lambda _name: object())
    monkeypatch.setattr(tasks, "make_tasks", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        guard, "require_idle_gpu", lambda *_args: pytest.fail("reached the GPU guard")
    )
    monkeypatch.setattr(
        evaluate, "load_policy", lambda *_args: pytest.fail("reached the model loader")
    )
    monkeypatch.setattr(
        "sys.argv",
        ["state-probe", "--output", str(tmp_path), "--layers", "1,,2"],
    )

    with pytest.raises(SystemExit) as raised:
        state_probe.main()

    assert raised.value.code == 2


@pytest.mark.parametrize("has_selection", [False, True])
def test_state_probe_reuse_discloses_legacy_or_preserves_recorded_selection(
    monkeypatch, tmp_path, capsys, has_selection
) -> None:
    """Catches reused captures guessing depth/fractions or replacing recorded provenance."""
    expected = (
        {
            "source": "registry-default",
            "requested": ["0.5", "1.0"],
            "fractions": [0.5, 1.0],
            "indices": [1, 2],
            "num_layers": 2,
        }
        if has_selection
        else {
            "source": "legacy-artifact",
            "requested": None,
            "fractions": None,
            "indices": [1, 2],
            "num_layers": None,
        }
    )
    metadata = {"layers": [1, 2]}
    if has_selection:
        metadata["layer_selection"] = expected
    state_probe.save_dataset(
        state_probe.ProbeDataset(
            layers=[1, 2],
            features={
                1: np.zeros((1, 1), dtype=np.float32),
                2: np.zeros((1, 1), dtype=np.float32),
            },
            labels={},
            task_ids=np.array(["test-read-0000-clean"]),
            meta=metadata,
        ),
        tmp_path / "state-base.npz",
    )
    monkeypatch.setattr(
        state_probe,
        "fit_probes",
        lambda dataset, _targets, layers, *_args, **_kwargs: (
            {"meta": dataset.meta, "layers_seen": layers, "targets": {}}
        ),
    )
    monkeypatch.setattr(state_probe, "render_markdown", lambda *_args: "# fake")
    monkeypatch.setattr(
        "sys.argv", ["state-probe", "--output", str(tmp_path), "--reuse"]
    )

    state_probe.main()

    payload = json.loads((tmp_path / "state-base.json").read_text(encoding="utf-8"))
    assert payload["layers_seen"] == [1, 2]
    assert payload["meta"]["layer_selection"] == expected

    out = capsys.readouterr().out
    events = _events(tmp_path)
    assert events[0]["kind"] == "start"
    assert events[0]["run"] == "state-probe"
    reused = [
        item
        for item in map(_flat, events)
        if item["kind"] == "info" and item["message"] == "reusing"
    ]
    assert [(item["path"], item["rows"]) for item in reused] == [
        (str(tmp_path / "state-base.npz"), 1)
    ]
    assert str(tmp_path / "state-base.npz") in out
    assert "# fake" in out.splitlines()


def test_state_probe_legacy_reuse_rejects_fractional_layers_without_guessing_depth(
    monkeypatch, tmp_path
) -> None:
    """Catches a fraction being resolved against an invented legacy model depth."""
    state_probe.save_dataset(
        state_probe.ProbeDataset(
            layers=[1],
            features={1: np.zeros((1, 1), dtype=np.float32)},
            labels={},
            task_ids=np.array(["test-read-0000-clean"]),
            meta={"layers": [1]},
        ),
        tmp_path / "state-base.npz",
    )
    monkeypatch.setattr(
        "sys.argv",
        ["state-probe", "--output", str(tmp_path), "--reuse", "--layers", "0.5"],
    )

    with pytest.raises(SystemExit) as raised:
        state_probe.main()

    assert raised.value.code == 2


def test_label_dataset_records_the_current_generator_version() -> None:
    dataset = state_probe.build_label_dataset(make_tasks("test", 1))

    assert dataset.meta["generator_version"] == GENERATOR_VERSION


def test_reanalysis_generator_version_requires_a_recording_or_explicit_binding() -> None:
    assert state_probe._reanalysis_generator_version({"generator_version": 2}, None) == 2
    assert state_probe._reanalysis_generator_version({}, 1) == 1
    with pytest.raises(ValueError, match="no generator_version"):
        state_probe._reanalysis_generator_version({}, None)
    with pytest.raises(ValueError, match="conflicts"):
        state_probe._reanalysis_generator_version({"generator_version": 2}, 1)


@pytest.mark.parametrize("invalid", [True, 0, GENERATOR_VERSION + 1, "2"])
def test_reanalysis_generator_version_rejects_invalid_bindings(invalid: object) -> None:
    with pytest.raises(ValueError, match="invalid"):
        state_probe._reanalysis_generator_version({"generator_version": invalid}, None)
    with pytest.raises(ValueError, match="invalid"):
        state_probe._reanalysis_generator_version({}, invalid)  # type: ignore[arg-type]


def test_capture_rejects_a_checkpoint_context_with_a_stale_generator_version() -> None:
    with pytest.raises(ValueError, match="checkpoint context generator_version"):
        state_probe.build_probe_dataset(
            None,
            None,
            [],
            [0],
            checkpoint_context={"generator_version": 1},
            mlx_runtime=object(),
        )


def test_saved_npz_replay_uses_the_recorded_historical_version_without_loaders(
    tmp_path, monkeypatch
) -> None:
    """The offline replay seam reads an NPZ's version, never a model or tokenizer."""
    dataset = state_probe.build_label_dataset(make_tasks("test", 1))
    dataset.meta["generator_version"] = 1
    path = state_probe.save_dataset(dataset, tmp_path / "tiny.npz")
    saved = state_probe.load_dataset(path)
    monkeypatch.setattr(
        state_probe,
        "load_policy",
        lambda *_args: pytest.fail("offline replay must not load a model"),
        raising=False,
    )
    replayed = state_probe._regenerate_tasks(saved, 20260902)

    assert saved.meta["generator_version"] == 1
    assert set(replayed) == set(saved.task_ids.tolist())


def test_saved_recovery_npz_offline_rows_use_historical_version_and_difficulty(tmp_path) -> None:
    task = next(
        item
        for item in make_tasks("train", 144, difficulty=3)
        if item.variant == "wrong_path"
    )
    dataset = state_probe.build_label_dataset([task], {task.task_id: 3})
    dataset.meta.update({"generator_version": 2, "data_seed": 20260902, "keep_last": 2})
    saved = state_probe.load_dataset(state_probe.save_dataset(dataset, tmp_path / "recovery.npz"))
    labels, surface, _metadata = state_probe._offline_rows(saved, data_seed=20260902)
    assert saved.difficulty.tolist() == [3] * len(saved)
    assert labels["pending_count"].shape[0] == len(saved)
    assert surface.shape[0] == len(saved)
    legacy = state_probe._regenerate_tasks(saved, 20260902)[task.task_id]
    assert legacy.steps != task.steps
    saved.meta.pop("generator_version")
    assert (
        state_probe._offline_rows(saved, data_seed=20260902, generator_version=2)[1].shape
        == surface.shape
    )


def test_historical_recovery_reanalysis_reports_version_without_model_seams(
    tmp_path, monkeypatch
) -> None:
    from local_llm_lab.pipeline import evaluate

    task = next(
        item
        for item in make_tasks("train", 144, difficulty=3)
        if item.variant == "wrong_path"
    )
    dataset = state_probe.build_label_dataset([task], {task.task_id: 3})
    dataset.meta.update({"generator_version": 2, "data_seed": 20260902, "keep_last": 2})
    saved = state_probe.load_dataset(state_probe.save_dataset(dataset, tmp_path / "legacy.npz"))
    monkeypatch.setattr(evaluate, "load_policy", lambda *_args: pytest.fail("offline loader"))
    monkeypatch.setattr(
        state_probe, "build_prompt", lambda *_args, **_kwargs: pytest.fail("offline prompt")
    )
    monkeypatch.setattr(
        state_probe,
        "_analyse_cohort",
        lambda *_args, **_kwargs: {"targets": {}},
    )
    legacy_surface = state_probe._offline_rows(saved, data_seed=20260902)[1]
    saved.meta["generator_version"] = 4
    head_surface = state_probe._offline_rows(saved, data_seed=20260902)[1]
    differing = np.flatnonzero(legacy_surface[:, 1] != head_surface[:, 1])
    assert len(differing) > 0
    saved.meta["generator_version"] = 2
    result = state_probe.reanalyse_dataset(saved, split_seeds=(1,), bootstrap_resamples=1)
    assert result["metadata"]["generator_version"] == 2
    assert saved.difficulty.tolist() == [3] * len(saved)


def test_saved_npz_replay_rejects_inconsistent_family_metadata() -> None:
    dataset = state_probe.build_label_dataset(make_tasks("test", 1))
    dataset.family[1] = "search"
    with pytest.raises(ValueError, match="inconsistent families"):
        state_probe._regenerate_tasks(dataset, 20260902)


def test_reanalyse_cli_forwards_explicit_generator_version(monkeypatch, tmp_path) -> None:
    dataset = state_probe.build_label_dataset(make_tasks("test", 1))
    dataset.meta.pop("generator_version")
    seen = []
    monkeypatch.setattr(state_probe, "load_dataset", lambda _path: dataset)
    monkeypatch.setattr(state_probe, "_captured_context", lambda _path: {})

    def stop_after_forwarding(*_args, **kwargs):
        seen.append(kwargs["generator_version"])
        raise RuntimeError("stop after reanalysis dispatch")

    monkeypatch.setattr(state_probe, "reanalyse_dataset", stop_after_forwarding)
    with pytest.raises(RuntimeError, match="stop after reanalysis dispatch"):
        state_probe._main_reanalyse(
            [
                "--input",
                str(tmp_path / "capture.npz"),
                "--output",
                str(tmp_path),
                "--generator-version",
                "1",
            ]
        )
    assert seen == [1]


def test_reanalysis_cli_writes_a_run_log_and_keeps_markdown_on_stdout(
    monkeypatch, tmp_path, capsys
) -> None:
    """R26 (issue #35, clause e): the offline report is as citable as a capture, and the
    markdown the CLI contracts to print stays on stdout unchanged."""
    import hashlib
    import sys

    capture_path = tmp_path / "capture.npz"
    capture_path.write_bytes(b"capture")
    dataset = state_probe.build_label_dataset(make_tasks("test", 1))
    monkeypatch.setattr(state_probe, "load_dataset", lambda _path: dataset)
    monkeypatch.setattr(state_probe, "_captured_context", lambda _path: {})
    monkeypatch.setattr(
        state_probe, "reanalyse_dataset", lambda *_args, **_kwargs: {"metadata": {}}
    )
    monkeypatch.setattr(
        state_probe, "render_reanalysis_markdown", lambda *_args: "# fake-reanalysis"
    )
    argv = ["state-probe", "reanalyse", "--input", str(capture_path), "--output", str(tmp_path)]
    monkeypatch.setattr("sys.argv", argv)

    state_probe._main_reanalyse(sys.argv[2:])

    out = capsys.readouterr().out
    assert (tmp_path / "run.log").is_file()
    events = _events(tmp_path)
    start = _flat(events[0])
    assert events[0]["kind"] == "start"
    assert events[0]["run"] == "state-probe-report"
    assert start["input"] == str(capture_path)
    assert start["input_sha256"] == hashlib.sha256(b"capture").hexdigest()
    assert isinstance(start["git_commit"], str)

    wrote = [
        item["path"]
        for item in map(_flat, events)
        if item["kind"] == "info" and item["message"] == "wrote"
    ]
    assert wrote == [
        str(tmp_path / "capture.reanalysis.json"),
        str(tmp_path / "capture.reanalysis.md"),
    ]
    assert "# fake-reanalysis" in out.splitlines()
    assert str(tmp_path / "capture.reanalysis.json") in out
    assert str(tmp_path / "capture.reanalysis.md") in out


def test_v4_row_labels_match_nonvacuous_progress_note_oracles() -> None:
    """Independent note parsing pins the v4 state labels to real generated progress."""
    assert GENERATOR_VERSION == 4
    counts = {"pending": 0, "loads": 0, "highest": 0, "values": 0, "batch": 0}
    for task in make_tasks("test", 144, perturb=False, difficulty=2):
        for index, step in enumerate(task.steps):
            labels = state_probe.row_labels(task, index)
            note = step.thought
            if step.action.name == "read_file" and "pending: " in note:
                pending = note.rsplit("pending: ", 1)[1].rstrip(".")
                expected = 0 if pending == "none" else len(pending.split(", "))
                if task.family == "batch_update" and expected:
                    expected -= 1  # batch's queue names the current head as well as its tail
                assert labels["pending_count"] == expected
                counts["pending"] += 1
            if "loads so far: " in note:
                observed = note.split("loads so far: ", 1)[1].split(".", 1)[0]
                values = [
                    int(part.rsplit("=", 1)[1])
                    for part in observed.split(", ")
                    if part != "none"
                ]
                assert labels["running_max"] == (max(values) if values else 0)
                counts["loads"] += 1
            if "highest so far: " in note:
                highest = note.split("highest so far: ", 1)[1].split(".", 1)[0]
                value = 0 if highest == "none" else int(
                    highest.split("=", 1)[1].split(";", 1)[0].split(" ", 1)[0]
                )
                assert labels["running_max"] == value
                counts["highest"] += 1
            if "values so far: " in note:
                observed = note.split("values so far: ", 1)[1].split("; split after ", 1)[0]
                values = [] if observed == "none" else observed.split(", ")
                split = int(note.split("split after ", 1)[1].split(" of ", 1)[0])
                assert labels["first_bucket_count"] == min(len(values), split)
                counts["values"] += 1
            if (
                task.family == "batch_update"
                and step.action.name in {"read_file", "replace_text"}
                and ("Inspected " in note or "Applied " in note)
            ):
                if "verified " in note:
                    assert labels["phase"] == "verify"
                elif step.action.name == "replace_text":
                    assert labels["phase"] == "apply"
                elif "Inspected " in note:
                    assert labels["phase"] == "inspect"
                counts["batch"] += 1
    assert all(count > 0 for count in counts.values())


# ------------------------------------------------------------- C7 bfloat16 refit (SPEC-004 §1)


_REFIT_SPLIT_SEEDS = (11, 13)
_REFIT_RESAMPLES = 4
_REFIT_LAYERS = (4, 8)
_REFIT_FEATURE_SEED = 917


def _refit_capture(*, data_seed: int = 20260902) -> Any:
    """A small synthetic capture: real task ids, fake activations, no model anywhere."""
    dataset = state_probe.build_label_dataset(make_tasks("test", 12))
    rng = np.random.default_rng(_REFIT_FEATURE_SEED)
    rows = len(dataset)
    dataset.layers = list(_REFIT_LAYERS)
    dataset.features = {
        layer: rng.standard_normal((rows, 6)).astype(np.float32) for layer in _REFIT_LAYERS
    }
    dataset.meta.update({"layers": list(_REFIT_LAYERS), "data_seed": data_seed})
    return dataset


def _refit_baseline(**overrides: Any) -> dict[str, Any]:
    """The baseline a refit is measured against, produced by the reanalysis itself.

    ``overrides`` reach ``reanalyse_dataset``'s own fit constants, so a test can prove the
    reader takes the writer's numbers rather than defaults that happen to agree with them.
    """
    return state_probe.reanalyse_dataset(
        _refit_capture(),
        split_seeds=_REFIT_SPLIT_SEEDS,
        bootstrap_resamples=_REFIT_RESAMPLES,
        data_seed=20260902,
        **overrides,
    )


def _refit_baseline_json(**overrides: Any) -> dict[str, Any]:
    """The baseline as ``refit-bf16`` READS it: the writer's output, through a JSON round trip.

    R38 (issue #70): ``_refit_baseline`` already comes from ``reanalyse_dataset``, but the CLI
    hands ``baseline_reanalysis_parameters`` a ``json.loads`` of a file, so a tuple that
    survives in memory and becomes a list on disk would be invisible to a test fed the dict.
    """
    return json.loads(_canonical(_refit_baseline(**overrides)))


def _canonical(payload: Any) -> str:
    return json.dumps(
        state_probe._json_compliant(payload), sort_keys=True, allow_nan=False, ensure_ascii=False
    )


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        -0.0,
        1.0,
        -1.0,
        2.0,
        1.5,  # exactly representable in bfloat16: unchanged
        float(np.float32(np.uint32(0x3F808000).view(np.float32))),  # a tie: rounds to even
        float(np.float32(np.uint32(0x3F818000).view(np.float32))),  # a tie the other way
        float(np.float32(np.uint32(0x7F7FFFFF).view(np.float32))),  # largest float32
        float(np.float32(np.uint32(0x00000001).view(np.float32))),  # smallest subnormal
        float(np.float32(np.uint32(0x00800000).view(np.float32))),  # smallest normal: survives
        float(np.float32(np.uint32(0x007FFFFF).view(np.float32))),  # largest subnormal: flushed
        1e-40,
        -1e-40,
        -3.14159265,
        65504.0,
        float("inf"),
        float("-inf"),
        float("nan"),
    ],
)
def test_bfloat16_rounding_helper_matches_the_mlx_cast_on_edge_values(value: float) -> None:
    """One rounding helper, pinned against the cast it stands in for (Section B plan §2)."""
    mx = pytest.importorskip("mlx.core")
    array = np.array([value], dtype=np.float32)
    rounded = state_probe._round_to_bfloat16(array)
    expected = np.array(mx.array(array).astype(mx.bfloat16).astype(mx.float32), dtype=np.float32)

    assert rounded.dtype == np.float32
    if np.isnan(expected[0]):
        assert np.isnan(rounded[0])
    else:
        assert rounded.view(np.uint32)[0] == expected.view(np.uint32)[0]


def test_bfloat16_rounding_helper_matches_the_mlx_cast_bitwise_on_a_random_array() -> None:
    """Every bit of a wide random draw, not just the easy magnitudes."""
    mx = pytest.importorskip("mlx.core")
    rng = np.random.default_rng(_REFIT_FEATURE_SEED)
    scales = np.float32(10.0) ** rng.integers(-44, 30, size=(64, 32)).astype(np.float32)
    values = (rng.standard_normal((64, 32)).astype(np.float32) * scales).astype(np.float32)
    assert np.any((values != 0) & (np.abs(values) < np.finfo(np.float32).tiny))  # subnormals
    rounded = state_probe._round_to_bfloat16(values)
    expected = np.array(mx.array(values).astype(mx.bfloat16).astype(mx.float32), dtype=np.float32)

    assert np.array_equal(rounded.view(np.uint32), expected.view(np.uint32))
    assert np.array_equal(state_probe._round_to_bfloat16(rounded), rounded)


def test_baseline_parameters_read_every_key_at_the_level_the_reanalysis_writes_it() -> None:
    """R38 on the refit's baseline reader (#70): non-default constants, so no default can pass.

    ``reanalyse_dataset`` is asked for a fit nothing defaults to. Had the reader kept its old
    ``fit.get(..., 10.0)`` fallbacks, every one of these three would have come back as the
    default and looked perfectly reasonable, which is exactly the failure R38 is about.
    """
    baseline = _refit_baseline_json(ridge_alpha=7.5, logistic_l2=0.25, logistic_steps=11)

    parameters = state_probe.baseline_reanalysis_parameters(baseline)

    assert parameters == {
        "split_seeds": _REFIT_SPLIT_SEEDS,
        "bootstrap_resamples": _REFIT_RESAMPLES,
        "data_seed": 20260902,
        "generator_version": baseline["metadata"]["generator_version"],
        "ridge_alpha": 7.5,
        "logistic_l2": 0.25,
        "logistic_steps": 11,
    }
    # The seeds survive the JSON round trip as a list and are returned as a tuple, which is
    # the one shape difference between the record in memory and the record on disk.
    assert baseline["metadata"]["split_seeds"] == list(_REFIT_SPLIT_SEEDS)


@pytest.mark.parametrize(
    "path",
    [
        ("metadata",),
        ("metadata", "split_seeds"),
        ("metadata", "bootstrap_resamples"),
        ("metadata", "data_seed"),
        ("metadata", "fit"),
        ("metadata", "fit", "ridge_alpha"),
        ("metadata", "fit", "logistic_l2"),
        ("metadata", "fit", "logistic_steps"),
    ],
)
def test_baseline_parameters_refuse_a_key_that_moved_off_its_level(path: tuple[str, ...]) -> None:
    """R38 step four: every required key renamed in turn, in the writer's own output.

    ``fit`` and its three constants join the seeds here. They used to fall back to
    ``reanalyse_dataset``'s defaults for a class of baseline that has never existed -- the
    ``fit`` block landed in the same commit as the writer -- so a moved block refitted at the
    defaults and reported the comparison as if it had matched the baseline's own constants.
    """
    baseline = _refit_baseline_json(ridge_alpha=7.5, logistic_l2=0.25, logistic_steps=11)
    parent = baseline
    for key in path[:-1]:
        parent = parent[key]
    parent[f"moved_{path[-1]}"] = parent.pop(path[-1])

    with pytest.raises(ValueError, match="refusing to guess"):
        state_probe.baseline_reanalysis_parameters(baseline)


def test_baseline_parameters_keep_generator_version_soft_for_the_ratified_baseline() -> None:
    """The one soft read that has a real subject, so it is pinned rather than tightened.

    ``state-base-mix.reanalysis.json`` -- the ratified P2 baseline every refit is measured
    against -- predates R23 and records no ``generator_version``. The refit binds one from
    ``--generator-version`` in that case and refuses a conflicting one, which is the branch
    this softness exists to reach; tightening it would refuse the only baseline on disk.
    """
    baseline = _refit_baseline_json()
    assert baseline["metadata"]["generator_version"] is not None

    baseline["metadata"]["moved_generator_version"] = baseline["metadata"].pop("generator_version")

    parameters = state_probe.baseline_reanalysis_parameters(baseline)

    assert parameters["generator_version"] is None
    # Nothing else about the record changes, which is why the CLI must supply the version
    # rather than this read reporting a problem.
    assert parameters["data_seed"] == 20260902


def test_refit_without_rounding_reproduces_the_baseline_exactly(tmp_path) -> None:
    """The `--no-round` control: same seeds, same code, byte-identical analyses."""
    baseline = _refit_baseline()
    parameters = state_probe.baseline_reanalysis_parameters(baseline)
    assert parameters["split_seeds"] == _REFIT_SPLIT_SEEDS
    assert parameters["bootstrap_resamples"] == _REFIT_RESAMPLES
    assert parameters["data_seed"] == 20260902
    assert parameters["logistic_steps"] == 120

    control = state_probe.refit_bf16(
        _refit_capture(), parameters=parameters, round_activations=False
    )

    assert _canonical(control["analyses"]) == _canonical(baseline["analyses"])
    assert control["readme"] == baseline["readme"]
    assert control["metadata"]["bfloat16_refit"]["rounding"] is False


def test_refit_rounding_changes_the_stored_activations_but_keeps_the_schema() -> None:
    """Rounding is applied to every layer and only to the layers."""
    baseline = _refit_baseline()
    parameters = state_probe.baseline_reanalysis_parameters(baseline)
    rounded = state_probe.refit_bf16(_refit_capture(), parameters=parameters)

    assert rounded["metadata"]["bfloat16_refit"]["rounding"] is True
    assert rounded["metadata"]["bfloat16_refit"]["rounding_helper"] == "_round_to_bfloat16"
    assert sorted(rounded["analyses"]) == sorted(baseline["analyses"])
    source = _refit_capture()
    for values in source.features.values():
        assert not np.array_equal(state_probe._round_to_bfloat16(values), values)


def test_refit_comparison_marks_a_flipped_support_flag() -> None:
    """A single perturbed flag is reported as a flip and counted; the rest stay unchanged."""
    baseline = _refit_baseline()
    refit = json.loads(json.dumps(state_probe._json_compliant(baseline)))
    cells = state_probe._support_cells(refit)
    key = next(key for key, entry in cells.items() if "holm_supported" in entry)
    cells[key]["holm_supported"] = not cells[key]["holm_supported"]
    cells[key]["intervals"]["margin_over_position"]["median"] += 0.25

    comparison = state_probe.compare_refit(baseline, refit, rounding=True)

    assert comparison["counts"]["flipped"] == 1
    assert comparison["counts"]["unchanged"] == comparison["counts"]["compared"] - 1
    flip = comparison["flips"][0]
    assert (flip["analysis"], flip["target"], flip["layer"]) == key
    assert flip["flipped_flags"] == ["holm_supported"]
    assert comparison["max_absolute_margin_change"]["value"] == pytest.approx(0.25)
    assert "flipped" in comparison["verdict"]

    unchanged = state_probe.compare_refit(baseline, baseline, rounding=True)
    assert unchanged["counts"]["flipped"] == 0
    assert unchanged["flips"] == []
    assert "supported set unchanged" in unchanged["verdict"].lower()


def test_refit_cli_refuses_a_baseline_whose_data_seed_disagrees(tmp_path) -> None:
    """Fail closed on provenance, like `reanalyse_dataset` itself."""
    capture = state_probe.save_dataset(_refit_capture(data_seed=20260901), tmp_path / "cap.npz")
    baseline_path = tmp_path / "cap.reanalysis.json"
    baseline_path.write_text(
        json.dumps(state_probe._json_compliant(_refit_baseline())), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="data seed"):
        state_probe._main_refit_bf16(
            [
                "--input",
                str(capture),
                "--baseline",
                str(baseline_path),
                "--output",
                str(tmp_path / "refit"),
            ]
        )


def test_refit_cli_writes_both_artifact_pairs_with_a_run_log(monkeypatch, tmp_path, capsys) -> None:
    """R26: identity with both input hashes, per-(analysis, target) progress, four files."""
    import hashlib
    import sys

    capture = state_probe.save_dataset(_refit_capture(), tmp_path / "cap.npz")
    baseline = _refit_baseline()
    baseline_path = tmp_path / "cap.reanalysis.json"
    baseline_path.write_text(json.dumps(state_probe._json_compliant(baseline)), encoding="utf-8")
    output = tmp_path / "refit"
    argv = [
        "state-probe",
        "refit-bf16",
        "--input",
        str(capture),
        "--baseline",
        str(baseline_path),
        "--output",
        str(output),
        "--no-round",
    ]
    monkeypatch.setattr("sys.argv", argv)

    state_probe._main_refit_bf16(sys.argv[2:])

    out = capsys.readouterr().out
    assert (output / "run.log").is_file()
    assert (output / "cap.reanalysis-bf16.json").is_file()
    assert (output / "cap.reanalysis-bf16.md").is_file()
    assert (output / "refit-comparison.json").is_file()
    assert (output / "refit-comparison.md").is_file()

    events = _events(output)
    start = _flat(events[0])
    assert events[0]["kind"] == "start"
    assert events[0]["run"] == "state-probe-refit-bf16"
    assert start["input"] == str(capture)
    assert start["input_sha256"] == hashlib.sha256(capture.read_bytes()).hexdigest()
    assert start["baseline"] == str(baseline_path)
    assert start["baseline_sha256"] == hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    assert start["rounding"] == "off"
    assert isinstance(start["git_commit"], str)

    progress = [item for item in map(_flat, events) if item["kind"] == "progress"]
    assert len(progress) == 2 * len(state_probe.REANALYSIS_TARGETS)
    assert {item["analysis"] for item in progress} == {"all_rows", "sft_disjoint"}
    assert {item["target"] for item in progress} == set(state_probe.REANALYSIS_TARGETS)

    comparison = json.loads((output / "refit-comparison.json").read_text(encoding="utf-8"))
    assert comparison["rounding"] is False
    assert comparison["rounding_helper"] == "_round_to_bfloat16"
    assert comparison["input_sha256"] == start["input_sha256"]
    assert comparison["baseline_sha256"] == start["baseline_sha256"]
    assert comparison["baseline_metadata"]["split_seeds"] == list(_REFIT_SPLIT_SEEDS)
    assert comparison["resolved_generator_version"] == GENERATOR_VERSION
    assert comparison["counts"]["flipped"] == 0
    assert comparison["verdict"] in out
    assert "| max absolute margin change |" in (output / "refit-comparison.md").read_text(
        encoding="utf-8"
    )


# ------------------------------------------------- SPEC-004 §2 / R29 compare and its sidecar


def _sidecar_records(
    *,
    advantage_layer: int | None = None,
    advantage: float = 0.0,
    layers: tuple[int, ...] = (4, 8),
    split_seeds: tuple[int, ...] = (20260903, 20260904),
    cohorts: tuple[str, ...] = ("all_rows", "sft_disjoint"),
    targets: tuple[str, ...] = ("pending_count",),
    task_offset: int = 0,
) -> list[dict[str, Any]]:
    """Per-row prediction records shaped exactly as ``_analyse_cohort`` collects them.

    The truth is a ramp; the position and surface baselines carry a fixed error and the probe
    carries the same error scaled by ``1 - advantage``, so an advantage of one makes the probe
    exact on that layer and leaves every other cell identical between the two sides.
    """
    records: list[dict[str, Any]] = []
    tasks = [f"p2-d0-read-{index + task_offset:04d}-clean" for index in range(12)]
    task_ids = np.array([task for task in tasks for _ in range(2)], dtype=str)
    difficulty = np.array([index % 2 for index, _ in enumerate(tasks) for _ in range(2)])
    actual = np.arange(len(task_ids), dtype=float)
    error = np.array([(-1.0) ** index * (index % 5) for index in range(len(task_ids))])
    for cohort in cohorts:
        for target in targets:
            for split_seed in split_seeds:
                for layer in layers:
                    scale = 1.0 - advantage if layer == advantage_layer else 1.0
                    base = {
                        "cohort": cohort,
                        "target": target,
                        "kind": "regression",
                        "layer": layer,
                        "split_seed": split_seed,
                        "row_ids": np.arange(len(task_ids)),
                        "task_ids": task_ids,
                        "difficulty": difficulty,
                        "actual": actual,
                        "scores": None,
                        "positive_label": None,
                    }
                    records.append(
                        {**base, "control": "probe", "predicted": actual + error * scale}
                    )
                    records.append(
                        {**base, "control": "position", "predicted": actual + error * 2.0}
                    )
                    records.append(
                        {**base, "control": "surface", "predicted": actual + error * 3.0}
                    )
    return records


def _sidecar_metadata(**overrides: Any) -> dict[str, Any]:
    metadata = {
        "schema_version": state_probe.PREDICTION_SIDECAR_SCHEMA_VERSION,
        "split_seeds": [20260903, 20260904],
        "layers": [4, 8],
        "layer_fractions": [0.25, 0.5],
        "generator_version": GENERATOR_VERSION,
        "data_seed": 20260902,
        "cohort_labels": dict(state_probe._COHORT_LABELS),
        "controls": list(state_probe._PREDICTION_CONTROLS),
        # R23 and R35: the basis for the bound version, and the four comparability
        # coordinates `compare` checks beyond R29's five.
        "generator_version_basis": "recorded in the capture metadata",
        "policy": "base",
        "derivative_method": {"kind": "captured_residual", "analysed_position": "last"},
        "prompt_template_kwargs": {"keep_last": 2, "condition": "intact"},
        "estimator_variant": {"fit": {"ridge_alpha": 1.0}},
    }
    metadata.update(overrides)
    return metadata


def _write_sidecar(path: Path, **kwargs: Any) -> dict[str, Any]:
    metadata_overrides = kwargs.pop("metadata", {})
    records = kwargs.pop("records", None)
    if records is None:
        records = _sidecar_records(**kwargs)
    state_probe.write_prediction_sidecar(path, records, _sidecar_metadata(**metadata_overrides))
    return state_probe.load_prediction_sidecar(path)


def _reanalysis_fixture() -> state_probe.ProbeDataset:
    """A small real-generator capture with planted signal, for the sidecar contract tests."""
    from local_llm_lab.pipeline.tasks import difficulty as task_difficulty

    tasks = [*make_tasks("train", 8), *make_tasks("test", 8, perturb=False)]
    difficulties = {}
    for task in tasks:
        split = task.task_id.split("-", 1)[0]
        index = int(task.task_id.rsplit("-", 2)[1])
        difficulties[task.task_id] = task_difficulty(split, index)
    dataset = state_probe.build_label_dataset(tasks, difficulties)
    pending = dataset.labels["pending_count"].astype(float)
    inspect = (dataset.labels["phase"] == "inspect").astype(float)
    rng = np.random.default_rng(417)
    dataset.layers = [0]
    dataset.features = {
        0: np.column_stack(
            [pending, inspect, pending + inspect, rng.normal(size=len(dataset))]
        ).astype(np.float32)
    }
    dataset.meta.update(
        {
            "layers": [0],
            "data_seed": 20260902,
            "keep_last": 2,
            "layer_selection": {"fractions": [0.5], "indices": [0], "num_layers": 2},
        }
    )
    return dataset


def test_reanalysis_predictions_are_optional_and_leave_the_result_unchanged() -> None:
    """R29's sidecar is additive: the JSON result is identical with and without it."""
    dataset = _reanalysis_fixture()
    kwargs: dict[str, Any] = {"split_seeds": (20260903,), "bootstrap_resamples": 2}
    plain = state_probe.reanalyse_dataset(dataset, **kwargs)
    with_rows, predictions = state_probe.reanalyse_dataset(
        dataset, **kwargs, return_predictions=True
    )

    def canonical(result: dict[str, Any]) -> str:
        payload = {key: value for key, value in result.items() if key != "metadata"}
        return json.dumps(state_probe._json_compliant(payload), sort_keys=True, allow_nan=False)

    assert canonical(plain) == canonical(with_rows)
    assert predictions


def test_reanalysis_sidecar_records_test_half_rows_for_every_control(tmp_path) -> None:
    dataset = _reanalysis_fixture()
    _result, predictions = state_probe.reanalyse_dataset(
        dataset,
        split_seeds=(20260903, 20260904),
        bootstrap_resamples=2,
        return_predictions=True,
    )
    path = state_probe.write_prediction_sidecar(
        tmp_path / "capture.predictions.npz", predictions, _sidecar_metadata(layers=[0])
    )
    sidecar = state_probe.load_prediction_sidecar(path)

    controls = {entry["control"] for entry in sidecar["predictions"]}
    assert controls == set(state_probe._PREDICTION_CONTROLS)
    assert {entry["cohort"] for entry in sidecar["predictions"]} == set(state_probe._COHORT_LABELS)
    assert {entry["split_seed"] for entry in sidecar["predictions"]} == {20260903, 20260904}
    group = sidecar["row_groups"][0]
    assert set(group) >= {"row_id", "task_id", "difficulty", "label", "cohort", "target", "kind"}
    assert len(group["row_id"]) == len(group["task_id"]) == len(group["difficulty"])
    # Test-half only: a proper subset of the cohort's rows, and every task id is a real one.
    assert 0 < len(group["row_id"]) < len(dataset)
    assert set(group["task_id"].tolist()) <= set(dataset.task_ids.tolist())
    assert sidecar["metadata"]["generator_version"] == GENERATOR_VERSION


def test_compare_supports_the_planted_layer_only(tmp_path) -> None:
    left = _write_sidecar(tmp_path / "left.predictions.npz")
    right = _write_sidecar(tmp_path / "right.predictions.npz", advantage_layer=8, advantage=1.0)

    results = state_probe.compare_predictions(left, right, resamples=32, seed=20260905)

    cells = results["comparisons"]["all_rows"]["targets"]["pending_count"]["layers"]
    planted = cells["8"]["margin_over_position"]["pooled"]
    flat = cells["4"]["margin_over_position"]["pooled"]
    assert planted["supported"] is True
    assert planted["difference"]["median"] > 0
    assert planted["direction"] == "right"
    assert flat["supported"] is False
    assert cells["8"]["margin_over_surface"]["pooled"]["supported"] is True


def test_compare_uses_one_resample_seed_list_for_every_cell(tmp_path, monkeypatch) -> None:
    left = _write_sidecar(tmp_path / "left.predictions.npz")
    right = _write_sidecar(tmp_path / "right.predictions.npz")
    seen: list[int] = []
    original = state_probe._compare_rng

    def recording(seed: int) -> Any:
        seen.append(int(seed))
        return original(seed)

    monkeypatch.setattr(state_probe, "_compare_rng", recording)
    resamples = 8
    state_probe.compare_predictions(left, right, resamples=resamples, seed=20260905)

    expected = state_probe.compare_resample_seeds(20260905, resamples)
    chunks = [seen[start : start + resamples] for start in range(0, len(seen), resamples)]
    assert len(chunks) > 1
    assert all(chunk == expected for chunk in chunks)


def test_compare_reports_within_difficulty_tables_and_marks_the_reportable_scope(
    tmp_path,
) -> None:
    left = _write_sidecar(tmp_path / "left.predictions.npz")
    right = _write_sidecar(tmp_path / "right.predictions.npz", advantage_layer=8, advantage=1.0)

    results = state_probe.compare_predictions(left, right, resamples=16, seed=20260905)

    disjoint = results["comparisons"]["sft_disjoint"]
    assert disjoint["reportable_scope"] == "by_difficulty"
    assert results["comparisons"]["all_rows"]["reportable_scope"] == "pooled"
    cell = disjoint["targets"]["pending_count"]["layers"]["8"]["margin_over_position"]
    assert set(cell["by_difficulty"]) == {"0", "1"}
    assert all(entry["n_tasks"] > 0 for entry in cell["by_difficulty"].values())
    markdown = state_probe.render_compare_markdown(results)
    assert "pending_count" in markdown
    assert "within difficulty" in markdown


@pytest.mark.parametrize(
    ("overrides", "kwargs", "reason"),
    [
        ({"split_seeds": [20260903]}, {"split_seeds": (20260903,)}, "split seeds"),
        ({"layer_fractions": [0.25, 0.75]}, {}, "layer"),
        ({"generator_version": GENERATOR_VERSION - 1}, {}, "generator version"),
        ({"cohort_labels": {"all_rows": "all_rows"}}, {"cohorts": ("all_rows",)}, "cohort"),
        ({}, {"task_offset": 100}, "task ids"),
    ],
)
def test_compare_refuses_on_every_mismatch(tmp_path, overrides, kwargs, reason) -> None:
    left = _write_sidecar(tmp_path / "left.predictions.npz")
    right = _write_sidecar(tmp_path / "right.predictions.npz", metadata=overrides, **kwargs)

    with pytest.raises(state_probe.CompareRefusal) as raised:
        state_probe.compare_predictions(left, right, resamples=4, seed=20260905)

    assert any(reason in item for item in raised.value.reasons)


def test_compare_cli_refuses_a_missing_sidecar_and_names_the_recompute_fallback(
    tmp_path, monkeypatch, capsys
) -> None:
    left_json = tmp_path / "left" / "a.reanalysis.json"
    right_json = tmp_path / "right" / "b.reanalysis.json"
    for path in (left_json, right_json):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    state_probe.write_prediction_sidecar(
        tmp_path / "left" / "a.predictions.npz", _sidecar_records(), _sidecar_metadata()
    )
    output = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "state-probe",
            "compare",
            "--left",
            str(left_json),
            "--right",
            str(right_json),
            "--output",
            str(output),
        ],
    )

    with pytest.raises(SystemExit) as raised:
        state_probe.main()

    assert raised.value.code == 2
    report = json.loads((output / "compare.json").read_text(encoding="utf-8"))
    assert report["refused"] is True
    assert any("b.predictions.npz" in reason for reason in report["reasons"])
    fallback = " ".join(report["reasons"])
    assert "recompute" in fallback and ".npz" in fallback
    assert "refused" in (output / "compare.md").read_text(encoding="utf-8")
    assert "refused" in capsys.readouterr().out


def test_compare_cli_writes_the_pair_of_reports_and_a_run_log(
    tmp_path, monkeypatch, capsys
) -> None:
    import hashlib

    paths = {}
    for side, kwargs in (("left", {}), ("right", {"advantage_layer": 8, "advantage": 1.0})):
        directory = tmp_path / side
        directory.mkdir()
        json_path = directory / f"{side}.reanalysis.json"
        json_path.write_text("{}\n", encoding="utf-8")
        state_probe.write_prediction_sidecar(
            directory / f"{side}.predictions.npz",
            _sidecar_records(**kwargs),
            _sidecar_metadata(),
        )
        paths[side] = json_path
    output = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "state-probe",
            "compare",
            "--left",
            str(paths["left"]),
            "--right",
            str(paths["right"]),
            "--output",
            str(output),
            "--resamples",
            "16",
            "--seed",
            "20260905",
        ],
    )

    state_probe.main()

    out = capsys.readouterr().out
    report = json.loads((output / "compare.json").read_text(encoding="utf-8"))
    assert report["refused"] is False
    assert report["metadata"]["resamples"] == 16
    assert report["metadata"]["bootstrap_unit"] == "task_id"
    events = _events(output)
    start = _flat(events[0])
    assert events[0]["run"] == "state-probe-compare"
    assert start["left_sha256"] == hashlib.sha256(b"{}\n").hexdigest()
    assert start["left_predictions_sha256"] and start["right_predictions_sha256"]
    assert "# P2 paired comparison" in out
    assert (output / "compare.md").is_file()


# ----------------- R29 addendum: Holm across the comparison cells (SPEC-004 §1.6 via §2)


def _compare_cells(results: dict[str, Any], cohort: str = "all_rows") -> list[dict[str, Any]]:
    """Every difference cell the comparison produced, pooled and within difficulty."""
    cells: list[dict[str, Any]] = []
    for entry in results["comparisons"][cohort]["targets"].values():
        for controls in entry["layers"].values():
            for margin in state_probe._COMPARE_MARGIN_KEYS.values():
                cells.append(controls[margin]["pooled"])
                cells.extend(controls[margin]["by_difficulty"].values())
    return cells


def test_compare_writes_holm_flags_and_records_the_family_on_every_cell(tmp_path) -> None:
    """R29 addendum: the comparison cells carry a Holm family, not only an interval flag."""
    left = _write_sidecar(tmp_path / "left.predictions.npz")
    right = _write_sidecar(tmp_path / "right.predictions.npz", advantage_layer=8, advantage=1.0)

    results = state_probe.compare_predictions(left, right, resamples=32)

    cells = _compare_cells(results)
    assert cells
    for cell in cells:
        assert set(cell) >= {"holm_adjusted_p", "holm_supported", "holm_family", "holm_family_size"}
    block = results["metadata"]["multiple_comparisons"]
    assert block["method"] == "Holm"
    assert block["family"] == "cohort x scope, spanning every (target, layer) cell in that scope"
    assert block["family"] == state_probe.COMPARE_HOLM_FAMILY
    families = results["comparisons"]["all_rows"]["multiple_comparisons"]
    # One target x two layers; each cell is a single member, its two controls collapsed.
    assert families["pooled"]["family"] == "all_rows/pooled"
    assert families["pooled"]["tested_cells"] == 2
    assert {
        cell["holm_family_size"] for cell in cells if cell["holm_family"].endswith("/pooled")
    } == {families["pooled"]["tested_cells"]}


def test_compare_holm_support_is_stricter_than_the_interval_flag(tmp_path) -> None:
    """A cell whose interval excludes zero is not Holm-supported at too few resamples."""
    left = _write_sidecar(tmp_path / "left.predictions.npz")
    right = _write_sidecar(tmp_path / "right.predictions.npz", advantage_layer=8, advantage=1.0)

    coarse = state_probe.compare_predictions(left, right, resamples=32)
    planted = coarse["comparisons"]["all_rows"]["targets"]["pending_count"]["layers"]["8"][
        "margin_over_position"
    ]["pooled"]
    assert planted["supported"] is True
    assert planted["difference"]["lower"] > 0
    assert planted["holm_supported"] is False
    assert planted["holm_adjusted_p"] > state_probe.COMPARE_HOLM_ALPHA

    fine = state_probe.compare_predictions(left, right, resamples=512)
    resolved = fine["comparisons"]["all_rows"]["targets"]["pending_count"]["layers"]["8"][
        "margin_over_position"
    ]["pooled"]
    # The interval flag is unchanged by the Holm addition; only the extra flag moves.
    assert resolved["supported"] is True
    assert resolved["holm_supported"] is True
    assert resolved["holm_adjusted_p"] <= state_probe.COMPARE_HOLM_ALPHA


def test_compare_markdown_renders_the_holm_support_column(tmp_path) -> None:
    left = _write_sidecar(tmp_path / "left.predictions.npz")
    right = _write_sidecar(tmp_path / "right.predictions.npz", advantage_layer=8, advantage=1.0)

    results = state_probe.compare_predictions(left, right, resamples=32)
    markdown = state_probe.render_compare_markdown(results)

    header = next(line for line in markdown.splitlines() if line.startswith("| layer |"))
    assert "Holm support" in header
    assert header.index("supported") < header.index("Holm support")
    lines = markdown.splitlines()
    start = lines.index(header)
    body = [
        line
        for line in lines[start + 2 :]
        if line.startswith("|") and line.split("|")[1].strip().isdigit()
    ]
    assert body
    assert all(line.count("|") == header.count("|") for line in body)
    column = header.split("|").index(" Holm support ")
    values = {line.split("|")[column].strip() for line in body}
    # At 32 resamples nothing clears Holm, and the within-difficulty scopes fall under R8's
    # row floor, so both the flag and the n/a form appear.
    assert "no" in values
    assert all(value in {"yes", "no"} or value.startswith("n/a (n=") for value in values)


def test_compare_and_the_reanalysis_form_the_same_holm_families() -> None:
    """The ruled family: both tools partition one synthetic set of cells identically.

    ``compare`` corrects over the family ``_analyse_cohort`` already uses, so the two tools
    cannot disagree about what a family is. The partitions are compared as sets of cell
    groupings, so this fails if either tool's shape changes: put the target back into either
    key and one side's grouping splits; merge ``compare``'s scopes and its member counts move.
    """
    cells = [
        (target, layer)
        for target in ("pending_count", "first_bucket_count")
        for layer in (4, 8)
    ]
    scopes = (state_probe._REANALYSIS_HOLM_SCOPE, "0", "1")

    reanalysis = state_probe._cohort_holm_families(
        [(target, layer, {"cell": (target, layer)}) for target, layer in cells]
    )
    compare = state_probe._compare_holm_families(
        [
            (target, layer, scope, [{"cell": (target, layer)}, {"cell": (target, layer)}])
            for scope in scopes
            for target, layer in cells
        ]
    )

    # The reanalysis takes one p per (target, layer) from the pooled margins, so it has a
    # single scope; `compare` takes a p per scope, so each scope is a family of its own.
    assert set(reanalysis) == {state_probe._REANALYSIS_HOLM_SCOPE}
    assert set(compare) == set(scopes)
    # No family is merged across scopes, and none is split by target or by layer.
    assert [len(members) for members in compare.values()] == [len(cells)] * len(scopes)
    reanalysis_groupings = {
        frozenset(entry["cell"] for entry in members) for members in reanalysis.values()
    }
    compare_groupings = {
        frozenset(controls[0]["cell"] for controls in members) for members in compare.values()
    }
    assert compare_groupings == reanalysis_groupings
    assert reanalysis_groupings == {frozenset(cells)}


def test_compare_holm_collapses_a_cell_to_the_larger_of_its_two_control_p_values() -> None:
    """A difference enters its family only if it holds against *both* baselines.

    This is ``_analyse_cohort``'s ``max(p_position, p_surface)``: collapsing by the smaller p
    would clear alpha on both cells here, and collapsing by the larger clears neither.
    """
    raw = {
        (4, "position"): 0.01,
        (4, "surface"): 0.30,
        (8, "position"): 0.02,
        (8, "surface"): 0.04,
    }
    cells = [
        (
            "pending_count",
            layer,
            "pooled",
            [
                {
                    "supported": True,
                    "raw_p": raw[(layer, control)],
                    "holm_eligible": True,
                    "holm_ineligible_reason": None,
                }
                for control in state_probe._COMPARE_CONTROLS
            ],
        )
        for layer in (4, 8)
    ]

    record = state_probe._apply_compare_holm(cells, cohort="all_rows")

    assert record["pooled"] == {
        "family": "all_rows/pooled",
        "tested_cells": 2,
        "excluded_cells": 0,
    }
    names = state_probe._COMPARE_CONTROLS
    larger = [max(raw[(layer, name)] for name in names) for layer in (4, 8)]
    smaller = [min(raw[(layer, name)] for name in names) for layer in (4, 8)]
    expected = state_probe._holm_adjust(larger)
    for (_target, _layer, _scope, controls), adjusted in zip(cells, expected, strict=True):
        # One adjusted p per cell, carried on both of its control rows.
        assert {control["holm_adjusted_p"] for control in controls} == {adjusted}
    alpha = state_probe.COMPARE_HOLM_ALPHA
    flags = [control["holm_supported"] for _t, _l, _s, controls in cells for control in controls]
    assert flags == [False] * len(flags)
    assert all(value <= alpha for value in state_probe._holm_adjust(smaller))


def test_compare_holm_excludes_a_whole_cell_when_r8_excludes_either_control() -> None:
    """R8's minimum governs membership by cell, and an excluded cell keeps its reason."""
    counts = {"n_tasks": 6, "n_rows_left": 12, "n_rows_right": 12}
    _eligible, floor = state_probe._compare_cell_eligibility(counts, 0.4)
    kept = {
        "supported": True,
        "raw_p": 0.01,
        "holm_eligible": True,
        "holm_ineligible_reason": None,
    }
    dropped = {
        "supported": True,
        "raw_p": 0.4,
        "holm_eligible": False,
        "holm_ineligible_reason": floor,
    }

    cells = [("pending_count", 4, "pooled", [kept, dropped])]

    record = state_probe._apply_compare_holm(cells, cohort="all_rows")

    assert record["pooled"] == {"family": "all_rows/pooled", "tested_cells": 0, "excluded_cells": 1}
    assert kept["holm_eligible"] is False
    assert kept["holm_adjusted_p"] is None
    assert kept["holm_supported"] is False
    assert floor in kept["holm_ineligible_reason"]
    assert dropped["holm_ineligible_reason"] == floor


def test_compare_holm_family_spans_every_target_of_the_cohort(tmp_path) -> None:
    """One family per (cohort, scope): every target's cells are corrected together."""
    targets = ("pending_count", "first_bucket_count")
    left = _write_sidecar(tmp_path / "left.predictions.npz", targets=targets)
    right = _write_sidecar(
        tmp_path / "right.predictions.npz", targets=targets, advantage_layer=8, advantage=1.0
    )

    results = state_probe.compare_predictions(left, right, resamples=32)

    analysis = results["comparisons"]["all_rows"]
    families = analysis["multiple_comparisons"]
    assert set(analysis["targets"]) == set(targets)
    assert families["pooled"]["family"] == "all_rows/pooled"
    # Two targets x two layers, each contributing one member rather than one per control.
    assert families["pooled"]["tested_cells"] == len(targets) * 2
    for target in targets:
        for controls in analysis["targets"][target]["layers"].values():
            rows = [
                controls[margin]["pooled"] for margin in state_probe._COMPARE_MARGIN_KEYS.values()
            ]
            assert {row["holm_family"] for row in rows} == {"all_rows/pooled"}
            assert {row["holm_family_size"] for row in rows} == {families["pooled"]["tested_cells"]}
            assert len({row["holm_adjusted_p"] for row in rows}) == 1
    # Either target's table states the same family, in the same words.
    markdown = state_probe.render_compare_markdown(results)
    stated = [line for line in markdown.splitlines() if "Holm family `all_rows/pooled`" in line]
    assert len(stated) == len(targets)
    assert len(set(stated)) == 1


# -------------------------------------------------- R35 cross-model comparability dimensions


@pytest.mark.parametrize(
    ("dimension", "override", "refuses"),
    [
        ("policy", "B", False),
        (
            "derivative_method",
            {"kind": "captured_residual", "analysed_position": "note_mean"},
            True,
        ),
        ("prompt_template_kwargs", {"keep_last": 0, "condition": "both"}, True),
        ("estimator_variant", {"fit": {"ridge_alpha": 2.0}}, True),
    ],
)
def test_compare_names_every_r35_dimension(tmp_path, dimension, override, refuses) -> None:
    """R35: each added dimension is either a named refusal or a named recorded difference."""
    left = _write_sidecar(tmp_path / "left.predictions.npz")
    right = _write_sidecar(tmp_path / "right.predictions.npz", metadata={dimension: override})

    if refuses:
        with pytest.raises(state_probe.CompareRefusal) as raised:
            state_probe.compare_predictions(left, right, resamples=4)
        assert any(dimension in reason for reason in raised.value.reasons)
        return

    results = state_probe.compare_predictions(left, right, resamples=4)
    comparability = results["metadata"]["comparability"]
    assert [entry["dimension"] for entry in comparability["named_differences"]] == [dimension]
    named = comparability["named_differences"][0]
    assert named["left"] == _sidecar_metadata()[dimension]
    assert named["right"] == override
    assert dimension in state_probe.render_compare_markdown(results)


def test_compare_records_all_nine_comparability_dimensions(tmp_path) -> None:
    """The five R29 refusal dimensions plus the four R35 ones, named in the artifact."""
    left = _write_sidecar(tmp_path / "left.predictions.npz")
    right = _write_sidecar(tmp_path / "right.predictions.npz")

    results = state_probe.compare_predictions(left, right, resamples=4)

    dimensions = results["metadata"]["comparability"]["dimensions"]
    assert [entry["dimension"] for entry in dimensions] == [
        "split_seeds",
        "layer_selection",
        "generator_version",
        "cohort_labels",
        "task_ids",
        "policy",
        "derivative_method",
        "prompt_template_kwargs",
        "estimator_variant",
    ]
    assert all(entry["equal"] for entry in dimensions)
    assert results["metadata"]["comparability"]["named_differences"] == []


# ------------------------------------------------- R23: the basis for the bound version


def test_reanalyse_metadata_records_the_generator_version_basis() -> None:
    dataset = _reanalysis_fixture()

    results = state_probe.reanalyse_dataset(dataset, split_seeds=(20260903,), bootstrap_resamples=2)

    assert results["metadata"]["generator_version"] == GENERATOR_VERSION
    assert "recorded" in results["metadata"]["generator_version_basis"]


def test_reanalyse_metadata_names_an_explicit_binding_as_its_own_basis() -> None:
    dataset = _reanalysis_fixture()
    dataset.meta.pop("generator_version", None)

    results = state_probe.reanalyse_dataset(
        dataset, split_seeds=(20260903,), bootstrap_resamples=2, generator_version=1
    )

    basis = results["metadata"]["generator_version_basis"]
    assert "R23" in basis and "explicit" in basis


def test_compare_artifact_carries_both_sides_generator_version_basis(tmp_path) -> None:
    left = _write_sidecar(tmp_path / "left.predictions.npz")
    right = _write_sidecar(tmp_path / "right.predictions.npz")

    results = state_probe.compare_predictions(left, right, resamples=4)

    basis = results["metadata"]["generator_version_basis"]
    assert basis["left"] == basis["right"] == _sidecar_metadata()["generator_version_basis"]
    assert basis["left"] in state_probe.render_compare_markdown(results)


def test_reanalyse_cli_stamps_the_r35_coordinates_and_basis_on_both_artifacts(
    monkeypatch, tmp_path, capsys
) -> None:
    """The reanalysis JSON and its R29 sidecar both name what `compare` will check."""
    import sys

    capture_path = tmp_path / "cap.npz"
    state_probe.save_dataset(_refit_capture(), capture_path)
    output = tmp_path / "result"
    monkeypatch.setattr(
        "sys.argv",
        [
            "state-probe",
            "reanalyse",
            "--input",
            str(capture_path),
            "--output",
            str(output),
            "--split-seeds",
            ",".join(map(str, _REFIT_SPLIT_SEEDS)),
            "--bootstrap-resamples",
            str(_REFIT_RESAMPLES),
        ],
    )

    state_probe._main_reanalyse(sys.argv[2:])
    capsys.readouterr()

    payload = json.loads((output / "cap.reanalysis.json").read_text(encoding="utf-8"))
    assert "recorded" in payload["metadata"]["generator_version_basis"]
    assert set(payload["metadata"]["r35"]) == set(state_probe.R35_DIMENSIONS)
    sidecar = state_probe.load_prediction_sidecar(output / "cap.predictions.npz")
    metadata = sidecar["metadata"]
    assert metadata["generator_version_basis"] == payload["metadata"]["generator_version_basis"]
    for dimension in state_probe.R35_DIMENSIONS:
        assert metadata[dimension] == payload["metadata"]["r35"][dimension]
    assert metadata["derivative_method"]["analysed_position"] == "last"
    assert metadata["estimator_variant"]["fit"] == payload["metadata"]["fit"]


# ------------------------- B1b: P2 conditions, dual capture positions, capture dtype (R18b)


# The generator's own default data seed, read rather than restated (briefing rule 1.8).
_GENERATOR_DATA_SEED = int(inspect.signature(make_tasks).parameters["seed"].default)
_NOTE_TEXT = "Pending 2 of 6: alpha, beta."
_TARGET_TURN = _NOTE_TEXT + '\n```json\n{"name": "read_file", "arguments": {"path": "a.txt"}}\n```'


class _SpanTokenizer:
    """Character-level ids, so every prompt is a strict token prefix of prompt + note."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]


class _MergingSpanTokenizer(_SpanTokenizer):
    """Characters merge pairwise: an odd-length prompt breaks the prefix property."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        pairs = [text[index : index + 2] for index in range(0, len(text), 2)]
        return [sum(map(ord, pair)) for pair in pairs]


class _Runtime(SimpleNamespace):
    def __init__(self) -> None:
        super().__init__()

    def eval(self, *values: Any) -> None:
        del values

    def clear_cache(self) -> None:
        return None

    def reset_peak_memory(self) -> None:
        return None

    def get_active_memory(self) -> int:
        return 0

    def get_cache_memory(self) -> int:
        return 0

    def get_peak_memory(self) -> int:
        return 0


def _token_capture(_model, token_ids, layers, *, positions="last", dtype="float32"):
    """One feature per token, equal to the token id, so a pooled mean is checkable by hand."""
    del dtype
    values = np.array([[float(token)] for token in token_ids], dtype=np.float32)
    picked = values[-1] if positions == "last" else values
    return {layer: picked + float(layer) for layer in layers}


def _one_row(prompt: str = "PROMPT", turn: str = _TARGET_TURN):
    return [
        {
            "messages": [
                {"role": "user", "content": "u"},
                {"role": "tool", "name": "read_file", "content": "observed"},
                {"role": "assistant", "content": turn},
            ],
            "metadata": {"step": 0},
        }
    ]


def _single_task() -> list[Any]:
    """One real generated task; the replay itself is stubbed by ``_install_single_row``."""
    return make_tasks("test", 1)[:1]


def _install_single_row(monkeypatch, *, prompt: str = "PROMPT", turn: str = _TARGET_TURN) -> None:
    monkeypatch.setattr(state_probe, "build_rows", lambda *_a, **_k: _one_row(turn=turn))
    monkeypatch.setattr(state_probe, "build_prompt", lambda *_a, **_k: prompt)
    monkeypatch.setattr(
        state_probe, "row_labels", lambda *_a: {name: 0.0 for name in state_probe.TARGETS}
    )
    monkeypatch.setattr(state_probe, "mlx_memory_snapshot", lambda *_a: {})


# --------------------------------------------------------------- conditions and difficulties


def test_capture_condition_names_the_four_spec_conditions() -> None:
    """SPEC-004 §2: the (--strip, --stub-observations) pair names one of four conditions."""
    assert state_probe.capture_condition(False, False) == "intact"
    assert state_probe.capture_condition(True, False) == "notes-stripped"
    assert state_probe.capture_condition(False, True) == "observations-stubbed"
    assert state_probe.capture_condition(True, True) == "both"
    assert set(state_probe.CONDITIONS.values()) == {
        "intact",
        "notes-stripped",
        "observations-stubbed",
        "both",
    }


@pytest.mark.parametrize("split", ["train", "valid", "test", "p2mix"])
def test_task_difficulties_is_unchanged_for_the_legacy_split_names(split: str) -> None:
    """The fix must not move a single legacy label: same ids, same levels, same order."""
    from local_llm_lab.pipeline.tasks import difficulty

    tasks = make_tasks(split, 8)
    legacy = {task.task_id: int(difficulty(split, index)) for index, task in enumerate(tasks)}

    assert state_probe.task_difficulties(split, tasks) == legacy


@pytest.mark.parametrize(("split", "level"), [("p2-d0", 0), ("p2-d1", 1), ("p2-d2", 2)])
def test_task_difficulties_reads_the_p2_splits_own_difficulty(split: str, level: int) -> None:
    """Known defect: ``difficulty(split, index)`` returns ``index % 2`` on an unknown split."""
    from local_llm_lab.pipeline.tasks import difficulty, make_p2_tasks

    tasks = make_p2_tasks(split, 6, _GENERATOR_DATA_SEED)
    levels = state_probe.task_difficulties(split, tasks)

    assert set(levels.values()) == {level}
    assert all(task.difficulty == level for task in tasks)
    if level != 2:  # the defect's signature: the positional formula disagrees on p2-d0/p2-d1
        assert [difficulty(split, index) for index in range(len(tasks))] != [level] * len(tasks)


def test_task_difficulties_falls_back_to_the_positional_rule_for_unlabelled_tasks() -> None:
    tasks = [SimpleNamespace(task_id=f"t{index}", difficulty=-1) for index in range(4)]

    assert state_probe.task_difficulties("p2mix", tasks) == {"t0": 0, "t1": 1, "t2": 0, "t3": 1}


def test_build_probe_dataset_stub_observations_hides_every_observation(monkeypatch) -> None:
    """``--stub-observations`` is ``keep_last = 0`` (Chief, 2026-09-05 05:40 decision 4)."""
    from local_llm_lab.pipeline.tasks import make_tasks as real_make_tasks

    seen: list[int] = []
    monkeypatch.setattr(
        state_probe,
        "build_rows",
        lambda _task, *, keep_last: seen.append(keep_last) or _one_row(),
    )
    monkeypatch.setattr(state_probe, "build_prompt", lambda *_a, **_k: "PROMPT")
    monkeypatch.setattr(
        state_probe, "row_labels", lambda *_a: {name: 0.0 for name in state_probe.TARGETS}
    )
    monkeypatch.setattr(state_probe, "mlx_memory_snapshot", lambda *_a: {})
    monkeypatch.setattr(state_probe, "capture_residuals", _token_capture)
    task = real_make_tasks("test", 1)[0]

    dataset = state_probe.build_probe_dataset(
        object(),
        _SpanTokenizer(),
        [task],
        [1],
        stub_observations=True,
        mlx_runtime=_Runtime(),
    )

    assert seen == [0]
    assert dataset.meta["keep_last"] == 0
    assert dataset.meta["stub_observations"] is True
    assert dataset.meta["condition"] == "observations-stubbed"


def test_build_probe_dataset_records_the_both_condition(monkeypatch) -> None:
    from local_llm_lab.pipeline.tasks import make_tasks as real_make_tasks

    _install_single_row(monkeypatch)
    monkeypatch.setattr(state_probe, "capture_residuals", _token_capture)

    dataset = state_probe.build_probe_dataset(
        object(),
        _SpanTokenizer(),
        real_make_tasks("test", 1)[:1],
        [1],
        True,
        stub_observations=True,
        mlx_runtime=_Runtime(),
    )

    assert dataset.meta["condition"] == "both"
    assert dataset.meta["strip"] is True


# ------------------------------------------------------------------- dual capture positions


def test_note_mean_averages_exactly_the_expert_note_tokens(monkeypatch) -> None:
    """SPEC-004 §2: the second position is the mean over the teacher-forced note tokens."""
    from local_llm_lab.probes.capture import note_token_span

    _install_single_row(monkeypatch)
    monkeypatch.setattr(state_probe, "capture_residuals", _token_capture)
    tokenizer = _SpanTokenizer()

    dataset = state_probe.build_probe_dataset(
        object(),
        tokenizer,
        _single_task(),
        [1, 3],
        capture_positions=("last", "note_mean"),
        mlx_runtime=_Runtime(),
    )

    joint, start, repairs = note_token_span(tokenizer, "PROMPT", _NOTE_TEXT)
    assert repairs == 0
    assert joint == tokenizer.encode("PROMPT" + _NOTE_TEXT)
    expected = float(np.mean([float(token) for token in joint[start:]]))
    assert dataset.note_features is not None
    for layer in (1, 3):
        assert dataset.note_features[layer].shape == (1, 1)
        assert dataset.note_features[layer][0, 0] == pytest.approx(expected + layer)
        # the last-token capture still sees the prompt alone, and is a different number
        assert dataset.features[layer][0, 0] == pytest.approx(
            float(tokenizer.encode("PROMPT")[-1]) + layer
        )
        assert dataset.features[layer][0, 0] != dataset.note_features[layer][0, 0]
    assert dataset.meta["capture_positions"] == ["last", "note_mean"]
    assert dataset.meta["note_token_counts"] == [len(joint) - start]
    assert dataset.meta["note_boundary_repairs"] == 0


def test_note_mean_widens_over_a_merged_tokenizer_boundary(monkeypatch) -> None:
    """Trap 6 (briefing §4): the boundary token merges, so the span widens onto the NATURAL
    joint sequence by exactly one token, and the repair is counted."""
    from local_llm_lab.probes.capture import note_token_span

    prompt = "odd"
    _install_single_row(monkeypatch, prompt=prompt)
    captured_ids: list[list[int]] = []

    def recording_capture(model, token_ids, layers, *, positions="last", dtype="float32"):
        captured_ids.append(list(token_ids))
        return _token_capture(model, token_ids, layers, positions=positions, dtype=dtype)

    monkeypatch.setattr(state_probe, "capture_residuals", recording_capture)
    tokenizer = _MergingSpanTokenizer()

    dataset = state_probe.build_probe_dataset(
        object(),
        tokenizer,
        _single_task(),
        [1],
        capture_positions=("last", "note_mean"),
        mlx_runtime=_Runtime(),
    )

    natural = tokenizer.encode(prompt + _NOTE_TEXT)
    joint, start, repairs = note_token_span(tokenizer, prompt, _NOTE_TEXT)
    assert (joint, repairs) == (natural, 1)
    # The sequence the model was actually run on is the natural one, not a rebuilt pair.
    assert captured_ids[-1] == natural
    assert natural != [*tokenizer.encode(prompt), *tokenizer.encode(_NOTE_TEXT)]
    # Widened by exactly one token, and that token is the merged boundary.
    assert start == len(tokenizer.encode(prompt)) - 1
    assert natural[start] == ord(prompt[-1]) + ord(_NOTE_TEXT[0])

    assert dataset.meta["note_boundary_repairs"] == 1
    assert dataset.meta["note_token_counts"] == [len(natural) - start]
    assert dataset.note_features is not None
    # The pooled mean covers exactly the widened span, merged boundary token included.
    assert dataset.note_features[1][0, 0] == pytest.approx(
        float(np.mean([float(token) for token in natural[start:]])) + 1.0
    )


def test_note_mean_refuses_a_tokenizer_that_is_not_prefix_stable(monkeypatch) -> None:
    """A boundary that would move more than one token raises rather than widen silently."""

    class _Unstable(_SpanTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            offset = 1 if len(text) > len("PROMPT") else 0
            return [ord(character) + offset for character in text]

    _install_single_row(monkeypatch)
    monkeypatch.setattr(state_probe, "capture_residuals", _token_capture)

    with pytest.raises(ValueError, match="more than one token"):
        state_probe.build_probe_dataset(
            object(),
            _Unstable(),
            _single_task(),
            [1],
            capture_positions=("last", "note_mean"),
            mlx_runtime=_Runtime(),
        )


def test_note_mean_pools_the_unstripped_target_note(monkeypatch) -> None:
    """--strip rewrites the context's notes, never the target whose residual is under test."""
    from local_llm_lab.probes.capture import note_token_span

    _install_single_row(monkeypatch)
    monkeypatch.setattr(state_probe, "capture_residuals", _token_capture)
    tokenizer = _SpanTokenizer()

    stripped = state_probe.build_probe_dataset(
        object(),
        tokenizer,
        _single_task(),
        [1],
        True,
        capture_positions=("last", "note_mean"),
        mlx_runtime=_Runtime(),
    )

    joint, start, _repairs = note_token_span(tokenizer, "PROMPT", _NOTE_TEXT)
    assert stripped.meta["condition"] == "notes-stripped"
    assert stripped.meta["note_token_counts"] == [len(joint) - start]
    assert stripped.note_features is not None
    assert stripped.note_features[1][0, 0] == pytest.approx(
        float(np.mean([float(token) for token in joint[start:]])) + 1.0
    )


def test_reanalyse_cli_names_the_position_and_keeps_the_legacy_stem(
    monkeypatch, tmp_path, capsys
) -> None:
    """``--position`` defaults to ``last``, whose artifact name is exactly the ratified one."""
    import sys

    dataset = _refit_capture()
    dataset.note_features = {layer: values + 1.0 for layer, values in dataset.features.items()}
    capture_path = tmp_path / "cap.npz"
    state_probe.save_dataset(dataset, capture_path)
    output = tmp_path / "result"

    for position, stem in (("last", "cap.reanalysis"), ("note_mean", "cap.reanalysis.note_mean")):
        monkeypatch.setattr(
            "sys.argv",
            [
                "state-probe",
                "reanalyse",
                "--input",
                str(capture_path),
                "--output",
                str(output),
                "--split-seeds",
                ",".join(map(str, _REFIT_SPLIT_SEEDS)),
                "--bootstrap-resamples",
                str(_REFIT_RESAMPLES),
                "--position",
                position,
            ],
        )
        state_probe._main_reanalyse(sys.argv[2:])
        capsys.readouterr()
        payload = json.loads((output / f"{stem}.json").read_text(encoding="utf-8"))
        assert payload["metadata"]["position"] == position


def test_expert_note_is_the_text_before_the_tool_call() -> None:
    assert state_probe.expert_note(_TARGET_TURN) == _NOTE_TEXT
    # An unparseable target keeps its whole content rather than losing the row.
    assert state_probe.expert_note("no call here") == "no call here"


def test_capture_positions_default_to_the_last_token_alone(monkeypatch) -> None:
    """Default ``("last",)``: every existing artifact and caller is unchanged."""
    _install_single_row(monkeypatch)
    monkeypatch.setattr(state_probe, "capture_residuals", _token_capture)

    dataset = state_probe.build_probe_dataset(
        object(),
        _SpanTokenizer(),
        _single_task(),
        [1],
        mlx_runtime=_Runtime(),
    )

    assert dataset.note_features is None
    assert dataset.meta["capture_positions"] == ["last"]


def test_dataset_round_trip_carries_the_note_mean_features(tmp_path) -> None:
    dataset = state_probe.ProbeDataset(
        layers=[1, 2],
        features={
            1: np.array([[1.0], [2.0]], dtype=np.float32),
            2: np.array([[3.0], [4.0]], dtype=np.float32),
        },
        note_features={
            1: np.array([[5.0], [6.0]], dtype=np.float32),
            2: np.array([[7.0], [8.0]], dtype=np.float32),
        },
        labels={},
        task_ids=np.array(["test-read-0000-clean", "test-read-0001-clean"]),
        meta={"layers": [1, 2], "capture_positions": ["last", "note_mean"]},
    )

    reloaded = state_probe.load_dataset(state_probe.save_dataset(dataset, tmp_path / "dual.npz"))

    assert reloaded.note_features is not None
    for layer in (1, 2):
        np.testing.assert_array_equal(reloaded.features[layer], dataset.features[layer])
        np.testing.assert_array_equal(reloaded.note_features[layer], dataset.note_features[layer])


def test_legacy_single_position_artifacts_still_load(tmp_path) -> None:
    dataset = state_probe.ProbeDataset(
        layers=[1],
        features={1: np.array([[1.0]], dtype=np.float32)},
        labels={},
        task_ids=np.array(["test-read-0000-clean"]),
        meta={"layers": [1]},
    )

    reloaded = state_probe.load_dataset(state_probe.save_dataset(dataset, tmp_path / "old.npz"))

    assert reloaded.note_features is None
    with pytest.raises(ValueError, match="note_mean"):
        reloaded.at_position("note_mean")


def test_at_position_selects_the_requested_feature_set() -> None:
    dataset = state_probe.ProbeDataset(
        layers=[1],
        features={1: np.array([[1.0]], dtype=np.float32)},
        note_features={1: np.array([[9.0]], dtype=np.float32)},
        labels={},
        task_ids=np.array(["test-read-0000-clean"]),
        meta={"layers": [1]},
    )

    assert dataset.at_position("last") is dataset
    assert dataset.at_position().features[1][0, 0] == 1.0
    assert dataset.at_position("note_mean").features[1][0, 0] == 9.0
    with pytest.raises(ValueError, match="position"):
        dataset.at_position("first")


def test_fit_probes_and_reanalyse_read_the_selected_position() -> None:
    """Default ``last`` leaves every existing artifact unchanged; ``note_mean`` uses the pool."""
    dataset = _refit_capture()
    dataset.note_features = {layer: values + 1.0 for layer, values in dataset.features.items()}

    last = state_probe.reanalyse_dataset(
        dataset, split_seeds=_REFIT_SPLIT_SEEDS, bootstrap_resamples=_REFIT_RESAMPLES
    )
    note = state_probe.reanalyse_dataset(
        dataset,
        split_seeds=_REFIT_SPLIT_SEEDS,
        bootstrap_resamples=_REFIT_RESAMPLES,
        position="note_mean",
    )

    assert last["metadata"]["position"] == "last"
    assert note["metadata"]["position"] == "note_mean"
    shifted = state_probe.ProbeDataset(
        layers=list(dataset.layers),
        features=dict(dataset.note_features),
        labels=dict(dataset.labels),
        task_ids=dataset.task_ids,
        family=dataset.family,
        step_index=dataset.step_index,
        difficulty=dataset.difficulty,
        meta=dict(dataset.meta),
    )
    control = state_probe.reanalyse_dataset(
        shifted, split_seeds=_REFIT_SPLIT_SEEDS, bootstrap_resamples=_REFIT_RESAMPLES
    )
    assert _canonical(note["analyses"]) == _canonical(control["analyses"])

    fitted = state_probe.fit_probes(dataset, ["pending_count"], [_REFIT_LAYERS[0]], 3)
    fitted_note = state_probe.fit_probes(
        dataset, ["pending_count"], [_REFIT_LAYERS[0]], 3, position="note_mean"
    )
    assert fitted["position"] == "last"
    assert fitted_note["position"] == "note_mean"


# ----------------------------------------------------------------------- capture dtype (R18b)


def _bfloat16_capture(_model, token_ids, layers, *, positions="last", dtype="float32"):
    """Blocks that emit bfloat16; ``float32`` upcasts before anything is pooled."""
    import mlx.core as mx

    values = mx.array(
        np.array([[float(token) / 3.0] for token in token_ids], dtype=np.float32)
    ).astype(mx.bfloat16)
    if dtype == "float32":
        values = values.astype(mx.float32)
    picked = values[-1] if positions == "last" else values
    return {layer: picked for layer in layers}


@pytest.mark.parametrize("capture_dtype", ["native", "float32"])
def test_capture_dtype_stores_float32_and_records_itself(monkeypatch, capture_dtype: str) -> None:
    """Rule 1.5 as amended by R18: native blocks, float32 only at storage."""
    import mlx.core as mx

    _install_single_row(monkeypatch)
    monkeypatch.setattr(state_probe, "capture_residuals", _bfloat16_capture)
    tokenizer = _SpanTokenizer()

    dataset = state_probe.build_probe_dataset(
        object(),
        tokenizer,
        _single_task(),
        [1],
        capture_positions=("last", "note_mean"),
        capture_dtype=capture_dtype,
        mlx_runtime=mx,
    )

    assert dataset.meta["capture_dtype"] == capture_dtype
    assert dataset.features[1].dtype == np.float32
    assert dataset.note_features is not None
    stored_last = dataset.features[1]
    # The last-token value is bfloat16-representable under both paths: widening is exact.
    np.testing.assert_array_equal(state_probe._round_to_bfloat16(stored_last), stored_last)

    note = dataset.note_features[1]
    if capture_dtype == "native":
        # Pooled in the block's own precision, so the mean is a bfloat16 number.
        np.testing.assert_array_equal(state_probe._round_to_bfloat16(note), note)
    else:
        assert not np.array_equal(state_probe._round_to_bfloat16(note), note)


def test_capture_dtype_native_adds_no_rounding_beyond_the_blocks_own(monkeypatch) -> None:
    """The native path must store the block's bfloat16 values, not re-round them."""
    import mlx.core as mx

    _install_single_row(monkeypatch)
    monkeypatch.setattr(state_probe, "capture_residuals", _bfloat16_capture)
    tokenizer = _SpanTokenizer()
    ids = tokenizer.encode("PROMPT")

    dataset = state_probe.build_probe_dataset(
        object(),
        tokenizer,
        _single_task(),
        [1],
        capture_dtype="native",
        mlx_runtime=mx,
    )

    expected = np.array(
        mx.array(np.array([float(ids[-1]) / 3.0], dtype=np.float32))
        .astype(mx.bfloat16)
        .astype(mx.float32),
        dtype=np.float32,
    )
    np.testing.assert_array_equal(dataset.features[1][0], expected)


def test_build_probe_dataset_rejects_an_unknown_capture_dtype_or_position() -> None:
    with pytest.raises(ValueError, match="capture_dtype"):
        state_probe.build_probe_dataset(
            object(), _SpanTokenizer(), [], [1], capture_dtype="float16"
        )
    with pytest.raises(ValueError, match="capture position"):
        state_probe.build_probe_dataset(
            object(), _SpanTokenizer(), [], [1], capture_positions=("first",)
        )


def test_spec_capture_dtype_reads_the_specs_own_name() -> None:
    from dataclasses import replace

    spec = load_model_spec("qwen35-4b")

    assert state_probe.spec_capture_dtype(spec) == "native"
    assert state_probe.spec_capture_dtype(replace(spec, probe_capture_dtype="float32")) == "float32"
    # Registry stand-ins in tests and older callers have no probes block at all.
    assert state_probe.spec_capture_dtype(object()) == "native"


def test_preflight_precision_block_is_none_when_neither_key_is_present(tmp_path) -> None:
    """R18a: no artifact, or an artifact carrying neither shape, records ``None``.

    A probe capture is not the place to gate on preflight evidence, and inventing a number
    would be worse than saying there is none.
    """
    spec = load_model_spec("qwen35-4b")

    assert state_probe.preflight_precision_block(spec, output_root=tmp_path) is None

    (tmp_path / f"{spec.name}.json").write_text(
        json.dumps({"residual_equivalence": {"passed": True}}), encoding="utf-8"
    )

    assert state_probe.preflight_precision_block(spec, output_root=tmp_path) is None


def test_preflight_precision_block_still_reads_the_older_top_level_shape(tmp_path) -> None:
    """The fallback route only -- and this hand-made record is why it must not stand alone.

    This shape is what the reader assumed and what the previous coverage supplied, so the test
    passed while production returned ``null`` on every real artifact: ``run_preflight`` nests
    the block under ``residual_equivalence`` and writes nothing at the top level. The
    writer-shaped case is pinned in ``tests/test_preflight.py``, against a record that
    ``run_preflight`` itself produced.
    """
    spec = load_model_spec("qwen35-4b")
    block = {"frobenius_relative_error": 0.004, "max_abs_error": 0.5}
    (tmp_path / f"{spec.name}.json").write_text(
        json.dumps({"fp32_manual_vs_native": block}), encoding="utf-8"
    )

    assert state_probe.preflight_precision_block(spec, output_root=tmp_path) == block


# ------------------------------------------------------------------------- the capture CLI


def _install_capture_cli(monkeypatch, tmp_path, recorded: dict[str, Any]):
    """Stub every model-touching seam of the capture CLI; nothing is loaded or run."""
    import sys

    from local_llm_lab import models
    from local_llm_lab.pipeline import evaluate
    from local_llm_lab.probes import guard, policies

    spec = load_model_spec("qwen35-4b")
    monkeypatch.setattr(models, "load_model_spec", lambda _name: spec)
    monkeypatch.setattr(guard, "require_idle_gpu", lambda *_a: None)
    monkeypatch.setattr(policies, "resolve_policy", lambda _name, _spec: None)
    monkeypatch.setattr(
        evaluate,
        "load_policy",
        lambda *_a: (object(), object(), SimpleNamespace(num_layers=4), object()),
    )
    monkeypatch.setattr(state_probe, "artifact_identity", lambda *_a: {})
    monkeypatch.setattr(state_probe, "set_mlx_cache_limit", lambda *_a: 0)
    monkeypatch.setitem(
        sys.modules,
        "mlx.core",
        SimpleNamespace(clear_cache=lambda: None, set_cache_limit=lambda _value: None),
    )

    def capture(_model, _tokenizer, tasks, layers, strip=False, **kwargs):
        recorded["tasks"] = list(tasks)
        recorded["strip"] = strip
        recorded["kwargs"] = kwargs
        return state_probe.ProbeDataset(
            layers=list(layers),
            features={layer: np.empty((0, 1), dtype=np.float32) for layer in layers},
            labels={},
            task_ids=np.array([], dtype=str),
            meta={"layers": list(layers), "condition": kwargs.get("condition", "intact")},
        )

    monkeypatch.setattr(state_probe, "build_probe_dataset", capture)
    monkeypatch.setattr(state_probe, "save_dataset", lambda _dataset, path: path)

    def fit(dataset, _targets, _layers, *_a, position="last", **_k):
        recorded["position"] = position
        return {"meta": dataset.meta, "targets": {}, "position": position}

    monkeypatch.setattr(state_probe, "fit_probes", fit)
    monkeypatch.setattr(state_probe, "render_markdown", lambda *_a: "# fake")
    return spec


@pytest.mark.parametrize("selector", [["--p2"], ["--splits", "p2-d0,p2-d1,p2-d2"]])
def test_capture_cli_p2_builds_the_three_probe_splits(monkeypatch, tmp_path, selector) -> None:
    """SPEC-004 §2 / R28: ``--p2`` draws through ``make_p2_tasks`` with explicit difficulty."""
    from local_llm_lab.pipeline.tasks import P2_SPLIT_NAMES, split_of_task_id

    recorded: dict[str, Any] = {}
    _install_capture_cli(monkeypatch, tmp_path, recorded)
    monkeypatch.setattr(
        "sys.argv",
        [
            "state-probe",
            "--model",
            "qwen35-4b",
            "--output",
            str(tmp_path),
            "--limit",
            "3",
            *selector,
        ],
    )

    state_probe.main()

    tasks = recorded["tasks"]
    assert len(tasks) == 3 * 3
    assert {split_of_task_id(task.task_id) for task in tasks} == set(P2_SPLIT_NAMES)
    levels = {split_of_task_id(task.task_id): task.difficulty for task in tasks}
    assert levels == {"p2-d0": 0, "p2-d1": 1, "p2-d2": 2}
    assert recorded["kwargs"]["difficulties"] == {task.task_id: task.difficulty for task in tasks}
    assert (tmp_path / "state-base-p2.npz").name  # the stem carries the plan
    events = _events(tmp_path)
    start = _flat(events[0])
    assert start["splits"] == list(P2_SPLIT_NAMES)
    assert start["condition"] == "intact"
    assert start["positions"] == ["last", "note_mean"]
    assert start["capture_dtype"] == "native"


def test_capture_cli_stub_observations_names_the_condition_in_the_stem(
    monkeypatch, tmp_path
) -> None:
    recorded: dict[str, Any] = {}
    _install_capture_cli(monkeypatch, tmp_path, recorded)
    monkeypatch.setattr(
        "sys.argv",
        [
            "state-probe",
            "--model",
            "qwen35-4b",
            "--output",
            str(tmp_path),
            "--limit",
            "1",
            "--strip",
            "--stub-observations",
            "--capture-positions",
            "last,note_mean",
            "--position",
            "note_mean",
        ],
    )

    state_probe.main()

    assert recorded["strip"] is True
    assert recorded["kwargs"]["stub_observations"] is True
    assert recorded["position"] == "note_mean"
    assert (tmp_path / "state-base-stripped-stubbed.json").is_file()
    start = _flat(_events(tmp_path)[0])
    assert start["condition"] == "both"
    assert start["positions"] == ["last", "note_mean"]


def test_capture_cli_keeps_the_legacy_stem_for_the_two_old_conditions(
    monkeypatch, tmp_path
) -> None:
    """Existing artifact names must not move: intact stays bare, --strip stays -stripped."""
    recorded: dict[str, Any] = {}
    _install_capture_cli(monkeypatch, tmp_path, recorded)
    monkeypatch.setattr(
        "sys.argv",
        ["state-probe", "--model", "qwen35-4b", "--output", str(tmp_path), "--limit", "1"],
    )
    state_probe.main()
    assert (tmp_path / "state-base.json").is_file()
    # No --p2: the default stays the single last-token capture, so cost and shards are unmoved.
    assert _flat(_events(tmp_path)[0])["positions"] == ["last"]

    monkeypatch.setattr(
        "sys.argv",
        [
            "state-probe",
            "--model",
            "qwen35-4b",
            "--output",
            str(tmp_path),
            "--limit",
            "1",
            "--strip",
        ],
    )
    state_probe.main()
    assert (tmp_path / "state-base-stripped.json").is_file()
