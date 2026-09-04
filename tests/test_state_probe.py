from __future__ import annotations

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


def _refit_baseline() -> dict[str, Any]:
    """The baseline a refit is measured against, produced by the reanalysis itself."""
    return state_probe.reanalyse_dataset(
        _refit_capture(),
        split_seeds=_REFIT_SPLIT_SEEDS,
        bootstrap_resamples=_REFIT_RESAMPLES,
        data_seed=20260902,
    )


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
        for split_seed in split_seeds:
            for layer in layers:
                scale = 1.0 - advantage if layer == advantage_layer else 1.0
                base = {
                    "cohort": cohort,
                    "target": "pending_count",
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
                records.append({**base, "control": "probe", "predicted": actual + error * scale})
                records.append({**base, "control": "position", "predicted": actual + error * 2.0})
                records.append({**base, "control": "surface", "predicted": actual + error * 3.0})
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
