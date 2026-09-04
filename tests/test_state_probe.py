from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from local_llm_lab.models import load_model_spec
from local_llm_lab.pipeline.tasks import GENERATOR_VERSION, make_tasks
from local_llm_lab.probes import state_probe


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


def test_build_probe_dataset_forwards_the_selected_spec_to_prompt_rendering(monkeypatch) -> None:
    selected = load_model_spec("qwen35-4b")
    task = SimpleNamespace(task_id="fake", family="read")
    seen = []

    monkeypatch.setattr(state_probe, "_checkpoint_signature", lambda *_args, **_kwargs: "fake")
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
    state_probe.build_probe_dataset(
        None, tokenizer, [task], [0], spec=selected, mlx_runtime=runtime
    )

    assert seen == [selected]


def test_main_loads_and_dispatches_the_selected_spec(monkeypatch, tmp_path) -> None:
    from local_llm_lab import models
    from local_llm_lab.pipeline import evaluate, tasks
    from local_llm_lab.probes import guard, policies

    selected = object()
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
    monkeypatch.setattr(policies, "resolve_policy", lambda *_args: None)
    monkeypatch.setattr(evaluate, "load_policy", lambda *_args: (None, None))
    monkeypatch.setattr(state_probe, "artifact_identity", lambda *_args: {})
    monkeypatch.setattr(state_probe, "set_mlx_cache_limit", lambda *_args: 0)

    def intercept(*_args, **kwargs):
        seen.append(("dispatch", kwargs["spec"]))
        raise RuntimeError("stop after dispatch")

    monkeypatch.setattr(state_probe, "build_probe_dataset", intercept)
    monkeypatch.setattr(
        "sys.argv",
        ["state-probe", "--model", "qwen35-4b", "--output", str(tmp_path), "--limit", "1"],
    )

    with pytest.raises(RuntimeError, match="stop after dispatch"):
        state_probe.main()

    assert seen == [("load", "qwen35-4b"), ("dispatch", selected)]


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


def test_v4_row_labels_match_nonvacuous_progress_note_oracles() -> None:
    """Independent note parsing pins the v4 state labels to real generated progress."""
    assert GENERATOR_VERSION == 4
    counts = {"pending": 0, "loads": 0, "values": 0, "batch": 0}
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
