from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from local_llm_lab import runlog as runlog_module
from local_llm_lab.models import load_model_spec
from local_llm_lab.pipeline import rollout
from local_llm_lab.pipeline.runner import Trajectory
from local_llm_lab.pipeline.tasks import make_tasks


def _install_fake_mlx(monkeypatch, events: list[tuple[object, ...]]) -> None:
    core = ModuleType("mlx.core")
    core.random = SimpleNamespace(seed=lambda value: events.append(("seed", value)))
    core.clear_cache = lambda: events.append(("clear_cache",))
    package = ModuleType("mlx")
    package.core = core
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)


def test_collect_rollouts_forwards_active_model_context_to_every_task(monkeypatch) -> None:
    spec = load_model_spec("qwen35-4b")
    model, tokenizer, view, resolved = object(), object(), object(), object()
    sampler = object()
    task = make_tasks("rollout-model-context", 1, seed=17)[0]
    trajectory = Trajectory(task.task_id, task.family, task.variant, "fake", task.prompt)
    trajectory.verdict = {"success": False, "clean": False}
    seen: list[tuple[object, ...]] = []
    runtime_events: list[tuple[object, ...]] = []

    _install_fake_mlx(monkeypatch, runtime_events)

    def fake_sampler(temperature: float):
        assert temperature == 0.0
        return sampler

    def fake_run(actual_model, actual_tokenizer, actual_task, **kwargs):
        assert kwargs["sampler"] is sampler
        seen.append(
            (
                actual_model,
                actual_tokenizer,
                actual_task,
                kwargs["spec"],
                kwargs["view"],
                kwargs["resolved"],
            )
        )
        return trajectory

    def fake_integrity(actual_task, steps, *, keep_last):
        assert actual_task is task
        assert steps is trajectory.steps
        assert keep_last == 2
        return SimpleNamespace(
            clean=True,
            as_dict=lambda: {"violations": [], "counts": {}, "clean": True},
        )

    monkeypatch.setattr(rollout, "make_sampler", fake_sampler)
    monkeypatch.setattr(rollout, "run_task", fake_run)
    monkeypatch.setattr(rollout, "check_trajectory", fake_integrity)

    rollout.collect_rollouts(
        model,
        tokenizer,
        [task],
        spec=spec,
        view=view,
        resolved=resolved,
        label="fake",
        samples=1,
        temperature=0.0,
        quiet=True,
        seed=17,
    )

    assert seen == [(model, tokenizer, task, spec, view, resolved)]
    assert runtime_events == [("seed", 17010)]


def test_run_rollout_loads_and_forwards_one_registry_spec(monkeypatch, tmp_path: Path) -> None:
    spec = load_model_spec("qwen35-4b")
    model, tokenizer, view = object(), object(), object()
    resolved = SimpleNamespace(as_dict=lambda: {"spec": {"name": spec.name}})
    tasks: list[object] = []
    rows: list[dict[str, object]] = []
    seen: list[tuple[object, ...]] = []

    _install_fake_mlx(monkeypatch, seen)

    def fake_registry(name: str):
        seen.append(("registry", name))
        assert name == "qwen35-4b"
        return spec

    def fake_loader(actual, adapter):
        seen.append(("loader", actual, adapter))
        assert actual is spec, "load_policy must receive the exact registry ModelSpec"
        assert adapter is None
        return model, tokenizer, view, resolved

    def fake_collect(actual_model, actual_tokenizer, actual_tasks, **kwargs):
        assert actual_model is model
        assert actual_tokenizer is tokenizer
        assert actual_tasks is tasks
        seen.append(
            (
                "collect",
                actual_model,
                actual_tokenizer,
                kwargs["spec"],
                kwargs["view"],
                kwargs["resolved"],
            )
        )
        return [], rows, {"pass_at_k": 0.0}

    def fake_write_jsonl(path: Path, actual_rows):
        assert path == tmp_path / "train.jsonl"
        assert actual_rows is rows
        return "digest"

    monkeypatch.setattr(rollout, "load_model_spec", fake_registry, raising=False)
    monkeypatch.setattr(rollout, "load_policy", fake_loader)
    monkeypatch.setattr(rollout, "collect_rollouts", fake_collect)
    monkeypatch.setattr(rollout, "make_tasks", lambda *_args, **_kwargs: tasks)
    monkeypatch.setattr(rollout, "write_jsonl", fake_write_jsonl)
    monkeypatch.setattr(rollout, "summary_table", lambda summary: json.dumps(summary))

    summary = rollout.run_rollout(
        model_name="qwen35-4b",
        adapter=None,
        label="fake",
        split="iter-model-context",
        limit=1,
        samples=1,
        temperature=0.0,
        output=tmp_path,
        transcript_dir=None,
        quiet=True,
    )

    assert seen[:2] == [("registry", "qwen35-4b"), ("loader", spec, None)]
    assert seen[2] == ("collect", model, tokenizer, spec, view, resolved)
    assert seen.count(("clear_cache",)) == 1
    assert summary["model"] == resolved.as_dict()
    assert json.loads((tmp_path / "summary.json").read_text())["model"] == resolved.as_dict()


# --------------------------------------------------------------- module main (R21, R26(a))


def _rollout_argv(output: Path, transcripts: Path) -> list[str]:
    return [
        "agent-v2-rollout",
        "--split",
        "guard-check",
        "--output",
        str(output),
        "--transcripts",
        str(transcripts),
        "--quiet",
    ]


def test_rollout_main_refuses_a_protected_dataset_directory(monkeypatch, tmp_path: Path) -> None:
    """R21: the module main must refuse PROTECTED_DATASETS as hard as the CLI boundary does."""
    from local_llm_lab.pipeline.data import ProtectedDatasetError
    from local_llm_lab.project import PROJECT_ROOT

    protected = PROJECT_ROOT / "data" / "agent_v2"
    monkeypatch.setattr(
        rollout, "run_rollout", lambda **_kwargs: pytest.fail("the R21 guard did not fire")
    )
    monkeypatch.setattr(sys, "argv", _rollout_argv(protected, tmp_path / "transcripts"))

    with pytest.raises(ProtectedDatasetError):
        rollout.main()

    assert not (tmp_path / "transcripts").exists()


def test_rollout_main_refuses_an_existing_manifest_with_no_override(
    monkeypatch, tmp_path: Path
) -> None:
    """R21: this stage has no override flag, so an existing dataset is simply refused."""
    from local_llm_lab.pipeline.data import DatasetWriteGuardError

    target = tmp_path / "rollouts"
    target.mkdir()
    (target / "manifest.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        rollout, "run_rollout", lambda **_kwargs: pytest.fail("the R21 guard did not fire")
    )
    monkeypatch.setattr(sys, "argv", _rollout_argv(target, tmp_path / "transcripts"))

    with pytest.raises(DatasetWriteGuardError) as error:
        rollout.main()

    assert "no overwrite path" in str(error.value)


def _interrupt_the_manifest_writer(monkeypatch, limit: int) -> None:
    """Truncate whichever writer the stamp uses and then fail, standing in for a full disk.

    ``runlog.write_text_atomic`` writes through ``runlog``'s own ``os.fdopen`` — the shim the
    sibling lane's ``tests/test_data.py`` interrupts — while a plain stamp writes through
    ``Path.write_text``. Interrupting both keeps the assertion on the outcome the R21 guard
    depends on (no partial sentinel at the destination) rather than on which writer is in use.
    """

    class _Truncating:
        def __init__(self, handle) -> None:
            self._handle = handle

        def __getattr__(self, name: str):
            return getattr(self._handle, name)

        def write(self, text: str) -> int:
            self._handle.write(text[:limit])
            raise OSError("no space left on device")

        def __enter__(self):
            return self

        def __exit__(self, *exc_info) -> bool:
            self._handle.close()
            return False

    class _InterruptingOS:
        """An ``os`` shim patched into runlog's namespace only, so nothing else is affected."""

        def __init__(self, real) -> None:
            self._real = real

        def __getattr__(self, name: str):
            return getattr(self._real, name)

        def fdopen(self, descriptor, *args, **kwargs):
            return _Truncating(self._real.fdopen(descriptor, *args, **kwargs))

    plain_write_text = Path.write_text

    def truncating_write_text(self, data, *args, **kwargs):
        plain_write_text(self, data[:limit], *args, **kwargs)
        raise OSError("no space left on device")

    monkeypatch.setattr(runlog_module, "os", _InterruptingOS(os))
    monkeypatch.setattr(Path, "write_text", truncating_write_text)


def test_an_interrupted_rollout_manifest_write_leaves_the_previous_manifest_intact(
    monkeypatch, tmp_path: Path
) -> None:
    """manifest.json is the R21 guard's own sentinel, so a half-written one is a live hazard.

    ``guard_dataset_write`` (pipeline/data.py:85) decides whether a later write is permitted by
    whether this file is there. With the rollout rows complete and the sentinel truncated, a
    later write would be waved straight over good data, so this stamp must be atomic: either
    the previous manifest survives whole or the new one lands whole.
    """
    target = tmp_path / "rollouts"
    target.mkdir()
    original = '{"stage": "rollout", "kept": true}\n'
    (target / "manifest.json").write_text(original, encoding="utf-8")

    _interrupt_the_manifest_writer(monkeypatch, 12)
    with pytest.raises(OSError, match="no space left on device"):
        rollout._write_stage_manifest(target, {"stage": "rollout", "split": "guard-check"})
    monkeypatch.undo()

    assert (target / "manifest.json").read_text(encoding="utf-8") == original
    leftovers = sorted(path.name for path in target.iterdir() if path.name.startswith("."))
    assert leftovers == [], "no partial temporary file may survive at the destination"


def test_rollout_main_writes_run_log_events_manifest_and_provenance(
    monkeypatch, tmp_path: Path
) -> None:
    """R26(a) plus the provenance stamp its sibling entry point already writes.

    ``branch.main`` records ``provenance.json`` beside the mined dataset and names it in the
    end event; the CLI's rollout stage does the same at ``cli.py:1021``. This entry point is
    the only rollout path that was leaving a dataset with no provenance beside it.
    """
    target = tmp_path / "rollouts"
    captured: dict[str, object] = {}
    summary = {"pass_at_k": 0.0, "kept_rows": 0, "seed": 11}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return dict(summary)

    monkeypatch.setattr(rollout, "run_rollout", fake_run)
    monkeypatch.setattr(sys, "argv", _rollout_argv(target, tmp_path / "transcripts"))

    rollout.main()

    assert (target / "run.log").is_file()
    events = [
        json.loads(line)
        for line in (target / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [event["kind"] for event in events][:1] == ["start"]
    assert len(events) >= 2
    identity = events[0]["fields"]
    assert identity["split"] == "guard-check"
    assert identity["model"] and identity["hf_id"]
    assert callable(captured["progress"])
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "rollout"
    assert manifest["split"] == "guard-check"
    assert manifest["seed"] == 11
    provenance = json.loads((target / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["extra"]["stage"] == "rollout"
    assert provenance["extra"]["summary"] == summary
    end = [event for event in events if event["message"] == "rollout complete"][-1]
    assert end["fields"]["provenance"] == str(target / "provenance.json")


def test_collect_rollouts_reports_progress_once_per_task(monkeypatch) -> None:
    """R26(g): no outer unit of work is silent."""
    spec = load_model_spec("qwen35-4b")
    tasks = make_tasks("rollout-progress", 2, seed=19)
    trajectory = Trajectory(tasks[0].task_id, tasks[0].family, tasks[0].variant, "fake", "p")
    trajectory.verdict = {"success": False, "clean": False}
    reported: list[tuple[int, int, str]] = []

    _install_fake_mlx(monkeypatch, [])
    monkeypatch.setattr(rollout, "make_sampler", lambda _temperature: object())
    monkeypatch.setattr(rollout, "run_task", lambda *_args, **_kwargs: trajectory)
    monkeypatch.setattr(
        rollout,
        "check_trajectory",
        lambda *_args, **_kwargs: SimpleNamespace(clean=False, as_dict=lambda: {}),
    )

    rollout.collect_rollouts(
        object(),
        object(),
        tasks,
        spec=spec,
        view=object(),
        resolved=object(),
        label="fake",
        samples=1,
        temperature=0.0,
        quiet=True,
        seed=19,
        progress=lambda step, total, label, **fields: reported.append((step, total, label)),
    )

    assert [(step, total) for step, total, _label in reported] == [(1, 2), (2, 2)]
