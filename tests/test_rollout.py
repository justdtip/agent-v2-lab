from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
