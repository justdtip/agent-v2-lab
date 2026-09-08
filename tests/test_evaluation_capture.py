"""Evaluation capture wiring without a model runtime."""

from types import SimpleNamespace

import pytest

from local_llm_lab.models import load_model_spec
from local_llm_lab.pipeline import evaluate
from local_llm_lab.pipeline.runner import Trajectory
from local_llm_lab.pipeline.tasks import make_tasks


def test_optional_capture_reaches_runner_without_changing_defaults(monkeypatch):
    task = make_tasks("train", 1)[0]
    received = []

    def run(*args, **kwargs):
        received.append(kwargs)
        return Trajectory(task.task_id, task.family, task.variant, "base", task.prompt)

    monkeypatch.setattr(evaluate, "run_task", run)
    monkeypatch.setattr(evaluate, "make_sampler", lambda _: None)
    monkeypatch.setattr(
        evaluate, "check_trajectory", lambda *_a, **_k: SimpleNamespace(as_dict=lambda: {})
    )
    common = dict(
        spec=load_model_spec("gemma3-4b"),
        view=None,
        resolved=None,
        label="base",
        quiet=True,
        use_cache=False,
    )
    evaluate.evaluate_tasks(None, None, [task], **common)
    capture = object()
    evaluate.evaluate_tasks(None, None, [task], capture=capture, **common)
    assert "capture" not in received[0]
    assert received[1]["capture"] is capture


def test_capture_requires_no_reuse_before_sampler_or_loading(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate, "make_sampler", lambda _: pytest.fail("sampler called"))
    monkeypatch.setattr(evaluate, "load_policy", lambda *_a: pytest.fail("model loaded"))
    with pytest.raises(ValueError, match="use_cache=False"):
        evaluate.evaluate_tasks(
            None, None, [], spec=None, view=None, resolved=None, label="base", capture=object()
        )
    with pytest.raises(ValueError, match="use_cache=False"):
        evaluate.run_evaluation(
            spec=None,
            adapter=None,
            label="base",
            split="train",
            limit=1,
            output=tmp_path / "eval.json",
            transcript_dir=None,
            capture=object(),
        )


def test_run_evaluation_passes_capture_to_stage(monkeypatch, tmp_path):
    reached = []
    capture = object()
    monkeypatch.setattr(evaluate, "_seed_model_rng", lambda _: None)
    monkeypatch.setattr(evaluate, "_clear_model_cache", lambda: None)
    monkeypatch.setattr(evaluate, "load_policy", lambda *_a: (None, None, None, None))

    def stage(*args, **kwargs):
        reached.append(kwargs)
        raise LookupError("stage reached")

    monkeypatch.setattr(evaluate, "evaluate_tasks", stage)
    with pytest.raises(LookupError, match="stage reached"):
        evaluate.run_evaluation(
            spec=load_model_spec("gemma3-4b"),
            adapter=None,
            label="base",
            split="train",
            limit=1,
            output=tmp_path / "eval.json",
            transcript_dir=None,
            capture=capture,
            use_cache=False,
        )
    assert reached[0]["capture"] is capture
    assert reached[0]["max_steps"] == 24
