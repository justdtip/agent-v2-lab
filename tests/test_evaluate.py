from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from local_llm_lab.pipeline import evaluate
from local_llm_lab.pipeline.runner import Trajectory
from local_llm_lab.pipeline.tasks import Task


def _trajectory(
    task_id: str,
    *,
    success: bool,
    clean: bool,
    difficulty: int,
    family: str = "read",
    variant: str = "clean",
    think_tokens: int = 0,
) -> Trajectory:
    return Trajectory(
        task_id=task_id,
        family=family,
        variant=variant,
        label="fake",
        prompt="fake prompt",
        verdict={
            "success": success,
            "clean": clean,
            "calls": 2,
            "schema_failures": 0,
            "executable_calls": 2,
            "errors": 0,
            "recovered_errors": 0,
        },
        turns=2,
        valid_turns=2,
        generated_tokens=4,
        difficulty=difficulty,
        integrity={"clean": clean, "counts": {} if clean else {"value_drop": 1}},
        think_tokens=think_tokens,
    )


@pytest.mark.parametrize(
    ("stress", "expected_faults"),
    [(False, None), (True, evaluate.STRESS_FAULTS)],
)
def test_evaluate_tasks_passes_stress_faults_to_runner(
    monkeypatch, stress: bool, expected_faults: object
) -> None:
    captured: dict[str, object] = {}
    task = Task(
        task_id="test-read-0000-clean",
        family="read",
        variant="clean",
        prompt="read a file",
        files={"a.txt": "x"},
        steps=(),
        expected_answer="x",
        required_tools=frozenset(),
    )

    monkeypatch.setattr(evaluate, "make_sampler", lambda _temperature: object())

    def fake_run_task(*_args, **kwargs):
        captured["faults"] = kwargs["faults"]
        return _trajectory(task.task_id, success=True, clean=True, difficulty=0)

    monkeypatch.setattr(evaluate, "run_task", fake_run_task)
    monkeypatch.setattr(
        evaluate,
        "check_trajectory",
        lambda *_args, **_kwargs: SimpleNamespace(
            as_dict=lambda: {"clean": True, "counts": {}}
        ),
    )

    evaluate.evaluate_tasks(
        object(), object(), [task], label="fake", stress=stress, quiet=True
    )

    assert captured["faults"] == expected_faults


def test_wilson_and_mcnemar_use_exact_small_sample_statistics() -> None:
    """Catch missing statistics, invalid counts, and asymmetric exact-binomial mistakes."""
    assert evaluate.wilson(0, 0) == (0.0, 0.0)
    assert evaluate.wilson(5, 10) == pytest.approx(
        (0.2365895936, 0.7634104064), abs=1e-10
    )
    with pytest.raises(ValueError, match="successes"):
        evaluate.wilson(3, 2)
    assert evaluate.mcnemar(
        {"a": True, "b": True, "c": False},
        {"a": False, "b": True, "c": True},
    ) == {"tasks": 3, "a_only": 1, "b_only": 1, "discordant": 2, "p_value": 1.0}
    assert evaluate.mcnemar(
        {"a": True, "b": True, "c": True, "d": False},
        {"a": False, "b": False, "c": True, "d": False},
    )["p_value"] == 0.5
    assert evaluate.mcnemar({"a": True}, {"a": True})["discordant"] == 0
    with pytest.raises(ValueError, match="same non-empty"):
        evaluate.mcnemar({"a": True}, {"b": True})


def test_summarize_exposes_rate_counts_and_wilson_intervals_for_all_groups() -> None:
    """Catch rounded-only summaries and missing difficulty or integrity attribution."""
    summary = evaluate.summarize(
        [
            _trajectory("valid-read-0000-clean", success=True, clean=True, difficulty=1),
            _trajectory(
                "valid-pointer_chain-0006-clean",
                success=False,
                clean=False,
                difficulty=2,
                family="pointer_chain",
            ),
        ]
    )
    assert summary["rate_counts"] == {
        "success": {"numerator": 1, "denominator": 2},
        "clean": {"numerator": 1, "denominator": 2},
        "valid_actions": {"numerator": 4, "denominator": 4},
        "schema_validity": {"numerator": 4, "denominator": 4},
        "executable_calls": {"numerator": 4, "denominator": 4},
        "integrity_clean": {"numerator": 1, "denominator": 2},
    }
    assert set(summary["wilson_95"]) == set(summary["rate_counts"])
    for group in ("by_family", "by_variant", "by_difficulty"):
        assert all(
            "rate_counts" in stats and "wilson_95" in stats
            for stats in summary[group].values()
        )
    assert summary["by_difficulty"]["1"]["successes"] == 1
    assert summary["wilson_95"]["integrity_clean"] == pytest.approx(evaluate.wilson(1, 2))


def test_summarize_aggregates_thinking_tokens_and_preserves_integrity() -> None:
    clean = _trajectory(
        "test-read-0000-clean",
        success=True,
        clean=True,
        difficulty=0,
        think_tokens=3,
    )
    affected = _trajectory(
        "test-read-0001-clean",
        success=False,
        clean=False,
        difficulty=0,
        think_tokens=8,
    )
    affected.integrity = {"clean": False, "counts": {"value_drop": 2}}

    summary = evaluate.summarize([clean, affected])

    assert summary["integrity"] == {
        "clean_trajectories": 1,
        "clean_rate": 0.5,
        "affected_trajectories": 1,
        "violations": 2,
        "by_kind": {"value_drop": {"violations": 2, "affected_trajectories": 1}},
        "by_family": {
            "read": {
                "trajectories": 2,
                "clean_trajectories": 1,
                "clean_rate": 0.5,
                "violations": 2,
                "affected_trajectories": 1,
            }
        },
        "failed_trajectories": 1,
        "failed_with_violation": 1,
        "failure_explained_rate": 1.0,
    }
    assert summary["think_tokens"] == 11
    assert summary["think_tokens_per_task"] == 5.5


def test_run_evaluation_records_explicit_difficulty_and_screen_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    """Catch a screen request that silently falls back to split defaults or loses provenance."""
    captured: dict[str, object] = {}

    def fake_load_policy(model_name: str, adapter: Path | None) -> tuple[object, object]:
        captured["model"] = model_name
        return object(), object()

    def fake_evaluate_tasks(_model, _tokenizer, tasks, **_kwargs):
        captured["tasks"] = tasks
        return [
            _trajectory(
                task.task_id,
                success=True,
                clean=True,
                difficulty=task.difficulty,
                family=task.family,
            )
            for task in tasks
        ]

    monkeypatch.setattr(evaluate, "load_policy", fake_load_policy)
    monkeypatch.setattr(evaluate, "evaluate_tasks", fake_evaluate_tasks)
    monkeypatch.setattr(evaluate, "_seed_model_rng", lambda _seed: None)
    monkeypatch.setattr(evaluate, "_clear_model_cache", lambda: None)
    summary = evaluate.run_evaluation(
        model_name="fake-model",
        adapter=None,
        label="fake",
        split="valid2",
        limit=None,
        difficulty=2,
        family_quotas={"default": 1, "long": 3},
        output=tmp_path / "eval.json",
        transcript_dir=None,
        quiet=True,
        seed=17,
    )

    assert len(captured["tasks"]) == 24
    assert {task.difficulty for task in captured["tasks"]} == {2}
    assert summary["model"] == "fake-model"
    assert summary["difficulty"] == 2
    assert summary["data_seed"] == 17
    payload = (tmp_path / "eval.json").read_text(encoding="utf-8")
    assert '"difficulty": 2' in payload and '"data_seed": 17' in payload


def test_run_evaluation_marks_mixed_default_difficulties_without_a_false_single_level(
    monkeypatch, tmp_path: Path
) -> None:
    """Catch metadata that labels an alternating split as its first task's difficulty."""

    monkeypatch.setattr(evaluate, "load_policy", lambda _model, _adapter: (object(), object()))
    monkeypatch.setattr(
        evaluate,
        "evaluate_tasks",
        lambda _model, _tokenizer, tasks, **_kwargs: [
            _trajectory(
                task.task_id,
                success=True,
                clean=True,
                difficulty=task.difficulty,
                family=task.family,
            )
            for task in tasks
        ],
    )
    monkeypatch.setattr(evaluate, "_seed_model_rng", lambda _seed: None)
    monkeypatch.setattr(evaluate, "_clear_model_cache", lambda: None)

    summary = evaluate.run_evaluation(
        model_name="fake-model",
        adapter=None,
        label="mixed",
        split="valid2",
        limit=4,
        output=tmp_path / "mixed.json",
        transcript_dir=None,
        quiet=True,
        seed=17,
    )

    assert summary["difficulty"] is None
    assert summary["difficulties"] == [0, 1]
