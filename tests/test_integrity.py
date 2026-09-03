from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_llm_lab.pipeline import evaluate, report
from local_llm_lab.pipeline.env import Simulator
from local_llm_lab.pipeline.integrity import (
    Fact,
    IntegrityReport,
    Violation,
    _render_evaluations,
    check_trajectory,
    completion_patterns,
    required_carry,
)
from local_llm_lab.pipeline.runner import Trajectory
from local_llm_lab.pipeline.tasks import (
    FAMILIES,
    VARIANTS,
    difficulty,
    make_tasks,
    task_from_id,
)


def _task(family: str):
    return next(task for task in make_tasks("test", 12) if task.family == family)


def _expert_trace(task) -> list[dict[str, object]]:
    simulator = Simulator.for_task(task)
    records = []
    for index, step in enumerate(task.steps):
        observation = simulator.execute(step.action)
        records.append(
            {
                "index": index,
                "thought": step.thought,
                "action": {"name": step.action.name, "arguments": step.action.arguments},
                "observation": observation,
                "raw": None,
            }
        )
    return records


def test_difficulty_override_preserves_default_mapping() -> None:
    assert difficulty("test", 3, override=1) == 1
    assert difficulty("train", 3, override=None) == 0
    assert difficulty("valid", 3, override=None) == 1
    assert difficulty("test", 3, override=None) == 2
    assert difficulty("fresh-split", 3, override=None) == 1
    with pytest.raises(ValueError, match="difficulty must be non-negative"):
        difficulty("test", 3, override=-1)


def test_make_tasks_sets_explicit_difficulty_deterministically() -> None:
    first = make_tasks("test", 24, difficulty=3)
    second = make_tasks("test", 24, difficulty=3)

    assert first == second
    assert all(task.difficulty == 3 for task in first)
    assert [task.task_id for task in first] == [
        f"test-{task.family}-{index:04d}-clean" for index, task in enumerate(first)
    ]


def test_run_d_generator_uses_pending_none_only_for_empty_action_queues() -> None:
    """Catch any completion wording that is not the exhausted-queue form."""
    patterns = completion_patterns()
    for level in range(4):
        for task in make_tasks("integrity", len(FAMILIES), difficulty=level, perturb=False):
            for step in task.steps:
                matches = [pattern.search(step.thought) for pattern in patterns]
                if any(matches):
                    assert "pending: none" in step.thought.casefold()


@pytest.mark.parametrize(
    "family",
    (
        "ledger_reconcile",
        "cross_reference",
        "conditional_update",
        "batch_update",
        "aggregate_report",
    ),
)
def test_run_d_required_carry_facts_are_literal_in_the_consuming_note(family: str) -> None:
    """Catch a long-family note that drops hidden values, keys, paths, or workers."""
    task = next(
        task for task in make_tasks("carry", len(FAMILIES), difficulty=3) if task.family == family
    )
    for step_index, facts in required_carry(task, keep_last=2).items():
        thought = task.steps[step_index].thought
        for fact in facts:
            value = fact.value.rsplit("/", 1)[-1]
            assert value in thought, (task.task_id, step_index, fact, thought)


@pytest.mark.parametrize("split", ["train", "valid", "test", "fresh-split"])
def test_task_from_id_rebuilds_clean_and_recovery_tasks(split: str) -> None:
    seed = 20260902
    generated = make_tasks(split, 72, seed)
    representatives = [next(task for task in generated if task.variant == "clean")]
    recovery = next((task for task in generated if task.variant != "clean"), None)
    if recovery is not None:
        representatives.append(recovery)

    for original in representatives:
        rebuilt = task_from_id(original.task_id, seed, difficulty=original.difficulty)
        assert rebuilt == original


@pytest.mark.parametrize(
    "task_id",
    [
        "invalid",
        "train-unknown-0000-clean",
        "train-read-nope-clean",
        "train-read-0001-clean",
        "train-read-0000-unknown",
        "train-read-0000-failed_edit",
    ],
)
def test_task_from_id_rejects_invalid_or_impossible_ids(task_id: str) -> None:
    with pytest.raises(ValueError, match=task_id):
        task_from_id(task_id, 20260902)


def test_task_from_id_does_not_depend_on_global_random_state() -> None:
    original = make_tasks("fresh-split", 48)[-1]
    random.seed(99)
    assert task_from_id(original.task_id, 20260902, difficulty=original.difficulty) == original
    with pytest.raises(ValueError, match=original.task_id):
        task_from_id(original.task_id, 20260902, difficulty=-1)


def test_task_from_id_requires_seed_and_accepts_positional_difficulty() -> None:
    original = make_tasks("test", 3, 20260902)[-1]
    assert task_from_id(original.task_id, 20260902, original.difficulty) == original
    with pytest.raises(TypeError):
        task_from_id(original.task_id)


def test_every_public_variant_name_is_covered_by_reconstruction_validation() -> None:
    assert set(VARIANTS) == {
        "clean",
        "wrong_path",
        "transient",
        "unknown_tool",
        "stale_path",
        "failed_edit",
    }


def test_integrity_report_as_dict_is_stable_and_sorted() -> None:
    first = Violation(2, "value_drop", "missing 18")
    second = Violation(2, "queue_loss", "omitted metric-4.txt")
    report = IntegrityReport(
        violations=(first, second),
        first_violation=first,
        counts={"queue_loss": 1, "value_drop": 1},
        clean=False,
    )
    assert report.as_dict() == {
        "violations": [
            {"step": 2, "kind": "value_drop", "detail": "missing 18"},
            {"step": 2, "kind": "queue_loss", "detail": "omitted metric-4.txt"},
        ],
        "first_violation": {"step": 2, "kind": "value_drop", "detail": "missing 18"},
        "counts": {"queue_loss": 1, "value_drop": 1},
        "clean": False,
    }
    assert IntegrityReport((), None, {}, True).as_dict() == {
        "violations": [],
        "first_violation": None,
        "counts": {},
        "clean": True,
    }


def test_completion_patterns_recognise_only_public_completion_forms() -> None:
    patterns = completion_patterns()
    assert len(patterns) == 6
    for text in (
        "first half: 1, 2 (FULL)",
        "highest so far: service-2=88 (Final)",
        "phase APPLY COMPLETE",
        "All 6 invoices READ",
        "all 4 workers Inspected",
        "ALL 3 edits APPLIED",
        "all 5 files verified",
        "PENDING: NONE",
        "task COMPLETE",
    ):
        assert any(pattern.search(text) for pattern in patterns), text
    assert not any(pattern.search("still pending: metric-2.txt") for pattern in patterns)


def test_required_carry_validates_keep_last() -> None:
    with pytest.raises(ValueError, match="keep_last must be non-negative"):
        required_carry(_task("read"), keep_last=-1)


def test_required_carry_tracks_approved_amounts_but_not_held_amounts() -> None:
    carry = required_carry(_task("ledger_reconcile"), keep_last=1)
    assert carry[7] == frozenset(
        {
            Fact("amount", "178", 1),
            Fact("amount", "40", 3),
            Fact("amount", "32", 4),
        }
    )
    assert all(fact.value not in {"38", "110"} for facts in carry.values() for fact in facts)


def test_required_carry_tracks_metric_subtotals_and_observation_indices() -> None:
    carry = required_carry(_task("aggregate_report"), keep_last=1)
    assert carry[7] == frozenset(
        {
            Fact("metric", "18", 1),
            Fact("metric", "14", 2),
            Fact("metric", "75", 3),
            Fact("metric", "72", 4),
            Fact("metric", "18", 5),
        }
    )


def test_required_carry_tracks_highest_load_worker_and_paths() -> None:
    loads = required_carry(_task("conditional_update"), keep_last=1)
    assert Fact("load", "service-2=88", 4) in loads[7]

    workers = required_carry(_task("batch_update"), keep_last=1)
    assert Fact("worker", "worker-0.ini mode=observe -> mode=strict", 0) in workers[5]
    assert Fact("worker", "worker-3.ini mode=strict -> mode=safe", 0) in workers[5]

    paths = required_carry(_task("list"), keep_last=0)
    assert paths[1] == frozenset({Fact("path", "summary-88.md", 0)})


def test_required_carry_tracks_hidden_next_keys_and_calculator_totals() -> None:
    next_keys = required_carry(_task("cross_reference"), keep_last=0)
    assert Fact("next_key", "REF-TEST-0008-9496-1", 1) in next_keys[2]

    totals = required_carry(_task("synthesis"), keep_last=0)
    assert totals[3] == frozenset({Fact("total", "568", 2)})


def test_expert_trace_helper_replays_real_simulator() -> None:
    task = _task("calculate")
    trace = _expert_trace(task)
    assert trace[0]["observation"] == "RESULT: 373"
    assert trace[-1]["observation"] == "FINISHED"


def _mutated_trace(family: str, index: int, replacement: str) -> tuple[object, list[dict]]:
    task = _task(family)
    trace = _expert_trace(task)
    trace[index] = {**trace[index], "thought": replacement}
    return task, trace


def test_check_trajectory_detects_each_violation_kind_once() -> None:
    task = _task("ledger_reconcile")
    clean = _expert_trace(task)
    assert check_trajectory(task, clean, keep_last=1).clean

    copied = [dict(step) for step in clean]
    copied_note = "Continuing invoice inspection before reconciling approved amounts."
    copied[1]["thought"] = copied_note
    copied[2]["thought"] = copied_note
    assert check_trajectory(task, copied, keep_last=1).counts == {"verbatim_copy": 1}

    dropped = [dict(step) for step in clean]
    dropped[7]["thought"] = str(dropped[7]["thought"]).replace("178, ", "")
    assert check_trajectory(task, dropped, keep_last=1).counts == {"value_drop": 1}

    premature = [dict(step) for step in clean]
    premature[4]["thought"] = f"{premature[4]['thought']} Task complete."
    assert check_trajectory(task, premature, keep_last=1).counts == {
        "premature_completion": 1
    }

    count = [dict(step) for step in clean]
    count[4]["thought"] = str(count[4]["thought"]).replace("3 of 6", "2 of 7")
    assert check_trajectory(task, count, keep_last=1).counts == {"count_mismatch": 1}

    stale = [dict(step) for step in clean]
    stale[4]["thought"] = str(stale[4]["thought"]).replace("approved: 178, 40", "approved: 999, 40")
    stale_report = check_trajectory(task, stale, keep_last=1)
    assert stale_report.counts == {"stale_fact": 1}

    queue = [dict(step) for step in clean]
    queue[4]["thought"] = str(queue[4]["thought"]).replace(", invoice-5-598.txt", "")
    assert check_trajectory(task, queue, keep_last=1).counts == {"queue_loss": 1}


def test_check_trajectory_covers_other_structured_fact_families() -> None:
    cases = (
        ("aggregate_report", "metric", "value_drop"),
        ("conditional_update", "load", "value_drop"),
        ("cross_reference", "next_key", "value_drop"),
        ("synthesis", "total", "value_drop"),
    )
    for family, fact_kind, kind in cases:
        task = _task(family)
        trace = _expert_trace(task)
        carries = required_carry(task, keep_last=0)
        index, facts = next(
            (index, facts)
            for index, facts in reversed(tuple(carries.items()))
            if any(fact.kind == fact_kind for fact in facts)
        )
        candidates = [fact for fact in facts if fact.kind == fact_kind]
        fact = next(
            fact
            for fact in candidates
            if sum(other.value == fact.value for other in candidates) == 1
        )
        old = fact.value.rsplit("/", 1)[-1]
        assert old in str(trace[index]["thought"])
        trace[index]["thought"] = str(trace[index]["thought"]).replace(old, "999999", 1)
        assert kind in check_trajectory(task, trace, keep_last=0).counts


def test_check_trajectory_visible_observation_satisfies_value_and_parse_errors_are_safe() -> None:
    task = _task("synthesis")
    trace = _expert_trace(task)
    trace[3]["thought"] = "Task complete."
    report = check_trajectory(task, trace, keep_last=1)
    assert "value_drop" not in report.counts

    malformed = [
        {"index": "bad", "thought": None, "action": None, "observation": None, "raw": "{"}
    ]
    assert isinstance(check_trajectory(task, malformed, keep_last=1), IntegrityReport)


def test_check_trajectory_deduplicates_patterns_and_orders_first_violation() -> None:
    task = _task("ledger_reconcile")
    trace = _expert_trace(task)
    trace[4]["thought"] = "Task complete. COMPLETE. pending: none."
    trace[5]["thought"] = trace[4]["thought"]
    report = check_trajectory(task, trace, keep_last=1)
    assert report.counts["premature_completion"] == 2
    assert report.first_violation == Violation(
        4, "value_drop", "missing required values: 178"
    )
    assert [violation.kind for violation in report.violations[:3]] == [
        "value_drop",
        "premature_completion",
        "queue_loss",
    ]


def _trajectory(task, *, success: bool = True) -> Trajectory:
    return Trajectory(
        task_id=task.task_id,
        family=task.family,
        variant=task.variant,
        label="fake",
        prompt=task.prompt,
        steps=_expert_trace(task),
        verdict={
            "success": success,
            "clean": success,
            "calls": len(task.steps),
            "schema_failures": 0,
            "executable_calls": len(task.steps),
            "errors": 0,
            "recovered_errors": 0,
            "reasons": [] if success else ["wrong answer: fake"],
        },
        turns=len(task.steps),
        valid_turns=len(task.steps),
    )


def test_evaluate_tasks_attaches_difficulty_and_integrity(monkeypatch) -> None:
    task = _task("calculate")
    trajectory = _trajectory(task)
    monkeypatch.setattr(evaluate, "make_sampler", lambda temperature: object())
    monkeypatch.setattr(evaluate, "run_task", lambda *args, **kwargs: trajectory)

    result = evaluate.evaluate_tasks(None, None, [task], label="fake", quiet=True)

    assert result == [trajectory]
    assert trajectory.difficulty == task.difficulty
    assert trajectory.integrity["clean"] is True


def test_failure_reason_prioritises_parse_integrity_loop_exhaustion_and_verdict() -> None:
    task = _task("calculate")
    trajectory = _trajectory(task, success=False)
    trajectory.parse_error = "bad"
    trajectory.loop_detected = True
    trajectory.exhausted = True
    trajectory.integrity = {
        "clean": False,
        "first_violation": {"step": 1, "kind": "value_drop", "detail": "missing 373"},
        "counts": {"value_drop": 1},
        "violations": [],
    }
    assert evaluate.failure_reason(trajectory) == "parse error"
    trajectory.parse_error = None
    assert evaluate.failure_reason(trajectory) == "value drop"
    trajectory.integrity = {"clean": True, "first_violation": None, "counts": {}, "violations": []}
    assert evaluate.failure_reason(trajectory) == "repetition loop"
    trajectory.loop_detected = False
    assert evaluate.failure_reason(trajectory) == "step budget exhausted"
    trajectory.exhausted = False
    assert evaluate.failure_reason(trajectory) == "wrong answer"


def test_summarize_reports_integrity_totals_by_kind_family_and_failed_denominator() -> None:
    ledger = _trajectory(_task("ledger_reconcile"), success=False)
    ledger.integrity = {
        "clean": False,
        "violations": [
            {"step": 4, "kind": "value_drop", "detail": "a"},
            {"step": 5, "kind": "value_drop", "detail": "b"},
        ],
        "first_violation": {"step": 4, "kind": "value_drop", "detail": "a"},
        "counts": {"value_drop": 2},
    }
    calculate = _trajectory(_task("calculate"), success=True)
    calculate.integrity = {"clean": True, "violations": [], "first_violation": None, "counts": {}}

    summary = evaluate.summarize([ledger, calculate])

    assert summary["integrity"] == {
        "clean_trajectories": 1,
        "clean_rate": 0.5,
        "affected_trajectories": 1,
        "violations": 2,
        "by_kind": {"value_drop": {"violations": 2, "affected_trajectories": 1}},
        "by_family": {
            "calculate": {
                "trajectories": 1,
                "clean_trajectories": 1,
                "clean_rate": 1.0,
                "violations": 0,
                "affected_trajectories": 0,
            },
            "ledger_reconcile": {
                "trajectories": 1,
                "clean_trajectories": 0,
                "clean_rate": 0.0,
                "violations": 2,
                "affected_trajectories": 1,
            },
        },
        "failed_trajectories": 1,
        "failed_with_violation": 1,
        "failure_explained_rate": 1.0,
    }


def test_write_report_serialises_dynamic_integrity_fields(tmp_path: Path) -> None:
    trajectory = _trajectory(_task("calculate"))
    trajectory.difficulty = 2
    trajectory.integrity = {"clean": True, "violations": [], "first_violation": None, "counts": {}}
    destination = tmp_path / "eval.json"

    evaluate.write_report(destination, {"tasks": 1}, [trajectory])

    record = json.loads(destination.read_text())["trajectories"][0]
    assert record["difficulty"] == 2
    assert record["integrity"]["clean"] is True
    assert trajectory.as_dict()["difficulty"] == record["difficulty"]
    assert trajectory.as_dict()["integrity"] == record["integrity"]


def test_report_render_adds_backward_compatible_integrity_clean_column() -> None:
    base = {
        "_file": "run.json",
        "label": "new",
        "split": "test",
        "successes": 1,
        "tasks": 2,
        "success_rate": 0.5,
        "clean_rate": 0.5,
        "valid_action_rate": 1.0,
        "schema_validity_rate": 1.0,
        "executable_call_rate": 1.0,
        "tool_errors": 0,
        "mean_steps": 2.0,
        "by_family": {},
    }
    legacy = {**base, "label": "legacy"}
    current = {**base, "integrity": {"clean_rate": 0.75}}
    rendered = report.render([legacy, current])
    assert "integrity-clean" in rendered
    assert "legacy" in rendered and "-" in rendered
    assert "75%" in rendered


def test_rollout_retains_only_integrity_clean_successes(monkeypatch) -> None:
    from local_llm_lab.pipeline import rollout

    task = _task("calculate")
    violating = _trajectory(task)
    violating.steps[1]["thought"] = violating.steps[0]["thought"]
    clean = _trajectory(task)
    trajectories = iter([violating, clean])
    seeded: list[int] = []
    fake_core = SimpleNamespace(random=SimpleNamespace(seed=seeded.append))
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=fake_core))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)
    monkeypatch.setattr(rollout, "make_sampler", lambda temperature: object())
    monkeypatch.setattr(rollout, "run_task", lambda *args, **kwargs: next(trajectories))
    monkeypatch.setattr(
        rollout,
        "trajectory_rows",
        lambda trajectory, task, keep_last: [
            {"label": trajectory.label, "clean": trajectory is clean}
        ],
    )

    all_trajectories, rows, summary = rollout.collect_rollouts(
        None,
        None,
        [task],
        label="fake",
        samples=2,
        temperature=0.7,
        keep_per_task=1,
        quiet=True,
    )

    assert all_trajectories == [violating, clean]
    assert rows == [{"label": "fake", "clean": True}]
    assert summary["kept_rows"] == 1
    assert summary["tasks_solved_at_least_once"] == 1
    assert violating.integrity["clean"] is False
    assert clean.integrity["clean"] is True


def _write_evaluation(path: Path, task, trace: list[dict], *, success: bool, label: str) -> None:
    trajectory = _trajectory(task, success=success)
    trajectory.steps = trace
    path.write_text(
        json.dumps(
            {
                "summary": {
                    "label": label,
                    "data_seed": 20260902,
                    "keep_last": 0,
                },
                "trajectories": [trajectory.as_dict()],
            }
        )
    )


def test_cli_treats_default_trajectory_difficulty_as_legacy(tmp_path: Path) -> None:
    task = _task("calculate")
    evaluation = tmp_path / "legacy.json"
    _write_evaluation(
        evaluation,
        task,
        _expert_trace(task),
        success=True,
        label="legacy",
    )
    saved = json.loads(evaluation.read_text())
    assert saved["trajectories"][0]["difficulty"] == -1

    rendered = _render_evaluations([evaluation], 20260902)

    assert "| legacy | 1/1 | 1/1 | 0 | 0/0 |" in rendered


def test_cli_helper_renders_deterministic_offline_comparison(tmp_path: Path) -> None:
    task = _task("calculate")
    clean = _expert_trace(task)
    affected = [dict(step) for step in clean]
    affected[1]["thought"] = affected[0]["thought"]
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    _write_evaluation(left, task, clean, success=True, label="left")
    _write_evaluation(right, task, affected, success=False, label="right")

    rendered = _render_evaluations([left, right], 1)

    assert rendered == _render_evaluations([left, right], 999)
    for heading in (
        "## Sources and configuration",
        "## Overall",
        "## Per-family",
        "## Violation kinds (affected trajectories)",
        "## Integrity-clean paired flips",
        "## Outcome paired flips",
        "## Memo acceptance checks",
    ):
        assert heading in rendered
    assert "| left | 1/1 | 1/1 | 0 | 0/0 |" in rendered
    assert "| right | 0/1 | 0/1 | 1 | 1/1 |" in rendered
    assert "| clean → affected | 1 |" in rendered
    assert "| success → failure | 1 |" in rendered
    assert "| cross_reference | verbatim_copy | 15 | 0 | FAIL |" in rendered


def test_cli_entry_point_is_registered() -> None:
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert 'agent-v2-integrity = "local_llm_lab.pipeline.integrity:main"' in pyproject


def test_cli_rejects_unpaired_task_sets(tmp_path: Path) -> None:
    first = _task("calculate")
    second = _task("read")
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    _write_evaluation(left, first, _expert_trace(first), success=True, label="left")
    _write_evaluation(right, second, _expert_trace(second), success=True, label="right")
    with pytest.raises(ValueError, match="task_id sets differ"):
        _render_evaluations([left, right], 20260902)


def test_retroactive_saved_evaluations_match_memo_contract(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    run_b = root / "outputs/agent-v2b/evals/runB-test180.json"
    run_c = root / "outputs/agent-v2c/evals/best-adapter-test.json"
    if not run_b.exists() or not run_c.exists():
        pytest.skip("saved B/C evaluations are not available")

    rendered = _render_evaluations([run_b, run_c], 20260902)
    destination = tmp_path / "comparison.md"
    destination.write_text(rendered)

    for row in (
        "| best-adapter | cross_reference | verbatim_copy | 15 |",
        "| best-adapter | batch_update | premature_completion | 14 |",
        "| best-adapter | ledger_reconcile | value_drop | 6 |",
        "| runB-180 | cross_reference | 14/15 | 1 |",
        "| best-adapter | cross_reference | 0/15 | 15 |",
        "| runB-180 | ledger_reconcile | 13/15 | 2 |",
        "| best-adapter | ledger_reconcile | 9/15 | 6 |",
        "| best-adapter | aggregate_report | 0/15 | 15 |",
        "| best-adapter | batch_update | 1/15 | 14 |",
        "| best-adapter | conditional_update | 3/15 | 12 |",
    ):
        assert row in rendered
    assert rendered.count("| PASS |") == 3
