from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest


@dataclass(frozen=True)
class _Task:
    task_id: str
    family: str
    prompt: str = "task"
    steps: tuple[object, ...] = ()


def _payload(trajectories, *, data_seed=17):
    return {"data_seed": data_seed, "trajectories": trajectories}


def _drop_report(step, values=None):
    """A fake integrity report with one value_drop, in the detail form integrity writes."""
    violation = SimpleNamespace(kind="value_drop", step=step)
    if values is not None:
        violation.detail = f"missing required values: {values}"
    return SimpleNamespace(violations=(violation,))


def _judgements(step, values="42", *, head_step=None, head_values=None):
    """A (bound, head) judgement pair; stable unless a head_* override disagrees."""
    from local_llm_lab.probes import patch

    bound = patch.DropJudgement(step, (values,), "generator_v1 replay")
    head = patch.DropJudgement(
        step if head_step is None else head_step,
        (values if head_values is None else head_values,),
        f"generator_v{patch.GENERATOR_VERSION} HEAD",
    )
    return bound, head


def _scoring_record(step, values="42"):
    from local_llm_lab.probes import patch

    return {
        "scoring_version_stable": True,
        "decision_step_bound": step,
        "decision_step_head": step,
        "dropped_values_bound": [values],
        "dropped_values_head": [values],
        "judged_under_bound": "generator_v1 replay",
        "judged_under_head": f"generator_v{patch.GENERATOR_VERSION} HEAD",
    }


def test_select_patch_cases_intersects_saved_evaluations(monkeypatch) -> None:
    from local_llm_lab.probes import patch

    tasks = {
        "test-aggregate_report-0-clean": _Task("test-aggregate_report-0-clean", "aggregate_report"),
        "test-ledger_reconcile-1-clean": _Task("test-ledger_reconcile-1-clean", "ledger_reconcile"),
    }
    monkeypatch.setattr(
        patch,
        "task_from_id",
        lambda task_id, seed, difficulty: (
            tasks[task_id] if seed == 19 and difficulty == 2 else None
        ),
    )
    # The HEAD scoring judgement (R24) runs at selection; the saved integrity carries no
    # detail here, so the bound values are unknown and the case cannot be stable.
    monkeypatch.setattr(
        patch, "check_trajectory", lambda task, steps, *, keep_last: _drop_report(1, "7")
    )
    passing = _payload(
        [
            {"task_id": key, "verdict": {"success": True}, "steps": [{"thought": "saved note"}]}
            for key in tasks
        ]
        + [{"task_id": "test-other-2-clean", "verdict": {"success": True}}],
        data_seed=11,
    )
    failing = _payload(
        [
            {
                "task_id": key,
                "family": task.family,
                "difficulty": 2,
                "verdict": {"success": False},
                "steps": [{"thought": "before"}, {"thought": "drop"}],
                "integrity": {"violations": [{"kind": "value_drop", "step": 1}]},
            }
            for key, task in tasks.items()
        ],
        data_seed=19,
    )

    cases, provenance = patch.select_patch_cases(passing, failing, keep_last=2)

    assert [(case.task.task_id, case.decision_step) for case in cases] == [
        ("test-aggregate_report-0-clean", 1),
        ("test-ledger_reconcile-1-clean", 1),
    ]
    assert cases[0].failing_steps[0]["thought"] == "before"
    assert cases[0].passing_steps == ({"thought": "saved note"},)
    assert cases[0].bound_judgement == patch.DropJudgement(1, None, "saved evaluation integrity")
    assert cases[0].head_judgement == patch.DropJudgement(
        1, ("7",), f"generator_v{patch.GENERATOR_VERSION} HEAD"
    )
    assert cases[0].scoring_version_stable is False
    assert provenance["data_seeds"] == {
        "passing": {"data_seed": 11, "data_seed_source": "evaluation"},
        "failing": {"data_seed": 19, "data_seed_source": "evaluation"},
    }
    assert provenance["eligibility"]["failing"]["eligibility_source"] == "evaluation"


def test_select_patch_cases_rejects_malformed_or_mismatched_payloads() -> None:
    from local_llm_lab.probes import patch

    with pytest.raises(ValueError, match="trajectories"):
        patch.select_patch_cases({}, {}, keep_last=2)
    with pytest.raises(ValueError, match="data_seed"):
        patch.select_patch_cases(_payload([], data_seed="bad"), _payload([]), keep_last=2)


def test_data_seed_override_is_refused_when_it_conflicts_with_the_saved_field() -> None:
    from local_llm_lab.probes import patch

    with pytest.raises(ValueError, match=r"passing evaluation data_seed 17 conflicts with --data-seed 23"):
        patch.select_patch_cases(_payload([]), _payload([]), keep_last=2, data_seed=23)


def test_data_seed_is_required_from_exactly_one_source() -> None:
    from local_llm_lab.probes import patch

    with pytest.raises(ValueError, match=r"passing evaluation lacks data_seed and no --data-seed"):
        patch.select_patch_cases({"trajectories": []}, _payload([]), keep_last=2)
    with pytest.raises(ValueError, match=r"--data-seed override must be an integer"):
        patch.select_patch_cases(_payload([]), _payload([]), keep_last=2, data_seed=True)


def test_data_seed_override_fills_only_the_absent_field_and_records_the_source(monkeypatch) -> None:
    from local_llm_lab.probes import patch

    seen = []
    monkeypatch.setattr(
        patch,
        "task_from_id",
        lambda task_id, seed, difficulty: seen.append(seed)
        or _Task("test-aggregate_report-0-clean", "aggregate_report"),
    )
    monkeypatch.setattr(patch, "check_trajectory", lambda *_args, **_kwargs: _drop_report(1))
    passing = {
        "trajectories": [
            {"task_id": "test-aggregate_report-0-clean", "verdict": {"success": True}}
        ]
    }
    failing = _payload(
        [
            {
                "task_id": "test-aggregate_report-0-clean",
                "difficulty": 2,
                "verdict": {"success": False},
                "steps": [{"thought": "before"}, {"thought": "drop"}],
                "integrity": {"violations": [{"kind": "value_drop", "step": 1}]},
            }
        ],
        data_seed=31,
    )

    cases, provenance = patch.select_patch_cases(passing, failing, keep_last=2, data_seed=31)

    assert provenance["data_seeds"] == {
        "passing": {"data_seed": 31, "data_seed_source": "flag"},
        "failing": {"data_seed": 31, "data_seed_source": "evaluation"},
    }
    assert seen == [31]
    assert len(cases) == 1 and cases[0].passing_steps is None


def _pre_schema_payloads():
    passing = {
        "trajectories": [
            {"task_id": "test-aggregate_report-0-clean", "verdict": {"success": True}}
        ]
    }
    failing = {
        "trajectories": [
            {
                "task_id": "test-aggregate_report-0-clean",
                "verdict": {"success": False},
                "steps": [{"thought": "before"}, {"thought": "drop"}],
            }
        ]
    }
    return passing, failing


def test_select_recomputes_missing_eligibility_fields_under_the_bound_version(monkeypatch) -> None:
    from local_llm_lab.probes import patch

    head_task = _Task("test-aggregate_report-0-clean", "aggregate_report")
    replayed_task = _Task("test-aggregate_report-0-clean", "aggregate_report", prompt="v1")
    monkeypatch.setattr(patch, "task_from_id", lambda *_args: head_task)
    replays = []
    monkeypatch.setattr(
        patch,
        "replay_task_from_id",
        lambda task_id, seed, version, difficulty: replays.append(
            (task_id, seed, version, difficulty)
        )
        or replayed_task,
    )
    checked = []

    def fake_check(task, steps, *, keep_last):
        checked.append((task, len(steps), keep_last))
        return _drop_report(1, "12")

    monkeypatch.setattr(patch, "check_trajectory", fake_check)
    passing, failing = _pre_schema_payloads()

    cases, provenance = patch.select_patch_cases(
        passing, failing, keep_last=2, data_seed=31, generator_version=1
    )

    assert [(case.task, case.decision_step) for case in cases] == [(head_task, 1)]
    assert replays == [("test-aggregate_report-0-clean", 31, 1, 2)]
    # Eligibility is judged under the replayed task, never HEAD; the HEAD judgement that
    # follows is the R24 scoring check over exactly the task ``_is_flip`` will see.
    assert checked == [(replayed_task, 2, 2), (head_task, 2, 2)]
    assert cases[0].bound_judgement == patch.DropJudgement(1, ("12",), "generator_v1 replay")
    assert cases[0].head_judgement == patch.DropJudgement(
        1, ("12",), f"generator_v{patch.GENERATOR_VERSION} HEAD"
    )
    assert cases[0].scoring_version_stable is True
    assert provenance["eligibility"] == {
        "passing": {"eligibility_source": "evaluation"},
        "failing": {
            "eligibility_source": "recomputed",
            "integrity": {"evaluation": 0, "recomputed": 1},
            "difficulty": {"evaluation": 0, "recomputed": 1},
            "recomputed_generator_version": 1,
            "generator_version_source": "flag",
            "generator_version_basis": (
                "explicit --generator-version binding; the evaluation records no "
                "generator_version"
            ),
        },
    }


def test_recompute_excludes_head_only_drops_judged_under_the_bound_version(monkeypatch) -> None:
    from local_llm_lab.probes import patch

    head_task = _Task("test-aggregate_report-0-clean", "aggregate_report")
    replayed_task = _Task("test-aggregate_report-0-clean", "aggregate_report", prompt="v1")
    monkeypatch.setattr(patch, "task_from_id", lambda *_args: head_task)
    monkeypatch.setattr(
        patch, "replay_task_from_id", lambda *_args: replayed_task
    )

    def fake_check(task, steps, *, keep_last):
        # A v4-only artifact: under the bound v1 replay there is NO value drop.
        assert task is replayed_task
        return SimpleNamespace(violations=())

    monkeypatch.setattr(patch, "check_trajectory", fake_check)
    passing, failing = _pre_schema_payloads()

    cases, provenance = patch.select_patch_cases(
        passing, failing, keep_last=2, data_seed=31, generator_version=1
    )

    assert cases == []
    assert provenance["eligibility"]["failing"]["integrity"] == {
        "evaluation": 0,
        "recomputed": 1,
    }
    assert provenance["eligibility"]["failing"]["recomputed_generator_version"] == 1


def test_generator_version_binding_resolves_records_conflicts_and_fails_closed(monkeypatch) -> None:
    from local_llm_lab.probes import patch

    head_task = _Task("test-aggregate_report-0-clean", "aggregate_report")
    replayed_task = _Task("test-aggregate_report-0-clean", "aggregate_report", prompt="v2")
    monkeypatch.setattr(patch, "task_from_id", lambda *_args: head_task)
    replays = []
    monkeypatch.setattr(
        patch,
        "replay_task_from_id",
        lambda task_id, seed, version, difficulty: replays.append(version) or replayed_task,
    )
    monkeypatch.setattr(
        patch,
        "check_trajectory",
        lambda task, steps, *, keep_last: SimpleNamespace(
            violations=(SimpleNamespace(kind="value_drop", step=1),)
        ),
    )
    passing, failing = _pre_schema_payloads()

    recorded = {**failing, "summary": {"generator_version": 2}}
    _cases, provenance = patch.select_patch_cases(
        passing, recorded, keep_last=2, data_seed=31
    )
    assert replays == [2]
    assert provenance["eligibility"]["failing"]["recomputed_generator_version"] == 2
    assert provenance["eligibility"]["failing"]["generator_version_source"] == "evaluation"
    assert (
        provenance["eligibility"]["failing"]["generator_version_basis"]
        == "recorded in the failing evaluation"
    )

    with pytest.raises(
        ValueError, match=r"generator_version 2 conflicts with --generator-version 1"
    ):
        patch.select_patch_cases(
            passing, recorded, keep_last=2, data_seed=31, generator_version=1
        )

    with pytest.raises(
        ValueError, match=r"records no generator_version and no --generator-version"
    ):
        patch.select_patch_cases(passing, failing, keep_last=2, data_seed=31)

    with pytest.raises(ValueError, match=r"--generator-version binding must be an integer"):
        patch.select_patch_cases(
            passing, failing, keep_last=2, data_seed=31, generator_version=True
        )


def test_verdict_filter_restricts_selection_to_the_spec_universe(monkeypatch) -> None:
    from local_llm_lab.probes import patch

    monkeypatch.setattr(
        patch,
        "task_from_id",
        lambda task_id, seed, difficulty: _Task(task_id, "aggregate_report"),
    )
    monkeypatch.setattr(patch, "check_trajectory", lambda *_args, **_kwargs: _drop_report(1))
    record = {
        "difficulty": 2,
        "steps": [{"thought": "before"}, {"thought": "drop"}],
        "integrity": {"violations": [{"kind": "value_drop", "step": 1}]},
    }
    passing = _payload(
        [
            {"task_id": task_id, "verdict": {"success": True}}
            for task_id in (
                "test-aggregate_report-0-clean",
                "test-aggregate_report-1-clean",
                "test-aggregate_report-2-clean",
            )
        ]
    )
    failing = _payload(
        [
            # Dropped a value but ultimately succeeded: outside SPEC-004 §5's universe.
            {**record, "task_id": "test-aggregate_report-0-clean", "verdict": {"success": True}},
            # No verdict at all: not provably failing, excluded.
            {**record, "task_id": "test-aggregate_report-1-clean"},
            {**record, "task_id": "test-aggregate_report-2-clean", "verdict": {"success": False}},
        ],
        data_seed=31,
    )

    cases, provenance = patch.select_patch_cases(passing, failing, keep_last=2)

    assert [case.task.task_id for case in cases] == ["test-aggregate_report-2-clean"]
    assert provenance["eligibility"]["failing"]["integrity"] == {
        "evaluation": 1,
        "recomputed": 0,
    }


def test_saved_eligibility_fields_win_and_recomputation_never_runs(monkeypatch) -> None:
    from local_llm_lab.probes import patch

    monkeypatch.setattr(
        patch,
        "task_from_id",
        lambda task_id, seed, difficulty: _Task(task_id, "aggregate_report"),
    )
    monkeypatch.setattr(
        patch,
        "replay_task_from_id",
        lambda *_args, **_kwargs: pytest.fail("recomputation must not run for saved fields"),
    )
    checked = []

    def fake_check(task, steps, *, keep_last):
        checked.append(task)
        return _drop_report(1, "12")

    monkeypatch.setattr(patch, "check_trajectory", fake_check)
    passing = {
        "trajectories": [
            {"task_id": "test-aggregate_report-0-clean", "verdict": {"success": True}}
        ]
    }
    failing = {
        "trajectories": [
            {
                "task_id": "test-aggregate_report-0-clean",
                "difficulty": 2,
                "verdict": {"success": False},
                "steps": [{"thought": "before"}, {"thought": "drop"}],
                "integrity": {
                    "violations": [
                        {"kind": "value_drop", "step": 1, "detail": "missing required values: 12"}
                    ]
                },
            }
        ]
    }

    cases, provenance = patch.select_patch_cases(passing, failing, keep_last=2, data_seed=31)

    assert len(cases) == 1 and cases[0].decision_step == 1
    # The only checker call is the R24 HEAD scoring judgement over the case's own task;
    # the bound judgement came from the saved integrity block, detail included.
    assert checked == [cases[0].task]
    assert cases[0].bound_judgement == patch.DropJudgement(
        1, ("12",), "saved evaluation integrity"
    )
    assert cases[0].scoring_version_stable is True
    assert provenance["eligibility"]["failing"] == {
        "eligibility_source": "evaluation",
        "integrity": {"evaluation": 1, "recomputed": 0},
        "difficulty": {"evaluation": 1, "recomputed": 0},
    }


def test_mixed_eligibility_sources_are_summarised_and_counted(monkeypatch) -> None:
    from local_llm_lab.probes import patch

    monkeypatch.setattr(
        patch,
        "task_from_id",
        lambda task_id, seed, difficulty: _Task(task_id, "aggregate_report"),
    )
    monkeypatch.setattr(
        patch,
        "replay_task_from_id",
        lambda task_id, seed, version, difficulty: _Task(task_id, "aggregate_report"),
    )
    monkeypatch.setattr(
        patch,
        "check_trajectory",
        lambda *_args, **_kwargs: SimpleNamespace(
            violations=(SimpleNamespace(kind="value_drop", step=0),)
        ),
    )
    passing = {
        "trajectories": [
            {"task_id": "test-aggregate_report-0-clean", "verdict": {"success": True}},
            {"task_id": "test-aggregate_report-0001-clean", "verdict": {"success": True}},
        ]
    }
    failing = {
        "trajectories": [
            {
                "task_id": "test-aggregate_report-0-clean",
                "difficulty": 2,
                "verdict": {"success": False},
                "steps": [{"thought": "before"}, {"thought": "drop"}],
                "integrity": {"violations": [{"kind": "value_drop", "step": 1}]},
            },
            {
                "task_id": "test-aggregate_report-0001-clean",
                "verdict": {"success": False},
                "steps": [{"thought": "drop"}],
            },
        ]
    }

    cases, provenance = patch.select_patch_cases(
        passing, failing, keep_last=2, data_seed=31, generator_version=1
    )

    assert [case.decision_step for case in cases] == [1, 0]
    assert provenance["eligibility"]["failing"] == {
        "eligibility_source": "mixed",
        "integrity": {"evaluation": 1, "recomputed": 1},
        "difficulty": {"evaluation": 1, "recomputed": 1},
        "recomputed_generator_version": 1,
        "generator_version_source": "flag",
        "generator_version_basis": (
            "explicit --generator-version binding; the evaluation records no "
            "generator_version"
        ),
    }


def test_scoring_stability_records_both_judgements_and_flags_disagreement(monkeypatch) -> None:
    from local_llm_lab.probes import patch

    monkeypatch.setattr(
        patch, "task_from_id", lambda task_id, *_args: _Task(task_id, "aggregate_report")
    )
    monkeypatch.setattr(
        patch,
        "replay_task_from_id",
        lambda task_id, *_args: _Task(task_id, "aggregate_report", prompt="v1"),
    )
    verdicts = {
        # (bound step, bound values), (head step, head values)
        "test-aggregate_report-0-clean": ((1, "12"), (1, "12")),  # identical: stable
        "test-aggregate_report-1-clean": ((1, "12"), (2, "12")),  # decision step differs
        "test-aggregate_report-2-clean": ((1, "12"), (1, "13")),  # dropped value differs
    }

    def fake_check(task, steps, *, keep_last):
        bound, head = verdicts[task.task_id]
        return _drop_report(*(bound if task.prompt == "v1" else head))

    monkeypatch.setattr(patch, "check_trajectory", fake_check)
    steps = [{"thought": "a"}, {"thought": "b"}, {"thought": "c"}]
    passing = {
        "trajectories": [{"task_id": key, "verdict": {"success": True}} for key in verdicts]
    }
    failing = {
        "trajectories": [
            {"task_id": key, "verdict": {"success": False}, "steps": steps} for key in verdicts
        ]
    }

    cases, _provenance = patch.select_patch_cases(
        passing, failing, keep_last=2, data_seed=31, generator_version=1
    )

    assert [case.decision_step for case in cases] == [1, 1, 1]
    assert [case.scoring_version_stable for case in cases] == [True, False, False]
    assert cases[1].scoring_record() == {
        "scoring_version_stable": False,
        "decision_step_bound": 1,
        "decision_step_head": 2,
        "dropped_values_bound": ["12"],
        "dropped_values_head": ["12"],
        "judged_under_bound": "generator_v1 replay",
        "judged_under_head": f"generator_v{patch.GENERATOR_VERSION} HEAD",
    }
    assert cases[2].scoring_record()["dropped_values_head"] == ["13"]
    assert cases[0].scoring_record() == {**_scoring_record(1, "12")}


def test_scoring_stability_fails_closed_without_recorded_values_or_judgements() -> None:
    from local_llm_lab.probes import patch

    head = patch.DropJudgement(1, ("12",), "head")
    assert patch.scoring_version_stable(patch.DropJudgement(1, ("12",), "bound"), head)
    assert not patch.scoring_version_stable(patch.DropJudgement(1, None, "bound"), head)
    assert not patch.scoring_version_stable(patch.DropJudgement(None, None, "bound"), head)
    assert not patch.scoring_version_stable(
        patch.DropJudgement(1, ("12", "13"), "bound"), head
    )
    assert patch.scoring_version_stable(
        patch.DropJudgement(1, ("13", "12"), "bound"),
        patch.DropJudgement(1, ("12", "13"), "head"),
    )
    with pytest.raises(ValueError, match="carries no bound/HEAD value-drop judgements"):
        patch.PatchCase(_Task("test-aggregate_report-0-clean", "aggregate_report"), 0, ()).scoring_version_stable


def test_dry_selection_recomputes_eligibility_for_evaluations_shaped_like_the_saved_runs() -> None:
    import re

    from local_llm_lab.pipeline.env import Simulator
    from local_llm_lab.pipeline.tasks import make_tasks
    from local_llm_lab.probes import patch

    task = next(item for item in make_tasks("test", 12, 11) if item.family == "ledger_reconcile")
    trace = []
    simulator = Simulator.for_task(task)
    for index, step in enumerate(task.steps):
        observation = simulator.execute(step.action)
        trace.append(
            {
                "index": index,
                "thought": step.thought,
                "action": {"name": step.action.name, "arguments": step.action.arguments},
                "observation": observation,
                "raw": None,
            }
        )
    mutated = [dict(step) for step in trace]
    mutated_index, match = next(
        (index, found)
        for index, step in enumerate(mutated)
        if (found := re.search(r"approved:\s*(\d+), ", str(step["thought"])))
    )
    thought = str(mutated[mutated_index]["thought"])
    mutated[mutated_index]["thought"] = thought[: match.start(1)] + thought[match.end(1) + 2 :]
    passing = {
        "trajectories": [
            {
                "task_id": task.task_id,
                "family": task.family,
                "variant": "clean",
                "verdict": {"success": True},
                "steps": trace,
            }
        ]
    }
    failing = {
        "trajectories": [
            {
                "task_id": task.task_id,
                "family": task.family,
                "variant": "clean",
                "verdict": {"success": False},
                "steps": mutated,
            }
        ]
    }

    # The fixture was generated at HEAD, so its generation-version binding is HEAD's.
    cases, provenance = patch.select_patch_cases(
        passing, failing, keep_last=1, data_seed=11, generator_version=patch.GENERATOR_VERSION
    )

    assert [(case.task.task_id, case.decision_step) for case in cases] == [
        (task.task_id, mutated_index)
    ]
    assert provenance["data_seeds"] == {
        "passing": {"data_seed": 11, "data_seed_source": "flag"},
        "failing": {"data_seed": 11, "data_seed_source": "flag"},
    }
    assert provenance["eligibility"]["failing"] == {
        "eligibility_source": "recomputed",
        "integrity": {"evaluation": 0, "recomputed": 1},
        "difficulty": {"evaluation": 0, "recomputed": 1},
        "recomputed_generator_version": patch.GENERATOR_VERSION,
        "generator_version_source": "flag",
        "generator_version_basis": (
            "explicit --generator-version binding; the evaluation records no "
            "generator_version"
        ),
    }
    _failing, counterfactual, note_provenance = patch.replay_counterfactual(cases[0])
    assert note_provenance["counterfactual_source"] == "passing_transcript"
    assert task.steps[mutated_index - 1].thought in counterfactual[2 * mutated_index]["content"]
    # Bound and HEAD are the same generator here, so the real checker's detail must parse to
    # the removed value on both sides and the case must be scoring-version stable.
    assert cases[0].scoring_version_stable is True
    assert cases[0].bound_judgement.dropped_values == cases[0].head_judgement.dropped_values
    assert match.group(1) in cases[0].bound_judgement.dropped_values


_SAVED_EVALS = (
    Path("outputs/agent-v2b/evals/runB-test180.json"),
    Path("outputs/agent-v2c/evals/best-adapter-test.json"),
)


@pytest.mark.skipif(
    not all(path.is_file() for path in _SAVED_EVALS),
    reason="protected saved evaluations not present on this checkout",
)
def test_real_saved_evaluations_select_exactly_the_spec_universe() -> None:
    """Pin the Director's retry selection: SPEC-004 §5's five cases under the v1 binding.

    Reads the protected evidence read-only. The seed and version literals here are
    fixture data: the runs' recorded configuration, not constants used by the tool.
    """
    import json

    from local_llm_lab.probes import patch

    passing = json.loads(_SAVED_EVALS[0].read_text(encoding="utf-8"))
    failing = json.loads(_SAVED_EVALS[1].read_text(encoding="utf-8"))

    cases, provenance = patch.select_patch_cases(
        passing, failing, keep_last=2, data_seed=20260902, generator_version=1
    )

    assert len(cases) == 5
    assert sorted(case.decision_step for case in cases) == [6, 7, 7, 7, 7]
    assert provenance["data_seeds"]["failing"]["data_seed_source"] == "flag"
    assert provenance["eligibility"]["failing"]["eligibility_source"] == "recomputed"
    assert provenance["eligibility"]["failing"]["integrity"] == {
        "evaluation": 0,
        "recomputed": 5,
    }
    assert provenance["eligibility"]["failing"]["recomputed_generator_version"] == 1
    assert provenance["eligibility"]["failing"]["generator_version_source"] == "flag"
    for case in cases:
        _failing, _counterfactual, note_provenance = patch.replay_counterfactual(case)
        assert note_provenance["counterfactual_source"] == "passing_transcript"
    # R24, as measured on these files: every case's decision step and dropped value are
    # identical under the v1 replay and HEAD, so all five are scoring-version stable.
    records = {case.task.task_id: case.scoring_record() for case in cases}
    assert {task_id: record["scoring_version_stable"] for task_id, record in records.items()} == {
        "test-ledger_reconcile-0031-clean": True,
        "test-ledger_reconcile-0127-clean": True,
        "test-ledger_reconcile-0139-clean": True,
        "test-ledger_reconcile-0163-clean": True,
        "test-ledger_reconcile-0175-clean": True,
    }
    assert {
        task_id: (record["decision_step_bound"], record["decision_step_head"], record["dropped_values_bound"])
        for task_id, record in records.items()
    } == {
        "test-ledger_reconcile-0031-clean": (7, 7, ["85"]),
        "test-ledger_reconcile-0127-clean": (7, 7, ["89"]),
        "test-ledger_reconcile-0139-clean": (7, 7, ["100"]),
        "test-ledger_reconcile-0163-clean": (7, 7, ["32"]),
        "test-ledger_reconcile-0175-clean": (6, 6, ["54"]),
    }
    assert all(
        record["dropped_values_bound"] == record["dropped_values_head"] for record in records.values()
    )


def test_position_groups_are_exact_and_fail_closed() -> None:
    from local_llm_lab.probes import patch

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return [ord(char) for char in text]

    tokenizer = Tokenizer()
    text = "SYS|TASK|OLD|NOTE|42|OBS0|OBS1|OBS2|!"
    groups = patch.position_groups(
        tokenizer,
        tokenizer.encode(text),
        prompt_text=text,
        system_text="SYS",
        task_text="TASK",
        previous_notes=["OLD", "NOTE|42"],
        note_values=["42"],
        observations=["OBS0", "OBS1", "OBS2"],
    )

    assert tuple(groups) == patch.PROMPT_GROUPS
    assert groups["note_value_tokens"] == (18, 19)
    assert groups["last_two_observations"] == (26, 27, 28, 29, 31, 32, 33, 34)
    assert groups["final_token"] == (36,)
    assert set(groups["note_value_tokens"]) <= set(groups["previous_notes"])
    with pytest.raises(ValueError, match="missing"):
        patch.position_groups(
            tokenizer,
            tokenizer.encode("SYS|TASK"),
            prompt_text="SYS|TASK",
            system_text="SYS",
            task_text="TASK",
            previous_notes=["MISSING"],
            note_values=[],
            observations=[],
        )


@pytest.mark.parametrize(
    ("family", "note", "expected_values"),
    [
        (
            "aggregate_report",
            "values so far: 17, 29; split after 2 of 4. First subtotal = 46; "
            "computing 30 + 40.",
            "172946",
        ),
        (
            "ledger_reconcile",
            "Invoices read: 3 of 3. approved: 12, 18; held (skip): 9. "
            "Summing approved amounts.",
            "1218",
        ),
    ],
)
def test_groups_for_extracts_canonical_family_value_tokens(
    family, note, expected_values
) -> None:
    from local_llm_lab.agent_protocol import Action
    from local_llm_lab.pipeline.protocol import assistant_message, tool_message
    from local_llm_lab.probes import patch

    class CharacterTokenizer:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return [ord(character) for character in text]

    tokenizer = CharacterTokenizer()
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": f"TASK-{family}"},
        assistant_message("Plan: inspect the state.", Action("list_files", {"directory": "/tmp"})),
        tool_message("list_files", "OBS0"),
        assistant_message(note, Action("calculate", {"expression": "1 + 1"})),
        tool_message("calculate", "OBS1"),
    ]
    prompt = "|".join(message["content"] for message in messages)

    prompt_ids = tokenizer.encode(prompt)
    groups, repairs, values = patch._groups_for(
        tokenizer, prompt_ids, messages, prompt_text=prompt, keep_last=2
    )
    assert repairs == {name: 0 for name in patch.PROMPT_GROUPS}

    assert all(groups[name] for name in patch.PROMPT_GROUPS)
    # Every value keeps its own span, in note order, for the R25 pairwise alignment.
    assert tuple(values) == tuple(patch._note_values(note))
    assert tuple(position for span in values.values() for position in span) == (
        groups["note_value_tokens"]
    )
    value_text = "".join(chr(prompt_ids[position]) for position in groups["note_value_tokens"])
    assert value_text == expected_values
    assert all(character.isdigit() for character in value_text)
    unrelated = "split after 2 of 4" if family == "aggregate_report" else "held (skip): 9"
    unrelated_start = prompt.index(unrelated, prompt.index(note))
    unrelated_positions = set(range(unrelated_start, unrelated_start + len(unrelated)))
    assert not unrelated_positions & set(groups["note_value_tokens"])


def test_random_positions_are_seeded_unique_and_distinct() -> None:
    from local_llm_lab.probes import patch

    first = patch.random_control_positions(range(12), (2, 3, 4), seed=9, label="case")
    second = patch.random_control_positions(range(12), (2, 3, 4), seed=9, label="case")

    assert first == second
    assert len(first) == 3 and len(set(first)) == 3
    assert set(first).isdisjoint({2, 3, 4})


def test_greedy_generate_uses_cached_masked_forwards(monkeypatch) -> None:
    from local_llm_lab.probes import patch

    class Tokenizer:
        def decode(self, ids):
            return "".join(chr(index) for index in ids)

    class View:
        num_layers = 1

        def __init__(self):
            self.lengths = []

        def make_cache(self):
            return [SimpleNamespace(offset=0)]

        def embed(self, ids):
            self.lengths.append(len(ids[0]) if hasattr(ids, "shape") else len(ids))
            return mx.zeros((1, self.lengths[-1], 1), dtype=mx.float32)

        def masks(self, h, cache):
            assert cache is not None
            return {"mask": h.shape[1]}

        def run_block(self, index, h, masks, cache_i):
            assert index == 0 and masks["mask"] == h.shape[1]
            cache_i.offset += h.shape[1]
            return h

        def final_norm(self, h):
            return h

        def unembed(self, h):
            logits = mx.zeros((1, h.shape[1], 128), dtype=mx.float32)
            logits[..., ord("x")] = 1
            return logits

    view = View()
    calls = []
    monkeypatch.setattr(
        patch,
        "turn_is_complete",
        lambda text: calls.append(text) or len(calls) == 2,
    )

    assert patch.greedy_generate(view, Tokenizer(), [1, 2, 3], max_tokens=5) == "xx"
    assert view.lengths == [3, 1]


def test_task_cell_aggregation_is_over_task_booleans_only() -> None:
    from local_llm_lab.probes import patch

    summary = patch.aggregate_task_flips({"task-a": [True, True], "task-b": [False]})

    assert summary["numerator"] == 1
    assert summary["denominator"] == 2
    assert summary["rate"] == 0.5
    assert len(summary["wilson_95"]) == 2


def test_flip_scoring_requires_the_decision_step_value_drop_to_disappear(monkeypatch) -> None:
    from local_llm_lab.probes import patch

    task = _Task("test-aggregate_report-0-clean", "aggregate_report")
    remaining = SimpleNamespace(violations=(SimpleNamespace(kind="value_drop", step=3),))
    moved = SimpleNamespace(violations=(SimpleNamespace(kind="value_drop", step=2),))
    monkeypatch.setattr(patch, "check_trajectory", lambda *_args, **_kwargs: remaining)
    assert not patch._is_flip(task, [], 3, keep_last=2)
    monkeypatch.setattr(patch, "check_trajectory", lambda *_args, **_kwargs: moved)
    assert patch._is_flip(task, [], 3, keep_last=2)


def test_patch_score_treats_an_unparseable_generation_as_non_flip(monkeypatch) -> None:
    from local_llm_lab.probes import patch

    class Hook:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    case = patch.PatchCase(
        _Task("test-aggregate_report-0-clean", "aggregate_report"),
        0,
        ({"thought": "bad"},),
    )
    monkeypatch.setattr(patch, "InjectionHook", Hook)
    monkeypatch.setattr(patch, "greedy_generate", lambda *_args, **_kwargs: "not a turn")
    monkeypatch.setattr(patch, "strip_thinking", lambda raw: (None, raw))
    monkeypatch.setattr(patch, "parse_turn", lambda _raw: (_ for _ in ()).throw(ValueError("bad")))

    assert not patch._score_patch(
        object(), object(), case, layer=1, source_rows=object(), target_positions=(0,),
        failing_ids=[1], keep_last=2, max_tokens=1,
    )


def test_replay_replaces_only_the_immediately_previous_note(monkeypatch) -> None:
    from local_llm_lab.probes import patch

    task = _Task(
        "test-aggregate_report-0-clean",
        "aggregate_report",
        steps=(SimpleNamespace(thought="expert zero"), SimpleNamespace(thought="expert one")),
    )
    case = patch.PatchCase(
        task,
        2,
        (
            {
                "thought": "old zero",
                "action": {"name": "read_file", "arguments": {"path": "a"}},
                "observation": "A",
            },
            {
                "thought": "old one",
                "action": {"name": "read_file", "arguments": {"path": "b"}},
                "observation": "B",
            },
        ),
    )
    monkeypatch.setattr(patch, "render_expert_note", lambda _task, index: f"expert {index}")

    failing, counterfactual, provenance = patch.replay_counterfactual(case)

    assert failing[:2] == counterfactual[:2]
    assert failing[2]["content"] == counterfactual[2]["content"]
    assert "old one" in failing[4]["content"]
    assert "expert 1" in counterfactual[4]["content"]
    assert failing[5:] == counterfactual[5:]
    assert provenance == {
        "counterfactual_source": f"generator_v{patch.GENERATOR_VERSION}",
        "counterfactual_basis": "passing trajectory steps unavailable",
    }


def _case_with_passing_steps(passing_steps):
    from local_llm_lab.probes import patch

    failing_steps = (
        {
            "thought": "old zero",
            "action": {"name": "read_file", "arguments": {"path": "a"}},
            "observation": "A",
        },
        {
            "thought": "old one",
            "action": {"name": "read_file", "arguments": {"path": "b"}},
            "observation": "B",
        },
        {
            "thought": "drop",
            "action": {"name": "calculate", "arguments": {"expression": "1 + 1"}},
        },
    )
    return patch.PatchCase(
        _Task("test-aggregate_report-0-clean", "aggregate_report"),
        2,
        failing_steps,
        passing_steps,
    )


def test_counterfactual_note_prefers_the_passing_runs_saved_note() -> None:
    from local_llm_lab.probes import patch

    case = _case_with_passing_steps(
        (
            {"thought": "pass zero", "action": {"name": "read_file", "arguments": {"path": "a"}}},
            {"thought": "pass one", "action": {"name": "read_file", "arguments": {"path": "b"}}},
            {"thought": "good decision", "action": {"name": "finish", "arguments": {}}},
        )
    )

    failing, counterfactual, provenance = patch.replay_counterfactual(case)

    assert "old one" in failing[4]["content"]
    assert "pass one" in counterfactual[4]["content"]
    assert failing[2]["content"] == counterfactual[2]["content"]
    assert provenance == {
        "counterfactual_source": "passing_transcript",
        "counterfactual_basis": (
            "passing actions equal failing actions at steps 0..1 (name and arguments)"
        ),
    }


@pytest.mark.parametrize(
    ("passing_steps", "reason"),
    [
        (
            (
                {"thought": "pass zero", "action": {"name": "read_file", "arguments": {"path": "a"}}},
                {"thought": "pass one", "action": {"name": "read_file", "arguments": {"path": "z"}}},
            ),
            "passing and failing actions diverge at step 1",
        ),
        (
            ({"thought": "pass zero", "action": {"name": "read_file", "arguments": {"path": "a"}}},),
            "passing trajectory has 1 steps, shorter than the 2-step decision prefix",
        ),
        (
            (
                {"thought": "pass zero"},
                {"thought": "pass one", "action": {"name": "read_file", "arguments": {"path": "b"}}},
            ),
            "passing step 0 is missing an action",
        ),
        (
            (
                {"thought": "pass zero", "action": {"name": "read_file", "arguments": {"path": "a"}}},
                {"action": {"name": "read_file", "arguments": {"path": "b"}}},
            ),
            "passing step 1 has no saved note text",
        ),
    ],
)
def test_counterfactual_note_falls_back_to_the_generator_and_records_why(
    monkeypatch, passing_steps, reason
) -> None:
    from local_llm_lab.probes import patch

    monkeypatch.setattr(patch, "render_expert_note", lambda _task, index: f"expert {index}")
    case = _case_with_passing_steps(passing_steps)

    note, provenance = patch.counterfactual_note(case)

    assert note == "expert 1"
    assert provenance == {
        "counterfactual_source": f"generator_v{patch.GENERATOR_VERSION}",
        "counterfactual_basis": reason,
    }


def test_patch_cli_is_registered_and_validates_before_loading(monkeypatch, tmp_path) -> None:
    from local_llm_lab.probes import patch

    loads = []
    monkeypatch.setattr(patch, "load_model_spec", lambda name: loads.append(name))
    monkeypatch.setattr(
        "sys.argv",
        [
            "agent-v2-probe-patch",
            "--passing-eval",
            str(tmp_path / "missing-b.json"),
            "--failing-eval",
            str(tmp_path / "missing-c.json"),
            "--output",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit):
        patch.main()

    assert loads == []


def test_patch_cli_forwards_registry_spec_and_writes_results(monkeypatch, tmp_path) -> None:
    from local_llm_lab.probes import guard, patch

    passing = tmp_path / "passing.json"
    failing = tmp_path / "failing.json"
    passing.write_text("{}", encoding="utf-8")
    failing.write_text("{}", encoding="utf-8")
    selected = SimpleNamespace(
        hf_id="fake/hf",
        policies={},
        probe_layer_fractions=(0.25, 0.5),
        resolve=lambda *_args: SimpleNamespace(num_layers=4, probe_layers=(1, 2)),
    )
    case = patch.PatchCase(_Task("test-aggregate_report-0-clean", "aggregate_report"), 0, ())
    seeds = {
        "passing": {"data_seed": 41, "data_seed_source": "flag"},
        "failing": {"data_seed": 41, "data_seed_source": "evaluation"},
    }
    eligibility = {
        "passing": {"eligibility_source": "evaluation"},
        "failing": {
            "eligibility_source": "recomputed",
            "integrity": {"evaluation": 0, "recomputed": 1},
            "difficulty": {"evaluation": 0, "recomputed": 1},
            "recomputed_generator_version": patch.GENERATOR_VERSION,
        },
    }
    seen = []
    monkeypatch.setattr(
        patch,
        "select_patch_cases",
        lambda *_args, **kwargs: seen.append(
            ("select", kwargs["data_seed"], kwargs["generator_version"])
        )
        or ([case], {"data_seeds": seeds, "eligibility": eligibility}),
    )
    monkeypatch.setattr(
        patch,
        "load_model_spec",
        lambda name: seen.append(("spec", name)) or selected,
    )
    monkeypatch.setattr(guard, "require_idle_gpu", lambda *_args: seen.append(("guard", None)))
    monkeypatch.setattr(
        patch,
        "resolve_policy",
        lambda name, spec: seen.append(("policy", name, spec)) or None,
    )
    monkeypatch.setattr(
        patch,
        "load_policy",
        lambda given, adapter: seen.append(("load", given.hf_id, adapter))
        or (object(), object(), SimpleNamespace(num_layers=4), object()),
    )
    monkeypatch.setattr(
        patch,
        "run_patch_probe",
        lambda *_args, **kwargs: seen.append(("probe", kwargs["spec"], kwargs["command"]))
        or {"groups": [], "layers": [], "controls": [], "cells": {}},
    )
    monkeypatch.setattr(patch, "render_markdown", lambda _payload: "# fake")
    argv = [
        "agent-v2-probe-patch",
        "--passing-eval",
        str(passing),
        "--failing-eval",
        str(failing),
        "--output",
        str(tmp_path),
        "--model",
        "qwen35-4b",
        "--policy",
        "base",
        "--layers",
        "1,0.5",
        "--seed",
        "9",
        "--data-seed",
        "41",
        "--generator-version",
        "1",
    ]
    monkeypatch.setattr("sys.argv", argv)

    patch.main()

    assert seen[0] == ("spec", "qwen35-4b")
    assert ("select", 41, 1) in seen
    assert ("policy", "base", selected) in seen and ("load", "fake/hf", None) in seen
    assert seen[-1] == ("probe", selected, argv)
    assert (tmp_path / "patch.json").is_file() and (tmp_path / "patch.md").read_text() == "# fake\n"
    payload = json.loads((tmp_path / "patch.json").read_text(encoding="utf-8"))
    assert payload["data_seeds"] == seeds
    assert payload["eligibility"] == eligibility
    assert payload["layer_selection"] == {
        "source": "cli",
        "requested": ["1", "0.5"],
        "fractions": [0.25, 0.5],
        "indices": [1, 2],
        "num_layers": 4,
    }


def test_patch_cli_rejects_malformed_layers_before_model_loading(monkeypatch, tmp_path) -> None:
    from local_llm_lab.probes import patch

    passing = tmp_path / "passing.json"
    failing = tmp_path / "failing.json"
    passing.write_text("{}", encoding="utf-8")
    failing.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        patch,
        "select_patch_cases",
        lambda *_args, **_kwargs: ([object()], {"data_seeds": {}, "eligibility": {}}),
    )
    loads = []
    monkeypatch.setattr(patch, "load_policy", lambda *_args: loads.append(True))
    monkeypatch.setattr(
        "sys.argv",
        [
            "agent-v2-probe-patch",
            "--passing-eval",
            str(passing),
            "--failing-eval",
            str(failing),
            "--output",
            str(tmp_path),
            "--layers",
            "garbage",
        ],
    )

    with pytest.raises(SystemExit):
        patch.main()

    assert loads == []


def test_patch_probe_routes_every_group_and_control_with_exact_trace(monkeypatch) -> None:
    from local_llm_lab.probes import patch

    cases = [
        patch.PatchCase(
            _Task("test-aggregate_report-0-clean", "aggregate_report"),
            0,
            ({"thought": "bad"},),
            None,
            *_judgements(0),
        ),
        patch.PatchCase(
            _Task("test-ledger_reconcile-1-clean", "ledger_reconcile"),
            0,
            ({"thought": "bad"},),
            None,
            *_judgements(0),
        ),
    ]
    # Per-prompt groups (six), plus each note value's own span.  The counterfactual note
    # carries one value the failing note dropped, so the two value cells differ (R25b).
    failing_groups = {
        "system_prompt": (0, 1),
        "task_prompt": (2,),
        "previous_notes": (3, 4),
        "note_value_tokens": (5,),
        "last_two_observations": (6, 7),
        "final_token": (8,),
    }
    counter_groups = {
        "system_prompt": (1, 2),
        "task_prompt": (3,),
        "previous_notes": (4, 5),
        "note_value_tokens": (6, 7),
        "last_two_observations": (7, 8),
        "final_token": (9,),
    }
    failing_values = {"12": (5,)}
    counter_values = {"12": (6,), "99": (7,)}
    # The alignment R25 prescribes for exactly these inputs, written out by hand.
    aligned_source = {
        "system_prompt": (1, 2),
        "task_prompt": (3,),
        "previous_notes": (4, 5),
        "shared_value_tokens": (6,),
        "dropped_value_slot": (7,),  # pooled to one row
        "last_two_observations": (7, 8),
        "final_token": (9,),
    }
    aligned_target = {
        "system_prompt": (0, 1),
        "task_prompt": (2,),
        "previous_notes": (3, 4),
        "shared_value_tokens": (5,),
        # 99 followed 12 in the counterfactual note, so its slot is the separator right
        # after 12's last token in the failing prompt.
        "dropped_value_slot": (6,),
        "last_two_observations": (6, 7),
        "final_token": (8,),
    }
    captures = []
    injected = []

    class View:
        num_layers = 1

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            if text.startswith("failing"):
                return [10 if text.endswith("0") else 110, 11, 12, 13, 14, 15, 16, 17, 18]
            return [20 if text.endswith("0") else 120, 21, 22, 23, 24, 25, 26, 27, 28, 29]

    class Hook:
        def __init__(self, _view, _layer, vector, *, at_positions, replace):
            assert replace
            injected.append((tuple(np.asarray(vector).reshape(-1)), tuple(at_positions)))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(patch.ArchitectureView, "from_model", lambda _model: View())
    monkeypatch.setattr(
        patch,
        "replay_counterfactual",
        lambda case: (
            [{"role": "user", "content": f"f{0 if '-0-' in case.task.task_id else 1}"}],
            [{"role": "user", "content": f"c{0 if '-0-' in case.task.task_id else 1}"}],
            {
                "counterfactual_source": (
                    "passing_transcript"
                    if "-0-" in case.task.task_id
                    else f"generator_v{patch.GENERATOR_VERSION}"
                ),
                "counterfactual_basis": "fixture",
            },
        ),
    )
    monkeypatch.setattr(
        patch,
        "build_prompt",
        lambda _tokenizer, messages, **_kwargs: (
            "failing" if messages[0]["content"].startswith("f") else "counter"
        ) + messages[0]["content"][-1],
    )
    monkeypatch.setattr(
        patch,
        "_groups_for",
        lambda _tokenizer, token_ids, _messages, **_kwargs: (
            (failing_groups, {name: 0 for name in patch.PROMPT_GROUPS}, failing_values)
            if token_ids[0] in (10, 110)
            else (counter_groups, {name: 0 for name in patch.PROMPT_GROUPS}, counter_values)
        ),
    )
    def capture(_view, ids, layers, *, positions):
        captures.append((tuple(ids), positions))
        assert positions == "all"
        base = {10: 100, 20: 200, 110: 300, 120: 400}[ids[0]]
        return {
            layer: mx.array([[base + layer + index] for index in range(len(ids))], dtype=mx.float32)
            for layer in layers
        }
    monkeypatch.setattr(patch, "capture_residuals", capture)
    monkeypatch.setattr(patch, "InjectionHook", Hook)

    def treatment_vector(name, base):
        rows = tuple(float(base + position) for position in aligned_source[name])
        # dropped_value_slot pools its source rows into one; here that is a single row.
        return (sum(rows) / len(rows),) if name == "dropped_value_slot" else rows

    def generation(*args, **_kwargs):
        vector, target = injected[-1]
        own_base = 201 if args[2][0] == 10 else 401
        own_treatments = {
            (treatment_vector(name, own_base), aligned_target[name])
            for name in patch.POSITION_GROUPS
        }
        if (vector, target) in own_treatments:
            return "treatment"
        if target in set(aligned_target.values()):
            return "unrelated"
        return "random"

    monkeypatch.setattr(patch, "greedy_generate", generation)
    monkeypatch.setattr(patch, "strip_thinking", lambda raw: (None, raw))
    monkeypatch.setattr(patch, "parse_turn", lambda raw: SimpleNamespace(thought=raw))
    monkeypatch.setattr(
        patch,
        "_is_flip",
        lambda _task, steps, *_args, **_kwargs: steps[0]["thought"] == "treatment",
    )
    resolved = SimpleNamespace(as_dict=lambda: {"name": "fake"})

    payload = patch.run_patch_probe(
        object(), Tokenizer(), cases, spec=object(), resolved=resolved, layers=[1], policy="base",
        keep_last=2, max_tokens=1, seed=7, command=["patch"],
    )

    assert len(captures) == 4
    assert len(injected) == len(cases) * len(patch.POSITION_GROUPS) * 3
    expected = []
    for name in patch.POSITION_GROUPS:
        for task_id, own_base, other_base in (
            (cases[0].task.task_id, 201, 401),
            (cases[1].task.task_id, 401, 201),
        ):
            target = aligned_target[name]
            source = aligned_source[name]
            expected.append((treatment_vector(name, own_base), target))
            expected.append((treatment_vector(name, other_base), target))
            rng = __import__("random").Random(f"7:{task_id}:1:{name}")
            random_source = tuple(
                sorted(rng.sample(sorted(set(range(10)) - set(source)), len(target)))
            )
            random_target = tuple(
                sorted(rng.sample(sorted(set(range(9)) - set(target)), len(target)))
            )
            expected.append(
                (tuple(float(own_base + position) for position in random_source), random_target)
            )
    assert injected == expected
    final_group = injected[-6:]
    assert [positions for _vector, positions in final_group[::3]] == [(8,), (8,)]
    assert [positions for _vector, positions in final_group[1::3]] == [(8,), (8,)]
    assert all(positions != (8,) for _vector, positions in final_group[2::3])
    assert payload["cells"]["1:system_prompt"]["treatment"]["rate"] == 1.0
    assert payload["cells"]["1:system_prompt"]["controls"]["unrelated_task"]["rate"] == 0.0
    assert payload["cells"]["1:system_prompt"]["controls"]["random_positions"]["rate"] == 0.0
    assert payload["cells"]["1:final_token"]["treatment"]["denominator"] == 2
    assert set(payload["cells"]["1:final_token"]["controls"]) == set(patch.CONTROLS)
    assert payload["artifact_schema"] == patch.ARTIFACT_SCHEMA
    # _groups_for is faked here, so no boundary repair is recorded for either prompt.
    no_repairs = {name: 0 for name in patch.PROMPT_GROUPS}
    alignment = {
        name: patch.GroupAlignment(
            group=name,
            rule="dropped_value_slot"
            if name == "dropped_value_slot"
            else "shared_value_identity"
            if name == "shared_value_tokens"
            else "identity",
            source_positions=() if name == "dropped_value_slot" else aligned_source[name],
            target_positions=aligned_target[name],
            source_cardinality=(
                len(aligned_source[name])
                if name == "dropped_value_slot"
                else len(counter_groups["note_value_tokens"])
                if name == "shared_value_tokens"
                else len(counter_groups[name])
            ),
            target_cardinality=(
                len(aligned_target[name])
                if name == "dropped_value_slot"
                else len(failing_groups["note_value_tokens"])
                if name == "shared_value_tokens"
                else len(failing_groups[name])
            ),
            pooled_sources=((7,),) if name == "dropped_value_slot" else (),
            slots=(
                {
                    "dropped_value": "99",
                    "slot_rule": "after_preceding_shared_value",
                    "preceding_shared_value": "12",
                    "slot_position": 6,
                    "overwritten_token_id": 16,
                    "overwritten_token_text": "",
                    "pooled_source_positions": [7],
                    "pooled_source_token_ids": [27],
                    "pooled_source_text": "",
                },
            )
            if name == "dropped_value_slot"
            else (),
            shared_values=("12",) if name in ("shared_value_tokens", "dropped_value_slot") else (),
            dropped_values=("99",) if name in ("shared_value_tokens", "dropped_value_slot") else (),
            residue={"count": 0, "positions": [], "token_ids": [], "text": ""},
        ).record()
        for name in patch.POSITION_GROUPS
    }
    assert payload["cases"] == [
        {
            "task_id": "test-aggregate_report-0-clean",
            "decision_step": 0,
            "counterfactual_source": "passing_transcript",
            "counterfactual_basis": "fixture",
            **_scoring_record(0),
            "boundary_repairs": {"failing": no_repairs, "counterfactual": no_repairs},
            "alignment": alignment,
        },
        {
            "task_id": "test-ledger_reconcile-1-clean",
            "decision_step": 0,
            "counterfactual_source": f"generator_v{patch.GENERATOR_VERSION}",
            "counterfactual_basis": "fixture",
            **_scoring_record(0),
            "boundary_repairs": {"failing": no_repairs, "counterfactual": no_repairs},
            "alignment": alignment,
        },
    ]
    assert payload["cells"]["1:dropped_value_slot"]["alignment"] == {
        case.task.task_id: alignment["dropped_value_slot"] for case in cases
    }
    assert payload["counterfactual_sources"] == {
        "passing_transcript": 1,
        f"generator_v{patch.GENERATOR_VERSION}": 1,
    }
    assert (payload["stable_cases"], payload["unstable_cases"]) == (2, 0)
    assert payload["excluded_cases"] == []
    assert payload["headline_task_ids"] == payload["selected_task_ids"]
    assert payload["scoring_generator_version"] == patch.GENERATOR_VERSION


def test_patch_probe_refuses_a_source_group_shorter_than_the_target(monkeypatch) -> None:
    """R25(a) aligns a LONGER source on its tail; a shorter one cannot be stretched."""
    from local_llm_lab.probes import patch

    cases = [
        patch.PatchCase(
            _Task("test-aggregate_report-0-clean", "aggregate_report"),
            0,
            ({"thought": "bad"},),
            None,
            *_judgements(0),
        ),
        patch.PatchCase(
            _Task("test-ledger_reconcile-1-clean", "ledger_reconcile"),
            0,
            ({"thought": "bad"},),
            None,
            *_judgements(0),
        ),
    ]
    groups = {name: (0,) for name in patch.PROMPT_GROUPS}
    groups["system_prompt"] = (0, 1)

    class View:
        num_layers = 1

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [0, *range(1, 10)] if text == "f" else [1, *range(2, 11)]

    monkeypatch.setattr(patch.ArchitectureView, "from_model", lambda _model: View())
    monkeypatch.setattr(
        patch,
        "replay_counterfactual",
        lambda _case: (
            [{"role": "user", "content": "f"}],
            [{"role": "user", "content": "c"}],
            {"counterfactual_source": "passing_transcript", "counterfactual_basis": "fixture"},
        ),
    )
    monkeypatch.setattr(
        patch,
        "build_prompt",
        lambda _tokenizer, messages, **_kwargs: messages[0]["content"],
    )
    monkeypatch.setattr(
        patch,
        "_groups_for",
        lambda _tokenizer, ids, _messages, **_kwargs: (
            groups if ids[0] == 0 else {**groups, "system_prompt": (0,)},
            {name: 0 for name in patch.PROMPT_GROUPS},
            {"7": (0,)} if ids[0] == 0 else {"7": (0,), "9": (1,)},
        ),
    )
    monkeypatch.setattr(
        patch,
        "capture_residuals",
        lambda _view, _ids, layers, **_kwargs: {
            layer: mx.zeros((10, 1)) for layer in layers
        },
    )
    resolved = SimpleNamespace(as_dict=lambda: {"name": "fake"})

    with pytest.raises(ValueError, match="treatment.*cardinality"):
        patch.run_patch_probe(
            object(),
            Tokenizer(),
            cases,
            spec=object(),
            resolved=resolved,
            layers=[1],
            policy="base",
            keep_last=2, max_tokens=1, seed=7, command=["patch"],
        )


def _probe_fixture(monkeypatch, cases):
    """Wire every model seam of run_patch_probe to fakes; returns the capture log."""
    from local_llm_lab.probes import patch

    captures = []

    class View:
        num_layers = 1

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return list(range(10)) if text == "f" else list(range(1, 11))

    class Hook:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(patch.ArchitectureView, "from_model", lambda _model: View())
    monkeypatch.setattr(
        patch,
        "replay_counterfactual",
        lambda case: (
            [{"role": "user", "content": "f"}],
            [{"role": "user", "content": "c"}],
            {"counterfactual_source": "passing_transcript", "counterfactual_basis": "fixture"},
        ),
    )
    monkeypatch.setattr(
        patch, "build_prompt", lambda _tokenizer, messages, **_kwargs: messages[0]["content"]
    )
    # Groups are per-prompt; the value maps give the counterfactual one extra value, which is
    # exactly the R25 case: the failing prompt has no position for it.
    monkeypatch.setattr(
        patch,
        "_groups_for",
        lambda _tokenizer, ids, _messages, **_kwargs: (
            {name: (0,) for name in patch.PROMPT_GROUPS},
            {name: 0 for name in patch.PROMPT_GROUPS},
            {"7": (0,)} if ids[0] == 0 else {"7": (0,), "9": (1,)},
        ),
    )

    def capture(_view, ids, layers, *, positions):
        captures.append(tuple(ids))
        return {layer: mx.zeros((10, 1)) for layer in layers}

    monkeypatch.setattr(patch, "capture_residuals", capture)
    monkeypatch.setattr(patch, "InjectionHook", Hook)
    monkeypatch.setattr(patch, "greedy_generate", lambda *_args, **_kwargs: "x")
    monkeypatch.setattr(patch, "strip_thinking", lambda raw: (None, raw))
    monkeypatch.setattr(patch, "parse_turn", lambda raw: SimpleNamespace(thought=raw))
    monkeypatch.setattr(patch, "_is_flip", lambda *_args, **_kwargs: True)
    return Tokenizer(), captures


def _probe_case(index, *, judgements):
    from local_llm_lab.probes import patch

    return patch.PatchCase(
        _Task(f"test-ledger_reconcile-{index}-clean", "ledger_reconcile"),
        1,
        ({"thought": "a"}, {"thought": "drop"}),
        None,
        *judgements,
    )


def test_patch_probe_excludes_unstable_cases_from_the_headline_and_lists_them(monkeypatch) -> None:
    from local_llm_lab.probes import patch

    cases = [
        _probe_case(0, judgements=_judgements(1)),
        _probe_case(1, judgements=_judgements(1, head_step=2)),
        _probe_case(2, judgements=_judgements(1)),
    ]
    tokenizer, captures = _probe_fixture(monkeypatch, cases)
    resolved = SimpleNamespace(as_dict=lambda: {"name": "fake"})

    payload = patch.run_patch_probe(
        object(), tokenizer, cases, spec=object(), resolved=resolved, layers=[1], policy="base",
        keep_last=2, max_tokens=1, seed=7, command=["patch"],
    )

    # Only the two stable cases were prepared (two captures each) and scored.
    assert len(captures) == 4
    assert (payload["stable_cases"], payload["unstable_cases"]) == (2, 1)
    assert payload["selected_task_ids"] == [case.task.task_id for case in cases]
    assert payload["headline_task_ids"] == [cases[0].task.task_id, cases[2].task.task_id]
    assert [record["task_id"] for record in payload["cases"]] == payload["headline_task_ids"]
    assert all(record["scoring_version_stable"] for record in payload["cases"])
    for cell in payload["cells"].values():
        assert cell["treatment"]["denominator"] == 2
        assert all(control["denominator"] == 2 for control in cell["controls"].values())
    assert payload["counterfactual_sources"] == {"passing_transcript": 2}
    assert payload["excluded_cases"] == [
        {
            "task_id": "test-ledger_reconcile-1-clean",
            "decision_step": 1,
            "counterfactual_source": "passing_transcript",
            "counterfactual_basis": "fixture",
            "scoring_version_stable": False,
            "decision_step_bound": 1,
            "decision_step_head": 2,
            "dropped_values_bound": ["42"],
            "dropped_values_head": ["42"],
            "judged_under_bound": "generator_v1 replay",
            "judged_under_head": f"generator_v{patch.GENERATOR_VERSION} HEAD",
            "excluded_reason": "scoring_version_unstable",
        }
    ]


def test_patch_probe_requires_two_stable_cases_for_the_unrelated_task_control(monkeypatch) -> None:
    from local_llm_lab.probes import patch

    cases = [
        _probe_case(0, judgements=_judgements(1)),
        _probe_case(1, judgements=_judgements(1, head_values="43")),
    ]
    tokenizer, _captures = _probe_fixture(monkeypatch, cases)
    resolved = SimpleNamespace(as_dict=lambda: {"name": "fake"})

    with pytest.raises(ValueError, match=r"scoring-version-stable.*1 stable, 1 unstable"):
        patch.run_patch_probe(
            object(), tokenizer, cases, spec=object(), resolved=resolved, layers=[1],
            policy="base", keep_last=2, max_tokens=1, seed=7, command=["patch"],
        )


def test_render_markdown_splits_stable_and_unstable_cases() -> None:
    from local_llm_lab.probes import patch

    stable = {"task_id": "test-ledger_reconcile-0-clean", **_scoring_record(1, "12")}
    unstable = {
        "task_id": "test-ledger_reconcile-1-clean",
        **_scoring_record(1, "12"),
        "scoring_version_stable": False,
        "decision_step_head": 2,
        "dropped_values_head": None,
        "excluded_reason": "scoring_version_unstable",
    }
    payload = {
        "groups": ["system_prompt"],
        "layers": [1],
        "controls": [],
        "cells": {"1:system_prompt": {"treatment": {"rate": 1.0, "wilson_95": [0.5, 1.0]}}},
        "stable_cases": 1,
        "unstable_cases": 1,
        "cases": [stable],
        "excluded_cases": [unstable],
    }

    text = patch.render_markdown(payload)
    lines = text.splitlines()

    assert "Headline over 1 scoring-version-stable case(s); 1 unstable case(s) excluded (R24)." in lines
    assert lines.index("## Treatment flip rate") < lines.index("| 1 | 1.000 [0.500, 1.000] |")
    assert "## Scoring version stability (R24)" in lines
    assert "| test-ledger_reconcile-0-clean | yes | 1 / 1 | 12 / 12 |" in lines
    unstable_header = lines.index("### Unstable cases (excluded from the headline)")
    assert lines.index("| test-ledger_reconcile-1-clean | no | 1 / 2 | 12 / unknown |") > unstable_header
    assert lines.index("| test-ledger_reconcile-0-clean | yes | 1 / 1 | 12 / 12 |") < unstable_header

    without = patch.render_markdown({**payload, "excluded_cases": [], "unstable_cases": 0})
    assert "None." in without.splitlines()


def test_render_markdown_summarises_counterfactual_note_sources() -> None:
    from local_llm_lab.probes import patch

    payload = {
        "groups": ["system_prompt"],
        "layers": [1],
        "controls": [],
        "cells": {"1:system_prompt": {"treatment": {"rate": 1.0, "wilson_95": [0.5, 1.0]}}},
        "counterfactual_sources": {"passing_transcript": 2, "generator_v4": 1},
    }

    text = patch.render_markdown(payload)

    assert "## Counterfactual note sources" in text
    assert "- passing_transcript: 2" in text
    assert "- generator_v4: 1" in text


# --- Position groups on the windowed prompt with a context-dependent tokenizer -----------
#
# The Director's P6 run crashed at ``_groups_for`` on real data: the replayed message list
# carries every observation, but ``build_prompt`` renders ``window_messages`` (stubs for all
# but the last ``keep_last``), and the template wraps each message in newlines so a
# standalone ``encode(text)`` is never a token-substring of the prompt (briefing trap 6).
# The fake below reproduces both: a newline merges into the following character, so
# ``encode("\n" + x) != encode("\n") + encode(x)``.

_FAKE_SPEC = SimpleNamespace(chat=SimpleNamespace(template_kwargs={}, thinking="on"))


class _MergingTokenizer:
    """Context-dependent fake: a newline merges with the character that follows it."""

    _TOKEN = __import__("re").compile(r"\n[^\n]|\n|[^\n]")

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def offsets(self, text):
        return [(match.start(), match.end()) for match in self._TOKEN.finditer(text)]

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [
            self.vocab.setdefault(token, len(self.vocab) + 1) for token in self._TOKEN.findall(text)
        ]

    def decode(self, ids):
        inverse = {index: token for token, index in self.vocab.items()}
        return "".join(inverse[index] for index in ids)

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, **_kwargs):
        from local_llm_lab.pipeline.protocol import generation_suffix

        assert tokenize is False
        parts = []
        for message in messages:
            if message["role"] == "tool":
                parts.append(f"[user]\n<tool_response>\n{message['content']}\n</tool_response>\n[/user]\n")
            else:
                parts.append(f"[{message['role']}]\n{message['content']}\n[/{message['role']}]\n")
        prompt = "".join(parts)
        return prompt + generation_suffix(_FAKE_SPEC) if add_generation_prompt else prompt


def _windowed_messages(*, observations=("OBS zero", "OBS one", "OBS two"), last_value="25"):
    """A replayed ledger case with three observations: more than ``keep_last`` = 2."""
    from local_llm_lab.agent_protocol import Action
    from local_llm_lab.pipeline.protocol import assistant_message, tool_message

    notes = [
        "Invoices read: 1 of 3. approved: 14; held (skip): 9. Reading the next invoice.",
        "Invoices read: 2 of 3. approved: 14, 18; held (skip): 9. Reading the next invoice.",
        f"Invoices read: 3 of 3. approved: 14, 18, {last_value}; held (skip): 9. Summing approved amounts.",
    ]
    # The system text is long enough that every group's random-position control can draw a
    # disjoint pool of equal cardinality from the rest of the prompt.
    system = "SYS rules for the agent. " + " ".join(f"Rule {index} applies." for index in range(24))
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "TASK reconcile the ledger"},
    ]
    for index, (note, observation) in enumerate(zip(notes, observations)):
        messages.append(assistant_message(note, Action("read_file", {"path": f"inv/{index}"})))
        messages.append(tool_message("read_file", observation))
    return messages


def _covering_positions(tokenizer, prompt, text, *, start=0):
    first = prompt.index(text, start)
    last = first + len(text)
    return tuple(
        index
        for index, (begin, end) in enumerate(tokenizer.offsets(prompt))
        if begin < last and end > first
    ), last


def test_groups_for_uses_the_windowed_view_and_char_offsets_with_a_merging_tokenizer() -> None:
    """Causes A and B together: hidden observations are not searched; merges are repaired."""
    from local_llm_lab.pipeline.protocol import build_prompt, window_messages
    from local_llm_lab.probes import patch

    tokenizer = _MergingTokenizer()
    messages = _windowed_messages()
    prompt = build_prompt(tokenizer, messages, spec=_FAKE_SPEC, keep_last=2)
    ids = tokenizer.encode(prompt)

    # The windowed prompt carries a stub for the first observation and the last two verbatim.
    windowed = window_messages(messages, 2)
    assert "OBS zero" not in prompt and windowed[3]["content"] in prompt
    assert all(message["content"] in prompt for message in windowed[4:])
    # Cause B, as measured on the fake: the text is present, its standalone ids are not.
    wrapped = tokenizer.encode("OBS one")
    assert not any(ids[index : index + len(wrapped)] == wrapped for index in range(len(ids)))

    groups, repairs, _values = patch._groups_for(
        tokenizer, ids, messages, prompt_text=prompt, keep_last=2
    )

    assert tuple(groups) == patch.PROMPT_GROUPS
    assert all(groups[name] for name in patch.PROMPT_GROUPS)
    assert repairs == {
        "system_prompt": 1,
        "task_prompt": 1,
        "previous_notes": 3,
        "note_value_tokens": 0,
        "last_two_observations": 2,
        "final_token": 0,
    }
    # (c) Each group is exactly the tokens covering its text in the rendered prompt.
    system, cursor = _covering_positions(tokenizer, prompt, messages[0]["content"])
    task, cursor = _covering_positions(tokenizer, prompt, messages[1]["content"], start=cursor)
    assert groups["system_prompt"] == system
    assert groups["task_prompt"] == task
    notes = ()
    for message in messages[2::2]:
        span, cursor = _covering_positions(tokenizer, prompt, message["content"], start=cursor)
        notes += span
    assert groups["previous_notes"] == notes
    last_note = messages[-2]["content"]
    note_start = prompt.index(last_note)
    values = ()
    for value in ("14", "18", "25"):
        span, _ = _covering_positions(tokenizer, prompt, value, start=note_start)
        values += span
    assert groups["note_value_tokens"] == values
    assert set(groups["note_value_tokens"]) <= set(groups["previous_notes"])
    assert tokenizer.decode([ids[index] for index in groups["note_value_tokens"]]) == "141825"
    observations = ()
    cursor = prompt.index(messages[1]["content"])
    for text in ("OBS one", "OBS two"):
        span, cursor = _covering_positions(tokenizer, prompt, text, start=cursor)
        observations += span
    assert groups["last_two_observations"] == observations
    assert tokenizer.decode([ids[index] for index in groups["last_two_observations"]]) == (
        "\nOBS one\nOBS two"
    )
    assert groups["final_token"] == (len(ids) - 1,)


def test_position_groups_refuses_ambiguous_and_missing_text_on_the_rendered_prompt() -> None:
    from local_llm_lab.pipeline.protocol import build_prompt
    from local_llm_lab.probes import patch

    tokenizer = _MergingTokenizer()
    duplicated = _windowed_messages(observations=("OBS zero", "OBS same", "OBS same"))
    prompt = build_prompt(tokenizer, duplicated, spec=_FAKE_SPEC, keep_last=2)
    with pytest.raises(ValueError, match="ambiguous token span for observation"):
        patch._groups_for(tokenizer, tokenizer.encode(prompt), duplicated, prompt_text=prompt, keep_last=2)

    messages = _windowed_messages()
    prompt = build_prompt(tokenizer, messages, spec=_FAKE_SPEC, keep_last=2)
    ids = tokenizer.encode(prompt)
    with pytest.raises(ValueError, match="missing token span for previous_note"):
        patch.position_groups(
            tokenizer,
            ids,
            prompt_text=prompt,
            system_text=messages[0]["content"],
            task_text=messages[1]["content"],
            previous_notes=["MISSING"],
            note_values=[],
            observations=["OBS one", "OBS two"],
        )
    with pytest.raises(ValueError, match="note_value token span is missing or ambiguous"):
        patch.position_groups(
            tokenizer,
            ids,
            prompt_text=prompt,
            system_text=messages[0]["content"],
            task_text=messages[1]["content"],
            previous_notes=[messages[-2]["content"]],
            note_values=["9999"],
            observations=["OBS one", "OBS two"],
        )
    # Cause A, stated directly: searching for a hidden observation is refused as missing.
    with pytest.raises(ValueError, match="missing token span for observation"):
        patch.position_groups(
            tokenizer,
            ids,
            prompt_text=prompt,
            system_text=messages[0]["content"],
            task_text=messages[1]["content"],
            previous_notes=[messages[-2]["content"]],
            note_values=[],
            observations=["OBS zero", "OBS one", "OBS two"],
        )


def test_groups_for_windows_with_the_threaded_keep_last() -> None:
    from local_llm_lab.pipeline.protocol import build_prompt
    from local_llm_lab.probes import patch

    tokenizer = _MergingTokenizer()
    messages = _windowed_messages()
    prompt = build_prompt(tokenizer, messages, spec=_FAKE_SPEC, keep_last=1)
    ids = tokenizer.encode(prompt)

    groups, _repairs, _values = patch._groups_for(
        tokenizer, ids, messages, prompt_text=prompt, keep_last=1
    )

    assert tokenizer.decode([ids[index] for index in groups["last_two_observations"]]) == "\nOBS two"
    with pytest.raises(ValueError, match="at least one token"):
        patch._groups_for(tokenizer, ids, messages, prompt_text=prompt, keep_last=0)


class _CharacterTokenizer(_MergingTokenizer):
    """Context-free fake sharing the template: only windowing (cause A) can bite."""

    _TOKEN = __import__("re").compile(r"[^\n]|\n")


@pytest.mark.parametrize(
    ("tokenizer_type", "expected_repairs"),
    [
        (_CharacterTokenizer, 0),
        (_MergingTokenizer, 1),
    ],
)
def test_run_patch_probe_resolves_groups_over_the_windowed_prompt(
    monkeypatch, tokenizer_type, expected_repairs
) -> None:
    """(a) The live crash path: real ``build_prompt``/``window_messages``/``position_groups``."""
    from local_llm_lab.probes import patch

    tokenizer = tokenizer_type()

    class View:
        num_layers = 1

    class Hook:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(patch.ArchitectureView, "from_model", lambda _model: View())
    monkeypatch.setattr(
        patch,
        "replay_counterfactual",
        lambda case: (
            _windowed_messages(),
            _windowed_messages(last_value="26"),
            {"counterfactual_source": "passing_transcript", "counterfactual_basis": "fixture"},
        ),
    )
    monkeypatch.setattr(
        patch,
        "capture_residuals",
        lambda _view, ids, layers, **_kwargs: {layer: mx.zeros((len(ids), 1)) for layer in layers},
    )
    monkeypatch.setattr(patch, "InjectionHook", Hook)
    monkeypatch.setattr(patch, "greedy_generate", lambda *_args, **_kwargs: "x")
    monkeypatch.setattr(patch, "strip_thinking", lambda raw: (None, raw))
    monkeypatch.setattr(patch, "parse_turn", lambda raw: SimpleNamespace(thought=raw))
    monkeypatch.setattr(patch, "_is_flip", lambda *_args, **_kwargs: True)
    cases = [_probe_case(0, judgements=_judgements(1)), _probe_case(1, judgements=_judgements(1))]
    resolved = SimpleNamespace(as_dict=lambda: {"name": "fake"})

    payload = patch.run_patch_probe(
        object(), tokenizer, cases, spec=_FAKE_SPEC, resolved=resolved, layers=[1],
        policy="base", keep_last=2, max_tokens=1, seed=7, command=["patch"],
    )

    repairs = {
        "system_prompt": expected_repairs,
        "task_prompt": expected_repairs,
        "previous_notes": 3 * expected_repairs,
        "note_value_tokens": 0,
        "last_two_observations": 2 * expected_repairs,
        "final_token": 0,
    }
    assert [record["boundary_repairs"] for record in payload["cases"]] == [
        {"failing": repairs, "counterfactual": repairs}
    ] * 2
    assert len(payload["cells"]) == len(patch.POSITION_GROUPS)


def test_real_tokenizer_resolves_and_aligns_the_seven_cells_on_the_first_case() -> None:
    """Smoke test on the cached 3B tokenizer (a tokenizer load is not model execution)."""
    import json
    import os

    from local_llm_lab.models import load_model_spec
    from local_llm_lab.pipeline.protocol import build_prompt
    from local_llm_lab.probes import patch
    from local_llm_lab.project import configure_local_cache

    if not all(path.is_file() for path in _SAVED_EVALS):
        pytest.skip("protected saved evaluations not present on this checkout")
    spec = load_model_spec("qwen25-coder-3b")
    snapshot = configure_local_cache() / "hub" / f"models--{spec.hf_id.replace('/', '--')}"
    if not snapshot.is_dir():
        pytest.skip("cached tokenizer snapshot not present on this machine")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from local_llm_lab.pipeline import cli

    tokenizer = cli._load_data_tokenizer(spec.hf_id)
    passing = json.loads(_SAVED_EVALS[0].read_text(encoding="utf-8"))
    failing = json.loads(_SAVED_EVALS[1].read_text(encoding="utf-8"))
    cases, _provenance = patch.select_patch_cases(
        passing, failing, keep_last=2, data_seed=20260902, generator_version=1
    )
    case = cases[0]
    assert case.task.task_id == "test-ledger_reconcile-0031-clean"
    replayed, counterfactual, _note = patch.replay_counterfactual(case)
    prompt = build_prompt(tokenizer, replayed, spec=spec, keep_last=2)
    ids = list(tokenizer.encode(prompt, add_special_tokens=False))

    groups, repairs, values = patch._groups_for(
        tokenizer, ids, replayed, prompt_text=prompt, keep_last=2
    )

    assert tuple(groups) == patch.PROMPT_GROUPS
    assert all(groups[name] for name in patch.PROMPT_GROUPS)
    _system, _task, _notes, observations = patch._message_contents(replayed, keep_last=2)
    assert len(observations) == 2
    covered = tokenizer.decode([ids[index] for index in groups["last_two_observations"]])
    assert all(observation in covered for observation in observations)
    assert repairs["last_two_observations"] > 0
    assert groups["final_token"] == (len(ids) - 1,)

    # R25 on the real pair: the counterfactual note carries 85, the failing note dropped it.
    source_prompt = build_prompt(tokenizer, counterfactual, spec=spec, keep_last=2)
    source_ids = list(tokenizer.encode(source_prompt, add_special_tokens=False))
    source_groups, _source_repairs, source_values = patch._groups_for(
        tokenizer, source_ids, counterfactual, prompt_text=source_prompt, keep_last=2
    )
    alignment = patch.align_groups(
        tokenizer,
        source_groups=source_groups,
        source_values=source_values,
        source_ids=source_ids,
        target_groups=groups,
        target_values=values,
        target_ids=ids,
    )

    assert tuple(alignment) == patch.POSITION_GROUPS
    assert list(values) == ["25", "122"] and list(source_values) == ["25", "122", "85"]
    # Every cell's cardinalities, measured on the real prompts (2287 and 2291 tokens).
    assert {
        name: (cell.source_cardinality, cell.target_cardinality, len(cell.target_positions))
        for name, cell in alignment.items()
    } == {
        "system_prompt": (359, 359, 359),
        "task_prompt": (69, 69, 69),
        "previous_notes": (663, 659, 659),
        "shared_value_tokens": (7, 5, 5),
        "dropped_value_slot": (2, 1, 1),
        "last_two_observations": (967, 967, 967),
        "final_token": (1, 1, 1),
    }
    assert {name: cell.rule for name, cell in alignment.items()} == {
        "system_prompt": "identity",
        "task_prompt": "identity",
        "previous_notes": "tail",
        "shared_value_tokens": "shared_value_identity",
        "dropped_value_slot": "dropped_value_slot",
        "last_two_observations": "identity",
        "final_token": "identity",
    }
    notes = alignment["previous_notes"]
    assert len(notes.unpatched_source_tokens) == 4
    assert notes.record()["unpatched_source_tokens"]["text"] == "Plan: list the"
    shared = alignment["shared_value_tokens"]
    assert shared.shared_values == ("25", "122") and shared.dropped_values == ("85",)
    assert len(shared.target_positions) == 5
    (slot,) = alignment["dropped_value_slot"].record()["slots"]
    assert slot["dropped_value"] == "85"
    assert slot["slot_rule"] == "after_preceding_shared_value"
    assert slot["preceding_shared_value"] == "122"
    assert slot["slot_position"] == values["122"][-1] + 1
    assert slot["overwritten_token_text"] == ";"
    assert slot["pooled_source_text"] == "85"
    assert len(slot["pooled_source_token_ids"]) == 2
    # Every selected case aligns; four of the five overwrite ";", the fifth ",".
    separators = []
    for other in cases:
        target_messages, source_messages, _prov = patch.replay_counterfactual(other)
        target_text = build_prompt(tokenizer, target_messages, spec=spec, keep_last=2)
        source_text = build_prompt(tokenizer, source_messages, spec=spec, keep_last=2)
        target_tokens = list(tokenizer.encode(target_text, add_special_tokens=False))
        source_tokens = list(tokenizer.encode(source_text, add_special_tokens=False))
        target_pack = patch._groups_for(
            tokenizer, target_tokens, target_messages, prompt_text=target_text, keep_last=2
        )
        source_pack = patch._groups_for(
            tokenizer, source_tokens, source_messages, prompt_text=source_text, keep_last=2
        )
        other_alignment = patch.align_groups(
            tokenizer,
            source_groups=source_pack[0],
            source_values=source_pack[2],
            source_ids=source_tokens,
            target_groups=target_pack[0],
            target_values=target_pack[2],
            target_ids=target_tokens,
        )
        assert other_alignment["previous_notes"].rule == "tail"
        (entry,) = other_alignment["dropped_value_slot"].record()["slots"]
        separators.append(entry["overwritten_token_text"])
    assert sorted(separators) == [",", ";", ";", ";", ";"]


class _ParityTokenizer(_CharacterTokenizer):
    """Not prefix-stable: ids depend on the length parity of the encoded text.

    A prefix whose parity differs from the full prompt disagrees from its first token, so the
    common prefix collapses to zero and an unbounded widening would silently move a boundary
    many tokens backwards.
    """

    def encode(self, text, add_special_tokens=False):
        offset = 1000 if len(text) % 2 else 0
        return [value + offset for value in super().encode(text, add_special_tokens=add_special_tokens)]


def test_token_span_refuses_a_boundary_that_moves_more_than_one_token():
    from local_llm_lab.probes import patch

    tokenizer = _ParityTokenizer()
    prompt = "abcdefgh"
    ids = tokenizer.encode(prompt)
    # "cde" starts at an even offset (prefix "ab" agrees) and ends at an odd one (prefix
    # "abcde" disagrees from token 0): the end boundary would collapse to 1.
    with pytest.raises(ValueError, match="moved more than one token"):
        patch._token_span(tokenizer, prompt, ids, (2, 5), label="task_prompt")


# --------------------------------------------------------------------- R25 alignment
#
# The failing (target) note drops a value the counterfactual (source) note carries, so the
# source note is longer and ``previous_notes``/``note_value_tokens`` differ in cardinality.
# ``InjectionHook(replace=True)`` maps source row *i* to ``at_positions[i]`` and refuses
# unequal counts (``capture.py:248-258``), and the target prompt has no position for the
# dropped value.  R25: unequal groups align on the tail, and the value cell splits into
# ``shared_value_tokens`` (identity-aligned) and ``dropped_value_slot`` (mean-pooled source
# rows written to the separator slot where the value should have appeared).


def _value_messages(values, *, observations=("OBS zero", "OBS one", "OBS two")):
    """Replayed ledger messages whose LAST note approves exactly ``values``.

    The two earlier notes are fixed, so the failing and counterfactual message lists differ
    only in the substituted note — the R22b construction.
    """
    from local_llm_lab.agent_protocol import Action
    from local_llm_lab.pipeline.protocol import assistant_message, tool_message

    notes = [
        "Invoices read: 1 of 3. approved: 41; held (skip): 9. Reading the next invoice.",
        "Invoices read: 2 of 3. approved: 41, 57; held (skip): 9. Reading the next invoice.",
        "Invoices read: 3 of 3. approved: "
        + ", ".join(values)
        + "; held (skip): 9. Summing approved amounts.",
    ]
    system = "SYS rules for the agent. " + " ".join(f"Rule {index} applies." for index in range(24))
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "TASK reconcile the ledger"},
    ]
    for index, (note, observation) in enumerate(zip(notes, observations, strict=True)):
        messages.append(assistant_message(note, Action("read_file", {"path": f"inv/{index}"})))
        messages.append(tool_message("read_file", observation))
    return messages


def _aligned(target_values, source_values, *, tokenizer_type=None, keep_last=2):
    """``align_groups`` over two real rendered prompts built from the fakes."""
    from local_llm_lab.pipeline.protocol import build_prompt
    from local_llm_lab.probes import patch

    tokenizer = (tokenizer_type or _MergingTokenizer)()
    target_messages = _value_messages(target_values)
    source_messages = _value_messages(source_values)
    target_prompt = build_prompt(tokenizer, target_messages, spec=_FAKE_SPEC, keep_last=keep_last)
    source_prompt = build_prompt(tokenizer, source_messages, spec=_FAKE_SPEC, keep_last=keep_last)
    target_ids = tokenizer.encode(target_prompt)
    source_ids = tokenizer.encode(source_prompt)
    target_groups, _tr, target_map = patch._groups_for(
        tokenizer, target_ids, target_messages, prompt_text=target_prompt, keep_last=keep_last
    )
    source_groups, _sr, source_map = patch._groups_for(
        tokenizer, source_ids, source_messages, prompt_text=source_prompt, keep_last=keep_last
    )
    alignment = patch.align_groups(
        tokenizer,
        source_groups=source_groups,
        source_values=source_map,
        source_ids=source_ids,
        target_groups=target_groups,
        target_values=target_map,
        target_ids=target_ids,
    )
    return SimpleNamespace(
        alignment=alignment,
        tokenizer=tokenizer,
        source_ids=source_ids,
        target_ids=target_ids,
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        source_groups=source_groups,
        target_groups=target_groups,
        source_values=source_map,
        target_values=target_map,
    )


def test_position_groups_names_seven_cells_including_the_split_value_cells() -> None:
    """R25(d): the value cell splits, and every other cell keeps its name and order."""
    from local_llm_lab.probes import patch

    assert patch.POSITION_GROUPS == (
        "system_prompt",
        "task_prompt",
        "previous_notes",
        "shared_value_tokens",
        "dropped_value_slot",
        "last_two_observations",
        "final_token",
    )
    # The per-prompt groups a tokenizer can resolve on one prompt alone are unchanged; the
    # two value cells exist only pairwise, which is why alignment is a separate seam.
    assert patch.PROMPT_GROUPS == (
        "system_prompt",
        "task_prompt",
        "previous_notes",
        "note_value_tokens",
        "last_two_observations",
        "final_token",
    )


def test_tail_alignment_takes_the_source_tail_and_records_the_leading_residue() -> None:
    """R25(a): the last |target| source positions patch the target; the residue is recorded."""
    fixture = _aligned(["41", "57"], ["41", "57", "62"])
    notes = fixture.alignment["previous_notes"]
    source = fixture.source_groups["previous_notes"]
    target = fixture.target_groups["previous_notes"]

    assert len(source) > len(target)
    assert notes.rule == "tail"
    assert notes.source_cardinality == len(source)
    assert notes.target_cardinality == len(target)
    assert notes.source_positions == tuple(source[-len(target) :])
    assert notes.target_positions == target
    assert notes.unpatched_source_tokens == tuple(source[: len(source) - len(target)])
    record = notes.record()
    assert record["unpatched_source_tokens"]["count"] == len(source) - len(target)
    assert record["unpatched_source_tokens"]["token_ids"] == [
        fixture.source_ids[position] for position in notes.unpatched_source_tokens
    ]
    assert record["unpatched_source_tokens"]["text"] == fixture.tokenizer.decode(
        [fixture.source_ids[position] for position in notes.unpatched_source_tokens]
    )
    # Equal-cardinality groups are untouched and record the identity rule with no residue.
    for name in ("system_prompt", "task_prompt", "last_two_observations", "final_token"):
        cell = fixture.alignment[name]
        assert cell.rule == "identity"
        assert cell.source_positions == fixture.source_groups[name]
        assert cell.unpatched_source_tokens == ()
        assert cell.record()["unpatched_source_tokens"]["count"] == 0


def test_shared_value_tokens_align_by_string_identity_when_order_differs() -> None:
    """R25(b): identity, not position — a reordered source note still pairs 41 with 41."""
    fixture = _aligned(["41", "57"], ["57", "62", "41"])
    shared = fixture.alignment["shared_value_tokens"]

    assert shared.rule == "shared_value_identity"
    assert len(shared.source_positions) == len(shared.target_positions)
    assert shared.record()["shared_values"] == ["57", "41"]
    assert shared.record()["dropped_values"] == ["62"]
    # Every paired source position decodes to the same digit as its target position.
    pairs = zip(shared.source_positions, shared.target_positions, strict=True)
    for source_position, target_position in pairs:
        assert fixture.tokenizer.decode([fixture.source_ids[source_position]]) == (
            fixture.tokenizer.decode([fixture.target_ids[target_position]])
        )
    expected_source = tuple(
        position for value in ("57", "41") for position in fixture.source_values[value]
    )
    expected_target = tuple(
        position for value in ("57", "41") for position in fixture.target_values[value]
    )
    assert shared.source_positions == expected_source
    assert shared.target_positions == expected_target
    assert shared.source_cardinality == len(fixture.source_groups["note_value_tokens"])
    assert shared.target_cardinality == len(fixture.target_groups["note_value_tokens"])


def test_dropped_value_slot_is_the_separator_after_the_preceding_shared_value() -> None:
    """R25(b): the slot is the single token right after the preceding shared value."""
    fixture = _aligned(["41", "57"], ["41", "57", "62"])
    slot_cell = fixture.alignment["dropped_value_slot"]

    assert slot_cell.rule == "dropped_value_slot"
    assert len(slot_cell.target_positions) == 1
    assert slot_cell.pooled_sources == (fixture.source_values["62"],)
    slot = slot_cell.target_positions[0]
    # "approved: 41, 57; held (skip): 9." — 62 would have followed 57, so the slot is the
    # separator token immediately after 57's last token, which is ";".
    assert slot == fixture.target_values["57"][-1] + 1
    assert fixture.tokenizer.decode([fixture.target_ids[slot]]) == ";"
    (record,) = slot_cell.record()["slots"]
    assert record == {
        "dropped_value": "62",
        "slot_rule": "after_preceding_shared_value",
        "preceding_shared_value": "57",
        "slot_position": slot,
        "overwritten_token_id": fixture.target_ids[slot],
        "overwritten_token_text": ";",
        "pooled_source_positions": list(fixture.source_values["62"]),
        "pooled_source_token_ids": [
            fixture.source_ids[position] for position in fixture.source_values["62"]
        ],
        "pooled_source_text": "62",
    }


def test_dropped_first_value_uses_the_separator_before_the_first_shared_value() -> None:
    """R25(b) fallback: no preceding shared value, so the slot precedes the first one."""
    fixture = _aligned(["41", "57"], ["62", "41", "57"])
    slot_cell = fixture.alignment["dropped_value_slot"]

    slot = slot_cell.target_positions[0]
    assert slot == fixture.target_values["41"][0] - 1
    (record,) = slot_cell.record()["slots"]
    assert record["slot_rule"] == "before_first_shared_value"
    assert record["preceding_shared_value"] is None
    assert record["dropped_value"] == "62"
    assert fixture.tokenizer.decode([fixture.target_ids[slot]]) == " "


def test_two_dropped_values_get_two_slots_in_source_order() -> None:
    """R25(b): more than one dropped value patches every slot in one injection."""
    fixture = _aligned(["41", "57"], ["41", "62", "57", "78"])
    slot_cell = fixture.alignment["dropped_value_slot"]

    assert len(slot_cell.target_positions) == 2
    assert slot_cell.pooled_sources == (
        fixture.source_values["62"],
        fixture.source_values["78"],
    )
    first, second = slot_cell.record()["slots"]
    assert (first["dropped_value"], second["dropped_value"]) == ("62", "78")
    assert first["preceding_shared_value"] == "41"
    assert second["preceding_shared_value"] == "57"
    assert first["slot_position"] == fixture.target_values["41"][-1] + 1
    assert second["slot_position"] == fixture.target_values["57"][-1] + 1
    assert len(set(slot_cell.target_positions)) == 2


def test_note_values_match_on_word_boundaries_not_substrings() -> None:
    """Issue #29: ``32`` inside ``132`` is not an ambiguous span, it is a different value."""
    fixture = _aligned(["132", "32"], ["132", "32", "78"])

    shared = fixture.alignment["shared_value_tokens"]
    assert shared.record()["shared_values"] == ["132", "32"]
    # The standalone 32 is located after the 132, never inside it.
    assert fixture.target_values["32"][0] > fixture.target_values["132"][-1]
    assert fixture.tokenizer.decode(
        [fixture.target_ids[position] for position in fixture.target_values["132"]]
    ) == "132"
    assert fixture.tokenizer.decode(
        [fixture.target_ids[position] for position in fixture.target_values["32"]]
    ) == "32"
    # A genuine duplicate is still refused.
    with pytest.raises(ValueError, match="note_value token span is missing or ambiguous"):
        _aligned(["41", "41"], ["41", "41", "78"])


def test_alignment_refuses_a_source_group_shorter_than_the_target() -> None:
    """Tail alignment needs at least |target| source positions; fail closed, never cycle."""
    with pytest.raises(ValueError, match="treatment.*cardinality"):
        _aligned(["41", "57", "62"], ["41", "57"])


def test_alignment_refuses_when_no_value_was_dropped() -> None:
    with pytest.raises(ValueError, match="dropped_value_slot"):
        _aligned(["41", "57"], ["41", "57"])


def test_random_control_pool_arithmetic_is_pinned_for_every_cell() -> None:
    """Issue #29: the post-alignment control draw, pinned per cell on the smallest fakes.

    The pool is ``range(length) - treatment``; the draw is seeded by
    ``f"{seed}:{task_id}:{layer}:{group}"``, so both sides are reproducible here.
    """
    import random as _random

    from local_llm_lab.probes import patch

    fixture = _aligned(["41", "57"], ["41", "57", "62"])
    source_length, target_length = len(fixture.source_ids), len(fixture.target_ids)
    for group in patch.POSITION_GROUPS:
        cell = fixture.alignment[group]
        cardinality = len(cell.target_positions)
        source_treatment = cell.treatment_source_positions
        drawn = patch._random_control_pair(
            source_length=source_length,
            target_length=target_length,
            source_treatment=source_treatment,
            target_treatment=cell.target_positions,
            cardinality=cardinality,
            seed=7,
            task_id="test-ledger_reconcile-0-clean",
            layer=1,
            group=group,
        )
        rng = _random.Random(f"7:test-ledger_reconcile-0-clean:1:{group}")
        expected = (
            tuple(
                sorted(
                    rng.sample(
                        sorted(set(range(source_length)) - set(source_treatment)), cardinality
                    )
                )
            ),
            tuple(
                sorted(
                    rng.sample(
                        sorted(set(range(target_length)) - set(cell.target_positions)),
                        cardinality,
                    )
                )
            ),
        )
        assert drawn == expected
        assert len(drawn[0]) == len(drawn[1]) == cardinality
        assert set(drawn[0]).isdisjoint(source_treatment)
        assert set(drawn[1]).isdisjoint(cell.target_positions)
    # dropped_value_slot draws exactly one non-treatment position per dropped value (R25c).
    assert len(fixture.alignment["dropped_value_slot"].target_positions) == 1
    # The refusal survives: a pool smaller than the treatment cannot satisfy it.
    with pytest.raises(ValueError, match="random control candidate pool cannot satisfy"):
        patch._random_control_pair(
            source_length=4,
            target_length=4,
            source_treatment=(0, 1, 2),
            target_treatment=(0, 1, 2),
            cardinality=3,
            seed=7,
            task_id="t",
            layer=1,
            group="previous_notes",
        )


def _alignment_probe(monkeypatch, target_values, source_values, *, layers=(1,)):
    """Run ``run_patch_probe`` over the fakes and return ``(payload, injections)``."""
    from local_llm_lab.probes import patch

    injected = []

    class View:
        num_layers = 1

    class Hook:
        def __init__(self, _view, _layer, rows, *, at_positions, replace):
            assert replace
            injected.append((np.asarray(rows), tuple(at_positions)))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(patch.ArchitectureView, "from_model", lambda _model: View())
    monkeypatch.setattr(
        patch,
        "replay_counterfactual",
        lambda case: (
            _value_messages(target_values),
            _value_messages(source_values),
            {"counterfactual_source": "passing_transcript", "counterfactual_basis": "fixture"},
        ),
    )
    # One row per position carrying that position's index, so a pooled row is checkable.
    monkeypatch.setattr(
        patch,
        "capture_residuals",
        lambda _view, ids, requested, **_kwargs: {
            layer: mx.array([[float(index)] for index in range(len(ids))], dtype=mx.float32)
            for layer in requested
        },
    )
    monkeypatch.setattr(patch, "InjectionHook", Hook)
    monkeypatch.setattr(patch, "greedy_generate", lambda *_args, **_kwargs: "x")
    monkeypatch.setattr(patch, "strip_thinking", lambda raw: (None, raw))
    monkeypatch.setattr(patch, "parse_turn", lambda raw: SimpleNamespace(thought=raw))
    monkeypatch.setattr(patch, "_is_flip", lambda *_args, **_kwargs: True)
    cases = [_probe_case(0, judgements=_judgements(1)), _probe_case(1, judgements=_judgements(1))]
    payload = patch.run_patch_probe(
        object(),
        _MergingTokenizer(),
        cases,
        spec=_FAKE_SPEC,
        resolved=SimpleNamespace(as_dict=lambda: {"name": "fake"}),
        layers=list(layers),
        policy="base",
        keep_last=2,
        max_tokens=1,
        seed=7,
        command=["patch"],
    )
    return payload, injected


def test_run_patch_probe_aligns_the_unequal_note_groups_end_to_end(monkeypatch) -> None:
    """The live crash path: unequal ``previous_notes``/value cardinalities now run."""
    from local_llm_lab.probes import patch

    payload, injected = _alignment_probe(monkeypatch, ["41", "57"], ["41", "57", "62"])

    assert payload["groups"] == list(patch.POSITION_GROUPS)
    assert set(payload["cells"]) == {f"1:{name}" for name in patch.POSITION_GROUPS}
    # Every injection writes exactly as many rows as it names target positions.
    for rows, positions in injected:
        assert rows.shape[0] == len(positions)
    record = payload["cases"][0]["alignment"]
    assert set(record) == set(patch.POSITION_GROUPS)
    assert record["previous_notes"]["rule"] == "tail"
    assert record["previous_notes"]["unpatched_source_tokens"]["count"] == (
        record["previous_notes"]["source_cardinality"]
        - record["previous_notes"]["target_cardinality"]
    )
    assert record["shared_value_tokens"]["rule"] == "shared_value_identity"
    assert record["shared_value_tokens"]["shared_values"] == ["41", "57"]
    assert record["dropped_value_slot"]["slots"][0]["dropped_value"] == "62"
    # R25(d): the cell carries the alignment facts a reader of the heat map needs.
    cell = payload["cells"]["1:dropped_value_slot"]["alignment"]
    assert set(cell) == {case["task_id"] for case in payload["cases"]}
    assert cell[payload["cases"][0]["task_id"]] == record["dropped_value_slot"]
    assert json.loads(json.dumps(payload["cases"][0]["alignment"])) == record
    markdown = patch.render_markdown(payload)
    assert "dropped_value_slot" in markdown and "Alignment (R25)" in markdown


def test_pooled_dropped_value_row_is_the_float32_mean_of_the_source_rows(monkeypatch) -> None:
    """R25(b) and briefing rule 1.5: one row per slot, the float32 mean of its source rows."""
    payload, injected = _alignment_probe(monkeypatch, ["41", "57"], ["41", "57", "62"])

    record = payload["cases"][0]["alignment"]["dropped_value_slot"]
    pooled_positions = record["slots"][0]["pooled_source_positions"]
    slot = record["slots"][0]["slot_position"]
    assert len(pooled_positions) > 1, "the pooling is only meaningful over several rows"
    treatments = [(rows, positions) for rows, positions in injected if positions == (slot,)]
    assert treatments, "the dropped-value slot was never patched"
    rows, _positions = treatments[0]
    assert rows.shape == (1, 1)
    assert rows.dtype == np.float32
    assert rows[0, 0] == pytest.approx(sum(pooled_positions) / len(pooled_positions))


def test_controls_resample_to_the_post_alignment_cardinality(monkeypatch) -> None:
    """R25(c): controls match the aligned target size, one draw per dropped value."""
    from local_llm_lab.probes import patch

    payload, injected = _alignment_probe(monkeypatch, ["41", "57"], ["41", "62", "57", "78"])

    record = payload["cases"][0]["alignment"]
    slots = tuple(entry["slot_position"] for entry in record["dropped_value_slot"]["slots"])
    assert len(slots) == 2
    # Three injections per (case, group): treatment, unrelated_task, random_positions.
    assert len(injected) == len(patch.POSITION_GROUPS) * len(payload["cases"]) * 3
    slot_injections = [
        (rows, positions) for rows, positions in injected if positions == slots
    ]
    assert len(slot_injections) == 2 * 2, "treatment and unrelated_task, for both cases"
    assert all(rows.shape[0] == len(slots) for rows, _positions in slot_injections)
    random_slot = [
        (rows, positions)
        for rows, positions in injected
        if len(positions) == len(slots) and set(positions).isdisjoint(slots)
    ]
    assert random_slot, "random_positions drew no non-treatment slot group"
    assert all(rows.shape[0] == len(positions) for rows, positions in random_slot)
    notes_target = tuple(record["previous_notes"]["target_positions"])
    notes_rows = [rows for rows, positions in injected if positions == notes_target]
    assert notes_rows and all(rows.shape[0] == len(notes_target) for rows in notes_rows)
