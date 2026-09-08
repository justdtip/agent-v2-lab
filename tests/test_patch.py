from __future__ import annotations

import hashlib
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


_FIXTURE_SEED = 17
"""The data seed the fixtures build tasks under. Fixture data, not a tool constant: the
generator is seeded by it and any value yields a well-formed task."""


def _payload(trajectories, *, data_seed=_FIXTURE_SEED, generator_version=None):
    payload = {"data_seed": data_seed, "trajectories": trajectories}
    if generator_version is not None:
        payload["generator_version"] = generator_version
    return payload


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
        # R30(2): every case records what its eligibility was judged under.
        "eligibility_basis": patch.BOUND_VS_HEAD_BASIS,
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
            "eligibility_basis": patch.BOUND_VS_HEAD_BASIS,
            "integrity": {"evaluation": 0, "recomputed": 1, "head_alone": 0},
            "difficulty": {"evaluation": 0, "recomputed": 1},
            "recomputed_generator_version": 1,
            "generator_version_source": "flag",
            "generator_version_basis": (
                "explicit --generator-version binding; the evaluation records no "
                "generator_version"
            ),
            # A pre-schema payload declares no version of its own, so the replay ran under
            # the flag and the input's own declaration stays empty rather than borrowing it.
            "evaluation_generator_version": None,
            "evaluation_generator_version_source": "flag",
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
        "head_alone": 0,
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
        "head_alone": 0,
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
        "eligibility_basis": patch.BOUND_VS_HEAD_BASIS,
        "integrity": {"evaluation": 1, "recomputed": 0, "head_alone": 0},
        "difficulty": {"evaluation": 1, "recomputed": 0},
        # Nothing recomputed, so no bound version was needed; the artifact still says what
        # the failing evaluation declared, which here -- a payload with no summary -- is
        # nothing, from neither a record nor a flag.
        "evaluation_generator_version": None,
        "evaluation_generator_version_source": "unbound",
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
        "eligibility_basis": patch.BOUND_VS_HEAD_BASIS,
        "integrity": {"evaluation": 1, "recomputed": 1, "head_alone": 0},
        "difficulty": {"evaluation": 1, "recomputed": 1},
        "recomputed_generator_version": 1,
        "generator_version_source": "flag",
        "generator_version_basis": (
            "explicit --generator-version binding; the evaluation records no "
            "generator_version"
        ),
        # The flag supplied the replay version; the evaluation itself declared none, and the
        # artifact must not let the flag's value pass for a recorded one.
        "evaluation_generator_version": None,
        "evaluation_generator_version_source": "flag",
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
        "eligibility_basis": patch.BOUND_VS_HEAD_BASIS,
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
        "eligibility_basis": patch.BOUND_VS_HEAD_BASIS,
        "integrity": {"evaluation": 0, "recomputed": 1, "head_alone": 0},
        "difficulty": {"evaluation": 0, "recomputed": 1},
        "recomputed_generator_version": patch.GENERATOR_VERSION,
        "generator_version_source": "flag",
        "generator_version_basis": (
            "explicit --generator-version binding; the evaluation records no "
            "generator_version"
        ),
        # This fixture's payload is a bare trajectory list with no summary, so the version in
        # force came from the flag and the evaluation declared none of its own.
        "evaluation_generator_version": None,
        "evaluation_generator_version_source": "flag",
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
        "head_alone": 0,
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
    """R27(1): an unparseable turn is ``parse_error``, never a flip, and its raw head is kept."""
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
    scoring = patch.CaseScoring(failing=(), dropped=("42",), canonical=("42",))
    monkeypatch.setattr(patch, "InjectionHook", Hook)
    monkeypatch.setattr(patch, "greedy_generate", lambda *_args, **_kwargs: "not a turn")
    monkeypatch.setattr(patch, "strip_thinking", lambda raw: (None, raw))
    monkeypatch.setattr(patch, "parse_turn", lambda _raw: (_ for _ in ()).throw(ValueError("bad")))

    scored = patch._score_patch(
        object(), object(), case, scoring=scoring, layer=1, source_rows=object(),
        target_positions=(0,), failing_ids=[1], keep_last=2, max_tokens=1,
    )

    assert not scored.is_flip
    assert scored.outcome == "parse_error"
    assert scored.raw_head == "not a turn"
    # The pre-R27 boolean is not even measured for a turn that never parsed.
    assert scored.value_drop_cleared is None


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
        name="qwen35-4b",
        hf_id="fake/hf",
        policies={},
        probe_layer_fractions=(0.25, 0.5),
        resolve=lambda *_args: SimpleNamespace(num_layers=4, probe_layers=(1, 2)),
    )
    case = patch.PatchCase(
        _Task("test-aggregate_report-0-clean", "aggregate_report"),
        0,
        (),
        None,
        *_judgements(0),
    )
    seeds = {
        "passing": {"data_seed": 41, "data_seed_source": "flag"},
        "failing": {"data_seed": 41, "data_seed_source": "evaluation"},
    }
    eligibility = {
        "passing": {"eligibility_source": "evaluation"},
        "failing": {
            "eligibility_source": "recomputed",
            "integrity": {"evaluation": 0, "recomputed": 1, "head_alone": 0},
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
    monkeypatch.setattr(
        patch,
        "require_preflight",
        lambda given, **kwargs: seen.append(("preflight", given.name, kwargs["skip"])),
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
        or (object(), object(), SimpleNamespace(num_layers=4), SimpleNamespace(as_dict=dict)),
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
    # SPEC-001 §10: the gate runs, and it runs before the guard and the load.
    assert seen.index(("preflight", "qwen35-4b", False)) < seen.index(("guard", None))
    assert seen.index(("preflight", "qwen35-4b", False)) < seen.index(("load", "fake/hf", None))
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

    # R26(e)/(g), issue #35: the run writes its own log pair, and the start event alone
    # identifies what ran.
    assert (tmp_path / "run.log").is_file()
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    start = events[0]
    assert start["kind"] == "start" and start["run"] == "p6-patch"
    assert start["command"] == argv
    identity = dict(start["fields"])
    assert identity.pop("git_commit")  # a commit or "unknown"; never absent
    assert identity == {
        "model": "qwen35-4b",
        "hf_id": "fake/hf",
        "policy": "base",
        "adapter": None,
        "passing_eval_sha256": hashlib.sha256(passing.read_bytes()).hexdigest(),
        "failing_eval_sha256": hashlib.sha256(failing.read_bytes()).hexdigest(),
        "layers": "1,0.5",
        "keep_last": 2,
        "data_seed": 41,
        "generator_version": 1,
    }
    selection = next(event for event in events if event["message"] == "selected cases")
    assert selection["fields"] == {
        "cases": 1,
        "stable": 1,
        "unstable": 0,
        "decision_steps": [0],
    }
    assert [event["fields"]["selected"] for event in events if event["message"] == "layers"] == [
        [1, 2]
    ]
    assert events[-1]["kind"] == "end" and events[-1]["status"] == "ok"


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
    def capture(_view, ids, layers, *, positions, dtype="float32"):
        del dtype
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

    # The generated NOTE is what R27 scores, so the fake returns a parseable value field:
    # the treatment restores the dropped value (a strict flip), everything else writes a
    # number outside the canonical set (a corruption, never a flip).
    def generation(*args, **_kwargs):
        vector, target = injected[-1]
        own_base = 201 if args[2][0] == 10 else 401
        own_treatments = {
            (treatment_vector(name, own_base), aligned_target[name])
            for name in patch.POSITION_GROUPS
        }
        if (vector, target) in own_treatments:
            return "approved: 42"
        return "approved: 77"

    monkeypatch.setattr(patch, "greedy_generate", generation)
    monkeypatch.setattr(patch, "strip_thinking", lambda raw: (None, raw))
    monkeypatch.setattr(patch, "parse_turn", lambda raw: SimpleNamespace(thought=raw))
    monkeypatch.setattr(patch, "_is_flip", lambda *_args, **_kwargs: False)
    resolved = SimpleNamespace(as_dict=lambda: {"name": "fake"})

    payload = patch.run_patch_probe(
        object(), Tokenizer(), cases, spec=object(), resolved=resolved, layers=[1], policy="base",
        keep_last=2, max_tokens=1, seed=7, command=["patch"],
    )

    assert len(captures) == 4
    # R27(5): the content control only runs where the cell's source rows actually cover the
    # dropped value's positions — here position 7 of the counterfactual prompt.
    content_swap_groups = ("dropped_value_slot", "last_two_observations")
    assert len(injected) == len(cases) * (
        len(patch.POSITION_GROUPS) * 3 + len(content_swap_groups)
    )
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
            if name in content_swap_groups:
                # The treatment rows with the row sourced from position 7 — the dropped
                # value's own row — replaced by the OTHER case's POOLED dropped-value row.
                # Its slot pools the single position 7, so the mean is that row itself.
                rows = list(treatment_vector(name, own_base))
                rows[0] = float(other_base + 7)
                expected.append((tuple(rows), target))
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
    # R27(1)/(2): the strict outcome, not a boolean, and the counts beside the rate.
    assert payload["cells"]["1:system_prompt"]["treatment"]["outcomes"] == {
        "flip": 2,
        "corrupted": 0,
        "unchanged": 0,
        "empty": 0,
        "parse_error": 0,
    }
    assert payload["cells"]["1:system_prompt"]["controls"]["unrelated_task"]["outcomes"] == {
        "flip": 0,
        "corrupted": 2,
        "unchanged": 0,
        "empty": 0,
        "parse_error": 0,
    }
    # R27(5): applicable only where the cell reads a dropped-value row.
    assert payload["cells"]["1:system_prompt"]["controls"]["content_swap"] == {
        "applicable": False,
        "reason": patch._CONTENT_SWAP_NOT_APPLICABLE,
        # R30(6): the n a control was NOT taken over is stated, not left blank.
        "applicable_cases": 0,
        "cases": 2,
    }
    assert payload["cells"]["1:final_token"]["controls"]["content_swap"]["applicable"] is False
    swap = payload["cells"]["1:dropped_value_slot"]["controls"]["content_swap"]
    assert swap["applicable"] is True and swap["denominator"] == 2
    assert swap["swapped_rows"] == {case.task.task_id: 1 for case in cases}
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
    # R27(3): the decision step is 0, so there is no prior observation to retain and the
    # dropped value cannot be visible in one.  R27(1): the value sets are recorded per case.
    hidden = {
        "dropped_value_visible_in_retained_observations": False,
        "visible_dropped_values": [],
        "retained_observations": 0,
    }
    value_sets = {
        "failing_values": [],
        "dropped_values_scored": ["42"],
        "canonical_values": [],
        "expected_values": ["42"],
        # R30(3)/(4): the fake task carries no canonical note, so there is no list field to
        # scope by; the required set is D alone, which is inside E, so the flip is reachable.
        "expected_field_scope": [],
        "flip_satisfiable": True,
        "failing_values_outside_canonical": [],
    }
    assert payload["cases"] == [
        {
            "task_id": "test-aggregate_report-0-clean",
            "decision_step": 0,
            "condition": patch.PRIMARY_CONDITION,
            "counterfactual_label": "empirically_passing_preferred",
            "counterfactual_source": "passing_transcript",
            "counterfactual_basis": "fixture",
            **_scoring_record(0),
            **hidden,
            "value_sets": value_sets,
            "boundary_repairs": {"failing": no_repairs, "counterfactual": no_repairs},
            "alignment": alignment,
        },
        {
            "task_id": "test-ledger_reconcile-1-clean",
            "decision_step": 0,
            "condition": patch.PRIMARY_CONDITION,
            "counterfactual_label": "empirically_passing_preferred",
            "counterfactual_source": f"generator_v{patch.GENERATOR_VERSION}",
            "counterfactual_basis": "fixture",
            **_scoring_record(0),
            **hidden,
            "value_sets": value_sets,
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
    assert (payload["headline_cases"], payload["visibility_excluded_cases"]) == (2, 0)
    assert payload["excluded_cases"] == []
    assert payload["headline_task_ids"] == payload["selected_task_ids"]
    assert payload["scoring_generator_version"] == patch.GENERATOR_VERSION
    assert payload["condition"] == patch.PRIMARY_CONDITION
    assert payload["outcomes"] == list(patch.OUTCOMES)
    # R27(2): aggregate-only is not permitted — one row per (case, layer, cell, condition).
    assert len(payload["generations"]) == len(injected)
    assert {row["condition"] for row in payload["generations"]} == {
        "treatment",
        *patch.CONTROLS,
    }
    treatment_rows = [
        row
        for row in payload["generations"]
        if row["condition"] == "treatment" and row["group"] == "system_prompt"
    ]
    assert [row["outcome"] for row in treatment_rows] == ["flip", "flip"]
    assert [row["note"] for row in treatment_rows] == ["approved: 42", "approved: 42"]
    assert [row["values"] for row in treatment_rows] == [["42"], ["42"]]
    assert json.loads(json.dumps(payload["generations"])) == payload["generations"]


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

    # R30a(2) moves where the refusal lands: ``align_groups`` still refuses the short source
    # (asserted directly in the R25 tests), but the case carrying it is now skipped rather
    # than raised through, and the section refuses on what is left — here, nothing.
    with pytest.raises(ValueError, match=r"scored 0 of 2 selected case\(s\)"):
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

    def capture(_view, ids, layers, *, positions, dtype="float32"):
        del dtype
        captures.append(tuple(ids))
        return {layer: mx.zeros((10, 1)) for layer in layers}

    monkeypatch.setattr(patch, "capture_residuals", capture)
    monkeypatch.setattr(patch, "InjectionHook", Hook)
    # R27 scores the generated NOTE, so the fake writes a parseable value field carrying
    # the case's dropped value: every generation is a strict flip.
    monkeypatch.setattr(patch, "greedy_generate", lambda *_args, **_kwargs: "approved: 42")
    monkeypatch.setattr(patch, "strip_thinking", lambda raw: (None, raw))
    monkeypatch.setattr(patch, "parse_turn", lambda raw: SimpleNamespace(thought=raw))
    monkeypatch.setattr(patch, "_is_flip", lambda *_args, **_kwargs: True)
    return Tokenizer(), captures


def _probe_case(index, *, judgements, observation=None, failing_note="drop"):
    """A probe case; ``observation`` puts a retained observation before the decision step,
    which is what R27(3)'s visibility measurement reads, and ``failing_note`` sets the note
    at that step, which is what R30(3)/(4) scope and satisfiability read."""
    from local_llm_lab.probes import patch

    first = {"thought": "a"}
    if observation is not None:
        first["observation"] = observation
    return patch.PatchCase(
        _Task(f"test-ledger_reconcile-{index}-clean", "ledger_reconcile"),
        1,
        (first, {"thought": failing_note}),
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
        assert all(
            control["denominator"] == 2
            for control in cell["controls"].values()
            # R27(5): a cell whose source rows cover no dropped-value position records
            # ``not_applicable`` rather than a fabricated denominator.
            if control.get("applicable", True)
        )
    assert payload["counterfactual_sources"] == {"passing_transcript": 2}
    assert payload["excluded_cases"] == [
        {
            "task_id": "test-ledger_reconcile-1-clean",
            "decision_step": 1,
            "condition": patch.PRIMARY_CONDITION,
            "counterfactual_label": "empirically_passing_preferred",
            "counterfactual_source": "passing_transcript",
            "counterfactual_basis": "fixture",
            "scoring_version_stable": False,
            "decision_step_bound": 1,
            "decision_step_head": 2,
            "dropped_values_bound": ["42"],
            "dropped_values_head": ["42"],
            "judged_under_bound": "generator_v1 replay",
            "judged_under_head": f"generator_v{patch.GENERATOR_VERSION} HEAD",
            "eligibility_basis": patch.BOUND_VS_HEAD_BASIS,
            "dropped_value_visible_in_retained_observations": False,
            "visible_dropped_values": [],
            "retained_observations": 0,
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


def test_patch_probe_reports_progress_per_capture_and_cell_without_touching_the_payload(
    monkeypatch,
) -> None:
    """R26(g): one line per prepared case, then one per (layer, group) cell after it is scored.

    The payload must be byte-identical with and without the callback: progress is reporting,
    never computation.
    """
    from local_llm_lab.probes import patch

    cases = [_probe_case(index, judgements=_judgements(1)) for index in range(3)]
    tokenizer, _captures = _probe_fixture(monkeypatch, cases)
    # The shared fixture's view has one layer; the cell grid needs more than one.
    monkeypatch.setattr(
        patch.ArchitectureView, "from_model", lambda _model: SimpleNamespace(num_layers=3)
    )
    resolved = SimpleNamespace(as_dict=lambda: {"name": "fake"})
    layers = [1, 3]

    def run(progress):
        return patch.run_patch_probe(
            object(), tokenizer, cases, spec=object(), resolved=resolved, layers=layers,
            policy="base", keep_last=2, max_tokens=1, seed=7, command=["patch"],
            progress=progress,
        )

    seen: list[tuple[int, int, str]] = []
    with_progress = run(lambda step, total, label: seen.append((step, total, label)))
    without_progress = run(None)

    cells = len(layers) * len(patch.POSITION_GROUPS)
    assert cells == len(layers) * 7
    assert seen == [
        (1, 3, "capture test-ledger_reconcile-0-clean"),
        (2, 3, "capture test-ledger_reconcile-1-clean"),
        (3, 3, "capture test-ledger_reconcile-2-clean"),
        *(
            (step, cells, f"layer {layer} {group}")
            for step, (layer, group) in enumerate(
                ((layer, group) for layer in layers for group in patch.POSITION_GROUPS),
                start=1,
            )
        ),
    ]
    assert json.dumps(with_progress, sort_keys=True) == json.dumps(
        without_progress, sort_keys=True
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
        "dropped_value_visible_in_retained_observations": True,
        "excluded_reason": "dropped_value_visible_in_retained_observations",
    }
    payload = {
        "groups": ["system_prompt"],
        "layers": [1],
        "controls": [],
        "cells": {"1:system_prompt": {"treatment": {"rate": 1.0, "wilson_95": [0.5, 1.0]}}},
        "stable_cases": 1,
        "unstable_cases": 1,
        "headline_cases": 1,
        "visibility_excluded_cases": 0,
        "cases": [{**stable, "dropped_value_visible_in_retained_observations": False}],
        "excluded_cases": [unstable],
    }

    text = patch.render_markdown(payload)
    lines = text.splitlines()

    assert (
        "Headline over 1 case(s) of 1 scoring-version-stable; 1 unstable case(s) excluded "
        "(R24) and 0 excluded because the dropped value is visible in the retained "
        "observations (R27)."
    ) in lines
    assert lines.index("## Treatment flip rate") < lines.index("| 1 | 1.000 [0.500, 1.000] |")
    assert "## Scoring version stability and value visibility (R24, R27)" in lines
    headline_row = "| test-ledger_reconcile-0-clean | yes | 1 / 1 | 12 / 12 | no | - |"
    assert headline_row in lines
    unstable_header = lines.index("### Excluded cases (never in the headline)")
    excluded_row = (
        "| test-ledger_reconcile-1-clean | no | 1 / 2 | 12 / unknown | yes | "
        "dropped_value_visible_in_retained_observations |"
    )
    assert lines.index(excluded_row) > unstable_header
    assert lines.index(headline_row) < unstable_header

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

# `generation_prefix` is declared per model now, and `protocol.generation_suffix` reads it, so a
# fake spec that omits it is a fake of a spec that cannot exist.
_FAKE_SPEC = SimpleNamespace(
    chat=SimpleNamespace(
        template_kwargs={},
        thinking="on",
        generation_prefix="<|im_start|>assistant\n",
        observation_role="tool",
    )
)


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
    # R30a(1) overturns this test's last clause: a genuine duplicate under one label used to
    # be refused as ambiguous, and is now the field's first and second value. The pairing
    # that ``align_groups`` does by string identity still needs one span per string, so the
    # canonical span is the first occurrence and the second contributes its tokens to the
    # group alone.
    duplicate = _aligned(["41", "41"], ["41", "41", "78"])
    assert duplicate.alignment["shared_value_tokens"].record()["shared_values"] == ["41"]
    located = duplicate.target_groups["note_value_tokens"]
    assert duplicate.tokenizer.decode(
        [duplicate.target_ids[position] for position in located]
    ) == "4141"
    canonical = duplicate.target_values["41"]
    assert canonical == located[: len(canonical)]


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


def _alignment_probe(monkeypatch, target_values, source_values, *, layers=(1,), other=None):
    """Run ``run_patch_probe`` over the fakes and return ``(payload, injections)``.

    ``other`` gives the SECOND case a different value pair, which is what the R27 content
    control needs: the unrelated case's dropped-value rows then sit at different positions
    from the treated case's, so a swapped row is distinguishable from an unswapped one.
    """
    from local_llm_lab.probes import patch

    injected = []
    pairs = {
        "test-ledger_reconcile-0-clean": (target_values, source_values),
        "test-ledger_reconcile-1-clean": other or (target_values, source_values),
    }

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
            _value_messages(pairs[case.task.task_id][0]),
            _value_messages(pairs[case.task.task_id][1]),
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
    monkeypatch.setattr(patch, "greedy_generate", lambda *_args, **_kwargs: "approved: 42")
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
    # Three injections per (case, group): treatment, unrelated_task, random_positions —
    # plus one content_swap (R27) on each cell whose source rows cover a dropped value.
    swap_cells = [
        key.split(":", 1)[1]
        for key, cell in payload["cells"].items()
        if cell["controls"]["content_swap"].get("applicable")
    ]
    assert set(swap_cells) == {"previous_notes", "dropped_value_slot"}
    assert len(injected) == len(patch.POSITION_GROUPS) * len(payload["cases"]) * 3 + len(
        swap_cells
    ) * len(payload["cases"])
    slot_injections = [
        (rows, positions) for rows, positions in injected if positions == slots
    ]
    assert len(slot_injections) == 3 * 2, (
        "treatment, unrelated_task and content_swap, for both cases"
    )
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


# --- R27: strict, auditable flip scoring ---------------------------------------------------
#
# The defect the ruling closes, in one sentence from the code: ``_is_flip`` asks only whether a
# ``value_drop`` violation remains at the decision step, and ``integrity._contradictory_fields``
# (``integrity.py:374-386``) marks a note's ``approved:`` field contradictory as soon as its
# numbers are not a SUBSET of the canonical field's, after which
# ``_fact_covered_by_stale_field`` (``:389-395``) treats every canonical value — the dropped one
# included — as covered.  A regenerated note carrying a WRONG number therefore loses its value
# drop and scored as a flip.  The fixture below is a real HEAD ledger task, so the demonstration
# runs through the real checker with no model and no tokenizer.

_R27_TASK_ID = "test-ledger_reconcile-0007-clean"
_R27_DECISION = 7
_R27_CANONICAL_VALUES = "34, 117, 145, 131"
_R27_FAILING_VALUES = "34, 117, 131"
_R27_DROPPED = "145"


def _real_ledger_case():
    """A real HEAD ledger task replayed through its simulator, with one value dropped.

    Structurally identical to the five P6 cases: an ``approved:`` list at the summing step
    that omits one amount whose observation has already left the window.
    """
    from local_llm_lab.pipeline.env import Simulator
    from local_llm_lab.pipeline.tasks import difficulty, task_from_id
    from local_llm_lab.probes import patch

    index = int(_R27_TASK_ID.rsplit("-", 2)[1])
    task = task_from_id(_R27_TASK_ID, _FIXTURE_SEED, difficulty("test", index))
    simulator = Simulator.for_task(task)
    steps = []
    for step in task.steps:
        observation = simulator.execute(step.action)
        steps.append(
            {
                "thought": step.thought,
                "action": {"name": step.action.name, "arguments": step.action.arguments},
                "observation": observation,
            }
        )
    canonical = task.steps[_R27_DECISION].thought
    assert _R27_CANONICAL_VALUES in canonical, "fixture drifted from the generator"
    steps[_R27_DECISION] = {
        **steps[_R27_DECISION],
        "thought": canonical.replace(_R27_CANONICAL_VALUES, _R27_FAILING_VALUES),
    }
    judgement = patch.DropJudgement(
        _R27_DECISION, (_R27_DROPPED,), f"generator_v{patch.GENERATOR_VERSION} HEAD"
    )
    case = patch.PatchCase(
        task, _R27_DECISION, tuple(steps), None, judgement, judgement
    )
    return case, task, steps, canonical


def _note_with(values: str) -> str:
    _case, task, _steps, _canonical = _real_ledger_case()
    return task.steps[_R27_DECISION].thought.replace(_R27_CANONICAL_VALUES, values)


def test_the_selected_case_really_carries_one_value_drop_at_the_decision_step() -> None:
    """The fixture is only evidence if the real checker agrees it is a value drop."""
    from local_llm_lab.pipeline.integrity import check_trajectory

    case, task, steps, _canonical = _real_ledger_case()

    report = check_trajectory(task, steps, keep_last=2)
    decision = [
        (violation.kind, violation.detail)
        for violation in report.violations
        if violation.step == _R27_DECISION
    ]

    assert decision == [("value_drop", f"missing required values: {_R27_DROPPED}")]
    assert case.scoring_version_stable is True


@pytest.mark.parametrize(
    ("label", "values", "pre_r27_flip", "outcome"),
    [
        # HEAD's wrong results, measured: a note carrying a number that is not in the
        # canonical list clears the value drop and scored as a flip.
        ("foreign_number", "34, 117, 131, 200", True, "corrupted"),
        # The same defect with a HELD amount moved into the approved list.
        ("held_number", "34, 117, 131, 99", True, "corrupted"),
        # The two cases HEAD already got right, pinned so the fix cannot regress them.
        ("restored", _R27_CANONICAL_VALUES, True, "flip"),
        ("unchanged", _R27_FAILING_VALUES, False, "unchanged"),
    ],
)
def test_pre_r27_scored_a_corrupted_note_as_a_flip_and_the_strict_rule_does_not(
    label, values, pre_r27_flip, outcome
) -> None:
    """R27(1), red-first: a wrong number is a corruption, never a flip.

    ``pre_r27_flip`` records what ``_is_flip`` says on this exact input at HEAD; two of the
    four rows are the defect (``True`` beside ``corrupted``).  Both are asserted, so the
    legacy diagnostic stays honest and the strict rule is pinned against it.
    """
    from local_llm_lab.probes import patch

    case, task, steps, _canonical = _real_ledger_case()
    note = _note_with(values)
    scored_steps = [dict(step) for step in steps]
    scored_steps[_R27_DECISION]["thought"] = note

    assert patch._is_flip(task, scored_steps, _R27_DECISION, keep_last=2) is pre_r27_flip
    scoring = patch.case_scoring(case)
    assert patch.classify_generation(patch._note_values(note), scoring) == outcome
    assert (outcome == "flip") is not (label in ("foreign_number", "held_number", "unchanged"))


def test_case_scoring_reads_the_value_sets_from_the_checkers_own_ground_truth() -> None:
    """R27(1): F, D and E, measured on the real fixture with ``_note_values``."""
    from local_llm_lab.probes import patch

    case, _task, _steps, _canonical = _real_ledger_case()

    scoring = patch.case_scoring(case)

    assert scoring.failing == ("34", "117", "131")
    assert scoring.dropped == (_R27_DROPPED,)
    assert scoring.canonical == ("34", "117", "145", "131")
    assert scoring.expected == frozenset({"34", "117", "145", "131"})
    assert scoring.required == frozenset({"34", "117", "131", "145"})
    # The failing note is a subset of the canonical field by construction; a non-empty list
    # here would mark an anomaly, never a silent adjustment.
    assert scoring.record()["failing_values_outside_canonical"] == []


def test_strict_scoring_refuses_a_turn_that_does_not_parse_or_carries_no_value(
    monkeypatch,
) -> None:
    """R27(1): both parse failures are ``parse_error``, and the raw head is kept."""
    from local_llm_lab.probes import patch

    scoring = patch.CaseScoring(
        failing=("1",), dropped=("2",), canonical=("1", "2"), fields=("approved",)
    )
    monkeypatch.setattr(patch, "strip_thinking", lambda raw: (None, raw))
    monkeypatch.setattr(
        patch, "parse_turn", lambda raw: (_ for _ in ()).throw(ValueError("bad"))
    )
    unparsed = patch.score_generation("x" * 400, scoring)

    assert unparsed.outcome == "parse_error"
    assert len(unparsed.raw_head) == 200 and unparsed.values == ()

    monkeypatch.setattr(patch, "parse_turn", lambda raw: SimpleNamespace(thought=raw))
    valueless = patch.score_generation("Reading the next invoice; pending: none.", scoring)

    # R30(5) split this out of ``parse_error``: the turn parsed, so nothing is unaccounted
    # for and no raw head is kept.  ``test_a_parseable_note_with_no_values_scores_empty``
    # carries the rest of the clause.
    assert valueless.outcome == "empty"
    assert valueless.raw_head == ""
    # A note that DOES parse to numbers is not a parse error, even when it is wrong.
    assert patch.score_generation("approved: 1, 2", scoring).outcome == "flip"
    assert patch.score_generation("approved: 1, 2, 3", scoring).outcome == "corrupted"
    assert patch.score_generation("approved: 1", scoring).outcome == "unchanged"


def test_dropped_value_visibility_uses_the_checkers_extraction_not_a_substring() -> None:
    """R27(3): ``_fact_is_visible`` over the retained window, never a digit-substring grep.

    The correction on issue #26: a substring test matched other numbers and reported three
    of the five P6 values as visible, which cannot be so — a visible fact is never scored as
    a drop.  The distractor line below contains ``145`` inside ``1450`` and inside prose, and
    must not count.
    """
    from local_llm_lab.probes import patch

    task = _Task("test-ledger_reconcile-0-clean", "ledger_reconcile")
    judgement = patch.DropJudgement(2, ("145",), "head")
    hidden = patch.PatchCase(
        task,
        2,
        (
            {"thought": "a", "observation": "path=inv/0 status=approved amount=145"},
            {"thought": "b", "observation": "context-9=historical 1450 for component 145x"},
            {"thought": "drop"},
        ),
        None,
        judgement,
        judgement,
    )
    visible = patch.PatchCase(
        task,
        2,
        (
            {"thought": "a", "observation": "noise"},
            {"thought": "b", "observation": "path=inv/1 status=approved amount=145"},
            {"thought": "drop"},
        ),
        None,
        judgement,
        judgement,
    )

    # keep_last=1 retains only the distractor, whose digits are not an ``amount`` fact.
    assert patch.dropped_value_visibility(hidden, keep_last=1) == {
        "dropped_value_visible_in_retained_observations": False,
        "visible_dropped_values": [],
        "retained_observations": 1,
    }
    # keep_last=2 retains the observation the amount actually came from.
    assert patch.dropped_value_visibility(hidden, keep_last=2)[
        "dropped_value_visible_in_retained_observations"
    ]
    assert patch.dropped_value_visibility(visible, keep_last=1) == {
        "dropped_value_visible_in_retained_observations": True,
        "visible_dropped_values": ["145"],
        "retained_observations": 1,
    }


def test_a_visible_dropped_value_excludes_the_case_from_the_headline(monkeypatch) -> None:
    """R27(3): True excludes, exactly as ``scoring_version_stable=False`` does, with a reason."""
    from local_llm_lab.probes import patch

    cases = [
        _probe_case(0, judgements=_judgements(1)),
        _probe_case(1, judgements=_judgements(1), observation="status=approved amount=42"),
        _probe_case(2, judgements=_judgements(1)),
    ]
    tokenizer, captures = _probe_fixture(monkeypatch, cases)
    resolved = SimpleNamespace(as_dict=lambda: {"name": "fake"})

    payload = patch.run_patch_probe(
        object(), tokenizer, cases, spec=object(), resolved=resolved, layers=[1], policy="base",
        keep_last=2, max_tokens=1, seed=7, command=["patch"],
    )

    assert len(captures) == 4, "the excluded case is never prepared or captured"
    assert (payload["stable_cases"], payload["unstable_cases"]) == (3, 0)
    assert (payload["headline_cases"], payload["visibility_excluded_cases"]) == (2, 1)
    assert payload["headline_task_ids"] == [cases[0].task.task_id, cases[2].task.task_id]
    assert [record["task_id"] for record in payload["excluded_cases"]] == [
        cases[1].task.task_id
    ]
    excluded = payload["excluded_cases"][0]
    assert excluded["excluded_reason"] == "dropped_value_visible_in_retained_observations"
    assert excluded["dropped_value_visible_in_retained_observations"] is True
    assert excluded["visible_dropped_values"] == ["42"]
    assert all(
        record["dropped_value_visible_in_retained_observations"] is False
        for record in payload["cases"]
    )
    for cell in payload["cells"].values():
        assert cell["treatment"]["denominator"] == 2


def test_content_swap_replaces_only_the_dropped_values_rows(monkeypatch) -> None:
    """R27(5): B's rows with the dropped value's positions carrying an unrelated value's.

    The residual fake gives position ``p`` the row ``[p]``, so the injected vector names the
    positions it was built from and the swap is checkable row by row.

    The replacement row is the float32 MEAN of the foreign value's rows, pooled exactly as
    R25(b) pools the treatment's own (``_alignment_rows``).  Review finding 10: taking that
    value's first token row unpooled made the control a different manipulation from the
    treatment, so the pooling is asserted here, not merely the positions.
    """
    payload, injected = _alignment_probe(
        monkeypatch,
        ["41", "57"],
        ["41", "57", "62"],
        other=(["41", "57"], ["41", "62", "57", "78"]),
    )

    first, second = payload["cases"]
    notes = first["alignment"]["previous_notes"]
    own_dropped = [
        position
        for slot in first["alignment"]["dropped_value_slot"]["slots"]
        for position in slot["pooled_source_positions"]
    ]
    # One pooled row per foreign dropped value: the mean of that value's own token rows.
    foreign_pooled = [
        sum(slot["pooled_source_positions"]) / len(slot["pooled_source_positions"])
        for slot in second["alignment"]["dropped_value_slot"]["slots"]
    ]
    assert len(second["alignment"]["dropped_value_slot"]["slots"]) == 2, (
        "the unrelated case must carry its own, different dropped-value rows"
    )
    assert all(
        len(slot["pooled_source_positions"]) > 1
        for slot in second["alignment"]["dropped_value_slot"]["slots"]
    ), "the pooling is only checkable when a foreign value spans several rows"
    source = notes["source_positions"]
    target = tuple(notes["target_positions"])
    indices = [index for index, position in enumerate(source) if position in own_dropped]
    assert indices, "the note cell must cover the dropped value's positions to be applicable"

    treatment = tuple(float(position) for position in source)
    swapped = list(treatment)
    for order, index in enumerate(indices):
        swapped[index] = foreign_pooled[order % len(foreign_pooled)]
    # A pooled mean is not any single foreign row, so this cannot pass by accident.
    assert not set(swapped[index] for index in indices) & {
        float(position)
        for slot in second["alignment"]["dropped_value_slot"]["slots"]
        for position in slot["pooled_source_positions"]
    }
    matches = [
        (np.asarray(rows), tuple(np.asarray(rows).reshape(-1)))
        for rows, positions in injected
        if positions == target
    ]
    vectors = [vector for _rows, vector in matches]

    assert treatment in vectors, "the treatment rows are unchanged"
    assert tuple(swapped) in vectors, "the content control writes the foreign value's rows"
    assert tuple(swapped) != treatment
    # Briefing rule 1.5: the pooled row is taken in float32, as R25(b) does for the treatment.
    swapped_rows = next(rows for rows, vector in matches if vector == tuple(swapped))
    assert swapped_rows.dtype == np.float32
    # Every non-dropped row is carried through untouched: only the value's rows move.
    assert [
        value for index, value in enumerate(swapped) if index not in indices
    ] == [value for index, value in enumerate(treatment) if index not in indices]
    control = payload["cells"]["1:previous_notes"]["controls"]["content_swap"]
    assert control["applicable"] is True
    assert control["swapped_rows"][first["task_id"]] == len(indices)

    # At ``dropped_value_slot`` every row IS a dropped-value row, so once the replacement is
    # pooled the content control and ``unrelated_task`` write the identical rows there. That
    # is the cell's own statement, and it only holds because both sides pool the same way.
    slot_target = tuple(first["alignment"]["dropped_value_slot"]["target_positions"])
    slot_vectors = {
        tuple(np.asarray(rows).reshape(-1))
        for rows, positions in injected
        if positions == slot_target
    }
    assert len(slot_vectors) == 2, (
        "the slot cell writes exactly two distinct row sets: the case's own pooled row "
        "(treatment) and the foreign pooled row (unrelated_task and content_swap)"
    )
    assert (foreign_pooled[0],) in slot_vectors


def test_content_swap_is_not_applicable_where_no_dropped_value_row_is_read(monkeypatch) -> None:
    """R27(5): cells that read none of those positions record ``not_applicable``, not a rate."""
    from local_llm_lab.probes import patch

    payload, _injected = _alignment_probe(monkeypatch, ["41", "57"], ["41", "57", "62"])

    applicable = {
        key.split(":", 1)[1]: cell["controls"]["content_swap"].get("applicable", False)
        for key, cell in payload["cells"].items()
    }

    # The dropped value's tokens sit inside the note, so the note cell and the slot cell
    # read them; the system prompt, task prompt, shared values, observations and the final
    # token do not.
    assert applicable == {
        "system_prompt": False,
        "task_prompt": False,
        "previous_notes": True,
        "shared_value_tokens": False,
        "dropped_value_slot": True,
        "last_two_observations": False,
        "final_token": False,
    }
    inert = payload["cells"]["1:system_prompt"]["controls"]["content_swap"]
    assert set(inert) == {"applicable", "reason", "applicable_cases", "cases"}
    assert "nothing to swap" in inert["reason"]
    # R30(6): even an inapplicable control states the n it was NOT taken over.
    assert (inert["applicable_cases"], inert["cases"]) == (0, 2)
    assert set(payload["cells"]["1:system_prompt"]["controls"]) == set(patch.CONTROLS)


def test_aggregate_counts_outcomes_over_generations_and_rates_over_cases() -> None:
    """R27(2): the case-level flip rate is unchanged; the counts are per generation."""
    from local_llm_lab.probes import patch

    summary = patch.aggregate_task_outcomes(
        {
            "task-a": ["corrupted", "flip"],
            "task-b": ["unchanged", "unchanged"],
            "task-c": ["parse_error"],
        }
    )

    assert (summary["numerator"], summary["denominator"], summary["rate"]) == (1, 3, 1 / 3)
    assert summary["outcomes"] == {
        "flip": 1,
        "corrupted": 1,
        "unchanged": 2,
        "empty": 0,
        "parse_error": 1,
    }
    assert summary["generations"] == 5
    assert summary["wilson_95"] == list(patch.wilson(1, 3))
    # The same rate the pre-R27 aggregate would report for the same cases.
    assert summary["rate"] == patch.aggregate_task_flips(
        {"task-a": [False, True], "task-b": [False, False], "task-c": [False]}
    )["rate"]
    with pytest.raises(ValueError, match="unknown R27 outcome"):
        patch.aggregate_task_outcomes({"task-a": ["flipped"]})


def test_secondary_condition_selects_the_family_without_the_passing_intersection(
    monkeypatch,
) -> None:
    """R27(5), issue #26: ``aggregate_report`` failures with the designed-correct note.

    The family fails under B and C alike, so the primary universe (pass under B, fail under
    C) is empty for it; the secondary universe drops that intersection and takes the
    generator note as the counterfactual instead.
    """
    from local_llm_lab.probes import patch

    tasks = {
        "test-aggregate_report-0-clean": _Task(
            "test-aggregate_report-0-clean", "aggregate_report"
        ),
        "test-ledger_reconcile-1-clean": _Task(
            "test-ledger_reconcile-1-clean", "ledger_reconcile"
        ),
    }
    monkeypatch.setattr(patch, "task_from_id", lambda task_id, seed, difficulty: tasks[task_id])
    monkeypatch.setattr(
        patch, "check_trajectory", lambda task, steps, *, keep_last: _drop_report(1, "7")
    )
    passing = _payload(
        [
            {
                "task_id": "test-ledger_reconcile-1-clean",
                "verdict": {"success": True},
                "steps": [{"thought": "saved note"}],
            }
        ]
    )
    failing = _payload(
        [
            {
                "task_id": key,
                "difficulty": 2,
                "verdict": {"success": False},
                "steps": [{"thought": "before"}, {"thought": "drop"}],
                "integrity": {"violations": [{"kind": "value_drop", "step": 1, "detail": "d: 7"}]},
            }
            for key in tasks
        ]
    )

    primary, primary_provenance = patch.select_patch_cases(passing, failing, keep_last=2)
    secondary, secondary_provenance = patch.select_patch_cases(
        passing, failing, keep_last=2, secondary_condition="aggregate_report"
    )

    # Only the ledger task has a pass/fail pair, which is why the primary run is ledger-only.
    assert [case.task.task_id for case in primary] == ["test-ledger_reconcile-1-clean"]
    assert primary[0].condition == patch.PRIMARY_CONDITION
    assert primary_provenance["condition"] == patch.PRIMARY_CONDITION
    assert [case.task.task_id for case in secondary] == ["test-aggregate_report-0-clean"]
    assert secondary[0].condition == "aggregate_report_secondary"
    assert secondary[0].counterfactual_label == "designed_correct"
    assert secondary_provenance["counterfactual_label"] == "designed_correct"
    # No passing steps, so ``counterfactual_note`` falls back to the HEAD generator note.
    assert secondary[0].passing_steps is None
    monkeypatch.setattr(patch, "render_expert_note", lambda _task, index: f"expert {index}")
    note, note_provenance = patch.counterfactual_note(secondary[0])
    assert note == "expert 0"
    assert note_provenance["counterfactual_source"] == f"generator_v{patch.GENERATOR_VERSION}"
    with pytest.raises(ValueError, match="unknown secondary condition"):
        patch.select_patch_cases(
            passing, failing, keep_last=2, secondary_condition="ledger_reconcile"
        )
    with pytest.raises(ValueError, match="one condition at a time"):
        patch.condition_of([*primary, *secondary])


def test_patch_cli_accepts_the_secondary_condition_flag(monkeypatch, tmp_path) -> None:
    """R27(5): the CLI shape, and the secondary section never merging into the headline."""
    from local_llm_lab.probes import guard, patch

    passing = tmp_path / "passing.json"
    failing = tmp_path / "failing.json"
    passing.write_text("{}", encoding="utf-8")
    failing.write_text("{}", encoding="utf-8")
    spec = SimpleNamespace(
        name="qwen35-4b",
        hf_id="fake/hf",
        policies={},
        probe_layer_fractions=(0.25,),
        resolve=lambda *_args: SimpleNamespace(num_layers=4, probe_layers=(1,)),
    )
    case = patch.PatchCase(
        _Task("test-ledger_reconcile-1-clean", "ledger_reconcile"), 0, (), None, *_judgements(0)
    )
    secondary_case = patch.PatchCase(
        _Task("test-aggregate_report-0-clean", "aggregate_report"),
        0,
        (),
        None,
        *_judgements(0),
        "aggregate_report_secondary",
    )
    selections = []

    def select(*_args, **kwargs):
        condition = kwargs.get("secondary_condition")
        selections.append(condition)
        provenance = {
            "data_seeds": {},
            "eligibility": {},
            "condition": "aggregate_report_secondary" if condition else patch.PRIMARY_CONDITION,
            "counterfactual_label": "designed_correct" if condition else "empirically_passing",
        }
        return ([secondary_case] if condition else [case]), provenance

    runs = []

    def run(_model, _tokenizer, cases, **_kwargs):
        runs.append([item.task.task_id for item in cases])
        return {"cells": {}, "groups": [], "layers": [], "controls": [], "condition": "x"}

    monkeypatch.setattr(patch, "select_patch_cases", select)
    monkeypatch.setattr(patch, "load_model_spec", lambda _name: spec)
    monkeypatch.setattr(patch, "require_preflight", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(guard, "require_idle_gpu", lambda *_args: None)
    monkeypatch.setattr(patch, "resolve_policy", lambda _name, _spec: None)
    monkeypatch.setattr(
        patch,
        "load_policy",
        lambda _spec, _adapter: (
            object(),
            object(),
            SimpleNamespace(num_layers=4),
            SimpleNamespace(as_dict=dict),
        ),
    )
    monkeypatch.setattr(patch, "run_patch_probe", run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "agent-v2-probe-patch",
            "--passing-eval", str(passing),
            "--failing-eval", str(failing),
            "--output", str(tmp_path),
            "--secondary-condition", "aggregate_report",
        ],
    )

    patch.main()

    assert selections == [None, "aggregate_report"]
    assert runs == [
        ["test-ledger_reconcile-1-clean"],
        ["test-aggregate_report-0-clean"],
    ]
    payload = json.loads((tmp_path / "patch.json").read_text(encoding="utf-8"))
    assert payload["secondary"]["counterfactual_label"] == "designed_correct"
    assert payload["secondary"]["selected_task_ids"] == ["test-aggregate_report-0-clean"]
    # The secondary population is a section, never merged into the headline's own keys.
    assert "test-aggregate_report-0-clean" not in json.dumps(
        {key: value for key, value in payload.items() if key != "secondary"}
    )
    assert "# Secondary condition (R27, issue #26)" in (tmp_path / "patch.md").read_text()


def test_render_markdown_carries_the_outcome_counts_and_the_content_control() -> None:
    """R27(7): the counts, the content control's applicability, and the secondary section."""
    from local_llm_lab.probes import patch

    counts = {"flip": 3, "corrupted": 1, "unchanged": 0, "empty": 0, "parse_error": 1}
    payload = {
        "groups": ["system_prompt"],
        "layers": [1],
        "controls": list(patch.CONTROLS),
        "outcomes": list(patch.OUTCOMES),
        "condition": patch.PRIMARY_CONDITION,
        "cells": {
            "1:system_prompt": {
                "treatment": {
                    "rate": 0.6,
                    "wilson_95": [0.2, 0.9],
                    "outcomes": counts,
                    "applicable_cases": 5,
                    "cases": 5,
                },
                "controls": {
                    "unrelated_task": {
                        "rate": 0.2,
                        "wilson_95": [0.0, 0.6],
                        "outcomes": counts,
                        "applicable_cases": 5,
                        "cases": 5,
                    },
                    "random_positions": {
                        "rate": 0.0,
                        "wilson_95": [0.0, 0.4],
                        "outcomes": counts,
                        "applicable_cases": 5,
                        "cases": 5,
                    },
                    "content_swap": {
                        "applicable": False,
                        "reason": "nothing to swap",
                        "applicable_cases": 0,
                        "cases": 5,
                    },
                },
            }
        },
        "secondary": {
            "condition": "aggregate_report_secondary",
            "counterfactual_label": "designed_correct",
            "selected_task_ids": ["test-aggregate_report-0-clean"],
            "error": "no eligible patch cases",
        },
    }

    lines = patch.render_markdown(payload).splitlines()

    assert "Condition: `primary`." in lines
    assert "## Outcome counts per cell (R27)" in lines
    # R30(5)/(6): the ``empty`` column, and the n each row was taken over.
    assert "| 1:system_prompt | treatment | n=5/5 | 3 | 1 | 0 | 0 | 1 |" in lines
    assert "| 1:system_prompt | content_swap | n=0/5 | n/a | n/a | n/a | n/a | n/a |" in lines
    assert "## Control: content_swap" in lines
    assert "| 1:system_prompt | n=0/5 | not_applicable | nothing to swap |" in lines
    assert "| 1:system_prompt | n=5/5 | 0.200 | [0.000, 0.600] |" in lines
    secondary = lines.index("# Secondary condition (R27, issue #26)")
    assert any("designed_correct" in line for line in lines[secondary:])
    assert any("Not scored: no eligible patch cases" in line for line in lines[secondary:])


@pytest.mark.skipif(
    not all(path.is_file() for path in _SAVED_EVALS),
    reason="protected saved evaluations not present on this checkout",
)
def test_real_saved_evaluations_measure_every_dropped_value_as_hidden() -> None:
    """R27(3) on the real evidence, re-measured with the checker's own extraction.

    Reads the protected evidence read-only, with no model and no tokenizer.  Pins the
    Chief's correction on issue #26: none of the five dropped values is visible in the
    retained observations (a visible fact is never scored as a drop), and four of the five
    are referenced in an EARLIER note of the failing prompt, which is why a flip does not
    require the injected content in those four.  The seed and version literals are fixture
    data: the runs' recorded configuration, not constants used by the tool.
    """
    import json

    from local_llm_lab.pipeline.integrity import _fact_is_referenced, required_carry
    from local_llm_lab.probes import patch

    passing = json.loads(_SAVED_EVALS[0].read_text(encoding="utf-8"))
    failing = json.loads(_SAVED_EVALS[1].read_text(encoding="utf-8"))
    cases, _provenance = patch.select_patch_cases(
        passing, failing, keep_last=2, data_seed=20260902, generator_version=1
    )

    assert len(cases) == 5
    visibility = {
        case.task.task_id: patch.dropped_value_visibility(case, keep_last=2) for case in cases
    }
    assert {
        task_id: record["dropped_value_visible_in_retained_observations"]
        for task_id, record in visibility.items()
    } == {
        "test-ledger_reconcile-0031-clean": False,
        "test-ledger_reconcile-0127-clean": False,
        "test-ledger_reconcile-0139-clean": False,
        "test-ledger_reconcile-0163-clean": False,
        "test-ledger_reconcile-0175-clean": False,
    }
    assert all(record["retained_observations"] == 2 for record in visibility.values())

    # The earlier-note references, measured with ``_fact_is_referenced``: the note index of
    # the first earlier note carrying the dropped value, or None when no note carries it.
    references = {}
    scoring = {}
    for case in cases:
        facts = {
            fact
            for fact in required_carry(case.task, keep_last=2)[case.decision_step]
            if fact.value in set(case.head_judgement.dropped_values or ())
        }
        references[case.task.task_id] = next(
            (
                index
                for index, record in enumerate(case.failing_steps[: case.decision_step])
                if any(_fact_is_referenced(fact, record.get("thought") or "") for fact in facts)
            ),
            None,
        )
        scoring[case.task.task_id] = patch.case_scoring(case)
    assert references == {
        "test-ledger_reconcile-0031-clean": 5,
        "test-ledger_reconcile-0127-clean": 5,
        "test-ledger_reconcile-0139-clean": 5,
        # 0163's value 32 has no text source anywhere in the failing prompt, so a flip there
        # would be genuine retrieval from the injected rows.
        "test-ledger_reconcile-0163-clean": None,
        "test-ledger_reconcile-0175-clean": 4,
    }
    # Every case admits a strict flip: D is inside E and F is a subset of E.
    for task_id, record in scoring.items():
        assert set(record.dropped) <= record.expected, task_id
        assert set(record.failing) <= record.expected, task_id
        assert record.record()["failing_values_outside_canonical"] == [], task_id
    assert {task_id: record.dropped for task_id, record in scoring.items()} == {
        "test-ledger_reconcile-0031-clean": ("85",),
        "test-ledger_reconcile-0127-clean": ("89",),
        "test-ledger_reconcile-0139-clean": ("100",),
        "test-ledger_reconcile-0163-clean": ("32",),
        "test-ledger_reconcile-0175-clean": ("54",),
    }

    # R27(5): the bound secondary universe, measured on the same files.
    secondary, secondary_provenance = patch.select_patch_cases(
        passing,
        failing,
        keep_last=2,
        data_seed=20260902,
        generator_version=1,
        secondary_condition="aggregate_report",
    )
    assert secondary, "the secondary condition must select at least one aggregate_report case"
    assert {case.task.family for case in secondary} == {"aggregate_report"}
    assert all(case.passing_steps is None for case in secondary)
    assert all(case.counterfactual_label == "designed_correct" for case in secondary)
    assert secondary_provenance["condition"] == "aggregate_report_secondary"
    assert not {case.task.task_id for case in secondary} & {
        case.task.task_id for case in cases
    }

    # R30(2), measured on the real files: judged under HEAD alone, no v1 replay is performed,
    # and the whole 15-case population is eligible instead of the 2 that survived R24's
    # bound-vs-HEAD comparison (the aggregate_report note template moved between v1 and v4).
    assert len(secondary) == 15
    assert secondary_provenance["eligibility"]["failing"]["integrity"] == {
        "evaluation": 0,
        "recomputed": 0,
        "head_alone": 15,
    }
    assert (
        secondary_provenance["eligibility"]["failing"]["eligibility_basis"]
        == patch.HEAD_ALONE_BASIS
    )
    assert all(case.scoring_version_stable for case in secondary)
    assert all(
        not patch.dropped_value_visibility(case, keep_last=2)[
            "dropped_value_visible_in_retained_observations"
        ]
        for case in secondary
    )
    # R30(4), measured: five failing notes carry only a computed subtotal that the canonical
    # value list does not contain, so no generation could ever score a flip for them.
    secondary_scoring = {case.task.task_id: patch.case_scoring(case) for case in secondary}
    assert {
        task_id: sorted(record.required - record.expected)
        for task_id, record in secondary_scoring.items()
        if not record.flip_satisfiable
    } == {
        "test-aggregate_report-0011-clean": ["32"],
        "test-aggregate_report-0107-clean": ["139"],
        "test-aggregate_report-0131-clean": ["115"],
        "test-aggregate_report-0155-clean": ["113"],
        "test-aggregate_report-0179-clean": ["67"],
    }
    assert sum(record.flip_satisfiable for record in secondary_scoring.values()) == 10
    # R30(1), measured: the two cases whose note uses run C's ``first half complete:`` style
    # now parse to a value set; before R30 the colon had to follow ``half`` directly and F
    # was empty for both.  Their scope is empty because the HEAD canonical note carries a
    # ``values so far`` field instead, so E falls back to the canonical note's own list
    # fields — still field-scoped, and both cases stay satisfiable.
    half_style = {
        case.task.task_id: case
        for case in secondary
        if "half complete" in (case.failing_steps[case.decision_step].get("thought") or "")
    }
    assert set(half_style) == {
        "test-aggregate_report-0035-clean",
        "test-aggregate_report-0143-clean",
    }
    for task_id, case in half_style.items():
        note = case.failing_steps[case.decision_step]["thought"]
        assert tuple(patch._note_value_fields(note)) == (
            "first half complete",
            "second half complete",
        ), task_id
        assert secondary_scoring[task_id].failing, task_id
        assert secondary_scoring[task_id].flip_satisfiable, task_id
    assert secondary_scoring["test-aggregate_report-0035-clean"].failing == (
        "67",
        "16",
        "71",
        "26",
    )
    # R30(3), measured on the primary: scoping to the failing note's own list field leaves the
    # five ledger cases exactly as they were, so the rerun is comparable.
    assert {
        case.task.task_id: patch.case_scoring(case).fields for case in cases
    } == dict.fromkeys((case.task.task_id for case in cases), ("approved",))
    assert all(patch.case_scoring(case).flip_satisfiable for case in cases)


# --- R30 (map:658-664): secondary condition and scorer refinements -------------------------
#
# Six clauses, each with its own test below:
#   (1) ``_note_values`` accepts words between ``half`` and the colon;
#   (2) the ``aggregate_report`` secondary condition's eligibility is judged under HEAD alone
#       with the basis recorded (no v1-bound side);
#   (3) the expected set is field-scoped to the failing note's list field;
#   (4) ``flip_unsatisfiable`` cases are marked and excluded from the headline;
#   (5) a parseable note with no values scores ``empty``;
#   (6) control rates are over applicable cases with n printed.


def test_note_values_accepts_words_between_half_and_the_colon() -> None:
    """R30(1): run C's ``first half complete:`` style, not only ``first half:``.

    Measured on the saved run: the ``aggregate_report`` note at the calculate step reads
    ``first half complete: 67 + 16; second half complete: 71 + 26.`` — the pre-R30 regex
    required the colon immediately after ``half``, so it extracted nothing and F was empty.
    """
    from local_llm_lab.probes import patch

    run_c = "first half complete: 67 + 16; second half complete: 71 + 26. Computing the first half."

    assert patch._note_values(run_c) == ["67", "16", "71", "26"]
    # The plain form still parses, and the order is document order (R25 pairs values by
    # string identity but ``_value_alignment`` reads ``list(source_values)`` in this order).
    assert patch._note_values("first half: 73, 44 (full); second half: 87, 89.") == [
        "73",
        "44",
        "87",
        "89",
    ]
    # The allowance stops at the clause: it cannot reach across ``;`` or ``.`` to a later colon.
    assert patch._note_values("first half; nothing here. approved: 5") == ["5"]
    # A label with words is reported under its own name, so the field scoping in R30(3) can
    # tell one list field from another.
    assert patch._note_value_fields(run_c) == {
        "first half complete": ("67", "16"),
        "second half complete": ("71", "26"),
    }
    # Subtotal and ``highest so far`` expressions are values but are NOT list fields.
    assert patch._note_value_fields("First subtotal = 32. Computing the rest.") == {}
    assert patch._note_values("First subtotal = 32. Computing the rest.") == ["32"]


def _aggregate_case(*, failing_note, canonical_note, dropped, step=1):
    """A case whose task carries a real canonical note at the decision step."""
    from local_llm_lab.probes import patch

    task = _Task(
        "test-aggregate_report-0023-clean",
        "aggregate_report",
        steps=(SimpleNamespace(thought="plan"), SimpleNamespace(thought=canonical_note)),
    )
    judgement = patch.DropJudgement(step, tuple(dropped), "head")
    return patch.PatchCase(
        task,
        step,
        ({"thought": "plan"}, {"thought": failing_note}),
        None,
        judgement,
        judgement,
    )


def test_expected_set_is_scoped_to_the_failing_notes_list_field() -> None:
    """R30(3): E comes from the canonical note's SAME list field, not every field.

    The reviewer's case-0023 shape: the canonical note carries both a value list and a
    computed subtotal (``values so far: 73, 44, 87, 89, 11, 89; … First subtotal = 204``).
    Flattening E across fields would admit 204 as a legitimate value, so a regenerated list
    ``values so far: 73, 44, 204`` would escape the corruption test. Field scoping refuses it.
    """
    from local_llm_lab.probes import patch

    canonical = (
        "values so far: 73, 44, 87, 89, 11, 89; split after 3 of 6. "
        "First subtotal = 204; computing 89 + 11 + 89."
    )
    case = _aggregate_case(
        failing_note="values so far: 73, 44, 87, 89, 89; split after 3 of 6.",
        canonical_note=canonical,
        dropped=("11",),
    )

    scoring = patch.case_scoring(case)

    assert scoring.fields == ("values so far",)
    # 204 is in the canonical note but not in the list field, so it is not expected.
    assert scoring.expected == frozenset({"73", "44", "87", "89", "11"})
    assert "204" in patch._note_values(canonical), "the flattened set would have held 204"
    assert patch.classify_generation(["73", "44", "87", "89", "11", "89"], scoring) == "flip"
    assert patch.classify_generation(["73", "44", "204"], scoring) == "corrupted"
    assert scoring.record()["expected_field_scope"] == ["values so far"]
    # A failing note with no list field at all falls back to the canonical note's own list
    # fields — still field-scoped, so the subtotal is still excluded.
    bare = _aggregate_case(
        failing_note="First subtotal = 204. Computing the second subtotal.",
        canonical_note=canonical,
        dropped=("11",),
    )
    bare_scoring = patch.case_scoring(bare)
    assert bare_scoring.fields == ()
    assert "204" not in bare_scoring.expected
    assert bare_scoring.record()["expected_field_scope"] == []


def test_flip_unsatisfiable_cases_are_marked_and_excluded(monkeypatch) -> None:
    """R30(4): required ⊄ expected means no generation could ever score ``flip``.

    Measured shape from the run: the failing note carries a computed subtotal that is not in
    the canonical value list, so F ⊄ E and the case can only ever be ``corrupted``. Scoring
    it would report a false 0.0, so it is excluded and the reason recorded.
    """
    from local_llm_lab.probes import patch

    canonical = (
        "values so far: 18, 14, 75, 72, 18, 79; split after 3 of 6. "
        "First subtotal = 107; computing 72 + 18 + 79."
    )
    unsatisfiable = _aggregate_case(
        failing_note="First subtotal = 32. Computing the second subtotal 75 + 72.",
        canonical_note=canonical,
        dropped=("14", "18"),
    )
    satisfiable = _aggregate_case(
        failing_note="values so far: 18, 75, 72, 79; split after 3 of 6.",
        canonical_note=canonical,
        dropped=("14",),
    )

    assert patch.case_scoring(unsatisfiable).flip_satisfiable is False
    assert patch.case_scoring(satisfiable).flip_satisfiable is True
    assert patch.case_scoring(unsatisfiable).record()["flip_satisfiable"] is False

    cases = [
        _probe_case(0, judgements=_judgements(1)),
        _probe_case(1, judgements=_judgements(1), failing_note="approved: 999"),
        _probe_case(2, judgements=_judgements(1)),
    ]
    tokenizer, captures = _probe_fixture(monkeypatch, cases)
    resolved = SimpleNamespace(as_dict=lambda: {"name": "fake"})

    payload = patch.run_patch_probe(
        object(), tokenizer, cases, spec=object(), resolved=resolved, layers=[1], policy="base",
        keep_last=2, max_tokens=1, seed=7, command=["patch"],
    )

    assert len(captures) == 4, "the unsatisfiable case is never prepared or captured"
    assert payload["headline_cases"] == 2
    assert payload["flip_unsatisfiable_cases"] == 1
    assert payload["headline_task_ids"] == [cases[0].task.task_id, cases[2].task.task_id]
    excluded = payload["excluded_cases"][0]
    assert excluded["task_id"] == cases[1].task.task_id
    assert excluded["excluded_reason"] == "flip_unsatisfiable"
    assert excluded["value_sets"]["flip_satisfiable"] is False
    for cell in payload["cells"].values():
        assert cell["treatment"]["denominator"] == 2


def test_a_parseable_note_with_no_values_scores_empty(monkeypatch) -> None:
    """R30(5): ``empty`` is its own outcome, and it still records the legacy diagnostic.

    A note that parses but carries no value list is not a parse failure — the model wrote a
    turn, it just did not restate the list — and folding it into ``parse_error`` hid that.
    """
    from local_llm_lab.probes import patch

    assert patch.OUTCOMES == ("flip", "corrupted", "unchanged", "empty", "parse_error")
    scoring = patch.CaseScoring(
        failing=("1",), dropped=("2",), canonical=("1", "2"), fields=("approved",)
    )
    monkeypatch.setattr(patch, "strip_thinking", lambda raw: (None, raw))
    monkeypatch.setattr(patch, "parse_turn", lambda raw: SimpleNamespace(thought=raw))

    scored = patch.score_generation("Reading the next invoice; pending: none.", scoring)

    assert scored.outcome == "empty"
    assert scored.values == ()
    assert scored.note == "Reading the next invoice; pending: none."
    # It parsed, so the raw head is not kept: nothing is unaccounted for.
    assert scored.raw_head == ""

    class Hook:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    case = patch.PatchCase(
        _Task("test-aggregate_report-0-clean", "aggregate_report"), 0, ({"thought": "bad"},)
    )
    monkeypatch.setattr(patch, "InjectionHook", Hook)
    monkeypatch.setattr(patch, "greedy_generate", lambda *_args, **_kwargs: "no values here.")
    monkeypatch.setattr(patch, "_is_flip", lambda *_args, **_kwargs: True)

    through = patch._score_patch(
        object(), object(), case, scoring=scoring, layer=1, source_rows=object(),
        target_positions=(0,), failing_ids=[1], keep_last=2, max_tokens=1,
    )

    assert through.outcome == "empty"
    # R30(5): unlike ``parse_error``, an ``empty`` generation still records the pre-R27 rule.
    assert through.value_drop_cleared is True

    counts = patch.aggregate_task_outcomes({"task-a": ["empty"], "task-b": ["flip"]})
    assert counts["outcomes"]["empty"] == 1
    assert counts["rate"] == 0.5


def test_control_rates_are_over_applicable_cases_with_n_recorded(monkeypatch) -> None:
    """R30(6): a control's rate covers only the cases it could be run on, and n is printed."""
    from local_llm_lab.probes import patch

    payload, _injected = _alignment_probe(monkeypatch, ["41", "57"], ["41", "57", "62"])

    for key, cell in payload["cells"].items():
        for name, control in cell["controls"].items():
            if not control.get("applicable", True):
                assert control["applicable_cases"] == 0, key
                assert control["cases"] == 2, key
                continue
            assert control["applicable_cases"] == control["denominator"], (key, name)
            assert control["cases"] == 2, (key, name)
        assert cell["treatment"]["applicable_cases"] == cell["treatment"]["denominator"]

    swap = payload["cells"]["1:previous_notes"]["controls"]["content_swap"]
    assert swap["applicable_cases"] == 2 and swap["cases"] == 2
    text = patch.render_markdown(payload)
    assert "n=2/2" in text
    assert "n=0/2" in text, "an inapplicable control prints its n, it does not hide"


def test_secondary_eligibility_is_judged_under_head_alone_with_the_basis_recorded(
    monkeypatch,
) -> None:
    """R30(2): no v1-bound side for the secondary condition, and the basis is in the artifact.

    Its counterfactual is already a HEAD generator note, so a version-bound judgement has
    nothing to bind to; judging it under R24's bound-vs-HEAD comparison excluded cases for a
    disagreement that cannot matter here.
    """
    from local_llm_lab.probes import patch

    tasks = {
        "test-aggregate_report-0-clean": _Task(
            "test-aggregate_report-0-clean", "aggregate_report"
        ),
        "test-ledger_reconcile-1-clean": _Task(
            "test-ledger_reconcile-1-clean", "ledger_reconcile"
        ),
    }
    monkeypatch.setattr(patch, "task_from_id", lambda task_id, seed, difficulty: tasks[task_id])
    replays = []

    def replay(task_id, seed, version, level):
        replays.append((task_id, version))
        return tasks[task_id]

    monkeypatch.setattr(patch, "replay_task_from_id", replay)
    # The bound replay and HEAD disagree on both the step and the values, which is what
    # excluded 13 of the 15 real cases before R30.
    monkeypatch.setattr(
        patch,
        "check_trajectory",
        lambda task, steps, *, keep_last: _drop_report(1, "7")
        if task is tasks["test-ledger_reconcile-1-clean"]
        else _drop_report(1, "7"),
    )
    passing = _payload(
        [
            {
                "task_id": "test-ledger_reconcile-1-clean",
                "verdict": {"success": True},
                "steps": [{"thought": "saved note"}],
            }
        ]
    )
    failing = _payload(
        [
            {
                "task_id": key,
                "difficulty": 2,
                "verdict": {"success": False},
                "steps": [{"thought": "before"}, {"thought": "drop"}],
            }
            for key in tasks
        ],
        generator_version=1,
    )

    secondary, provenance = patch.select_patch_cases(
        passing, failing, keep_last=2, secondary_condition="aggregate_report"
    )

    (case,) = secondary
    assert case.task.task_id == "test-aggregate_report-0-clean"
    # No v1 replay was performed for this universe at all: HEAD alone.
    assert replays == []
    assert case.bound_judgement is case.head_judgement
    assert case.scoring_version_stable is True
    assert case.eligibility_basis == patch.HEAD_ALONE_BASIS
    record = case.scoring_record()
    assert record["eligibility_basis"] == patch.HEAD_ALONE_BASIS
    assert record["judged_under_bound"] == record["judged_under_head"]
    assert provenance["eligibility"]["failing"]["eligibility_basis"] == patch.HEAD_ALONE_BASIS
    # The primary universe keeps the R24 bound-vs-HEAD comparison and does replay.
    primary, primary_provenance = patch.select_patch_cases(passing, failing, keep_last=2)
    assert replays == [("test-ledger_reconcile-1-clean", 1)]
    assert primary[0].eligibility_basis == patch.BOUND_VS_HEAD_BASIS
    assert (
        primary_provenance["eligibility"]["failing"]["eligibility_basis"]
        == patch.BOUND_VS_HEAD_BASIS
    )


# ------------------------------- SPEC-001 §9/§10 closure: preflight, provenance, R18a/R18b


def _p6_cli_fixture(monkeypatch, tmp_path: Path, capture_dtype: str = "native"):
    """The P6 CLI reduced to files and fakes: no eval payload parsing, no model."""
    from local_llm_lab.probes import patch

    passing = tmp_path / "passing.json"
    failing = tmp_path / "failing.json"
    passing.write_text("{}", encoding="utf-8")
    failing.write_text("{}", encoding="utf-8")
    selected = SimpleNamespace(
        name="qwen35-4b",
        hf_id="fake/hf",
        policies={},
        probe_layer_fractions=(0.25, 0.5),
        probes=SimpleNamespace(capture_dtype=capture_dtype),
    )
    case = patch.PatchCase(
        _Task("test-aggregate_report-0-clean", "aggregate_report"),
        0,
        (),
        None,
        *_judgements(0),
    )
    monkeypatch.setattr(
        patch,
        "select_patch_cases",
        lambda *_args, **_kwargs: ([case], {"data_seeds": {}, "eligibility": {}}),
    )
    monkeypatch.setattr(patch, "load_model_spec", lambda _name: selected)
    monkeypatch.setattr(patch, "resolve_policy", lambda _name, _spec: None)
    argv = [
        "agent-v2-probe-patch",
        "--passing-eval",
        str(passing),
        "--failing-eval",
        str(failing),
        "--output",
        str(tmp_path / "p6"),
        "--model",
        "qwen35-4b",
        "--policy",
        "base",
        "--layers",
        "1",
    ]
    monkeypatch.setattr("sys.argv", argv)
    return selected


def test_patch_cli_refuses_to_load_without_preflight_evidence(monkeypatch, tmp_path) -> None:
    """SPEC-001 §10: no preflight artifact for the resolved model, no model load."""
    from local_llm_lab.pipeline import preflight
    from local_llm_lab.probes import guard, patch

    _p6_cli_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(preflight, "_OUTPUT_DIRECTORY", tmp_path / "preflight")
    monkeypatch.setattr(
        guard, "require_idle_gpu", lambda *_args: pytest.fail("reached the GPU guard")
    )
    monkeypatch.setattr(
        patch, "load_policy", lambda *_args: pytest.fail("reached the model loader")
    )

    with pytest.raises(SystemExit) as raised:
        patch.main()

    assert "qwen35-4b" in str(raised.value)
    assert "preflight" in str(raised.value)


def test_patch_cli_writes_provenance_beside_the_artifact(monkeypatch, tmp_path) -> None:
    """SPEC-001 §9: a completed P6 run records what produced it."""
    from local_llm_lab.probes import guard, patch

    order: list[str] = []
    _p6_cli_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(
        patch,
        "require_preflight",
        lambda spec, **kwargs: order.append(f"preflight:{spec.name}:{kwargs['skip']}"),
    )
    monkeypatch.setattr(guard, "require_idle_gpu", lambda *_args: order.append("guard"))
    monkeypatch.setattr(
        patch,
        "load_policy",
        lambda *_args: order.append("load")
        or (
            object(),
            object(),
            SimpleNamespace(num_layers=4),
            SimpleNamespace(as_dict=dict),
        ),
    )
    monkeypatch.setattr(
        patch,
        "run_patch_probe",
        lambda *_args, **_kwargs: {"groups": [], "layers": [], "controls": [], "cells": {}},
    )
    monkeypatch.setattr(patch, "render_markdown", lambda _payload: "# fake")

    patch.main()

    assert order == ["preflight:qwen35-4b:False", "guard", "load"]
    output = tmp_path / "p6"
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["extra"]["stage"] == "p6-patch"
    assert provenance["extra"]["artifacts"] == [
        str(output / "patch.json"),
        str(output / "patch.md"),
    ]


def test_patch_probe_records_the_precision_block_and_capture_dtype(monkeypatch) -> None:
    """R18a/R18b: P6 captures at the registry dtype and carries the preflight deviation."""
    from local_llm_lab.probes import patch

    dtypes: list[str] = []
    block = {"frobenius_relative": 0.004, "elementwise_max": 0.02}
    # Two stable cases: the unrelated-task control needs a second case to draw from.
    cases = [_probe_case(0, judgements=_judgements(1)), _probe_case(1, judgements=_judgements(1))]
    tokenizer, _captures = _probe_fixture(monkeypatch, cases)
    monkeypatch.setattr(patch, "preflight_precision_block", lambda _spec: dict(block))

    def capture(_view, ids, layers, *, positions, dtype):
        del positions
        dtypes.append(dtype)
        return {layer: mx.zeros((len(ids), 1)) for layer in layers}

    monkeypatch.setattr(patch, "capture_residuals", capture)
    spec = SimpleNamespace(name="qwen35-4b", probes=SimpleNamespace(capture_dtype="native"))

    payload = patch.run_patch_probe(
        object(),
        tokenizer,
        cases,
        spec=spec,
        resolved=SimpleNamespace(as_dict=dict),
        layers=[1],
        policy="base",
        keep_last=2,
        max_tokens=1,
        seed=7,
        command=["patch"],
    )

    assert payload["fp32_manual_vs_native"] == block
    assert payload["capture_dtype"] == "native"
    assert set(dtypes) == {"native"}


# ------------------------- R38 slice 2: patch.py against evaluate.py, the real writer


def _write_real_evaluation(path: Path, records, *, seed: int, label: str) -> dict:
    """An evaluation artifact written by ``evaluate.write_report`` -- the writer patch.py reads.

    ``_load_payload`` parses whatever ``run_evaluation`` left on disk, so the fixture has to be
    that file and not a dict shaped like someone's memory of it. ``write_report`` frames the
    payload, ``summarize`` fills the rates, ``evaluation_metadata`` fills the identity block,
    and ``Trajectory`` is the record class whose ``as_dict`` becomes each trajectory -- so
    every key ``_records``, ``_generator_version_binding`` and ``select_patch_cases`` reach for
    lands where and only where a real run puts it.

    ``records`` are ``(task_id, family, success, steps)``; the artifact is returned parsed, so
    a test can move a key in it and write it back.
    """
    from local_llm_lab.models import ResolvedSpec, load_model_spec
    from local_llm_lab.pipeline import evaluate
    from local_llm_lab.pipeline.runner import Trajectory

    resolved = ResolvedSpec(
        # ``ModelSpec.resolve`` needs loaded weights; this suite has none, so the record is
        # built directly over a real registry declaration. patch.py never reads these numbers.
        spec=load_model_spec("qwen35-4b"),
        num_layers=4,
        hidden_size=8,
        vocab_size=32,
        tie_word_embeddings=True,
        layer_types=("full_attention",) * 4,
        lora_keys=("self_attn.q_proj",),
        trainable_parameters=64,
        probe_layers=(1, 2, 3),
        cache_strategy="none",
        cache_strategy_reason="explicit:none",
        snapshot_revision=None,
        jvp_method="untested",
    )
    trajectories = [
        Trajectory(
            task_id=task_id,
            family=family,
            variant="clean",
            label=label,
            prompt="p",
            steps=list(steps),
            turns=len(steps),
            valid_turns=len(steps),
            difficulty=2,
            # ``evaluate_tasks`` stores an integrity block on every trajectory, so a real
            # artifact always carries the key and ``select_patch_cases`` always takes its
            # saved-judgement branch -- never the recomputing one the legacy files take.
            integrity=_real_integrity(success),
            verdict={
                "success": success,
                "clean": success,
                "errors": 0,
                "recovered_errors": 0,
                "reasons": [],
                "calls": len(steps),
                "schema_failures": 0,
                "executable_calls": len(steps),
            },
        )
        for task_id, family, success, steps in records
    ]
    summary = evaluate.summarize(trajectories)
    summary.update(
        evaluate.evaluation_metadata(
            label=label,
            resolved=resolved,
            adapter=None,
            split="test",
            difficulties=[2],
            stress=False,
            temperature=0.0,
            keep_last=2,
            seed=seed,
            use_cache=True,
            elapsed_seconds=0.0,
        )
    )
    evaluate.write_report(path, summary, trajectories)
    return json.loads(path.read_text(encoding="utf-8"))


_REAL_STEPS = ({"thought": "before"}, {"thought": "drop"})


def _real_integrity(clean: bool) -> dict:
    """The integrity block ``evaluate_tasks`` stores, built by the class that produces it.

    ``evaluate.py:153`` assigns ``check_trajectory(...).as_dict()``, so ``patch._saved_judgement``
    is reading ``IntegrityReport.as_dict()`` and not a shape invented here. Running the real
    checker would need a real task and trace; the report it returns is built directly instead.
    """
    from local_llm_lab.pipeline.integrity import IntegrityReport, Violation

    if clean:
        return IntegrityReport((), None, {}, True).as_dict()
    violation = Violation(1, "value_drop", "missing required values: 12")
    return IntegrityReport((violation,), violation, {"value_drop": 1}, False).as_dict()


def _real_pair(tmp_path: Path, *, seed: int) -> tuple[dict, dict]:
    """A passing and a failing artifact over one task id, both written by ``write_report``."""
    task_id = "test-aggregate_report-0-clean"
    passing = _write_real_evaluation(
        tmp_path / "passing.json",
        [(task_id, "aggregate_report", True, _REAL_STEPS)],
        seed=seed,
        label="passing",
    )
    failing = _write_real_evaluation(
        tmp_path / "failing.json",
        [(task_id, "aggregate_report", False, _REAL_STEPS)],
        seed=seed,
        label="failing",
    )
    return passing, failing


def test_records_finds_the_data_seed_where_the_writer_puts_it(tmp_path: Path) -> None:
    """R38 on ``patch.py:_records``: the recorded seed is inside ``summary``, not beside it.

    The reader looked at ``payload["data_seed"]`` alone. ``write_report`` frames the file as
    ``{"summary": ..., "trajectories": ...}`` and ``evaluation_metadata`` puts ``data_seed``
    in the summary, so the recorded seed was invisible to it: every real artifact took the
    "field is absent" branch, demanded ``--data-seed``, and -- the part that matters -- took
    the flag's value without ever comparing it to the one on disk. The R22a conflict check
    could not fire, because nothing ever reached it.

    The old hand-made ``_payload`` helper above builds ``{"data_seed": ..., ...}`` at the top
    level, which is the shape that certified the belief rather than the seam.
    """
    from local_llm_lab.probes import patch

    seed = 4242
    _passing, failing = _real_pair(tmp_path, seed=seed)

    assert failing["summary"]["data_seed"] == seed, "the writer records it inside the summary"
    assert "data_seed" not in failing, "and nowhere else"

    records, resolved_seed, source = patch._records(failing, "failing")
    assert (resolved_seed, source) == (seed, "evaluation")
    assert len(records) == 1

    # The conflict the reader claimed to catch and could not reach.
    conflict = rf"data_seed {seed} conflicts with --data-seed {seed + 1}"
    with pytest.raises(ValueError, match=conflict):
        patch._records(failing, "failing", data_seed=seed + 1)
    assert patch._records(failing, "failing", data_seed=seed)[1:] == (seed, "evaluation")


@pytest.mark.parametrize("level", ["summary", "top"])
def test_records_data_seed_goes_red_when_the_key_moves_off_its_level(
    tmp_path: Path, level: str
) -> None:
    """Move ``data_seed`` and the reader must change its answer, not keep the old one.

    Removed entirely: the flag is required again, and its value is reported as the flag's.
    Hoisted to the top level: still found, because a pre-``evaluation_metadata`` artifact put
    it there and this reader must keep reading those.
    """
    from local_llm_lab.probes import patch

    seed = 4242
    _passing, failing = _real_pair(tmp_path, seed=seed)
    moved = failing["summary"].pop("data_seed")
    if level == "top":
        failing["data_seed"] = moved
        assert patch._records(failing, "failing")[1:] == (seed, "evaluation")
        return
    with pytest.raises(ValueError, match="lacks data_seed"):
        patch._records(failing, "failing")
    assert patch._records(failing, "failing", data_seed=seed + 1)[1:] == (seed + 1, "flag")


def test_generator_version_binding_prefers_the_recorded_version_over_the_flag(
    tmp_path: Path,
) -> None:
    """A1 red-first: this reader's recorded branch had no writer feeding it until now.

    ``_generator_version_binding`` has always preferred the recorded version, refused a
    conflicting flag and returned ``(None, "")`` when neither existed. Nothing in this
    repository wrote the field, so only the last of the three ever ran and every P6 selection
    was version-bound by hand. With ``evaluation_metadata`` recording it, all three do.
    """
    from local_llm_lab.probes import patch

    _passing, failing = _real_pair(tmp_path, seed=4242)
    assert failing["summary"]["generator_version"] == patch.GENERATOR_VERSION

    # Preferred with no flag at all, and confirmed by a matching one.
    assert patch._generator_version_binding(failing, "failing", None) == (
        patch.GENERATOR_VERSION,
        "evaluation",
    )
    assert patch._generator_version_binding(failing, "failing", patch.GENERATOR_VERSION) == (
        patch.GENERATOR_VERSION,
        "evaluation",
    )

    # A flag that disagrees is refused by name, so a v4 artifact can no longer be replayed
    # under the v1 binding the Director's retry used on the legacy files.
    conflict = rf"generator_version {patch.GENERATOR_VERSION} conflicts with --generator-version 1"
    with pytest.raises(ValueError, match=conflict):
        patch._generator_version_binding(failing, "failing", 1)

    # Absent still fails closed: the eighteen saved evaluations record none.
    legacy = json.loads(json.dumps(failing))
    legacy["summary"].pop("generator_version")
    assert patch._generator_version_binding(legacy, "failing", None) == (None, "")
    assert patch._generator_version_binding(legacy, "failing", 1) == (1, "flag")


def test_generator_version_binding_reads_the_legacy_top_level_when_the_summary_loses_it(
    tmp_path: Path,
) -> None:
    """Moving the key one level up must still resolve; moving it out must stop resolving."""
    from local_llm_lab.probes import patch

    _passing, failing = _real_pair(tmp_path, seed=4242)
    hoisted = json.loads(json.dumps(failing))
    hoisted["generator_version"] = hoisted["summary"].pop("generator_version")
    assert patch._generator_version_binding(hoisted, "failing", None) == (
        patch.GENERATOR_VERSION,
        "evaluation",
    )
    hoisted.pop("generator_version")
    assert patch._generator_version_binding(hoisted, "failing", None) == (None, "")


def test_selection_records_the_inputs_generator_version_in_both_conditions(
    monkeypatch, tmp_path: Path
) -> None:
    """A1 in the patch artifact, on the primary AND the secondary path.

    ``recomputed_generator_version`` is written only inside the ``integrity_counts`` branch, so
    it reaches the artifact only when a case was recomputed -- never on the secondary path,
    which counts ``head_alone`` and recomputes nothing, and never on any run against an
    artifact this writer produced, because every trajectory it writes carries an ``integrity``
    key and takes the saved branch. The input's own declaration is therefore recorded
    unconditionally, and named for the input, because the secondary section declares
    ``HEAD_ALONE_BASIS``: a bound version number under that basis would contradict it on the
    face of the artifact.
    """
    from local_llm_lab.probes import patch

    task = _Task("test-aggregate_report-0-clean", "aggregate_report")
    monkeypatch.setattr(patch, "task_from_id", lambda task_id, seed, difficulty: task)
    monkeypatch.setattr(
        patch, "replay_task_from_id", lambda *_args, **_kwargs: pytest.fail("no replay expected")
    )
    monkeypatch.setattr(
        patch, "check_trajectory", lambda *_args, **_kwargs: _drop_report(1, "12")
    )
    passing, failing = _real_pair(tmp_path, seed=4242)

    primary, primary_provenance = patch.select_patch_cases(passing, failing, keep_last=2)
    secondary, secondary_provenance = patch.select_patch_cases(
        passing, failing, keep_last=2, secondary_condition="aggregate_report"
    )

    assert [case.task.task_id for case in primary] == [task.task_id]
    assert [case.task.task_id for case in secondary] == [task.task_id]
    for provenance, basis in (
        (primary_provenance, patch.BOUND_VS_HEAD_BASIS),
        (secondary_provenance, patch.HEAD_ALONE_BASIS),
    ):
        block = provenance["eligibility"]["failing"]
        assert block["eligibility_basis"] == basis
        assert block["evaluation_generator_version"] == patch.GENERATOR_VERSION
        assert block["evaluation_generator_version_source"] == "evaluation"
        # No recomputation happened on either path, so no bound version is claimed. Under
        # HEAD_ALONE_BASIS one would be a contradiction; under BOUND_VS_HEAD it would be a
        # replay that never ran.
        assert "recomputed_generator_version" not in block
    # The primary path judged from the artifact's own integrity block and the secondary under
    # HEAD alone. Neither recomputed, which is why the conditional block writes nothing on
    # either -- against an artifact this writer produced, it never will.
    assert primary_provenance["eligibility"]["failing"]["integrity"] == {
        "evaluation": 1,
        "recomputed": 0,
        "head_alone": 0,
    }
    assert secondary_provenance["eligibility"]["failing"]["integrity"] == {
        "evaluation": 0,
        "recomputed": 0,
        "head_alone": 1,
    }


def test_written_patch_artifact_carries_the_recorded_version_in_both_sections(
    monkeypatch, tmp_path: Path
) -> None:
    """The Chief's question, answered on the file rather than on the provenance dict.

    ``main`` copies ``selection_provenance["eligibility"]`` into ``payload["eligibility"]``
    and ``secondary_provenance["eligibility"]`` into ``payload["secondary"]["eligibility"]``,
    so the field reaches both halves of ``patch.json``. The secondary block is assembled
    before ``run_patch_probe`` is called for it, which is what makes the record survive a
    secondary condition that refuses to score.

    Only ``run_patch_probe`` is stubbed here -- it needs a loaded policy. ``select_patch_cases``
    is the real one, reading the real artifact, so what is asserted is the field's route to
    disk and not a fixture's echo of it.
    """
    from local_llm_lab.probes import guard, patch

    task = _Task("test-aggregate_report-0-clean", "aggregate_report")
    monkeypatch.setattr(patch, "task_from_id", lambda task_id, seed, difficulty: task)
    monkeypatch.setattr(
        patch, "check_trajectory", lambda *_args, **_kwargs: _drop_report(1, "12")
    )
    passing_path, failing_path = tmp_path / "passing.json", tmp_path / "failing.json"
    _real_pair(tmp_path, seed=4242)
    assert passing_path.is_file() and failing_path.is_file()

    spec = SimpleNamespace(
        name="qwen35-4b",
        hf_id="fake/hf",
        policies={},
        probe_layer_fractions=(0.25,),
        resolve=lambda *_args: SimpleNamespace(num_layers=4, probe_layers=(1,)),
    )
    monkeypatch.setattr(patch, "load_model_spec", lambda _name: spec)
    monkeypatch.setattr(patch, "require_preflight", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(guard, "require_idle_gpu", lambda *_args: None)
    monkeypatch.setattr(patch, "resolve_policy", lambda _name, _spec: None)
    monkeypatch.setattr(
        patch,
        "load_policy",
        lambda _spec, _adapter: (
            object(),
            object(),
            SimpleNamespace(num_layers=4),
            SimpleNamespace(as_dict=dict),
        ),
    )
    monkeypatch.setattr(
        patch,
        "run_patch_probe",
        lambda *_args, **_kwargs: {
            "cells": {},
            "groups": [],
            "layers": [],
            "controls": [],
            "condition": "x",
            "scoring_generator_version": patch.GENERATOR_VERSION,
        },
    )
    output = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "agent-v2-probe-patch",
            "--passing-eval", str(passing_path),
            "--failing-eval", str(failing_path),
            "--output", str(output),
            "--secondary-condition", "aggregate_report",
        ],
    )

    patch.main()

    written = json.loads((output / "patch.json").read_text(encoding="utf-8"))
    for block, basis in (
        (written["eligibility"]["failing"], patch.BOUND_VS_HEAD_BASIS),
        (written["secondary"]["eligibility"]["failing"], patch.HEAD_ALONE_BASIS),
    ):
        assert block["evaluation_generator_version"] == patch.GENERATOR_VERSION
        assert block["evaluation_generator_version_source"] == "evaluation"
        assert block["eligibility_basis"] == basis
        # A HEAD-alone section must not carry a bound version, and does not: nothing was
        # replayed under one on either path.
        assert "recomputed_generator_version" not in block
    # HEAD is what the flip scoring ran under, recorded in both sections by run_patch_probe.
    assert written["scoring_generator_version"] == patch.GENERATOR_VERSION
    assert written["secondary"]["scoring_generator_version"] == patch.GENERATOR_VERSION


# --- R30a (SPEC-004 §5a): locating note values in a family whose values repeat -------------

_GENERATOR_SEED = 20260902
"""The data seed the saved runs were generated under; the fixtures below name the same
tasks the aborted secondary condition selected, so they are that run's own notes."""


def _generator_note(step_index: int, *, generator_version: int | None = None) -> str:
    """One real ``aggregate_report`` note, rendered by the generator that writes them.

    R38 applies here literally: a hand-written ambiguous note is the author writing down
    what they believe the generator produces, and this scorer broke on the difference. Task
    ``0011`` is the first ``aggregate_report`` of the test split, and the run that scored
    nothing selected it; its metric list repeats ``18``, so its own notes carry the
    repetition without a fixture inventing any. ``generator_version`` names the note
    template: HEAD's single ``values so far`` list when omitted, and the two-field
    ``first half``/``second half`` shape of version 1, which is the shape run C writes.
    """
    from local_llm_lab.pipeline.tasks import make_tasks, render_expert_note, replay_task_from_id

    task_id = "test-aggregate_report-0011-clean"
    if generator_version is None:
        task = next(
            task for task in make_tasks("test", 720, _GENERATOR_SEED) if task.task_id == task_id
        )
    else:
        task = replay_task_from_id(task_id, _GENERATOR_SEED, generator_version)
    return render_expert_note(task, step_index)


def _locate_note_values(note: str):
    """Run ``_groups_for`` over a one-note prompt and return (prompt, groups, value spans).

    The character tokenizer makes a token position a character position, so a located span
    can be compared directly with a character span measured on the note itself.
    """
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
        {"role": "user", "content": "TASK-aggregate_report"},
        assistant_message("Plan: inspect the state.", Action("list_files", {"directory": "/m"})),
        tool_message("list_files", "OBS0"),
        assistant_message(note, Action("calculate", {"expression": "1 + 1"})),
        tool_message("calculate", "OBS1"),
    ]
    prompt = "|".join(message["content"] for message in messages)
    groups, _repairs, spans = patch._groups_for(
        tokenizer, tokenizer.encode(prompt), messages, prompt_text=prompt, keep_last=2
    )
    return prompt, groups, spans


def test_a_repeated_note_value_is_located_by_its_field_occurrence() -> None:
    """R30a(1): the k-th number under a label is that label's k-th value.

    The generator's own step-7 note lists ``18`` twice and then names ``18``, ``14`` and
    ``75`` again in the subtotal it is about to compute. Searching the whole note for a
    unique match refuses every one of those; searching inside the field that carries them
    gives each a position by construction.
    """
    from local_llm_lab.probes import patch

    note = _generator_note(7)
    assert note.count("18") > 1, "the fixture must be the generator's repeating note"
    prompt, groups, _spans = _locate_note_values(note)

    values = patch._note_values(note)
    located = "".join(chr(ord(prompt[position])) for position in groups["note_value_tokens"])
    # Every value keeps its own tokens, repeats included, in note order.
    assert located == "".join(values)
    assert len(groups["note_value_tokens"]) == sum(len(value) for value in values)
    # The subtotal arithmetic after the list is never a match candidate, and neither is the
    # ``split after 3 of 6`` clause, whose numbers are not values at all.
    note_start = prompt.index(note)
    tail_start = note_start + note.index(";")
    assert max(groups["note_value_tokens"]) < tail_start


def test_moving_a_value_between_fields_moves_the_span_it_is_located_at() -> None:
    """R30a(1) is a claim about fields, so the test moves a value from one to the other.

    Version 1's note -- the shape run C writes -- carries two fields and ``18`` in both.
    Moving the second ``18`` into the first field changes which label owns it and changes
    the note's value order; a locator reading each value inside its own field follows it,
    and one holding any earlier notion of where the fields are does not.
    """
    from local_llm_lab.probes import patch

    note = _generator_note(7, generator_version=1)
    moved = note.replace(
        "first half complete: 18 + 14 + 75; second half complete: 72 + 18 + 79",
        "first half complete: 18 + 14 + 75 + 18; second half complete: 72 + 79",
    )
    assert moved != note, "the fixture note must carry the two fields the move needs"
    # The move is the test: the value order itself changes, so an expectation copied from
    # the unmoved note cannot pass.
    assert patch._note_values(note) == ["18", "14", "75", "72", "18", "79"]
    assert patch._note_values(moved) == ["18", "14", "75", "18", "72", "79"]

    for text, owner in ((note, "second half complete"), (moved, "first half complete")):
        prompt, groups, _spans = _locate_note_values(text)
        fields = {
            " ".join(match.group(1).split()).casefold(): (match.start(2), match.end(2))
            for match in patch._VALUE_FIELD.finditer(text)
        }
        start, end = fields[owner]
        offset = prompt.index(text)
        # The repeated 18 is the note's fourth or fifth value depending on the move; take it
        # by position in the located run rather than by name, since the name is ambiguous.
        index = patch._note_values(text).index("18", 1)
        positions = groups["note_value_tokens"]
        before = sum(len(value) for value in patch._note_values(text)[:index])
        span = positions[before : before + len("18")]
        assert all(offset + start <= position < offset + end for position in span)


def _residue_note() -> str:
    """A real generator note the field-scoped locator still refuses.

    R38 again: the residue is measured, not imagined, and it was re-measured when the class it
    used to belong to disappeared. This note's ``76`` appears twice -- once under ``loads so
    far`` and again under ``highest so far`` -- and sits under no *list* field, so the
    field-scoped locator cannot give it a position and falls back to the unique search, which
    refuses a value matching twice. It is the residue R30a(2) turns into a skipped case rather
    than an abandoned section.

    The earlier fixture here was a ``ledger_reconcile`` note whose only value ended a clause
    before a full stop. That refused for too FEW matches, not too many, and widening
    ``_value_pattern``'s trailing lookahead removed the whole class -- which is why this
    fixture had to move. A residue of repetition survives that change; a residue of
    clause-final punctuation does not.
    """
    from local_llm_lab.pipeline.tasks import render_expert_note, replay_task_from_id

    task = replay_task_from_id("test-conditional_update-0009-clean", _GENERATOR_SEED, 4)
    return render_expert_note(task, 3)


def _locator_refusal() -> ValueError:
    """The refusal the real locator raises on the real residue note, not a copy of its text."""
    from local_llm_lab.probes import patch

    with pytest.raises(ValueError) as raised:
        _locate_note_values(_residue_note())
    assert isinstance(raised.value, ValueError)
    del patch
    return raised.value


def _skipping_fixture(monkeypatch, cases, skipped_task_id):
    """The shared probe fixture, with each case's prompts named after its own task.

    ``_probe_fixture`` gives every case the same two prompts, so nothing downstream can tell
    them apart; naming them lets exactly one case's locator refuse.
    """
    from local_llm_lab.probes import patch

    # Taken before the fixture runs: ``_probe_fixture`` replaces ``_groups_for`` with a fake
    # that cannot refuse anything, so the real locator has to be asked first.
    refusal = _locator_refusal()
    _tokenizer, captures = _probe_fixture(monkeypatch, cases)

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return list(range(10)) if text.startswith("f") else list(range(1, 11))

    monkeypatch.setattr(
        patch,
        "replay_counterfactual",
        lambda case: (
            [{"role": "user", "content": f"f{case.task.task_id}"}],
            [{"role": "user", "content": f"c{case.task.task_id}"}],
            {"counterfactual_source": "passing_transcript", "counterfactual_basis": "fixture"},
        ),
    )

    def groups_for(_tokenizer, ids, messages, **_kwargs):
        if messages[0]["content"].endswith(skipped_task_id):
            raise refusal
        return (
            {name: (0,) for name in patch.PROMPT_GROUPS},
            {name: 0 for name in patch.PROMPT_GROUPS},
            {"7": (0,)} if ids[0] == 0 else {"7": (0,), "9": (1,)},
        )

    monkeypatch.setattr(patch, "_groups_for", groups_for)
    return Tokenizer(), captures, refusal


def test_an_unlocatable_case_is_skipped_with_its_reason_and_the_section_scores_the_rest(
    monkeypatch,
) -> None:
    """R30a(2): one unlocatable case cost all fifteen; it now costs itself.

    The refusal is the real locator's, raised on a real note, so what is asserted is the
    control flow around it and not a fixture's imitation of the failure.
    """
    from local_llm_lab.probes import patch

    cases = [_probe_case(index, judgements=_judgements(1)) for index in range(3)]
    skipped = cases[1].task.task_id
    tokenizer, captures, refusal = _skipping_fixture(monkeypatch, cases, skipped)
    resolved = SimpleNamespace(as_dict=lambda: {"name": "fake"})

    payload = patch.run_patch_probe(
        object(), tokenizer, cases, spec=object(), resolved=resolved, layers=[1], policy="base",
        keep_last=2, max_tokens=1, seed=7, command=["patch"],
    )

    # Two cases prepared and captured, one skipped before any residual was taken.
    assert len(captures) == 4
    assert (payload["scored_cases"], payload["skipped_cases"]) == (2, 1)
    assert payload["selected_cases"] == len(cases)
    assert payload["skipped_reasons"] == {"note_values_unlocatable": 1}
    assert payload["headline_task_ids"] == [cases[0].task.task_id, cases[2].task.task_id]
    assert payload["headline_cases"] == 2
    # The skip is recorded in the primary's own container, with the locator's own words.
    skips = [
        record for record in payload["excluded_cases"]
        if record["excluded_reason"] == "note_values_unlocatable"
    ]
    assert [record["task_id"] for record in skips] == [skipped]
    assert skips[0]["excluded_detail"] == str(refusal)
    assert "missing or ambiguous" in skips[0]["excluded_detail"]


def test_a_section_refuses_below_its_minimum_scored_count_naming_both_numbers(
    monkeypatch,
) -> None:
    """R30a(2)/(3): the section refuses only when too few placeable cases remain."""
    from local_llm_lab.probes import patch

    cases = [_probe_case(index, judgements=_judgements(1)) for index in range(3)]
    tokenizer, _captures, _refusal = _skipping_fixture(
        monkeypatch, cases, cases[1].task.task_id
    )
    resolved = SimpleNamespace(as_dict=lambda: {"name": "fake"})

    with pytest.raises(ValueError, match=r"2 of 3 selected"):
        patch.run_patch_probe(
            object(), tokenizer, cases, spec=object(), resolved=resolved, layers=[1],
            policy="base", keep_last=2, max_tokens=1, seed=7, command=["patch"],
            minimum_scored=3,
        )


def test_the_markdown_reports_the_scored_and_skipped_counts_out_of_the_selection(
    monkeypatch,
) -> None:
    """R30a(2): a reader must see what the rates were taken over without opening the JSON."""
    from local_llm_lab.probes import patch

    cases = [_probe_case(index, judgements=_judgements(1)) for index in range(3)]
    tokenizer, _captures, _refusal = _skipping_fixture(
        monkeypatch, cases, cases[1].task.task_id
    )
    resolved = SimpleNamespace(as_dict=lambda: {"name": "fake"})

    payload = patch.run_patch_probe(
        object(), tokenizer, cases, spec=object(), resolved=resolved, layers=[1], policy="base",
        keep_last=2, max_tokens=1, seed=7, command=["patch"],
    )

    markdown = patch.render_markdown(payload)
    assert "2 of 3 selected case(s) scored; 1 skipped" in markdown
    assert "note_values_unlocatable" in markdown


def test_an_unscored_secondary_section_does_not_end_the_run_ok(monkeypatch, tmp_path) -> None:
    """R30a(4): the ``error`` field was honest and the end line was not.

    The 2026-09-05 run wrote its refusal into the artifact and then closed ``status=ok``,
    because nothing raised — the record reported that the code finished, not that the
    measurement happened. The section's state now reaches the end event, the health block,
    the top of the markdown and the exit status, so no reader has to open the JSON to learn
    that a scheduled section scored nothing.
    """
    from local_llm_lab.probes import guard, patch

    passing, failing = tmp_path / "passing.json", tmp_path / "failing.json"
    passing.write_text("{}", encoding="utf-8")
    failing.write_text("{}", encoding="utf-8")
    spec = SimpleNamespace(
        name="qwen35-4b",
        hf_id="fake/hf",
        policies={},
        probe_layer_fractions=(0.25,),
        resolve=lambda *_args: SimpleNamespace(num_layers=4, probe_layers=(1,)),
    )
    primary_case = patch.PatchCase(
        _Task("test-ledger_reconcile-1-clean", "ledger_reconcile"), 0, (), None, *_judgements(0)
    )
    secondary_case = patch.PatchCase(
        _Task("test-aggregate_report-0-clean", "aggregate_report"),
        0,
        (),
        None,
        *_judgements(0),
        "aggregate_report_secondary",
    )

    def select(*_args, **kwargs):
        condition = kwargs.get("secondary_condition")
        provenance = {
            "data_seeds": {},
            "eligibility": {},
            "condition": "aggregate_report_secondary" if condition else patch.PRIMARY_CONDITION,
            "counterfactual_label": "designed_correct" if condition else "empirically_passing",
        }
        return ([secondary_case] if condition else [primary_case]), provenance

    minimums = []

    def run(_model, _tokenizer, cases, **kwargs):
        minimums.append(kwargs.get("minimum_scored"))
        if cases[0].task.family == "aggregate_report":
            raise ValueError("P6 section scored 1 of 15 selected case(s), below the minimum of 5")
        return {"cells": {}, "groups": [], "layers": [], "controls": [], "condition": "x"}

    monkeypatch.setattr(patch, "select_patch_cases", select)
    monkeypatch.setattr(patch, "load_model_spec", lambda _name: spec)
    monkeypatch.setattr(patch, "require_preflight", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(guard, "require_idle_gpu", lambda *_args: None)
    monkeypatch.setattr(patch, "resolve_policy", lambda _name, _spec: None)
    monkeypatch.setattr(
        patch,
        "load_policy",
        lambda _spec, _adapter: (
            object(),
            object(),
            SimpleNamespace(num_layers=4),
            SimpleNamespace(as_dict=dict),
        ),
    )
    monkeypatch.setattr(patch, "run_patch_probe", run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "agent-v2-probe-patch",
            "--passing-eval", str(passing),
            "--failing-eval", str(failing),
            "--output", str(tmp_path),
            "--secondary-condition", "aggregate_report",
        ],
    )

    with pytest.raises(SystemExit, match="secondary"):
        patch.main()

    # The floor the ruling names reaches the section that was ruled on.
    assert minimums == [None, patch.MINIMUM_SCORED_CASES]
    # The artifact is still written: an unscored section is a finding, not a lost run.
    payload = json.loads((tmp_path / "patch.json").read_text(encoding="utf-8"))
    assert payload["secondary"]["error"].startswith("P6 section scored 1 of 15")
    assert payload["health"] == {
        "verdict": "unscored_section",
        "status": "error",
        "unscored_sections": ["secondary"],
        "section_errors": {"secondary": payload["secondary"]["error"]},
    }
    # The end event carries the same state as the health block, not "ok".
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    end = next(event for event in events if event["kind"] == "end")
    assert end["status"] == "error"
    assert end["fields"]["unscored_sections"] == ["secondary"]
    # And the markdown says it before anything a reader could mistake for a result.
    markdown = (tmp_path / "patch.md").read_text(encoding="utf-8").splitlines()
    assert "secondary" in markdown[2]
    assert markdown[2].startswith("**Unscored")


def test_a_fully_scored_run_still_ends_ok_and_carries_a_healthy_block(
    monkeypatch, tmp_path
) -> None:
    """The complement: R30a(4) must not turn every run into an error."""
    from local_llm_lab.probes import guard, patch

    passing, failing = tmp_path / "passing.json", tmp_path / "failing.json"
    passing.write_text("{}", encoding="utf-8")
    failing.write_text("{}", encoding="utf-8")
    spec = SimpleNamespace(
        name="qwen35-4b",
        hf_id="fake/hf",
        policies={},
        probe_layer_fractions=(0.25,),
        resolve=lambda *_args: SimpleNamespace(num_layers=4, probe_layers=(1,)),
    )
    case = patch.PatchCase(
        _Task("test-ledger_reconcile-1-clean", "ledger_reconcile"), 0, (), None, *_judgements(0)
    )
    monkeypatch.setattr(
        patch,
        "select_patch_cases",
        lambda *_args, **_kwargs: ([case], {"data_seeds": {}, "eligibility": {}}),
    )
    monkeypatch.setattr(patch, "load_model_spec", lambda _name: spec)
    monkeypatch.setattr(patch, "require_preflight", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(guard, "require_idle_gpu", lambda *_args: None)
    monkeypatch.setattr(patch, "resolve_policy", lambda _name, _spec: None)
    monkeypatch.setattr(
        patch,
        "load_policy",
        lambda _spec, _adapter: (
            object(),
            object(),
            SimpleNamespace(num_layers=4),
            SimpleNamespace(as_dict=dict),
        ),
    )
    monkeypatch.setattr(
        patch,
        "run_patch_probe",
        lambda *_args, **_kwargs: {
            "cells": {}, "groups": [], "layers": [], "controls": [], "condition": "x"
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "agent-v2-probe-patch",
            "--passing-eval", str(passing),
            "--failing-eval", str(failing),
            "--output", str(tmp_path),
        ],
    )

    patch.main()

    payload = json.loads((tmp_path / "patch.json").read_text(encoding="utf-8"))
    assert payload["health"] == {
        "verdict": "scored",
        "status": "ok",
        "unscored_sections": [],
        "section_errors": {},
    }
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    end = next(event for event in events if event["kind"] == "end")
    assert end["status"] == "ok"
    assert end["fields"]["unscored_sections"] == []
    assert not (tmp_path / "patch.md").read_text(encoding="utf-8").startswith("**Unscored")


def test_a_value_ending_a_clause_is_located_rather_than_refused_for_too_few_hits() -> None:
    """The trailing lookahead rejected a full stop of any kind, so a clause-final value matched
    nowhere and its note was refused for having too FEW occurrences rather than too many.

    R38: the note comes from the generator, not from this file. ``replay_task_from_id`` step 8 of
    ``test-ledger_reconcile-0007-clean`` under generator version 1 renders
    ``Approved total = 359. Inspecting the summary before editing.`` -- a real note, found by
    scanning the split rather than constructed to make the point.

    The decimal exclusion the old form was written for is kept: ``72`` must still not match inside
    ``72.5``. Splitting the lookahead separates the two cases the single character class conflated.
    """
    import re

    from local_llm_lab.pipeline.tasks import render_expert_note, replay_task_from_id
    from local_llm_lab.probes.patch import _note_values, _value_pattern

    note = render_expert_note(
        replay_task_from_id("test-ledger_reconcile-0007-clean", 20260902, 1), 8
    )
    assert note.rstrip().startswith("Approved total = 359."), "the generator still renders this note"

    values = _note_values(note)
    assert values, "the note carries a value to locate"
    for value in values:
        assert len(list(re.finditer(_value_pattern(value), note))) == 1

    # The old form, kept here as the thing being changed rather than described: it found nothing.
    superseded = rf"(?<![0-9.\-]){re.escape(values[0])}(?![0-9.])"
    assert not list(re.finditer(superseded, note))

    # And the case the exclusion existed for is unaffected.
    assert not list(re.finditer(_value_pattern("72"), "the rate was 72.5 today"))
    assert len(list(re.finditer(_value_pattern("32"), "values 132 and 32 differ"))) == 1
