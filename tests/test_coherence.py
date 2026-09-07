"""The coherence standard (UNIFIED-RUN-2026-09-06 section 2 revised; rulings of 10:12).

These tests assert the recorded rulings rather than derive a rule (R38(f)): the four causes as
competing risks, per-error recovery within a three-step window by the same tool with any
argument (the 10:12 ruling as corrected at 13:30: the generator's experts recover by re-calling
the failed tool with the corrected argument, and a rule that does not recognise the expert
demonstration is wrong by construction), the unknown_tool clause, symmetric tail censoring for
the tool-error cause, no pooled length as a statistic, and the Kaplan-Meier median as a
descriptive marked "not reached". The generator-based test runs the expert step sequences of the
recovery variants through the simulator and asserts no tool-error event.
"""
from __future__ import annotations

from local_llm_lab.pipeline.coherence import (
    CAUSES,
    coherence_summary,
    loop_step,
    primary_arguments,
    trajectory_events,
)
from local_llm_lab.pipeline.runner import Trajectory


def _step(index: int, name: str, arguments: dict, observation: str) -> dict:
    return {"index": index, "action": {"name": name, "arguments": arguments}, "observation": observation}


def _trajectory(steps: list[dict], *, success: bool, integrity: dict | None = None, horizon: int = 6, **extra) -> Trajectory:
    trajectory = Trajectory(task_id="t", family="read", variant="clean", label="x", prompt="p", steps=steps)
    trajectory.verdict = {"success": success}
    trajectory.integrity = integrity or {"first_violation": None, "counts": {}, "clean": True}
    trajectory.horizon = horizon
    for key, value in extra.items():
        setattr(trajectory, key, value)
    return trajectory


def test_primary_arguments_come_from_the_schema() -> None:
    primaries = primary_arguments()
    assert primaries["read_file"] == "path"
    assert primaries["replace_text"] == "path"  # first required parameter, named in the docstring
    assert primaries["finish"] == "answer"


def test_a_clean_success_is_coherent_to_completion_with_no_events() -> None:
    steps = [_step(1, "read_file", {"path": "a"}, "contents"), _step(2, "finish", {"answer": "x"}, "done")]
    record = trajectory_events(_trajectory(steps, success=True))
    assert record["coherent_to_completion"] is True
    assert record["first_cause"] is None
    assert all(record["events"][cause] is None for cause in CAUSES)


def test_the_integrity_violation_carries_its_step() -> None:
    steps = [_step(i, "read_file", {"path": f"f{i}"}, "contents") for i in range(1, 5)]
    integrity = {"first_violation": {"step": 3, "kind": "verbatim_copy", "detail": ""}, "counts": {"verbatim_copy": 1}, "clean": False}
    record = trajectory_events(_trajectory(steps, success=True, integrity=integrity))
    assert record["events"]["integrity"] == 3
    assert record["first_cause"] == "integrity"
    assert record["coherent_to_completion"] is False  # success with an event is not coherent to completion


def test_a_tool_error_recovered_within_the_window_by_the_same_tool_is_not_an_event() -> None:
    steps = [
        _step(1, "read_file", {"path": "a"}, "ERROR: no such file: a"),
        _step(2, "read_file", {"path": "b"}, "contents"),  # a correction: same tool, different path
        _step(3, "read_file", {"path": "c"}, "contents"),
        _step(4, "read_file", {"path": "d"}, "contents"),
        _step(5, "read_file", {"path": "e"}, "contents"),
        _step(6, "finish", {"answer": "x"}, "done"),
    ]
    assert trajectory_events(_trajectory(steps, success=True))["events"]["tool_error"] is None


def test_a_tool_error_answered_only_by_other_tools_is_an_event_at_its_own_step() -> None:
    steps = [
        _step(1, "read_file", {"path": "a"}, "ERROR: no such file: a"),
        _step(2, "search_files", {"query": "b"}, "hits"),
        _step(3, "list_files", {"directory": "c"}, "files"),
        _step(4, "calculate", {"expression": "1+1"}, "2"),
        _step(5, "read_file", {"path": "e"}, "contents"),  # the same tool, but beyond the window
        _step(6, "read_file", {"path": "f"}, "contents"),
        _step(7, "read_file", {"path": "g"}, "contents"),
    ]
    record = trajectory_events(_trajectory(steps, success=False))
    assert record["events"]["tool_error"] == 1
    assert record["first_cause"] == "tool_error"


def test_the_window_counts_executed_steps_not_indices() -> None:
    steps = [
        _step(1, "read_file", {"path": "a"}, "ERROR: no such file: a"),
        {"index": 2, "parse_error": "no fenced block"},  # not executed: does not consume the window
        _step(3, "search_files", {"query": "q"}, "hits"),
        _step(4, "list_files", {"directory": "d"}, "files"),
        _step(5, "read_file", {"path": "b"}, "contents"),  # the third executed step after the error
        _step(6, "read_file", {"path": "c"}, "contents"),
        _step(7, "read_file", {"path": "d"}, "contents"),
        _step(8, "read_file", {"path": "e"}, "contents"),
    ]
    assert trajectory_events(_trajectory(steps, success=False))["events"]["tool_error"] is None


def test_the_tail_is_censored_for_the_tool_error_cause_whatever_the_outcome() -> None:
    steps = [
        _step(1, "read_file", {"path": "a"}, "contents"),
        _step(2, "read_file", {"path": "b"}, "contents"),
        _step(3, "read_file", {"path": "c"}, "ERROR: no such file: c"),  # fewer than three executed steps follow
        _step(4, "finish", {"answer": "x"}, "done"),
    ]
    for success in (False, True):
        record = trajectory_events(_trajectory(steps, success=success))
        assert record["events"]["tool_error"] is None
        assert record["tool_error_censored_tail"] is True
        assert record["coherent_to_completion"] is success


def test_an_unknown_tool_is_recovered_by_any_later_successful_call_within_the_window() -> None:
    steps = [
        _step(1, "open_file", {"file": "a"}, "ERROR: invalid call: unknown tool: open_file; available tools are read_file"),
        _step(2, "read_file", {"path": "a"}, "contents"),
        _step(3, "finish", {"answer": "x"}, "done"),
    ]
    assert trajectory_events(_trajectory(steps, success=True))["events"]["tool_error"] is None
    steps = [
        _step(1, "open_file", {"file": "a"}, "ERROR: invalid call: unknown tool: open_file; available tools are read_file"),
        _step(2, "open_file", {"file": "a"}, "ERROR: invalid call: unknown tool: open_file; available tools are read_file"),
        _step(3, "open_file", {"file": "a"}, "ERROR: invalid call: unknown tool: open_file; available tools are read_file"),
        _step(4, "open_file", {"file": "a"}, "ERROR: invalid call: unknown tool: open_file; available tools are read_file"),
        _step(5, "read_file", {"path": "a"}, "contents"),
        _step(6, "read_file", {"path": "b"}, "contents"),
        _step(7, "read_file", {"path": "c"}, "contents"),
        _step(8, "read_file", {"path": "d"}, "contents"),
    ]
    record = trajectory_events(_trajectory(steps, success=False))
    assert record["events"]["tool_error"] == 1  # never recovered within three executed steps: an event at its own step


def test_the_generator_experts_of_the_recovery_variants_carry_no_tool_error_event() -> None:
    """The expert demonstration is the definition of recovery in this programme (the Research
    Division, 13:30): a rule that scores the expert's own correction as an unrecovered error is
    wrong by construction. The expert actions are executed through the simulator so the
    observations are the real ones."""
    from local_llm_lab.pipeline.env import Simulator
    from local_llm_lab.pipeline.tasks import make_tasks

    tasks = make_tasks("train", 120, 20260902, perturb=True, difficulty=0)
    seen: dict[str, int] = {}
    for task in tasks:
        if task.variant not in ("wrong_path", "stale_path", "transient", "unknown_tool", "failed_edit"):
            continue
        if seen.get(task.variant, 0) >= 2:
            continue
        seen[task.variant] = seen.get(task.variant, 0) + 1
        simulator = Simulator.for_task(task)
        steps = [
            {"index": i, "action": {"name": step.action.name, "arguments": dict(step.action.arguments)}, "observation": simulator.execute(step.action)}
            for i, step in enumerate(task.steps, 1)
        ]
        assert any(str(step["observation"]).startswith("ERROR") for step in steps), (task.variant, task.task_id)
        record = trajectory_events(_trajectory(steps, success=True, horizon=task.horizon))
        assert record["events"]["tool_error"] is None, (task.variant, task.task_id, steps)
        assert record["events"]["invalid_action"] is None, (task.variant, task.task_id)
    assert set(seen) == {"wrong_path", "stale_path", "transient", "unknown_tool", "failed_edit"}, seen


def test_a_refused_call_other_than_an_unknown_tool_is_an_invalid_action() -> None:
    steps = [
        _step(1, "read_file", {}, "ERROR: invalid call: missing required argument(s): path; expected: path"),
        _step(2, "finish", {"answer": "x"}, "done"),
    ]
    record = trajectory_events(_trajectory(steps, success=True))
    assert record["events"]["invalid_action"] == 1
    assert record["events"]["tool_error"] is None


def test_a_parse_error_step_is_an_invalid_action() -> None:
    steps = [_step(1, "read_file", {"path": "a"}, "contents"), {"index": 2, "parse_error": "no fenced block"}]
    assert trajectory_events(_trajectory(steps, success=False))["events"]["invalid_action"] == 2


def test_the_loop_step_is_recomputed_from_the_stored_steps() -> None:
    steps = [_step(i, "read_file", {"path": "same"}, "contents") for i in range(1, 6)]  # three identical calls close at step 3
    record = trajectory_events(_trajectory(steps, success=False))
    assert record["events"]["loop"] == 3
    assert record["first_cause"] == "loop"


def test_the_summary_reports_causes_apart_and_no_pooled_length_statistic() -> None:
    clean = _trajectory([_step(1, "read_file", {"path": "a"}, "contents"), _step(2, "finish", {"answer": "x"}, "done")], success=True)
    looped = _trajectory([_step(i, "read_file", {"path": "same"}, "contents") for i in range(1, 5)], success=False, family="search")
    errored = _trajectory([_step(1, "read_file", {"path": "a"}, "ERROR: no such file: a")] + [_step(i, "search_files", {"query": f"q{i}"}, "hits") for i in range(2, 8)], success=False, family="search")  # never re-calls the failed tool
    summary = coherence_summary([clean, looped, errored], max_steps=8)
    overall = summary["overall"]
    assert overall["tasks"] == 3 and overall["coherent_to_completion"] == 1
    assert overall["events_by_cause"] == {"integrity": 0, "loop": 1, "invalid_action": 0, "tool_error": 1}
    assert abs(sum(overall["cumulative_incidence_by_cause"].values()) - 2 / 3) < 1e-3  # incidences are rounded to four places
    assert "wilson_95" in overall and "coherent_to_completion" in overall["wilson_95"]
    assert "median_coherence_length" not in overall  # no pooled length as a statistic
    assert overall["km_median_step_descriptive"] in ("not reached", 1, 2, 3, 4, 5, 6, 7, 8)
    # at step 3 only the looped run is still at risk: the clean run finished at step 2 and the
    # errored run's first event was at step 1, so the loop hazard there is 1 of 1
    assert overall["at_risk_per_step"][2] == 1
    assert overall["hazard_by_cause_per_step"]["loop"][2] == 1.0
    assert overall["hazard_by_cause_per_step"]["tool_error"][0] == round(1 / 3, 4)  # step 1: one of three at risk
    assert set(summary["by_family"]) == {"read", "search"}
    assert "km_median_step_descriptive" in summary["not_comparable_across_cells"]
    assert "hazard_by_cause_per_step" in summary["comparable_across_cells"]


def test_the_first_event_is_scaled_by_the_task_horizon() -> None:
    steps = [_step(i, "read_file", {"path": "same"}, "contents") for i in range(1, 5)]
    record = trajectory_events(_trajectory(steps, success=False, horizon=6))
    assert record["first_step"] == 3 and record["first_step_over_horizon"] == 0.5


def test_a_step_without_an_index_key_is_indexed_by_position_and_agrees_with_the_key() -> None:
    """Hand-built steps carry no ``"index"``; the seven ``test_patch`` failures were that.

    The fixture builds the same trajectory twice, once with the key and once without, because a
    scheme that fell back to position but disagreed with the key would pass a test that only
    checked the keyless case. Both must give the same answer, since ``first_cause`` is a minimum
    across causes and the integrity cause is on the key-or-position scale by construction.
    """
    from types import SimpleNamespace

    def parse_error_at(position: int, *, with_index_key: bool):
        steps = []
        for n in range(position + 1):
            step: dict = {"thought": "t", "raw": "r"}
            if with_index_key:
                step["index"] = n
            if n == position:
                step["parse_error"] = "no fenced block"
            else:
                step["action"] = {"name": "read_file", "arguments": {"path": "/a"}}
                step["observation"] = "contents"
            steps.append(step)
        return SimpleNamespace(steps=steps, integrity={}, success=False, horizon=-1)

    without = trajectory_events(parse_error_at(3, with_index_key=False))
    assert without["events"]["invalid_action"] == 3
    assert without == trajectory_events(parse_error_at(3, with_index_key=True))


def test_the_index_rule_is_the_integrity_checkers_own() -> None:
    """The two modules must not drift: ``first_cause`` is a minimum across causes and the
    integrity cause is whatever the checker reported, so a divergence would silently compare
    positions against saved indices."""
    from local_llm_lab.pipeline.coherence import _step_index
    from local_llm_lab.pipeline.integrity import _policy_index

    cases = [
        ({}, 4, 8),
        ({"index": 2}, 4, 8),
        ({"index": 9}, 4, 8),
        ({"index": -1}, 4, 8),
        ({"index": True}, 4, 8),
        ({"index": "3"}, 4, 8),
        ({"index": 0}, 0, 8),
    ]
    for record, position, horizon in cases:
        mine = _step_index(record, position, horizon)
        assert mine == _policy_index(record, position, horizon), record


def test_the_loop_cause_shares_the_saved_index_scale_when_the_horizon_exceeds_the_length() -> None:
    """Deputy's reproduction, 7 September: three identical calls closing a loop carry saved
    indices 3, 4, 5 with task horizon 6, so every saved index is valid at the horizon and at or
    beyond the trajectory's length. If ``loop_step`` scaled by the length alone, the loop cause
    would read 2 (position) against integrity's 5 (saved index) and ``first_cause`` would flip
    to ``loop`` on that artefact. Runner-written steps cannot produce this shape (index equals
    position there); hand-built ones can, which is why the earlier fixtures did not catch it."""
    from types import SimpleNamespace

    call = {"name": "read_file", "arguments": {"path": "same"}}
    steps = [
        {"index": n, "thought": "t", "raw": "r", "action": call, "observation": "contents"}
        for n in (3, 4, 5)
    ]
    trajectory = SimpleNamespace(
        steps=steps,
        success=False,
        horizon=6,
        integrity={"first_violation": {"step": 5, "rule": "carry"}},
    )
    record = trajectory_events(trajectory)
    assert record["events"]["loop"] == 5
    assert record["events"]["integrity"] == 5
    assert record["first_cause"] == "integrity"
    assert loop_step(steps) == 2  # the length scale, when no horizon is given
    assert loop_step(steps, 6) == 5
