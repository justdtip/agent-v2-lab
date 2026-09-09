"""How a trajectory's outcome is recorded: repetition, truncation, and the answer's bound.

Three columns that each existed in a form that could report the wrong thing quietly:

- ``longest_identical_run`` reads 1 on a model alternating two failing calls, while
  ``loop_detected`` reads true. That column was quoted across a day of comparison tables.
- a turn cut off by the token cap was scored as a wrong answer, which charges the model for a
  budget we chose.
- ``success`` requires exact normalised equality, which is stricter than several task prompts
  ask for, so a pass rate was carrying an unstated bound.

Each test below is written against the shape that fooled the old column, not against a
convenient one.
"""

from __future__ import annotations

from types import SimpleNamespace

from local_llm_lab.agent_protocol import Action
from local_llm_lab.models import load_model_spec
from local_llm_lab.pipeline import runner
from local_llm_lab.pipeline.env import Simulator
from local_llm_lab.pipeline.protocol import render_turn
from local_llm_lab.pipeline.runner import (
    detect_loop,
    longest_period_run,
    repetition_metrics,
)
from local_llm_lab.pipeline.tasks import Task


def _step(index: int, name: str, arguments: dict, observation: str = "ok") -> dict:
    return {
        "index": index,
        "thought": "",
        "action": {"name": name, "arguments": arguments},
        "observation": observation,
        "raw": "",
    }


def _task(expected: str = "x", required: frozenset[str] = frozenset()) -> Task:
    return Task(
        task_id="test-outcomes-0001-clean",
        family="runner",
        variant="clean",
        prompt="finish with x",
        files={},
        steps=(),
        expected_answer=expected,
        required_tools=required,
    )


def test_the_two_cycle_that_defeated_the_old_column() -> None:
    """The defect itself: alternating two calls loops, and the old column reads 1."""
    alternating = [
        _step(index, "read_file", {"path": "a" if index % 2 else "b"}, "ERROR: not found")
        for index in range(10)
    ]
    metrics = repetition_metrics(alternating)
    assert metrics["longest_identical_run"] == 1, (
        "the old column is kept exactly as it was, so an old table stays readable"
    )
    assert metrics["longest_period_run"] == 10
    assert metrics["cycle_period"] == 2
    assert detect_loop(alternating), "the trajectory is looping while the old column reads 1"


def test_a_repeated_single_call_reports_the_same_number_in_both_columns() -> None:
    repeated = [_step(index, "calculate", {"expression": "1+1"}, "RESULT: 2") for index in range(7)]
    metrics = repetition_metrics(repeated)
    assert metrics["longest_identical_run"] == 7
    assert metrics["longest_period_run"] == 7
    assert metrics["cycle_period"] == 1, "period 1 means the two columns describe one repetition"


def test_a_three_cycle_and_a_run_of_distinct_calls() -> None:
    three = [_step(index, f"tool{index % 3}", {"n": index % 3}) for index in range(9)]
    metrics = repetition_metrics(three)
    assert metrics["cycle_period"] == 3 and metrics["longest_period_run"] == 9

    distinct = [_step(index, f"tool{index}", {"n": index}) for index in range(6)]
    metrics = repetition_metrics(distinct)
    assert metrics["longest_identical_run"] == 1, (
        "the old column calls a single non-repeating call a run of one, and keeping that is "
        "the whole point of not recomputing it"
    )
    assert metrics["longest_period_run"] == 0
    assert metrics["cycle_period"] == 0, "no repetition means no period, not period 1"

    empty: list[dict] = []
    assert repetition_metrics(empty)["longest_identical_run"] == 0


def test_a_period_needs_two_full_cycles_before_it_is_called_one() -> None:
    """``abcab`` ends mid-cycle; reading a period off it would find one in any sequence."""
    partial = [_step(index, name, {}) for index, name in enumerate("abcab")]
    assert longest_period_run(runner.call_signatures(partial), 3) == 0
    complete = [_step(index, name, {}) for index, name in enumerate("abcabc")]
    assert longest_period_run(runner.call_signatures(complete), 3) == 6


def test_parse_error_steps_are_not_calls() -> None:
    steps = [
        {"index": 0, "raw": "", "parse_error": "x"},
        *[_step(i, "read_file", {}) for i in range(3)],
    ]
    metrics = repetition_metrics(steps)
    assert metrics["executed_calls"] == 3, "an unparsed turn carries no call"
    assert metrics["longest_identical_run"] == 3


def test_a_turn_that_hits_the_cap_without_a_parseable_action_is_truncated(monkeypatch) -> None:
    spec = load_model_spec("qwen35-4b")
    monkeypatch.setattr(
        runner,
        "build_prompt",
        lambda tokenizer, messages, *, spec, keep_last, generation=True: "turn",
    )

    def cut_off(model, tokenizer, prompt, sampler, max_tokens, turn_cache, *, spec):
        return "a note and then the budget ran o", max_tokens, 0

    monkeypatch.setattr(runner, "generate_turn_with_count", cut_off)
    trajectory = runner.run_task(
        object(),
        SimpleNamespace(encode=lambda text: list(range(len(text)))),
        _task(),
        sampler=None,
        spec=spec,
        view=None,
        resolved=None,
        max_steps=1,
        max_tokens=17,
        use_cache=False,
    )
    assert trajectory.truncated is True
    assert trajectory.steps[0]["truncated"] is True
    assert trajectory.parse_error is not None


def test_a_short_unparseable_turn_is_not_truncated(monkeypatch) -> None:
    """The control: unparseable is not the same fact as cut off, and only one is our doing."""
    spec = load_model_spec("qwen35-4b")
    monkeypatch.setattr(
        runner,
        "build_prompt",
        lambda tokenizer, messages, *, spec, keep_last, generation=True: "turn",
    )

    def gave_up_early(model, tokenizer, prompt, sampler, max_tokens, turn_cache, *, spec):
        return "not a tool call at all", 4, 0

    monkeypatch.setattr(runner, "generate_turn_with_count", gave_up_early)
    trajectory = runner.run_task(
        object(),
        SimpleNamespace(encode=lambda text: list(range(len(text)))),
        _task(),
        sampler=None,
        spec=spec,
        view=None,
        resolved=None,
        max_steps=1,
        max_tokens=200,
        use_cache=False,
    )
    assert trajectory.parse_error is not None
    assert trajectory.truncated is False
    assert trajectory.steps[0]["truncated"] is False


def _verdict_for(answer: str, expected: str) -> dict:
    simulator = Simulator.for_task(_task(expected=expected))
    simulator.execute(Action("finish", {"answer": answer}))
    return simulator.verdict().as_dict()


def test_contains_expected_carries_the_bound_on_every_pass_rate() -> None:
    exact = _verdict_for("x", "x")
    assert exact["success"] is True and exact["contains_expected"] is True

    wordy = _verdict_for("The answer is x", "x")
    assert wordy["success"] is False, "the grader requires exact equality and still does"
    assert wordy["contains_expected"] is True, (
        "the answer is present; the strict grade is stricter than the prompt asks for"
    )

    wrong = _verdict_for("y", "x")
    assert wrong["success"] is False and wrong["contains_expected"] is False

    silent = Simulator.for_task(_task()).verdict().as_dict()
    assert silent["success"] is False and silent["contains_expected"] is False


def test_contains_expected_uses_the_same_normaliser_as_success() -> None:
    """The two grades must differ only in exact-versus-containment, never in normalisation."""
    fenced = _verdict_for("`x`.", "x")
    assert fenced["success"] is True and fenced["contains_expected"] is True
    spaced = _verdict_for("  X  ", "x")
    assert spaced["success"] is True and spaced["contains_expected"] is True


def test_run_task_records_both_repetition_columns(monkeypatch) -> None:
    spec = load_model_spec("qwen35-4b")
    monkeypatch.setattr(
        runner,
        "build_prompt",
        lambda tokenizer, messages, *, spec, keep_last, generation=True: "turn",
    )
    finish = render_turn("done", Action("finish", {"answer": "x"}))

    def one_call(model, tokenizer, prompt, sampler, max_tokens, turn_cache, *, spec):
        return finish, 8, 0

    monkeypatch.setattr(runner, "generate_turn_with_count", one_call)
    trajectory = runner.run_task(
        object(),
        SimpleNamespace(encode=lambda text: list(range(len(text)))),
        _task(),
        sampler=None,
        spec=spec,
        view=None,
        resolved=None,
        max_steps=1,
        use_cache=False,
    )
    assert set(trajectory.repetition) == {
        "longest_identical_run",
        "longest_period_run",
        "cycle_period",
        "distinct_calls",
        "executed_calls",
    }
    assert trajectory.repetition["executed_calls"] == 1
    assert trajectory.as_dict()["repetition"] == trajectory.repetition
