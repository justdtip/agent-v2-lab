"""The relation test (D10): two files with swapped states, generated, replayed, and scored."""

from __future__ import annotations

from local_llm_lab.agent_protocol import Action
from local_llm_lab.pipeline.env import Simulator
from local_llm_lab.pipeline.state_programme import diagnostics as dx
from local_llm_lab.pipeline.state_programme import family
from local_llm_lab.pipeline.state_programme.run import ScriptedPolicy, _relation_rows, run_episode


def _context(row, pair, target):
    return dx.Context.from_trajectory(row["steps"], target=target, directory=pair.directory)


def test_the_two_episodes_swap_the_states_and_share_one_prompt() -> None:
    pair = family.make_relation_pairs("fx", 1, 1, seed=3)[0]
    e1, e2 = pair.episode_1, pair.episode_2
    assert e1.prompt == e2.prompt
    assert pair.file_a in e1.files and pair.file_b not in e1.files
    assert pair.file_b in e2.files and pair.file_a not in e2.files
    assert {p for p in e1.files if p != pair.file_a} == {p for p in e2.files if p != pair.file_b}
    name_a, name_b = pair.file_a.rsplit("/", 1)[-1], pair.file_b.rsplit("/", 1)[-1]
    assert name_a in e1.prompt and name_b in e1.prompt
    assert family.NO_SUMMARY in e1.prompt


def test_both_experts_replay_cleanly_and_the_answers_name_both_files() -> None:
    for level in (0, 2):
        for pair in family.make_relation_pairs("fx", 3, level, seed=11):
            for _arm, task in pair.episodes():
                sim = Simulator.for_task(task)
                for step in task.steps:
                    sim.execute(step.action)
                verdict = sim.verdict()
                assert verdict.success and verdict.clean, (task.task_id, verdict.reasons)
                assert family.NO_SUMMARY in task.expected_answer


def test_the_expert_scores_one_on_d10_and_zero_on_the_a_arm() -> None:
    pair = family.make_relation_pairs("fx", 1, 0, seed=5)[0]
    rows = _relation_rows(pair, ScriptedPolicy())
    assert [r["arm"] for r in rows] == ["T1", "T2"]
    assert "D10" not in rows[0]["scores"] and rows[1]["scores"]["D10"] == 1.0
    assert rows[1]["scores"]["D10_follows_A"] == 0.0
    assert rows[1]["relation_target"] == pair.file_b


def test_a_policy_that_acts_on_b_by_a_state_scores_zero_on_d10() -> None:
    """Episode 2: A absent, B present. A policy that reads A anyway is acting on A's state."""
    pair = family.make_relation_pairs("fx", 1, 0, seed=8)[0]
    read_a = ("Reading A regardless.", Action("read_file", {"path": pair.file_a}))
    wrong = ScriptedPolicy(deviations={1: read_a})
    row1 = run_episode(pair.episode_1, ScriptedPolicy())
    row2 = run_episode(pair.episode_2, wrong)
    c1 = _context(row1, pair, pair.file_b)
    c2 = _context(row2, pair, pair.file_b)
    assert dx.d10_relation(c1, c2) == 0.0
    assert dx.relation_scores(c1, c2) == {"D10": 0.0, "D10_follows_A": 1.0}


def test_an_unreached_relation_scores_none_on_both_arms() -> None:
    empty = dx.Context.from_trajectory([], target="x/notes/b.md", directory="x/notes")
    assert dx.relation_scores(empty, empty) == {"D10": None, "D10_follows_A": None}
