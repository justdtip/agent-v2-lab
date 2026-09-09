"""The existence family: matched pairs, the state only in the tool result, and the false result."""

from __future__ import annotations

import random

import pytest

from local_llm_lab.pipeline import tasks as task_module
from local_llm_lab.pipeline.env import Simulator
from local_llm_lab.pipeline.state_programme import family
from local_llm_lab.pipeline.tasks import FAMILIES


def _replay(task):
    sim = Simulator.for_task(task)
    observations = [sim.execute(step.action) for step in task.steps]
    return sim, observations


def test_the_two_arms_differ_only_in_the_target_file_and_say_nothing_about_it() -> None:
    pair = family.make_existence_pairs("fx", 1, 1, seed=3)[0]
    e, a = pair.exists, pair.absent

    assert e.prompt == a.prompt, "one string, both arms"
    assert pair.target not in e.prompt and family.NO_SUMMARY in e.prompt
    assert set(e.files) - set(a.files) == {pair.target}
    assert all(e.files[p] == a.files[p] for p in a.files)
    assert e.task_id.endswith("-E") and a.task_id.endswith("-A")
    assert e.task_id[:-2] == a.task_id[:-2] == pair.pair_id


def test_both_experts_replay_cleanly_through_the_real_environment() -> None:
    for level in (0, 1, 2):
        for pair in family.make_existence_pairs("fx", 3, level, seed=11):
            for _arm, task in pair.arms():
                sim, _ = _replay(task)
                verdict = sim.verdict()
                assert verdict.success and verdict.clean, (task.task_id, verdict.reasons)


def test_the_state_enters_through_the_listing_and_the_experts_diverge_only_after_it() -> None:
    pair = family.make_existence_pairs("fx", 1, 0, seed=5)[0]
    _, obs_e = _replay(pair.exists)
    _, obs_a = _replay(pair.absent)

    assert pair.exists.steps[0] == pair.absent.steps[0], "the first action is identical"
    assert pair.target in obs_e[0] and pair.target not in obs_a[0]
    assert pair.exists.steps[1].action.name == "read_file"
    assert pair.absent.steps[1].action.name == "finish"


def test_pairs_are_deterministic_in_the_seed_and_distinct_across_it() -> None:
    a = family.make_existence_pairs("fx", 4, 1, seed="s1")
    b = family.make_existence_pairs("fx", 4, 1, seed="s1")
    c = family.make_existence_pairs("fx", 4, 1, seed="s2")
    assert [p.exists.files for p in a] == [p.exists.files for p in b]
    assert [p.target for p in a] != [p.target for p in c]


# ------------------------------------------------------------------- the reliability instrument


def test_the_false_listing_is_well_formed_and_contradicts_the_truth_in_each_arm() -> None:
    pair = family.make_existence_pairs("fx", 1, 1, seed=9)[0]
    truthful_e = Simulator.for_task(pair.exists).execute(pair.exists.steps[0].action)
    truthful_a = Simulator.for_task(pair.absent).execute(pair.absent.steps[0].action)
    false_e = family.false_listing(pair.exists, arm="E")
    false_a = family.false_listing(pair.absent, arm="A")

    for text in (false_e, false_a):
        assert text.startswith("FILES: ") and not text.startswith("ERROR")
    assert pair.target in truthful_e and pair.target not in false_e
    assert pair.target not in truthful_a
    assert "summary-" in false_a and false_a != truthful_a
    # Same directory, same distractors: only the target's presence is lied about.
    listed = set(false_e.split(": ", 1)[1].split(", "))
    assert listed == set(truthful_e.split(": ", 1)[1].split(", ")) - {pair.target}


def test_a_false_observation_is_applied_by_the_environment_before_validation() -> None:
    pair = family.make_existence_pairs("fx", 1, 0, seed=2)[0]
    falsified = family.with_false_observation(pair.exists, arm="E")
    sim = Simulator.for_task(falsified)
    first = sim.execute(falsified.steps[0].action)

    assert first == family.false_listing(pair.exists, arm="E")
    assert pair.target not in first, "the model is told the file is not there"
    assert falsified.variant == "false_observation"
    assert falsified.faults[0].call_index == 0
    # A second listing is truthful: the fault is at one call index, like every other fault.
    assert pair.target in sim.execute(falsified.steps[0].action)


def test_the_rate_is_applied_per_arm_with_a_seed_and_reported_per_episode() -> None:
    pairs = family.make_existence_pairs("fx", 50, 0, seed=1)
    rows = family.apply_reliability(pairs, rate=0.3, seed=7)
    again = family.apply_reliability(pairs, rate=0.3, seed=7)

    assert len(rows) == 100
    assert [r[3] for r in rows] == [r[3] for r in again], "seeded"
    share = sum(r[3] for r in rows) / len(rows)
    assert 0.15 < share < 0.45
    assert all((t.variant == "false_observation") == f for _, _, t, f in rows)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        family.apply_reliability(pairs, rate=1.5, seed=0)


# ---------------------------------------------------------------------------- registration


def test_the_maker_is_registered_beside_list_and_not_in_the_family_cycle() -> None:
    """A thirteenth entry in FAMILIES would move every task id after it in every recorded corpus."""
    assert "existence" in task_module._MAKERS
    assert "existence" not in FAMILIES
    assert len(FAMILIES) == 12
    probe = task_module._MAKERS["existence"]("train", 0, 1, random.Random("probe:1"))
    assert probe.family == family.FAMILY and probe.task_id.endswith("-E")
    # The variant probe reads the maker's step shape and must not choke on it.
    assert "clean" in task_module.applicable_variants("existence", 1)
