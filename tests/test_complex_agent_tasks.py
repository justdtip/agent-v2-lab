from local_llm_lab.agent_tasks import ToolSimulator
from local_llm_lab.complex_agent_tasks import make_complex_tasks
from local_llm_lab.generate_agent_data import build_sft_rows


def test_complex_expert_trajectories_pass() -> None:
    for split in ("train", "valid", "test"):
        for task in make_complex_tasks(split, 18):
            simulator = ToolSimulator(task)
            for action in task.expert_actions:
                assert not simulator.execute(action).startswith("ERROR:")
            assert simulator.success, task.task_id


def test_complex_splits_are_isolated_and_test_is_longer() -> None:
    train = make_complex_tasks("train", 18)
    test = make_complex_tasks("test", 18)
    assert {task.task_id for task in train}.isdisjoint(task.task_id for task in test)
    assert {task.prompt for task in train}.isdisjoint(task.prompt for task in test)
    assert sum(map(lambda task: len(task.expert_actions), test)) > sum(
        map(lambda task: len(task.expert_actions), train)
    )


def test_every_complex_action_is_a_supervised_target() -> None:
    for task in make_complex_tasks("train", 6):
        assert len(build_sft_rows(task)) == len(task.expert_actions)


def test_complex_tasks_have_visible_planning_checkpoints() -> None:
    for task in make_complex_tasks("train", 6):
        assert task.expert_actions[0].name == "set_plan"
        assert task.expert_actions[-2].name == "update_plan"
        assert {"set_plan", "update_plan"}.issubset(task.required_tools)


def test_complex_horizons_exceed_original_curriculum() -> None:
    tasks = make_complex_tasks("test", 60)
    horizons = [len(task.expert_actions) for task in tasks]
    assert min(horizons) >= 7
    assert max(horizons) >= 13
