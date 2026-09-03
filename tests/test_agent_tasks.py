from local_llm_lab.agent_tasks import ToolSimulator, make_tasks
from local_llm_lab.generate_agent_data import build_sft_rows


def test_all_expert_trajectories_pass() -> None:
    for split in ("train", "valid", "test"):
        for task in make_tasks(split, 18):
            simulator = ToolSimulator(task)
            for action in task.expert_actions:
                assert not simulator.execute(action).startswith("ERROR:")
            assert simulator.success, task.task_id


def test_splits_have_distinct_task_ids_and_content() -> None:
    train = make_tasks("train", 12)
    test = make_tasks("test", 12)
    assert {task.task_id for task in train}.isdisjoint(task.task_id for task in test)
    assert {task.prompt for task in train}.isdisjoint(task.prompt for task in test)


def test_each_assistant_action_becomes_a_target() -> None:
    task = make_tasks("train", 1)[0]
    rows = build_sft_rows(task)
    assert len(rows) == len(task.expert_actions)
    assert all(row["messages"][-1]["role"] == "assistant" for row in rows)
    assert all(row["tools"] for row in rows)
