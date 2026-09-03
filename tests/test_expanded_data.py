from local_llm_lab.build_expanded_data import DECISION_REPEATS, decision_balanced, target_action


def _tool_row(name: str) -> dict:
    return {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": name, "arguments": "{}"}}],
            }
        ]
    }


def test_target_action_distinguishes_tools_from_chat() -> None:
    assert target_action(_tool_row("set_plan")) == "set_plan"
    assert target_action({"messages": [{"role": "assistant", "content": "hello"}]}) is None


def test_decision_balancing_oversamples_high_level_actions() -> None:
    plan = _tool_row("set_plan")
    read = _tool_row("read_file")
    rows = decision_balanced([plan, read])
    assert rows.count(plan) == DECISION_REPEATS["set_plan"]
    assert rows.count(read) == 1
