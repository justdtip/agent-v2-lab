from local_llm_lab.chat_replay import make_chat_prompts


def test_chat_prompt_splits_are_distinct_and_balanced() -> None:
    train = make_chat_prompts("train", 12)
    test = make_chat_prompts("test", 12)
    assert {item.prompt_id for item in train}.isdisjoint(item.prompt_id for item in test)
    assert {item.messages[-1]["content"] for item in train}.isdisjoint(
        item.messages[-1]["content"] for item in test
    )
    assert len({item.category for item in train}) == 6


def test_follow_up_prompts_have_conversation_history() -> None:
    prompts = make_chat_prompts("train", 6)
    follow_up = next(item for item in prompts if item.category == "follow_up")
    assert [message["role"] for message in follow_up.messages] == [
        "user",
        "assistant",
        "user",
    ]
