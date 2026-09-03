from local_llm_lab.rewards import (
    combined_reward,
    exact_match_reward,
    extract_answer,
    format_reward,
    numeric_match_reward,
)


def test_extracts_answer_tag() -> None:
    assert extract_answer("work\n<answer> 42 </answer>") == "42"


def test_extracts_last_final_line() -> None:
    assert extract_answer("FINAL: draft\nmore work\nFINAL: done") == "done"


def test_exact_match_normalizes_case_and_whitespace() -> None:
    assert exact_match_reward("FINAL:  Blue   Whale ", "blue whale") == 1.0


def test_numeric_match_accepts_equivalent_decimal() -> None:
    assert numeric_match_reward("FINAL: 12.0", "12") == 1.0


def test_numeric_match_rejects_wrong_answer() -> None:
    assert numeric_match_reward("FINAL: 13", "12") == 0.0


def test_format_rewards_machine_readable_markers() -> None:
    assert format_reward("<answer>yes</answer>") == 1.0
    assert format_reward("FINAL: yes") == 0.8
    assert format_reward("yes") == 0.0


def test_combined_reward_prefers_correct_formatted_answer() -> None:
    answer = "12"
    assert combined_reward("FINAL: 12", answer) > combined_reward("12", answer)
    assert combined_reward("12", answer) > combined_reward("FINAL: 13", answer)
