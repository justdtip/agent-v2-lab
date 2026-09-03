from __future__ import annotations

import argparse
import re
from decimal import Decimal, InvalidOperation

_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_FINAL_LINE = re.compile(r"^\s*FINAL\s*:\s*(.*?)\s*$", re.MULTILINE | re.IGNORECASE)
_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")


def normalize_text(value: str) -> str:
    """Normalize a textual answer for deterministic comparisons."""
    return " ".join(value.casefold().strip().split())


def extract_answer(response: str) -> str:
    """Extract an answer tag or FINAL line, falling back to the whole response."""
    tagged = _ANSWER_TAG.search(response)
    if tagged:
        return tagged.group(1).strip()
    final_lines = _FINAL_LINE.findall(response)
    if final_lines:
        return final_lines[-1].strip()
    return response.strip()


def _last_number(value: str) -> Decimal | None:
    matches = _NUMBER.findall(value.replace(",", ""))
    if not matches:
        return None
    try:
        return Decimal(matches[-1])
    except InvalidOperation:
        return None


def exact_match_reward(response: str, ground_truth: str) -> float:
    return float(normalize_text(extract_answer(response)) == normalize_text(ground_truth))


def numeric_match_reward(response: str, ground_truth: str) -> float:
    predicted = _last_number(extract_answer(response))
    expected = _last_number(ground_truth)
    return float(predicted is not None and expected is not None and predicted == expected)


def format_reward(response: str, _ground_truth: str = "") -> float:
    """Reward an explicit machine-readable answer marker."""
    if _ANSWER_TAG.search(response):
        return 1.0
    if _FINAL_LINE.search(response):
        return 0.8
    return 0.0


def conciseness_reward(response: str, _ground_truth: str = "") -> float:
    """A gentle preference for useful short answers, not a hard length cutoff."""
    words = response.split()
    if 2 <= len(words) <= 40:
        return 1.0
    if len(words) <= 80:
        return 0.5
    return 0.0


def combined_reward(response: str, ground_truth: str) -> float:
    """Layered reward suitable for the included toy arithmetic experiment."""
    correctness = max(
        exact_match_reward(response, ground_truth),
        numeric_match_reward(response, ground_truth),
    )
    score = (
        0.75 * correctness
        + 0.20 * format_reward(response, ground_truth)
        + 0.05 * conciseness_reward(response, ground_truth)
    )
    return round(score, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the example reward function.")
    parser.add_argument("--response")
    parser.add_argument("--answer")
    args = parser.parse_args()

    if args.response is not None or args.answer is not None:
        if args.response is None or args.answer is None:
            parser.error("--response and --answer must be provided together")
        print(f"answer:      {extract_answer(args.response)!r}")
        print(f"exact:       {exact_match_reward(args.response, args.answer):.2f}")
        print(f"numeric:     {numeric_match_reward(args.response, args.answer):.2f}")
        print(f"format:      {format_reward(args.response, args.answer):.2f}")
        print(f"conciseness: {conciseness_reward(args.response, args.answer):.2f}")
        print(f"combined:    {combined_reward(args.response, args.answer):.2f}")
        return

    examples = (
        ("The result is twelve.\nFINAL: 12", "12"),
        ("<answer>12</answer>", "12"),
        ("I think the answer is 12", "12"),
        ("FINAL: 13", "12"),
    )
    print("Example reward distribution:")
    for response, answer in examples:
        print(f"  {combined_reward(response, answer):.2f}  {response!r}")
