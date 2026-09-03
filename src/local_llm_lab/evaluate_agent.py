from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from local_llm_lab.agent_protocol import (
    SYSTEM_PROMPT,
    TOOL_SPECS,
    ActionParseError,
    assistant_tool_message,
    parse_action,
    tool_result_message,
)
from local_llm_lab.agent_tasks import ToolSimulator, make_tasks
from local_llm_lab.project import PROJECT_ROOT, configure_local_cache

DEFAULT_MODEL = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"


@dataclass
class EvaluationRecord:
    task_id: str
    category: str
    prompt: str
    success: bool
    steps: int
    valid_actions: int
    tool_errors: int
    parse_error: str | None
    expected_answer: str
    actual_answer: str | None
    calls: list[dict[str, Any]]
    observations: list[dict[str, str]]
    responses: list[str]


def evaluate_task(
    model: Any,
    tokenizer: Any,
    sampler: Any,
    task: Any,
    max_steps: int,
    max_tokens: int,
    system_prompt: str = SYSTEM_PROMPT,
    tool_specs: list[dict[str, Any]] = TOOL_SPECS,
) -> EvaluationRecord:
    from mlx_lm import generate

    simulator = ToolSimulator(task)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task.prompt},
    ]
    responses = []
    observations = []
    parse_error = None
    valid_actions = 0

    for _ in range(max_steps):
        prompt = tokenizer.apply_chat_template(
            messages,
            tools=tool_specs,
            add_generation_prompt=True,
            tokenize=False,
        )
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        ).strip()
        responses.append(response)
        try:
            action = parse_action(response)
        except ActionParseError as error:
            parse_error = str(error)
            break

        valid_actions += 1
        result = simulator.execute(action)
        observations.append({"tool": action.name, "result": result})
        if action.name == "finish":
            break
        messages.append(assistant_tool_message(action))
        messages.append(tool_result_message(action.name, result))

    return EvaluationRecord(
        task_id=task.task_id,
        category=task.category,
        prompt=task.prompt,
        success=simulator.success,
        steps=len(simulator.calls),
        valid_actions=valid_actions,
        tool_errors=simulator.errors,
        parse_error=parse_error,
        expected_answer=task.expected_answer,
        actual_answer=simulator.finished_answer,
        calls=[asdict(action) for action in simulator.calls],
        observations=observations,
        responses=responses,
    )


def summarize(records: list[EvaluationRecord]) -> dict[str, Any]:
    successes = sum(record.success for record in records)
    total_responses = sum(len(record.responses) for record in records)
    valid_actions = sum(record.valid_actions for record in records)
    by_category = {}
    for category in sorted({record.category for record in records}):
        selected = [record for record in records if record.category == category]
        by_category[category] = {
            "successes": sum(record.success for record in selected),
            "tasks": len(selected),
            "success_rate": round(sum(record.success for record in selected) / len(selected), 4),
        }
    return {
        "successes": successes,
        "tasks": len(records),
        "success_rate": round(successes / len(records), 4),
        "valid_action_rate": round(valid_actions / total_responses, 4) if total_responses else 0.0,
        "tool_errors": sum(record.tool_errors for record in records),
        "mean_steps": round(sum(record.steps for record in records) / len(records), 3),
        "by_category": by_category,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a base or adapted model as an agent.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.limit < 1 or args.max_steps < 1 or args.max_tokens < 1:
        parser.error("limit, max-steps, and max-tokens must be positive")

    configure_local_cache()
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler

    adapter = str(args.adapter.resolve()) if args.adapter else None
    model, tokenizer = load(args.model, adapter_path=adapter)
    sampler = make_sampler(temp=0.0)
    tasks = make_tasks(args.split, args.limit)

    records = []
    started = time.monotonic()
    for index, task in enumerate(tasks, 1):
        record = evaluate_task(model, tokenizer, sampler, task, args.max_steps, args.max_tokens)
        records.append(record)
        mark = "PASS" if record.success else "FAIL"
        print(f"[{index:02d}/{len(tasks):02d}] {mark} {task.task_id} ({record.steps} steps)")

    summary = summarize(records)
    summary.update(
        {
            "model": args.model,
            "adapter": adapter,
            "split": args.split,
            "elapsed_seconds": round(time.monotonic() - started, 2),
        }
    )
    output = args.output or (
        PROJECT_ROOT
        / "outputs"
        / "agent-3b"
        / ("adapted-eval.json" if adapter else "base-eval.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"summary": summary, "records": [asdict(record) for record in records]},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output}")
