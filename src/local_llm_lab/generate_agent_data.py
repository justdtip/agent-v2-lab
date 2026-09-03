from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from local_llm_lab.agent_protocol import (
    SYSTEM_PROMPT,
    TOOL_SPECS,
    assistant_tool_message,
    tool_result_message,
)
from local_llm_lab.agent_tasks import AgentTask, ToolSimulator, make_tasks
from local_llm_lab.project import PROJECT_ROOT


def build_sft_rows(
    task: AgentTask,
    *,
    system_prompt: str = SYSTEM_PROMPT,
    tool_specs: list[dict[str, Any]] = TOOL_SPECS,
) -> list[dict[str, Any]]:
    """Expand a trajectory so each agent action is a separately supervised target."""
    context: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task.prompt},
    ]
    rows = []
    simulator = ToolSimulator(task)
    for step, action in enumerate(task.expert_actions):
        assistant = assistant_tool_message(action)
        rows.append(
            {
                "messages": context + [assistant],
                "tools": tool_specs,
                "metadata": {
                    "task_id": task.task_id,
                    "category": task.category,
                    "step": step,
                },
            }
        )
        context.append(assistant)
        result = simulator.execute(action)
        if action.name != "finish":
            context.append(tool_result_message(action.name, result))
    if not simulator.success:
        raise RuntimeError(f"expert trajectory failed its verifier: {task.task_id}")
    return rows


def write_split(
    path: Path,
    tasks: list[AgentTask],
    *,
    system_prompt: str = SYSTEM_PROMPT,
    tool_specs: list[dict[str, Any]] = TOOL_SPECS,
) -> int:
    rows = [
        row
        for task in tasks
        for row in build_sft_rows(task, system_prompt=system_prompt, tool_specs=tool_specs)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return len(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic agent tool-use data.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "agent_sft")
    parser.add_argument("--train-tasks", type=int, default=180)
    parser.add_argument("--valid-tasks", type=int, default=30)
    parser.add_argument("--test-tasks", type=int, default=40)
    args = parser.parse_args()

    counts = {
        "train": args.train_tasks,
        "valid": args.valid_tasks,
        "test": args.test_tasks,
    }
    if any(count < 1 for count in counts.values()):
        parser.error("all split sizes must be positive")

    output = args.output.resolve()
    manifest: dict[str, Any] = {
        "seed": 20260902,
        "protocol_version": 1,
        "splits": {},
    }
    for split, task_count in counts.items():
        tasks = make_tasks(split, task_count)
        path = output / f"{split}.jsonl"
        row_count = write_split(path, tasks)
        category_counts = {
            category: sum(task.category == category for task in tasks)
            for category in sorted({task.category for task in tasks})
        }
        manifest["splits"][split] = {
            "tasks": task_count,
            "action_targets": row_count,
            "categories": category_counts,
            "sha256": file_sha256(path),
        }
        print(f"{split:5s}: {task_count:3d} tasks -> {row_count:3d} action targets")
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote agent SFT data to {output}")
