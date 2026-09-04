from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from local_llm_lab.complex_agent_protocol import COMPLEX_SYSTEM_PROMPT, COMPLEX_TOOL_SPECS
from local_llm_lab.complex_agent_tasks import make_complex_tasks
from local_llm_lab.generate_agent_data import file_sha256, write_split
from local_llm_lab.pipeline.data import guard_dataset_write
from local_llm_lab.project import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate higher-complexity agent trajectories.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "complex_agent_sft")
    parser.add_argument("--train-tasks", type=int, default=180)
    parser.add_argument("--valid-tasks", type=int, default=36)
    parser.add_argument("--test-tasks", type=int, default=60)
    args = parser.parse_args()
    counts = {
        "train": args.train_tasks,
        "valid": args.valid_tasks,
        "test": args.test_tasks,
    }
    if any(count < 1 for count in counts.values()):
        parser.error("all split sizes must be positive")

    output = args.output.resolve()
    # R21: refuse before the first split is written — a protected directory unconditionally, an
    # existing dataset outright, because this stage has no override flag.
    guard_dataset_write(output, override_flag=None)
    manifest: dict[str, Any] = {
        "seed": 20260902,
        "protocol_version": 2,
        "curriculum": "long-horizon with test-time length extrapolation",
        "splits": {},
    }
    for split, task_count in counts.items():
        tasks = make_complex_tasks(split, task_count)
        path = output / f"{split}.jsonl"
        row_count = write_split(
            path,
            tasks,
            system_prompt=COMPLEX_SYSTEM_PROMPT,
            tool_specs=COMPLEX_TOOL_SPECS,
        )
        horizons = [len(task.expert_actions) for task in tasks]
        categories = sorted({task.category for task in tasks})
        manifest["splits"][split] = {
            "tasks": task_count,
            "action_targets": row_count,
            "categories": {
                category: sum(task.category == category for task in tasks)
                for category in categories
            },
            "min_horizon": min(horizons),
            "max_horizon": max(horizons),
            "mean_horizon": round(sum(horizons) / len(horizons), 3),
            "sha256": file_sha256(path),
        }
        print(
            f"{split:5s}: {task_count:3d} tasks -> {row_count:4d} targets; "
            f"horizon {min(horizons)}-{max(horizons)}"
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote complex agent SFT data to {output}")
