from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from local_llm_lab.agent_tasks import make_tasks
from local_llm_lab.evaluate_agent import DEFAULT_MODEL, evaluate_task
from local_llm_lab.project import PROJECT_ROOT, configure_local_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base and adapter action traces.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--adapter",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "agent-3b" / "best-adapter",
    )
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=6)
    args = parser.parse_args()
    if args.task_index < 0 or args.max_steps < 1:
        parser.error("task-index must be non-negative and max-steps must be positive")

    adapter = args.adapter.resolve()
    if not (adapter / "adapters.safetensors").is_file():
        raise SystemExit(f"Trained adapter not found: {adapter}")
    task = make_tasks(args.split, args.task_index + 1)[args.task_index]
    configure_local_cache()

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0)
    base, tokenizer = load(args.model)
    base_record = evaluate_task(base, tokenizer, sampler, task, args.max_steps, 160)
    del base
    mx.clear_cache()

    tuned, tokenizer = load(args.model, adapter_path=str(adapter))
    tuned_record = evaluate_task(tuned, tokenizer, sampler, task, args.max_steps, 160)
    payload = {
        "task": {"task_id": task.task_id, "prompt": task.prompt},
        "base": asdict(base_record),
        "adapted": asdict(tuned_record),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
