from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from local_llm_lab.complex_agent_protocol import COMPLEX_SYSTEM_PROMPT, COMPLEX_TOOL_SPECS
from local_llm_lab.complex_agent_tasks import make_complex_tasks
from local_llm_lab.depth_expansion import ExpansionSpec, load_expanded_model
from local_llm_lab.evaluate_agent import DEFAULT_MODEL, evaluate_task, summarize
from local_llm_lab.project import PROJECT_ROOT, configure_local_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate long-horizon agent behavior.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--policy", choices=("base", "agent", "expanded"), default="agent")
    parser.add_argument(
        "--base-adapter",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "agent-3b" / "best-adapter",
    )
    parser.add_argument("--extension", type=Path)
    parser.add_argument(
        "--extension-weights",
        type=Path,
        help="Optional checkpoint file; defaults to EXTENSION/extension.safetensors.",
    )
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=18)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.limit < 1 or args.max_steps < 1 or args.max_tokens < 1:
        parser.error("limit, max-steps, and max-tokens must be positive")
    if args.policy == "expanded" and args.extension is None:
        parser.error("--extension is required for the expanded policy")

    configure_local_cache()
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler

    if args.policy == "base":
        model, tokenizer = load(args.model)
        architecture = {"policy": "base", "adapter": None, "added_layers": 0}
    elif args.policy == "agent":
        adapter = args.base_adapter.resolve()
        model, tokenizer = load(args.model, adapter_path=str(adapter))
        architecture = {"policy": "agent", "adapter": str(adapter), "added_layers": 0}
    else:
        extension = args.extension.resolve()
        spec = ExpansionSpec.load(extension / "expansion.json")
        weights = (
            args.extension_weights.resolve()
            if args.extension_weights
            else extension / "extension.safetensors"
        )
        model, tokenizer, added_indices = load_expanded_model(
            args.model,
            args.base_adapter,
            spec,
            weights,
        )
        architecture = {
            "policy": "expanded",
            "adapter": str(args.base_adapter.resolve()),
            "extension": str(extension),
            "extension_weights": str(weights),
            "added_layers": len(added_indices),
            "added_layer_indices": list(added_indices),
        }

    sampler = make_sampler(temp=0.0)
    tasks = make_complex_tasks(args.split, args.limit)
    records = []
    started = time.monotonic()
    for index, task in enumerate(tasks, 1):
        record = evaluate_task(
            model,
            tokenizer,
            sampler,
            task,
            args.max_steps,
            args.max_tokens,
            system_prompt=COMPLEX_SYSTEM_PROMPT,
            tool_specs=COMPLEX_TOOL_SPECS,
        )
        records.append(record)
        mark = "PASS" if record.success else "FAIL"
        print(
            f"[{index:02d}/{len(tasks):02d}] {mark} {task.task_id} ({record.steps} steps)",
            flush=True,
        )

    summary = summarize(records)
    summary.update(
        {
            "model": args.model,
            "split": args.split,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "architecture": architecture,
        }
    )
    output = args.output or (
        PROJECT_ROOT / "outputs" / "complex-agent" / f"{args.policy}-eval.json"
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
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {output}", flush=True)
