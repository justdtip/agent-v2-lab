from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

from local_llm_lab.agent_protocol import ActionParseError
from local_llm_lab.arch import ArchitectureView
from local_llm_lab.models import ModelSpec, ResolvedSpec, load_model_spec
from local_llm_lab.pipeline.data import write_jsonl
from local_llm_lab.pipeline.env import Simulator
from local_llm_lab.pipeline.evaluate import DEFAULT_MODEL, load_policy, make_sampler
from local_llm_lab.pipeline.protocol import (
    DEFAULT_KEEP_LAST,
    END_OF_TURN,
    SYSTEM_PROMPT,
    assistant_message,
    build_prompt,
    parse_turn,
    tool_message,
)
from local_llm_lab.pipeline.protocol import render_completion as protocol_render_completion
from local_llm_lab.pipeline.runner import generate_turn, run_task
from local_llm_lab.pipeline.tasks import Task, make_tasks
from local_llm_lab.pipeline.transcript import Transcript
from local_llm_lab.project import PROJECT_ROOT

_LEGACY_SPEC = load_model_spec(DEFAULT_MODEL)


def render_completion(thought: str, action: Any) -> str:
    """Compatibility delegate for consumers that still import the legacy branch seam."""
    return protocol_render_completion(thought, action, spec=_LEGACY_SPEC).removesuffix("\n")


def _continue(
    model: Any,
    tokenizer: Any,
    task: Task,
    messages: list[dict[str, Any]],
    simulator: Simulator,
    *,
    spec: ModelSpec,
    sampler: Any,
    remaining_steps: int,
    max_tokens: int,
    keep_last: int,
) -> bool:
    """Roll the policy forward from an arbitrary state; return whether the verifier passes."""
    for _ in range(remaining_steps):
        prompt = build_prompt(tokenizer, messages, spec=spec, keep_last=keep_last)
        raw = generate_turn(model, tokenizer, prompt, sampler, max_tokens)
        try:
            turn = parse_turn(raw)
        except ActionParseError:
            return False
        observation = simulator.execute(turn.action)
        if turn.action.name == "finish":
            break
        messages.append(assistant_message(turn.thought, turn.action))
        messages.append(tool_message(turn.action.name, observation))
    return simulator.verdict().success


def mine_pairs(  # noqa: C901 - branch outcome collection remains an established single transaction.
    model: Any,
    tokenizer: Any,
    task: Task,
    *,
    spec: ModelSpec,
    # DEBT(R20): required once the condition-4 slice threads branch.build_prompt. That slice
    # drops both defaults and updates the two optional-path callers that omit them today,
    # tests/test_branch.py:164 and :210.
    view: ArchitectureView | None = None,
    resolved: ResolvedSpec | None = None,
    branches: int,
    temperature: float,
    max_steps: int,
    max_tokens: int,
    keep_last: int,
    max_pairs_per_point: int = 2,
    transcript: Transcript | None = None,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Branch a verified greedy trajectory at every step and label alternatives by outcome.

    At each branch point the prefix is identical, so (chosen, rejected) differ only in the
    decision the policy made there: an exact step-level preference pair.
    """
    import mlx.core as mx

    greedy = make_sampler(0.0)
    sampler = make_sampler(temperature)
    seed_trajectory = run_task(
        model,
        tokenizer,
        task,
        sampler=greedy,
        spec=spec,
        view=view,
        resolved=resolved,
        label="seed",
        max_steps=max_steps,
        max_tokens=max_tokens,
        keep_last=keep_last,
        transcript=transcript,
    )
    stats = {
        "task_id": task.task_id,
        "seed_success": seed_trajectory.success,
        "branch_points": 0,
        "branches": 0,
        "branch_failures": 0,
        "pairs": 0,
    }
    if not seed_trajectory.success:
        return [], stats

    pairs: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.prompt},
    ]
    simulator = Simulator.for_task(task)
    for point, step in enumerate(seed_trajectory.steps):
        action = step["action"]
        raw = step.get("raw")
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"seed trajectory step {point} is missing a raw completion")
        seed_completion = raw
        prompt = build_prompt(tokenizer, messages, spec=spec, keep_last=keep_last)
        stats["branch_points"] += 1
        good: list[str] = [seed_completion]
        bad: list[str] = []
        for branch in range(branches):
            mx.random.seed(seed * 100_003 + point * 101 + branch)
            raw = generate_turn(model, tokenizer, prompt, sampler, max_tokens)
            completion = raw.replace(END_OF_TURN, "") + END_OF_TURN
            stats["branches"] += 1
            if completion in good or completion in bad:
                continue
            try:
                turn = parse_turn(raw)
            except ActionParseError:
                bad.append(completion)
                stats["branch_failures"] += 1
                continue
            branch_sim = copy.deepcopy(simulator)
            branch_messages = copy.deepcopy(messages)
            observation = branch_sim.execute(turn.action)
            if turn.action.name == "finish":
                success = branch_sim.verdict().success
            else:
                branch_messages.append(assistant_message(turn.thought, turn.action))
                branch_messages.append(tool_message(turn.action.name, observation))
                success = _continue(
                    model,
                    tokenizer,
                    task,
                    branch_messages,
                    branch_sim,
                    spec=spec,
                    sampler=greedy,
                    remaining_steps=max_steps - point - 1,
                    max_tokens=max_tokens,
                    keep_last=keep_last,
                )
            (good if success else bad).append(completion)
            stats["branch_failures"] += int(not success)
        emitted = 0
        for rejected in bad:
            for chosen in good:
                if emitted >= max_pairs_per_point:
                    break
                pairs.append(
                    {
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                        "metadata": {"task_id": task.task_id, "family": task.family, "step": point},
                    }
                )
                emitted += 1
        stats["pairs"] += emitted
        if transcript is not None and bad:
            transcript.step(
                point,
                f"branch point: {len(bad)} failing alternative(s) vs {len(good)} passing",
                None,
                None,
                raw=bad[0],
            )
        # advance the shared prefix along the verified seed trajectory
        observation = simulator.execute(_action(action))
        if action["name"] == "finish":
            break
        messages.append(assistant_message(step["thought"], _action(action)))
        messages.append(tool_message(action["name"], observation))
    return pairs, stats


def _action(payload: dict[str, Any]) -> Any:
    from local_llm_lab.agent_protocol import Action

    return Action(payload["name"], dict(payload["arguments"]))


def run_branch_mining(
    *,
    model_name: str,
    adapter: Path | None,
    split: str,
    limit: int,
    branches: int,
    temperature: float,
    output: Path,
    transcript_dir: Path | None,
    max_steps: int = 24,
    max_tokens: int = 200,
    keep_last: int = DEFAULT_KEEP_LAST,
    quiet: bool = False,
    seed: int = 20260902,
) -> dict[str, Any]:
    import mlx.core as mx

    spec = load_model_spec(model_name)
    model, tokenizer, view, resolved = load_policy(spec, adapter)
    tasks = make_tasks(split, limit, seed)
    started = time.monotonic()
    all_pairs: list[dict[str, Any]] = []
    stats = []
    for number, task in enumerate(tasks, 1):
        transcript = Transcript(stream=None if quiet else sys.stdout, directory=transcript_dir)
        pairs, task_stats = mine_pairs(
            model,
            tokenizer,
            task,
            spec=spec,
            view=view,
            resolved=resolved,
            branches=branches,
            temperature=temperature,
            max_steps=max_steps,
            max_tokens=max_tokens,
            keep_last=keep_last,
            transcript=transcript,
            seed=seed + number,
        )
        all_pairs.extend(pairs)
        stats.append(task_stats)
        print(
            f"[{number:02d}/{len(tasks):02d}] {task.task_id}: seed {'PASS' if task_stats['seed_success'] else 'FAIL'}, "
            f"{task_stats['pairs']} pairs from {task_stats['branches']} branches",
            flush=True,
        )
    output.mkdir(parents=True, exist_ok=True)
    digest = write_jsonl(output / "pairs.jsonl", all_pairs)
    summary = {
        "model": resolved.as_dict(),
        "adapter": None if adapter is None else str(adapter.resolve()),
        "split": split,
        "tasks": len(tasks),
        "seed_successes": sum(s["seed_success"] for s in stats),
        "branch_points": sum(s["branch_points"] for s in stats),
        "branches": sum(s["branches"] for s in stats),
        "branch_failures": sum(s["branch_failures"] for s in stats),
        "pairs": len(all_pairs),
        "pairs_sha256": digest,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "per_task": stats,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"\nMined {len(all_pairs)} preference pairs from {summary['branch_points']} branch points -> {output / 'pairs.jsonl'}"
    )
    del model
    mx.clear_cache()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine step-level preference pairs by branching verified trajectories."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--adapter", type=Path, default=PROJECT_ROOT / "outputs" / "agent-v2" / "best-adapter"
    )
    parser.add_argument("--split", default="pref1")
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--branches", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--keep-last", type=int, default=DEFAULT_KEEP_LAST)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--transcripts", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.split in {"train", "valid", "test"}:
        parser.error("use a fresh split name so evaluation tasks are never mined")
    run_branch_mining(
        model_name=args.model,
        adapter=args.adapter,
        split=args.split,
        limit=args.limit,
        branches=args.branches,
        temperature=args.temperature,
        output=args.output or PROJECT_ROOT / "data" / "preferences" / args.split,
        transcript_dir=args.transcripts
        or PROJECT_ROOT / "outputs" / "agent-v2" / "transcripts" / f"branch-{args.split}",
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        keep_last=args.keep_last,
        quiet=args.quiet,
    )
