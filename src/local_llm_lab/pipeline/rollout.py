from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from local_llm_lab.pipeline.data import write_jsonl
from local_llm_lab.pipeline.evaluate import DEFAULT_MODEL, load_policy, make_sampler, summarize
from local_llm_lab.pipeline.integrity import check_trajectory
from local_llm_lab.pipeline.protocol import DEFAULT_KEEP_LAST
from local_llm_lab.pipeline.runner import Trajectory, run_task, trajectory_rows
from local_llm_lab.pipeline.tasks import Task, make_tasks
from local_llm_lab.pipeline.transcript import Transcript, summary_table
from local_llm_lab.project import PROJECT_ROOT


def collect_rollouts(
    model: Any,
    tokenizer: Any,
    tasks: list[Task],
    *,
    label: str,
    samples: int,
    temperature: float,
    keep_per_task: int = 2,
    max_steps: int = 24,
    max_tokens: int = 200,
    keep_last: int = DEFAULT_KEEP_LAST,
    transcript_dir: Path | None = None,
    quiet: bool = False,
    seed: int = 20260902,
) -> tuple[list[Trajectory], list[dict[str, Any]], dict[str, Any]]:
    """Sample the policy on fresh tasks; keep verifier-approved trajectories as new supervision."""
    import mlx.core as mx

    sampler = make_sampler(temperature)
    stream = None if quiet else sys.stdout
    all_trajectories: list[Trajectory] = []
    rows: list[dict[str, Any]] = []
    solved_tasks = 0
    for number, task in enumerate(tasks, 1):
        candidates = []
        for sample in range(samples):
            mx.random.seed(seed * 1000 + number * 10 + sample)
            transcript = Transcript(stream=stream, directory=transcript_dir)
            trajectory = run_task(
                model,
                tokenizer,
                task,
                sampler=sampler,
                label=f"{label}-s{sample}",
                max_steps=max_steps,
                max_tokens=max_tokens,
                keep_last=keep_last,
                transcript=transcript,
            )
            integrity = check_trajectory(task, trajectory.steps, keep_last=keep_last)
            trajectory.difficulty = task.difficulty
            trajectory.integrity = integrity.as_dict()
            all_trajectories.append(trajectory)
            if trajectory.success and integrity.clean:
                candidates.append(trajectory)
        seen: set[tuple[str, ...]] = set()
        kept = 0
        for trajectory in sorted(candidates, key=lambda t: (not t.verdict["clean"], t.turns)):
            signature = tuple(
                json.dumps(step["action"], sort_keys=True) for step in trajectory.steps
            )
            if signature in seen:
                continue
            seen.add(signature)
            rows.extend(trajectory_rows(trajectory, task, keep_last=keep_last))
            kept += 1
            if kept >= keep_per_task:
                break
        solved_tasks += bool(candidates)
        if quiet:
            print(
                f"[{number:02d}/{len(tasks):02d}] {task.task_id}: {len(candidates)}/{samples} passed, kept {kept}",
                flush=True,
            )
    summary = summarize(all_trajectories)
    summary.update(
        {
            "label": label,
            "samples_per_task": samples,
            "temperature": temperature,
            "tasks_solved_at_least_once": solved_tasks,
            "pass_at_k": round(solved_tasks / len(tasks), 4),
            "kept_rows": len(rows),
        }
    )
    return all_trajectories, rows, summary


def run_rollout(
    *,
    model_name: str,
    adapter: Path | None,
    label: str,
    split: str,
    limit: int,
    samples: int,
    temperature: float,
    output: Path,
    transcript_dir: Path | None,
    keep_per_task: int = 2,
    max_steps: int = 24,
    max_tokens: int = 200,
    keep_last: int = DEFAULT_KEEP_LAST,
    quiet: bool = False,
    seed: int = 20260902,
) -> dict[str, Any]:
    import mlx.core as mx

    model, tokenizer = load_policy(model_name, adapter)
    tasks = make_tasks(split, limit, seed)
    started = time.monotonic()
    trajectories, rows, summary = collect_rollouts(
        model,
        tokenizer,
        tasks,
        label=label,
        samples=samples,
        temperature=temperature,
        keep_per_task=keep_per_task,
        max_steps=max_steps,
        max_tokens=max_tokens,
        keep_last=keep_last,
        transcript_dir=transcript_dir,
        quiet=quiet,
        seed=seed,
    )
    summary.update(
        {
            "model": model_name,
            "adapter": None if adapter is None else str(adapter.resolve()),
            "split": split,
            "elapsed_seconds": round(time.monotonic() - started, 2),
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    digest = write_jsonl(output / "train.jsonl", rows)
    summary["train_sha256"] = digest
    with (output / "rollouts.jsonl").open("w", encoding="utf-8") as handle:
        for trajectory in trajectories:
            record = trajectory.as_dict()
            record["difficulty"] = getattr(trajectory, "difficulty", -1)
            record["integrity"] = getattr(trajectory, "integrity", None)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nrollout {label} on {split}: pass@{samples}={summary['pass_at_k']:.1%}")
    print(summary_table(summary))
    print(f"Kept {len(rows)} verified rows -> {output / 'train.jsonl'}")
    del model
    mx.clear_cache()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect verifier-approved rollouts for expert iteration."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--adapter", type=Path, default=PROJECT_ROOT / "outputs" / "agent-v2" / "best-adapter"
    )
    parser.add_argument(
        "--split", default="iter1", help="Any name other than train/valid/test yields fresh tasks."
    )
    parser.add_argument("--limit", type=int, default=96)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--keep-per-task", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--keep-last", type=int, default=DEFAULT_KEEP_LAST)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--transcripts", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.split in {"train", "valid", "test"}:
        parser.error("rollouts must use a fresh split name so evaluation data is never trained on")
    if min(args.limit, args.samples, args.keep_per_task, args.max_steps, args.max_tokens) < 1:
        parser.error("limit, samples, keep-per-task, max-steps, and max-tokens must be positive")
    output = args.output or PROJECT_ROOT / "data" / "rollouts" / args.split
    transcripts = (
        args.transcripts
        or PROJECT_ROOT / "outputs" / "agent-v2" / "transcripts" / f"rollout-{args.split}"
    )
    run_rollout(
        model_name=args.model,
        adapter=args.adapter,
        label=f"rollout-{args.split}",
        split=args.split,
        limit=args.limit,
        samples=args.samples,
        temperature=args.temperature,
        output=output,
        transcript_dir=transcripts,
        keep_per_task=args.keep_per_task,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        keep_last=args.keep_last,
        quiet=args.quiet,
    )
