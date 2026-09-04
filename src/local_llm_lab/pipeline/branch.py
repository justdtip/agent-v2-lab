from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from local_llm_lab.agent_protocol import ActionParseError
from local_llm_lab.arch import ArchitectureView
from local_llm_lab.models import ModelSpec, ResolvedSpec, load_model_spec
from local_llm_lab.pipeline.data import guard_dataset_write, write_jsonl
from local_llm_lab.pipeline.env import Simulator
from local_llm_lab.pipeline.evaluate import DEFAULT_MODEL, load_policy, make_sampler
from local_llm_lab.pipeline.protocol import (
    DEFAULT_KEEP_LAST,
    SYSTEM_PROMPT,
    assistant_message,
    build_prompt,
    parse_turn,
    tool_message,
)
from local_llm_lab.pipeline.protocol import render_completion as protocol_render_completion
from local_llm_lab.pipeline.runner import generate_turn, run_task
from local_llm_lab.pipeline.tasks import GENERATOR_VERSION, Task, make_tasks
from local_llm_lab.pipeline.transcript import Transcript
from local_llm_lab.project import PROJECT_ROOT
from local_llm_lab.provenance import write_provenance
from local_llm_lab.runlog import RunLog, git_commit, write_text_atomic

_LEGACY_SPEC = load_model_spec(DEFAULT_MODEL)


def _write_stage_manifest(target: Path, payload: dict[str, Any]) -> None:
    """Stamp this stage's data output with manifest.json after it succeeds.

    The R21 guard fires only on this file, so without the stamp the guard call in ``main``
    would be inert on a re-run. Same shape as the pipeline CLI's stamp (cli.py:177), which
    this entry point cannot import: the CLI imports ``run_branch_mining`` from here.

    The write goes through ``runlog.write_text_atomic`` because this file is the guard's own
    sentinel: ``guard_dataset_write`` (pipeline/data.py:85) decides whether a later write is
    permitted by whether it is there, so a truncated one is worse than none — it can wave a
    later write straight over mined pairs that are in fact complete.
    """
    target.mkdir(parents=True, exist_ok=True)
    write_text_atomic(target / "manifest.json", json.dumps(payload, indent=2) + "\n")


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
    # DEBT(R20): the mine_pairs limb is discharged -- view and resolved are required here and
    # every caller now supplies them. One limb of the condition-4 slice remains: routing
    # cli.py::_load_training_base through load_policy(adapter=None, lazy=False). DEFERRED --
    # src/local_llm_lab/pipeline/cli.py is owned by an uncommitted lane sitting at the Chief's
    # gate, so no other lane may edit it; R20 expires when that lane lands and closes it.
    view: ArchitectureView,
    resolved: ResolvedSpec,
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
    # R2: the turn terminator is a property of the run's model, not of the module-level
    # compatibility constant, which is evaluated at import time from the legacy 3B spec.
    end_of_turn = spec.chat.end_of_turn
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
            completion = raw.replace(end_of_turn, "") + end_of_turn
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
    progress: Callable[..., None] | None = None,
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
        if progress is not None:
            # R26(g): one event per outer unit of work, so no run is silent for longer
            # than a single task.
            progress(
                number,
                len(tasks),
                task.task_id,
                seed_success=task_stats["seed_success"],
                pairs=task_stats["pairs"],
                branches=task_stats["branches"],
            )
    output.mkdir(parents=True, exist_ok=True)
    digest = write_jsonl(output / "pairs.jsonl", all_pairs)
    summary = {
        "model": resolved.as_dict(),
        "adapter": None if adapter is None else str(adapter.resolve()),
        "split": split,
        "seed": seed,
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
    output = args.output or PROJECT_ROOT / "data" / "preferences" / args.split
    # R21: this entry point writes under data/ without going through the CLI boundary, so it
    # carries the guard itself. The stage has no overwrite flag, hence override_flag=None.
    guard_dataset_write(output, override_flag=None)
    transcripts = (
        args.transcripts
        or PROJECT_ROOT / "outputs" / "agent-v2" / "transcripts" / f"branch-{args.split}"
    )
    spec = load_model_spec(args.model)
    # R26(a)/(e): the log opens before any work so run.log alone identifies the run.
    with RunLog.open(
        output,
        name="agent-v2-branch",
        command=sys.argv,
        identity={
            "model": spec.name,
            "hf_id": spec.hf_id,
            "split": args.split,
            "adapter": None if args.adapter is None else str(args.adapter),
            "limit": args.limit,
            "branches": args.branches,
            "output": str(output),
            "git_commit": git_commit(),
        },
    ) as log:
        summary = run_branch_mining(
            model_name=args.model,
            adapter=args.adapter,
            split=args.split,
            limit=args.limit,
            branches=args.branches,
            temperature=args.temperature,
            output=output,
            transcript_dir=transcripts,
            max_steps=args.max_steps,
            max_tokens=args.max_tokens,
            keep_last=args.keep_last,
            quiet=args.quiet,
            progress=log.progress,
        )
        _write_stage_manifest(
            output,
            {
                "stage": "branch",
                "generator_version": GENERATOR_VERSION,
                "model": args.model,
                "split": args.split,
                "seed": summary.get("seed"),
                "adapter": None if args.adapter is None else str(args.adapter),
                "summary": summary,
            },
        )
        provenance = write_provenance(
            output,
            resolved=None,
            spec=spec,
            extra={"stage": "branch", "summary": summary},
        )
        log.info(
            "branch mining complete",
            pairs=summary.get("pairs"),
            branch_points=summary.get("branch_points"),
            manifest=str(output / "manifest.json"),
            provenance=str(provenance),
        )
