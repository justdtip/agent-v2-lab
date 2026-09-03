from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from local_llm_lab.pipeline.env import Fault
from local_llm_lab.pipeline.integrity import check_trajectory
from local_llm_lab.pipeline.protocol import DEFAULT_KEEP_LAST
from local_llm_lab.pipeline.runner import Trajectory, run_task
from local_llm_lab.pipeline.tasks import Task, make_tasks
from local_llm_lab.pipeline.transcript import Transcript, summary_table
from local_llm_lab.project import PROJECT_ROOT, configure_local_cache

DEFAULT_MODEL = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"
STRESS_FAULTS = (Fault(call_index=1),)


def load_policy(model_name: str, adapter: Path | None) -> tuple[Any, Any]:
    configure_local_cache()
    from mlx_lm import load

    return load(model_name, adapter_path=None if adapter is None else str(adapter.resolve()))


def make_sampler(temperature: float) -> Any:
    from mlx_lm.sample_utils import make_sampler as _make

    return _make(temp=temperature)


def evaluate_tasks(
    model: Any,
    tokenizer: Any,
    tasks: list[Task],
    *,
    label: str,
    temperature: float = 0.0,
    max_steps: int = 24,
    max_tokens: int = 200,
    keep_last: int = DEFAULT_KEEP_LAST,
    stress: bool = False,
    transcript_dir: Path | None = None,
    quiet: bool = False,
    use_cache: bool = True,
) -> list[Trajectory]:
    sampler = make_sampler(temperature)
    stream = None if quiet else sys.stdout
    trajectories = []
    for number, task in enumerate(tasks, 1):
        transcript = Transcript(stream=stream, directory=transcript_dir)
        trajectory = run_task(
            model,
            tokenizer,
            task,
            sampler=sampler,
            label=label,
            max_steps=max_steps,
            max_tokens=max_tokens,
            keep_last=keep_last,
            faults=STRESS_FAULTS if stress else None,
            transcript=transcript,
            use_cache=use_cache,
        )
        trajectory.difficulty = task.difficulty
        trajectory.integrity = check_trajectory(
            task, trajectory.steps, keep_last=keep_last
        ).as_dict()
        trajectories.append(trajectory)
        if quiet:
            mark = "PASS" if trajectory.success else "FAIL"
            print(
                f"[{number:02d}/{len(tasks):02d}] {mark} {task.task_id} ({trajectory.turns} turns)",
                flush=True,
            )
    return trajectories


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile of ``values`` for ``q`` in [0, 100]; 0.0 for an empty sample."""
    if not 0 <= q <= 100:
        raise ValueError("q must be between 0 and 100")
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def failure_reason(trajectory: Trajectory) -> str:
    """Single attributed reason for a failed trajectory; the first matching rule wins."""
    if trajectory.parse_error:
        return "parse error"
    integrity = getattr(trajectory, "integrity", {})
    first_violation = integrity.get("first_violation") if isinstance(integrity, dict) else None
    if isinstance(first_violation, dict) and first_violation.get("kind"):
        return str(first_violation["kind"]).replace("_", " ")
    if trajectory.loop_detected:
        return "repetition loop"
    if trajectory.exhausted:
        return "step budget exhausted"
    return (trajectory.verdict.get("reasons") or ["unknown"])[0].split(":")[0]


def _ratio(numerator: int | float, denominator: int | float, digits: int = 4) -> float:
    return round(numerator / denominator, digits) if denominator else 0.0


def _trajectory_totals(trajectories: list[Trajectory]) -> dict[str, Any]:
    totals: dict[str, Any] = {
        "successes": 0,
        "clean": 0,
        "turns": 0,
        "valid": 0,
        "calls": 0,
        "schema_failures": 0,
        "executable": 0,
        "generated_tokens": 0,
        "tool_errors": 0,
        "recovered_errors": 0,
        "loop_failures": 0,
        "exhausted": 0,
        "latencies": [],
    }
    for trajectory in trajectories:
        totals["successes"] += int(trajectory.success)
        totals["clean"] += int(bool(trajectory.verdict.get("clean")))
        totals["turns"] += trajectory.turns
        totals["valid"] += trajectory.valid_turns
        totals["calls"] += trajectory.verdict.get("calls", 0)
        totals["schema_failures"] += trajectory.verdict.get("schema_failures", 0)
        totals["executable"] += trajectory.verdict.get("executable_calls", 0)
        totals["generated_tokens"] += trajectory.generated_tokens
        totals["tool_errors"] += trajectory.verdict.get("errors", 0)
        totals["recovered_errors"] += trajectory.verdict.get("recovered_errors", 0)
        totals["loop_failures"] += int(trajectory.loop_detected and not trajectory.success)
        totals["exhausted"] += int(trajectory.exhausted and not trajectory.success)
        totals["latencies"].append(trajectory.elapsed_seconds)
    return totals


def _failure_reasons(trajectories: list[Trajectory]) -> dict[str, int]:
    reasons: dict[str, int] = {}
    for trajectory in trajectories:
        if trajectory.success:
            continue
        reason = failure_reason(trajectory)
        reasons[reason] = reasons.get(reason, 0) + 1
    return dict(sorted(reasons.items(), key=lambda item: -item[1]))


def _integrity_values(trajectory: Trajectory) -> tuple[dict[str, Any], bool]:
    integrity = trajectory.integrity
    if not isinstance(integrity, dict):
        integrity = {"clean": True, "counts": {}}
    counts = integrity.get("counts", {})
    if not isinstance(counts, dict):
        counts = {}
    return counts, bool(integrity.get("clean", not counts))


def _integrity_family_bucket() -> dict[str, Any]:
    return {
        "trajectories": 0,
        "clean_trajectories": 0,
        "violations": 0,
        "affected_trajectories": 0,
    }


def _integrity_summary(
    trajectories: list[Trajectory], *, failed: int
) -> dict[str, Any]:
    by_kind: dict[str, dict[str, int]] = {}
    by_family: dict[str, dict[str, Any]] = {}
    clean = 0
    affected = 0
    violations = 0
    failed_with_violation = 0
    for trajectory in trajectories:
        counts, is_clean = _integrity_values(trajectory)
        count = sum(int(value) for value in counts.values())
        clean += int(is_clean)
        affected += int(not is_clean)
        violations += count
        failed_with_violation += int(not trajectory.success and not is_clean)
        family = by_family.setdefault(trajectory.family, _integrity_family_bucket())
        family["trajectories"] += 1
        family["clean_trajectories"] += int(is_clean)
        family["violations"] += count
        family["affected_trajectories"] += int(not is_clean)
        for kind, value in counts.items():
            bucket = by_kind.setdefault(
                str(kind), {"violations": 0, "affected_trajectories": 0}
            )
            bucket["violations"] += int(value)
            bucket["affected_trajectories"] += int(value > 0)
    for family in by_family.values():
        family["clean_rate"] = _ratio(
            family["clean_trajectories"], family["trajectories"]
        )
    return {
        "clean_trajectories": clean,
        "clean_rate": _ratio(clean, len(trajectories)),
        "affected_trajectories": affected,
        "violations": violations,
        "by_kind": dict(sorted(by_kind.items())),
        "by_family": dict(sorted(by_family.items())),
        "failed_trajectories": failed,
        "failed_with_violation": failed_with_violation,
        "failure_explained_rate": _ratio(failed_with_violation, failed),
    }


def _group(trajectories: list[Trajectory], key: str) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for trajectory in trajectories:
        bucket = table.setdefault(getattr(trajectory, key), {"successes": 0, "tasks": 0})
        bucket["tasks"] += 1
        bucket["successes"] += int(trajectory.success)
    for bucket in table.values():
        bucket["success_rate"] = _ratio(bucket["successes"], bucket["tasks"])
    return dict(sorted(table.items()))


def summarize(trajectories: list[Trajectory]) -> dict[str, Any]:
    totals = _trajectory_totals(trajectories)
    count = len(trajectories)
    successes = totals["successes"]
    failed = count - totals["successes"]
    return {
        "tasks": count,
        "successes": successes,
        "success_rate": _ratio(successes, count),
        "clean_successes": totals["clean"],
        "clean_rate": _ratio(totals["clean"], count),
        "valid_action_rate": _ratio(totals["valid"], totals["turns"]),
        "schema_validity_rate": _ratio(
            totals["calls"] - totals["schema_failures"], totals["calls"]
        ),
        "executable_call_rate": _ratio(totals["executable"], totals["calls"]),
        "tool_errors": totals["tool_errors"],
        "recovered_errors": totals["recovered_errors"],
        "loop_failures": totals["loop_failures"],
        "exhausted": totals["exhausted"],
        "generated_tokens": totals["generated_tokens"],
        "tokens_per_success": (
            round(totals["generated_tokens"] / successes, 1) if successes else None
        ),
        "latency_p50_seconds": round(percentile(totals["latencies"], 50), 2),
        "latency_p95_seconds": round(percentile(totals["latencies"], 95), 2),
        "mean_steps": _ratio(totals["turns"], count, 2),
        "by_family": _group(trajectories, "family"),
        "by_variant": _group(trajectories, "variant"),
        "failure_reasons": _failure_reasons(trajectories),
        "integrity": _integrity_summary(trajectories, failed=failed),
    }


def write_report(path: Path, summary: dict[str, Any], trajectories: list[Trajectory]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for trajectory in trajectories:
        record = trajectory.as_dict()
        record["difficulty"] = getattr(trajectory, "difficulty", -1)
        record["integrity"] = getattr(trajectory, "integrity", None)
        records.append(record)
    payload = {"summary": summary, "trajectories": records}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_evaluation(
    *,
    model_name: str,
    adapter: Path | None,
    label: str,
    split: str,
    limit: int,
    output: Path,
    transcript_dir: Path | None,
    stress: bool = False,
    temperature: float = 0.0,
    max_steps: int = 24,
    max_tokens: int = 200,
    keep_last: int = DEFAULT_KEEP_LAST,
    quiet: bool = False,
    use_cache: bool = True,
    seed: int = 20260902,
) -> dict[str, Any]:
    import mlx.core as mx

    mx.random.seed(seed)
    model, tokenizer = load_policy(model_name, adapter)
    tasks = make_tasks(split, limit, seed)
    started = time.monotonic()
    trajectories = evaluate_tasks(
        model,
        tokenizer,
        tasks,
        label=label,
        temperature=temperature,
        max_steps=max_steps,
        max_tokens=max_tokens,
        keep_last=keep_last,
        stress=stress,
        transcript_dir=transcript_dir,
        quiet=quiet,
        use_cache=use_cache,
    )
    summary = summarize(trajectories)
    summary.update(
        {
            "label": label,
            "model": model_name,
            "adapter": None if adapter is None else str(adapter.resolve()),
            "split": split,
            "stress": stress,
            "temperature": temperature,
            "keep_last": keep_last,
            "data_seed": seed,
            "kv_cache": use_cache,
            "elapsed_seconds": round(time.monotonic() - started, 2),
        }
    )
    write_report(output, summary, trajectories)
    print(f"\n{label} on {split} ({'stress' if stress else 'clean'}):")
    print(summary_table(summary))
    print(f"Wrote {output}")
    if transcript_dir is not None:
        print(f"Transcripts in {transcript_dir}")
    del model
    mx.clear_cache()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a policy on v2 agent tasks with live transcripts."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--adapter", type=Path, help="Adapter directory; omit for the untouched base."
    )
    parser.add_argument(
        "--label", help="Name shown in transcripts; defaults to the adapter name or 'base'."
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument(
        "--stress",
        action="store_true",
        help="Inject a transient tool error on every task's second call.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--keep-last", type=int, default=DEFAULT_KEEP_LAST)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--transcripts", type=Path)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cross-turn KV cache reuse (a speed-only change verified not to alter outputs).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print pass/fail lines; transcripts still go to disk.",
    )
    args = parser.parse_args()
    if args.limit < 1 or args.max_steps < 1 or args.max_tokens < 1 or args.keep_last < 0:
        parser.error("limit, max-steps, max-tokens must be positive and keep-last non-negative")
    label = args.label or (args.adapter.name if args.adapter else "base")
    stem = f"{label}-{args.split}{'-stress' if args.stress else ''}"
    output = args.output or PROJECT_ROOT / "outputs" / "agent-v2" / "evals" / f"{stem}.json"
    transcripts = args.transcripts or PROJECT_ROOT / "outputs" / "agent-v2" / "transcripts" / stem
    run_evaluation(
        model_name=args.model,
        adapter=args.adapter,
        label=label,
        split=args.split,
        limit=args.limit,
        output=output,
        transcript_dir=transcripts,
        stress=args.stress,
        temperature=args.temperature,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        keep_last=args.keep_last,
        quiet=args.quiet,
        use_cache=not args.no_cache,
    )
