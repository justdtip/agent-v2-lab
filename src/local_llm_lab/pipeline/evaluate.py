from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from local_llm_lab.arch import ArchitectureView
from local_llm_lab.models import ModelSpec, ResolvedSpec, load_model_spec
from local_llm_lab.pipeline.env import Fault
from local_llm_lab.pipeline.integrity import check_trajectory, git_tree_dirty
from local_llm_lab.pipeline.protocol import DEFAULT_KEEP_LAST
from local_llm_lab.pipeline.coherence import coherence_summary, trajectory_events
from local_llm_lab.pipeline.runner import Trajectory, run_task
from local_llm_lab.pipeline.tasks import (
    GENERATOR_VERSION,
    Task,
    family_balanced_tasks,
    make_tasks,
)
from local_llm_lab.pipeline.transcript import Transcript, summary_table
from local_llm_lab.project import PROJECT_ROOT, configure_local_cache
from local_llm_lab.runlock import load_weights
from local_llm_lab.runlog import RunLog, git_commit

DEFAULT_MODEL = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"
STRESS_FAULTS = (Fault(call_index=1),)


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Return the unclamped-precision Wilson score interval for a binomial rate."""
    if n < 0 or successes < 0 or successes > n:
        raise ValueError("successes must be between zero and n")
    if n == 0:
        return (0.0, 0.0)
    proportion = successes / n
    denominator = 1 + z * z / n
    centre = (proportion + z * z / (2 * n)) / denominator
    radius = z / denominator * math.sqrt(
        proportion * (1 - proportion) / n + z * z / (4 * n * n)
    )
    return (max(0.0, centre - radius), min(1.0, centre + radius))


def mcnemar(a: dict[str, bool], b: dict[str, bool]) -> dict[str, Any]:
    """Calculate the exact two-sided McNemar test for matching non-empty task IDs."""
    if not a or a.keys() != b.keys():
        raise ValueError("McNemar inputs must contain the same non-empty task ids")
    a_only = sum(bool(a[key]) and not bool(b[key]) for key in a)
    b_only = sum(bool(b[key]) and not bool(a[key]) for key in a)
    discordant = a_only + b_only
    tail = sum(math.comb(discordant, i) for i in range(min(a_only, b_only) + 1))
    return {
        "tasks": len(a),
        "a_only": a_only,
        "b_only": b_only,
        "discordant": discordant,
        "p_value": 1.0 if discordant == 0 else min(1.0, 2 * tail / (2**discordant)),
    }


def _seed_model_rng(seed: int) -> None:
    """Seed MLX only at the model boundary so fakes can exercise evaluation wiring."""
    import mlx.core as mx

    mx.random.seed(seed)


def _clear_model_cache() -> None:
    """Release model allocations after an evaluation without exposing MLX to callers."""
    import mlx.core as mx

    mx.clear_cache()


def load_policy(
    spec: ModelSpec,
    adapter: Path | None,
    *,
    lazy: bool = False,
) -> tuple[Any, Any, ArchitectureView, ResolvedSpec]:
    """Load one declared policy together with its architecture view and resolved spec.

    Resolution happens here so that no caller rebuilds an ``ArchitectureView`` or repeats
    ``ModelSpec.resolve``; every stage then records the same resolved declaration. ``lazy``
    loads parameters on demand for structural inspection only (the preflight stage).

    Weights come through ``runlock.load_weights``, never ``mlx_lm.load``: this function is the
    shared loader for every v2 stage and probe, so it is where the model-run lock belongs
    (issue 83). It is taken before the load and held until the process exits, because that is
    how long the weights stay resident.
    """
    configure_local_cache()
    model, tokenizer = load_weights(
        spec.hf_id,
        adapter_path=None if adapter is None else str(adapter.resolve()),
        lazy=lazy,
    )
    return model, tokenizer, ArchitectureView.from_model(model), spec.resolve(model, tokenizer)


def make_sampler(temperature: float) -> Any:
    from mlx_lm.sample_utils import make_sampler as _make

    return _make(temp=temperature)


def evaluate_tasks(
    model: Any,
    tokenizer: Any,
    tasks: list[Task],
    *,
    spec: ModelSpec,
    view: ArchitectureView,
    resolved: ResolvedSpec,
    label: str,
    temperature: float = 0.0,
    max_steps: int = 24,
    max_tokens: int = 200,
    keep_last: int = DEFAULT_KEEP_LAST,
    stress: bool = False,
    transcript_dir: Path | None = None,
    quiet: bool = False,
    use_cache: bool = True,
    log: RunLog | None = None,
) -> list[Trajectory]:
    """Run every task under one loaded policy, carrying its model context into each rollout.

    ``log`` is optional so the function stays callable from a test or a notebook without a
    run directory; when given, it receives one progress event per task (R26(g)).
    """
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
            spec=spec,
            view=view,
            resolved=resolved,
            label=label,
            max_steps=max_steps,
            max_tokens=max_tokens,
            keep_last=keep_last,
            faults=STRESS_FAULTS if stress else None,
            transcript=transcript,
            use_cache=use_cache,
        )
        trajectory.difficulty = task.difficulty
        trajectory.horizon = task.horizon  # for the scale-free first-event fraction (coherence.py)
        trajectory.integrity = check_trajectory(
            task, trajectory.steps, keep_last=keep_last
        ).as_dict()
        trajectories.append(trajectory)
        if log is not None:
            # R26(g): the evaluated task is this stage's outer unit of work, so the run is
            # never silent for longer than one rollout.
            log.progress(
                number,
                len(tasks),
                "task",
                task_id=task.task_id,
                family=task.family,
                difficulty=task.difficulty,
                success=trajectory.success,
                turns=trajectory.turns,
            )
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


def coherence_first_cause(trajectory: Trajectory) -> str | None:
    """The first incoherence cause of a trajectory, or None (pipeline/coherence.py)."""
    return trajectory_events(trajectory)["first_cause"]


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
        "think_tokens": 0,
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
        totals["think_tokens"] += trajectory.think_tokens
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
        bucket = table.setdefault(
            str(getattr(trajectory, key)),
            {"successes": 0, "tasks": 0, "loop_failures": 0, "exhausted": 0},
        )
        bucket["tasks"] += 1
        bucket["successes"] += int(trajectory.success)
        # UNIFIED-RUN-2026-09-06 (10:12, item d): loops and exhaustion per cell, the
        # pre-registered measurement of the note-form hypothesis; the totals already existed
        # only globally (the Research Division, 12:35).
        bucket["loop_failures"] += int(trajectory.loop_detected and not trajectory.success)
        bucket["exhausted"] += int(trajectory.exhausted and not trajectory.success)
    for bucket in table.values():
        bucket["success_rate"] = _ratio(bucket["successes"], bucket["tasks"])
        bucket["rate_counts"] = {
            "success": {"numerator": bucket["successes"], "denominator": bucket["tasks"]}
        }
        bucket["wilson_95"] = {"success": wilson(bucket["successes"], bucket["tasks"])}
    return dict(sorted(table.items()))


def summarize(trajectories: list[Trajectory], *, max_steps: int = 24) -> dict[str, Any]:
    totals = _trajectory_totals(trajectories)
    count = len(trajectories)
    successes = totals["successes"]
    failed = count - totals["successes"]
    integrity = _integrity_summary(trajectories, failed=failed)
    rate_counts = {
        "success": {"numerator": successes, "denominator": count},
        "clean": {"numerator": totals["clean"], "denominator": count},
        "valid_actions": {"numerator": totals["valid"], "denominator": totals["turns"]},
        "schema_validity": {
            "numerator": totals["calls"] - totals["schema_failures"],
            "denominator": totals["calls"],
        },
        "executable_calls": {"numerator": totals["executable"], "denominator": totals["calls"]},
        "integrity_clean": {
            "numerator": integrity["clean_trajectories"],
            "denominator": count,
        },
    }
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
        "think_tokens": totals["think_tokens"],
        "think_tokens_per_task": _ratio(totals["think_tokens"], count, 2),
        "tokens_per_success": (
            round(totals["generated_tokens"] / successes, 1) if successes else None
        ),
        "latency_p50_seconds": round(percentile(totals["latencies"], 50), 2),
        "latency_p95_seconds": round(percentile(totals["latencies"], 95), 2),
        "mean_steps": _ratio(totals["turns"], count, 2),
        "by_family": _group(trajectories, "family"),
        "by_variant": _group(trajectories, "variant"),
        "by_difficulty": _group(trajectories, "difficulty"),
        "failure_reasons": _failure_reasons(trajectories),
        "integrity": integrity,
        # The capability standard (UNIFIED-RUN-2026-09-06 section 2 revised): coherence to
        # completion with the four causes as competing risks; see pipeline/coherence.py.
        "coherence": coherence_summary(trajectories, max_steps=max_steps),
        "coherent_to_completion_rate": _ratio(
            sum(1 for t in trajectories if t.success and coherence_first_cause(t) is None), count
        ),
        "rate_counts": rate_counts,
        "wilson_95": {
            key: wilson(counts["numerator"], counts["denominator"])
            for key, counts in rate_counts.items()
        },
    }


def evaluation_metadata(
    *,
    label: str,
    resolved: ResolvedSpec,
    adapter: Path | None,
    split: str,
    difficulties: list[int],
    stress: bool,
    temperature: float,
    keep_last: int,
    seed: int,
    use_cache: bool,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """The run-identity half of an evaluation summary, which :func:`summarize` cannot know.

    ``summarize`` sees trajectories only, so every key a downstream reader uses to say *which*
    run an artifact is — ``label``, ``split``, ``data_seed``, ``keep_last``, ``model`` — is
    added here.  Split out of :func:`run_evaluation` (R38) so a test can build an artifact
    carrying exactly the keys a real run records without loading a policy: a hand-written
    summary would certify ``report.load_summaries`` and ``integrity._analyse_evaluation``
    against their authors' belief about this block rather than against the block itself.
    """
    return {
        "label": label,
        "model": resolved.as_dict(),
        "adapter": None if adapter is None else str(adapter.resolve()),
        "split": split,
        # A cohort spanning two levels has no single difficulty; ``difficulties`` keeps the set.
        "difficulty": difficulties[0] if len(difficulties) == 1 else None,
        "difficulties": difficulties,
        "stress": stress,
        "temperature": temperature,
        "keep_last": keep_last,
        "data_seed": seed,
        # The generator that made these tasks, recorded beside the seed that placed them.
        # Both readers of this artifact already prefer a recorded version over their
        # ``--generator-version`` flag and refuse a conflicting one, but no writer recorded
        # one, so the preference had nothing to prefer and every replay was bound by hand.
        # ``_evaluation_identity`` has always put it in run.log; run.log is not the artifact
        # an offline reader is handed.  Absence still fails closed, which is what keeps the
        # saved pre-versioning evaluations requiring an explicit binding.
        "generator_version": GENERATOR_VERSION,
        "kv_cache": use_cache,
        "elapsed_seconds": elapsed_seconds,
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


def _evaluation_identity(
    spec: ModelSpec, adapter: Path | None, label: str, split: str, seed: int
) -> dict[str, Any]:
    """R26(e), issue #35: what the start event names, so run.log alone identifies the run."""
    return {
        "model": spec.name,
        "hf_id": spec.hf_id,
        "policy": label,
        "adapter": None if adapter is None else str(Path(adapter).resolve()),
        "split": split,
        "data_seed": seed,
        "generator_version": GENERATOR_VERSION,
        "git_commit": git_commit(),
        "git_dirty": git_tree_dirty(),
    }


def run_evaluation(
    *,
    spec: ModelSpec,
    adapter: Path | None,
    label: str,
    split: str,
    limit: int | None,
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
    difficulty: int | None = None,
    family_quotas: dict[str, int] | None = None,
) -> dict[str, Any]:
    # R26(a): the log opens before the RNG seeding and the policy load, so a run that dies
    # loading weights still leaves run.log and events.jsonl beside the evaluation it never
    # wrote. The evaluation artifact is a file, so the run directory is its parent.
    output = Path(output)
    log = RunLog.open(
        output.parent,
        name="eval",
        command=list(sys.argv),
        identity=_evaluation_identity(spec, adapter, label, split, seed),
    )
    completed = False
    try:
        _seed_model_rng(seed)
        log.info(
            "loading policy",
            hf_id=spec.hf_id,
            adapter=None if adapter is None else str(adapter),
        )
        model, tokenizer, view, resolved = load_policy(spec, adapter)
        tasks = (
            family_balanced_tasks(
                split,
                difficulty=difficulty if difficulty is not None else 0,
                per_family=family_quotas,
                seed=seed,
            )
            if family_quotas is not None
            else make_tasks(split, limit or 0, seed, difficulty=difficulty)
        )
        if limit is not None:
            tasks = tasks[:limit]
        log.info("tasks resolved", count=len(tasks), split=split, stress=stress)
        started = time.monotonic()
        trajectories = evaluate_tasks(
            model,
            tokenizer,
            tasks,
            spec=spec,
            view=view,
            resolved=resolved,
            label=label,
            temperature=temperature,
            max_steps=max_steps,
            max_tokens=max_tokens,
            keep_last=keep_last,
            stress=stress,
            transcript_dir=transcript_dir,
            quiet=quiet,
            use_cache=use_cache,
            log=log,
        )
        summary = summarize(trajectories, max_steps=max_steps)
        summary.update(
            evaluation_metadata(
                label=label,
                resolved=resolved,
                adapter=adapter,
                split=split,
                difficulties=sorted({task.difficulty for task in tasks}),
                stress=stress,
                temperature=temperature,
                keep_last=keep_last,
                seed=seed,
                use_cache=use_cache,
                elapsed_seconds=round(time.monotonic() - started, 2),
            )
        )
        write_report(output, summary, trajectories)
        print(f"\n{label} on {split} ({'stress' if stress else 'clean'}):")
        print(summary_table(summary))
        print(f"Wrote {output}")
        if transcript_dir is not None:
            print(f"Transcripts in {transcript_dir}")
        del model
        _clear_model_cache()
        completed = True
        return summary
    finally:
        # R26(c)'s incomplete_run rule on a non-training stage: an evaluation that stopped
        # before writing its report scored nothing anything downstream may select on.
        log.close(status="ok" if completed else "error", incomplete_run=not completed)


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
        spec=load_model_spec(args.model),
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
