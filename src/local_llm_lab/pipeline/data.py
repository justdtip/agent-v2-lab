from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from local_llm_lab.pipeline.env import Simulator
from local_llm_lab.pipeline.protocol import (
    DEFAULT_KEEP_LAST,
    SYSTEM_PROMPT,
    assistant_message,
    tool_message,
    window_messages,
)
from local_llm_lab.pipeline.tasks import GENERATOR_VERSION, Task, make_tasks


def build_rows(task: Task, *, keep_last: int = DEFAULT_KEEP_LAST) -> list[dict[str, Any]]:
    """Replay a task through the simulator, emitting one supervised row per expert decision.

    Rows see exactly the windowed context the policy will see at inference time. Steps that
    deliberately fail (recovery variants) are executed for their observation but never become
    training targets. Rows carry no ``tools`` key: the system message (``SYSTEM_PROMPT``)
    already contains the rendered tool list, and the assistant target is plain text with the
    call in a fenced JSON block.
    """
    simulator = Simulator.for_task(task)
    fault_indices = {fault.call_index for fault in task.faults}
    context: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.prompt},
    ]
    rows = []
    previous_failed = False
    for index, step in enumerate(task.steps):
        assistant = assistant_message(step.thought, step.action)
        if step.supervise:
            rows.append(
                {
                    "messages": [*window_messages(context, keep_last), assistant],
                    "metadata": {
                        "task_id": task.task_id,
                        "family": task.family,
                        "variant": task.variant,
                        "step": index,
                        "source": "expert",
                        # True when the previous observation was an error, so this target is
                        # the corrective action. These are the scarcest and most important
                        # decisions in the set, and are oversampled in the training split.
                        "recovery": previous_failed,
                    },
                }
            )
        context.append(assistant)
        result = simulator.execute(step.action)
        failed = result.startswith("ERROR")
        previous_failed = failed
        if not step.supervise and not failed:
            raise RuntimeError(f"{task.task_id}: unsupervised step {index} did not fail")
        if step.supervise and failed and index not in fault_indices:
            raise RuntimeError(f"{task.task_id}: expert step {index} failed: {result}")
        if step.action.name != "finish":
            context.append(tool_message(step.action.name, result))
    verdict = simulator.verdict()
    if not verdict.success:
        raise RuntimeError(
            f"{task.task_id}: expert trajectory failed verification: {verdict.reasons}"
        )
    return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_dataset(
    output: Path,
    counts: dict[str, int],
    *,
    seed: int = 20260902,
    keep_last: int = DEFAULT_KEEP_LAST,
    chat_dir: Path | None = None,
    chat_repeats: int = 1,
    recovery_repeats: int | dict[str, int] = 1,
    extra_dirs: list[Path] | None = None,
) -> dict[str, Any]:
    """Write train/valid/test JSONL plus a manifest describing exactly what went in.

    ``recovery_repeats`` oversamples corrective decisions in the training split only, either as
    one factor for all of them or as a per-variant mapping. Re-observation recoveries (relist
    after a stale path, re-read after a failed edit) are hosted by only a few families, and the
    run is shorter than one epoch, so without oversampling a given target may never be visited.
    Held-out splits are never reweighted.
    """
    manifest: dict[str, Any] = {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "keep_last": keep_last,
        "protocol": "v2-state-carrying-notes-fenced-json",
        "recovery_repeats": recovery_repeats,
        "splits": {},
    }
    for split, count in counts.items():
        tasks = make_tasks(split, count, seed)
        expert_rows = [row for task in tasks for row in build_rows(task, keep_last=keep_last)]
        recovery_rows = sum(row["metadata"].get("recovery", False) for row in expert_rows)
        if split == "train":
            expert_rows = [
                row for row in expert_rows for _ in range(_repeats(row, recovery_repeats))
            ]
        chat_rows: list[dict[str, Any]] = []
        if chat_dir is not None and (chat_dir / f"{split}.jsonl").is_file():
            chat_rows = read_jsonl(chat_dir / f"{split}.jsonl") * chat_repeats
        extra_rows: list[dict[str, Any]] = []
        if split == "train":
            for extra in extra_dirs or []:
                extra_rows.extend(read_jsonl(extra / "train.jsonl"))
        rows = [*expert_rows, *chat_rows, *extra_rows]
        random.Random(f"{seed}:{split}").shuffle(rows)
        digest = write_jsonl(output / f"{split}.jsonl", rows)
        horizons = [task.horizon for task in tasks]
        manifest["splits"][split] = {
            "tasks": count,
            "rows": len(rows),
            "expert_rows": len(expert_rows),
            "recovery_targets": recovery_rows,
            "recovery_rows_after_repeats": sum(
                row["metadata"].get("recovery", False) for row in expert_rows
            ),
            "chat_rows": len(chat_rows),
            "extra_rows": len(extra_rows),
            "variants": _histogram(task.variant for task in tasks),
            "families": _histogram(task.family for task in tasks),
            "min_horizon": min(horizons),
            "max_horizon": max(horizons),
            "mean_horizon": round(sum(horizons) / len(horizons), 2),
            "sha256": digest,
        }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _repeats(row: dict[str, Any], recovery_repeats: int | dict[str, int]) -> int:
    """How many times a training row is emitted; only corrective decisions are repeated."""
    metadata = row.get("metadata", {})
    if not metadata.get("recovery"):
        return 1
    if isinstance(recovery_repeats, int):
        return max(1, recovery_repeats)
    return max(1, int(recovery_repeats.get(metadata.get("variant", ""), 1)))


def _histogram(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
