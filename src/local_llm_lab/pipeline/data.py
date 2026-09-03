from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

from local_llm_lab.models import ModelSpec
from local_llm_lab.pipeline.env import Simulator
from local_llm_lab.pipeline.protocol import (
    DEFAULT_KEEP_LAST,
    SYSTEM_PROMPT,
    assistant_message,
    build_prompt,
    generation_suffix,
    parse_turn,
    render_completion,
    strip_thinking,
    tool_message,
    window_messages,
)
from local_llm_lab.pipeline.tasks import GENERATOR_VERSION, Task, make_tasks


@dataclass(frozen=True)
class SplitSpec:
    """One deterministic logical task chunk and the role output it contributes to."""

    count: int
    difficulty: int | None = None
    perturb: bool | None = None
    role: Literal["train", "valid", "test"] = "train"

    def __post_init__(self) -> None:
        if not isinstance(self.count, int) or isinstance(self.count, bool) or self.count < 1:
            raise ValueError("split count must be a positive integer")
        if self.difficulty is not None and (
            not isinstance(self.difficulty, int)
            or isinstance(self.difficulty, bool)
            or self.difficulty < 0
        ):
            raise ValueError("split difficulty must be a non-negative integer or None")
        if self.perturb is not None and not isinstance(self.perturb, bool):
            raise ValueError("split perturb must be a boolean or None")
        if self.role not in {"train", "valid", "test"}:
            raise ValueError("split role must be train, valid, or test")


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


def render_rows(
    rows: list[dict[str, Any]], tokenizer: Any, *, spec: ModelSpec
) -> list[dict[str, Any]]:
    """Add non-mutating canonical prompt/completion fields for in-process training."""
    training_spec = _training_spec(spec)
    rendered: list[dict[str, Any]] = []
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("rendered row must contain a non-empty messages list")
        target = messages[-1]
        if not isinstance(target, dict) or target.get("role") != "assistant":
            raise ValueError("rendered row must end with an assistant target")
        content = target.get("content")
        if not isinstance(content, str):
            raise ValueError("assistant target content must be a string")
        thinking, remainder = strip_thinking(content)
        turn = parse_turn(remainder)
        completion = render_completion(turn.thought, turn.action, spec=training_spec)
        if thinking is not None and spec.chat.thinking == "trained":
            completion = f"<think>{thinking}</think>\n\n{completion}"
        copy_row = copy.deepcopy(row)
        copy_row["prompt"] = build_prompt(
            tokenizer, messages[:-1], spec=training_spec, generation=True
        )
        copy_row["completion"] = completion
        rendered.append(copy_row)
    return rendered


def _training_spec(spec: ModelSpec) -> ModelSpec:
    """Disable inference-only thinking while retaining explicitly trained reasoning rows."""
    if spec.chat.thinking != "inference":
        return spec
    return replace(
        spec,
        chat=replace(
            spec.chat,
            thinking="off",
            template_kwargs={**spec.chat.template_kwargs, "enable_thinking": False},
        ),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    """Atomically write dataset JSONL through a flushed, same-directory temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_dataset(
    output: Path,
    splits: dict[str, SplitSpec | int],
    *,
    seed: int = 20260902,
    keep_last: int = DEFAULT_KEEP_LAST,
    chat_dir: Path | None = None,
    chat_repeats: int = 1,
    recovery_repeats: int | dict[str, int] = 1,
    extra_dirs: list[Path] | None = None,
    tokenizer: Any | None = None,
    spec: ModelSpec | None = None,
) -> dict[str, Any]:
    """Write role JSONL files from deterministic, declaration-ordered logical chunks.

    ``recovery_repeats`` oversamples corrective decisions in the training split only, either as
    one factor for all of them or as a per-variant mapping. Re-observation recoveries (relist
    after a stale path, re-read after a failed edit) are hosted by only a few families, and the
    run is shorter than one epoch, so without oversampling a given target may never be visited.
    Held-out splits are never reweighted.
    """
    if (tokenizer is None) != (spec is None):
        raise ValueError("tokenizer and spec must be provided together for rendered rows")
    rendering_spec = _training_spec(spec) if spec is not None else None
    manifest: dict[str, Any] = {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "keep_last": keep_last,
        "protocol": "v2-state-carrying-notes-fenced-json",
        "recovery_repeats": recovery_repeats,
        "splits": {},
    }
    if rendering_spec is not None:
        manifest["model"] = asdict(spec)
        manifest["rendering"] = {
            "thinking": rendering_spec.chat.thinking,
            "template_kwargs": rendering_spec.chat.template_kwargs,
            "generation_suffix": generation_suffix(rendering_spec),
        }
    normalized = _normalize_splits(splits)
    role_chunks: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    role_chat_added = {"train": False, "valid": False, "test": False}
    role_extra_added = False
    for split, split_spec in normalized.items():
        role = split_spec.role
        rows, info, added_chat, added_extra = _build_logical_chunk(
            split,
            split_spec,
            seed=seed,
            keep_last=keep_last,
            chat_dir=chat_dir,
            chat_repeats=chat_repeats,
            recovery_repeats=recovery_repeats,
            extra_dirs=extra_dirs,
            tokenizer=tokenizer,
            spec=spec,
            add_chat=not role_chat_added[role],
            add_extra=role == "train" and not role_extra_added,
        )
        role_chat_added[role] = role_chat_added[role] or added_chat
        role_extra_added = role_extra_added or added_extra
        role_chunks[role].extend(rows)
        manifest["splits"][split] = info
    manifest["outputs"] = {}
    for role, rows in role_chunks.items():
        digest = write_jsonl(output / f"{role}.jsonl", rows)
        manifest["outputs"][role] = {
            "rows": len(rows),
            "sha256": digest,
            "chat_rows": sum(
                info["chat_rows"] for info in manifest["splits"].values() if info["role"] == role
            ),
            "extra_rows": sum(
                info["extra_rows"] for info in manifest["splits"].values() if info["role"] == role
            ),
        }
    # A single legacy logical split is also its role output, so keep the historical hash seam.
    for split, info in manifest["splits"].items():
        same_role = [name for name, spec in normalized.items() if spec.role == info["role"]]
        if same_role == [split]:
            info["sha256"] = manifest["outputs"][info["role"]]["sha256"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _build_logical_chunk(
    split: str,
    split_spec: SplitSpec,
    *,
    seed: int,
    keep_last: int,
    chat_dir: Path | None,
    chat_repeats: int,
    recovery_repeats: int | dict[str, int],
    extra_dirs: list[Path] | None,
    tokenizer: Any | None,
    spec: ModelSpec | None,
    add_chat: bool,
    add_extra: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], bool, bool]:
    """Generate one named chunk before its independent deterministic shuffle."""
    tasks = make_tasks(
        split,
        split_spec.count,
        seed,
        perturb=split_spec.perturb,
        difficulty=split_spec.difficulty,
    )
    expert_rows = [
        {
            **row,
            "metadata": {
                **row["metadata"],
                "difficulty": split_spec.difficulty,
                "perturb": split_spec.perturb,
            },
        }
        for task in tasks
        for row in build_rows(task, keep_last=keep_last)
    ]
    recovery_targets = sum(row["metadata"].get("recovery", False) for row in expert_rows)
    if split_spec.role == "train":
        expert_rows = [row for row in expert_rows for _ in range(_repeats(row, recovery_repeats))]
    chat_rows, added_chat = _role_chat_rows(chat_dir, split_spec.role, chat_repeats, add_chat)
    extra_rows = _extra_rows(extra_dirs) if add_extra else []
    rows = [*expert_rows, *chat_rows, *extra_rows]
    if tokenizer is not None and spec is not None:
        rows = render_rows(rows, tokenizer, spec=spec)
    random.Random(f"{seed}:{split}").shuffle(rows)
    horizons = [task.horizon for task in tasks]
    info = {
        "tasks": split_spec.count,
        "count": split_spec.count,
        "difficulty": split_spec.difficulty,
        "perturb": split_spec.perturb,
        "role": split_spec.role,
        "rows": len(rows),
        "expert_rows": len(expert_rows),
        "recovery_targets": recovery_targets,
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
        "sha256": _rows_hash(rows),
    }
    return rows, info, added_chat, add_extra


def _role_chat_rows(
    chat_dir: Path | None, role: str, repeats: int, enabled: bool
) -> tuple[list[dict[str, Any]], bool]:
    if not enabled or chat_dir is None:
        return [], False
    path = chat_dir / f"{role}.jsonl"
    if not path.is_file():
        return [], False
    return read_jsonl(path) * repeats, True


def _extra_rows(extra_dirs: list[Path] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for extra in extra_dirs or []:
        rows.extend(read_jsonl(extra / "train.jsonl"))
    return rows


def _normalize_splits(splits: dict[str, SplitSpec | int]) -> dict[str, SplitSpec]:
    """Accept legacy direct callers while keeping all generation role-aware internally."""
    normalized: dict[str, SplitSpec] = {}
    for name, value in splits.items():
        if isinstance(value, SplitSpec):
            normalized[name] = value
        elif isinstance(value, int) and not isinstance(value, bool):
            role: Literal["train", "valid", "test"] = name if name in {"train", "valid", "test"} else "train"  # type: ignore[assignment]
            normalized[name] = SplitSpec(value, role=role)
        else:
            raise ValueError(f"{name}: split must be a SplitSpec")
    return normalized


def _rows_hash(rows: list[dict[str, Any]]) -> str:
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
