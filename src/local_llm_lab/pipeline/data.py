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
    ActionParseError,
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
from local_llm_lab.project import PROJECT_ROOT
from local_llm_lab.runlog import write_text_atomic

# DEBT(R21): the same non-atomic manifest write survives at three sites owned by other lanes
# this cycle — `pipeline/cli.py:184` (`_write_stage_manifest`, which also stamps the rollout and
# branch outputs at `cli.py:1009` and `cli.py:1224`) and `pipeline/cli.py:703` (health.json) —
# and `probes/state_probe.py:2845` duplicates this helper. Route all four through
# `runlog.write_text_atomic`; see design_specifications/complete/R21-RENDER-GUARD-REPORT.md.

# Ruling R21: these datasets are irreplaceable (runs A and B cannot be regenerated from the
# current source; the generator revision that produced them is unrecoverable). No write path
# may target them, with or without an override flag.
PROTECTED_DATASETS: frozenset[str] = frozenset(
    {
        "data/agent_v2",
        "data/agent_v2b",
        "data/agent_v2c",
        "data/chat_replay",
    }
)

_ROLES = ("train", "valid", "test")

# The only row sources whose assistant target may legally lack a tool call and render
# verbatim. Every data/chat_replay row carries exactly this tag (240/48/60 verified), and the
# ratified R21 round requires an explicit allowlist: any other unparseable target fails
# loudly, because in a controlled arm silent is the failure mode to fear.
CHAT_COMPLETION_SOURCES: frozenset[str] = frozenset({"pre-expansion-policy-replay"})


class DatasetWriteGuardError(RuntimeError):
    """A dataset write was refused because the target already holds a manifest."""


class ProtectedDatasetError(DatasetWriteGuardError):
    """A dataset write targeted an irreplaceable directory; no override exists."""


class DatasetManifestMissingError(RuntimeError):
    """A dataset directory was read before anything stamped it with a manifest."""


class DatasetRenderError(RuntimeError):
    """A render request named a source that is not a readable dataset directory."""


def _names_same_directory(candidate: Path, protected: Path) -> bool:
    """Compare by on-disk identity, because case-insensitive filesystems alias paths.

    ``resolve()`` canonicalises symlinks but preserves the case the caller typed, so on a
    case-insensitive filesystem a mis-cased alias of a protected directory compares unequal
    as a path while naming the same directory on disk. ``samefile`` (st_dev + st_ino) closes
    that hole; plain path equality remains the fallback when either path does not exist yet.
    """
    try:
        return os.path.samefile(candidate, protected)
    except OSError:
        return candidate == protected


def guard_dataset_write(
    output: Path, *, overwrite: bool = False, override_flag: str | None = "--force-overwrite"
) -> bool:
    """Refuse to clobber an existing dataset at the write boundary (ruling R21).

    Returns True when an existing ``manifest.json`` is being replaced under an explicit
    override, so the caller can record that fact in the new manifest. Directories listed in
    ``PROTECTED_DATASETS`` refuse unconditionally — matched by file identity, not path
    spelling, and covering every subpath, because pollution of an irreplaceable directory
    must be impossible, not merely refused at its root. ``override_flag`` names the
    caller's overwrite flag in the refusal message; a stage with no override passes ``None``.
    """
    resolved = Path(output).resolve()
    protected_roots = [
        (name, (PROJECT_ROOT / name).resolve()) for name in sorted(PROTECTED_DATASETS)
    ]
    for candidate in (resolved, *resolved.parents):
        for name, protected in protected_roots:
            if _names_same_directory(candidate, protected):
                raise ProtectedDatasetError(
                    f"refusing to write into {resolved}: it is {name} or lies inside it — "
                    "an irreplaceable dataset (PROTECTED_DATASETS) that can never be "
                    "written to; no override exists. Write to a new directory instead."
                )
    if (resolved / "manifest.json").is_file():
        if not overwrite:
            hint = (
                f"Pass {override_flag} to replace the existing dataset, or choose"
                if override_flag
                else "This stage has no overwrite path; choose"
            )
            raise DatasetWriteGuardError(
                f"refusing to write into {resolved}: it already contains manifest.json. "
                f"{hint} a new output directory."
            )
        return True
    return False


def require_dataset_manifest(directory: Path) -> Path:
    """Refuse to READ a dataset directory that carries no ``manifest.json`` (ruling on #73).

    The manifest is the commit point. Every dataset writer emits its role files first and
    stamps ``manifest.json`` last, through :func:`runlog.write_text_atomic`, so a directory
    holding rows and no manifest is a write that died between the two: the rows look complete
    and their completeness is unknown. ``guard_dataset_write`` deliberately leaves such a
    directory unprotected — a crashed run must be re-runnable without a manual delete — so the
    refusal has to live on this side. Half-written is unusable downstream and overwritable
    upstream, which is the pair the ruling asks for.

    PRESENCE ONLY. This reads whether the file exists and never opens it, so an empty object,
    an unparseable byte or a manifest describing some other dataset all pass. That is the whole
    contract, and test fixtures across the suite lean on it: several stand up a guarded
    directory with a hand-written ``manifest.json`` holding ``{}`` beside rows that came from a
    real writer. A change that starts reading the CONTENTS here breaks every one of them, which
    is the point of saying so — those fixtures must then be rebuilt through a writer that emits
    whatever the new check requires, not padded until the check passes.

    Returns the manifest path, so a caller that goes on to read it need not spell it twice.
    """
    directory = Path(directory)
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        raise DatasetManifestMissingError(
            f"refusing to read {directory}: it holds no manifest.json, so it is either not a "
            "dataset directory or a write that died before its commit point. Re-run the stage "
            "that produces it."
        )
    return manifest


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
    """Add non-mutating canonical prompt/completion fields for in-process training.

    Only rows whose metadata source is in ``CHAT_COMPLETION_SOURCES`` may lack a tool call
    and render their content verbatim; any other unparseable target raises — in a
    controlled arm, silent is the failure mode to fear.
    """
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
        try:
            turn = parse_turn(remainder)
        except ActionParseError as error:
            metadata = row.get("metadata")
            source = metadata.get("source") if isinstance(metadata, dict) else None
            if source not in CHAT_COMPLETION_SOURCES:
                raise ValueError(
                    f"row from source {source!r} has no parseable tool call; only "
                    f"{sorted(CHAT_COMPLETION_SOURCES)} rows may render verbatim"
                ) from error
            # Replayed chat rows end in a plain assistant turn with no tool call; their
            # completion is the verbatim content under the model's turn terminator.
            completion = content + training_spec.chat.end_of_turn + "\n"
        else:
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
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write role JSONL files from deterministic, declaration-ordered logical chunks.

    ``recovery_repeats`` oversamples corrective decisions in the training split only, either as
    one factor for all of them or as a per-variant mapping. Re-observation recoveries (relist
    after a stale path, re-read after a failed edit) are hosted by only a few families, and the
    run is shorter than one epoch, so without oversampling a given target may never be visited.
    Held-out splits are never reweighted.

    The R21 write guard runs before any work: an existing dataset is only replaced under an
    explicit ``overwrite``, recorded in the new manifest, and protected datasets never are.
    """
    overridden = guard_dataset_write(output, overwrite=overwrite)
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
    if overridden:
        manifest["force_overwrite"] = True
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
    write_text_atomic(output / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    return manifest


def render_dataset(
    source: Path,
    output: Path,
    tokenizer: Any,
    *,
    spec: ModelSpec,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Re-render an existing dataset's rows for ``spec`` without regenerating any task (R21).

    Task content (``messages`` and ``metadata``) is carried byte-identical from the source and
    only the ``prompt``/``completion`` rendering changes, so a cross-model training arm stays a
    controlled comparison. The generator is never invoked; the source manifest's
    ``generator_version`` is carried through unchanged (``None`` when the source MANIFEST
    predates generator versioning — a source with no manifest at all is now refused outright,
    ruling on #73), never replaced with the current one.
    """
    source = Path(source).resolve()
    overridden = guard_dataset_write(output, overwrite=overwrite)
    if not source.is_dir():
        raise DatasetRenderError(f"render source {source} is not a directory")
    missing = [role for role in _ROLES if not (source / f"{role}.jsonl").is_file()]
    if missing:
        names = ", ".join(f"{role}.jsonl" for role in missing)
        raise DatasetRenderError(f"render source {source} is missing {names}")
    # The source's own commit point, checked after the role files so a directory that is not a
    # dataset at all still says which file it is missing. All three roles can be present and
    # complete-looking while the write that produced them died before the stamp, and a render
    # of rows of unknown completeness carries that unknowability into a training arm.
    manifest_path = require_dataset_manifest(source)
    rendering_spec = _training_spec(spec)
    source_block: dict[str, Any] = {"directory": str(source)}
    source_manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_block["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    source_block["sha256"] = {}
    rendered_roles: dict[str, list[dict[str, Any]]] = {}
    for role in _ROLES:
        path = source / f"{role}.jsonl"
        source_block["sha256"][role] = hashlib.sha256(path.read_bytes()).hexdigest()
        rendered_roles[role] = render_rows(read_jsonl(path), tokenizer, spec=spec)
    manifest: dict[str, Any] = {
        "stage": "render",
        "generator_version": source_manifest.get("generator_version"),
        "seed": source_manifest.get("seed"),
        "keep_last": source_manifest.get("keep_last"),
        "protocol": source_manifest.get("protocol"),
        "model": asdict(spec),
        "rendering": {
            "thinking": rendering_spec.chat.thinking,
            "template_kwargs": rendering_spec.chat.template_kwargs,
            "generation_suffix": generation_suffix(rendering_spec),
        },
        "source": source_block,
        "outputs": {},
    }
    if overridden:
        manifest["force_overwrite"] = True
    for role, rows in rendered_roles.items():
        digest = write_jsonl(output / f"{role}.jsonl", rows)
        manifest["outputs"][role] = {"rows": len(rows), "sha256": digest}
    output.mkdir(parents=True, exist_ok=True)
    write_text_atomic(output / "manifest.json", json.dumps(manifest, indent=2) + "\n")
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
    # #73: the retention rows come out of a dataset directory `chat_replay.py` stamps last, so
    # this reader owes it the same manifest check as any other. Checked before the per-role
    # file test, because "this role was not generated" and "this whole directory is a crashed
    # write" are different answers and only the first may be absorbed into an empty list.
    require_dataset_manifest(chat_dir)
    path = chat_dir / f"{role}.jsonl"
    if not path.is_file():
        return [], False
    return read_jsonl(path) * repeats, True


def _extra_rows(extra_dirs: list[Path] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for extra in extra_dirs or []:
        # #73: an --extra directory is a rollout output, and `rollout.py` writes train.jsonl
        # before its manifest, so an unstamped one is a sampling run that died mid-write. Its
        # kept rows would otherwise be mixed into training as though the run had finished.
        require_dataset_manifest(extra)
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
