"""P2: the internal environment model, as linear state probes (design §5).

The question is whether the residual stream at the last prompt token -- the position the note
is about to be generated from -- already encodes the task state the note carries, and whether
that encoding is something the adapter created or something the base already had.

Three things make this honest. First, every label comes from the *generator*, never from
parsing the note: the note is the thing under test, so reading labels off it would guarantee a
positive result. Second, every fit is reported next to a shuffled-label control fit on the
same split, and the split is by task id so no task contributes rows to both halves.

Third -- and this is what the first run forced (§5, "Confound found on first run") -- every fit
is also reported next to a **position-only baseline**: a lookup table keyed on
``(family, step index)`` fitted on the same training rows. The generator makes state an almost
exact function of position within a family at a fixed difficulty, so a probe that decodes only
"where am I in the trajectory" scores as if it decoded state. The reportable quantity is the
probe's *margin* over that baseline, never its raw accuracy. Two further tools address the
same confound: ``--mix-difficulty`` builds the dataset from three splits at three difficulties
so the same ``(family, step)`` occurs with different true states, and ``--within-position``
subtracts each cell's mean activation before fitting, which removes everything position can
explain by construction.

The decisive contrast is ``--strip``: the same rows with the notes' state fields removed by
:func:`capture.strip_state_fields`. If the *margin* collapses, the state lived in the note and
the model was reading it; if it survives, the model is holding it internally. (Raw accuracy
would not collapse either way, because position stays decodable after stripping.)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from local_llm_lab.pipeline.data import build_rows
from local_llm_lab.pipeline.protocol import DEFAULT_KEEP_LAST, build_prompt
from local_llm_lab.pipeline.tasks import Task
from local_llm_lab.probes import stats
from local_llm_lab.probes.capture import capture_residuals, strip_state_fields

__all__ = [
    "TARGETS",
    "ProbeDataset",
    "artifact_identity",
    "build_label_dataset",
    "build_probe_dataset",
    "derive_position_keys",
    "fit_probes",
    "load_dataset",
    "main",
    "position_baseline",
    "position_determinism",
    "render_markdown",
    "resolve_within_position",
    "row_labels",
    "save_dataset",
    "set_mlx_cache_limit",
    "split_by_task",
    "task_difficulties",
    "validate_mixed_design",
]

# target -> kind. "regression" is fitted with ridge and scored by R^2/MAE; "categorical" and
# "binary" with (multinomial) logistic regression, scored by accuracy/macro-F1, plus AUC for
# the binary one.
TARGETS: dict[str, str] = {
    "pending_count": "regression",
    "phase": "categorical",
    "prev_error": "binary",
    "next_tool": "categorical",
    "running_max": "regression",
    "first_bucket_count": "regression",
}
CATEGORICAL = {name for name, kind in TARGETS.items() if kind in ("categorical", "binary")}
PHASES = ("inspect", "apply", "verify", "other")
_LOAD_RE = re.compile(r"^load=(\d+)$", re.MULTILINE)
# The three splits --mix-difficulty draws from: train is difficulty 0 throughout, p2mix is a
# rollout-style name so tasks.difficulty alternates 0/1, and test is difficulty 2. train and
# p2mix perturb by default, so they carry the recovery variants; test is generated clean.
MIX_PLAN: tuple[tuple[str, bool | None], ...] = (("train", None), ("p2mix", None), ("test", False))
MIN_WITHIN_CELLS = 5
MIN_WITHIN_TRAIN_ROWS = 48
MIN_WITHIN_TEST_ROWS = 24
DEFAULT_MLX_CACHE_LIMIT_MIB = 512
CHECKPOINT_VERSION = 1


# --------------------------------------------------------------------------- ground truth


def _read_run(names: list[str], index: int) -> tuple[int, int]:
    """Start and stop of the maximal contiguous run of ``read_file`` steps containing ``index``."""
    if names[index] != "read_file":
        return index, index
    start = index
    while start > 0 and names[start - 1] == "read_file":
        start -= 1
    stop = index + 1
    while stop < len(names) and names[stop] == "read_file":
        stop += 1
    return start, stop


def _phase(task: Task, names: list[str], index: int) -> str:
    """``inspect``/``apply``/``verify``/``other`` for one decision.

    Only ``batch_update`` has all three phases, and it marks them structurally: everything
    before the first ``replace_text`` is inspection, the replacements (and any re-read forced
    between them by a failed edit) are the apply phase, and the reads after the last
    replacement are verification. Every other family has no apply/verify distinction, so a read
    inside a read loop counts as inspection and everything else is ``other``.
    """
    if task.family == "batch_update":
        replaces = [position for position, name in enumerate(names) if name == "replace_text"]
        if names[index] == "replace_text":
            return "apply"
        if names[index] != "read_file":
            return "other"
        if not replaces or index < replaces[0]:
            return "inspect"
        return "apply" if index < replaces[-1] else "verify"
    start, stop = _read_run(names, index)
    return "inspect" if names[index] == "read_file" and stop - start >= 2 else "other"


def _previous_observation_failed(task: Task, index: int) -> bool:
    """Replay the simulator over the earlier steps, exactly as :func:`build_rows` does."""
    from local_llm_lab.pipeline.env import Simulator

    simulator = Simulator.for_task(task)
    failed = False
    for step in task.steps[:index]:
        failed = simulator.execute(step.action).startswith("ERROR")
    return failed


def _paths_read_before(task: Task, index: int) -> list[str]:
    seen: list[str] = []
    for step in task.steps[:index]:
        if step.action.name != "read_file":
            continue
        path = str(step.action.arguments.get("path", ""))
        if path and path not in seen:
            seen.append(path)
    return seen


def _running_max(task: Task, index: int) -> float | None:
    """conditional_update: the highest ``load=`` seen in the service files read so far."""
    if task.family != "conditional_update":
        return None
    loads = [
        int(match.group(1))
        for path in _paths_read_before(task, index)
        for match in [_LOAD_RE.search(task.files.get(path, ""))]
        if match
    ]
    return float(max(loads)) if loads else 0.0


def _first_bucket_count(task: Task, index: int) -> float | None:
    """aggregate_report: how many metric values have gone into the first half so far."""
    if task.family != "aggregate_report":
        return None
    metrics = sorted(path for path in task.files if "/metric-" in path)
    split_at = len(metrics) // 2
    read = sum(1 for path in _paths_read_before(task, index) if path in metrics)
    return float(min(read, split_at))


def row_labels(task: Task, step_index: int) -> dict[str, Any]:
    """Ground-truth state at one decision point, taken from the generator.

    Nothing here parses the progress note. ``pending_count`` is the number of ``read_file``
    steps still to come in the current contiguous read loop (which is what the note's
    ``pending:`` list enumerates), 0 outside a read loop; ``running_max`` and
    ``first_bucket_count`` are ``None`` for families that do not have them, so those probes are
    fitted on their own family's rows only.
    """
    if not 0 <= step_index < len(task.steps):
        raise ValueError(f"step {step_index} out of range for {task.task_id}")
    names = [step.action.name for step in task.steps]
    start, stop = _read_run(names, step_index)
    del start
    return {
        "task_id": task.task_id,
        "family": task.family,
        "variant": task.variant,
        "step": step_index,
        "pending_count": float(stop - step_index - 1) if names[step_index] == "read_file" else 0.0,
        "phase": _phase(task, names, step_index),
        "prev_error": float(_previous_observation_failed(task, step_index)),
        "next_tool": names[step_index],
        "running_max": _running_max(task, step_index),
        "first_bucket_count": _first_bucket_count(task, step_index),
    }


# --------------------------------------------------------------------------- dataset


def derive_position_keys(task_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover ``(family, step_index)`` from task ids alone, for ``.npz`` files written before
    those arrays were stored.

    Ids are ``{split}-{family}-{index:04d}-{variant}`` and rows of one task are contiguous, so
    the family is the middle field and the step is the row's order within its task. That order
    is the *supervised row* index rather than the generator's step index (``build_rows`` skips
    unsupervised steps), which is why the real arrays are stored now and this is a fallback.
    """
    families: list[str] = []
    steps: list[int] = []
    seen: dict[str, int] = {}
    for raw in np.asarray(task_ids, dtype=str).tolist():
        task_id = str(raw)
        parts = task_id.split("-")
        families.append("-".join(parts[1:-2]) if len(parts) >= 4 else task_id)
        steps.append(seen.get(task_id, 0))
        seen[task_id] = seen.get(task_id, 0) + 1
    return np.array(families, dtype=str), np.array(steps, dtype=np.int64)


@dataclass
class ProbeDataset:
    """Last-prompt-token activations plus ground-truth labels for a set of supervised rows.

    ``family``, ``step_index`` and ``difficulty`` are per-row and exist so the position-only
    baseline can be fitted: they are the confound's coordinates. When they are not supplied
    (an old ``.npz``, or a synthetic dataset in a test) they are derived from ``task_ids`` and
    the difficulty is recorded as ``-1``, "unknown".
    """

    layers: list[int]
    features: dict[int, np.ndarray]
    labels: dict[str, np.ndarray]
    task_ids: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)
    family: np.ndarray | None = None
    step_index: np.ndarray | None = None
    difficulty: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.task_ids = np.asarray(self.task_ids, dtype=str)
        derived_family, derived_steps = derive_position_keys(self.task_ids)
        self.family = derived_family if self.family is None else np.asarray(self.family, dtype=str)
        self.step_index = (
            derived_steps
            if self.step_index is None
            else np.asarray(self.step_index).astype(np.int64)
        )
        self.difficulty = (
            np.full(len(self.task_ids), -1, dtype=np.int64)
            if self.difficulty is None
            else np.asarray(self.difficulty).astype(np.int64)
        )

    def __len__(self) -> int:
        return len(self.task_ids)

    @property
    def cells(self) -> np.ndarray:
        """The ``(family, step)`` cell of every row, as a string key."""
        family = np.asarray(self.family, dtype=str).tolist()
        steps = np.asarray(self.step_index).astype(np.int64).tolist()
        return np.array(
            [f"{name}|{step}" for name, step in zip(family, steps, strict=True)], dtype=str
        )


def _strip_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {**message, "content": strip_state_fields(message["content"])}
        if message.get("role") == "assistant"
        else dict(message)
        for message in messages
    ]


def task_difficulties(split: str, tasks: list[Task]) -> dict[str, int]:
    """``task_id -> difficulty level`` for tasks made by ``make_tasks(split, ...)``.

    The level comes from :func:`tasks.difficulty` applied to the generator index, which is the
    position in the list ``make_tasks`` returned -- not from parsing the id.
    """
    from local_llm_lab.pipeline.tasks import difficulty

    return {task.task_id: int(difficulty(split, index)) for index, task in enumerate(tasks)}


def build_label_dataset(
    tasks: list[Task], difficulties: dict[str, int] | None = None
) -> ProbeDataset:
    """Build the generator-truth portion of P2 without loading or running a model."""
    labels: dict[str, list[Any]] = {name: [] for name in TARGETS}
    task_ids: list[str] = []
    families: list[str] = []
    step_indices: list[int] = []
    levels: list[int] = []
    for task in tasks:
        for row in build_rows(task):
            step = int(row["metadata"]["step"])
            truth = row_labels(task, step)
            for name in TARGETS:
                labels[name].append(truth[name])
            task_ids.append(task.task_id)
            families.append(task.family)
            step_indices.append(step)
            levels.append(int((difficulties or {}).get(task.task_id, -1)))
    return ProbeDataset(
        layers=[0],
        features={0: np.zeros((len(task_ids), 1), dtype=np.float32)},
        labels={
            name: (
                np.array(
                    [value if value is not None else "" for value in values], dtype=object
                ).astype(str)
                if name in CATEGORICAL
                else np.array(
                    [np.nan if value is None else float(value) for value in values],
                    dtype=np.float64,
                )
            )
            for name, values in labels.items()
        },
        task_ids=np.array(task_ids, dtype=str),
        family=np.array(families, dtype=str),
        step_index=np.array(step_indices, dtype=np.int64),
        difficulty=np.array(levels, dtype=np.int64),
        meta={
            "rows": len(task_ids),
            "tasks": len(tasks),
            "layers": [0],
            "difficulties": {
                str(level): int(count) for level, count in sorted(Counter(levels).items())
            },
        },
    )


def build_probe_dataset(
    model: Any,
    tokenizer: Any,
    tasks: list[Task],
    layers: list[int],
    strip: bool = False,
    *,
    keep_last: int = DEFAULT_KEEP_LAST,
    difficulties: dict[str, int] | None = None,
    progress: Any = None,
    mlx_runtime: Any = None,
    memory_progress: Any = None,
    checkpoint_dir: Path | None = None,
    checkpoint_context: dict[str, Any] | None = None,
) -> ProbeDataset:
    """Capture the last prompt token's residual stream for every supervised row.

    The rows come from :func:`build_rows`, so the context is exactly the windowed context the
    policy sees at inference. Those messages are already windowed, so ``build_prompt`` is
    called with ``keep_last`` large enough to leave them alone (windowing twice would re-stub
    an already-stubbed observation and change the text).

    ``difficulties`` maps task id to difficulty level (see :func:`task_difficulties`); rows of
    tasks it does not mention are recorded as ``-1``.
    """
    from local_llm_lab.pipeline.jlens import _encode

    if mlx_runtime is None:
        import mlx.core as mlx_runtime

    checkpoint_context = dict(checkpoint_context or {})
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    features: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    labels: dict[str, list[Any]] = {name: [] for name in TARGETS}
    task_ids: list[str] = []
    families: list[str] = []
    step_indices: list[int] = []
    levels: list[int] = []
    prompt_token_sum = 0
    max_prompt_tokens = 0
    resumed_tasks = 0
    for number, task in enumerate(tasks, 1):
        difficulty = int((difficulties or {}).get(task.task_id, -1))
        signature = _checkpoint_signature(
            task,
            layers,
            strip=strip,
            keep_last=keep_last,
            difficulty=difficulty,
            context=checkpoint_context,
        )
        checkpoint_path = (
            checkpoint_dir / f"{number:04d}.npz" if checkpoint_dir is not None else None
        )
        part: ProbeDataset | None = None
        resumed = False
        try:
            if checkpoint_path is not None and checkpoint_path.is_file():
                part = load_dataset(checkpoint_path)
                if part.meta.get("checkpoint") != signature:
                    raise ValueError(
                        f"checkpoint {checkpoint_path} does not match this capture; "
                        "use a fresh output directory"
                    )
                resumed = True
                resumed_tasks += 1
            else:
                mlx_runtime.reset_peak_memory()
                task_features: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
                task_labels: dict[str, list[Any]] = {name: [] for name in TARGETS}
                task_ids_part: list[str] = []
                families_part: list[str] = []
                step_indices_part: list[int] = []
                levels_part: list[int] = []
                task_lengths: list[int] = []
                for row in build_rows(task, keep_last=keep_last):
                    context = row["messages"][:-1]
                    if strip:
                        context = _strip_messages(context)
                    prompt = build_prompt(tokenizer, context, keep_last=len(context))
                    token_ids = _encode(tokenizer, prompt)
                    task_lengths.append(len(token_ids))
                    captured = capture_residuals(model, token_ids, layers, positions="last")
                    materialized = _materialize_residuals(captured, layers, mlx_runtime)
                    del captured
                    for layer in layers:
                        task_features[layer].append(materialized[layer])
                    step = int(row["metadata"]["step"])
                    truth = row_labels(task, step)
                    for name in TARGETS:
                        task_labels[name].append(truth[name])
                    task_ids_part.append(task.task_id)
                    families_part.append(task.family)
                    step_indices_part.append(step)
                    levels_part.append(difficulty)
                part = ProbeDataset(
                    layers=list(layers),
                    features={layer: np.stack(values) for layer, values in task_features.items()},
                    labels=_label_arrays(task_labels),
                    task_ids=np.array(task_ids_part, dtype=str),
                    family=np.array(families_part, dtype=str),
                    step_index=np.array(step_indices_part, dtype=np.int64),
                    difficulty=np.array(levels_part, dtype=np.int64),
                    meta={
                        "rows": len(task_ids_part),
                        "tasks": 1,
                        "layers": list(layers),
                        "strip": bool(strip),
                        "keep_last": keep_last,
                        "prompt_token_sum": int(sum(task_lengths)),
                        "mean_prompt_tokens": (
                            float(np.mean(task_lengths)) if task_lengths else 0.0
                        ),
                        "max_prompt_tokens": int(np.max(task_lengths)) if task_lengths else 0,
                        "checkpoint": signature,
                    },
                )
                if checkpoint_path is not None:
                    save_dataset(part, checkpoint_path)
        finally:
            mlx_runtime.clear_cache()

        assert part is not None
        for layer in layers:
            features[layer].extend(np.asarray(part.features[layer], dtype=np.float32))
        for name in TARGETS:
            labels[name].extend(np.asarray(part.labels[name]).tolist())
        task_ids.extend(np.asarray(part.task_ids, dtype=str).tolist())
        families.extend(np.asarray(part.family, dtype=str).tolist())
        step_indices.extend(np.asarray(part.step_index, dtype=np.int64).tolist())
        levels.extend(np.asarray(part.difficulty, dtype=np.int64).tolist())
        prompt_token_sum += int(
            part.meta.get(
                "prompt_token_sum", round(float(part.meta.get("mean_prompt_tokens", 0)) * len(part))
            )
        )
        max_prompt_tokens = max(max_prompt_tokens, int(part.meta.get("max_prompt_tokens", 0)))
        memory = mlx_memory_snapshot(mlx_runtime)
        if resumed:
            # A resumed task does no MLX work, so the runtime's process-wide peak belongs to
            # an earlier capture and would be misleading if attributed to this shard.
            memory["peak_bytes"] = None
        memory.update(
            {
                "task": number,
                "task_id": task.task_id,
                "rows": len(task_ids),
                "resumed": resumed,
            }
        )
        if memory_progress is not None:
            memory_progress(memory)
        if progress is not None:
            progress(number, len(tasks), len(task_ids))
    return ProbeDataset(
        layers=list(layers),
        features={layer: np.stack(values) for layer, values in features.items()},
        labels=_label_arrays(labels),
        task_ids=np.array(task_ids, dtype=str),
        family=np.array(families, dtype=str),
        step_index=np.array(step_indices, dtype=np.int64),
        difficulty=np.array(levels, dtype=np.int64),
        meta={
            "rows": len(task_ids),
            "tasks": len(tasks),
            "layers": list(layers),
            "strip": bool(strip),
            "keep_last": keep_last,
            "checkpointed_tasks": len(tasks) if checkpoint_dir is not None else 0,
            "resumed_tasks": resumed_tasks,
            "difficulties": {
                str(level): int(count) for level, count in sorted(Counter(levels).items())
            },
            "mean_prompt_tokens": float(prompt_token_sum / len(task_ids)) if task_ids else 0.0,
            "max_prompt_tokens": max_prompt_tokens,
        },
    )


def _label_arrays(labels: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    return {
        name: (
            np.array([value if value is not None else "" for value in values], dtype=object).astype(
                str
            )
            if name in CATEGORICAL
            else np.array(
                [np.nan if value is None else float(value) for value in values],
                dtype=np.float64,
            )
        )
        for name, values in labels.items()
    }


def _materialize_residuals(
    captured: dict[int, Any], layers: list[int], mlx_runtime: Any
) -> dict[int, np.ndarray]:
    values = [captured[layer] for layer in layers]
    mlx_runtime.eval(*values)
    return {
        layer: np.array(value, dtype=np.float32, copy=True)
        for layer, value in zip(layers, values, strict=True)
    }


def mlx_memory_snapshot(mlx_runtime: Any) -> dict[str, int]:
    return {
        "active_bytes": int(mlx_runtime.get_active_memory()),
        "cache_bytes": int(mlx_runtime.get_cache_memory()),
        "peak_bytes": int(mlx_runtime.get_peak_memory()),
    }


def set_mlx_cache_limit(mlx_runtime: Any, limit_mib: int) -> int:
    if limit_mib <= 0:
        raise ValueError("MLX cache limit must be positive")
    return int(mlx_runtime.set_cache_limit(int(limit_mib) * 2**20))


def _canonical_value(value: Any) -> Any:
    """Convert task metadata to a deterministic, JSON-serialisable representation."""
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical_value(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _canonical_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (set, frozenset)):
        converted = [_canonical_value(item) for item in value]
        return sorted(
            converted,
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ),
        )
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def artifact_identity(reference: str | Path | None) -> str | None:
    """Return a stable identity for local weights or a cached Hugging Face snapshot.

    Checkpoint reuse is safe only when the exact model and adapter artifacts are unchanged.
    Local artifacts are content-hashed. Model repository names resolve to the immutable cached
    snapshot selected by the loader, without permitting a network fetch here.
    """
    if reference is None:
        return None
    path = Path(reference).expanduser()
    if path.exists():
        root = path if path.is_dir() else path.parent
        files = sorted(
            (candidate for candidate in path.rglob("*") if candidate.is_file())
            if path.is_dir()
            else [path]
        )
        digest = hashlib.sha256()
        for candidate in files:
            relative = candidate.relative_to(root).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            with candidate.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"

    from huggingface_hub import scan_cache_dir

    try:
        cache = scan_cache_dir()
    except Exception as error:
        raise RuntimeError(
            f"cannot inspect the local Hugging Face cache for {reference!s}"
        ) from error
    revisions = [
        revision
        for repository in cache.repos
        if repository.repo_type == "model" and repository.repo_id == str(reference)
        for revision in repository.revisions
    ]
    if not revisions:
        raise RuntimeError(f"loaded model artifact {reference!s} is absent from the local cache")
    # The loader's unqualified repository reference resolves through `main`. Falling back to
    # the most recently modified cached revision covers older cache layouts that lack refs.
    selected = max(
        revisions,
        key=lambda revision: (
            "main" in revision.refs,
            float(getattr(revision, "last_modified", 0.0)),
            revision.commit_hash,
        ),
    )
    return f"hf:{reference}@{selected.commit_hash}"


def _checkpoint_signature(
    task: Task,
    layers: list[int],
    *,
    strip: bool,
    keep_last: int,
    difficulty: int,
    context: dict[str, Any],
) -> dict[str, Any]:
    task_payload = json.dumps(
        _canonical_value(task), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return {
        "version": CHECKPOINT_VERSION,
        "task_id": task.task_id,
        "task_sha256": hashlib.sha256(task_payload.encode("utf-8")).hexdigest(),
        "layers": list(layers),
        "strip": bool(strip),
        "keep_last": int(keep_last),
        "difficulty": int(difficulty),
        "context": context,
    }


def save_dataset(dataset: ProbeDataset, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        f"layer_{layer}": dataset.features[layer] for layer in dataset.layers
    }
    payload.update({f"label_{name}": values for name, values in dataset.labels.items()})
    payload["task_ids"] = dataset.task_ids
    payload["family"] = np.asarray(dataset.family, dtype=str)
    payload["step_index"] = np.asarray(dataset.step_index).astype(np.int64)
    payload["difficulty"] = np.asarray(dataset.difficulty).astype(np.int64)
    payload["meta"] = np.array(json.dumps(dataset.meta))
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def load_dataset(path: Path) -> ProbeDataset:
    """Load a captured dataset, tolerating files written before the per-row position arrays."""
    with np.load(path, allow_pickle=False) as handle:
        meta = json.loads(str(handle["meta"]))
        layers = list(meta["layers"])
        has_position = "family" in handle and "step_index" in handle
        has_difficulty = "difficulty" in handle
        if not has_position:
            warnings.warn(
                f"{path} has no per-row family/step_index arrays (captured before the "
                "position-only baseline existed); deriving them from the task ids, so the "
                "step index is the supervised-row index rather than the generator's step.",
                stacklevel=2,
            )
        return ProbeDataset(
            layers=layers,
            features={layer: handle[f"layer_{layer}"] for layer in layers},
            labels={name: handle[f"label_{name}"] for name in TARGETS if f"label_{name}" in handle},
            task_ids=handle["task_ids"],
            meta=meta,
            family=handle["family"] if has_position else None,
            step_index=handle["step_index"] if has_position else None,
            difficulty=handle["difficulty"] if has_difficulty else None,
        )


# --------------------------------------------------------------------------- fitting


def split_by_task(
    task_ids: np.ndarray,
    seed: int,
    train_fraction: float = 0.7,
    *,
    difficulty: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Row indices for a train/test split that never puts one task on both sides.

    With ``difficulty`` given, the cut is taken *within* each difficulty level, so every level
    is represented on both sides; otherwise a mixed-difficulty dataset could put all of the
    hard tasks in one half and the position baseline would be measured on the wrong thing.
    """
    task_ids = np.asarray(task_ids, dtype=str)
    levels: dict[str, int] = {}
    if difficulty is not None:
        for task_id, level in zip(task_ids.tolist(), np.asarray(difficulty).tolist(), strict=True):
            levels.setdefault(str(task_id), int(level))
    rng = np.random.default_rng(seed)
    train_tasks: set[str] = set()
    unique = sorted(set(task_ids.tolist()))
    groups = defaultdict(list)
    for task_id in unique:
        groups[levels.get(task_id, 0)].append(task_id)
    for level in sorted(groups):
        members = np.array(groups[level])
        order = rng.permutation(len(members))
        cut = max(1, int(round(train_fraction * len(members))))
        train_tasks.update(members[order[:cut]].tolist())
    is_train = np.array([task_id in train_tasks for task_id in task_ids.tolist()])
    return np.flatnonzero(is_train), np.flatnonzero(~is_train)


def _standardise(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    return (train - mean) / scale, (other - mean) / scale


def ridge_fit(features: np.ndarray, targets: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge with an unpenalised bias, solved in whichever form is smaller.

    With ``n < d`` (the usual case here: a few hundred rows against 2048 dimensions) the dual
    form ``w = X'(XX' + aI)^-1 y`` costs ``n^3`` instead of ``d^3``.
    """
    n, d = features.shape
    centre = targets.mean()
    centred = targets - centre
    if n < d:
        gram = features @ features.T + alpha * np.eye(n)
        weights = features.T @ np.linalg.solve(gram, centred)
    else:
        gram = features.T @ features + alpha * np.eye(d)
        weights = np.linalg.solve(gram, features.T @ centred)
    return np.concatenate([weights, [centre]])


def _predict_linear(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return features @ weights[:-1] + weights[-1]


def _spectral_norm(matrix: np.ndarray, iterations: int = 20, seed: int = 0) -> float:
    """Largest singular value by power iteration, for a safe gradient-descent step size."""
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=matrix.shape[1])
    vector /= np.linalg.norm(vector) + 1e-12
    for _ in range(iterations):
        vector = matrix.T @ (matrix @ vector)
        norm = np.linalg.norm(vector)
        if norm < 1e-12:
            return 0.0
        vector /= norm
    return float(np.sqrt(norm))


def logistic_fit(
    features: np.ndarray,
    classes: np.ndarray,
    n_classes: int,
    *,
    l2: float = 0.01,
    steps: int = 400,
) -> np.ndarray:
    """Multinomial logistic regression by full-batch gradient descent with L2.

    The loss is convex and its gradient is Lipschitz with constant ``0.5 * s_max^2 / n + l2``
    (``s_max`` the largest singular value of the design matrix), so a step size of one over
    that constant converges without any tuning or line search -- which is why the step size is
    computed rather than guessed.
    """
    n, d = features.shape
    design = np.hstack([features, np.ones((n, 1))])
    onehot = np.zeros((n, n_classes))
    onehot[np.arange(n), classes] = 1.0
    weights = np.zeros((d + 1, n_classes))
    lipschitz = 0.5 * _spectral_norm(design) ** 2 / n + l2
    step = 1.0 / max(lipschitz, 1e-8)
    penalty = np.ones((d + 1, 1))
    penalty[-1, 0] = 0.0  # never regularise the bias
    for _ in range(steps):
        scores = design @ weights
        scores -= scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        gradient = design.T @ (probabilities - onehot) / n + l2 * penalty * weights
        weights -= step * gradient
    return weights


def _predict_logits(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    scores = np.hstack([features, np.ones((len(features), 1))]) @ weights
    scores -= scores.max(axis=1, keepdims=True)
    probabilities = np.exp(scores)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _majority(labels: Any) -> Any:
    """The commonest label, ties broken by the label's string order (so it is deterministic)."""
    counts = Counter(np.asarray(labels).tolist())
    return min(counts.items(), key=lambda item: (-item[1], str(item[0])))[0]


def _score(
    kind: str,
    predicted: np.ndarray,
    actual: np.ndarray,
    *,
    classes: np.ndarray | None = None,
    positive_scores: np.ndarray | None = None,
    majority: Any = None,
) -> dict[str, float]:
    """The metric dict every fit (probe, control, baseline) reports, for one set of rows."""
    if kind == "regression":
        return {"r2": stats.r2(predicted, actual), "mae": stats.mae(predicted, actual)}
    assert classes is not None
    result = {
        "accuracy": stats.accuracy(predicted, actual),
        "macro_f1": stats.macro_f1(predicted, actual, classes),
    }
    if majority is not None:
        result["majority_baseline"] = float(np.mean(actual == majority))
    if kind == "binary" and len(classes) == 2 and positive_scores is not None:
        result["auc"] = stats.auc(positive_scores, (actual == classes[-1]).astype(int))
    return result


def _grouped(
    kind: str,
    predicted: np.ndarray,
    actual: np.ndarray,
    groups: np.ndarray,
    **score_kwargs: Any,
) -> dict[str, dict[str, float]]:
    """The same metrics recomputed inside each group (used for the difficulty breakdown)."""
    out: dict[str, dict[str, float]] = {}
    positive = score_kwargs.pop("positive_scores", None)
    for level in sorted(set(np.asarray(groups).tolist())):
        mask = np.asarray(groups) == level
        if not mask.any():
            continue
        out[str(level)] = {
            **_score(
                kind,
                predicted[mask],
                actual[mask],
                positive_scores=None if positive is None else positive[mask],
                **score_kwargs,
            ),
            "n": int(mask.sum()),
        }
    return out


def _fit_one(
    kind: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    alpha: float,
    l2: float,
    steps: int,
    groups: np.ndarray | None = None,
    subset: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit one probe and score it. ``subset`` restricts *evaluation* to some test rows."""
    keep = np.ones(len(test_y), dtype=bool) if subset is None else np.asarray(subset, dtype=bool)
    grouping = None if groups is None else np.asarray(groups)[keep]
    if kind == "regression":
        weights = ridge_fit(train_x, train_y.astype(np.float64), alpha)
        predicted = _predict_linear(test_x, weights)[keep]
        actual = test_y.astype(np.float64)[keep]
        result: dict[str, Any] = _score(kind, predicted, actual)
        if grouping is not None:
            result["by_difficulty"] = _grouped(kind, predicted, actual, grouping)
        return result
    classes = np.array(sorted(set(train_y.tolist())))
    lookup = {label: index for index, label in enumerate(classes)}
    train_codes = np.array([lookup[label] for label in train_y])
    weights = logistic_fit(train_x, train_codes, len(classes), l2=l2, steps=steps)
    probabilities = _predict_logits(test_x, weights)[keep]
    predicted = classes[np.argmax(probabilities, axis=1)]
    actual = test_y[keep]
    positive = probabilities[:, list(classes).index(classes[-1])] if len(classes) == 2 else None
    majority = _majority(train_y)
    result = _score(
        kind, predicted, actual, classes=classes, positive_scores=positive, majority=majority
    )
    if grouping is not None:
        result["by_difficulty"] = _grouped(
            kind,
            predicted,
            actual,
            grouping,
            classes=classes,
            positive_scores=positive,
            majority=majority,
        )
    return result


# --------------------------------------------------------------------- the position confound


def _defined(values: np.ndarray, target: str) -> np.ndarray:
    """Rows whose label exists for this target (families without it are dropped per target)."""
    if target in CATEGORICAL:
        return np.asarray(values) != ""
    return ~np.isnan(np.asarray(values).astype(np.float64))


def position_baseline(
    dataset: ProbeDataset,
    target: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    """A lookup table on ``(family, step index)``, fitted and scored on the same split.

    This is the control the first run forced (§5). It never sees an activation: it predicts the
    training mean (regression) or majority class (categorical) of the rows sharing the test
    row's ``(family, step)`` cell, falling back to the global training mean or majority when the
    cell is unseen. If the activation probe does not beat this, the probe has decoded position,
    not state.

    ``train_idx`` and ``test_idx`` are row indices into ``dataset``; ``groups`` (one value per
    test row) adds a breakdown, and is used for difficulty.
    """
    kind = TARGETS[target]
    cells = dataset.cells
    values = dataset.labels[target]
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    train_keys = cells[train_idx].tolist()
    test_keys = cells[test_idx].tolist()
    train_y, test_y = values[train_idx], values[test_idx]
    if kind == "regression":
        train_y = train_y.astype(np.float64)
        test_y = test_y.astype(np.float64)
        sums: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for key, value in zip(train_keys, train_y.tolist(), strict=True):
            sums[key] += value
            counts[key] += 1
        fallback = float(train_y.mean()) if len(train_y) else 0.0
        cell_mean = {key: sums[key] / counts[key] for key in sums}
        predicted = np.array([cell_mean.get(key, fallback) for key in test_keys])
        result: dict[str, Any] = _score(kind, predicted, test_y)
        result["unseen_cells"] = int(sum(1 for key in test_keys if key not in cell_mean))
        if groups is not None:
            result["by_difficulty"] = _grouped(kind, predicted, test_y, np.asarray(groups))
        return result
    classes = np.array(sorted(set(train_y.tolist())))
    tallies: dict[str, Counter] = defaultdict(Counter)
    for key, label in zip(train_keys, train_y.tolist(), strict=True):
        tallies[key][label] += 1
    majority = _majority(train_y)
    cell_vote = {key: _majority(list(counter.elements())) for key, counter in tallies.items()}
    predicted = np.array([cell_vote.get(key, majority) for key in test_keys])
    positive = None
    if kind == "binary" and len(classes) == 2:
        share = {
            key: counter[classes[-1]] / sum(counter.values()) for key, counter in tallies.items()
        }
        default = float(np.mean(train_y == classes[-1]))
        positive = np.array([share.get(key, default) for key in test_keys])
    result = _score(
        kind, predicted, test_y, classes=classes, positive_scores=positive, majority=majority
    )
    result["unseen_cells"] = int(sum(1 for key in test_keys if key not in cell_vote))
    if groups is not None:
        result["by_difficulty"] = _grouped(
            kind,
            predicted,
            test_y,
            np.asarray(groups),
            classes=classes,
            positive_scores=positive,
            majority=majority,
        )
    return result


def position_determinism(dataset: ProbeDataset, targets: list[str]) -> dict[str, dict[str, float]]:
    """How often ``(family, step)`` already fixes the label, per target.

    A fraction near 1.0 means the dataset *is* the confound: knowing where you are in the
    trajectory is knowing the state, and no probe on it can be interpreted. Mixing difficulties
    (``--mix-difficulty``) is what pushes this down.
    """
    cells = dataset.cells
    out: dict[str, dict[str, float]] = {}
    for target in targets:
        values = dataset.labels.get(target)
        if values is None:
            continue
        rows = np.flatnonzero(_defined(values, target))
        buckets: dict[str, list[Any]] = defaultdict(list)
        for row in rows.tolist():
            buckets[str(cells[row])].append(values[row])
        constant = 0
        multi = 0
        multi_constant = 0
        within_error = 0.0
        matched = 0
        for labels in buckets.values():
            if TARGETS[target] == "regression":
                numbers = np.asarray(labels, dtype=np.float64)
                is_constant = bool(float(numbers.max() - numbers.min()) <= 1e-12)
                within_error += float(np.sum((numbers - numbers.mean()) ** 2))
            else:
                is_constant = len(set(np.asarray(labels).tolist())) == 1
                matched += int(np.sum(np.asarray(labels) == _majority(labels)))
            constant += int(is_constant)
            if len(labels) > 1:
                multi += 1
                multi_constant += int(is_constant)
        out[target] = {
            "rows": int(len(rows)),
            "cells": int(len(buckets)),
            "constant_fraction": float(constant / len(buckets)) if buckets else float("nan"),
            "multi_row_cells": int(multi),
            "multi_row_constant_fraction": float(multi_constant / multi) if multi else float("nan"),
            "explained": _explained(target, values[rows], within_error, matched),
        }
    return out


def _explained(target: str, values: np.ndarray, within_error: float, matched: int) -> float:
    """How much of the label the cell alone accounts for, in sample.

    For regression this is the variance share ``1 - SS_within / SS_total``; for classes it is
    the accuracy of predicting each row's own cell's majority. A cell can fail the
    constant-label test and still leave almost nothing over, which is exactly the case for
    ``pending_count``, so this is the number that sizes the confound.
    """
    if len(values) == 0:
        return float("nan")
    if TARGETS[target] == "regression":
        numbers = np.asarray(values, dtype=np.float64)
        total = float(np.sum((numbers - numbers.mean()) ** 2))
        return float(1.0 - within_error / total) if total > 0 else float("nan")
    return float(matched / len(values)) if len(values) else float("nan")


def _cell_centre_features(
    features: np.ndarray, keys: list[str], train_rows: np.ndarray
) -> np.ndarray:
    """Subtract each row's ``(family, step)`` cell mean, computed on training rows only."""
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = defaultdict(int)
    for row in np.asarray(train_rows).tolist():
        key = keys[row]
        sums[key] = features[row] if key not in sums else sums[key] + features[row]
        counts[key] += 1
    fallback = features[train_rows].mean(axis=0)
    means = {key: sums[key] / counts[key] for key in sums}
    return features - np.stack([means.get(key, fallback) for key in keys])


def _cell_centre_targets(values: np.ndarray, keys: list[str], train_rows: np.ndarray) -> np.ndarray:
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    numbers = np.asarray(values, dtype=np.float64)
    for row in np.asarray(train_rows).tolist():
        sums[keys[row]] += float(numbers[row])
        counts[keys[row]] += 1
    fallback = float(numbers[train_rows].mean()) if len(train_rows) else 0.0
    means = {key: sums[key] / counts[key] for key in sums}
    return numbers - np.array([means.get(key, fallback) for key in keys])


def _varying_training_cells(
    values: np.ndarray,
    keys: list[str],
    train_rows: np.ndarray,
    kind: str,
) -> set[str]:
    """Training cells with genuine within-cell target variation.

    Eligibility is learned from the training half only. Looking at held-out labels to decide
    which cells to score would leak test information into the analysis; accepting constant
    training cells would dilute the result with rows whose state is fixed by position.
    """
    seen: dict[str, list[Any]] = defaultdict(list)
    for row in np.asarray(train_rows).tolist():
        seen[keys[row]].append(values[row])
    varying: set[str] = set()
    for key, labels in seen.items():
        if len(labels) < 2:
            continue
        if kind == "regression":
            numbers = np.asarray(labels, dtype=np.float64)
            if float(numbers.max() - numbers.min()) > 1e-12:
                varying.add(key)
        elif len(set(np.asarray(labels).tolist())) > 1:
            varying.add(key)
    return varying


def validate_mixed_design(
    dataset: ProbeDataset,
    targets: list[str],
    seed: int = 20260903,
    *,
    min_cells: int = MIN_WITHIN_CELLS,
    min_train_rows: int = MIN_WITHIN_TRAIN_ROWS,
    min_test_rows: int = MIN_WITHIN_TEST_ROWS,
) -> dict[str, Any]:
    """Gate a mixed dataset on leakage-free within-position support for every target.

    Raw labels can remain highly predictable from position even after difficulty mixing because
    many task states are structurally positional. The run is usable only when every requested
    target has enough *training-identified* varying cells and held-out rows for the residualised
    analysis that removes that remaining signal.
    """
    levels = sorted({int(level) for level in np.asarray(dataset.difficulty).tolist()})
    failures: list[str] = []
    if levels != [0, 1, 2]:
        failures.append(f"difficulty levels are {levels}, expected [0, 1, 2]")
    cells = dataset.cells
    support: dict[str, dict[str, int]] = {}
    for target in targets:
        values = dataset.labels[target]
        rows = np.flatnonzero(_defined(values, target))
        if not len(rows):
            failures.append(f"{target}: no labelled rows")
            support[target] = {
                "eligible_cells": 0,
                "n_train": 0,
                "n_test": 0,
                "excluded_test_rows": 0,
            }
            continue
        selected = values[rows]
        keys = cells[rows].tolist()
        train_rows, test_rows = split_by_task(
            dataset.task_ids[rows], seed, difficulty=np.asarray(dataset.difficulty)[rows]
        )
        eligible = _varying_training_cells(selected, keys, train_rows, TARGETS[target])
        n_train = int(sum(keys[row] in eligible for row in train_rows.tolist()))
        n_test = int(sum(keys[row] in eligible for row in test_rows.tolist()))
        support[target] = {
            "eligible_cells": len(eligible),
            "n_train": n_train,
            "n_test": n_test,
            "excluded_test_rows": int(len(test_rows) - n_test),
        }
        if len(eligible) < min_cells:
            failures.append(f"{target}: only {len(eligible)} varying training cells")
        if n_train < min_train_rows:
            failures.append(f"{target}: only {n_train} eligible training rows")
        if n_test < min_test_rows:
            failures.append(f"{target}: only {n_test} eligible test rows")
    return {
        "status": "PASS" if not failures else "FAIL",
        "difficulty_levels": levels,
        "thresholds": {
            "min_cells": min_cells,
            "min_train_rows": min_train_rows,
            "min_test_rows": min_test_rows,
        },
        "targets": support,
        "failures": failures,
    }


# --------------------------------------------------------------------------- the fit


def fit_probes(
    dataset: ProbeDataset,
    targets: list[str],
    layers: list[int],
    seed: int = 20260903,
    *,
    alpha: float = 10.0,
    l2: float = 0.01,
    steps: int = 400,
    train_fraction: float = 0.7,
    within_position: bool = False,
) -> dict[str, Any]:
    """Fit one probe per (target, layer) beside two controls on the same split.

    ``alpha`` and ``l2`` are the ridge and logistic penalties on standardised features; the
    logistic default matches the effective strength of scikit-learn's ``C=1`` at these sample
    sizes, and the shuffled control is what says whether either is letting the fit overfit.

    The second control is the position-only baseline, and ``margin`` (probe minus baseline on
    the headline metric) is the number to report. With ``within_position`` a third fit is added
    per layer, on activations with their ``(family, step)`` cell mean removed and, for
    regression, targets centred the same way. Both regression and categorical targets are
    trained and evaluated only in cells that contain target variation in the training half;
    eligibility never consults held-out labels, and unseen test cells are excluded explicitly.

    Rows whose label is undefined for their family (``running_max`` outside
    ``conditional_update``, ``first_bucket_count`` outside ``aggregate_report``) are dropped
    for that target only. A target with a single class in the training half is skipped and says
    so, rather than reporting a meaningless perfect score.
    """
    results: dict[str, Any] = {
        "seed": seed,
        "meta": dataset.meta,
        "within_position": bool(within_position),
        "position_determinism": position_determinism(dataset, targets),
        "targets": {},
    }
    cells = dataset.cells
    levels = np.asarray(dataset.difficulty).astype(np.int64)
    rng = np.random.default_rng(seed + 1)
    for target in targets:
        kind = TARGETS[target]
        values = dataset.labels[target]
        rows = np.flatnonzero(_defined(values, target))
        entry: dict[str, Any] = {
            "kind": kind,
            "rows": int(len(rows)),
            "position_determinism": results["position_determinism"].get(target),
            "layers": {},
        }
        if len(rows) < 8:
            entry["skipped"] = "fewer than 8 labelled rows"
            results["targets"][target] = entry
            continue
        selected = values[rows]
        shuffled = selected[rng.permutation(len(rows))]
        train_rows, test_rows = split_by_task(
            dataset.task_ids[rows], seed, train_fraction, difficulty=levels[rows]
        )
        if len(train_rows) == 0 or len(test_rows) == 0:
            entry["skipped"] = "the task split left one side empty"
            results["targets"][target] = entry
            continue
        if kind != "regression" and len(set(selected[train_rows].tolist())) < 2:
            entry["skipped"] = "only one class in the training half"
            results["targets"][target] = entry
            continue
        entry["classes"] = sorted(set(selected.tolist())) if kind != "regression" else None
        entry["n_train"] = int(len(train_rows))
        entry["n_test"] = int(len(test_rows))
        test_levels = levels[rows][test_rows]
        known = sorted({level for level in test_levels.tolist() if level >= 0})
        groups = test_levels if len(known) > 1 else None
        entry["position_baseline"] = position_baseline(
            dataset, target, rows[train_rows], rows[test_rows], groups=groups
        )
        headline = _headline_key(kind, entry["position_baseline"])
        keys = cells[rows].tolist()
        for layer in layers:
            features = dataset.features[layer][rows].astype(np.float64)
            train_x, test_x = _standardise(features[train_rows], features[test_rows])
            fit = _fit_one(
                kind,
                train_x,
                selected[train_rows],
                test_x,
                selected[test_rows],
                alpha=alpha,
                l2=l2,
                steps=steps,
                groups=groups,
            )
            control = _fit_one(
                kind,
                train_x,
                shuffled[train_rows],
                test_x,
                shuffled[test_rows],
                alpha=alpha,
                l2=l2,
                steps=steps,
            )
            layer_entry: dict[str, Any] = {"probe": fit, "shuffled_control": control}
            layer_entry["margin"] = float(fit[headline] - entry["position_baseline"][headline])
            if within_position:
                layer_entry["within_position"] = _within_position_fit(
                    kind,
                    features,
                    keys,
                    selected,
                    train_rows,
                    test_rows,
                    alpha=alpha,
                    l2=l2,
                    steps=steps,
                )
            entry["layers"][str(layer)] = layer_entry
        results["targets"][target] = entry
    return results


def _within_position_fit(
    kind: str,
    features: np.ndarray,
    keys: list[str],
    selected: np.ndarray,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    *,
    alpha: float,
    l2: float,
    steps: int,
) -> dict[str, Any]:
    """Fit after removing each training cell's mean, without consulting test labels.

    Only cells whose *training rows* contain target variation are eligible. Held-out rows from
    unseen or constant training cells are excluded: their training mean is unknowable or leaves
    no state residual to decode, so including them would reintroduce position or merely inflate
    the denominator with guaranteed zeros.
    """
    eligible = _varying_training_cells(selected, keys, train_rows, kind)
    within_train = np.array(
        [row for row in train_rows.tolist() if keys[row] in eligible], dtype=np.int64
    )
    within_test = np.array(
        [row for row in test_rows.tolist() if keys[row] in eligible], dtype=np.int64
    )
    support = {
        "eligible_cells": int(len(eligible)),
        "n_train": int(len(within_train)),
        "n_test": int(len(within_test)),
        "excluded_test_rows": int(len(test_rows) - len(within_test)),
    }
    if not eligible:
        return {**support, "skipped": "no training cell has within-position target variation"}
    if len(within_train) < 8 or len(within_test) < 2:
        return {**support, "skipped": "insufficient train/test rows in varying training cells"}

    centred = _cell_centre_features(features, keys, train_rows)
    train_x, test_x = _standardise(centred[within_train], centred[within_test])
    if kind == "regression":
        centred_y = _cell_centre_targets(selected, keys, train_rows)
        result = _fit_one(
            kind,
            train_x,
            centred_y[within_train],
            test_x,
            centred_y[within_test],
            alpha=alpha,
            l2=l2,
            steps=steps,
        )
        return {**result, **support}
    if len(set(selected[within_train].tolist())) < 2:
        return {**support, "skipped": "only one class remains in varying training cells"}
    result = _fit_one(
        kind,
        train_x,
        selected[within_train],
        test_x,
        selected[within_test],
        alpha=alpha,
        l2=l2,
        steps=steps,
    )
    return {**result, **support}


# --------------------------------------------------------------------------- rendering


_HEADLINE = {"regression": "r2", "categorical": "accuracy", "binary": "auc"}


def _headline_key(kind: str, metrics: dict[str, Any]) -> str:
    """The metric a table leads with, falling back when AUC is undefined for a binary fit."""
    preferred = _HEADLINE[kind]
    if preferred in metrics:
        return preferred
    return "accuracy" if "accuracy" in metrics else "r2"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _determinism_section(results: dict[str, Any]) -> list[str]:
    determinism = results.get("position_determinism") or {}
    if not determinism:
        return []
    out = [
        "## position determinism",
        "",
        "Fraction of (family, step) cells whose label is already constant, and the share of "
        "the label the cell explains in sample (variance share, or cell-majority accuracy). "
        "Near 1.0 means position determines state in this dataset and no probe on it can be "
        "read as a state probe; --mix-difficulty is what lowers it.",
        "",
    ]
    rows = [
        [
            target,
            f"{entry['constant_fraction']:.3f}",
            f"{entry['explained']:.3f}",
            str(entry["cells"]),
            f"{entry['multi_row_constant_fraction']:.3f}",
            str(entry["multi_row_cells"]),
            str(entry["rows"]),
        ]
        for target, entry in determinism.items()
    ]
    out.extend(
        _table(
            [
                "target",
                "constant",
                "explained",
                "cells",
                "constant (>1 row)",
                "cells (>1 row)",
                "rows",
            ],
            rows,
        )
    )
    out.append("")
    return out


def _preflight_section(results: dict[str, Any]) -> list[str]:
    preflight = results.get("meta", {}).get("deconfounding_preflight")
    if not preflight:
        return []
    rows = [
        [
            target,
            str(entry["eligible_cells"]),
            str(entry["n_train"]),
            str(entry["n_test"]),
            str(entry["excluded_test_rows"]),
        ]
        for target, entry in preflight["targets"].items()
    ]
    out = [
        "## deconfounding gate",
        "",
        f"**{preflight['status']}** — difficulty levels {preflight['difficulty_levels']}. "
        "Mixed runs require leakage-free within-position analysis; raw probe margins remain "
        "diagnostics because the generator retains structural position signal.",
        "",
        *_table(
            ["target", "varying train cells", "train rows", "test rows", "excluded test rows"],
            rows,
        ),
        "",
    ]
    if preflight.get("failures"):
        out.extend([*(f"- {failure}" for failure in preflight["failures"]), ""])
    return out


def _verdict(entry: dict[str, Any], metric: str) -> str:
    margins = {
        layer: values["margin"]
        for layer, values in entry["layers"].items()
        if values.get("margin") is not None and not np.isnan(values["margin"])
    }
    if not margins:
        return "No margin over the position baseline could be computed."
    best = max(margins, key=lambda layer: margins[layer])
    if margins[best] > 0:
        return (
            f"Verdict: probe beats position baseline by {margins[best]:.3f} {metric} "
            f"at layer {best}."
        )
    return (
        f"Verdict: probe does not beat position baseline "
        f"(best margin {margins[best]:+.3f} {metric} at layer {best})."
    )


def _difficulty_rows(entry: dict[str, Any], metric: str) -> list[str]:
    """Per-difficulty breakdown at the layer with the largest margin, beside the baseline."""
    with_groups = {
        layer: values
        for layer, values in entry["layers"].items()
        if "by_difficulty" in values.get("probe", {})
    }
    baseline = entry.get("position_baseline", {}).get("by_difficulty")
    if not with_groups or not baseline:
        return []
    best = max(with_groups, key=lambda layer: with_groups[layer]["margin"])
    probe = with_groups[best]["probe"]["by_difficulty"]
    rows = []
    for level in sorted(set(probe) | set(baseline)):
        probe_value = probe.get(level, {}).get(metric, float("nan"))
        base_value = baseline.get(level, {}).get(metric, float("nan"))
        rows.append(
            [
                level,
                str(probe.get(level, baseline.get(level, {})).get("n", "?")),
                f"{probe_value:.3f}",
                f"{base_value:.3f}",
                f"{probe_value - base_value:+.3f}",
            ]
        )
    return [
        "",
        f"By difficulty at layer {best}:",
        "",
        *_table(
            ["difficulty", "test rows", metric, f"{metric} (position)", "margin"],
            rows,
        ),
    ]


def _within_rows(entry: dict[str, Any], metric: str, second: str) -> list[str]:
    layers = {
        layer: values["within_position"]
        for layer, values in entry["layers"].items()
        if "within_position" in values
    }
    if not layers:
        return []
    out = [
        "",
        "Within position (training-cell mean activation removed; regression targets are "
        "cell-centred too). Only cells with target variation in the training half are scored, "
        "and held-out labels never determine eligibility. Whatever survives here is not position:",
        "",
    ]
    rows = []
    for layer, values in layers.items():
        if values.get("skipped"):
            rows.append(
                [
                    layer,
                    "-",
                    "-",
                    str(values.get("eligible_cells", 0)),
                    str(values.get("n_test", 0)),
                    values["skipped"],
                ]
            )
            continue
        key = _headline_key(entry["kind"], values)
        degenerate = entry["kind"] == "regression" and np.isnan(values[key])
        rows.append(
            [
                layer,
                f"{values[key]:.3f}",
                f"{values.get(second, float('nan')):.3f}",
                str(values["eligible_cells"]),
                str(values["n_test"]),
                "no within-cell variance left in the target" if degenerate else "",
            ]
        )
    out.extend(_table(["layer", metric, second, "eligible cells", "test rows", "note"], rows))
    return out


def render_markdown(results: dict[str, Any], label: str) -> str:
    """One table per target: the headline metric per layer, with both controls beside it."""
    out = [f"# P2: linear state probes -- {label} (design §5)", ""]
    meta = results.get("meta", {})
    fitted = [entry for entry in results["targets"].values() if "n_train" in entry]
    split = f"{fitted[0]['n_train']}/{fitted[0]['n_test']}" if fitted else "?"
    splits = ", ".join(meta.get("splits", [])) or results.get("split", "?")
    difficulties = meta.get("difficulties")
    out.append(
        f"{meta.get('rows', '?')} rows from {meta.get('tasks', '?')} tasks, "
        f"splits {splits}, notes {'STRIPPED' if meta.get('strip') else 'intact'}, "
        f"{'difficulties ' + json.dumps(difficulties) + ', ' if difficulties else ''}"
        f"split by task id ({split} train/test rows), seed {results.get('seed')}.\n"
    )
    out.extend(_preflight_section(results))
    out.extend(_determinism_section(results))
    for target, entry in results["targets"].items():
        out.append(f"## {target} ({entry['kind']})")
        out.append("")
        if entry.get("skipped"):
            out.append(f"Skipped: {entry['skipped']}.\n")
            continue
        baseline = entry.get("position_baseline", {})
        metric = _headline_key(entry["kind"], baseline)
        second = "mae" if entry["kind"] == "regression" else "macro_f1"
        header = [
            "layer",
            metric,
            f"{metric} (shuffled)",
            f"{metric} (position)",
            "margin",
            second,
            f"{second} (shuffled)",
            f"{second} (position)",
        ]
        rows = []
        for layer, values in entry["layers"].items():
            probe = values["probe"]
            control = values["shuffled_control"]
            rows.append(
                [
                    layer,
                    f"{probe[metric]:.3f}",
                    f"{control.get(metric, float('nan')):.3f}",
                    f"{baseline.get(metric, float('nan')):.3f}",
                    f"{values['margin']:+.3f}",
                    f"{probe[second]:.3f}",
                    f"{control.get(second, float('nan')):.3f}",
                    f"{baseline.get(second, float('nan')):.3f}",
                ]
            )
        out.extend(_table(header, rows))
        out.append("")
        out.append(_verdict(entry, metric))
        out.extend(_difficulty_rows(entry, metric))
        out.extend(_within_rows(entry, metric, second))
        note = f"{entry['rows']} labelled rows"
        if entry.get("classes"):
            note += f"; classes {entry['classes']}"
        determinism = entry.get("position_determinism")
        if determinism:
            note += f"; {determinism['constant_fraction']:.3f} of cells position-determined"
        out.append("")
        out.append(note + ".\n")
    return "\n".join(out)


# --------------------------------------------------------------------------- CLI


def _build_plan(args: argparse.Namespace, splits: list[str]) -> list[tuple[str, bool | None]]:
    return list(MIX_PLAN) if args.mix_difficulty else [(name, None) for name in splits]


def resolve_within_position(mix_difficulty: bool, requested: bool | None) -> bool:
    """Mixed runs must use the analysis that removes their remaining position signal."""
    if mix_difficulty and requested is False:
        raise ValueError("--mix-difficulty requires within-position analysis")
    return bool(mix_difficulty) if requested is None else bool(requested)


def main() -> None:
    from local_llm_lab.pipeline.evaluate import load_policy
    from local_llm_lab.pipeline.tasks import make_tasks
    from local_llm_lab.probes.guard import add_gpu_arguments, require_idle_gpu
    from local_llm_lab.probes.policies import POLICY_NAMES, resolve_policy

    parser = argparse.ArgumentParser(
        description="P2: linear state probes over the residual stream."
    )
    parser.add_argument(
        "--policy", default="base", help=f"one of {POLICY_NAMES} or an adapter directory"
    )
    parser.add_argument("--model", default="mlx-community/Qwen2.5-Coder-3B-Instruct-4bit")
    parser.add_argument("--splits", default="train", help="Comma-separated split names.")
    parser.add_argument("--split", default=None, help="Deprecated alias for a single --splits.")
    parser.add_argument(
        "--mix-difficulty",
        action="store_true",
        help="Build from train (difficulty 0), p2mix (alternating 0/1) and clean test "
        "(difficulty 2), so the same (family, step) occurs with different true states.",
    )
    parser.add_argument("--limit", type=int, default=120, help="Tasks per split.")
    parser.add_argument("--layers", default="6,12,18,24,30,35")
    parser.add_argument("--strip", action="store_true", help="Remove state fields from every note.")
    parser.add_argument(
        "--within-position",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fit after removing each training (family, step) cell mean. Automatically enabled "
        "and cannot be disabled with --mix-difficulty.",
    )
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--data-seed", type=int, default=20260902, help="Task generator seed.")
    parser.add_argument("--keep-last", type=int, default=DEFAULT_KEEP_LAST)
    parser.add_argument(
        "--mlx-cache-limit-mib",
        type=int,
        default=DEFAULT_MLX_CACHE_LIMIT_MIB,
        help="Maximum MLX allocator cache in MiB; reclaimed at every task boundary.",
    )
    parser.add_argument("--reuse", action="store_true", help="Reuse a captured .npz if present.")
    add_gpu_arguments(parser)
    args = parser.parse_args()

    if args.split is not None:
        if args.splits != "train":
            parser.error("pass either --split or --splits, not both")
        args.splits = args.split
    splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    if not splits:
        parser.error("--splits needs at least one split name")
    layers = [int(part) for part in args.layers.split(",") if part.strip()]
    targets = [part for part in args.targets.split(",") if part.strip()]
    unknown = [target for target in targets if target not in TARGETS]
    if unknown:
        parser.error(f"unknown targets {unknown}; known: {list(TARGETS)}")
    if args.mlx_cache_limit_mib <= 0:
        parser.error("--mlx-cache-limit-mib must be positive")
    try:
        within_position = resolve_within_position(args.mix_difficulty, args.within_position)
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    plan = _build_plan(args, splits)
    suffix = (
        "-mix" if args.mix_difficulty else ("" if splits == ["train"] else "-" + "_".join(splits))
    )
    stem = f"state-{args.policy}{'-stripped' if args.strip else ''}{suffix}"
    npz_path = args.output / f"{stem}.npz"
    preflight: dict[str, Any] | None = None

    if args.reuse and npz_path.is_file():
        dataset = load_dataset(npz_path)
        if args.mix_difficulty:
            preflight = validate_mixed_design(dataset, targets, args.seed)
            if preflight["status"] != "PASS":
                parser.error(
                    "mixed-difficulty preflight failed: " + "; ".join(preflight["failures"])
                )
        print(f"Reusing {npz_path} ({len(dataset)} rows)")
    else:
        tasks: list[Task] = []
        difficulties: dict[str, int] = {}
        for name, perturb in plan:
            made = (
                make_tasks(name, args.limit, args.data_seed)
                if perturb is None
                else make_tasks(name, args.limit, args.data_seed, perturb=perturb)
            )
            tasks.extend(made)
            difficulties.update(task_difficulties(name, made))
        if args.mix_difficulty:
            label_dataset = build_label_dataset(tasks, difficulties)
            preflight = validate_mixed_design(label_dataset, targets, args.seed)
            if preflight["status"] != "PASS":
                parser.error(
                    "mixed-difficulty preflight failed: " + "; ".join(preflight["failures"])
                )
            print(
                "within-position preflight PASS: "
                + ", ".join(
                    f"{target}={entry['eligible_cells']} cells/{entry['n_test']} test rows"
                    for target, entry in preflight["targets"].items()
                ),
                flush=True,
            )
        require_idle_gpu(parser, args, "capturing activations")
        adapter = resolve_policy(args.policy)
        import mlx.core as mx

        previous_cache_limit = set_mlx_cache_limit(mx, args.mlx_cache_limit_mib)
        model: Any = None
        checkpoint_dir = args.output / f"{stem}.checkpoints"
        latest_memory: dict[str, Any] = {}
        try:
            model, tokenizer = load_policy(args.model, adapter)
            model_artifact = artifact_identity(args.model)
            adapter_artifact = artifact_identity(adapter)
            print(f"{len(tasks)} tasks from {[name for name, _ in plan]}", flush=True)
            print(
                f"MLX cache limited to {args.mlx_cache_limit_mib} MiB; "
                f"task checkpoints in {checkpoint_dir}",
                flush=True,
            )

            def memory_progress(sample: dict[str, Any]) -> None:
                latest_memory.clear()
                latest_memory.update(sample)

            def progress(number: int, total: int, rows: int) -> None:
                mib = 2**20
                source = "resumed" if latest_memory.get("resumed") else "captured"
                peak = latest_memory.get("peak_bytes")
                peak_text = "n/a" if peak is None else f"{peak / mib:.0f} MiB"
                print(
                    f"[{number:03d}/{total:03d}] {rows} rows available ({source}); "
                    f"MLX active={latest_memory.get('active_bytes', 0) / mib:.0f} MiB, "
                    f"cache={latest_memory.get('cache_bytes', 0) / mib:.0f} MiB, "
                    f"task peak={peak_text}",
                    flush=True,
                )

            dataset = build_probe_dataset(
                model,
                tokenizer,
                tasks,
                layers,
                args.strip,
                keep_last=args.keep_last,
                difficulties=difficulties,
                progress=progress,
                mlx_runtime=mx,
                memory_progress=memory_progress,
                checkpoint_dir=checkpoint_dir,
                checkpoint_context={
                    "model": args.model,
                    "model_artifact": model_artifact,
                    "policy": args.policy,
                    "adapter_artifact": adapter_artifact,
                    "splits": [name for name, _ in plan],
                    "mix_difficulty": bool(args.mix_difficulty),
                    "data_seed": args.data_seed,
                },
            )
        finally:
            model = None
            mx.clear_cache()
            mx.set_cache_limit(previous_cache_limit)
        dataset.meta["splits"] = [name for name, _ in plan]
        dataset.meta["mix_difficulty"] = bool(args.mix_difficulty)
        dataset.meta["mlx_cache_limit_mib"] = args.mlx_cache_limit_mib
        if preflight is not None:
            dataset.meta["deconfounding_preflight"] = preflight
        save_dataset(dataset, npz_path)
        print(f"Wrote {npz_path} ({len(dataset)} rows)")

    if preflight is not None:
        dataset.meta["deconfounding_preflight"] = preflight
    results = fit_probes(dataset, targets, layers, args.seed, within_position=within_position)
    results["policy"] = args.policy
    results["splits"] = [name for name, _ in plan]
    results["split"] = ",".join(name for name, _ in plan)
    (args.output / f"{stem}.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    markdown = render_markdown(results, f"{args.policy}{' (stripped notes)' if args.strip else ''}")
    (args.output / f"{stem}.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    print(f"Wrote {args.output / f'{stem}.json'}")
