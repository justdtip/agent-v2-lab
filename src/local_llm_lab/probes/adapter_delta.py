"""P5: what the adapters wrote -- the exact model diff (design §8).

A LoRA adapter over a shared base *is* an exact model diff, so the crosscoder question ("what
is model-specific?") can be answered directly instead of by training a crosscoder. For each of
the 252 modules this measures how big the update is relative to the weight it edits, how much
of the rank-16 budget it used, whether independent runs wrote into the same subspace, and what
the update's output directions read out as.

Orientation. ``mlx_lm.tuner.lora.LoRALinear`` stores ``lora_a`` as ``(input_dims, r)`` and
``lora_b`` as ``(r, output_dims)`` (lora.py:88-93) and its forward pass is
``y = linear(x) + scale * ((dropout(x) @ lora_a) @ lora_b)`` (lora.py:95-98). So in the base
weight's own ``(out, in)`` orientation the update is ``scale * (lora_a @ lora_b).T``, which is
exactly what ``fuse()`` adds to ``linear.weight`` (lora.py:52-53). ``test_probes.py`` checks
this against mlx-lm's own class rather than trusting the reading.

Memory. Large MLP deltas are tens of megabytes each in float32, so
materialising all 252 would cost ~10 GB. Nothing here needs the dense matrix: because
``deltaW = L R`` with ``L = scale * lora_b.T`` (out x r) and ``R = lora_a.T`` (r x in), a QR of
each factor reduces the SVD to an r x r problem, giving *exact* singular values and left
singular vectors, and ``||deltaW||_F = sqrt(sum(s^2))`` exactly. The dense array returned by
:func:`load_adapter_deltas` is therefore an unevaluated MLX graph; evaluating one is fine,
evaluating all of them is not.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from local_llm_lab.arch import ArchitectureView
from local_llm_lab.models import ResolvedSpec

__all__ = [
    "DEFAULT_SCALE",
    "analyse_adapter",
    "compare_adapters",
    "delta_from_factors",
    "effective_rank_90",
    "left_singular_vectors",
    "load_adapter_deltas",
    "main",
    "principal_angle_cosines",
    "random_subspace_cosine",
    "readout_update_directions",
    "run_block_ablation",
    "run_label",
    "relative_norm",
    "weight_norm",
    "render_markdown",
    "spectrum",
]

DEFAULT_SCALE = 20.0  # mlx_lm.tuner.lora.LoRALinear's own default, used when a config omits it
_LAYER_RE = re.compile(r"layers\.(\d+)\.")


# --------------------------------------------------------------------------- loading


def _load_tensors(path: Path) -> dict[str, np.ndarray]:
    """Adapter tensors as numpy arrays, without touching MLX (so without touching the GPU)."""
    from safetensors.numpy import load_file

    return load_file(str(path))


def adapter_scale(adapter_dir: Path) -> float:
    """The LoRA ``scale`` the adapter was trained with, from its ``adapter_config.json``."""
    config_path = Path(adapter_dir) / "adapter_config.json"
    if not config_path.is_file():
        return DEFAULT_SCALE
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parameters = config.get("lora_parameters") or {}
    return float(parameters.get("scale", DEFAULT_SCALE))


def delta_from_factors(lora_a: Any, lora_b: Any, scale: float) -> Any:
    """``deltaW = scale * (lora_a @ lora_b).T``: the update in the base weight's ``(out, in)``
    orientation, matching ``LoRALinear.fuse`` (mlx_lm/tuner/lora.py:52-53)."""
    import mlx.core as mx

    a = mx.array(np.asarray(lora_a, dtype=np.float32))
    b = mx.array(np.asarray(lora_b, dtype=np.float32))
    return (float(scale) * (a @ b)).T


def run_label(path: str | Path) -> str:
    """A label that distinguishes runs: every run's adapter directory is called
    ``best-adapter``, so the run name is the directory above it (``agent-v2b``)."""
    directory = Path(path)
    return (
        directory.parent.name if directory.name in ("best-adapter", "adapters") else directory.name
    )


def _unique_labels(paths: list[Path]) -> list[str]:
    labels = [run_label(path) for path in paths]
    if len(set(labels)) == len(labels):
        return labels
    return [f"{label}#{index}" for index, label in enumerate(labels)]


def module_layer(name: str) -> int | None:
    match = _LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def module_type(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def load_adapter_deltas(adapter_dir: str | Path) -> dict[str, tuple[Any, dict[str, Any]]]:
    """``{module: (deltaW, info)}`` for every LoRA module in ``adapters.safetensors``.

    ``deltaW`` is ``(out, in)`` float32 and **lazy** (see the module docstring). ``info``
    carries the float32 factors (``lora_a``, ``lora_b``), the ``scale``, the rank, the layer
    index and the module type, which is what every statistic below is computed from.
    """
    directory = Path(adapter_dir)
    tensors = _load_tensors(directory / "adapters.safetensors")
    scale = adapter_scale(directory)
    names = sorted(
        {key.rsplit(".", 1)[0] for key in tensors if key.endswith((".lora_a", ".lora_b"))}
    )
    deltas: dict[str, tuple[Any, dict[str, Any]]] = {}
    for name in names:
        lora_a = np.asarray(tensors[f"{name}.lora_a"], dtype=np.float32)
        lora_b = np.asarray(tensors[f"{name}.lora_b"], dtype=np.float32)
        if lora_a.shape[1] != lora_b.shape[0]:
            raise ValueError(
                f"{name}: lora_a {lora_a.shape} does not compose with lora_b {lora_b.shape}"
            )
        info = {
            "module": name,
            "layer": module_layer(name),
            "type": module_type(name),
            "scale": scale,
            "rank": int(lora_a.shape[1]),
            "shape": [int(lora_b.shape[1]), int(lora_a.shape[0])],
            "lora_a": lora_a,
            "lora_b": lora_b,
        }
        deltas[name] = (delta_from_factors(lora_a, lora_b, scale), info)
    return deltas


# --------------------------------------------------------------------------- spectrum


def _factors(info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """``(L, R)`` with ``deltaW = L @ R``, ``L`` = ``(out, r)`` and ``R`` = ``(r, in)``."""
    left = float(info["scale"]) * np.asarray(info["lora_b"], dtype=np.float32).T
    right = np.asarray(info["lora_a"], dtype=np.float32).T
    return left, right


def _thin_svd(info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Exact ``(U, s)`` of ``deltaW`` via QR of both rank-r factors; never forms ``deltaW``."""
    left, right = _factors(info)
    q_left, r_left = np.linalg.qr(left)
    q_right, r_right = np.linalg.qr(right.T)
    del q_right
    u_small, values, _ = np.linalg.svd(r_left @ r_right.T)
    return q_left @ u_small, values


def spectrum(info: dict[str, Any], top: int = 16) -> np.ndarray:
    """Singular values of ``deltaW``, largest first (at most ``rank`` are non-zero)."""
    return _thin_svd(info)[1][:top].astype(np.float32)


def left_singular_vectors(info: dict[str, Any], k: int = 2) -> np.ndarray:
    """The top-``k`` left singular vectors of ``deltaW``: the *output* directions it writes."""
    return _thin_svd(info)[0][:, :k].astype(np.float32)


def effective_rank_90(values: np.ndarray, fraction: float = 0.9) -> int:
    """How many singular values hold ``fraction`` of the squared energy (0 for a zero update)."""
    energy = np.asarray(values, dtype=np.float64) ** 2
    total = float(energy.sum())
    if total <= 0.0:
        return 0
    return int(np.searchsorted(np.cumsum(energy) / total, fraction) + 1)


def delta_norm(info: dict[str, Any]) -> float:
    """``||deltaW||_F``, from the singular values (exact, and never forms ``deltaW``)."""
    return float(np.sqrt((spectrum(info, top=int(info["rank"])) ** 2).sum()))


def principal_angle_cosines(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Cosines of the principal angles between the column spaces of ``left`` and ``right``.

    Both are orthonormalised first, so the cosines are the singular values of ``Qa.T @ Qb``:
    1.0 for identical subspaces, 0.0 for orthogonal ones.
    """
    q_left, _ = np.linalg.qr(np.asarray(left, dtype=np.float64))
    q_right, _ = np.linalg.qr(np.asarray(right, dtype=np.float64))
    values = np.linalg.svd(q_left.T @ q_right, compute_uv=False)
    return np.clip(values, 0.0, 1.0)


# --------------------------------------------------------------------------- base weights


def resolve_module(model: Any, name: str) -> Any:
    """Walk a dotted module path such as ``model.layers.7.mlp.down_proj``.

    Descends through a ``LoRALinear`` wrapper's ``.linear`` when the model was loaded with the
    adapter attached, so the same name resolves to the base weight either way.
    """
    view = model if isinstance(model, ArchitectureView) else ArchitectureView.from_model(model)
    node: Any = view.model
    for part in name.split("."):
        node = node[int(part)] if part.isdigit() else getattr(node, part)
    return getattr(node, "linear", node)


def base_weight(model: Any, name: str) -> Any:
    """The base weight of ``name`` as a dense float32 ``(out, in)`` array, dequantised if needed."""
    import mlx.core as mx

    module = resolve_module(model, name)
    weight = module.weight
    if hasattr(module, "scales"):
        extra = {"mode": module.mode} if hasattr(module, "mode") else {}
        weight = mx.dequantize(
            weight,
            module.scales,
            module.biases,
            group_size=module.group_size,
            bits=module.bits,
            **extra,
        )
    return weight.astype(mx.float32)


def weight_norm(model: Any, name: str) -> float:
    """``||W||_F`` of a base weight, accumulated in float32.

    The dequantised weight is float16 on this checkpoint and a Frobenius norm sums ~20 million
    squares, which overflows float16 to infinity; the cast is what keeps the ratio finite.
    """
    import mlx.core as mx

    value = float(mx.sqrt(mx.sum(base_weight(model, name).astype(mx.float32) ** 2)).item())
    if value == 0.0 or not np.isfinite(value):
        raise ValueError(f"{name}: base weight norm is {value}")
    return value


def relative_norm(info: dict[str, Any], weight: Any) -> float:
    """``||deltaW||_F / ||W||_F`` for one module, given its dense base weight."""
    import mlx.core as mx

    base = float(mx.sqrt(mx.sum(weight.astype(mx.float32) ** 2)).item())
    if base == 0.0:
        raise ValueError(f"{info['module']}: base weight has zero Frobenius norm")
    return delta_norm(info) / base


# --------------------------------------------------------------------------- per-adapter report


def analyse_adapter(
    adapter_dir: str | Path,
    model: Any = None,
    *,
    top: int = 16,
    norm_cache: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Per-module statistics for one adapter: norms, spectrum, and effective rank.

    ``model`` is the *base* model; when given, ``relative_norm`` is filled in by dequantising
    each base weight one at a time. Without it the report carries absolute ``||deltaW||_F``
    only (and no model is loaded, so no GPU is touched). Every run shares the same base, so
    ``norm_cache`` lets a caller comparing several adapters pay for the 252 dequantisations
    once.
    """
    directory = Path(adapter_dir)
    deltas = load_adapter_deltas(directory)
    modules = []
    for name, (_delta, info) in deltas.items():
        values = spectrum(info, top=top)
        record = {
            "module": name,
            "layer": info["layer"],
            "type": info["type"],
            "shape": info["shape"],
            "rank": info["rank"],
            "scale": info["scale"],
            "delta_norm": delta_norm(info),
            "singular_values": [float(value) for value in values],
            "effective_rank_90": effective_rank_90(values),
            "relative_norm": None,
        }
        if model is not None:
            if norm_cache is None or name not in norm_cache:
                base = weight_norm(model, name)
                if norm_cache is not None:
                    norm_cache[name] = base
            else:
                base = norm_cache[name]
            record["relative_norm"] = record["delta_norm"] / base
        modules.append(record)
    modules.sort(
        key=lambda record: (record["layer"] if record["layer"] is not None else -1, record["type"])
    )
    return {
        "adapter": str(directory.resolve()),
        "scale": adapter_scale(directory),
        "modules": modules,
        "effective_rank_histogram": _histogram(record["effective_rank_90"] for record in modules),
    }


def _histogram(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def random_subspace_cosine(
    dimension: int, top: int, *, trials: int = 8, seed: int = 20260903
) -> float:
    """Mean principal-angle cosine between two *random* ``top``-dimensional subspaces.

    This is the null the cross-run agreement is read against: no probe result is reported
    without its control, and "0.4" only means something next to what chance gives at the same
    dimensions.
    """
    rng = np.random.default_rng(seed)
    scores = [
        float(
            np.mean(
                principal_angle_cosines(
                    rng.normal(size=(dimension, top)), rng.normal(size=(dimension, top))
                )
            )
        )
        for _ in range(trials)
    ]
    return float(np.mean(scores))


def compare_adapters(paths: list[str | Path], *, top: int = 16) -> dict[str, Any]:
    """Principal-angle agreement between the column spaces of ``deltaW`` across runs.

    For every module present in all runs, the top-``top`` left singular vectors of each run's
    update are compared pairwise; the reported number is the mean cosine of the principal
    angles (1.0 = the same subspace, 0.0 = orthogonal). Compare the reported values with the
    random-subspace control at the actual output dimension rather than a fixed reference size.
    """
    directories = [Path(path) for path in paths]
    if len(directories) < 2:
        raise ValueError("compare_adapters needs at least two adapters")
    labels = _unique_labels(directories)
    bases: dict[str, dict[str, np.ndarray]] = {}
    for label, directory in zip(labels, directories, strict=True):
        vectors = {}
        for name, (_delta, info) in load_adapter_deltas(directory).items():
            vectors[name] = left_singular_vectors(info, k=top)
        bases[label] = vectors
    shared = sorted(set.intersection(*(set(vectors) for vectors in bases.values())))
    pairs = [(labels[i], labels[j]) for i in range(len(labels)) for j in range(i + 1, len(labels))]
    per_module: list[dict[str, Any]] = []
    for name in shared:
        record: dict[str, Any] = {
            "module": name,
            "layer": module_layer(name),
            "type": module_type(name),
        }
        for left, right in pairs:
            cosines = principal_angle_cosines(bases[left][name], bases[right][name])
            record[f"{left}|{right}"] = float(np.mean(cosines))
        per_module.append(record)
    summary: dict[str, dict[str, float]] = {}
    for left, right in pairs:
        key = f"{left}|{right}"
        by_type = {
            module: float(
                np.mean([record[key] for record in per_module if record["type"] == module])
            )
            for module in sorted({str(record["type"]) for record in per_module})
        }
        by_type["all"] = float(np.mean([record[key] for record in per_module]))
        summary[key] = by_type
    null: dict[str, float] = {}
    for module_name in {record["type"] for record in per_module}:
        example = next(record["module"] for record in per_module if record["type"] == module_name)
        out_dim = int(next(iter(bases.values()))[example].shape[0])
        null[module_name] = random_subspace_cosine(out_dim, top)
    null["all"] = float(np.mean(list(null.values())))
    return {
        "adapters": labels,
        "modules": per_module,
        "mean_cosine_by_type": summary,
        "null_mean_cosine_by_type": null,
        "top": top,
    }


# --------------------------------------------------------------------------- block ablation


def _layer_blocks(num_layers: int, blocks: int) -> list[tuple[int, ...]]:
    """Partition every layer into consecutive near-equal non-empty blocks."""
    if not 1 <= blocks <= num_layers:
        raise ValueError("blocks must be between 1 and num_layers")
    width, remainder = divmod(num_layers, blocks)
    result: list[tuple[int, ...]] = []
    start = 0
    for index in range(blocks):
        stop = start + width + int(index < remainder)
        result.append(tuple(range(start, stop)))
        start = stop
    return result


def _success_record(successes: int, tasks: int) -> dict[str, Any]:
    """Return exact counts, their rate, and the repository-standard Wilson interval."""
    from local_llm_lab.pipeline.evaluate import wilson

    lower, upper = wilson(successes, tasks)
    return {
        "successes": successes,
        "tasks": tasks,
        "success_rate": successes / tasks if tasks else 0.0,
        "wilson_95": (lower, upper),
    }


def _aggregate_screen_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(int(summary["successes"]) for summary in summaries)
    tasks = sum(int(summary["tasks"]) for summary in summaries)
    family_counts: dict[str, list[int]] = {}
    for summary in summaries:
        for family, values in summary.get("by_family", {}).items():
            counts = family_counts.setdefault(str(family), [0, 0])
            counts[0] += int(values["successes"])
            counts[1] += int(values["tasks"])
    return {
        "overall": _success_record(successes, tasks),
        "by_family": {
            family: _success_record(counts[0], counts[1])
            for family, counts in sorted(family_counts.items())
        },
    }


def _evaluate_ablation_condition(
    view: ArchitectureView,
    *,
    name: str,
    block_index: int | None,
    kept_layers: tuple[int, ...],
    removed_layers: tuple[int, ...],
    screen: list[dict[str, Any]],
    evaluate_condition: Callable[[str, int, dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    from local_llm_lab.probes.capture import lora_block_mask

    evaluations: list[dict[str, Any]] = []
    with lora_block_mask(view, kept_layers) as masked_modules:
        for cell_index, cell in enumerate(screen):
            summary = evaluate_condition(name, cell_index, cell)
            evaluations.append(
                {"cell_index": cell_index, "screen": dict(cell), "summary": summary}
            )
    aggregate = _aggregate_screen_summaries(
        [evaluation["summary"] for evaluation in evaluations]
    )
    return {
        "name": name,
        "block_index": block_index,
        "kept_layers": list(kept_layers),
        "removed_layers": list(removed_layers),
        "masked_modules": int(masked_modules),
        **aggregate,
        "evaluations": evaluations,
    }


def run_block_ablation(
    view: ArchitectureView,
    *,
    blocks: int,
    screen: list[dict[str, Any]],
    evaluate_condition: Callable[[str, int, dict[str, Any]], dict[str, Any]],
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Evaluate full, empty, and leave-one-block-out adapter conditions on one screen.

    ``progress`` is an optional ``(step, total, label)`` callback fired once per condition
    once that condition's screen has been evaluated (R26(g)); the record it returns is
    identical with and without it.
    """
    if not screen:
        raise ValueError("screen must contain at least one evaluation cell")
    partitions = _layer_blocks(int(view.num_layers), blocks)
    all_layers = tuple(range(int(view.num_layers)))
    condition_total = len(partitions) + 2
    condition_number = 0

    def note(name: str) -> None:
        nonlocal condition_number
        condition_number += 1
        if progress is not None:
            progress(condition_number, condition_total, f"condition {name}")

    full = _evaluate_ablation_condition(
        view,
        name="full_adapter",
        block_index=None,
        kept_layers=all_layers,
        removed_layers=(),
        screen=screen,
        evaluate_condition=evaluate_condition,
    )
    note("full_adapter")
    empty = _evaluate_ablation_condition(
        view,
        name="empty_adapter",
        block_index=None,
        kept_layers=(),
        removed_layers=all_layers,
        screen=screen,
        evaluate_condition=evaluate_condition,
    )
    note("empty_adapter")
    conditions = []
    for block_index, removed in enumerate(partitions):
        removed_set = set(removed)
        kept = tuple(layer for layer in all_layers if layer not in removed_set)
        conditions.append(
            _evaluate_ablation_condition(
                view,
                name=f"remove_block_{block_index}",
                block_index=block_index,
                kept_layers=kept,
                removed_layers=removed,
                screen=screen,
                evaluate_condition=evaluate_condition,
            )
        )
        note(f"remove_block_{block_index}")
    full_rate = float(full["overall"]["success_rate"])
    most_costly = max(
        conditions,
        key=lambda condition: (
            full_rate - float(condition["overall"]["success_rate"]),
            -int(condition["block_index"]),
        ),
    )
    return {
        "blocks": [list(block) for block in partitions],
        "controls": {"full_adapter": full, "empty_adapter": empty},
        "conditions": conditions,
        "most_costly_removal": {
            "condition": most_costly["name"],
            "block_index": most_costly["block_index"],
            "removed_layers": most_costly["removed_layers"],
            "success_rate_cost": full_rate
            - float(most_costly["overall"]["success_rate"]),
        },
    }


@contextmanager
def _reuse_loaded_policy(
    model: Any, tokenizer: Any, view: ArchitectureView, resolved: ResolvedSpec
) -> Iterator[None]:
    """Make evaluator cells reuse one loaded policy, restoring its loader even on failure."""
    from local_llm_lab.pipeline import evaluate

    original = evaluate.load_policy
    evaluate.load_policy = lambda *_args, **_kwargs: (model, tokenizer, view, resolved)
    try:
        yield
    finally:
        evaluate.load_policy = original


# --------------------------------------------------------------------------- direction readouts


def _jlens_module() -> Any:
    """Lazy import keeps adapter inspection free of MLX model work until readout is requested."""
    from local_llm_lab.pipeline import jlens

    return jlens


def _default_residual_types(
    deltas: dict[str, tuple[Any, dict[str, Any]]],
    hidden_size: int,
) -> set[str]:
    """Select only unambiguous residual-update outputs from parsed adapter metadata.

    A square hidden-to-hidden update can be an internal attention projection as easily as a
    residual update. Default discovery therefore only admits outputs with the residual width and
    a distinct input width. Ambiguous paths remain available through the explicit ``types``
    filter at the public call site.
    """
    selected: set[str] = set()
    for _delta, info in deltas.values():
        shape = _update_shape(info)
        if shape is None:
            continue
        output_size, input_size = shape
        if output_size == hidden_size and input_size != hidden_size:
            selected.add(str(info["type"]))
    return selected


def _update_shape(info: dict[str, Any]) -> tuple[int, int] | None:
    """Return an adapter update's `(output, input)` shape when record metadata is valid."""
    shape = info.get("shape", ())
    if len(shape) != 2:
        return None
    return int(shape[0]), int(shape[1])


def readout_update_directions(
    view: ArchitectureView,
    tokenizer: Any,
    adapter_dir: str | Path,
    layers: list[int],
    *,
    types: tuple[str, ...] | None = None,
    directions: int = 2,
    corpus_size: int = 8,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Read top output directions of parsed adapter updates as tokens.

    Without an explicit ``types`` filter, parsed shape metadata and the view's residual width
    select only unambiguous residual-update outputs. Ambiguous hidden-to-hidden internal paths
    require an intentional caller filter instead of being assumed from a module name. Every
    selected record still must output the residual width, including explicit filters.

    A direction produced by block ``L`` lands in the layer ``L + 1`` residual stream, so that is
    the layer whose tail Jacobian the J-lens uses.

    **These are readouts of an update *direction*, not of an activation the model ever has.**
    The sign of a singular vector is arbitrary, so ``+v`` and ``-v`` are both reported and
    neither is "the" direction; read them as a pair.
    """
    jlens = _jlens_module()
    corpus_ids = [jlens.encode(tokenizer, text) for text in jlens.DEFAULT_CORPUS[:corpus_size]]
    deltas = load_adapter_deltas(adapter_dir)
    requested_types = (
        types if types is not None else _default_residual_types(deltas, int(view.hidden_size))
    )
    hidden_size = int(view.hidden_size)
    records: list[dict[str, Any]] = []
    for name, (_delta, info) in sorted(deltas.items()):
        if info["type"] not in requested_types or info["layer"] not in layers:
            continue
        shape = _update_shape(info)
        if shape is None or shape[0] != hidden_size:
            continue
        if types is None and shape[1] == hidden_size:
            # The default type discovery is intentionally conservative, but a type can
            # occur on multiple paths. Recheck each record so one non-square residual
            # update cannot admit a same-typed square internal projection.
            continue
        vectors = left_singular_vectors(info, k=directions)
        values = spectrum(info, top=directions)
        for index in range(vectors.shape[1]):
            if vectors.shape[0] != hidden_size:
                continue
            direction = _as_mx(vectors[:, index])
            readouts: dict[str, Any] = {}
            mapped, _stats = jlens.jlens_map(view, int(info["layer"]) + 1, direction, corpus_ids)
            for sign_name, sign in (("+v", 1.0), ("-v", -1.0)):
                probe = sign * direction
                readouts[sign_name] = {
                    "logit_lens": jlens.logit_lens(view, probe, tokenizer, k=top_k),
                    "jlens": jlens.readout(view, sign * mapped, tokenizer, k=top_k),
                }
            records.append(
                {
                    "module": name,
                    "layer": info["layer"],
                    "readout_layer": int(info["layer"]) + 1,
                    "type": info["type"],
                    "direction": index,
                    "singular_value": float(values[index]),
                    "sign_ambiguous": True,
                    "readouts": readouts,
                }
            )
    return records


def _as_mx(vector: np.ndarray) -> Any:
    import mlx.core as mx

    return mx.array(np.ascontiguousarray(vector, dtype=np.float32))


# --------------------------------------------------------------------------- rendering


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _append_direction_readouts(out: list[str], readouts: list[dict[str, Any]]) -> None:
    out.extend(
        [
            "## What the update writes (readout of an update *direction*)",
            "",
            "Singular vectors have an arbitrary sign, so both `+v` and `-v` are shown; neither "
            "is privileged. These are directions the update writes, not activations the model "
            "was observed to hold.\n",
        ]
    )
    rows = []
    for record in readouts:
        for sign, lenses in record["readouts"].items():
            rows.append(
                [
                    record["module"],
                    str(record["direction"]),
                    f"{record['singular_value']:.3f}",
                    sign,
                    ", ".join(repr(token) for token, _p, _i in lenses["jlens"][:8]),
                    ", ".join(repr(token) for token, _p, _i in lenses["logit_lens"][:8]),
                ]
            )
    out.extend(
        [
            _table(
                ["module", "dir", "sigma", "sign", "j-lens top tokens", "logit-lens top tokens"],
                rows,
            ),
            "",
        ]
    )


def render_markdown(payload: dict[str, Any]) -> str:
    """Markdown summary: per-layer update size, rank use, cross-run agreement, readouts."""
    out: list[str] = ["# P5: adapter delta (design §8)", ""]
    metric = "relative_norm" if payload.get("has_base") else "delta_norm"
    label = "||dW||/||W||" if payload.get("has_base") else "||dW||_F"
    if not payload.get("has_base"):
        out.append(
            "> Base model not loaded, so update size is reported as the absolute Frobenius "
            "norm of the update instead of relative to the weight it edits.\n"
        )
    for run in payload["runs"]:
        out.append(f"## {run['label']}: {label} by layer and module type")
        out.append("")
        types = sorted({str(module["type"]) for module in run["modules"]})
        layers = sorted({m["layer"] for m in run["modules"] if m["layer"] is not None})
        rows = []
        for layer in layers:
            row = [str(layer)]
            for module_name in types:
                values = [
                    m[metric]
                    for m in run["modules"]
                    if m["layer"] == layer and m["type"] == module_name and m[metric] is not None
                ]
                row.append(f"{values[0]:.4f}" if values else "-")
            rows.append(row)
        out.append(_table(["layer", *types], rows))
        out.append("")
        out.append(
            f"### {run['label']}: effective rank (90% of squared energy), rank {run['modules'][0]['rank']} budget"
        )
        out.append("")
        histogram = run["effective_rank_histogram"]
        out.append(
            _table(
                ["effective rank", "modules"],
                [[key, str(value)] for key, value in histogram.items()],
            )
        )
        out.append("")
    comparison = payload.get("comparison")
    if comparison:
        out.append(
            "## Cross-run agreement: mean cosine of principal angles between update subspaces"
        )
        out.append("")
        pairs = list(comparison["mean_cosine_by_type"])
        types = sorted(
            name for name in comparison["mean_cosine_by_type"][pairs[0]] if name != "all"
        )
        rows = []
        for pair in pairs:
            values = comparison["mean_cosine_by_type"][pair]
            rows.append([pair, *[f"{values[name]:.3f}" for name in types], f"{values['all']:.3f}"])
        null = comparison.get("null_mean_cosine_by_type", {})
        if null:
            rows.append(
                [
                    "random subspaces (control)",
                    *[f"{null[name]:.3f}" for name in types],
                    f"{null['all']:.3f}",
                ]
            )
        out.append(_table(["pair", *types, "all"], rows))
        out.append("")
        out.append(
            f"Top-{comparison['top']} left singular vectors per module. The control row is the "
            "mean principal-angle cosine between two random subspaces of the same shape.\n"
        )
    readouts = payload.get("readouts")
    if readouts:
        _append_direction_readouts(out, readouts)
    return "\n".join(out)


def _format_success(record: dict[str, Any]) -> str:
    lower, upper = record["wilson_95"]
    return (
        f"{record['successes']}/{record['tasks']} ({record['success_rate']:.3f}; "
        f"95% CI {lower:.3f}–{upper:.3f})"
    )


def _render_ablation_markdown(payload: dict[str, Any]) -> str:
    """Render full/empty controls and leave-one-block-out family rates with intervals."""
    controls = payload["controls"]
    conditions = payload["conditions"]
    out = [
        "# P5 item 5: adapter layer-block ablation",
        "",
        "Removal cost is measured against the full adapter: full-adapter success rate minus "
        "the success rate after removing one consecutive block.",
        "",
        "## Anchors",
        "",
        _table(
            ["condition", "kept layers", "removed layers", "overall success (Wilson 95%)"],
            [
                [
                    "Full adapter",
                    ", ".join(str(layer) for layer in controls["full_adapter"]["kept_layers"]),
                    "none",
                    _format_success(controls["full_adapter"]["overall"]),
                ],
                [
                    "Empty adapter",
                    "none",
                    ", ".join(str(layer) for layer in controls["empty_adapter"]["removed_layers"]),
                    _format_success(controls["empty_adapter"]["overall"]),
                ],
            ],
        ),
        "",
        "## Leave-one-block-out conditions",
        "",
    ]
    rows: list[list[str]] = []
    for condition in conditions:
        rows.append(
            [
                condition["name"],
                ", ".join(str(layer) for layer in condition["removed_layers"]),
                "overall",
                _format_success(condition["overall"]),
            ]
        )
        for family, record in condition["by_family"].items():
            rows.append(
                [
                    condition["name"],
                    ", ".join(str(layer) for layer in condition["removed_layers"]),
                    family,
                    _format_success(record),
                ]
            )
    out.extend(
        [
            _table(["condition", "removed layers", "family", "success (Wilson 95%)"], rows),
            "",
        ]
    )
    costly = payload["most_costly_removal"]
    out.extend(
        [
            "## Most costly removal",
            "",
            f"`{costly['condition']}` (layers {costly['removed_layers']}) reduced success by "
            f"{costly['success_rate_cost']:.3f} relative to the full adapter.",
            "",
        ]
    )
    return "\n".join(out)


def _screen_from_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    select = config.get("select")
    screen = select.get("screen") if isinstance(select, dict) else None
    if not isinstance(screen, list) or not screen:
        raise ValueError("screen config must define a non-empty select.screen list")
    cells: list[dict[str, Any]] = []
    for index, cell in enumerate(screen):
        valid = (
            isinstance(cell, dict)
            and isinstance(cell.get("split"), str)
            and isinstance(cell.get("difficulty"), int)
            and isinstance(cell.get("per_family"), dict)
        )
        if not valid:
            raise ValueError(
                f"select.screen[{index}] must define split, difficulty, and per_family"
            )
        cells.append(dict(cell))
    return cells


def _run_ablation_cli(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    require_idle_gpu: Callable[[argparse.ArgumentParser, argparse.Namespace, str], None],
) -> None:
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.pipeline import evaluate
    from local_llm_lab.pipeline.cli import load_config
    from local_llm_lab.runlog import RunLog, git_commit, sha256_of

    spec = load_model_spec(args.model)
    # R26(e), issue #35: the log opens before the GPU guard and the model load, so the whole
    # run — including a guard refusal — is in run.log, and the start line says what ran.
    with RunLog.open(
        args.output,
        name="block-ablation",
        command=sys.argv,
        identity={
            "model": args.model,
            "hf_id": spec.hf_id,
            "adapter": str(args.adapter.resolve()),
            "screen": str(args.screen.resolve()),
            "screen_sha256": sha256_of(args.screen),
            "blocks": args.blocks,
            "git_commit": git_commit(),
        },
    ) as log:
        config = load_config(args.screen)
        try:
            screen = _screen_from_config(config)
        except ValueError as error:
            parser.error(str(error))
        log.info(
            "conditions",
            conditions=args.blocks + 2,
            blocks=args.blocks,
            screen_cells=len(screen),
        )
        require_idle_gpu(parser, args, "loading the adapter policy for block ablation")
        log.info("loading policy", model=args.model, policy=str(args.adapter))
        model, tokenizer, view, resolved_spec = evaluate.load_policy(spec, args.adapter)
        try:
            _layer_blocks(int(view.num_layers), args.blocks)
        except ValueError as error:
            parser.error(str(error))
        resolved = resolved_spec.as_dict()
        eval_config = config.get("eval") if isinstance(config.get("eval"), dict) else {}
        args.output.mkdir(parents=True, exist_ok=True)

        def evaluate_condition(
            condition: str, cell_index: int, cell: dict[str, Any]
        ) -> dict[str, Any]:
            evaluation_dir = args.output / "evaluations" / condition
            filename = f"{cell_index:02d}-{cell['split']}-d{cell['difficulty']}.json"
            return evaluate.run_evaluation(
                spec=spec,
                adapter=args.adapter,
                label=f"{args.adapter.name}-{condition}-{cell_index:02d}",
                split=cell["split"],
                limit=None,
                output=evaluation_dir / filename,
                transcript_dir=None,
                stress=False,
                temperature=0.0,
                max_steps=int(eval_config.get("max_steps", 24)),
                max_tokens=int(eval_config.get("max_tokens", 200)),
                keep_last=int(config.get("keep_last", 2)),
                quiet=True,
                use_cache=True,
                seed=int(config.get("seed", 20260902)),
                difficulty=int(cell["difficulty"]),
                family_quotas=dict(cell["per_family"]),
            )

        with _reuse_loaded_policy(model, tokenizer, view, resolved_spec):
            result = run_block_ablation(
                view,
                blocks=args.blocks,
                screen=screen,
                evaluate_condition=evaluate_condition,
                progress=log.progress,
            )
        payload = {
            "command": shlex.join(sys.argv),
            "model": resolved,
            "adapter": str(args.adapter.resolve()),
            "screen": {"path": str(args.screen.resolve()), "cells": screen},
            "block_count": args.blocks,
            "control": "full_adapter success rate",
            **result,
        }
        (args.output / "ablation.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (args.output / "ablation.md").write_text(
            _render_ablation_markdown(payload) + "\n", encoding="utf-8"
        )
        log.info(
            "wrote",
            json=str(args.output / "ablation.json"),
            md=str(args.output / "ablation.md"),
        )


# --------------------------------------------------------------------------- CLI


def _reject_ablation_static_flags(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.no_base is not None or args.top is not None or args.readout_layers is not None:
        parser.error("--no-base, --top, and --readout-layers require static analysis")


def main() -> None:
    from local_llm_lab.probes.guard import add_gpu_arguments, require_idle_gpu

    parser = argparse.ArgumentParser(
        description="P5: analyse what each LoRA adapter wrote (exact model diff, design §8)."
    )
    parser.add_argument("--adapters", nargs="+", type=Path)
    parser.add_argument("--adapter", type=Path, help="One adapter directory for --ablate.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="mlx-community/Qwen2.5-Coder-3B-Instruct-4bit")
    parser.add_argument(
        "--no-base",
        action="store_true",
        default=None,
        help="Skip everything that needs the base model: relative norms and direction readouts.",
    )
    parser.add_argument("--top", type=int, help="Singular vectors per module (default: 16).")
    parser.add_argument(
        "--readout-layers",
        help="Comma-separated blocks whose down_proj/o_proj update directions are read out.",
    )
    parser.add_argument(
        "--ablate",
        action="store_true",
        help="Evaluate full, empty, and leave-one-block-out adapter conditions.",
    )
    parser.add_argument("--blocks", type=int, help="Consecutive blocks for --ablate (default: 6).")
    parser.add_argument("--screen", type=Path, help="Pipeline config whose select.screen is evaluated.")
    add_gpu_arguments(parser)
    args = parser.parse_args()

    if args.ablate:
        if args.adapters:
            parser.error("--adapters belongs to static analysis; use --adapter with --ablate")
        _reject_ablation_static_flags(parser, args)
        if args.adapter is None or args.screen is None:
            parser.error("--ablate requires --adapter and --screen")
        args.blocks = 6 if args.blocks is None else args.blocks
        if args.blocks < 1:
            parser.error("--blocks must be positive")
        _run_ablation_cli(parser, args, require_idle_gpu)
        return
    if args.adapter is not None or args.blocks is not None or args.screen is not None:
        parser.error("--adapter, --blocks, and --screen require --ablate")
    if not args.adapters:
        parser.error("static analysis requires --adapters")
    args.no_base = False if args.no_base is None else args.no_base
    args.top = 16 if args.top is None else args.top
    args.readout_layers = "" if args.readout_layers is None else args.readout_layers
    readout_layers = [int(part) for part in args.readout_layers.split(",") if part.strip()]

    from local_llm_lab.runlog import RunLog, git_commit

    adapters = [str(Path(adapter).resolve()) for adapter in args.adapters]
    # R26(e), issue #35: opened before the GPU guard and the base-model load; `--no-base`
    # never resolves a ModelSpec, so the model string is recorded as the CLI gave it.
    with RunLog.open(
        args.output,
        name="adapter-delta",
        command=sys.argv,
        identity={
            "model": args.model,
            "adapters": adapters,
            "readout_layers": readout_layers,
            "top": args.top,
            "no_base": args.no_base,
            "git_commit": git_commit(),
        },
    ) as log:
        log.info(
            "selection",
            adapters=len(adapters),
            layers=readout_layers,
            top=args.top,
            no_base=args.no_base,
        )
        model = tokenizer = None
        if not args.no_base:
            require_idle_gpu(parser, args, "loading the base model for relative norms")
            from local_llm_lab.models import load_model_spec
            from local_llm_lab.pipeline.evaluate import load_policy

            spec = load_model_spec(args.model)
            log.info("loading policy", model=args.model, policy="base", hf_id=spec.hf_id)
            model, tokenizer, _view, _resolved = load_policy(spec, None)

        runs = []
        norm_cache: dict[str, float] = {}
        labels = _unique_labels(list(args.adapters))
        for index, (adapter, label) in enumerate(zip(args.adapters, labels, strict=True), start=1):
            report = analyse_adapter(adapter, model, top=args.top, norm_cache=norm_cache)
            report["label"] = label
            runs.append(report)
            log.info(
                f"{report['label']}: {len(report['modules'])} modules, "
                f"effective rank histogram {report['effective_rank_histogram']}"
            )
            log.progress(index, len(labels), f"adapter {label}")

        payload: dict[str, Any] = {
            "model": args.model,
            "has_base": model is not None,
            "runs": runs,
            "comparison": compare_adapters(args.adapters, top=args.top)
            if len(args.adapters) > 1
            else None,
            "readouts": None,
        }
        if model is not None and readout_layers:
            payload["readouts"] = readout_update_directions(
                ArchitectureView.from_model(model), tokenizer, args.adapters[0], readout_layers
            )

        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "delta.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (args.output / "delta.md").write_text(
            render_markdown(payload) + "\n", encoding="utf-8"
        )
        log.info(
            "wrote",
            json=str(args.output / "delta.json"),
            md=str(args.output / "delta.md"),
        )
