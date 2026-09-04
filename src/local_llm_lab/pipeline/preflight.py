"""Deterministic, pre-load compatibility evidence for registered base models.

Artifact schema version 3 (R32) adds the two memory answers a version 2 artifact could not
give.  ``memory`` records ``registry_budget_gib`` (the declaration), ``device_working_set_gib``
(what ``mx.device_info()`` says the runtime will grant, or ``None`` with
``device_working_set_note`` saying why), the resolved ``budget_gib`` = the minimum of the two,
``budget_source`` naming which one bound, and ``within_budget`` judged against that minimum.

``training_footprint`` records the training peak for the longest trained row.  The estimate is
a *calibrated upper envelope* - ``peak_gib = slope * tokens + intercept``, fitted per
recurrence form to measured training peaks (issue #51 lane 2) - not an analytic sum of
retained recurrence state; see ``_CALIBRATION_POINTS`` for the measurements and why the
analytic sum was replaced.

The block carries ``batch_size`` with its ``batch_size_source``, ``chunk``, ``max_row_tokens``
and ``max_row_tokens_source``, the introspected ``linear_attention_layers`` /
``linear_attention_state_shape`` / ``linear_attention_state_shape_source``,
``grad_checkpoint`` with the ``retained_recurrence_layers`` it implies and the reason,
``headroom_fraction``, the ``recurrence_mode`` the arm's configuration selects with its
``recurrence_mode_source``, the ``calibration`` provenance and any
``calibration_domain_departures``, and one ``estimates`` entry per calibrated form
(``floor``, ``unrolled``, ``chunked``, ``chunkwise``) carrying ``chunk``, ``coefficients``,
the ``points`` they were fitted from, ``single_observation`` with its ``caveat``,
``estimated_train_peak_bytes``, ``estimated_train_peak_gib``, ``fits_with_headroom`` and
whether it ``gates``.  ``gated_estimate`` names the gating entry and ``passed`` is its
``fits_with_headroom``.  Without a row count the block is ``skipped`` with ``skip_reason`` and
``passed`` true, so ``agent-pipeline preflight --model <name>`` keeps its existing meaning; a
recurrence form with no calibration points is ``refused`` instead, with ``passed`` false.

The report's own ``passed`` is MODEL-LEVEL evidence only - residual equivalence passed, the
JVP is finite, and the loaded model sits within the resolved memory budget.  The training
footprint is deliberately outside it: the estimate describes what one arm's *training*
configuration would peak at, and a probe loads no optimiser, takes no gradient through a
training step and never reaches ``max_seq_length``, so a training peak cannot be evidence
against it.  The footprint gates ``require_preflight(consumer="training")`` instead, which
requires a block that was actually computed - not ``skipped``, not ``refused`` - and passed;
``consumer="view"`` never reads it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from math import ceil, sqrt
from pathlib import Path
from typing import Any, Literal, NamedTuple

from local_llm_lab.models import ModelSpec, ResolvedSpec, load_model_spec
from local_llm_lab.project import PROJECT_ROOT, configure_local_cache

__all__ = [
    "artifact_path",
    "cached_revision",
    "longest_row_tokens",
    "require_preflight",
    "run_residual_control",
    "run_preflight",
    "write_report",
]

# 3 (R32): the memory block records the registry and device budgets separately and the report
# carries a ``training_footprint`` block, which gates the training consumer rather than the
# report's own ``passed``; a version 2 artifact cannot answer either question.
_SCHEMA_VERSION = 3
_OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "preflight"
_FIXED_PROMPT = "Explain why a model preflight check protects a local training run."
_FIXED_MESSAGES = [{"role": "user", "content": _FIXED_PROMPT}]
_THINKING_MODES = ("unsupported", "off", "inference", "trained")
_TRAINED_SPLITS = ("train", "valid")
_LINEAR_ATTENTION_KIND = "linear_attention"
# R32(d): the estimate must clear the budget with room to spare, because it counts the terms
# that scale with the row and not every transient the runtime allocates around them.
_TRAINING_HEADROOM_FRACTION = 0.10
# ``mx.device_info()`` names the working set Metal will actually grant; the registry budget is
# a declaration of intent and may exceed it (22 GiB declared, 17.8 GiB granted on the M4 Pro).
_DEVICE_WORKING_SET_KEY = "max_recommended_working_set_size"
_NO_DEVICE_WORKING_SET = "no device working set reported; the registry value stands"
# The recurrence's state shape, read from the module that owns it (mlx-lm's ``GatedDeltaNet``
# stores each as a plain int) and, failing that, from the text module's own config dataclass.
_MODULE_STATE_FIELDS = (
    ("heads_v", "num_v_heads"),
    ("dim_v", "head_v_dim"),
    ("dim_k", "head_k_dim"),
)
_CONFIG_STATE_FIELDS = (
    ("heads_v", "linear_num_value_heads"),
    ("dim_v", "linear_value_head_dim"),
    ("dim_k", "linear_key_head_dim"),
)

# ---------------------------------------------------------------------------------------
# The calibrated training-peak envelope (issue #51 lane 2, R32(b) and (d)).
#
# The gate's estimate is a fit to measured training peaks, not an analytic sum of retained
# recurrence state.  Two findings from the measurement lane forced the change.
#
# 1. The analytic sum is wrong in *shape*, not merely in scale.
#    ``gated_delta_chunked.training_state_bytes`` retains ``ceil(T / C) + C`` states, which at
#    997 tokens is a 2.12x spread between chunk 128 and chunk 32.  Measured peaks at that row
#    length: 11.85, 11.85 and 12.66 GiB at chunks 32, 64 and 128 - a 1.07x spread, 1.12x on
#    the recurrence-only delta above the floor.  Chunk length barely moves real memory, and
#    the analytic model says it dominates, so the envelope below does not read the chunk at
#    all.  ``training_state_bytes`` is left in place: it is still the analytic *description*
#    of retained state, it is simply no longer what the gate estimates with.
# 2. It models a recurrence the arm will not run.  Binding the chunked form whenever a chunk
#    is configured ignores ``train.gated_delta_mode``, and the chunked form cannot train run
#    D's longest row at all (two OOM points below) while the chunkwise form can.  The mode is
#    therefore selected from the arm's configuration; see ``_select_recurrence_mode``.
#
# Editing ``_CALIBRATION_POINTS`` re-fits every envelope: the coefficients are derived from
# the table at import, never written down beside it, so the two cannot drift apart.
_CALIBRATION_DEVICE = (
    "Apple M4 Pro, 24 GiB unified memory, max_recommended_working_set_size 17.8 GiB"
)
_CALIBRATION_PROCEDURE = (
    "one process per point running the real mlx_lm train loop: batch 1, one row, one "
    "optimiser step, gradient checkpointing on, 4B backbone"
)
_CALIBRATION_UNIT = "GiB (bytes / 1024**3), the unit the resolved budget is in"
_CALIBRATION_BATCH_SIZE = 1
_CALIBRATION_GRAD_CHECKPOINT = True
_OK, _OOM = "ok", "oom"
# The shared base term: what one training step costs with the recurrence's backward pass out
# of the picture.  Every recurrence mode is this plus its own retained state.
_FLOOR_MODE = "floor"


class _CalibrationPoint(NamedTuple):
    """One measured training peak.

    ``peak_gib`` of an ``ok`` point is the observed peak.  ``peak_gib`` of an ``oom`` point is
    a LOWER bound - the run died at that allocation and its true peak is unknown but larger -
    so it constrains the envelope from below and must never be read as an observation.
    """

    mode: str
    tokens: int
    chunk: int | None
    peak_gib: float
    outcome: str


_CALIBRATION_POINTS: tuple[_CalibrationPoint, ...] = (
    _CalibrationPoint(_FLOOR_MODE, 997, None, 5.36, _OK),
    _CalibrationPoint(_FLOOR_MODE, 1591, None, 6.61, _OK),
    _CalibrationPoint(_FLOOR_MODE, 2085, None, 7.88, _OK),
    _CalibrationPoint(_FLOOR_MODE, 2874, None, 9.90, _OK),
    _CalibrationPoint("unrolled", 495, None, 11.38, _OK),
    _CalibrationPoint("unrolled", 997, None, 19.17, _OOM),
    _CalibrationPoint("chunked", 997, 32, 11.85, _OK),
    _CalibrationPoint("chunked", 997, 64, 11.85, _OK),
    _CalibrationPoint("chunked", 997, 128, 12.66, _OK),
    _CalibrationPoint("chunked", 1591, 64, 16.93, _OK),
    _CalibrationPoint("chunked", 2085, 64, 19.29, _OOM),
    _CalibrationPoint("chunked", 2874, 64, 19.22, _OOM),
    _CalibrationPoint("chunkwise", 2874, 64, 10.12, _OK),
)

# How a mode's cost *above the floor* is allowed to move with the row length.  Each shape is
# the recurrence's own retained-state behaviour, not a curve picked to fit the points.
_RECURRENCE_SHAPES = {
    # The library's reference loop leaves one state per token in the autograd graph, so its
    # cost above the floor is proportional to the row.  The OOM point is what proves the
    # shape: a flat offset fitted to the 495-token point predicts 12.60 GiB at 997 tokens,
    # where the run demonstrably died past 19.17.
    "unrolled": "proportional",
    # Measured at two row lengths and three chunk lengths.  The chunk spread is folded into
    # the intercept rather than modelled, because it is far smaller than the row's effect.
    "chunked": "affine",
    # ONE observation.  Its slope is the floor's - the chunkwise form retains a bounded number
    # of states rather than one per token - and the single point anchors the offset.  Every
    # estimate carries ``single_observation`` and a caveat so nobody reads it as settled.
    "chunkwise": "flat",
}

# What a single-observation envelope's slope rests on, said plainly in the artifact.
_SHAPE_CAVEATS = {
    "proportional": (
        "Its slope is that point divided by its row length, on the recurrence's own "
        "one-state-per-token shape rather than on a measured trend."
    ),
    "flat": (
        "Its slope is the no-recurrence floor's, not measured for this form; the point fixes "
        "only the offset."
    ),
}

# ``train.gated_delta_mode`` names an installer (``cli._GATED_DELTA_INSTALLERS``) while the
# calibration table names a measured recurrence form.  This mapping is the whole set of mode
# names the gate will estimate for; anything else is refused by name rather than guessed at.
_DEFAULT_GATED_DELTA_MODE = "checkpointed"
_CONFIGURED_MODE_CALIBRATION = {"checkpointed": "chunked", "chunkwise": "chunkwise"}


class _Envelope(NamedTuple):
    """An upper envelope ``peak_gib = slope * tokens + intercept`` for one recurrence mode."""

    mode: str
    shape: str
    slope_gib_per_token: float
    intercept_gib: float
    recurrence_slope_gib_per_token: float
    recurrence_intercept_gib: float
    points: tuple[_CalibrationPoint, ...]
    single_observation: bool


def _calibration_points(mode: str, outcome: str | None = None) -> tuple[_CalibrationPoint, ...]:
    """The table's entries for one mode, in table order."""
    return tuple(
        point
        for point in _CALIBRATION_POINTS
        if point.mode == mode and (outcome is None or point.outcome == outcome)
    )


def _least_squares_slope(samples: Sequence[tuple[int, float]]) -> float:
    """Least-squares slope of the sampled value against the row length."""
    mean_tokens = sum(tokens for tokens, _ in samples) / len(samples)
    mean_value = sum(value for _, value in samples) / len(samples)
    spread = sum((tokens - mean_tokens) ** 2 for tokens, _ in samples)
    if spread == 0.0:
        raise ValueError("an affine calibration needs points at two or more row lengths")
    return (
        sum((tokens - mean_tokens) * (value - mean_value) for tokens, value in samples) / spread
    )


def _fit(shape: str, samples: Sequence[tuple[int, float]]) -> tuple[float, float]:
    """Fit ``(slope, intercept)`` of the declared shape, sitting at or above every sample.

    Least squares alone runs *through* a cloud of points; a memory gate needs a line that runs
    over it.  The affine fit therefore keeps the least-squares slope - the trend the points
    actually show - and lifts the intercept by the largest shortfall, so the binding point is
    named by the data rather than chosen by hand.
    """
    if shape == "proportional":
        return max(value / tokens for tokens, value in samples), 0.0
    if shape == "flat":
        return 0.0, max(value for _, value in samples)
    slope = _least_squares_slope(samples)
    return slope, max(value - slope * tokens for tokens, value in samples)


def _close_envelope(
    slope: float, intercept: float, samples: Sequence[tuple[int, float]]
) -> float:
    """Lift ``intercept`` until no sample sits above ``slope * tokens + intercept``.

    The fits above are exact in real arithmetic, but a mode's coefficients are the sum of the
    floor's and its own, and that sum can land a fraction of an ULP under the very point it
    was fitted to.  This pass removes that, and is a no-op wherever the fit already clears.
    """
    shortfall = max(value - (slope * tokens + intercept) for tokens, value in samples)
    return intercept + shortfall if shortfall > 0.0 else intercept


def _build_envelopes() -> dict[str, _Envelope]:
    """Fit one envelope per mode: the shared no-recurrence floor plus that mode's own state."""
    floor_ok = _calibration_points(_FLOOR_MODE, _OK)
    floor_samples = [(point.tokens, point.peak_gib) for point in floor_ok]
    floor_slope, floor_intercept = _fit("affine", floor_samples)
    floor_intercept = _close_envelope(floor_slope, floor_intercept, floor_samples)
    envelopes = {
        _FLOOR_MODE: _Envelope(
            mode=_FLOOR_MODE,
            shape="affine",
            slope_gib_per_token=floor_slope,
            intercept_gib=floor_intercept,
            recurrence_slope_gib_per_token=0.0,
            recurrence_intercept_gib=0.0,
            points=_calibration_points(_FLOOR_MODE),
            single_observation=len(floor_ok) == 1,
        )
    }
    for mode, shape in _RECURRENCE_SHAPES.items():
        mode_ok = _calibration_points(mode, _OK)
        samples = [(point.tokens, point.peak_gib) for point in mode_ok]
        recurrence_slope, recurrence_intercept = _fit(
            shape,
            [(tokens, peak - (floor_slope * tokens + floor_intercept)) for tokens, peak in samples],
        )
        slope = floor_slope + recurrence_slope
        intercept = _close_envelope(slope, floor_intercept + recurrence_intercept, samples)
        envelopes[mode] = _Envelope(
            mode=mode,
            shape=shape,
            slope_gib_per_token=slope,
            intercept_gib=intercept,
            recurrence_slope_gib_per_token=recurrence_slope,
            recurrence_intercept_gib=intercept - floor_intercept,
            points=_calibration_points(mode),
            single_observation=len(mode_ok) == 1,
        )
    return envelopes


_ENVELOPES = _build_envelopes()


def _estimated_peak_gib(mode: str, tokens: int) -> float:
    """The calibrated upper envelope on the training peak, in GiB."""
    envelope = _ENVELOPES[mode]
    return envelope.slope_gib_per_token * tokens + envelope.intercept_gib


def artifact_path(spec: ModelSpec, output_root: Path | None = None) -> Path:
    """Return the direct, stable evidence location for one registered model."""
    return (output_root or _OUTPUT_DIRECTORY) / f"{spec.name}.json"


def cached_revision(spec: ModelSpec) -> str | None:
    """Read the pinned local Hub revision without importing or loading a model."""
    configure_local_cache()
    safe_id = spec.hf_id.replace("/", "--")
    path = Path(os.environ["HF_HOME"]) / "hub" / f"models--{safe_id}" / "refs" / "main"
    try:
        revision = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return revision or None


def longest_row_tokens(
    data_dir: Path,
    tokenizer: Any,
    *,
    max_seq_length: int,
    splits: Sequence[str] = _TRAINED_SPLITS,
) -> int:
    """Return the token count of the longest row the trainer will actually see.

    The count comes from ``RenderedRowsDataset``, the same class ``stage_train`` hands to
    mlx-lm, so the number the memory gate uses is the number the recurrence will run on -
    truncation at ``max_seq_length`` included.  Only the splits training reads are counted:
    a test row nothing trains on is no reason to size a training footprint (the B4 start
    failure was exactly a test row deciding a training question).
    """
    from local_llm_lab.tuner_data import load_rendered_splits

    datasets = load_rendered_splits(
        Path(data_dir), tokenizer, max_seq_length=max_seq_length, splits=tuple(splits)
    )
    lengths = [len(dataset[index]) for dataset in datasets for index in range(len(dataset))]
    if not lengths:
        raise ValueError(f"{data_dir} holds no rows in splits {tuple(splits)}")
    return max(lengths)


def write_report(report: Mapping[str, Any], path: Path) -> Path:
    """Persist stable preflight metadata without the array-report writer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_preflight(
    model_name: str,
    *,
    loader: Callable[..., tuple[Any, Any, Any, ResolvedSpec]] | None = None,
    output_root: Path | None = None,
    spec_loader: Callable[[str], ModelSpec] | None = None,
    view_factory: Callable[[Any], Any] | None = None,
    resolver: Callable[[ModelSpec, Any, Any], ResolvedSpec] | None = None,
    jvp: Callable[..., Any] | None = None,
    revision_reader: Callable[[ModelSpec], str | None] | None = None,
    array_api: Any | None = None,
    device_info: Callable[[], Mapping[str, Any]] | None = None,
    data_dir: Path | None = None,
    max_row_tokens: int | None = None,
    gated_delta_chunk: int | None = None,
    gated_delta_mode: str | None = None,
) -> Path:
    """Inspect one registered model and write a deterministic compatibility artifact.

    The optional collaborators make this model-loading stage wholly fake-testable.  The default
    loader and numerical runtime remain late imports so all ordinary CLI paths stay lightweight.
    ``view_factory`` and ``resolver`` override what the loader already returns; leaving them
    unset keeps preflight on exactly the view and resolved spec every other stage receives.

    R32 adds five training-memory inputs, every one of them optional so
    ``agent-pipeline preflight --model <name>`` keeps its existing meaning.  ``device_info``
    supplies the runtime's recommended working set (``mx.device_info()`` by default), which
    caps the registry budget.  ``data_dir`` or ``max_row_tokens`` supplies the longest training
    row; without a row count the footprint records why it was skipped rather than estimating on
    a number it lacks, and ``require_preflight(consumer="training")`` then refuses the artifact
    until one is supplied.  ``gated_delta_chunk`` and ``gated_delta_mode`` are the arm's
    ``train.`` fields of those names, and together they select which measured recurrence form
    the estimate is drawn from.

    The footprint never fails this command: it describes one arm's training configuration, so a
    failed, skipped or refused estimate is written into its own block and the command still
    exits zero.  Only model-level evidence sets the report's ``passed``.
    """
    if spec_loader is None:
        spec_loader = load_model_spec
    spec = spec_loader(model_name)
    if loader is None:
        loader = _default_loader
    if jvp is None:
        from local_llm_lab.pipeline.jlens import jacobian_vector_product

        jvp = jacobian_vector_product
    if revision_reader is None:
        revision_reader = cached_revision
    if array_api is None:
        import mlx.core as mx

        array_api = mx
    if device_info is None:
        device_info = getattr(array_api, "device_info", None)

    model, tokenizer, view, resolved = loader(spec, None, lazy=True)
    if view_factory is not None:
        view = view_factory(model)
    if resolver is not None:
        resolved = resolver(spec, model, tokenizer)
    revision = resolved.snapshot_revision or revision_reader(spec)
    if not revision:
        raise SystemExit(
            f"no cached revision for {spec.name}; run agent-pipeline preflight --model {spec.name} "
            "only after the local snapshot is available"
        )

    token_ids = _fixed_token_ids(tokenizer, array_api)
    residual = _residual_equivalence(view, token_ids, array_api)
    jvp_result = _jvp_result(view, token_ids, jvp, array_api)
    prompts = _thinking_prompts(spec, tokenizer)
    parameter_bytes = _parameter_tree_bytes(model)
    activation_bytes = _activation_bytes(spec, view)
    memory = _memory_block(
        spec,
        parameter_bytes=parameter_bytes,
        activation_bytes=activation_bytes,
        device_info=device_info,
    )
    row_tokens, row_tokens_source = _resolve_row_tokens(
        spec, tokenizer, data_dir=data_dir, max_row_tokens=max_row_tokens
    )
    footprint = _training_footprint(
        spec,
        view,
        budget_gib=memory["budget_gib"],
        max_row_tokens=row_tokens,
        max_row_tokens_source=row_tokens_source,
        gated_delta_chunk=gated_delta_chunk,
        gated_delta_mode=gated_delta_mode,
    )
    cache = list(view.make_cache())
    layer_counts = _layer_counts(view)

    report = {
        "architecture": {
            "hidden_size": int(view.hidden_size),
            "layer_counts": layer_counts,
            "num_layers": int(view.num_layers),
            "tie_word_embeddings": bool(view.tie_word_embeddings),
            "vocab_size": int(view.vocab_size),
        },
        "cache": {
            "entry_types": [type(entry).__name__ for entry in cache],
            "strategy": resolved.cache_strategy,
            "strategy_reason": resolved.cache_strategy_reason,
        },
        "hf_id": spec.hf_id,
        "jvp": jvp_result,
        "lora": {
            "keys": list(resolved.lora_keys),
            "trainable_parameters": int(resolved.trainable_parameters),
        },
        "memory": memory,
        "model_name": spec.name,
        "passed": False,
        "residual_equivalence": residual,
        "schema_version": _SCHEMA_VERSION,
        "snapshot_revision": revision,
        "thinking_prompts": prompts,
        "training_footprint": footprint,
    }
    # Model-level evidence only.  ``footprint["passed"]`` is deliberately absent: it judges one
    # arm's training configuration, and a probe - which loads no optimiser and never reaches
    # ``max_seq_length`` - must not be refused a model over a training peak it never allocates.
    # ``require_preflight(consumer="training")`` is where the footprint gates instead.
    report["passed"] = bool(
        residual["passed"] and jvp_result["finite"] and report["memory"]["within_budget"]
    )
    path = write_report(report, artifact_path(spec, output_root))
    if report["passed"]:
        return path
    raise SystemExit(f"preflight failed for {spec.name}; evidence written to {path}")


def run_residual_control(
    model_name: str,
    *,
    output_path: Path,
    loader: Callable[..., tuple[Any, Any, Any, ResolvedSpec]] | None = None,
    spec_loader: Callable[[str], ModelSpec] | None = None,
    view_factory: Callable[[Any], Any] | None = None,
    revision_reader: Callable[[ModelSpec], str | None] | None = None,
    array_api: Any | None = None,
) -> Path:
    """Write one narrow, no-cache residual control without broader preflight work."""
    if spec_loader is None:
        spec_loader = load_model_spec
    spec = spec_loader(model_name)
    if loader is None:
        loader = _default_loader
    if revision_reader is None:
        revision_reader = cached_revision
    if array_api is None:
        import mlx.core as mx

        array_api = mx
    model, tokenizer, view, _resolved = loader(spec, None, lazy=True)
    if view_factory is not None:
        view = view_factory(model)
    revision = revision_reader(spec)
    if not revision:
        raise SystemExit(f"no cached revision for {spec.name}; cannot write residual control")
    ids = _fixed_token_ids(tokenizer, array_api)
    comparisons = _residual_comparisons(view, ids, array_api)
    report = {
        "fp32_manual_vs_native": comparisons["fp32_manual_vs_native"],
        "hf_id": spec.hf_id,
        "model_name": spec.name,
        "native_manual_vs_native": comparisons["native_manual_vs_native"],
        "schema_version": _SCHEMA_VERSION,
        "snapshot_revision": revision,
        "token_identity": {
            "ids": [int(token) for token in ids[0].tolist()],
            "token_count": int(ids.shape[1]),
        },
    }
    return write_report(report, output_path)


def require_preflight(
    spec: ModelSpec,
    *,
    consumer: Literal["view", "training"] = "view",
    skip: bool = False,
    output_root: Path | None = None,
    revision_reader: Callable[[ModelSpec], str | None] | None = None,
    action: Callable[[], Any] | None = None,
) -> dict[str, Any] | None:
    """Reject stale, malformed, or failed evidence before callers can load a model.

    Everything up to and including ``_training_evidence_passed`` is required of every consumer.
    The two consumer branches below add what only one of them needs: the training footprint for
    a run that will hold an optimiser and a backward graph, and the view equivalence for a
    consumer that reads the residual stream.  Neither may be demanded of the other.

    ``action`` is deliberately called only after validation and exists solely as a narrow test
    seam proving the guard's order.  Production callers leave it unset.
    """
    if consumer not in {"view", "training"}:
        _reject(spec, f"unknown preflight consumer {consumer!r}")
    if skip:
        return None
    if revision_reader is None:
        revision_reader = cached_revision
    path = artifact_path(spec, output_root)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _reject(spec, "preflight artifact is missing")
    except (OSError, json.JSONDecodeError):
        _reject(spec, "preflight artifact is malformed or unreadable")
    if not isinstance(record, dict):
        _reject(spec, "preflight artifact is not a JSON object")
    if record.get("schema_version") != _SCHEMA_VERSION:
        _reject(spec, "preflight artifact schema version is missing or unsupported")
    if record.get("model_name") != spec.name:
        _reject(spec, "preflight artifact model name does not match the registered model")
    if record.get("hf_id") != spec.hf_id:
        _reject(spec, "preflight artifact hf_id does not match the registered model")
    revision = revision_reader(spec)
    if not revision:
        _reject(spec, "cached snapshot revision is absent")
    if record.get("snapshot_revision") != revision:
        _reject(spec, "preflight artifact snapshot revision is stale")
    if not _training_evidence_passed(record):
        _reject(spec, "preflight training evidence did not pass")
    if consumer == "training":
        footprint_reason = _footprint_rejection(record)
        if footprint_reason is not None:
            _reject(spec, footprint_reason)
    if consumer == "view" and not _view_evidence_passed(record):
        _reject(spec, "preflight view evidence did not pass")
    if action is not None:
        action()
    return record


def _default_loader(
    spec: ModelSpec, adapter: Path | None, *, lazy: bool
) -> tuple[Any, Any, Any, ResolvedSpec]:
    """Route preflight through the one shared policy loader so both paths agree."""
    from local_llm_lab.pipeline.evaluate import load_policy

    return load_policy(spec, adapter, lazy=lazy)


def _fixed_token_ids(tokenizer: Any, array_api: Any) -> Any:
    try:
        ids = list(tokenizer.encode(_FIXED_PROMPT, add_special_tokens=False))
    except TypeError:
        ids = list(tokenizer.encode(_FIXED_PROMPT))
    if not ids:
        raise ValueError("the preflight prompt tokenized to no token ids")
    ids = (ids * ceil(64 / len(ids)))[:64]
    return array_api.array(ids).astype(array_api.int32)[None, :]


def _residual_comparisons(view: Any, ids: Any, array_api: Any) -> dict[str, dict[str, Any]]:
    """Measure both R18a comparisons against the one native forward reference.

    ``native_manual_vs_native`` shares the reference's dtype, so its derived tolerance is the
    shared dtype's rounding budget and any structural defect lands far outside it.  The
    ``fp32_manual_vs_native`` gap mixes a genuine precision difference into the error, which is
    why it may be reported but never gated.
    """
    fp32_manual = _manual_final_residual(view, ids)
    native_manual = view.diagnostic_native_final_residual(ids)
    native_reference = _native_final_residual(view, ids)
    return {
        "fp32_manual_vs_native": _residual_metrics(
            fp32_manual, native_reference, num_layers=view.num_layers, array_api=array_api
        ),
        "native_manual_vs_native": _residual_metrics(
            native_manual, native_reference, num_layers=view.num_layers, array_api=array_api
        ),
    }


_NATIVE_FROBENIUS_TOLERANCE = 1e-4


def _native_gate_passed(metrics: Mapping[str, Any]) -> bool:
    """Both R18a conditions: the derived elementwise floor AND the Frobenius-relative bound.

    The derived floor alone would pass a distributed deviation as large as its relative
    tolerance (about 6e-2 on a deep bfloat16 model) - exactly the signature of a wrong norm
    weight, a mis-scaled residual, or a mask defect.  Failing the floor implies failing the
    bound, so the conjunction only closes that distributed band; it never loosens the gate.
    """
    return bool(
        metrics["within_tolerance"]
        and metrics["frobenius_relative_error"] <= _NATIVE_FROBENIUS_TOLERANCE
    )


def _residual_equivalence(view: Any, ids: Any, array_api: Any) -> dict[str, Any]:
    """Gate on the native-dtype loop; record the float32 gap as evidence, never as a gate."""
    comparisons = _residual_comparisons(view, ids, array_api)
    fp32 = comparisons["fp32_manual_vs_native"]
    native = comparisons["native_manual_vs_native"]
    return {
        "criterion": "native_dtype_rms_roundoff_and_frobenius_relative",
        "fp32_manual_vs_native": {
            **fp32,
            "gates": False,
            "purpose": (
                "precision-gap measurement: the float32 capture path against the native "
                "forward, reported for the capture-dtype decision and never gated"
            ),
        },
        "frobenius_relative_tolerance": _NATIVE_FROBENIUS_TOLERANCE,
        "gated_comparison": "native_manual_vs_native",
        "native_manual_vs_native": {
            **native,
            "gates": True,
            "purpose": (
                "structural equivalence: the manual block loop in the model's native dtype "
                "against the native forward, expected exact or within both the derived "
                "rounding floor and the Frobenius-relative bound"
            ),
        },
        "passed": _native_gate_passed(native),
        "token_count": int(ids.shape[1]),
    }


def _manual_final_residual(view: Any, ids: Any) -> Any:
    manual = view.embed(ids)
    masks = view.masks(manual, None)
    for index in range(view.num_layers):
        manual = view.run_block(index, manual, masks, None)
    return view.final_norm(manual)


def _residual_metrics(
    actual: Any, reference: Any, *, num_layers: int, array_api: Any
) -> dict[str, Any]:
    """Describe an error against the precision and scale of its native reference.

    Frobenius norms accumulate in float32 so a low-precision reference cannot corrupt the
    measurement of its own error (R18a reports Frobenius relative plus elementwise maxima).
    """
    if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers <= 0:
        raise ValueError(f"num_layers must be a positive integer; got {num_layers!r}")
    info = array_api.finfo(reference.dtype)
    scale = _scalar(array_api.max(array_api.abs(reference)))
    floor = max(scale, float(info.smallest_normal))
    rounding_steps = 2 * num_layers + 1
    relative_tolerance = sqrt(rounding_steps) * float(info.eps)
    absolute_tolerance = relative_tolerance * floor
    delta = actual - reference
    max_abs_error = _scalar(array_api.max(array_api.abs(delta)))
    frobenius_error = _scalar(
        array_api.sqrt(array_api.sum(array_api.square(delta.astype(array_api.float32))))
    )
    frobenius_scale = max(
        _scalar(
            array_api.sqrt(array_api.sum(array_api.square(reference.astype(array_api.float32))))
        ),
        float(info.smallest_normal),
    )
    return {
        "absolute_tolerance": absolute_tolerance,
        "frobenius_relative_error": frobenius_error / frobenius_scale,
        "max_abs_error": max_abs_error,
        "max_relative_error": max_abs_error / floor,
        "reference_dtype": str(reference.dtype),
        "reference_epsilon": float(info.eps),
        "reference_scale": scale,
        "relative_tolerance": relative_tolerance,
        "rounding_steps": rounding_steps,
        "within_tolerance": max_abs_error <= absolute_tolerance,
    }


def _native_final_residual(view: Any, ids: Any) -> Any:
    native = view.text_module(ids)
    if hasattr(native, "last_hidden_state"):
        return native.last_hidden_state
    if isinstance(native, tuple):
        return native[0]
    return native


def _jvp_result(view: Any, ids: Any, jvp: Callable[..., Any], array_api: Any) -> dict[str, Any]:
    layer = view.num_layers // 2
    primal = view.residuals(ids, [layer])[layer].astype(array_api.float32)
    tangent = array_api.ones_like(primal).astype(array_api.float32)
    method = "forward"
    try:
        result = jvp(view, layer, primal, tangent, method=method)
        finite = _is_finite(result, array_api)
    except Exception:
        finite = False
    if not finite:
        method = "finite_difference"
        result = jvp(view, layer, primal, tangent, method=method)
        finite = _is_finite(result, array_api)
    if not finite:
        return {"finite": False, "layer": layer, "method": method}
    return {"finite": True, "layer": layer, "method": method}


def _thinking_prompts(spec: ModelSpec, tokenizer: Any) -> list[dict[str, Any]]:
    prompts = []
    for mode in _THINKING_MODES:
        kwargs = dict(spec.chat.template_kwargs)
        if mode == "unsupported":
            kwargs.pop("enable_thinking", None)
        else:
            kwargs["enable_thinking"] = mode in {"inference", "trained"}
        prompt = tokenizer.apply_chat_template(
            _FIXED_MESSAGES, add_generation_prompt=True, tokenize=False, **kwargs
        )
        try:
            count = len(tokenizer.encode(prompt, add_special_tokens=False))
        except TypeError:
            count = len(tokenizer.encode(prompt))
        prompts.append({"mode": mode, "prompt": prompt, "token_count": count})
    return prompts


def _parameter_tree_bytes(model: Any) -> int:
    """Count materialized parameter buffers without evaluating or dequantizing them."""
    parameters = model.parameters() if callable(getattr(model, "parameters", None)) else model
    return sum(_leaf_nbytes(value) for value in _tree_values(parameters))


def _tree_values(value: Any):
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _tree_values(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _tree_values(child)
    else:
        yield value


def _leaf_nbytes(value: Any) -> int:
    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    size = getattr(value, "size", None)
    itemsize = getattr(value, "itemsize", None)
    if size is not None and itemsize is not None:
        return int(size) * int(itemsize)
    raise ValueError(f"parameter leaf {type(value).__name__} has no byte size")


def _activation_bytes(spec: ModelSpec, view: Any) -> int:
    batch_size = int(spec.train["batch_size"])
    max_seq_length = int(spec.train["max_seq_length"])
    return batch_size * max_seq_length * int(view.hidden_size) * int(view.num_layers) * 4


def _memory_block(
    spec: ModelSpec,
    *,
    parameter_bytes: int,
    activation_bytes: int,
    device_info: Callable[[], Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Resolve R32(b)'s budget: the registry declaration capped by what the device grants.

    Both numbers are recorded because they answer different questions - what the recipe asked
    for and what this machine will give it - and the B4 failure was the gate judging 3.85 GiB
    against a 22 GiB declaration the device never honoured.  The registry file keeps its
    declared intent; the minimum is resolved here, at the only point that knows the device.
    """
    device_gib, note = _device_working_set_gib(device_info)
    registry_gib = float(spec.memory_budget_gib)
    budget_gib = registry_gib if device_gib is None else min(registry_gib, device_gib)
    total_bytes = parameter_bytes + activation_bytes
    total_gib = total_bytes / 1024**3
    return {
        "activation_bytes": activation_bytes,
        "budget_gib": budget_gib,
        "budget_source": "registry" if budget_gib == registry_gib else "device",
        "device_working_set_gib": device_gib,
        "device_working_set_note": note,
        "parameter_bytes": parameter_bytes,
        "registry_budget_gib": registry_gib,
        "total_bytes": total_bytes,
        "total_gib": total_gib,
        "within_budget": total_gib <= budget_gib,
    }


def _device_working_set_gib(
    device_info: Callable[[], Mapping[str, Any]] | None,
) -> tuple[float | None, str]:
    """Read the runtime's recommended working set, and say why when there is none.

    A runtime that cannot report one must leave the registry value standing rather than
    resolve a budget of zero, so every failure mode here returns ``None`` with its reason.
    """
    if device_info is None:
        return None, "no device info source available; the registry value stands"
    try:
        info = device_info()
    except Exception as error:  # pragma: no cover - defensive: runtime-specific failures
        return None, f"device info unavailable ({type(error).__name__}); the registry value stands"
    value = info.get(_DEVICE_WORKING_SET_KEY) if isinstance(info, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None, _NO_DEVICE_WORKING_SET
    return float(value) / 1024**3, f"{_DEVICE_WORKING_SET_KEY} reported by the runtime"


def _resolve_row_tokens(
    spec: ModelSpec,
    tokenizer: Any,
    *,
    data_dir: Path | None,
    max_row_tokens: int | None,
) -> tuple[int | None, str | None]:
    """Return the longest training row's token count and where the number came from."""
    if max_row_tokens is not None:
        if isinstance(max_row_tokens, bool) or not isinstance(max_row_tokens, int):
            raise ValueError(f"max_row_tokens must be an integer; got {max_row_tokens!r}")
        if max_row_tokens <= 0:
            raise ValueError(f"max_row_tokens must be positive; got {max_row_tokens}")
        return max_row_tokens, "max_row_tokens"
    if data_dir is None:
        return None, None
    tokens = longest_row_tokens(
        Path(data_dir), tokenizer, max_seq_length=int(spec.train["max_seq_length"])
    )
    return tokens, f"data:{data_dir}"


def _training_footprint(
    spec: ModelSpec,
    view: Any,
    *,
    budget_gib: float,
    max_row_tokens: int | None,
    max_row_tokens_source: str | None,
    gated_delta_chunk: int | None,
    gated_delta_mode: str | None,
) -> dict[str, Any]:
    """Estimate the training peak for the longest row (R32(d), R15 condition 1).

    The estimate is the calibrated upper envelope for the recurrence form the arm will run,
    evaluated at the longest trained row - a fit to measured peaks, not a sum of terms.  Every
    calibrated mode is recorded so a reader can see what the alternatives would cost; the one
    the configuration selects is the one that gates.

    What this block gates is the training consumer, in ``_footprint_rejection`` - never the
    report's own ``passed``, which is model-level evidence a probe may rely on.  Nothing here
    can be built without a row count, so a run without one records why it was skipped and
    keeps ``passed`` true, leaving ``agent-pipeline preflight --model <name>`` intact; that
    ``passed`` says only that nothing was measured, and a training run is refused on it.  A
    recurrence form with no calibration points is different in kind: it is refused outright,
    because guessing at an unmeasured form is exactly what this estimate exists to stop.
    """
    batch_size = int(spec.train["batch_size"])
    grad_checkpoint = bool(spec.train.get("grad_checkpoint", False))
    layers = sum(
        1
        for index in range(int(view.num_layers))
        if str(view.layer_kind(index)) == _LINEAR_ATTENTION_KIND
    )
    shape, shape_source = _linear_attention_state_shape(view, layers)
    retained = 1 if grad_checkpoint else layers
    mode, mode_source = _select_recurrence_mode(
        layers=layers, chunk=gated_delta_chunk, configured_mode=gated_delta_mode
    )
    block: dict[str, Any] = {
        "batch_size": batch_size,
        # The registry's declaration, which is what sizes ``activation_bytes`` too.  An arm
        # config may train at a smaller batch (B4 runs 1 against the registry's 2); the
        # envelope was calibrated at batch 1 and does not model batch, so a departure is
        # named in ``calibration_domain_departures`` rather than scaled for.
        "batch_size_source": "registry train.batch_size",
        "calibration": _calibration_record(),
        "calibration_domain_departures": _calibration_domain_departures(
            batch_size, grad_checkpoint
        ),
        "chunk": gated_delta_chunk,
        "estimates": {},
        "gated_estimate": None,
        "grad_checkpoint": grad_checkpoint,
        "headroom_fraction": _TRAINING_HEADROOM_FRACTION,
        "linear_attention_layers": layers,
        # Recorded as architecture evidence.  The envelope is a fit to measured peaks and does
        # not read the state shape, so a shape the introspection cannot find no longer skips
        # the gate - it only leaves this field null with its reason beside it.
        "linear_attention_state_shape": shape,
        "linear_attention_state_shape_source": shape_source,
        "max_row_tokens": max_row_tokens,
        "max_row_tokens_source": max_row_tokens_source,
        "passed": True,
        "recurrence_mode": mode,
        "recurrence_mode_source": mode_source,
        "refused": False,
        "retained_recurrence_layers": retained,
        "retained_recurrence_layers_reason": (
            "gradient checkpointing bounds the live recurrence graph to one decoder layer"
            if grad_checkpoint
            else "no gradient checkpointing: every linear-attention layer retains its graph"
        ),
        "skip_reason": None,
        "skipped": True,
    }
    if mode is None:
        block["refused"] = True
        block["passed"] = False
        block["skip_reason"] = mode_source
        return block
    if max_row_tokens is None:
        block["skip_reason"] = (
            "no training row token count; pass --data <dir> or --max-row-tokens <int>"
        )
        return block

    for name in (_FLOOR_MODE, *_RECURRENCE_SHAPES):
        block["estimates"][name] = _calibrated_estimate(
            name,
            max_row_tokens,
            chunk=gated_delta_chunk,
            budget_gib=budget_gib,
            gates=name == mode,
        )
    block["gated_estimate"] = mode
    block["passed"] = bool(block["estimates"][mode]["fits_with_headroom"])
    block["skipped"] = False
    return block


def _select_recurrence_mode(
    *, layers: int, chunk: int | None, configured_mode: str | None
) -> tuple[str | None, str]:
    """Choose the calibrated mode the run will actually take, and say where the choice came from.

    The name is validated before anything else, exactly as ``cli._training_backbone`` rejects
    an unknown ``train.gated_delta_mode`` before it looks at the chunk: a form the calibration
    lane never measured is refused by name rather than silently estimated as something else.
    """
    named = _DEFAULT_GATED_DELTA_MODE if configured_mode is None else str(configured_mode)
    calibrated = _CONFIGURED_MODE_CALIBRATION.get(named)
    if calibrated is None or calibrated not in _RECURRENCE_SHAPES:
        return None, (
            f"no calibration points for train.gated_delta_mode={named!r}; the measured "
            f"recurrence forms are {sorted(_CONFIGURED_MODE_CALIBRATION)}"
        )
    if layers == 0:
        return _FLOOR_MODE, "no linear-attention layers: this backbone runs no recurrence"
    if chunk is None:
        return "unrolled", (
            "no train.gated_delta_chunk: nothing is installed, so the library's unrolled "
            "reference loop runs"
        )
    source = "train.gated_delta_mode" + ("" if configured_mode is not None else " default")
    return calibrated, f"{source}={named}"


def _calibrated_estimate(
    mode: str, tokens: int, *, chunk: int | None, budget_gib: float, gates: bool
) -> dict[str, Any]:
    """One mode's envelope evaluated at this row, with everything needed to re-derive it."""
    envelope = _ENVELOPES[mode]
    peak_gib = _estimated_peak_gib(mode, tokens)
    return {
        "caveat": _envelope_caveat(envelope),
        # The arm's chunk, recorded where it applies.  The envelope does not read it: at 997
        # tokens the measured spread across chunks 32, 64 and 128 is 1.07x, against the 2.12x
        # the analytic retained-state sum predicts.
        "chunk": chunk if mode in ("chunked", "chunkwise") else None,
        "coefficients": {
            "intercept_gib": envelope.intercept_gib,
            "recurrence_intercept_gib": envelope.recurrence_intercept_gib,
            "recurrence_slope_gib_per_token": envelope.recurrence_slope_gib_per_token,
            "shape": envelope.shape,
            "slope_gib_per_token": envelope.slope_gib_per_token,
        },
        "estimated_train_peak_bytes": round(peak_gib * 1024**3),
        "estimated_train_peak_gib": peak_gib,
        "fits_with_headroom": peak_gib * (1.0 + _TRAINING_HEADROOM_FRACTION) <= budget_gib,
        "gates": gates,
        "points": [point._asdict() for point in envelope.points],
        "single_observation": envelope.single_observation,
    }


def _envelope_caveat(envelope: _Envelope) -> str | None:
    """Say, in the artifact, when an envelope rests on one measurement (R32(d) honesty).

    A single point fixes an offset and nothing else: the curve's slope comes from the shape
    the recurrence is known to have, not from the data, so the number must not be read as a
    well-determined fit.
    """
    if not envelope.single_observation:
        return None
    (point,) = _calibration_points(envelope.mode, _OK)
    return (
        f"single observation: the {envelope.mode} envelope rests on one measured point "
        f"({point.tokens} tokens, {point.peak_gib} GiB). {_SHAPE_CAVEATS[envelope.shape]} "
        "Every other row length is an extrapolation, and the envelope is not well determined."
    )


def _calibration_record() -> dict[str, Any]:
    """The provenance of the fit: where the numbers came from and under what conditions."""
    floor = _ENVELOPES[_FLOOR_MODE]
    return {
        "batch_size": _CALIBRATION_BATCH_SIZE,
        "device": _CALIBRATION_DEVICE,
        "floor_coefficients": {
            "intercept_gib": floor.intercept_gib,
            "shape": floor.shape,
            "slope_gib_per_token": floor.slope_gib_per_token,
        },
        "floor_points": [point._asdict() for point in floor.points],
        "grad_checkpoint": _CALIBRATION_GRAD_CHECKPOINT,
        "procedure": _CALIBRATION_PROCEDURE,
        "unit": _CALIBRATION_UNIT,
    }


def _calibration_domain_departures(batch_size: int, grad_checkpoint: bool) -> list[str]:
    """Name every way this arm sits outside the conditions every point was measured under.

    The envelope is a fit in one variable - the row length - so nothing else it was held
    fixed at can be scaled for.  Where the arm differs, the artifact says so instead.
    """
    departures = []
    if batch_size != _CALIBRATION_BATCH_SIZE:
        departures.append(
            f"train.batch_size is {batch_size}; every calibration point was measured at "
            f"batch {_CALIBRATION_BATCH_SIZE} and the envelope does not model batch"
        )
    if grad_checkpoint != _CALIBRATION_GRAD_CHECKPOINT:
        departures.append(
            "gradient checkpointing is off; every calibration point was measured with it on, "
            "and the envelope does not model the unbounded retained graph"
        )
    return departures


def _linear_attention_state_shape(view: Any, layers: int) -> tuple[dict[str, int] | None, str]:
    """Read the recurrence's ``(heads_v, dim_v, dim_k)`` from the loaded architecture.

    The shapes are properties of the model, not of the recipe, so they are introspected rather
    than declared in the registry: the module that owns the recurrence carries them, and the
    text module's config carries them under its own names when the module does not.
    """
    if not layers:
        return None, "no linear-attention layers"
    blocks = getattr(view, "blocks", None)
    if blocks is not None:
        for index in range(int(view.num_layers)):
            if str(view.layer_kind(index)) != _LINEAR_ATTENTION_KIND:
                continue
            for candidate in _module_candidates(blocks[index]):
                shape = _state_shape_from(candidate, _MODULE_STATE_FIELDS)
                if shape is not None:
                    return shape, "recurrence module"
            break
    shape = _state_shape_from(getattr(view, "text_module", None), _CONFIG_STATE_FIELDS)
    if shape is not None:
        return shape, "text module config"
    shape = _state_shape_from(
        getattr(getattr(view, "text_module", None), "args", None), _CONFIG_STATE_FIELDS
    )
    if shape is not None:
        return shape, "text module config"
    return None, "neither the recurrence module nor the text module config exposes it"


def _module_candidates(block: Any):
    """Yield a decoder block and its direct members, whichever way the block stores them."""
    yield block
    if isinstance(block, Mapping):
        yield from block.values()
        return
    try:
        members = vars(block)
    except TypeError:
        return
    yield from members.values()


def _state_shape_from(source: Any, fields: tuple[tuple[str, str], ...]) -> dict[str, int] | None:
    """Return the state shape only when every dimension is a positive integer."""
    if source is None:
        return None
    shape: dict[str, int] = {}
    for key, attribute in fields:
        value = getattr(source, attribute, None)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return None
        shape[key] = int(value)
    return shape


def _layer_counts(view: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for index in range(view.num_layers):
        kind = str(view.layer_kind(index))
        counts[kind] = counts.get(kind, 0) + 1
    return dict(sorted(counts.items()))


def _is_finite(value: Any, array_api: Any) -> bool:
    return bool(_scalar(array_api.all(array_api.isfinite(value))))


def _scalar(value: Any) -> float:
    item = getattr(value, "item", None)
    return float(item() if callable(item) else value)


def _training_evidence_passed(record: Mapping[str, Any]) -> bool:
    """The evidence EVERY consumer needs, whatever it means to do with the model.

    Named for the training run it was first written for, but a probe wants all of it too - the
    loaded model within budget, the four thinking prompts, the LoRA keys it may load an adapter
    over.  The one training-specific condition, the footprint, lives in ``_footprint_rejection``
    so that this predicate can keep running for every consumer.
    """
    memory = record.get("memory")
    if not isinstance(memory, Mapping) or memory.get("within_budget") is not True:
        return False
    prompts = record.get("thinking_prompts")
    if not isinstance(prompts, list):
        return False
    modes = [entry.get("mode") if isinstance(entry, Mapping) else None for entry in prompts]
    if modes != list(_THINKING_MODES):
        return False
    if not all(
        isinstance(entry, Mapping)
        and isinstance(entry.get("prompt"), str)
        and bool(entry["prompt"])
        and isinstance(entry.get("token_count"), int)
        and not isinstance(entry["token_count"], bool)
        and entry["token_count"] > 0
        for entry in prompts
    ):
        return False
    lora = record.get("lora")
    return bool(
        isinstance(lora, Mapping)
        and isinstance(lora.get("keys"), list)
        and lora["keys"]
        and all(isinstance(key, str) and key for key in lora["keys"])
        and isinstance(lora.get("trainable_parameters"), int)
        and not isinstance(lora["trainable_parameters"], bool)
        and lora["trainable_parameters"] > 0
    )


def _footprint_rejection(record: Mapping[str, Any]) -> str | None:
    """Why the training footprint bars a training run, or ``None`` when it clears one.

    R32(d) joins R15 condition 1: an estimate that failed its 10% headroom is a run that dies at
    step one.  An estimate that was never computed is no better - a ``skipped`` block records
    ``passed`` so ``agent-pipeline preflight --model <name>`` keeps its meaning, but passing a
    training run on it would clear the gate on a number nobody has.  Both are rejected here, and
    only here, because only a training consumer allocates a training peak.

    ``refused`` is read before ``skipped``: a refused block never reaches the point that clears
    ``skipped``, so testing ``skipped`` first would report a missing row count for a run whose
    real problem is a recurrence form nobody measured.
    """
    footprint = record.get("training_footprint")
    if not isinstance(footprint, Mapping):
        return "preflight artifact carries no training footprint block"
    if footprint.get("refused") is True:
        return "preflight training footprint refused the configured recurrence form as unmeasured"
    if footprint.get("skipped") is True:
        return (
            "preflight training footprint was skipped for want of a training row count "
            "(--data <dir> or --max-row-tokens <int>)"
        )
    if footprint.get("passed") is not True:
        return "preflight training footprint estimate does not fit the memory budget with headroom"
    return None


def _view_evidence_passed(record: Mapping[str, Any]) -> bool:
    residual = record.get("residual_equivalence")
    jvp = record.get("jvp")
    return bool(
        record.get("passed") is True
        and isinstance(residual, Mapping)
        and residual.get("passed") is True
        and isinstance(jvp, Mapping)
        and jvp.get("finite") is True
    )


def _reject(spec: ModelSpec, reason: str) -> None:
    raise SystemExit(
        f"{reason} for {spec.name}; run agent-pipeline preflight --model {spec.name} "
        "or use --skip-preflight-check to override"
    )
