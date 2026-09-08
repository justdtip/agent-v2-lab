"""Full-basis, same-position pre-norm Jacobians (requirements §§2/3.3).

Plans and resource decisions are pure Python. Native arithmetic is imported only
inside measurement calls, after the standalone CLI has acquired the primary lock.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def check_bounds(bounds: dict) -> dict:
    if set(bounds) != {"atol", "rtol"} or not all(
        np.isfinite(v) and v >= 0 for v in bounds.values()
    ):
        raise ValueError("explicit nonnegative finite atol/rtol bounds required")
    return dict(bounds)


def sample_positions(rows: list[dict], *, split: str, count: int, seed: int) -> list[dict]:
    """Equal span allocation; uniform prompt then token within span, with replacement.

    One draw is one prompt and one same source/target position. Balanced cycles
    make every running prefix differ by at most one draw between available spans.
    """
    if split not in {"fit", "held"} or count <= 0:
        raise ValueError("positive count and fit/held split required")
    available = defaultdict(dict)
    for i, row in enumerate(rows):
        if row["split"] != split:
            continue
        if not row["ids"] or len(row["ids"]) != len(row["spans"]):
            raise ValueError("token/span mismatch")
        for p, span in enumerate(row["spans"]):
            available[span].setdefault(i, []).append(p)
    if not available:
        raise ValueError(f"no {split} positions")
    rng = np.random.default_rng(seed)
    result, spans = [], sorted(available)
    while len(result) < count:
        for span in rng.permutation(spans).tolist():
            if len(result) == count:
                break
            i = int(rng.choice(sorted(available[span])))
            p = int(rng.choice(available[span][i]))
            row = rows[i]
            result.append(
                dict(
                    draw=len(result),
                    row=i,
                    position=p,
                    span=span,
                    source=row["source"],
                    index=row["index"],
                    split=split,
                    identity={
                        k: row[k] for k in ("task_id", "step_index", "token_start") if k in row
                    },
                )
            )
    return result


def make_plan(
    rows,
    *,
    layers,
    hidden_size,
    corpus_sha256,
    snapshot_sha256,
    seed,
    self_bounds,
    response_bounds,
    stability_bounds,
    held_count,
    working_set_bytes,
    initial_peak_bytes,
    step_coefficient=None,
):
    layers = list(layers)
    if len(layers) < 3 or layers != list(range(1, max(layers) + 1)) or hidden_size <= 0:
        raise ValueError("all nonfinal layers (at least three) and positive dimension required")
    if held_count < 150:
        raise ValueError("at least 150 held positions required for mean validation")
    memory_gate(initial_peak_bytes, working_set_bytes)
    fit = sample_positions(rows, split="fit", count=400, seed=seed)
    held = sample_positions(rows, split="held", count=held_count, seed=seed + 1)
    if {p["span"] for p in fit} != {p["span"] for p in held}:
        raise ValueError("held sampling must cover the fit spans")
    # Shortest fit prompt bounds the full-sequence reference's operational cost.
    self_row = min({p["row"] for p in fit}, key=lambda i: (len(rows[i]["ids"]), i))
    plan = dict(
        schema_version=1,
        allocator_cache_limit_bytes=0,
        rows_sha256=digest(rows),
        corpus_sha256=corpus_sha256,
        snapshot_sha256=snapshot_sha256,
        layers=layers,
        hidden_size=hidden_size,
        seeds=dict(
            positions=seed, held=seed + 1, self_directions=seed + 2, validation_directions=seed + 3
        ),
        fit_positions=fit,
        held_positions=held,
        sampling="equal span; uniform prompt then token; with replacement; same position",
        self_row=self_row,
        self_position=len(rows[self_row]["ids"]) - 1,
        self_layers=[layers[0], layers[len(layers) // 2], layers[-1]],
        self_bounds=check_bounds(self_bounds),
        response_bounds=check_bounds(response_bounds),
        stability_bounds=check_bounds(stability_bounds),
        working_set_bytes=working_set_bytes,
        initial_peak_bytes=initial_peak_bytes,
        batches=[8, 16, 32, 64],
        paths=["restore", "broadcast"],
        epsilon="float32: 0.01 * norm(full sequence primal) / norm(one tangent); zero primal: 0.01",
        stopping=dict(
            interval=10,
            diagnostic_floor=100,
            minimum=150,
            ceiling=400,
            consecutive=2,
            relative_frobenius=0.002,
        ),
    )

    if step_coefficient is not None:
        if not np.isfinite(step_coefficient) or step_coefficient <= 0:
            raise ValueError("finite positive position-step coefficient required")
        plan["step_coefficient"] = float(step_coefficient)
        plan["epsilon"] = (
            "float32: c * norm(selected position primal) / norm(one tangent); zero norm: stop"
        )
    return plan


def position_step_kwargs(plan):
    """Legacy plans replay unchanged; revised plans bind all four numerical stages."""
    return {"step_coefficient": plan["step_coefficient"]} if "step_coefficient" in plan else {}


def freeze_plan(path: Path, plan: dict) -> dict:
    frozen = dict(plan, plan_sha256=digest(plan))
    write_record(path, frozen)
    return frozen


def write_record(path: Path, record: dict) -> None:
    with Path(path).open("x") as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def read_plan(path: Path, rows: list[dict]) -> dict:
    plan = json.loads(Path(path).read_text())
    payload = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if digest(payload) != plan.get("plan_sha256"):
        raise ValueError("plan digest mismatch")
    if digest(rows) != plan["rows_sha256"]:
        raise ValueError("corpus rows changed after plan freeze")
    expected = make_plan(
        rows,
        layers=plan["layers"],
        hidden_size=plan["hidden_size"],
        corpus_sha256=plan["corpus_sha256"],
        snapshot_sha256=plan["snapshot_sha256"],
        seed=plan["seeds"]["positions"],
        self_bounds=plan["self_bounds"],
        response_bounds=plan["response_bounds"],
        stability_bounds=plan["stability_bounds"],
        held_count=len(plan["held_positions"]),
        working_set_bytes=plan["working_set_bytes"],
        initial_peak_bytes=plan["initial_peak_bytes"],
        **position_step_kwargs(plan),
    )
    if expected != payload:
        raise ValueError("plan does not match fixed protocol")
    return plan


@dataclass
class Convergence:
    n: int = 0
    mean: Any = None
    previous: Any = None
    stable_batches: int = 0
    curve: list = field(default_factory=list)
    accepted: bool = False
    reason: str = "sampling"
    span_means: dict = field(default_factory=dict)

    def add(self, matrix, *, span="all") -> bool:
        if self.reason != "sampling":
            raise ValueError("convergence already terminated")
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
            raise ValueError("finite square Jacobian required")
        if self.mean is not None and matrix.shape != self.mean.shape:
            raise ValueError("Jacobian shape changed")
        self.n += 1
        mean, count = self.span_means.get(span, (np.zeros_like(matrix), 0))
        self.span_means[span] = (mean + (matrix - mean) / (count + 1), count + 1)
        self.mean = np.mean([m for m, _ in self.span_means.values()], axis=0, dtype=np.float32)
        if self.n % 10 == 0:
            norm = float(np.linalg.norm(self.mean.astype(np.float64)))
            change = (
                None
                if self.previous is None
                else float(np.linalg.norm((self.mean - self.previous).astype(np.float64)))
            )
            relative = None if change is None or norm == 0 else change / norm
            stable = relative is not None and relative < 0.002
            self.stable_batches = self.stable_batches + 1 if stable and self.n >= 100 else 0
            self.curve.append(
                dict(
                    positions=self.n,
                    relative_change=relative,
                    decision_eligible=self.n >= 100,
                    stable_batches=self.stable_batches,
                )
            )
            self.previous = self.mean.copy()
            self.accepted = self.n >= 150 and self.stable_batches >= 2
            if self.accepted:
                self.reason = "converged"
            elif self.n == 400:
                self.reason = "ceiling_without_convergence"
        return self.accepted


def memory_gate(projected_bytes: float, working_set_bytes: float) -> None:
    if (
        not np.isfinite([projected_bytes, working_set_bytes]).all()
        or min(projected_bytes, working_set_bytes) <= 0
    ):
        raise ValueError("positive finite memory projection and working set required")
    if projected_bytes > 0.6 * working_set_bytes:
        raise ValueError("memory projection exceeds 0.6 working set; Director window required")


def time_gate(projected_seconds: dict) -> None:
    if not projected_seconds or any(
        not np.isfinite(t) or t <= 0 for t in projected_seconds.values()
    ):
        raise ValueError("finite positive per-layer time projections required")
    if any(t > 25 * 60 for t in projected_seconds.values()):
        raise ValueError("projected cost exceeds 25 minutes/layer; Director choice required")


class MemoryStopped(ValueError):
    """A measured workload invalidated its envelope; retain exact stop evidence."""

    def __init__(self, report):
        super().__init__(report["reason"])
        self.report = report


@dataclass
class WorkloadMemoryGuard:
    projected_bytes: float
    working_set_bytes: float
    read_peak: Any
    context: dict = field(default_factory=dict)
    recent: list = field(default_factory=list)
    batch_widths: set = field(default_factory=set)

    def __post_init__(self):
        memory_gate(self.projected_bytes, self.working_set_bytes)

    def __call__(self, workload, **details):
        peak = self.read_peak()
        if "actual_batch_size" in details:
            self.batch_widths.add(details["actual_batch_size"])
        observed = dict(
            workload=workload,
            **self.context,
            **details,
            peak_bytes=float(peak) if np.isfinite(peak) else None,
            projected_peak_bytes=self.projected_bytes,
            working_set_bytes=self.working_set_bytes,
            cap_bytes=0.6 * self.working_set_bytes,
        )
        # Keep bounded evidence rather than retaining millions of successful basis calls.
        self.recent.append(observed)
        self.recent[:] = self.recent[-8:]
        reason = None
        if not np.isfinite(peak) or peak < 0:
            reason = "nonfinite or negative measured memory peak"
        elif peak > 0.6 * self.working_set_bytes:
            reason = "measured memory peak exceeded 0.6 working set"
        elif peak > self.projected_bytes:
            reason = "memory projection falsified by measured workload peak"
        if reason:
            raise MemoryStopped(dict(reason=reason, **observed, recent_workloads=list(self.recent)))


def _check_workload(guard, workload, **details):
    if guard is not None:
        guard(workload, **details)


def unit_directions(count: int, dimension: int, seed: int) -> np.ndarray:
    directions = np.random.default_rng(seed).normal(size=(count, dimension)).astype(np.float32)
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def copy_cache(cache, *, batch_size: int = 1):
    """Own all mutable native arrays; only known batch axes may be expanded.

    Empty KV state is handled without accessing KVCache.state. Temporal offsets
    remain scalars; per-sequence lengths/padding repeat on their batch axis only.
    Unknown subclasses fail closed rather than losing private cache semantics.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache, KVCache

    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("positive integer batch size required")

    def copied(a):
        if a is None:
            return None
        if not isinstance(a, mx.array) or a.ndim < 1 or a.shape[0] != 1:
            raise ValueError("unsupported cache array/batch state")
        return mx.array(mx.broadcast_to(a, (batch_size, *a.shape[1:])))

    result = []
    for entry in cache:
        if type(entry) is KVCache:
            if set(vars(entry)) != {"keys", "values", "offset"}:
                raise ValueError("unsupported KV cache metadata")
            new = KVCache()
            new.keys, new.values = copied(entry.keys), copied(entry.values)
            new.offset = entry.offset
            mx.eval(new.keys, new.values) if new.keys is not None else None
        elif type(entry) is ArraysCache:
            if set(vars(entry)) != {"cache", "lengths", "left_padding"}:
                raise ValueError("unsupported ArraysCache metadata")
            new = ArraysCache(len(entry.cache))
            new.cache = [copied(a) for a in entry.cache]
            new.lengths, new.left_padding = copied(entry.lengths), copied(entry.left_padding)
            mx.eval([a for a in [*new.cache, new.lengths, new.left_padding] if a is not None])
        else:
            raise ValueError(f"unsupported native cache: {type(entry).__name__}")
        result.append(new)
    return result


def pre_norm_tail(view, layer, h, cache=None):
    """Run blocks[layer:] only; masks see uniformly unadvanced prefix offsets."""
    import mlx.core as mx

    h = h.astype(mx.float32)
    masks = view.masks(h, cache)
    for index in range(layer, view.num_layers):
        h = view.run_block(index, h, masks, None if cache is None else cache[index])
    return h.astype(mx.float32)


@dataclass(frozen=True)
class PositionState:
    view: Any
    layer: int
    position: int
    full_primal: Any
    primal: Any
    prefix_cache: Any
    epsilon: float
    primal_norm: float
    step_coefficient: float = 0.01


def with_position_step(state, coefficient):
    """§15: normalize by only the perturbed float32 position, preserving native state."""
    selected = np.asarray(state.full_primal[0, state.position], dtype=np.float32)
    norm = float(np.linalg.norm(selected))
    if not np.isfinite([norm, coefficient]).all() or min(norm, coefficient) <= 0:
        raise ValueError("finite positive selected-position norm and coefficient required")
    return replace(
        state, primal_norm=norm, epsilon=coefficient * norm, step_coefficient=float(coefficient)
    )


def prepare_position(
    view, ids, layer: int, position: int, *, guard=None, step_coefficient=None
) -> PositionState:
    """Cache the prefix through *all* blocks, derive current residual on a clone.

    Legacy plans use the full-sequence norm. Revised §15 plans use only the
    selected position. The full residual is retained for the uncached reference.
    """
    import mlx.core as mx

    if not 1 <= layer < view.num_layers or not 0 <= position < len(ids):
        raise ValueError("nonfinal residual layer and valid position required")
    full = view.embed(ids)
    masks = view.masks(full, None)
    for index in range(layer):
        full = view.run_block(index, full, masks, None)
    full = full.astype(mx.float32)
    mx.eval(full)
    _check_workload(guard, "prepare_full_primal")
    norm = float(mx.linalg.norm(full).item())
    if not np.isfinite(norm):
        raise ValueError("nonfinite full primal")
    epsilon = 0.01 * norm if norm else 0.01
    cache = view.make_cache()
    if position:
        prefix = view.embed(ids[:position])
        prefix_output = pre_norm_tail(view, 0, prefix, cache)
        mx.eval(prefix_output)
        _check_workload(guard, "prepare_prefix")
        del prefix_output
        # Materialize and own the complete snapshot, including SSM metadata.
        cache = copy_cache(cache)
        _check_workload(guard, "prepare_prefix_copy")
    current_cache = copy_cache(cache) if position else view.make_cache()
    _check_workload(guard, "prepare_current_cache")
    h = view.embed(ids[position : position + 1])
    masks = view.masks(h, current_cache)
    for index in range(layer):
        h = view.run_block(index, h, masks, current_cache[index])
    mx.eval(h)
    _check_workload(guard, "prepare_current_primal")
    state = PositionState(view, layer, position, full, h.astype(mx.float32), cache, epsilon, norm)
    return with_position_step(state, step_coefficient) if step_coefficient is not None else state


def _directions(state, directions):
    d = np.asarray(directions, dtype=np.float32)
    if d.ndim != 2 or d.shape[1] != state.view.hidden_size or not len(d):
        raise ValueError("nonempty direction rows matching hidden size required")
    norms = np.linalg.norm(d, axis=1)
    if not np.isfinite(d).all() or np.any(norms == 0):
        raise ValueError("finite nonzero directions required")
    return d, norms


def reference_responses(state, directions, *, epsilon_scale=1.0, guard=None):
    """Full-sequence uncached central FD; perturb only the selected source row."""
    import mlx.core as mx

    directions, norms = _directions(state, directions)
    if not np.isfinite(epsilon_scale) or epsilon_scale <= 0:
        raise ValueError("positive epsilon scale required")
    responses = []
    for number, (direction, norm) in enumerate(zip(directions, norms, strict=True)):
        eps = float(
            finite_difference_steps(
                state.primal_norm,
                np.array([norm]),
                epsilon_scale,
                coefficient=getattr(state, "step_coefficient", 0.01),
            )[0]
        )
        tangent = mx.zeros_like(state.full_primal)
        tangent[0, state.position] = mx.array(direction)
        plus = pre_norm_tail(state.view, state.layer, state.full_primal + eps * tangent)
        mx.eval(plus)
        _check_workload(guard, "reference_plus", direction=number)
        minus = pre_norm_tail(state.view, state.layer, state.full_primal - eps * tangent)
        mx.eval(minus)
        _check_workload(guard, "reference_minus", direction=number)
        response = (plus[0, state.position] - minus[0, state.position]) / (2 * eps)
        responses.append(np.array(response, dtype=np.float32))
        _check_workload(guard, "reference_response", direction=number)
    return np.stack(responses)


def cached_responses(state, directions, *, mode, batch_size=8, epsilon_scale=1.0, guard=None):
    """Each direction uses the reference epsilon, independent of row/batch norm.

    Restore is serial per direction; broadcast batches direction rows. Each plus
    and minus call owns a fresh cache. No candidate ever advances the snapshot.
    """
    import mlx.core as mx

    directions, norms = _directions(state, directions)
    if mode not in {"restore", "broadcast"} or batch_size not in (8, 16, 32, 64):
        raise ValueError("restore/broadcast path and batch8/16/32/64 required")
    if not np.isfinite(epsilon_scale) or epsilon_scale <= 0:
        raise ValueError("positive epsilon scale required")
    width = 1 if mode == "restore" else batch_size
    results = []
    for start in range(0, len(directions), width):
        d = mx.array(directions[start : start + width])[:, None, :]
        eps = mx.array(
            finite_difference_steps(
                state.primal_norm,
                norms[start : start + width],
                epsilon_scale,
                coefficient=getattr(state, "step_coefficient", 0.01),
            )
        )[:, None, None]
        primal = mx.broadcast_to(state.primal, d.shape)
        outputs = []
        for sign in (1, -1):
            cache = copy_cache(state.prefix_cache, batch_size=d.shape[0])
            _check_workload(
                guard,
                "candidate_cache_copy",
                direction_start=start,
                sign=sign,
                actual_batch_size=d.shape[0],
            )
            value = pre_norm_tail(state.view, state.layer, primal + sign * eps * d, cache)
            mx.eval(value)
            _check_workload(
                guard,
                "candidate_tail",
                direction_start=start,
                sign=sign,
                actual_batch_size=d.shape[0],
            )
            outputs.append(value)
        response = (outputs[0] - outputs[1]) / (2 * eps)
        results.append(np.array(response[:, 0], dtype=np.float32))
        _check_workload(
            guard, "candidate_response", direction_start=start, actual_batch_size=d.shape[0]
        )
    return np.concatenate(results)


def full_jacobian(state, *, mode, batch_size=8, guard=None):
    """Measure every basis column; avoid a hidden-size squared device basis."""
    dimension = state.view.hidden_size
    matrix = np.empty((dimension, dimension), dtype=np.float32)
    for start in range(0, dimension, batch_size):
        end = min(start + batch_size, dimension)
        basis = np.zeros((end - start, dimension), dtype=np.float32)
        basis[np.arange(end - start), np.arange(start, end)] = 1
        matrix[:, start:end] = cached_responses(
            state, basis, mode=mode, batch_size=batch_size, guard=guard
        ).T
    return matrix


def self_check(view, rows, plan, *, batch_size=8):
    """Six independent comparisons; bounds come only from the frozen plan."""
    import mlx.core as mx

    from local_llm_lab.pipeline.live_lens.validation import response_agreement

    device_working = mx.device_info()["max_recommended_working_set_size"]
    if plan["working_set_bytes"] > device_working:
        raise ValueError("declared working set exceeds actual device working set")
    guard = WorkloadMemoryGuard(
        plan["initial_peak_bytes"] * batch_size / 8,
        plan["working_set_bytes"],
        mx.get_peak_memory,
        context=dict(stage="self_check", batch_size=batch_size),
    )
    guard("before_self_check")
    directions = unit_directions(16, view.hidden_size, plan["seeds"]["self_directions"])
    # Preserve the 16 scientific directions, repeat their rows to exercise the
    # configured candidate width rather than silently testing only batch16.
    row_indices = np.arange(max(16, batch_size)) % 16
    candidate_directions = directions[row_indices]
    checks = []
    for layer in plan["self_layers"]:
        guard.context.update(layer=layer, mode="reference", epsilon_scale=1.0)
        state = prepare_position(
            view,
            rows[plan["self_row"]]["ids"],
            layer,
            plan["self_position"],
            guard=guard,
            **position_step_kwargs(plan),
        )
        guard("self_check_prepared")
        reference = reference_responses(state, directions, guard=guard)
        guard("self_check_reference")
        guard.context["epsilon_scale"] = 0.5
        half = reference_responses(state, directions, epsilon_scale=0.5, guard=guard)
        guard("self_check_half_reference")
        stable = response_agreement(reference, half, **plan["stability_bounds"], stable=True)
        for mode in ("restore", "broadcast"):
            guard.context.update(mode=mode, epsilon_scale=1.0)
            guard.batch_widths.clear()
            candidate = cached_responses(
                state, candidate_directions, mode=mode, batch_size=batch_size, guard=guard
            )
            guard("self_check_candidate")
            check = response_agreement(
                reference[row_indices],
                candidate,
                **plan["self_bounds"],
                stable=stable["outcome"] == "pass",
            )
            checks.append(
                dict(
                    layer=layer,
                    mode=mode,
                    **check,
                    epsilon=state.epsilon,
                    reference_stability=stable,
                    candidate_rows=len(candidate_directions),
                    actual_batch_sizes=sorted(guard.batch_widths),
                )
            )
    return dict(
        plan_sha256=plan["plan_sha256"],
        directions=16,
        batch_size=batch_size,
        checks=checks,
        direction_seed=plan["seeds"]["self_directions"],
    )


def require_self_check(plan, proof, *, batch_size=8):
    expected = {(layer, m) for layer in plan["self_layers"] for m in ("restore", "broadcast")}
    checks = proof.get("checks", [])
    if (
        proof.get("plan_sha256") != plan["plan_sha256"]
        or proof.get("directions") != 16
        or proof.get("batch_size") != batch_size
        or len(checks) != len(expected)
        or {(c.get("layer"), c.get("mode")) for c in checks} != expected
        or any(c.get("outcome") != "pass" for c in checks)
        or any(
            c.get("actual_batch_sizes") != ([1] if c.get("mode") == "restore" else [batch_size])
            for c in checks
        )
    ):
        raise ValueError("missing or failed cached/reference self-check proof")


def benchmark_plan(plan, proof, measure):
    """Pure scheduling seam. measure(layer, mode, batch) times fresh native work.

    Each observation must include an independently passed reference check. Larger
    batches are conservatively projected from the entire preceding process peak;
    this deliberately does not infer a Director window from spare byte headroom.
    """
    require_self_check(plan, proof)
    observations, skipped, selected = [], [], {}
    previous_peak = plan["initial_peak_bytes"]
    previous_batch = 8
    for batch in plan["batches"]:
        projection = max(
            plan["initial_peak_bytes"] * batch / 8, previous_peak * batch / previous_batch
        )
        try:
            memory_gate(projection, plan["working_set_bytes"])
        except ValueError as error:
            skipped.append(dict(batch_size=batch, projected_bytes=projection, reason=str(error)))
            continue
        batch_peak = 0
        for layer in plan["layers"]:
            for mode in ("restore", "broadcast"):
                observation = measure(layer, mode, batch)
                if observation.get("reference_passed") is not True:
                    raise ValueError("benchmark requires independent candidate reference proof")
                values = [
                    observation[k]
                    for k in ("elapsed_s", "preparation_s", "peak_bytes", "directions")
                ]
                if not np.isfinite(values).all() or min(values) <= 0:
                    raise ValueError("invalid benchmark measurement")
                memory_gate(observation["peak_bytes"], plan["working_set_bytes"])
                if observation["peak_bytes"] > projection:
                    raise ValueError("memory projection falsified by measured candidate peak")
                projected_s = len(plan["fit_positions"]) * (
                    observation["preparation_s"]
                    + observation["elapsed_s"]
                    * np.ceil(plan["hidden_size"] / observation["directions"])
                )
                row = dict(
                    layer=layer,
                    mode=mode,
                    batch_size=batch,
                    projected_peak_bytes=projection,
                    projected_seconds=float(projected_s),
                    **observation,
                )
                observations.append(row)
                key = str(layer)
                if key not in selected or projected_s < selected[key]["projected_seconds"]:
                    selected[key] = row
                batch_peak = max(batch_peak, observation["peak_bytes"])
        previous_peak, previous_batch = batch_peak, batch
    status, reason = "ready", "all layers projected within limits"
    try:
        time_gate({k: v["projected_seconds"] for k, v in selected.items()})
    except ValueError as error:
        status, reason = "stop_required", str(error)
    return dict(
        plan_sha256=plan["plan_sha256"],
        status=status,
        reason=reason,
        observations=observations,
        skipped=skipped,
        selected=selected,
        projected_positions=len(plan["fit_positions"]),
        buffer_count_safety="not established by byte measurements",
    )


def require_benchmark(plan, report):
    selected = report.get("selected", {})
    if (
        report.get("plan_sha256") != plan["plan_sha256"]
        or report.get("status") != "ready"
        or set(selected) != {str(layer) for layer in plan["layers"]}
        or report.get("projected_positions") != 400
    ):
        raise ValueError("missing or stopped benchmark proof")
    for layer, row in selected.items():
        if (
            row not in report.get("observations", [])
            or row.get("reference_passed") is not True
            or str(row.get("layer")) != layer
            or row.get("mode") not in plan["paths"]
            or row.get("batch_size") not in plan["batches"]
        ):
            raise ValueError("invalid selected benchmark proof")
        memory_gate(row["peak_bytes"], plan["working_set_bytes"])
        memory_gate(row["projected_peak_bytes"], plan["working_set_bytes"])
    time_gate({k: v["projected_seconds"] for k, v in selected.items()})


def benchmark(view, rows, plan, proof, *, progress=None):
    """Time batch8 first, then larger eligible candidates on worst planned contexts.

    Prefix and full-sequence extremes may be different prompts: time both, retain
    the worse reading. Re-run R52 for every new batch before it may be timed.
    """
    import time

    import mlx.core as mx

    proven = {8: proof}

    def measure(layer, mode, batch):
        if batch not in proven:
            proven[batch] = self_check(view, rows, plan, batch_size=batch)
        require_self_check(plan, proven[batch], batch_size=batch)
        by_length = max(
            plan["fit_positions"] + plan["held_positions"], key=lambda s: len(rows[s["row"]]["ids"])
        )
        by_prefix = max(plan["fit_positions"] + plan["held_positions"], key=lambda s: s["position"])
        guard = WorkloadMemoryGuard(
            plan["initial_peak_bytes"] * batch / 8,
            plan["working_set_bytes"],
            mx.get_peak_memory,
            context=dict(stage="benchmark", layer=layer, mode=mode, batch_size=batch),
        )
        guard("before_benchmark_candidate")
        observations = []
        for extreme, sample in enumerate((by_length, by_prefix)):
            guard.context.update(extreme=extreme, sample=sample)
            started = time.monotonic()
            state = prepare_position(
                view,
                rows[sample["row"]]["ids"],
                layer,
                sample["position"],
                guard=guard,
                **position_step_kwargs(plan),
            )
            guard("benchmark_prepared")
            preparation_s = time.monotonic() - started
            width = min(batch, view.hidden_size)
            directions = np.zeros((width, view.hidden_size), dtype=np.float32)
            directions[np.arange(width), np.arange(width)] = 1
            started = time.monotonic()
            cached_responses(state, directions, mode=mode, batch_size=batch, guard=guard)
            guard("benchmark_derivative")
            observations.append(
                dict(
                    elapsed_s=time.monotonic() - started, preparation_s=preparation_s, sample=sample
                )
            )
            del state
        result = dict(
            elapsed_s=max(o["elapsed_s"] for o in observations),
            preparation_s=max(o["preparation_s"] for o in observations),
            peak_bytes=mx.get_peak_memory(),
            directions=width,
            reference_passed=True,
            extremes=observations,
        )
        if progress:
            progress(
                dict(
                    event="benchmark_candidate", layer=layer, mode=mode, batch_size=batch, **result
                )
            )
        return result

    report = benchmark_plan(plan, proof, measure)
    report["batch_self_checks"] = {str(k): v for k, v in proven.items()}
    return report


@dataclass(frozen=True)
class JacobianResult:
    maps: dict
    per_layer: dict
    counts: dict
    elapsed_s: float
    peak_memory_gib: float


class FitStopped(ValueError):
    def __init__(self, report):
        super().__init__(report["reason"])
        self.report = report


def fit_jacobian(view, rows, plan, proof, benchmark_report, *, progress=None):
    """Measure all basis directions for each planned draw; never accept the ceiling.

    Callers persist the frozen plan, check and benchmark records before entering.
    A fresh process must remeasure benchmark immediately before this long fit.
    """
    import time

    import mlx.core as mx

    require_self_check(plan, proof)
    require_benchmark(plan, benchmark_report)
    if view.hidden_size != plan["hidden_size"] or list(range(1, view.num_layers)) != plan["layers"]:
        raise ValueError("view does not match frozen plan")
    if digest(rows) != plan["rows_sha256"]:
        raise ValueError("corpus changed after frozen plan")
    started = time.monotonic()
    maps, per_layer = {}, {}
    tokens_done = 0
    for layer in plan["layers"]:
        layer_start = time.monotonic()
        candidate = benchmark_report["selected"][str(layer)]
        guard = WorkloadMemoryGuard(
            max(
                candidate["projected_peak_bytes"],
                max(o["peak_bytes"] for o in benchmark_report["observations"]),
            ),
            plan["working_set_bytes"],
            mx.get_peak_memory,
            context=dict(
                stage="fit", layer=layer, mode=candidate["mode"], batch_size=candidate["batch_size"]
            ),
        )
        guard("before_fit_layer")
        c = Convergence()
        for sample in plan["fit_positions"]:
            guard.context["sample"] = sample
            state = prepare_position(
                view,
                rows[sample["row"]]["ids"],
                layer,
                sample["position"],
                guard=guard,
                **position_step_kwargs(plan),
            )
            guard("fit_prepared")
            matrix = full_jacobian(
                state, mode=candidate["mode"], batch_size=candidate["batch_size"], guard=guard
            )
            guard("fit_basis")
            c.add(matrix, span=sample["span"])
            tokens_done += len(rows[sample["row"]]["ids"])
            del state, matrix
            memory_gate(mx.get_peak_memory(), plan["working_set_bytes"])
            if c.n % 10 == 0 and progress:
                progress(
                    dict(
                        event="convergence",
                        layer=layer,
                        tokens_done=tokens_done,
                        **c.curve[-1],
                        reason=c.reason,
                    )
                )
            if c.accepted:
                break
        info = dict(
            positions=c.n,
            selected_positions=plan["fit_positions"][: c.n],
            convergence_curve=c.curve,
            reason=c.reason,
            accepted=c.accepted,
            span_counts={s: n for s, (_, n) in c.span_means.items()},
            elapsed_s=time.monotonic() - layer_start,
            peak_memory_gib=mx.get_peak_memory() / 2**30,
            mode=candidate["mode"],
            batch_size=candidate["batch_size"],
        )
        per_layer[str(layer)] = info
        if not c.accepted:
            raise FitStopped(dict(reason=c.reason, failed_layer=layer, per_layer=per_layer))
        maps[layer] = c.mean.copy()
    return JacobianResult(
        maps,
        per_layer,
        {"fit": {"positions_per_layer": {k: v["positions"] for k, v in per_layer.items()}}},
        time.monotonic() - started,
        mx.get_peak_memory() / 2**30,
    )


def prepare_plan(prepared, path, *, config_path=None):
    """Freeze or verify a plan without importing MLX or invoking a loader.

    config_path supplies exactly seed, held_count, working_set_bytes,
    initial_peak_bytes and self/response/stability_bounds. The initial peak is a
    declared conservative envelope including model, full reference and caches.
    An optional step_coefficient freezes the measured §15 position-local rule.
    Values exceeding the normal memory window still stop here.
    """
    config = json.loads((Path(prepared.snapshot["snapshot_path"]) / "config.json").read_text())
    config = config.get("text_config", config)
    dimension, depth = config["hidden_size"], config["num_hidden_layers"]
    if config_path is not None:
        parameters = json.loads(Path(config_path).read_text())
        expected = {
            "seed",
            "held_count",
            "working_set_bytes",
            "initial_peak_bytes",
            "self_bounds",
            "response_bounds",
            "stability_bounds",
        }
        if set(parameters) - {"step_coefficient"} != expected:
            raise ValueError(f"plan config requires {sorted(expected)}; optional step_coefficient")
        plan = make_plan(
            prepared.rows,
            layers=range(1, depth),
            hidden_size=dimension,
            corpus_sha256=prepared.corpus_manifest_sha256,
            snapshot_sha256=prepared.snapshot["snapshot_sha256"],
            **parameters,
        )
        return freeze_plan(path, plan)
    plan = read_plan(path, prepared.rows)
    if (
        plan["corpus_sha256"] != prepared.corpus_manifest_sha256
        or plan["snapshot_sha256"] != prepared.snapshot["snapshot_sha256"]
        or plan["hidden_size"] != dimension
        or plan["layers"] != list(range(1, depth))
    ):
        raise ValueError("plan corpus/snapshot/architecture mismatch")
    return plan


def run_jacobian_stage(view, rows, plan, *, stage, record_dir, progress=None):
    """Persist R52 proof before timing, timing before fit, and all gate failures.

    Every invocation performs fresh checks; a benchmark file from another process
    never substitutes for current measurements before a long fit.
    """
    from local_llm_lab.pipeline.lens_fitting.validation import validate_maps

    if stage not in {"check", "benchmark", "fit"}:
        raise ValueError("unknown Jacobian stage")
    record_dir = Path(record_dir)
    try:
        proof = self_check(view, rows, plan)
        write_record(record_dir / "self-check.json", proof)
        require_self_check(plan, proof)
        if stage == "check":
            return None
        report = benchmark(view, rows, plan, proof, progress=progress)
        write_record(record_dir / "benchmark.json", report)
        require_benchmark(plan, report)
        if stage == "benchmark":
            return None
        result = fit_jacobian(view, rows, plan, proof, report, progress=progress)
        write_record(
            record_dir / "fit.json",
            dict(
                per_layer=result.per_layer,
                counts=result.counts,
                elapsed_s=result.elapsed_s,
                peak_memory_gib=result.peak_memory_gib,
            ),
        )
        validation = validate_maps(view, rows, result.maps, plan, report, progress=progress)
        write_record(record_dir / "validation.json", validation)
        return result, validation
    except (ValueError, RuntimeError) as error:
        failure = dict(plan_sha256=plan["plan_sha256"], status="stop_required", reason=str(error))
        if isinstance(error, (FitStopped, MemoryStopped)):
            failure.update(error.report)
        write_record(record_dir / "stop.json", failure)
        raise


def finite_difference_steps(primal_norm, tangent_norms, epsilon_scale=1.0, *, coefficient=0.01):
    """Direction-normalized step; caller supplies the frozen norm and coefficient."""
    norms = np.asarray(tangent_norms, dtype=np.float32)
    if (
        not np.isfinite(primal_norm)
        or primal_norm < 0
        or not np.isfinite(norms).all()
        or np.any(norms <= 0)
        or not np.isfinite(epsilon_scale)
        or epsilon_scale <= 0
        or not np.isfinite(coefficient)
        or coefficient <= 0
    ):
        raise ValueError("finite primal and positive tangent norms/epsilon scale required")
    steps = 0.01 * primal_norm / norms if primal_norm else np.full_like(norms, 0.01)
    factor = epsilon_scale if coefficient == 0.01 else epsilon_scale * coefficient / 0.01
    return steps * factor
