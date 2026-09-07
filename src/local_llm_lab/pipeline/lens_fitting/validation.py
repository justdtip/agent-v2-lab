"""Held-position mean-to-mean validation; existing verdicts own refit authority."""

from __future__ import annotations

import numpy as np

from local_llm_lab.pipeline.lens_fitting.jacobian import check_bounds
from local_llm_lab.pipeline.live_lens.validation import layer_verdict, response_agreement


def validate_layer(matrix, samples, directions, measure, *, response_bounds, stability_bounds):
    """measure(sample, directions, epsilon_scale) returns input-direction rows.

    Equal span means use the same source/target position and conditional prompt
    sampling as fitting. Epsilon and epsilon/2 must agree before judging the map.
    Concept and output hooks remain inconclusive/descriptive until pre-registration.
    """
    response_bounds, stability_bounds = map(check_bounds, (response_bounds, stability_bounds))
    matrix, directions = np.asarray(matrix), np.asarray(directions)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or directions.shape != (32, matrix.shape[0])
    ):
        raise ValueError("32 unit directions and a square map required")
    if not np.allclose(np.linalg.norm(directions, axis=1), 1, atol=1e-6, rtol=1e-6):
        raise ValueError("directions must be unit vectors")
    if not samples:
        raise ValueError("held positions required")
    grouped, checks = {}, []
    for sample in samples:
        a = np.asarray(measure(sample, directions, 1.0), dtype=np.float32)
        b = np.asarray(measure(sample, directions, 0.5), dtype=np.float32)
        if a.shape != directions.shape or b.shape != directions.shape:
            raise ValueError("measured direction response shape mismatch")
        stability = response_agreement(a, b, **stability_bounds, stable=True)
        checks.append(dict(sample=sample, **stability))
        total, count = grouped.get(sample["span"], (np.zeros_like(a), 0))
        grouped[sample["span"]] = total + a, count + 1
    measured = np.mean([total / n for total, n in grouped.values()], axis=0, dtype=np.float32)
    predicted = directions @ matrix.T
    agreement = response_agreement(
        measured, predicted, **response_bounds, stable=all(c["outcome"] == "pass" for c in checks)
    )
    return dict(
        map_check=agreement,
        stability_checks=checks,
        span_counts={s: n for s, (_, n) in grouped.items()},
        averaging="equal span means; uniform prompt then token; same source/target position",
        verdict=layer_verdict(agreement["outcome"], "inconclusive", None),
    )


def validate_maps(view, rows, maps, plan, benchmark_report, *, progress=None):
    """32 common random unit directions; held means share fitting's span weights."""
    import mlx.core as mx

    from local_llm_lab.pipeline.lens_fitting.jacobian import (
        WorkloadMemoryGuard,
        cached_responses,
        digest,
        prepare_position,
        require_benchmark,
        unit_directions,
    )

    require_benchmark(plan, benchmark_report)
    if digest(rows) != plan["rows_sha256"] or set(maps) != set(plan["layers"]):
        raise ValueError("validation corpus or layers mismatch")
    samples = plan["held_positions"]
    if any(rows[s["row"]]["split"] != "held" or s["split"] != "held" for s in samples):
        raise ValueError("validation requires held positions exclusively")
    if {s["span"] for s in samples} != {s["span"] for s in plan["fit_positions"]}:
        raise ValueError("validation requires the fit span support")
    directions = unit_directions(32, view.hidden_size, plan["seeds"]["validation_directions"])
    results = {}
    for layer in plan["layers"]:
        selected = benchmark_report["selected"][str(layer)]
        guard = WorkloadMemoryGuard(
            max(
                selected["projected_peak_bytes"],
                max(o["peak_bytes"] for o in benchmark_report["observations"]),
            ),
            plan["working_set_bytes"],
            mx.get_peak_memory,
            context=dict(
                stage="validation",
                layer=layer,
                mode=selected["mode"],
                batch_size=selected["batch_size"],
            ),
        )
        guard("before_validation_layer")
        # Preserve only one prepared prompt between epsilon and half-epsilon calls.
        current, state = None, None

        def measure(sample, directions, scale, *, layer=layer, selected=selected, guard=guard):
            nonlocal current, state
            guard.context.update(sample=sample, epsilon_scale=scale)
            if current != sample:
                state = prepare_position(
                    view, rows[sample["row"]]["ids"], layer, sample["position"], guard=guard
                )
                guard("validation_prepared")
                current = sample
            response = cached_responses(
                state,
                directions,
                mode=selected["mode"],
                batch_size=selected["batch_size"],
                epsilon_scale=scale,
                guard=guard,
            )
            guard("validation_response")
            return response

        results[str(layer)] = validate_layer(
            maps[layer],
            samples,
            directions,
            measure,
            response_bounds=plan["response_bounds"],
            stability_bounds=plan["stability_bounds"],
        )
        if progress:
            progress(dict(event="validation", layer=layer, **results[str(layer)]["verdict"]))
    return dict(
        plan_sha256=plan["plan_sha256"],
        directions=32,
        direction_seed=plan["seeds"]["validation_directions"],
        per_layer=results,
    )
