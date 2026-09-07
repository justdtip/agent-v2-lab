"""Position-local finite-difference diagnostic specified by requirements §15.

The production fitter remains unchanged until the sweep and original self-check pass.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .epsilon_diagnostic import BlockLedger, atomic_record, compare, validate_scope
from .jacobian import (
    cached_responses,
    finite_difference_steps,
    prepare_position,
    reference_responses,
    unit_directions,
)

COEFFICIENTS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
BUDGET_SECONDS = 1200


def position_local_state(state):
    """Change only the step normalization; retain both native primals and caches."""
    selected = np.asarray(state.full_primal[0, state.position], dtype=np.float32)
    norm = float(np.linalg.norm(selected))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("finite positive selected-position norm required")
    return replace(state, primal_norm=norm, epsilon=0.01 * norm)


def plateaus(points):
    """Return every eligible upper point; an isolated passing point is insufficient."""
    if [p["coefficient"] for p in points] != list(COEFFICIENTS):
        raise ValueError("complete ascending preregistered coefficient grid required")
    return [
        p["coefficient"]
        for previous, p in zip(points, points[1:], strict=False)
        if previous["failed_coordinates"] == p["failed_coordinates"] == 0
    ]


def run_sweep(
    view,
    rows,
    plan,
    directory,
    *,
    started,
    guard,
    provenance,
    prepare=prepare_position,
    reference=reference_responses,
    cached=cached_responses,
    progress=None,
):
    directory = Path(directory)
    validate_scope(rows, plan)
    directions = unit_directions(16, plan["hidden_size"], plan["seeds"]["self_directions"])
    np.save(directory / "directions.npy", directions, allow_pickle=False)
    report = dict(
        status="incomplete",
        diagnostic_only=True,
        production_acceptance=False,
        coefficient_grid=list(COEFFICIENTS),
        budget_seconds=BUDGET_SECONDS,
        normalization="float32 norm of the selected full-sequence residual position",
        step="c * norm(h_t) / norm(direction)",
        zero_norm="stop, no absolute fallback",
        selection="largest common upper point with passing stability at it and next smaller grid point",
        plan_sha256=plan["plan_sha256"],
        provenance=provenance,
        row=0,
        position=438,
        input_ids=rows[0]["ids"],
        direction_sha256=hashlib.sha256(directions.tobytes()).hexdigest(),
        measurements=[],
        stability=[],
        primals=[],
        eligible={},
        self_check=[],
    )
    with (directory / "block-ledger.jsonl").open("x") as ledger_stream:

        def emit(event):
            ledger_stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            ledger_stream.flush()
            if event["event"] == "materialized_workload":
                os.fsync(ledger_stream.fileno())

        wrapped = BlockLedger(view, emit, started=started, budget=BUDGET_SECONDS)

        def measured_guard(workload, **details):
            wrapped.confirm(workload)
            guard(workload, **details)

        def checkpoint(event):
            report.update(
                elapsed_s=time.monotonic() - started,
                block_accounting=wrapped.summary(),
                progress=event,
            )
            atomic_record(directory / "diagnostic.json", report)
            if progress:
                progress(event)

        def prepare_layer(layer):
            wrapped.phase = dict(layer=layer, operation="prepare")
            original = prepare(wrapped, rows[0]["ids"], layer, 438, guard=measured_guard)
            corrected = position_local_state(original)
            report["primals"].append(
                dict(
                    layer=layer, full_norm=original.primal_norm, position_norm=corrected.primal_norm
                )
            )
            return corrected

        def measure(state, coefficient, mode):
            wrapped.phase = dict(layer=state.layer, coefficient=coefficient, operation=mode)
            checkpoint(dict(event="measurement_started", **wrapped.phase))
            kwargs = dict(epsilon_scale=coefficient / 0.01, guard=measured_guard)
            response = (
                reference(state, directions, **kwargs)
                if mode == "reference"
                else cached(state, directions, mode=mode, batch_size=8, **kwargs)
            )
            response = np.asarray(response, dtype=np.float32)
            measured_guard("sweep_response", **wrapped.phase)
            if response.shape != directions.shape:
                raise ValueError("response shape differs from fixed directions")
            filename = f"measurement-{len(report['measurements']) + 1:03}-layer-{state.layer}-c-{coefficient:g}-{mode}.npy"
            np.save(directory / filename, response, allow_pickle=False)
            report["measurements"].append(
                dict(
                    layer=state.layer,
                    coefficient=coefficient,
                    mode=mode,
                    file=filename,
                    file_sha256=hashlib.sha256((directory / filename).read_bytes()).hexdigest(),
                    epsilon=finite_difference_steps(
                        state.primal_norm, np.linalg.norm(directions, axis=1), coefficient / 0.01
                    ).tolist(),
                    nonfinite_coordinates=int((~np.isfinite(response)).sum()),
                )
            )
            checkpoint(dict(event="measurement_completed", **wrapped.phase))
            return response

        try:
            checkpoint(dict(event="sweep_started"))
            for layer in plan["self_layers"]:
                state = prepare_layer(layer)
                points = []
                for c in COEFFICIENTS:
                    full = measure(state, c, "reference")
                    half = measure(state, c / 2, "reference")
                    point = dict(
                        layer=layer,
                        coefficient=c,
                        **compare(full, half, **plan["stability_bounds"]),
                    )
                    points.append(point)
                    report["stability"].append(point)
                    checkpoint(
                        dict(
                            event="stability_completed",
                            layer=layer,
                            coefficient=c,
                            failed_coordinates=point["failed_coordinates"],
                            max_error=point["max_error"],
                        )
                    )
                report["eligible"][str(layer)] = plateaus(points)
                checkpoint(dict(event="layer_completed", layer=layer, eligible=plateaus(points)))
                del state, full, half
            common = set(COEFFICIENTS)
            for values in report["eligible"].values():
                common.intersection_update(values)
            report["selected_per_layer"] = {
                k: max(v) if v else None for k, v in report["eligible"].items()
            }
            report["selected_common_coefficient"] = max(common) if common else None
            if not common:
                report["status"] = (
                    "instrument_limited"
                    if any(not v for v in report["eligible"].values())
                    else "no_common_coefficient"
                )
            else:
                c = max(common)
                # Re-run the original stability and both cache paths at the selected rule.
                for layer in plan["self_layers"]:
                    state = prepare_layer(layer)
                    ref = measure(state, c, "reference")
                    half = measure(state, c / 2, "reference")
                    stable = compare(ref, half, **plan["stability_bounds"])
                    for mode in ("restore", "broadcast"):
                        candidate = measure(state, c, mode)
                        agreement = compare(ref, candidate, **plan["self_bounds"])
                        report["self_check"].append(
                            dict(
                                layer=layer,
                                mode=mode,
                                coefficient=c,
                                stability=stable,
                                agreement=agreement,
                                passed=stable["failed_coordinates"]
                                == agreement["failed_coordinates"]
                                == 0,
                            )
                        )
                    del state, ref, half, candidate
                report["status"] = (
                    "self_check_passed"
                    if all(r["passed"] for r in report["self_check"])
                    else "self_check_failed"
                )
            if wrapped.pending:
                raise RuntimeError("unconfirmed native work at end of sweep")
        except BaseException as error:
            report.update(status="incomplete", error_type=type(error).__name__, error=str(error))
        finally:
            checkpoint(dict(event="sweep_finished", status=report["status"]))
    return report
