"""Approved four-scale descriptive diagnostic; never a production self-check proof."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from .jacobian import (
    cached_responses,
    digest,
    finite_difference_steps,
    prepare_position,
    reference_responses,
    unit_directions,
)

SCALES = (0.125, 0.25, 0.5, 1.0)
ENDPOINTS = (0.125, 1.0)


def atomic_record(path, record):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def compare(reference, candidate, *, atol, rtol):
    """The first argument supplies the tolerance denominator, including stability."""
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("matching direction-by-coordinate responses required")
    finite = np.isfinite(a) & np.isfinite(b)
    with np.errstate(invalid="ignore", over="ignore"):
        error = np.abs(a - b)
        failed = ~finite | (error > atol + rtol * np.abs(a))

    def metrics(e, f, valid):
        return dict(
            max_error=float(e.max()) if valid.all() else None,
            rms_error=float(np.sqrt(np.mean(e**2))) if valid.all() else None,
            failed_coordinates=int(f.sum()),
            total_coordinates=int(f.size),
            nonfinite_coordinates=int((~valid).sum()),
        )

    return dict(
        atol=atol,
        rtol=rtol,
        denominator="first/reference response",
        **metrics(error, failed, finite),
        per_direction=[metrics(e, f, v) for e, f, v in zip(error, failed, finite, strict=True)],
    )


class DeadlineExceeded(RuntimeError):
    pass


class BlockLedger:
    """Single wrapper around actual run_block calls; records partial-block work."""

    def __init__(self, view, emit, *, started, clock=time.monotonic, budget=480):
        self.view, self.emit, self.started, self.clock, self.budget = (
            view,
            emit,
            started,
            clock,
            budget,
        )
        self.phase = {}
        self.head = "0" * 64
        self.calls = self.block_tokens = 0
        self.attempts = 0
        self.pending = []
        self.confirmed_tokens = 0

    def __getattr__(self, name):
        return getattr(self.view, name)

    def check(self):
        if self.clock() - self.started >= self.budget:
            raise DeadlineExceeded("480-second CLI budget exhausted before next native block call")

    def run_block(self, index, h, *args, **kwargs):
        self.check()
        self.attempts += 1
        fields = dict(
            call=self.attempts,
            block_index=int(index),
            batch=int(h.shape[0]),
            sequence=int(h.shape[1]),
            block_tokens=int(h.shape[0] * h.shape[1]),
            phase=dict(self.phase),
        )
        self._record(dict(event="block_started", **fields))
        try:
            result = self.view.run_block(index, h, *args, **kwargs)
        except BaseException as error:
            self._record(dict(event="block_failed", error=type(error).__name__, **fields))
            raise
        self.calls += 1
        self.block_tokens += fields["block_tokens"]
        self.pending.append(fields)
        self._record(dict(event="block_returned", **fields))
        return result

    def _record(self, event):
        event = dict(event, previous_sha256=self.head)
        self.head = digest(event)
        self.emit(dict(event, sha256=self.head))

    def confirm(self, workload):
        if self.pending:
            self.confirmed_tokens += sum(e["block_tokens"] for e in self.pending)
            self._record(
                dict(
                    event="materialized_workload",
                    workload=workload,
                    calls=[e["call"] for e in self.pending],
                )
            )
            self.pending.clear()

    def summary(self):
        return dict(
            attempted_block_calls=self.attempts,
            returned_block_calls=self.calls,
            scheduled_block_tokens=self.block_tokens,
            materialized_block_tokens=self.confirmed_tokens,
            pending_block_calls=len(self.pending),
            ledger_sha256=self.head,
            accounting="batch * sequence per actual run_block; partial-block computations; "
            "returned calls may be lazily evaluated; workload guards follow materialization",
        )


def validate_scope(rows, plan):
    if (
        plan["self_row"],
        plan["self_position"],
        plan["self_layers"],
        plan["seeds"]["self_directions"],
    ) != (0, 438, [1, 16, 31], 20260904):
        raise ValueError("diagnostic requires the original frozen row/position/layers/seed")
    if len(rows[0]["ids"]) != 439 or digest(rows) != plan["rows_sha256"]:
        raise ValueError("diagnostic requires the original frozen 439-token input")
    if plan["self_bounds"] != dict(atol=0.0003, rtol=0.003) or plan["stability_bounds"] != dict(
        atol=0.003, rtol=0.03
    ):
        raise ValueError("original diagnostic tolerances required")


def run_diagnostic(
    view,
    rows,
    plan,
    directory,
    *,
    started,
    guard,
    provenance,
    clock=time.monotonic,
    prepare=prepare_position,
    reference=reference_responses,
    cached=cached_responses,
    progress=None,
):
    """Injected numerical callbacks keep schedule/accounting tests entirely CPU-only."""
    directory = Path(directory)
    validate_scope(rows, plan)
    directions = unit_directions(16, plan["hidden_size"], 20260904)
    np.save(directory / "directions.npy", directions, allow_pickle=False)
    report = dict(
        status="incomplete",
        diagnostic_only=True,
        production_acceptance=False,
        provenance=provenance,
        plan_sha256=plan["plan_sha256"],
        snapshot_sha256=plan["snapshot_sha256"],
        input_ids=rows[0]["ids"],
        input_ids_sha256=digest(rows[0]["ids"]),
        direction_sha256=hashlib.sha256(directions.tobytes()).hexdigest(),
        direction_dtype=str(directions.dtype),
        direction_shape=list(directions.shape),
        direction_seed=20260904,
        scales=list(SCALES),
        endpoints=list(ENDPOINTS),
        row=0,
        position=438,
        layers=[1, 16, 31],
        batch_size=8,
        budget_seconds=480,
        primals=[],
        measurements=[],
        comparisons=[],
    )
    with (directory / "block-ledger.jsonl").open("x") as ledger_stream:

        def emit(event):
            ledger_stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            ledger_stream.flush()
            if event["event"] == "materialized_workload":
                os.fsync(ledger_stream.fileno())

        wrapped = BlockLedger(view, emit, started=started, clock=clock)

        def measured_guard(workload, **details):
            wrapped.confirm(workload)
            guard(workload, **details)

        def checkpoint(event):
            if getattr(guard, "recent", None):
                event = dict(event, peak_memory_bytes=guard.recent[-1]["peak_bytes"])
                report["peak_memory_bytes"] = guard.recent[-1]["peak_bytes"]
            ledger_stream.flush()
            os.fsync(ledger_stream.fileno())
            report.update(
                elapsed_s=clock() - started, block_accounting=wrapped.summary(), progress=event
            )
            atomic_record(directory / "diagnostic.json", report)
            if progress:
                progress(event)

        try:
            checkpoint(dict(event="diagnostic_started"))
            for layer in (1, 16, 31):
                wrapped.check()
                wrapped.phase = dict(layer=layer, operation="prepare")
                checkpoint(wrapped.phase)
                state = prepare(wrapped, rows[0]["ids"], layer, 438, guard=measured_guard)
                primal_info = dict(layer=layer, full_primal_norm=float(state.primal_norm))
                if hasattr(state, "full_primal"):
                    selected = np.asarray(state.full_primal[0, 438], dtype=np.float32)
                    primal_info["selected_position_norm"] = (
                        float(np.linalg.norm(selected.astype(np.float64)))
                        if np.isfinite(selected).all()
                        else None
                    )
                report["primals"].append(primal_info)
                previous = None
                for scale in SCALES:
                    modes = (
                        ("reference", "restore", "broadcast")
                        if scale in ENDPOINTS
                        else ("reference",)
                    )
                    ref = None
                    for mode in modes:
                        wrapped.check()
                        wrapped.phase = dict(layer=layer, scale=scale, operation=mode)
                        checkpoint(wrapped.phase)
                        kwargs = dict(epsilon_scale=scale, guard=measured_guard)
                        response = (
                            reference(state, directions, **kwargs)
                            if mode == "reference"
                            else cached(state, directions, mode=mode, batch_size=8, **kwargs)
                        )
                        response = np.asarray(response, dtype=np.float32)
                        measured_guard("diagnostic_response", layer=layer, scale=scale, mode=mode)
                        if response.shape != directions.shape:
                            raise ValueError("response shape differs from direction grid")
                        filename = f"layer-{layer}-scale-{scale:g}-{mode}.npy"
                        np.save(directory / filename, response, allow_pickle=False)
                        norms = [
                            float(np.linalg.norm(row.astype(np.float64)))
                            if np.isfinite(row).all()
                            else None
                            for row in response
                        ]
                        report["measurements"].append(
                            dict(
                                layer=layer,
                                scale=scale,
                                mode=mode,
                                epsilon=finite_difference_steps(
                                    state.primal_norm, np.linalg.norm(directions, axis=1), scale
                                ).tolist(),
                                response_norms=norms,
                                nonfinite_coordinates=int((~np.isfinite(response)).sum()),
                                file=filename,
                                file_sha256=hashlib.sha256(
                                    (directory / filename).read_bytes()
                                ).hexdigest(),
                            )
                        )
                        if mode == "reference":
                            ref = response
                            if previous is not None:
                                report["comparisons"].append(
                                    dict(
                                        kind="adjacent_stability",
                                        layer=layer,
                                        reference_scale=scale,
                                        candidate_scale=scale / 2,
                                        **compare(ref, previous, atol=0.003, rtol=0.03),
                                    )
                                )
                        else:
                            report["comparisons"].append(
                                dict(
                                    kind="cache_reference",
                                    layer=layer,
                                    scale=scale,
                                    mode=mode,
                                    **compare(ref, response, atol=0.0003, rtol=0.003),
                                )
                            )
                        checkpoint(dict(event="measurement_completed", **wrapped.phase))
                    previous = ref
                del state, previous, ref, response
            wrapped.check()
            if wrapped.pending:
                raise RuntimeError("unconfirmed block calls at diagnostic end")
            report["status"] = (
                "completed_with_nonfinite_failures"
                if any(m["nonfinite_coordinates"] for m in report["measurements"])
                else "completed"
            )
        except BaseException as error:
            report.update(status="incomplete", error_type=type(error).__name__, error=str(error))
            if hasattr(error, "report"):
                report["stop_details"] = error.report
        finally:
            checkpoint(dict(event="diagnostic_finished", status=report["status"]))
    return report
