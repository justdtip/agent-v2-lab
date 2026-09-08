"""Bounded regression timing/memory probe; no lens artifacts or quality readings.

Run only after a committed registration and a free primary model slot. R47(b)
projects each larger size before executing it. The first two sizes require an
explicit inspected bound. This script cannot grant an above-cap run window.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from local_llm_lab.pipeline.lens_fitting.memory_policy import ladder, project_peak


def run_ladder(lengths, measure, *, fixed_bytes, initial_bound_bytes, cap_bytes, emit):
    """The measurement callback is unreachable for a refused larger row."""
    if not all(math.isfinite(v) and v > 0 for v in (initial_bound_bytes, cap_bytes)):
        raise ValueError("finite positive initial bound and cap are required")
    rows = []
    for tokens in lengths:
        projected = (
            initial_bound_bytes
            if len(rows) < 2
            else project_peak(rows, tokens, fixed_bytes=fixed_bytes)
        )
        if projected > cap_bytes:
            stop = {
                "event": "stopped",
                "reason": "initial bound exceeds cap"
                if len(rows) < 2
                else "projection exceeds cap",
                "next_tokens": tokens,
                "projected_peak_bytes": projected,
                "cap_bytes": cap_bytes,
            }
            emit(stop)
            return rows, stop
        emit({"event": "before_size", "tokens": tokens, "projected_peak_bytes": projected})
        row = measure(tokens)
        row["projected_peak_bytes"] = projected
        rows.append(row)
        emit({"event": "measured", **row})
        peak = row["peak_bytes"]
        if not math.isfinite(peak) or peak > cap_bytes:
            stop = {
                "event": "stopped",
                "reason": "measured peak exceeded cap; breach, not pre-launch protection",
                "tokens": tokens,
                "peak_bytes": peak,
                "cap_bytes": cap_bytes,
            }
            emit(stop)
            return rows, stop
        if len(rows) <= 2 and peak > initial_bound_bytes:
            stop = {
                "event": "stopped",
                "reason": "measured peak falsified initial bound",
                "tokens": tokens,
                "peak_bytes": peak,
                "projected_peak_bytes": initial_bound_bytes,
                "cap_bytes": cap_bytes,
            }
            emit(stop)
            return rows, stop
    return rows, None


def project_time(rows, lengths, *, one_layer_solve_s, nonfinal_layers):
    left, right = rows[-2:]
    t0, t1 = left["tokens"], right["tokens"]
    s0, s1 = left["seconds_per_sequence"], right["seconds_per_sequence"]
    exponent = max(1.0, math.log(s1 / s0) / math.log(t1 / t0))
    forward = sum(s1 * (length / t1) ** exponent for length in lengths)
    solves = one_layer_solve_s * nonfinal_layers
    return {
        "timing_exponent": exponent,
        "forward_seconds": forward,
        "solve_seconds": solves,
        "projected_fit_seconds": forward + solves,
        "scope": (
            "projection only; excludes loading and artifact serialization; "
            "not an under-hour acceptance"
        ),
    }


def validate_registration_source(path: Path, residual_source: str) -> None:
    """Native preflight requires a JSON declaration matching the requested producer."""
    from local_llm_lab.pipeline.lens_fitting.regression import RESIDUAL_SOURCES

    if residual_source not in RESIDUAL_SOURCES:
        raise ValueError("invalid residual_source")
    try:
        registration = json.loads(path.read_text())
    except json.JSONDecodeError:
        if residual_source == "hand_run":
            return  # Preserve the historical default with plain-text registrations.
        raise ValueError("native registration must declare residual_source in JSON") from None
    if not isinstance(registration, dict) or registration.get("residual_source") != residual_source:
        raise ValueError("registration residual_source does not match requested producer")


def calibration_rows(source: dict, tokens: int) -> list[dict]:
    """Masked rows use a suffix ending at the source end; unmasked rows retain prefixes.

    Only original scored positions inside that suffix survive, translated by its
    start. At full source length this is exactly the frozen fit row's selection.
    """
    from local_llm_lab.pipeline.lens_fitting.regression import validated_score_positions

    if (
        isinstance(tokens, bool)
        or not isinstance(tokens, int)
        or not 0 < tokens <= len(source["ids"])
    ):
        raise ValueError("calibration tokens must select a nonempty source window")
    positions = validated_score_positions(source)
    start = 0 if positions is None else len(source["ids"]) - tokens
    selected = (
        {}
        if positions is None
        else {"score_positions": [i - start for i in positions if i >= start]}
    )
    if positions is not None and not selected["score_positions"]:
        raise ValueError("calibration suffix contains no scored positions")
    return [
        {"ids": source["ids"][start : start + tokens], "split": split, **selected}
        for split in ("fit", "held")
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--planned-lens", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--initial-bound-gib", type=float, required=True)
    parser.add_argument("--residual-source", choices=("hand_run", "native"), default="hand_run")
    parser.add_argument("--statistics-storage", choices=("memory", "split_spill"), default="memory")
    parser.add_argument("--scratch-parent", type=Path)
    parser.add_argument("--diagnostic-tokens", type=int)
    args = parser.parse_args(argv)
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.pipeline.lens_fitting.runtime import load_runtime, prepare_fit
    from local_llm_lab.pipeline.live_lens.instruments import file_sha256

    if not math.isfinite(args.initial_bound_gib) or args.initial_bound_gib <= 0:
        parser.error("initial inspected bound must be positive and finite")
    from local_llm_lab.pipeline.lens_fitting.memory_policy import (
        binding,
        check_projection,
        device_working_set,
        require_owned_window,
    )

    registration_sha = file_sha256(args.registration)
    validate_registration_source(args.registration, args.residual_source)
    if args.report.exists():
        raise FileExistsError(args.report)
    device_bytes = device_working_set()  # owned window checked before MLX device query
    check_projection(args.initial_bound_gib * 2**30, device_bytes)
    prepared = prepare_fit(args.corpus, load_model_spec(args.model), args.planned_lens)
    maximum = max(len(row["ids"]) for row in prepared.rows)
    if args.diagnostic_tokens is not None and not 0 < args.diagnostic_tokens <= maximum:
        parser.error("diagnostic tokens must be positive and within the frozen corpus")
    lengths = [args.diagnostic_tokens] if args.diagnostic_tokens is not None else ladder(maximum)
    source = max((r for r in prepared.rows if r["split"] == "fit"), key=lambda r: len(r["ids"]))
    if len(source["ids"]) < max(lengths):
        parser.error("the maximum sequence is held; preflight needs a fit row reaching that size")
    for tokens in lengths:
        calibration_rows(source, tokens)  # Reject an unusable scoring window before model loading.
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with (
        TemporaryDirectory(prefix="lens-preflight-", dir=args.scratch_parent) as scratch,
        args.report.open("x") as report,
    ):

        def emit(event):
            line = json.dumps(event, allow_nan=False)
            report.write(line + "\n")
            report.flush()
            print(line, flush=True)

        emit(
            {
                "event": "begin",
                "qualification_binding": binding(
                    prepared, args.residual_source, args.statistics_storage
                ),
                "statistics_storage": args.statistics_storage,
                "mode": "diagnostic" if args.diagnostic_tokens is not None else "qualification",
                "diagnostic_repetitions_per_split": 4
                if args.diagnostic_tokens is not None
                else None,
                "model": prepared.snapshot,
                "corpus_manifest_sha256": prepared.corpus_manifest_sha256,
                "registration_sha256": registration_sha,
                "residual_source": args.residual_source,
                "source_sha256": file_sha256(Path(__file__)),
                "initial_bound_bytes": args.initial_bound_gib * 2**30,
                "lengths": lengths,
                "source_sequence_index": source["index"],
                "source_input_positions": len(source["ids"]),
                "source_scored_positions": len(source.get("score_positions", source["ids"])),
                "calibration_window_rule": (
                    "suffix ending at registered source end; intersect original score_positions "
                    "and translate by suffix start; no new scored positions"
                    if "score_positions" in source
                    else "legacy unmasked source prefix"
                ),
                "purpose": (
                    "two copies of one fit row exercise both statistics slots; "
                    "no held data or quality output"
                ),
                "buffer_cap": (
                    "bytes do not measure buffer count; "
                    "accumulation-side count remains unobservable"
                ),
            }
        )
        try:
            require_owned_window()
            check_projection(args.initial_bound_gib * 2**30, device_bytes)
            loaded = load_runtime(prepared)
            import mlx.core as mx

            from local_llm_lab.pipeline.lens_fitting.regression import accumulate, solve_layer

            loaded.model.eval()
            # Keep idle cached allocations from obscuring the active graph measurement.
            # This is a byte-cache setting, explicitly not a buffer-count safeguard.
            old_cache_limit = mx.set_cache_limit(0)
            mx.clear_cache()
            resident = mx.get_active_memory()
            device_bytes = mx.device_info()["max_recommended_working_set_size"]
            check_projection(mx.get_peak_memory(), device_bytes)
            sums_bytes = (
                (2 if args.statistics_storage == "split_spill" else 4)
                * 4
                * loaded.view.hidden_size**2
                * (loaded.view.num_layers - 1)
            )
            emit(
                {
                    "event": "loaded",
                    "lock": str(loaded.lock_path),
                    "resident_bytes": resident,
                    "load_peak_bytes": mx.get_peak_memory(),
                    "statistics_bytes": sums_bytes,
                    "working_set_bytes": device_bytes,
                    "cap_bytes": 0.6 * device_bytes,
                    "previous_cache_limit_bytes": old_cache_limit,
                    "cache_limit_bytes": 0,
                    "measurement_scope": (
                        "MLX active peak; allocator cache disabled; not OS-wide memory"
                    ),
                }
            )
            retained = None
            shape_scratch = None

            def measure(tokens):
                nonlocal retained, shape_scratch
                retained = None
                if shape_scratch is not None:
                    shutil.rmtree(shape_scratch)
                mx.clear_cache()
                mx.reset_peak_memory()
                require_owned_window()
                rows = calibration_rows(source, tokens)
                if args.diagnostic_tokens is not None:
                    rows = rows * 4
                shape_scratch = Path(scratch) / str(tokens)
                shape_scratch.mkdir()

                def memory_progress(event):
                    emit({**event, "event": "memory_phase"})
                    if event.get("phase") == "before_forward":
                        require_owned_window()
                    peak = event["peak_memory_bytes"]
                    limit = 0.6 * device_bytes
                    if tokens in lengths[:2]:
                        limit = min(limit, args.initial_bound_gib * 2**30)
                    if not math.isfinite(peak) or peak > limit:
                        raise ValueError(
                            "measured memory breach or initial-bound falsification; "
                            "stop before another calibration row, not first-spike prevention"
                        )

                started = time.perf_counter()
                retained, counts = accumulate(
                    loaded.view,
                    rows,
                    residual_source=args.residual_source,
                    statistics_storage=args.statistics_storage,
                    scratch_dir=shape_scratch,
                    memory_progress=memory_progress,
                )
                elapsed = time.perf_counter() - started
                return {
                    "tokens": tokens,
                    "peak_bytes": mx.get_peak_memory(),
                    "seconds_per_sequence": elapsed / len(rows),
                    "residual_source": args.residual_source,
                    "counts": counts,
                    "source_window_start": len(source["ids"]) - tokens
                    if "score_positions" in source
                    else 0,
                    "source_window_end": len(source["ids"])
                    if "score_positions" in source
                    else tokens,
                    "scored_positions_per_sequence": len(
                        rows[0].get("score_positions", rows[0]["ids"])
                    ),
                }

            rows, stop = run_ladder(
                lengths,
                measure,
                fixed_bytes=resident + sums_bytes,
                initial_bound_bytes=args.initial_bound_gib * 2**30,
                cap_bytes=0.6 * device_bytes,
                emit=emit,
            )
            if stop is not None:
                emit({"event": "end", "status": "stopped", "fit_started": False})
                return 2
            if args.diagnostic_tokens is not None:
                emit({"event": "end", "status": "diagnostic_complete", "fit_started": False})
                return 0
            # One complete fixed-alpha grid, using calibration sums, only for timing.
            # Quality values and maps are discarded and never written/read as findings.
            mx.clear_cache()
            # The solve adds a fixed number of dense matrices, not a longer token
            # graph. Budget sixteen float32 matrices for inputs, CPU solve work,
            # candidate/best solutions and error products, before running the grid.
            dense_bytes = 4 * loaded.view.hidden_size**2
            active_floor = mx.get_active_memory()
            host_reserve = 2 * (loaded.view.num_layers - 1) * dense_bytes
            solve_bound = active_floor + 16 * dense_bytes + host_reserve
            emit(
                {
                    "event": "full_fit_envelope",
                    "dense_matrix_bytes": dense_bytes,
                    "nonfinal_layers": loaded.view.num_layers - 1,
                    "active_floor_bytes": active_floor,
                    "host_maps_and_serialization_bytes": host_reserve,
                    "solver_workspace_bytes": 16 * dense_bytes,
                    "projected_peak_bytes": solve_bound,
                    "scope": (
                        "MLX active floor plus analytical host/solve reserves; "
                        "not measured process RSS"
                    ),
                }
            )
            emit(
                {
                    "event": "before_solve",
                    "projected_peak_bytes": solve_bound,
                    "basis": (
                        "active bytes plus sixteen dense float32 matrices and two all-layer "
                        "host map sets (fit plus artifact readback)"
                    ),
                }
            )
            if solve_bound > 0.6 * device_bytes:
                emit(
                    {
                        "event": "end",
                        "status": "stopped",
                        "reason": "solve projection exceeds cap",
                        "fit_started": False,
                    }
                )
                return 2
            mx.reset_peak_memory()
            started = time.perf_counter()
            require_owned_window()
            temporary = solve_layer(retained["fit"][1], retained["held"][1])
            solve_s = time.perf_counter() - started
            del temporary, retained
            solve_peak = mx.get_peak_memory()
            emit(
                {
                    "event": "measured_solve",
                    "peak_bytes": solve_peak,
                    "projected_peak_bytes": solve_bound,
                    "elapsed_s": solve_s,
                }
            )
            if solve_peak > 0.6 * device_bytes:
                emit(
                    {
                        "event": "end",
                        "status": "stopped",
                        "reason": "measured solve peak exceeded cap; breach",
                        "peak_bytes": solve_peak,
                        "fit_started": False,
                    }
                )
                return 2
            emit(
                {
                    "event": "timing",
                    **project_time(
                        rows,
                        [len(r["ids"]) for r in prepared.rows],
                        one_layer_solve_s=solve_s,
                        nonfinal_layers=loaded.view.num_layers - 1,
                    ),
                    "one_layer_fixed_grid_seconds": solve_s,
                }
            )
            emit({"event": "end", "status": "measured", "fit_started": False})
            return 0
        except BaseException as error:
            emit({"event": "end", "status": "error", "error": f"{type(error).__name__}: {error}"})
            raise


if __name__ == "__main__":
    raise SystemExit(main())
