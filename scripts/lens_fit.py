"""Standalone all-layer lens fitting; input validation runs before model imports."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("regression", "jacobian"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--layers", choices=("all",), default="all")
    parser.add_argument(
        "--residual-source",
        choices=("hand_run", "native"),
        default="hand_run",
        help=(
            "which forward produces the residuals a regression fit reads. `hand_run` is this "
            "repository's decoder loop, which every Qwen fit used; `native` taps the model's own "
            "forward and is what a family the loop does not yet describe needs. Explicit with no "
            "default inference from the model, and recorded in the fit's provenance."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--revision", default="main", help="Existing local Hub revision; offline only"
    )
    parser.add_argument(
        "--jacobian-stage", choices=("plan", "check", "benchmark", "fit"), default="fit"
    )
    parser.add_argument("--jacobian-plan", type=Path)
    parser.add_argument(
        "--plan-config", type=Path, help="Explicit frozen bounds, seeds and memory envelope JSON"
    )
    parser.add_argument(
        "--record-dir", type=Path, help="New directory for append-only Jacobian stage records"
    )
    parser.add_argument("--statistics-storage", choices=("memory", "split_spill"), default="memory")
    parser.add_argument("--scratch-parent", type=Path)
    parser.add_argument("--memory-preflight", type=Path)
    args = parser.parse_args(argv)
    if args.kind == "regression" and args.memory_preflight is None:
        parser.error("regression requires matching successful --memory-preflight evidence")
    if args.kind == "jacobian":
        if args.jacobian_plan is None:
            parser.error("Jacobian stages require --jacobian-plan")
        if (args.jacobian_stage == "plan") != (args.plan_config is not None):
            parser.error("only the plan stage requires --plan-config")
        if args.jacobian_stage != "plan" and args.record_dir is None:
            parser.error("native Jacobian stages require a new --record-dir")
    elif args.jacobian_plan or args.plan_config or args.record_dir or args.jacobian_stage != "fit":
        parser.error("Jacobian stage arguments require --kind jacobian")
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.pipeline.lens_fitting.runtime import (
        configure_allocator_cache,
        load_runtime,
        prepare_fit,
        resource_snapshot,
    )

    started = time.monotonic()
    try:
        prepared = prepare_fit(
            args.corpus,
            load_model_spec(args.model),
            args.out,
            kind=args.kind,
            revision=args.revision,
        )
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    plan = None
    if args.kind == "jacobian":
        from local_llm_lab.pipeline.lens_fitting.jacobian import (
            prepare_plan,
            run_jacobian_stage,
            write_record,
        )

        try:
            plan = prepare_plan(prepared, args.jacobian_plan, config_path=args.plan_config)
            if args.jacobian_stage == "plan":
                print(json.dumps({"event": "plan_frozen", "plan_sha256": plan["plan_sha256"]}))
                return 0
            args.record_dir.mkdir(parents=True, exist_ok=False)
            write_record(args.record_dir / "run-plan.json", plan)
        except (OSError, ValueError, KeyError) as error:
            parser.error(str(error))
    qualification = None
    if args.kind == "regression":
        from local_llm_lab.pipeline.lens_fitting.memory_policy import (
            binding,
            device_working_set,
            guard_runtime_event,
            require_owned_window,
            validate_evidence,
        )

        qualification = validate_evidence(
            args.memory_preflight,
            binding(prepared, args.residual_source, args.statistics_storage),
            device_working_set(),
        )
        require_owned_window()
    loaded = load_runtime(prepared)
    from local_llm_lab.pipeline.lens_fitting.artifacts import write_lens
    from local_llm_lab.pipeline.lens_fitting.regression import ALPHA_GRID, fit_regression
    from local_llm_lab.pipeline.live_lens.instruments import LensIdentity

    if qualification is not None:
        from local_llm_lab.pipeline.lens_fitting.memory_policy import check_projection

        check_projection(
            resource_snapshot()["peak_memory_gib"] * 2**30, qualification["working_set_bytes"]
        )
    loaded.model.eval()
    allocator_cache = configure_allocator_cache()

    def enrich(event):
        counts = event.get("counts", {})
        tokens = (
            event["tokens_done"]
            if "tokens_done" in event
            else sum(
                counts[split].get("positions", 0)
                for split in ("fit", "held")
                if isinstance(counts.get(split), dict)
            )
        )
        return {
            **event,
            "elapsed_s": time.monotonic() - started,
            "tokens_done": tokens,
            **resource_snapshot(),
        }

    def progress(event):
        report = enrich(event)
        if qualification is not None:
            guard_runtime_event(
                {"phase": "progress", "peak_memory_bytes": report["peak_memory_gib"] * 2**30},
                qualification,
            )
        print(json.dumps(report), flush=True)

    def memory_progress(event):
        guard_runtime_event(event, qualification)
        print(json.dumps({**event, "event": "memory_phase"}), flush=True)

    validation = None
    if args.kind == "regression":
        require_owned_window()
        result = fit_regression(
            loaded.view,
            prepared.rows,
            progress=progress,
            residual_source=args.residual_source,
            statistics_storage=args.statistics_storage,
            scratch_parent=args.scratch_parent,
            memory_progress=memory_progress,
        )
    else:
        write_record(
            args.record_dir / "runtime.json",
            {
                "snapshot": loaded.snapshot,
                "model_run_lock": str(loaded.lock_path),
                "allocator_cache": allocator_cache,
                "allocator_scope": (
                    "unused allocation bytes only; does not establish live buffer-count safety"
                ),
            },
        )
        with (args.record_dir / "progress.jsonl").open("x") as events:

            def progress(event):
                event = enrich(event)
                events.write(json.dumps(event, allow_nan=False) + "\n")
                events.flush()
                print(json.dumps(event), flush=True)

            try:
                outcome = run_jacobian_stage(
                    loaded.view,
                    prepared.rows,
                    plan,
                    stage=args.jacobian_stage,
                    record_dir=args.record_dir,
                    progress=progress,
                )
            except (ValueError, RuntimeError) as error:
                print(json.dumps({"event": "stop_required", "reason": str(error)}))
                return 2
        if outcome is None:
            return 0
        result, validation = outcome
    metadata = {
        "schema_version": 1,
        "kind": args.kind,
        "domain": prepared.manifest["domain"],
        "snapshot": loaded.snapshot,
        "model_name": loaded.spec.name,
        "corpus_manifest_sha256": prepared.corpus_manifest_sha256,
        "corpus_sequences_sha256": prepared.manifest["sequences"]["sha256"],
        "counts": result.counts,
        # Top level as well as inside counts: a reader deciding whether two fits are comparable
        # should not have to dig for the forward that produced them.
        "residual_source": args.residual_source,
        "per_layer": result.per_layer,
        "lambda_grid": ALPHA_GRID,
        "penalty_rule": "alpha * mean_diag(total fit XTX) = n * lambda_per_position",
        "held_split_role": (
            "sequence-level alpha selection; agentic fit/held may share trajectories"
        ),
        "generalisation_evaluation": "disjoint pilot episodes; not measured by this fit",
        "r2_definition": "uncentered: 1 - held SSE / held sum(Y squared)",
        "seeds": {},
        "elapsed_s": time.monotonic() - started,
        "fit_elapsed_s": result.elapsed_s,
        "peak_memory_gib": result.peak_memory_gib,
        "peak_memory_scope": "MLX process peak, including model loading",
        "model_run_lock": str(loaded.lock_path),
        "cache_strategy": loaded.resolved.cache_strategy,
        "allocator_cache": allocator_cache,
        "statistics_storage": args.statistics_storage if args.kind == "regression" else None,
        "memory_qualification": qualification,
    }
    if args.kind == "jacobian":
        for key in ("lambda_grid", "penalty_rule", "r2_definition"):
            metadata.pop(key)
        metadata.update(
            {
                "held_split_role": (
                    "equal-span held mean derivative validation; no held fit positions"
                ),
                "seeds": plan["seeds"],
                "plan_sha256": plan["plan_sha256"],
                "run_plan": plan,
                "validation": validation,
                "record_dir": str(args.record_dir),
                "orientation": "output-by-input; prediction directions @ J.T",
            }
        )
    if qualification is not None:
        # Peak is a conservative upper bound for the current MLX active floor here.
        peak = resource_snapshot()["peak_memory_gib"] * 2**30
        guard_runtime_event(
            {"phase": "artifact_start", "peak_memory_bytes": peak, "active_memory_bytes": peak},
            qualification,
        )
    written = write_lens(
        prepared.output,
        result.maps,
        hidden_size=loaded.view.hidden_size,
        num_layers=loaded.view.num_layers,
        metadata=metadata,
        # Issue 99: a lens we fit carries the model it was fitted on, inside the archive whose
        # digest every later reader pins.
        identity=LensIdentity(loaded.spec.base, loaded.view.num_layers, loaded.spec.training),
    )
    print(
        json.dumps(
            {
                "event": "done",
                "npz": str(prepared.output),
                "npz_sha256": written["npz_sha256"],
                "elapsed_s": written["elapsed_s"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
