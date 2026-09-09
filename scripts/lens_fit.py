"""Standalone all-layer lens fitting; input validation runs before model imports.

Both fits here are MLX. The backend is read once, after argument validation and before the first
import that reaches MLX, and a non-MLX backend is refused rather than allowed to fall through —
see the comment at that seam for why a refusal is the right answer and not a temporary one.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: The torch stage, which exists on the CUDA line only. Named once so the dispatch below and the
#: refusal beside it cannot drift apart, and so `error.name` is compared against a constant rather
#: than a string repeated at both ends.
TORCH_STAGE_MODULE = "local_llm_lab.pipeline.lens_fitting.fit_torch"


def load_model_spec_for(name: str):
    """The registry lookup, deferred so a torch box does not import the MLX side to reach it."""
    from local_llm_lab.models import load_model_spec

    return load_model_spec(name)


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
    args = parser.parse_args(argv)
    if args.kind == "jacobian":
        if args.jacobian_plan is None:
            parser.error("Jacobian stages require --jacobian-plan")
        if (args.jacobian_stage == "plan") != (args.plan_config is not None):
            parser.error("only the plan stage requires --plan-config")
        if args.jacobian_stage != "plan" and args.record_dir is None:
            parser.error("native Jacobian stages require a new --record-dir")
    elif args.jacobian_plan or args.plan_config or args.record_dir or args.jacobian_stage != "fit":
        parser.error("Jacobian stage arguments require --kind jacobian")
    # The backend is read here and nowhere else in this script, so the MLX path below is reached by
    # exactly the same code it always was — the shape `stage_train` uses (plan §16.5). The import is
    # deferred rather than module-scope for the same reason it is there: `device` is stdlib-only,
    # but everything after this point reaches MLX, and a torch-only box must get its answer before
    # any of that is imported.
    from local_llm_lab import device

    backend = device.backend()
    if backend != "mlx":
        # The dispatch, and the refusal behind it. `fit_torch` lives on the CUDA line only, so this
        # one file is identical on both branches and neither has to carry a divergent copy of the
        # seam: where the stage exists it runs, and where it does not the refusal explains itself.
        #
        # The `error.name` test is not defensive tidiness. Catching ImportError broadly here would
        # turn *any* import failure inside the stage — a missing torch, a typo in a submodule — into
        # "not implemented on this backend", which is the over-broad-catch shape this repository has
        # now found four times. Only the stage module itself being absent reaches the refusal;
        # anything else is a real failure and is raised with its own name.
        #
        # `import_module`, not `from ... import fit_torch`: the `from` form raises with `name` set
        # to the *package*, so the discrimination above would compare against the wrong string and
        # re-raise on the one case the refusal exists for. Found by this file's own tests.
        try:
            stage = importlib.import_module(TORCH_STAGE_MODULE)
        except ModuleNotFoundError as error:
            if error.name != TORCH_STAGE_MODULE:
                raise
        else:
            return stage.run(args, load_model_spec_for(args.model), progress=None)

        # Deliberately a refusal and not a fall-through. Both fits below are MLX: `regression.py`
        # has no torch port on this branch and `jacobian.py` is this repository's finite-difference
        # estimator. Falling through on a torch-only box fails several imports deep, with an error
        # about a missing module rather than about a missing fit, which is the failure this guard
        # exists to replace. `run_native.py` shipped once without one; this is that lesson applied
        # before rather than after.
        #
        # The estimator sentence is the load-bearing one. When the torch path lands it will be
        # upstream's exact autograd, not a port of the finite-difference stage, so the two backends
        # will produce *different instruments* fitted on the same corpus. That difference is the
        # golden test's finding and it must be declared in provenance, never absorbed by a CLI that
        # silently gives each box whichever estimator it can run.
        print(
            json.dumps(
                {
                    "event": "refused",
                    "reason": "lens fitting has no torch implementation yet",
                    "backend": backend,
                    "backend_source": (
                        f"${device.BACKEND_ENV}" if os.environ.get(device.BACKEND_ENV)
                        else "first installed, MLX first"
                    ),
                    "kind": args.kind,
                    "implemented_on": "mlx",
                    "will_live_in": TORCH_STAGE_MODULE,
                    "estimator_here": "finite difference (lens_fitting/jacobian.py)",
                    "estimator_there": "exact autograd (upstream jlens)",
                    "not_interchangeable": (
                        "the two estimators are different instruments on the same corpus; the "
                        "difference is a measurement, so neither backend may stand in for the other"
                    ),
                    "override": (
                        f"{device.BACKEND_ENV}=mlx if MLX is installed on this box"
                    ),
                }
            )
        )
        return 3
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
    loaded = load_runtime(prepared)
    from local_llm_lab.pipeline.lens_fitting.artifacts import write_lens
    from local_llm_lab.pipeline.lens_fitting.regression import ALPHA_GRID, fit_regression
    from local_llm_lab.pipeline.live_lens.instruments import LensIdentity

    loaded.model.eval()
    allocator_cache = configure_allocator_cache()

    def enrich(event):
        counts = event.get("counts", {})
        tokens = event.get(
            "tokens_done",
            sum(counts.get(split, {}).get("positions", 0) for split in ("fit", "held")),
        )
        return {
            **event,
            "elapsed_s": time.monotonic() - started,
            "tokens_done": tokens,
            **resource_snapshot(),
        }

    def progress(event):
        print(json.dumps(enrich(event)), flush=True)

    validation = None
    if args.kind == "regression":
        result = fit_regression(
            loaded.view,
            prepared.rows,
            progress=progress,
            residual_source=args.residual_source,
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
        "model": loaded.snapshot | {"name": loaded.spec.name},
        "checkpoint": loaded.snapshot,
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
    written = write_lens(
        prepared.output,
        result.maps,
        hidden_size=loaded.view.hidden_size,
        num_layers=loaded.view.num_layers,
        metadata=metadata,
        # Issue 99: a lens we fit carries the model it was fitted on, inside the archive whose
        # digest every later reader pins.
        identity=LensIdentity(
            base=loaded.spec.base,
            num_layers=loaded.view.num_layers,
            training=loaded.spec.training,
        ),
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
