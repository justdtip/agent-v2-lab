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
    parser.add_argument("--kind", choices=("regression",), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--layers", choices=("all",), default="all")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--revision", default="main", help="Existing local Hub revision; offline only"
    )
    args = parser.parse_args(argv)
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.pipeline.lens_fitting.runtime import load_runtime, prepare_fit

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
    loaded = load_runtime(prepared)
    from local_llm_lab.pipeline.lens_fitting.artifacts import write_lens
    from local_llm_lab.pipeline.lens_fitting.regression import ALPHA_GRID, fit_regression

    loaded.model.eval()
    result = fit_regression(
        loaded.view, prepared.rows, progress=lambda event: print(json.dumps(event), flush=True)
    )
    metadata = {
        "schema_version": 1,
        "kind": args.kind,
        "domain": prepared.manifest["domain"],
        "model": loaded.snapshot | {"name": loaded.spec.name},
        "corpus_manifest_sha256": prepared.corpus_manifest_sha256,
        "corpus_sequences_sha256": prepared.manifest["sequences"]["sha256"],
        "counts": result.counts,
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
    }
    written = write_lens(
        prepared.output,
        result.maps,
        hidden_size=loaded.view.hidden_size,
        num_layers=loaded.view.num_layers,
        metadata=metadata,
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
