"""Run only the approved original-full-primal four-scale epsilon diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--record-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.pipeline.lens_fitting.epsilon_diagnostic import (
        DeadlineExceeded,
        atomic_record,
        run_diagnostic,
        validate_scope,
    )
    from local_llm_lab.pipeline.lens_fitting.jacobian import (
        WorkloadMemoryGuard,
        memory_gate,
        prepare_plan,
    )
    from local_llm_lab.pipeline.lens_fitting.runtime import (
        configure_allocator_cache,
        load_runtime,
        prepare_fit,
    )
    from local_llm_lab.pipeline.live_lens.instruments import file_sha256

    args.record_dir.mkdir(parents=True, exist_ok=False)

    def progress(event):
        event = dict(event, elapsed_s=time.monotonic() - started)
        with (args.record_dir / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
            stream.flush()
            import os

            os.fsync(stream.fileno())
        print(json.dumps(event, allow_nan=False), flush=True)

    try:
        progress(dict(event="preflight_started"))
        spec = load_model_spec(args.model)
        manifest = json.loads(args.corpus.read_text())
        # prepare_fit validates this unused sentinel path; no lens is ever written.
        sentinel = (
            args.record_dir
            / f"{spec.name.replace('/', '--')}-{manifest['domain']}-jacobian-unused.npz"
        )
        prepared = prepare_fit(args.corpus, spec, sentinel, kind="jacobian", revision=args.revision)
        plan = prepare_plan(prepared, args.plan)
        validate_scope(prepared.rows, plan)
        memory_gate(8 * 2**30, plan["working_set_bytes"])
        root = Path(__file__).resolve().parents[1]
        source_paths = [
            Path(__file__).resolve(),
            *sorted((root / "src/local_llm_lab/pipeline/lens_fitting").glob("*.py")),
            root / "src/local_llm_lab/arch.py",
            root / "src/local_llm_lab/pipeline/evaluate.py",
        ]
        provenance = dict(
            snapshot=prepared.snapshot,
            plan_file_sha256=file_sha256(args.plan),
            corpus_manifest_sha256=prepared.corpus_manifest_sha256,
            source_sha256={str(p.relative_to(root)): file_sha256(p) for p in source_paths},
        )
        atomic_record(args.record_dir / "provenance.json", provenance)
        if time.monotonic() - started >= 480:
            raise DeadlineExceeded("CLI budget exhausted before load")
        progress(dict(event="load_started"))
        loaded = load_runtime(prepared)
        import mlx.core as mx

        loaded.model.eval()
        allocator = configure_allocator_cache()
        working = mx.device_info()["max_recommended_working_set_size"]
        guard = WorkloadMemoryGuard(8 * 2**30, working, mx.get_peak_memory)
        guard("after_load")
        provenance.update(
            model_run_lock=str(loaded.lock_path),
            allocator=allocator,
            actual_working_set_bytes=working,
            peak_limit_bytes=8 * 2**30,
        )
        atomic_record(args.record_dir / "provenance.json", provenance)
        report = run_diagnostic(
            loaded.view,
            prepared.rows,
            plan,
            args.record_dir,
            started=started,
            guard=guard,
            provenance=provenance,
            progress=progress,
        )
        peak = float(mx.get_peak_memory())
        import math

        report["peak_memory_bytes"] = peak if math.isfinite(peak) else None
        report["peak_memory_scope"] = "actual MLX process peak including load"
        atomic_record(args.record_dir / "diagnostic.json", report)
        return 0 if report["status"] == "completed" else 2
    except BaseException as error:
        record = dict(
            status="incomplete",
            diagnostic_only=True,
            production_acceptance=False,
            error_type=type(error).__name__,
            error=str(error),
            elapsed_s=time.monotonic() - started,
        )
        if hasattr(error, "report"):
            record["stop_details"] = error.report
        atomic_record(args.record_dir / "stop.json", record)
        progress(dict(event="stop", **record))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
