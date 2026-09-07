"""Run requirements §15; concurrent loading requires an explicit one-run flag."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "revision"):
        parser.add_argument("--" + name, required=True)
    for name in ("corpus", "plan", "record-dir", "registration"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--allow-concurrent-run", action="store_true")
    args = parser.parse_args()
    from local_llm_lab import runlock
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.pipeline.lens_fitting.epsilon_diagnostic import atomic_record, validate_scope
    from local_llm_lab.pipeline.lens_fitting.jacobian import WorkloadMemoryGuard, prepare_plan
    from local_llm_lab.pipeline.lens_fitting.position_step_diagnostic import run_sweep
    from local_llm_lab.pipeline.lens_fitting.runtime import (
        configure_allocator_cache,
        load_runtime,
        prepare_fit,
        primary_worktree,
    )
    from local_llm_lab.pipeline.live_lens.instruments import file_sha256

    directory = args.record_dir.resolve()
    registration = args.registration.read_text()
    if "Director-authorized concurrent load" not in registration and args.allow_concurrent_run:
        parser.error("concurrent run requires its written registration")
    directory.mkdir(parents=True, exist_ok=False)

    def progress(event):
        row = dict(event, elapsed_s=time.monotonic() - started)
        with (directory / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(row, allow_nan=False), flush=True)

    try:
        spec = load_model_spec(args.model)
        sentinel = directory / f"{spec.name}-agentic-jacobian-unused.npz"
        prepared = prepare_fit(args.corpus, spec, sentinel, kind="jacobian", revision=args.revision)
        plan = prepare_plan(prepared, args.plan)
        validate_scope(prepared.rows, plan)
        root = Path(__file__).resolve().parents[1]
        primary = primary_worktree()
        sources = [
            Path(__file__).resolve(),
            root / "src/local_llm_lab/arch.py",
            root / "src/local_llm_lab/runlock.py",
            *sorted((root / "src/local_llm_lab/pipeline/lens_fitting").glob("*.py")),
        ]
        provenance = dict(
            pid=os.getpid(),
            snapshot=prepared.snapshot,
            corpus_manifest_sha256=prepared.corpus_manifest_sha256,
            plan_file_sha256=file_sha256(args.plan),
            registration_sha256=file_sha256(args.registration),
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            source_sha256={str(p.relative_to(root)): file_sha256(p) for p in sources},
            concurrent_authorized=args.allow_concurrent_run,
            primary_state_before={
                str(p): p.read_text() if p.exists() else None
                for p in (primary / "outputs/.model-run.lock", primary / "outputs/.box-window.json")
            },
        )
        atomic_record(directory / "provenance.json", provenance)
        if args.allow_concurrent_run:
            # Process-local exception only. Existing primary lock/window remain untouched.
            private_lock = directory / ".model-run.lock"
            runlock.LOCK_RELATIVE_PATH = private_lock
            runlock.hold_model_run_lock(
                path=private_lock,
                check_processes=False,
                session="Codex Director-authorized concurrent §15 diagnostic",
            )
        else:
            runlock.PROJECT_ROOT = primary
            runlock.hold_model_run_lock(path=primary / runlock.LOCK_RELATIVE_PATH)
        import mlx.core as mx

        limit = 6 * 2**30
        previous_limit = mx.set_memory_limit(limit)
        allocator = configure_allocator_cache()
        progress(dict(event="load_started", pid=os.getpid(), allocation_limit_bytes=limit))
        loaded = load_runtime(prepared)
        loaded.model.eval()
        guard = WorkloadMemoryGuard(
            limit, mx.device_info()["max_recommended_working_set_size"], mx.get_peak_memory
        )
        guard("after_load")
        provenance.update(
            model_run_lock=str(loaded.lock_path),
            allocator=allocator,
            allocation_limit_bytes=limit,
            previous_allocation_limit_bytes=previous_limit,
        )
        atomic_record(directory / "provenance.json", provenance)
        report = run_sweep(
            loaded.view,
            prepared.rows,
            plan,
            directory,
            started=started,
            guard=guard,
            provenance=provenance,
            progress=progress,
        )
        report["peak_memory_bytes"] = float(mx.get_peak_memory())
        atomic_record(directory / "diagnostic.json", report)
        progress(
            dict(
                event="done", status=report["status"], peak_memory_bytes=report["peak_memory_bytes"]
            )
        )
        return 0 if report["status"] == "self_check_passed" else 2
    except BaseException as error:
        atomic_record(
            directory / "failure.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                elapsed_s=time.monotonic() - started,
            ),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
