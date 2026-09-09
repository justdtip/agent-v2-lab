"""One registered native regression run, under the normal shared window and lock.

Guarded like every other records script that reaches the model. This one holds the model-run lock
and loads a 4B checkpoint, so running it by hand -- by a reader retracing the record, or by a test
sweep -- starts a model process outside R47, R48 and the issue-83 lock. It arrived on this branch
with the regression record on 2026-09-09 and the repository-rule tests caught it immediately; the
guard is added rather than the rule relaxed.

The record's content is untouched. This adds a refusal and changes nothing the run produced.
"""
import sys as _sys

if __name__ == "__main__" and "--i-am-a-record" not in _sys.argv:
    _sys.exit(
        "refusing to run: this file is the record of the native regression fit, not a launcher. "
        "It takes the model-run lock and loads a 4B checkpoint. Re-run it deliberately with "
        "--i-am-a-record, inside an announced box window."
    )

import hashlib
import json
import runpy
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.dont_write_bytecode = True


def timeout(signum, frame):
    raise TimeoutError("registered 65-minute load/fit/save deadline exceeded")


def main():
    started = time.monotonic()
    output = PROJECT / "models/jlens/gemma3-4b-bf16-prose-regression-native.npz"
    status = "error"
    try:
        for relative, digest in json.loads((ROOT / "source-hashes.json").read_text()).items():
            with (PROJECT / relative).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != digest:
                raise ValueError("registered source changed: " + relative)
        from local_llm_lab import runlock
        runlock.hold_model_run_lock()
        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(65 * 60)
        import mlx.core as mx
        working = mx.device_info()["max_recommended_working_set_size"]
        cap = min(14 * 2**30, int(0.9 * working))
        mx.set_memory_limit(cap)
        mx.set_cache_limit(0)
        print(json.dumps({"event": "registered_begin", "allocation_cap_bytes": cap,
                          "working_set_bytes": working, "residual_source": "native"}), flush=True)
        argv = ["--kind", "regression", "--model", "gemma3-4b-bf16",
                "--corpus", str(PROJECT / "data/lens-fitting/gemma3-4b-bf16-prose-128/manifest.json"),
                "--residual-source", "native", "--layers", "all", "--out", str(output)]
        result = runpy.run_path(str(PROJECT / "scripts/lens_fit.py"))["main"](argv)
        status = "completed" if result == 0 else "failed"
        return result
    finally:
        signal.alarm(0)
        with (ROOT / "run-end.json").open("x") as stream:
            json.dump({"status": status, "elapsed_s": time.monotonic() - started,
                       "artifact_exists": output.exists()}, stream, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
