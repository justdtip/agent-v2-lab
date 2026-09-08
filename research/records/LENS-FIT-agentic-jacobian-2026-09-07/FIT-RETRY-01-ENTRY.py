"""One Director-authorized fit retry; only this process redirects its model lock."""

import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path("/Users/daniel.tipton/worktrees/lens-fitting")
RECORD = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from local_llm_lab import runlock  # noqa: E402

proof = json.loads((RECORD / "position-step-01/diagnostic.json").read_text())
if proof["status"] != "self_check_passed" or proof["selected_common_coefficient"] != 0.03:
    raise SystemExit("The registered corrected-rule self-check must pass before retry")
private_lock = RECORD / ".fit-retry-01-model-run.lock"
runlock.LOCK_RELATIVE_PATH = private_lock
runlock.hold_model_run_lock(
    path=private_lock,
    check_processes=False,
    session="Codex Director-authorized concurrent fit retry01",
)
import mlx.core as mx  # noqa: E402

mx.set_memory_limit(6 * 2**30)
mx.set_cache_limit(0)
source = ROOT / "src/local_llm_lab/pipeline/lens_fitting"
identity = dict(
    owned_pid=os.getpid(),
    concurrent_authorized=True,
    private_lock=str(private_lock),
    allocation_limit_bytes=6 * 2**30,
    diagnostic_sha256=hashlib.sha256(
        (RECORD / "position-step-01/diagnostic.json").read_bytes()
    ).hexdigest(),
    source_sha256={
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [ROOT / "scripts/lens_fit.py", *sorted(source.glob("*.py"))]
    },
)
with (RECORD / "fit-retry-01-source.json").open("x") as stream:
    json.dump(identity, stream, indent=2)
    stream.write("\n")
print(json.dumps(dict(event="fit_retry_started", **identity)), flush=True)
sys.argv = [
    str(ROOT / "scripts/lens_fit.py"),
    "--kind",
    "jacobian",
    "--model",
    "qwen35-4b",
    "--corpus",
    str(ROOT / "data/lens-fitting/qwen35-4b-agentic/manifest.json"),
    "--out",
    str(ROOT / "models/jlens/qwen35-4b-agentic-jacobian-position-step.npz"),
    "--revision",
    "32f3e8ecf65426fc3306969496342d504bfa13f3",
    "--jacobian-stage",
    "fit",
    "--jacobian-plan",
    str(RECORD / "position-step-plan.json"),
    "--record-dir",
    str(RECORD / "fit-retry-01"),
]
runpy.run_path(sys.argv[0], run_name="__main__")
