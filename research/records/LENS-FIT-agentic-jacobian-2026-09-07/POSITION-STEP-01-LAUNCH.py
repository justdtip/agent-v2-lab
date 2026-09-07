"""Supervise only the Director-authorized concurrent diagnostic child."""

import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

WORKTREE = Path("/Users/daniel.tipton/worktrees/lens-fitting")
PRIMARY = Path("/Users/daniel.tipton/Desktop/An app")
RECORD = Path(__file__).resolve().parent
RUN = RECORD / "position-step-01"
HEARTBEAT = PRIMARY / "design_specifications/live/HEARTBEAT-LOG-2026-09-04.md"
PYTHON = PRIMARY / ".venv/bin/python"
command = [
    str(PYTHON),
    str(WORKTREE / "scripts/lens_position_step_diagnostic.py"),
    "--model",
    "qwen35-4b",
    "--corpus",
    str(WORKTREE / "data/lens-fitting/qwen35-4b-agentic/manifest.json"),
    "--plan",
    str(RECORD / "plan.json"),
    "--revision",
    "32f3e8ecf65426fc3306969496342d504bfa13f3",
    "--record-dir",
    str(RUN),
    "--registration",
    str(RECORD / "POSITION-STEP-01-REGISTRATION.md"),
    "--allow-concurrent-run",
]


def now():
    return datetime.now(UTC).isoformat()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def append(message):
    with HEARTBEAT.open("a") as stream:
        stream.write("\n- " + now() + " " + message + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def pressure():
    result = subprocess.run(
        ["/usr/sbin/sysctl", "kern.memorystatus_vm_pressure_level", "vm.swapusage"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    match = re.search(r"kern.memorystatus_vm_pressure_level:\s*(\d+)", result.stdout)
    return (int(match.group(1)) if match and result.returncode == 0 else None), result.stdout


if RUN.exists():
    raise SystemExit("Fresh run directory required")
child = None
reason = None
exit_code = None
started = time.monotonic()


def interrupted(sig, frame):
    raise SystemExit(128 + sig)


for sig in (signal.SIGTERM, signal.SIGHUP):
    signal.signal(sig, interrupted)
try:
    level, observation = pressure()
    if level is None:
        raise RuntimeError("pressure supervision unavailable before load")
    append(
        "(**Director-authorized concurrent diagnostic: Codex position-step01 begins**). "
        "The Director explicitly overrides single-load exclusivity "
        "for this diagnosis and conditional retry. "
        "Existing Deputy evaluation, primary model lock and window stay intact. "
        "Only the Codex child uses a private record-directory lock; 6 GiB allocation limit, "
        "cache0, 20-minute diagnostic deadline, pressure supervised, no foreign signals. "
        "Frozen §15 grid, original bounds, source and registration in "
        "`codex/lens-fitting:research/records/LENS-FIT-agentic-jacobian-2026-09-07/`. "
        "This is a concurrent exception, not an exclusive window or general release."
    )
    with (
        (RECORD / "position-step-01.log").open("x") as output,
        (RECORD / "position-step-01-supervision.jsonl").open("x") as supervision,
    ):
        env = dict(os.environ, PYTHONPATH=str(WORKTREE / "src"), PYTHONDONTWRITEBYTECODE="1")
        child = subprocess.Popen(
            command,
            cwd=WORKTREE,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        launch = dict(
            started_utc=now(),
            owned_pid=child.pid,
            wrapper_pid=os.getpid(),
            command=command,
            concurrent_authorized=True,
            private_lock=str(RUN / ".model-run.lock"),
        )
        save(RECORD / "position-step-01-launch.json", launch)
        print(json.dumps(launch), flush=True)
        critical = 0
        while child.poll() is None:
            level, observation = pressure()
            sample = dict(
                time_utc=now(),
                elapsed_s=time.monotonic() - started,
                owned_pid=child.pid,
                kernel_pressure=level,
                observation=observation,
            )
            supervision.write(json.dumps(sample) + "\n")
            supervision.flush()
            os.fsync(supervision.fileno())
            print(json.dumps(sample), flush=True)
            critical = critical + 1 if level == 4 else 0
            if level is None or critical >= 2 or time.monotonic() - started >= 1260:
                reason = "pressure unavailable/critical or diagnostic deadline exceeded"
                break
            with contextlib.suppress(subprocess.TimeoutExpired):
                child.wait(timeout=30)
        exit_code = child.poll()
finally:
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, signal.SIG_IGN)
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=30)
    if child is not None:
        exit_code = child.returncode
    ended = dict(
        ended_utc=now(),
        exit_code=exit_code,
        reason=reason,
        elapsed_s=time.monotonic() - started,
        owned_child_exited=child is None or child.poll() is not None,
        private_lock_remaining=(RUN / ".model-run.lock").exists(),
    )
    save(RECORD / "position-step-01-end.json", ended)
    append(
        "(**Codex concurrent diagnostic position-step01 ends**). "
        + json.dumps(ended)
        + " Primary lock/window and foreign processes were not modified; "
        "this end line does not release their ownership."
    )
    print(json.dumps(ended), flush=True)
sys.exit(exit_code if exit_code is not None else 2)
