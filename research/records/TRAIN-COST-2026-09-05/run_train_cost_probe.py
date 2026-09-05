"""Launcher: one child per (variant, row length), projection-gated, kernel-pressure guarded (R46, R47).

The launcher imports no MLX, so it may start children freely. Before each child it projects the
peak from the last two ok rows of that variant (linear in tokens) and refuses any length whose
projection exceeds SHARE_CAP of the working set; a child that fails is the end of its variant. A
guard thread reads kern.memorystatus_vm_pressure_level every 10 s and terminates the child on two
consecutive critical samples (R46(iv)); swap free is logged beside it. Every child's stdout is
echoed line by line, so the progress line is visible from outside.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHILD = HERE / "train_cost_probe.py"
PY = "/Users/daniel.tipton/Desktop/An app/.venv/bin/python"
ROWS = HERE / "rows.jsonl"
LOG = HERE / "launcher.log"
SHARE_CAP = float(os.environ.get("SHARE_CAP", "0.85"))
LENGTHS = [int(x) for x in os.environ.get("LENGTHS", "512,1024,2048,2688,4096,6144,8192,12288,16384").split(",")]
VARIANTS = os.environ.get("VARIANTS", "A,B").split(",")
WS_GIB = 17.76
CHILD_TIMEOUT = 1800


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def sysctl(name: str) -> str:
    return subprocess.run([shutil.which("sysctl"), "-n", name], capture_output=True, text=True).stdout.strip()


def libmlx_holders() -> list[str]:
    out = subprocess.run([shutil.which("lsof"), "-w", "-d", "txt", "-F", "pn"], capture_output=True, text=True).stdout
    pid, found = None, []
    for line in out.splitlines():
        if line.startswith("p"):
            pid = line[1:]
        elif line.startswith("n") and line.endswith("libmlx.dylib") and pid:
            found.append(pid)
    return sorted(set(found))


def guard(proc: subprocess.Popen, stop: threading.Event, record: dict) -> None:
    critical = 0
    while not stop.wait(10):
        level = sysctl("kern.memorystatus_vm_pressure_level")
        swap = sysctl("vm.swapusage")
        critical = critical + 1 if level == "4" else 0
        log(f"guard pid={proc.pid} pressure={level} critical_run={critical} swap=[{swap}]")
        if critical >= 2:
            record["killed"] = f"kernel pressure critical on two consecutive samples; swap {swap}"
            log(f"guard KILL pid={proc.pid}: {record['killed']}")
            proc.terminate()
            return


def project(points: list[tuple[int, float]], tokens: int) -> float | None:
    if len(points) < 2:
        return None
    (t0, p0), (t1, p1) = points[-2], points[-1]
    if t1 == t0:
        return None
    slope = (p1 - p0) / (t1 - t0)
    return p1 + slope * (tokens - t1)


def main() -> int:
    holders = libmlx_holders()
    if holders:
        log(f"refusing to start: libmlx mapped in pids {holders}")
        return 2
    log(f"start variants={VARIANTS} lengths={LENGTHS} share_cap={SHARE_CAP}")
    for variant in VARIANTS:
        points: list[tuple[int, float]] = []
        for tokens in LENGTHS:
            projected = project(points, tokens)
            if projected is not None and projected / WS_GIB > SHARE_CAP:
                log(f"{variant} {tokens}: projected peak {projected:.2f} GiB = {projected / WS_GIB:.2f} of the working set, above cap; variant stops here")
                with ROWS.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"variant": variant, "tokens": tokens, "ok": False, "skipped": "projection above cap", "projected_peak_gib": projected}) + "\n")
                break
            out = HERE / f"row-{variant}-{tokens}.json"
            log(f"{variant} {tokens}: launching (projection {'n/a' if projected is None else f'{projected:.2f} GiB'})")
            proc = subprocess.Popen(
                [PY, str(CHILD), "--variant", variant, "--tokens", str(tokens), "--out", str(out)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                env={**os.environ, "PYTHONPATH": "/Users/daniel.tipton/Desktop/An app/src", "MODEL_RUN_SESSION": "CRO train-cost probe"},
            )
            stop, record = threading.Event(), {}
            t = threading.Thread(target=guard, args=(proc, stop, record), daemon=True)
            t.start()
            started = time.time()
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        log(f"  [{variant} {tokens}] {line[:300]}")
                    if time.time() - started > CHILD_TIMEOUT:
                        record["killed"] = "child timeout"
                        proc.terminate()
                proc.wait(timeout=60)
            finally:
                stop.set()
            row = json.loads(out.read_text()) if out.exists() else {"variant": variant, "tokens": tokens, "ok": False, "error": "no row written"}
            row["exit_status"] = proc.returncode
            row["killed"] = record.get("killed")
            with ROWS.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            if row.get("ok"):
                points.append((tokens, row["peak_gib"]))
                log(f"{variant} {tokens}: ok peak {row['peak_gib']:.2f} GiB ({row['peak_ws_share']:.2f} ws) step {row['step_s']:.1f}s ({row['step_tokens_per_s']:.0f} tok/s)")
            else:
                log(f"{variant} {tokens}: FAILED exit={proc.returncode} killed={record.get('killed')} error={str(row.get('error'))[:200]}; variant stops here")
                break
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
