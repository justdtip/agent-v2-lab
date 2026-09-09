"""Repeat this record's CPU-only acceptance; never import either native MLX package."""

from __future__ import annotations

import argparse
import builtins
import contextlib
import json
import os
import platform
import sys
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import psutil

from local_llm_lab import runlock
from local_llm_lab.spawn import run
from research.acceptance.provenance import MEASURED_HERE, Measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("tests", nargs="+")
    args = parser.parse_args()
    if os.environ.get("LLL_BACKEND") != "torch" or os.environ.get("LLL_DEVICE") != "cpu":
        parser.error("set LLL_BACKEND=torch and LLL_DEVICE=cpu")
    original_import = builtins.__import__

    def guarded_import(name, *pos, **kw):
        if name.split(".")[0] in {"mlx", "mlx_lm"}:
            raise ImportError("native MLX is prohibited in this CPU fixture run")
        return original_import(name, *pos, **kw)

    builtins.__import__ = guarded_import
    root = Path(__file__).resolve().parents[3]
    output = Path(__file__).resolve().parent
    commit = run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    diff = run(
        ["git", "-C", str(root), "status", "--short"], capture_output=True, text=True, check=True
    ).stdout
    window, lock = runlock.read_window(), runlock.read_lock()
    basis = "this CPU fixture run; an active desktop, not certified idle"
    report = {
        "source_commit": commit,
        "working_tree_status": diff,
        "started_utc": datetime.now(UTC).isoformat(),
        "device": "cpu",
        "platform": platform.platform(),
        "checkpoint_sha256": None,
        "scope": "synthetic fixtures and source contracts; no checkpoint loaded",
        "box_state": {
            "idle": "not established",
            "cpu_percent_before": Measured(
                psutil.cpu_percent(interval=1), MEASURED_HERE, basis
            ).as_dict(),
            "load_average": Measured(list(os.getloadavg()), MEASURED_HERE, basis).as_dict(),
            "window": window.describe() if window else None,
            "lock": lock.describe() if lock else None,
        },
        "tests": args.tests,
        "native_mlx_imports": "blocked",
    }
    import pytest
    import torch

    torch.set_num_threads(1)
    report["torch"] = torch.__version__
    xml = output / f"{args.name}.xml"
    start = time.monotonic()
    with (
        (output / f"{args.name}.log").open("w") as stream,
        contextlib.redirect_stdout(stream),
        contextlib.redirect_stderr(stream),
    ):
        status = int(pytest.main([*args.tests, "--junitxml", str(xml)]))
    suites = ET.parse(xml).getroot()
    report["counts"] = {
        key: Measured(
            sum(int(suite.get(key, 0)) for suite in suites), MEASURED_HERE, basis
        ).as_dict()
        for key in ("tests", "failures", "errors", "skipped")
    }
    report["exit_code"] = status
    report["elapsed_seconds"] = Measured(time.monotonic() - start, MEASURED_HERE, basis).as_dict()
    report["ended_utc"] = datetime.now(UTC).isoformat()
    report["native_mlx_modules_loaded"] = sorted(
        name for name in sys.modules if name.split(".")[0] in {"mlx", "mlx_lm"}
    )
    (output / f"{args.name}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"name": args.name, "exit_code": status, "counts": report["counts"]}))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
