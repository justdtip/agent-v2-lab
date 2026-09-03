"""Refuse to load the 3B checkpoint while an evaluation is already using the GPU.

A pipeline evaluation and a probe both want the whole model resident; running them together
either thrashes unified memory or silently slows the evaluation whose numbers are being
collected. Every probe CLI that loads the real model checks first.
"""

from __future__ import annotations

import argparse
import subprocess

__all__ = ["GPU_PROCESS_PATTERN", "add_gpu_arguments", "gpu_users", "require_idle_gpu"]

GPU_PROCESS_PATTERN = "agent-pipeline|agent-v2-eval|agent-v2-rollout|agent-v2-branch"


def gpu_users(pattern: str = GPU_PROCESS_PATTERN) -> list[str]:
    """Lines from ``pgrep -fl <pattern>``, excluding this process; empty when the GPU is free."""
    import os

    try:
        completed = subprocess.run(
            ["pgrep", "-fl", pattern], capture_output=True, text=True, check=False, timeout=10
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - pgrep is always present
        return []
    mine = str(os.getpid())
    return [
        line
        for line in completed.stdout.splitlines()
        if line.strip() and line.split(" ", 1)[0] != mine
    ]


def add_gpu_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allow-busy-gpu",
        action="store_true",
        help="Load the model even if an evaluation or rollout appears to be running.",
    )


def require_idle_gpu(
    parser: argparse.ArgumentParser, args: argparse.Namespace, action: str
) -> None:
    """Abort with a clear message when another job holds the GPU, unless overridden."""
    if getattr(args, "allow_busy_gpu", False):
        return
    busy = gpu_users()
    if busy:
        listed = "\n  ".join(busy)
        parser.error(
            f"refusing {action}: another job is using the GPU:\n  {listed}\n"
            "Wait for it to finish, or pass --allow-busy-gpu."
        )
