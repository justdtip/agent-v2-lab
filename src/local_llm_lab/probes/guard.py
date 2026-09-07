"""Refuse to load a checkpoint while another process on this machine already holds MLX.

A pipeline evaluation and a probe both want the whole model resident; running them together
either thrashes unified memory or silently slows the evaluation whose numbers are being
collected. Every probe CLI that loads the real model checks first.

**The check is not this module's own.** It delegates to :func:`runlock.running_model_processes`,
which asks the kernel which processes have ``libmlx.dylib`` mapped. What used to be here was a
second, name-based definition of "a model is already loaded" -- ``pgrep -fl`` against a fixed
pattern of job names -- and issue 84 asked for it to be deleted rather than repaired. Three
reasons, all of them things that had already happened:

* **It missed real processes.** Detection by name sees only jobs whose command line matches the
  pattern. ``ctxmax.py`` held 16.6-17.2 GiB for twenty-three minutes on 2026-09-05 and was
  invisible to every name check on the machine (R45).
* **It was about to block itself.** The pattern excluded only the caller's own pid, so the first
  probe entry point whose command line contained one of those names would refuse to run because
  it found *itself*. That is the same self-match that cost eight minutes during the hosted-lens
  conversion, sitting latent here.
* **It forked.** ``pgrep`` through ``subprocess`` was safe only because it ran before the first
  MLX allocation -- a property of call order that nothing enforced and that any reordering broke
  silently (R45, issue 84 item 2).

Two definitions of one condition is how the disagreement between them recurs, so there is now one.
"""

from __future__ import annotations

import argparse

from local_llm_lab import runlock

__all__ = ["add_gpu_arguments", "gpu_users", "require_idle_gpu"]


def gpu_users() -> list[runlock.MappedProcess]:
    """Processes outside this one that hold MLX; empty when no model is resident.

    Kept as a name because the probe CLIs and their tests use it, but it is now a thin call to
    the lock's own check rather than a second implementation of it.
    """
    return runlock.running_model_processes()


def add_gpu_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allow-busy-gpu",
        action="store_true",
        help=(
            "Skip this CLI's early check for another process holding MLX. The model-run lock "
            "repeats the check at the load and has no override, so this only moves where a "
            "refusal happens."
        ),
    )


def require_idle_gpu(
    parser: argparse.ArgumentParser, args: argparse.Namespace, action: str
) -> None:
    """Abort with the lock's own refusal when another process holds MLX, unless overridden.

    The override skips only this early check. ``runlock.load_weights`` takes the lock and
    makes the same check again with no override, so ``--allow-busy-gpu`` moves a refusal
    from here to the load and buys nothing else; the refusal below does not offer it as a
    way past.
    """
    if getattr(args, "allow_busy_gpu", False):
        return
    busy = gpu_users()
    if busy:
        parser.error(
            f"{runlock.refusal_for_processes(busy)}\nThis command stops here rather than {action}."
        )
