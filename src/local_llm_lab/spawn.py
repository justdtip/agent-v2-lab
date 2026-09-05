"""Start child processes without forking, because forking here kills the interpreter.

The failure this exists to prevent, observed on this machine on 2026-09-05 (macOS crash report
``Python-2026-09-05-202845.ips``): ``pytest`` brings up Metal during collection, a later
``subprocess.Popen`` took CPython's ``fork`` path, and the parent's fork handler tried to unlock
a ``malloc`` lock held by the Metal driver's memory-pool-decay thread. libplatform aborted the
process with "BUG IN CLIENT OF LIBPLATFORM: Unlock of an os_unfair_lock not owned by current
thread", the crashing queue being ``com.apple.AGXMetal.MemoryPoolDecay`` inside
``AGX::PooledAllocator::shrink``.

**Forking is not dangerous in itself. Forking is dangerous in an interpreter where Metal has
been initialised** — which in this repository is any process that has touched MLX, including
every model-loading run and the whole test suite, which imports MLX at collection. Between
``fork`` and ``exec`` only the forking thread survives, so any lock another thread was holding
at that instant is held forever, and the Metal driver holds such locks on its own schedule. The
window is invisible, the abort is immediate, and retrying does not help: given the conditions it
is deterministic, not intermittent.

``posix_spawn`` has no such window, and CPython 3.13 chooses it in ``Popen._execute_child`` only
when *every* one of these holds (``subprocess.py``, the ``_USE_POSIX_SPAWN`` branch):

* ``close_fds`` is false — macOS has no ``POSIX_SPAWN_CLOSEFROM``, so the default ``True``
  forces the fork path on this platform;
* the executable has a directory component, i.e. a bare ``ps`` or ``git`` forks;
* ``cwd`` is ``None``, ``preexec_fn`` is ``None``, ``pass_fds`` is empty;
* ``start_new_session`` is false and ``process_group`` is ``-1``;
* no ``user``/``group``/``extra_groups``/``umask`` override;
* every stdio descriptor CPython plumbs is either ``-1`` (inherited) or above 2. This one is
  not reachable from any argument here — the pipes ``capture_output`` and ``stdout=PIPE``
  create are allocated well above 2 in a normal process — but it belongs in the list, because
  the list claims to be complete and a process started with 0, 1 or 2 already closed would
  fall back to ``fork`` with nothing in the call to explain why.

``env``, ``capture_output``, ``text`` and ``timeout`` are all fine.

So this module resolves the program to an absolute path, sets ``close_fds=False``, and
**refuses** the arguments that would silently put the call back on the fork path rather than
accepting them and hoping. A refusal is a compile-time-ish error in a test; an accepted ``cwd``
is an abort in the middle of a 70-minute training run.

*The cost of ``close_fds=False``, stated here so it is not deleted later as a stray flag.* The
child inherits this process's open file descriptors instead of having them closed for it. That
is accepted because on macOS it is the only way to reach ``posix_spawn`` at all, and because the
children started here are our own short-lived utilities (``ps``, a fixture's dummy process) that
neither run untrusted code nor outlive the parent. It would not be acceptable for a long-lived
child, a child running code we did not write, or a child that must not see a descriptor holding
credentials — none of which occur in this repository. Do not "tidy" it to ``True``: that
reintroduces the abort above, and ``tests/test_spawn.py`` fails when it does.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from typing import Any

__all__ = ["UnsafeSpawnError", "popen", "resolve_program", "run"]


class UnsafeSpawnError(ValueError):
    """The call as written would have taken CPython's fork path."""


#: Arguments that force the fork path, each with the way to get the same effect without it.
_FORK_FORCING: dict[str, str] = {
    "close_fds": (
        "this helper fixes it at False; macOS has no POSIX_SPAWN_CLOSEFROM, so True forks"
    ),
    "cwd": "pass absolute paths in argv and env instead of changing the child's directory",
    "preexec_fn": "there is nothing safe to run between fork and exec in this interpreter",
    "start_new_session": "have the child call os.setsid() itself, after exec",
    "process_group": "have the child call os.setpgid(0, 0) itself, after exec",
    "pass_fds": "hand the child a path rather than an inherited descriptor",
    "shell": "put /bin/sh -c in argv, so the exec that actually happens is visible",
    "executable": "name the program in argv[0]; this helper resolves it to an absolute path",
    "user": "not supported: it forks",
    "group": "not supported: it forks",
    "extra_groups": "not supported: it forks",
    "umask": "not supported: it forks",
}


def resolve_program(program: str) -> str:
    """Return an absolute path for ``program``, because a bare name forks.

    ``shutil.which`` is the resolution the shell would do, done here where the result can be
    checked: a program that is not on the path is a clear error rather than a fork.
    """
    if "/" in program:
        return program
    found = shutil.which(program)
    if found is None:
        raise UnsafeSpawnError(
            f"{program!r} is not on PATH, so it has no absolute path and the call would fork"
        )
    return found


def _vetted(argv: Sequence[str], kwargs: dict[str, Any]) -> list[str]:
    """Refuse fork-forcing arguments and resolve ``argv[0]``."""
    offenders = [name for name in kwargs if name in _FORK_FORCING]
    if offenders:
        reasons = "; ".join(f"{name}: {_FORK_FORCING[name]}" for name in sorted(offenders))
        raise UnsafeSpawnError(
            f"{', '.join(sorted(offenders))} would put this call on CPython's fork path, "
            f"which aborts an interpreter that has initialised Metal. {reasons}"
        )
    if not argv:
        raise UnsafeSpawnError("argv must name a program")
    return [resolve_program(str(argv[0])), *(str(part) for part in argv[1:])]


def run(argv: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
    """``subprocess.run`` restricted to arguments that reach ``posix_spawn``.

    Every process started under ``src/`` and ``tests/`` goes through this function or
    ``popen``; ``tests/test_repository_rules.py`` fails on a direct ``subprocess`` call.
    """
    return subprocess.run(_vetted(argv, kwargs), close_fds=False, **kwargs)


def popen(argv: Sequence[str], **kwargs: Any) -> subprocess.Popen[Any]:
    """``subprocess.Popen`` restricted to arguments that reach ``posix_spawn``.

    For a child that has to outlive the call — a fixture's dummy process, a long-running
    utility. Same restrictions and the same reason as ``run``.
    """
    return subprocess.Popen(_vetted(argv, kwargs), close_fds=False, **kwargs)
