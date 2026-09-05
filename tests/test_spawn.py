"""How this repository starts child processes, and the guard that keeps it that way.

Forking is not dangerous in itself. It is dangerous *in this interpreter*: pytest imports MLX
during collection, which brings Metal up, and a ``fork`` there aborts the process in libplatform
when the parent's fork handler meets a lock the Metal driver's memory-pool-decay thread was
holding (see ``local_llm_lab.spawn`` for the crash report). The abort is deterministic given the
conditions, arrives as a kill in the middle of an unrelated test, and cannot be retried past.

Two things therefore have to hold, and both are tested here rather than assumed:

* ``local_llm_lab.spawn`` reaches ``posix_spawn`` for the arguments it accepts, and refuses the
  arguments that would silently drop back to ``fork``;
* the suite's fork guard in ``conftest.py`` actually fires. It stands on private CPython names,
  and a private name that moves turns a guard into dead code that still reports success — the
  same shape as a counting test passing after the property it counted was gone. So the guard is
  made to fire here, deliberately, and the names it patches are checked for still being the
  ones CPython uses.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

# `tests/` is not a package, so pytest puts it on `sys.path` and the suite's own conftest
# imports as a top-level module.
from conftest import FORK_HOOKS, PRE_EXISTING_FORK_SITES, ForkInThisInterpreter

from local_llm_lab import runlog, spawn
from local_llm_lab.project import PROJECT_ROOT

# --------------------------------------------------------------------------- the guard fires


def test_the_fork_guard_fires_on_a_forking_subprocess_call() -> None:
    """A guard nobody has watched fail is not a guard.

    A bare program name has no directory component, which is one of CPython's conditions for
    ``posix_spawn``, so this call really does take the fork path — and under the guard it
    raises on the line that caused it instead of aborting the interpreter three tests later.
    """
    with pytest.raises(ForkInThisInterpreter, match="spawn.run"):
        subprocess.run(["ps"], capture_output=True, check=False)


def test_the_fork_guard_fires_on_close_fds_which_is_the_default() -> None:
    """The trap: an absolute path is not enough, because ``close_fds`` defaults to True."""
    with pytest.raises(ForkInThisInterpreter):
        subprocess.run(["/bin/ps"], capture_output=True, check=False, close_fds=True)


def test_the_fork_guard_fires_on_a_direct_os_fork() -> None:
    """``subprocess`` is not the only way to fork; ``multiprocessing`` reaches ``os.fork`` too."""
    with pytest.raises(ForkInThisInterpreter):
        os.fork()


def test_the_fork_guard_lets_a_listed_site_through() -> None:
    """The other half of the contract: the guard refuses *new* forks, not all of them.

    `runlog.git_commit` shells out with a bare `git` and therefore forks. It is on
    `PRE_EXISTING_FORK_SITES` — reported in the issue-83 hand-off rather than fixed in that
    slice — so the guard hands the call to the real primitive instead of refusing it. Without
    this half the guard failed 114 tests the first time it was run against the whole suite,
    which is the shape of a guard that gets switched off rather than fixed.

    `git_commit` swallows `OSError`, `ValueError` and `SubprocessError` and answers "unknown",
    but `ForkInThisInterpreter` is a `RuntimeError` and would propagate — so a real commit hash
    here means the fork genuinely happened.
    """
    assert "src/local_llm_lab/runlog.py" in PRE_EXISTING_FORK_SITES

    commit = runlog.git_commit(PROJECT_ROOT)

    assert len(commit) == 40 and commit != "unknown"


def test_the_fork_guard_hooks_are_still_the_names_cpython_uses() -> None:
    """If a CPython release moves either private name, fail here rather than go quiet.

    Checking the attribute exists is not enough: ``subprocess._fork_exec`` has to be the global
    that ``Popen._execute_child`` actually looks up, or patching the module attribute guards
    nothing. ``co_names`` says which globals that function reads.
    """
    for module_name, attribute in FORK_HOOKS:
        module = sys.modules[module_name]
        assert hasattr(module, attribute), f"{module_name}.{attribute} has moved"

    read_by_execute_child = subprocess.Popen._execute_child.__code__.co_names
    assert "_fork_exec" in read_by_execute_child
    # The other half of the same branch: if this goes, `posix_spawn` is no longer the escape.
    assert "_posix_spawn" in read_by_execute_child


def test_posix_spawn_is_available_and_still_needs_close_fds_false() -> None:
    """The conditions `spawn` is built around, asserted rather than remembered.

    When a future macOS/CPython gains ``POSIX_SPAWN_CLOSEFROM`` this test fails, and that is the
    signal that ``close_fds=False`` — and its cost, an inherited descriptor table — can be
    dropped rather than carried forever as a flag nobody dares touch.
    """
    assert subprocess._USE_POSIX_SPAWN
    assert not subprocess._HAVE_POSIX_SPAWN_CLOSEFROM


# ------------------------------------------------------------------------------ the helper


def test_the_spawn_helper_never_forks() -> None:
    """The end-to-end proof, and it works only because the guard above is armed.

    ``ps`` is resolved to an absolute path and ``close_fds`` is forced False, so this reaches
    ``posix_spawn``. Undo either and the guard turns this into a failure rather than letting a
    forking helper ship.
    """
    completed = spawn.run(["ps", "-o", "pid="], capture_output=True, text=True, timeout=10)

    assert completed.returncode == 0
    assert completed.stdout.strip()


def test_a_long_lived_child_also_reaches_posix_spawn() -> None:
    """``popen`` is the form the lock fixtures use, so it carries the same guarantee."""
    child = spawn.popen([sys.executable, "-c", "pass"])
    try:
        assert child.wait(timeout=30) == 0
    finally:
        if child.poll() is None:  # pragma: no cover - only on a failed run
            child.kill()
            child.wait(timeout=30)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("cwd", "/tmp"),
        ("preexec_fn", os.setsid),
        ("start_new_session", True),
        ("process_group", 0),
        ("close_fds", True),
        ("shell", True),
        ("pass_fds", (3,)),
        ("executable", "/bin/ps"),
    ],
)
def test_the_helper_refuses_every_argument_that_would_fork(keyword: str, value: object) -> None:
    """Each of these silently returns the call to the fork path, so each is refused loudly."""
    with pytest.raises(spawn.UnsafeSpawnError, match=keyword):
        spawn.run(["/bin/ps"], **{keyword: value})


def test_a_refusal_says_how_to_get_the_same_effect_without_forking() -> None:
    """A refusal that only says no gets worked around; this one names the alternative."""
    with pytest.raises(spawn.UnsafeSpawnError) as error:
        spawn.popen([sys.executable, "-c", "pass"], start_new_session=True)

    assert "os.setsid()" in str(error.value)


def test_a_bare_program_name_is_resolved_to_an_absolute_path() -> None:
    """The resolution the shell would do, done where the result can be checked."""
    resolved = spawn.resolve_program("ps")

    assert os.path.isabs(resolved) and os.path.basename(resolved) == "ps"
    assert spawn.resolve_program("/bin/ps") == "/bin/ps"


def test_a_program_that_is_not_on_the_path_is_an_error_not_a_fork() -> None:
    with pytest.raises(spawn.UnsafeSpawnError, match="not on PATH"):
        spawn.run(["definitely-not-a-real-program-93f1a"])


def test_an_empty_argv_is_refused() -> None:
    with pytest.raises(spawn.UnsafeSpawnError, match="must name a program"):
        spawn.run([])
