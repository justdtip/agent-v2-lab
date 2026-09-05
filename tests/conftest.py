"""Suite-wide guards.

Two things live here, both from issue 83.

**The model-run lock's blast radius.** Several tests drive ``evaluate.load_policy`` and
``prefer.run_prefer`` against a faked ``mlx_lm``, and both now reach the lock on the way to
weights, so without a redirect the suite takes the lock that gates every model-loading run on
this machine — refusing its own next test, and leaving a lock file behind that blocks the next
real run. That is the exact failure the lock exists to prevent, arriving from the one process
nobody thinks of as a launcher.

**Forking.** ``pytest`` imports MLX during collection, which initialises Metal, and a ``fork``
in an interpreter where Metal is up aborts the process in libplatform (see
``local_llm_lab.spawn`` for the crash report and the mechanism). The abort looks like a random
kill in the middle of an unrelated test, so the guard below turns it into a named exception on
the line that caused it. Forking is not dangerous in itself — it is dangerous *here*, because
of what collection has already loaded into this interpreter.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from local_llm_lab import runlock

#: Captured at import, before any fixture can redirect it: where the lock really lives.
MACHINE_LOCK_PATH = runlock.default_lock_path()

#: The unstubbed process probes, likewise captured before the redirect below replaces them, so
#: that the one test which has to see the real ``ps``/``lsof`` argument lists can reach them.
REAL_PROCESS_TABLE = runlock._process_table
REAL_MAPPED_PIDS = runlock._mapped_pids


@pytest.fixture
def machine_lock_path() -> Path:
    """The real ``outputs/.model-run.lock``, past the redirect below."""
    return MACHINE_LOCK_PATH


@pytest.fixture(scope="session")
def _suite_lock_path(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("model-run-lock") / ".model-run.lock"


@pytest.fixture(autouse=True)
def _the_model_run_lock_is_never_the_machines(_suite_lock_path, monkeypatch):
    """Point every default acquire at a temporary path, and give it back after each test.

    The two process-check probes are stubbed empty for the same reason and one more: host state
    must not be able to fail the suite -- an unrelated model-loading process on this machine is
    a fact about the host, not about the code -- and the suite must not shell out to ``lsof``
    and ``ps`` once per test. Tests that exercise the check inject their own tables or run real
    child processes against a library of their own.
    """
    monkeypatch.setattr(runlock, "default_lock_path", lambda: _suite_lock_path)
    # Explicit no-op lambdas, not the `list`/`dict` builtins. `running_model_processes` calls
    # `_mapped_pids(library)` positionally, so the `dict` builtin evaluated
    # `dict("libmlx.dylib")` and raised ValueError from inside the guard. It reached the six
    # tests that drive `load_policy` and `run_prefer` against a faked `mlx_lm`, and nothing
    # else, because every test that names the lock explicitly passes `check_processes=False`.
    # `test_the_default_check_runs_under_these_stubs` now holds this signature down.
    monkeypatch.setattr(runlock, "_process_table", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(runlock, "_mapped_pids", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runlock, "_held", None)
    yield
    runlock._release_held()


def check_and_reclaim_machine_lock(path: Path) -> str | None:
    """Report a lock that appeared during the suite, removing it only when it is ours.

    The obvious cleanup -- unlink whatever is there -- is wrong, and wrong in exactly the way
    `runlock` refuses to be one file away. A real run started while the suite is running holds
    that lock with a model resident; deleting it hands the machine to the next launcher while
    16 GiB is still in use. So ownership is decided by the recorded pid: our own lock is a suite
    bug and is cleaned up, and anyone else's is reported and left exactly where it is.

    Returns the failure message, or ``None`` when there is nothing to report. Separated from the
    fixture so the decision can be exercised directly -- see
    `test_the_suite_never_removes_a_lock_it_does_not_own`.
    """
    held = runlock.read_lock(path)
    if held is None:
        return None
    if held.pid == os.getpid():
        path.unlink(missing_ok=True)
        return f"the suite took the machine's model-run lock:\n{held.describe()}"
    return (
        "a model-run lock appeared while this suite was running and it is NOT this process's, "
        "so it has been left alone -- a real run may be holding it with a model resident:\n"
        + held.describe()
    )


@pytest.fixture(scope="session", autouse=True)
def _the_suite_leaves_no_lock_on_the_machine():
    """Fail loudly if a test took the real lock; never remove one this process does not own."""
    existed = MACHINE_LOCK_PATH.exists()
    yield
    if existed:
        return
    problem = check_and_reclaim_machine_lock(MACHINE_LOCK_PATH)
    if problem is not None:
        pytest.fail(problem)


# ------------------------------------------------------------------------------- forking


class ForkInThisInterpreter(RuntimeError):
    """A fork was attempted in a process where Metal is up. That aborts, so refuse instead."""


#: The private CPython names the guard stands on. ``subprocess.Popen._execute_child`` looks
#: ``_fork_exec`` up as a module global, so replacing the module attribute is enough — but a
#: private name is a name that can move, and a guard whose hook has quietly stopped existing is
#: dead code that still reports success. ``tests/test_spawn.py`` fails when either name goes.
FORK_HOOKS = (("subprocess", "_fork_exec"), ("os", "fork"))

#: Modules that started processes before ``local_llm_lab.spawn`` existed, and still fork today.
#: They are **reported, not fixed** in the issue-83 slice (each needs its own argument change,
#: its own callers checked, and putting seven untested edits next to the mechanism that gates
#: every model-loading run is the wrong trade), so the guard lets them through and refuses
#: everything else. Its contract is therefore "no *new* forks", not "no forks".
#:
#: This is the same list ``test_repository_rules.py`` exempts from the static rule, imported
#: from here so the two cannot drift, and
#: ``test_the_spawn_exemptions_are_all_still_load_bearing`` fails when an entry goes stale.
#:
#: ``provenance.py`` **has left this list** (issue 84 item 1), and the ordering is what decided
#: it: ``write_provenance`` runs *after* ``load_policy`` at the end of every probe and every
#: ``cli`` stage, so it was the one site that forked with a model already resident and Metal
#: fully up. An abort there is ``SIGABRT``, which means ``atexit`` never runs and the lock is
#: orphaned -- the failure this whole slice exists to prevent, arriving from the provenance
#: write at the end of a *successful* run. It now goes through ``spawn.run`` with the repository
#: root in argv as ``git -C``, and ``test_a_provenance_write_after_a_load_does_not_fork`` fails
#: if that is undone. The seven that remain are issue 84 items 2 and 3; ``runlog.git_commit``,
#: ``integrity.git_tree_dirty`` and ``probes/guard.py`` run before the load, and this list is
#: empty when 84 closes.
PRE_EXISTING_FORK_SITES = frozenset(
    {
        "src/local_llm_lab/chat.py",
        "src/local_llm_lab/check_env.py",
        "src/local_llm_lab/pipeline/integrity.py",
        "src/local_llm_lab/probes/guard.py",
        "src/local_llm_lab/runlog.py",
        "src/local_llm_lab/train_sft.py",
        "tests/test_probes.py",
    }
)

#: Paths the *static* rule exempts but the guard does not: the helper itself, and the test that
#: proves the guard fires, which has to fork on purpose.
SPAWN_SANCTIONED_PATHS = frozenset({"src/local_llm_lab/spawn.py", "tests/test_spawn.py"})

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SUBPROCESS_SOURCE = subprocess.__file__


def _forking_caller() -> str | None:
    """The repository-relative file that asked for a child, past ``subprocess``'s own frames.

    ``subprocess.run`` reaches ``_fork_exec`` through three of its own frames, so the first
    frame outside ``subprocess.py`` and this file is the code that actually made the call.
    A caller outside the repository (a library) is not exempt and returns ``None``.
    """
    frame = sys._getframe(1)
    while frame is not None:
        filename = frame.f_code.co_filename
        if filename not in (_SUBPROCESS_SOURCE, __file__):
            try:
                return Path(filename).resolve().relative_to(_REPOSITORY_ROOT).as_posix()
            except ValueError:
                return None
        frame = frame.f_back
    return None


def _fork_refusal(real):
    """Wrap one fork primitive so a new caller is refused and a listed one is let through."""

    def refuse(*args, **kwargs):
        caller = _forking_caller()
        if caller in PRE_EXISTING_FORK_SITES:
            return real(*args, **kwargs)
        raise ForkInThisInterpreter(
            f"{caller or 'a library'} started a child by forking. pytest imports MLX at "
            "collection, so Metal is up, and forking here aborts the interpreter in "
            "libplatform rather than raising (see local_llm_lab.spawn). Start the child with "
            "local_llm_lab.spawn.run or spawn.popen, which reach posix_spawn instead."
        )

    return refuse


@pytest.fixture(scope="session", autouse=True)
def _forking_is_refused_rather_than_fatal():
    """Turn a *new* fork anywhere in the suite into an exception on the line that caused it."""
    with pytest.MonkeyPatch.context() as patch:
        for module, attribute in FORK_HOOKS:
            target = sys.modules[module]
            patch.setattr(target, attribute, _fork_refusal(getattr(target, attribute)))
        yield
