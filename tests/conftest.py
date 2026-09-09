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


#: `default_window_path` as it is before the suite-wide redirect below, for the handful of tests
#: that are *about* where the window lives and must see the real function.
REAL_DEFAULT_WINDOW_PATH = runlock.default_window_path


@pytest.fixture
def unredirected_window_path(monkeypatch):
    """Give one test the real `default_window_path` back, for tests about the path itself."""
    monkeypatch.setattr(runlock, "default_window_path", REAL_DEFAULT_WINDOW_PATH)
    return REAL_DEFAULT_WINDOW_PATH


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
    # And the window, for the same reason and one the lock did not have: `_acquire` consults it
    # (issue 95), so a **real** window open on this machine failed three of these tests the first
    # time one was. Host state must not be able to fail the suite; a test that exercises the
    # window passes its own path.
    monkeypatch.setattr(
        runlock, "default_window_path", lambda: _suite_lock_path.parent / ".box-window.json"
    )
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

#: Modules the fork guard lets through. **Now none**, so the guard's contract is "no forks",
#: not "no *new* forks". The set was created by the issue-83 slice for the eight files that
#: started processes before ``local_llm_lab.spawn`` existed and still forked; they were
#: reported there rather than fixed, because seven untested edits next to the mechanism that
#: gates every model-loading run was the wrong trade, and a guard that fails 114 tests on its
#: first run is the shape of a guard people switch off.
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
#: if that is undone.
#:
#: **Empty, and it stays empty** (issue 84 closed). The other seven went through the helper in
#: three shapes, one per reason they had been exempted: ``git -C`` where a directory was passed
#: to git (``runlog``, ``integrity``); ``/bin/sh -c 'cd … && exec …'`` where a child genuinely
#: needs the project root, because ``mlx_lm``'s YAML resolves ``data:`` against it (``chat``,
#: ``train_sft``); and simply dropping a ``cwd`` that turned out to be decorative
#: (``tests/test_probes.py`` -- the repository root holds ``src/``, not the package, so the
#: child never imported through it). ``probes/guard.py`` did not migrate: its check was deleted.
#:
#: A new entry here is not a way to pass the rule.
#: ``test_the_fork_exemption_list_is_empty_and_stays_empty`` fails on any addition. If a file
#: needs to start a process, it goes through ``local_llm_lab.spawn``; if it cannot, that is a
#: finding about the file.
PRE_EXISTING_FORK_SITES: frozenset[str] = frozenset()

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


#: Test files whose import loads MLX, so Metal is up for the whole session once any of them is
#: collected. Pinned rather than computed, and the pin is checked against an AST import closure by
#: `tests/test_repository_rules.py`, which imports this tuple rather than keeping a second copy:
#: one list, two enforcers, the same shape as the fork-site sets above.
TESTS_THAT_LOAD_MLX = (
    "test_arch.py",
    "test_capture.py",
    "test_gated_delta_chunked.py",
    "test_gated_delta_chunkwise.py",
    "test_history_cache.py",
    "test_jlens.py",
    "test_jspace_sweep.py",
    "test_lens_native_integration.py",
    "test_lens_regression.py",
    "test_live_lens_native.py",
    "test_patch.py",
    "test_pipeline.py",
    "test_preflight.py",
    "test_probes.py",
    "test_state_swap.py",
    "test_tuner_data.py",
)


#: Test files that can reach MLX **at any scope**, including a function-local import that only
#: runs when the test does. A superset of the tuple above, and the right one for the collector:
#: `TESTS_THAT_LOAD_MLX` answers "what maps MLX on import", which is the question the closure
#: walker asks, and the collector's question is "what can map MLX at all". The five extra files
#: import mlx inside a function or a fixture and map it exactly as surely when that runs.
#:
#: Pinned like its sibling and checked against an any-scope AST walk by
#: `tests/test_repository_rules.py`, so a new function-local import cannot quietly rejoin the
#: set of files a window lets through.
#: Where the collector leaves what it did, for the terminal summary to report.
_WINDOW_SKIP = pytest.StashKey[object]()

TESTS_THAT_CAN_REACH_MLX = tuple(
    sorted(
        {
            *TESTS_THAT_LOAD_MLX,
            "test_adapter_delta.py",
            # Joined 2026-09-08 (issue 96): its new test reads `mlx_lm.tuner.trainer`'s source to
            # check the library still passes no seed, so this file reaches MLX at that scope.
            "test_cli.py",
            "test_cache_equivalence.py",
            "test_lens_jacobian.py",
            "test_metal_cache_limit.py",
            "test_runner.py",
            "test_state_probe.py",
        }
    )
)


def pytest_ignore_collect(collection_path, config):
    """On a box without MLX, the files that can reach it are left uncollected and named.

    The body lives in `runlock.ignore_when_mlx_absent`, like the window collector's, so the
    nested run in `tests/test_conftest_without_mlx.py` can import it and say "not installed".
    """
    return runlock.ignore_when_mlx_absent(collection_path.name, TESTS_THAT_CAN_REACH_MLX, config)


def pytest_collection_modifyitems(config, items):
    """Skip the MLX-loading files while another seat holds the box window (issue 95).

    Skipped rather than refused. A refusal makes the whole suite red for somebody who has done
    nothing wrong, and the natural response to a red suite is to run it again -- which is the
    collision. A skip keeps the rest of the suite honest, says whose slot this is, and leaves
    nothing to rerun.

    The holder's own suite is untouched: `blocking_window` returns None when the window's nonce
    matches the token the announcing command exported, and every child of that command inherits
    it.

    The body lives in `runlock` so this hook and the nested run that proves it fires are one
    implementation rather than two that agree today.
    """
    from local_llm_lab.runlock import mark_items_for_a_foreign_window

    config.stash[_WINDOW_SKIP] = mark_items_for_a_foreign_window(items, TESTS_THAT_CAN_REACH_MLX)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    runlock.report_mlx_absent(terminalreporter, config)
    """Say, where a reader will meet it, that a window skipped part of this run.

    A skip is the right response to somebody else's window and a silent one is its own hazard: on
    2026-09-08 a reviewer read "exit 0" from a run whose whole `test_cli.py` the collector had
    skipped, and reported the file green. The exit code was honest and the summary line was there;
    what was missing was anything that said the omission had a *cause*.
    """
    skipped = config.stash.get(_WINDOW_SKIP, None)
    if not skipped:
        return
    window, count = skipped
    if not count:
        # A window was open and this run collected nothing it covers. Saying so would be noise
        # on every unrelated run made during somebody's block.
        return
    terminalreporter.write_sep("=", "box window", yellow=True)
    terminalreporter.write_line(
        f"{count} test(s) were skipped, not run: the box window is held by {window.describe()}."
    )
    terminalreporter.write_line(
        "This run did NOT exercise the files that can reach MLX. Re-run after the end line "
        "before reporting them green."
    )
