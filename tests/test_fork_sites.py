"""Issue 84 items 2 and 3: the last seven files that started a process by forking.

One place a reader can find how each was fixed and why it needed its own fix. Every entry in
``PRE_EXISTING_FORK_SITES`` was exempted for a concrete reason -- it passed ``cwd``, or it named a
bare program -- and CPython takes ``posix_spawn`` only when ``cwd`` is ``None`` and the program is
absolute. A fork in an interpreter that has initialised Metal aborts (R45), and an abort is
``SIGABRT``, so ``atexit`` never runs and the model-run lock is orphaned.

The conftest guard turns any fork in this interpreter into ``ForkInThisInterpreter`` on the line
that caused it, so the tests here that actually run the migrated code fail outright if a migration
is reverted. They do not need to assert on the exception.
"""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest

from local_llm_lab import chat, check_env, runlog, train_sft
from local_llm_lab.pipeline import integrity

_REPO_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------- the two git helpers: `-C`, not `cwd`


@pytest.mark.parametrize(
    ("build", "verb"),
    [(integrity._git_argv, "status"), (runlog._git_argv, "rev-parse")],
)
def test_the_git_helpers_put_the_directory_in_argv_rather_than_in_subprocess_cwd(
    build, verb: str
) -> None:
    """``git -C <dir>`` is the same request by a mechanism that does not fork.

    Asserted on the argv rather than on the result, because the result is identical either way --
    which is exactly why a reviewer cannot tell the two apart without looking here.
    """
    assert build(Path("/somewhere"), verb) == ["git", "-C", "/somewhere", verb]
    assert build(None, verb) == ["git", verb]


@pytest.mark.parametrize(
    ("call", "not_determined"),
    [(integrity.git_tree_dirty, None), (runlog.git_commit, "unknown")],
)
def test_the_git_helpers_keep_reporting_not_determined_rather_than_raising(
    call, not_determined, tmp_path: Path
) -> None:
    """The contract both helpers document: provenance never kills a run.

    ``git -C`` on a directory that is not a repository exits non-zero, which is the same branch
    the old ``cwd`` form took, so the migration changes the spawn path and nothing else.
    """
    assert call(tmp_path) == not_determined


def test_the_git_helpers_read_this_repository_through_the_spawn_helper() -> None:
    """Both helpers, against real git, under the fork guard.

    The shape tests above would pass against a helper that built a correct argv and then never
    ran. This one runs, so a migration that is reverted fails here rather than silently.
    """
    commit = runlog.git_commit(_REPO_ROOT)
    assert len(commit) == 40 and set(commit) <= set("0123456789abcdef")
    assert isinstance(integrity.git_tree_dirty(_REPO_ROOT), bool)


# ---------------------------------- the two mlx_lm launchers: a visible `cd` in argv, not `cwd`


@pytest.mark.parametrize("module", [chat, train_sft])
def test_a_project_root_child_carries_its_directory_as_a_visible_shell_exec(module) -> None:
    """``mlx_lm``'s YAML resolves ``data:`` and ``adapter_path:`` against the working directory.

    So unlike the git helpers, the directory here cannot be dropped and git has no ``-C`` to move
    it into. It moves into argv as ``/bin/sh -c 'cd … && exec …'``, the form ``spawn``'s docstring
    names: the exec that actually happens stays visible, and ``exec`` leaves no extra process.
    """
    argv = module._in_project_root(["mlx_lm.lora", "--config", "a b.yaml"])
    program, flag, script = argv

    assert (program, flag) == ("/bin/sh", "-c")
    assert len(argv) == 3, "anything after -c is an argument to the script, not part of it"
    assert script.startswith(f"cd {shlex.quote(str(module.PROJECT_ROOT))} && exec ")
    # Quoting is not decorative here: this repository's own root contains a space, so an
    # unquoted `cd` would run in a directory that does not exist and mlx_lm would read the
    # wrong dataset rather than fail.
    assert shlex.split(script)[1] == str(module.PROJECT_ROOT)
    assert shlex.split(script)[-1] == "a b.yaml"


# --------------------------------------------------------- the bare program name: `sysctl`


def test_memory_gib_reads_sysctl_through_the_helper() -> None:
    """``sysctl`` was exempted for naming a bare program; the helper resolves it to a path.

    Runs for real, so the resolution is exercised rather than described. The value is only
    sanity-checked: this is a test of the spawn path, not of the machine.
    """
    memory = check_env._memory_gib()
    assert memory is not None and 1 < memory < 4096


def test_memory_gib_still_reports_none_when_the_child_fails(monkeypatch) -> None:
    """Unchanged contract: ``check_env`` degrades to "not determined" rather than raising."""
    from local_llm_lab import spawn

    def refuse(*_args, **_kwargs):
        raise OSError("no sysctl here")

    monkeypatch.setattr(spawn, "run", refuse)
    assert check_env._memory_gib() is None
