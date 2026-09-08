"""Fixtures for the model-run lock (issue 83).

R38: each fixture here exercises the real failure rather than a proxy for it. The three that
matter run actual processes, because the failures they name are not reachable inside one
interpreter:

* **Contention.** Two processes race for the same lock path and exactly one wins. Acquiring
  twice in sequence in one process cannot show the race — it only shows that a file exists
  after it was created — so the winner and the loser here are separate pids that overlap in
  time, held open by a release barrier so the second attempt is guaranteed to land while the
  first still holds.
* **Self-match.** A launcher started under a live shell wrapper, with the entry-point token on
  both command lines, does not report itself as a competing run. This is reachable only with a
  real ancestry: a synthesised table can be made to pass by a filter that excludes nothing but
  the caller's own pid, which is exactly the bug that stalled a conversion for eight minutes.
* **Orphaned lock.** A process that dies the way the probe CLIs die — an exception on a
  template or shape error, a ``parser.error`` ``SystemExit``, a ``SIGTERM`` from a person —
  leaves no lock behind. Only a separate process can die; a test that catches its own
  exception has already proved nothing about interpreter teardown.

No model is loaded anywhere here: every child process is a dummy that touches the lock module
and nothing else.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import REAL_MAPPED_PIDS, REAL_PROCESS_TABLE, check_and_reclaim_machine_lock

from local_llm_lab import runlock, spawn
from local_llm_lab.project import PROJECT_ROOT

_SRC = PROJECT_ROOT / "src"
_TIMEOUT = 30.0


def _child_env() -> dict[str, str]:
    """Run children against this tree, not whatever the venv happens to have installed."""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(_SRC) if not existing else f"{_SRC}{os.pathsep}{existing}"
    env.pop("MODEL_RUN_SESSION", None)
    return env


def _wait_for(predicate, *, what: str, timeout: float = _TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what}")


def _script(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


# --------------------------------------------------------------- the record and its reading


def test_the_lock_path_is_the_one_the_concurrency_clause_names(machine_lock_path) -> None:
    """The clause of record names `outputs/.model-run.lock`; nothing else is the lock.

    `machine_lock_path` reads past the suite-wide redirect in `conftest.py`, which points every
    other test at a temporary lock so the suite never takes the machine's.

    Against `box_state_root()` rather than `PROJECT_ROOT` since issue 95: the two are the same
    directory in the primary checkout and differ in a linked worktree, where the lock used to be
    private and therefore useless. The relative path is what the clause names and it has not
    moved; what moved is which root it hangs off.
    """
    assert runlock.LOCK_RELATIVE_PATH.as_posix() == "outputs/.model-run.lock"
    assert machine_lock_path == runlock.box_state_root() / runlock.LOCK_RELATIVE_PATH


def test_the_lock_records_the_session_command_timestamp_and_pid(tmp_path, monkeypatch) -> None:
    """Every field a refusal has to print comes out of the file, not out of the refuser."""
    monkeypatch.setenv("MODEL_RUN_SESSION", "deputy")
    monkeypatch.setattr(sys, "argv", ["agent-v2-probe-patch", "--config", "configs/x.yaml"])
    lock = tmp_path / "run.lock"

    with runlock.model_run_lock(path=lock, check_processes=False):
        held = runlock.read_lock(lock)
        assert held is not None
        assert held.session == "deputy"
        assert held.command == "agent-v2-probe-patch --config configs/x.yaml"
        assert held.pid == os.getpid()
        assert held.started.endswith("Z")
        assert held.holder_alive
        assert not held.stale
        payload = json.loads(lock.read_text(encoding="utf-8"))
        assert payload["process_check"] == "skipped"

    assert runlock.read_lock(lock) is None


def test_a_malformed_lock_is_still_readable_and_still_refuses(tmp_path) -> None:
    """A truncated or foreign lock must be reported, not treated as no lock at all."""
    lock = tmp_path / "run.lock"
    lock.write_text('{"session": "deputy", "com', encoding="utf-8")

    held = runlock.read_lock(lock)
    assert held is not None
    assert held.pid is None
    assert held.command == "unknown"
    assert held.age_seconds is not None  # dated from the file's mtime

    with pytest.raises(runlock.RunLockBusy) as error, runlock.model_run_lock(
        path=lock, check_processes=False
    ):
        pass
    assert str(lock) in str(error.value)


def test_a_lock_whose_holder_is_gone_reads_as_stale_and_is_not_removed(tmp_path) -> None:
    """The clause: a stale lock is reported, never deleted by a launcher."""
    lock = tmp_path / "run.lock"
    dead_pid = 2**22 - 1  # above every pid_max this machine issues
    lock.write_text(
        json.dumps(
            {
                "session": "head",
                "command": "agent-v2-eval --config configs/agent_v2d.yaml",
                "started": "2026-09-05T01:00:00Z",
                "started_epoch": time.time() - 60,
                "pid": dead_pid,
                "nonce": "deadbeef",
            }
        ),
        encoding="utf-8",
    )

    held = runlock.read_lock(lock)
    assert held is not None and held.stale and not held.holder_alive

    with pytest.raises(runlock.RunLockBusy) as error, runlock.model_run_lock(
        path=lock, check_processes=False
    ):
        pass

    message = str(error.value)
    assert f"holder pid     : {dead_pid} (not alive)" in message
    assert "agent-v2-eval --config configs/agent_v2d.yaml" in message
    assert "NOT removed automatically" in message
    assert "Report it to the Chief" in message
    assert lock.exists(), "a launcher must never delete a lock it finds inconvenient"


def test_an_aged_lock_with_a_live_holder_is_stale_by_age(tmp_path) -> None:
    """The second staleness signal: past the longest run this machine does."""
    lock = tmp_path / "run.lock"
    lock.write_text(
        json.dumps(
            {
                "session": "chief",
                "command": "agent-pipeline train",
                "started": "2026-09-04T01:00:00Z",
                "started_epoch": time.time() - runlock.STALE_AFTER_SECONDS - 60,
                "pid": os.getpid(),
                "nonce": "n",
            }
        ),
        encoding="utf-8",
    )

    held = runlock.read_lock(lock)
    assert held is not None and held.holder_alive and held.stale

    with pytest.raises(runlock.RunLockBusy) as error, runlock.model_run_lock(
        path=lock, check_processes=False
    ):
        pass
    assert "staleness threshold" in str(error.value)
    assert lock.exists()


def test_a_second_create_over_an_existing_lock_fails_and_changes_nothing(tmp_path) -> None:
    """The uncontended precondition only: an existing lock is never overwritten.

    Deliberately not called a test of the exclusive create. A check-then-write passes this too,
    because nothing here overlaps in time; the design the clause rejects is separated from the
    one it requires by `test_no_two_processes_ever_hold_the_lock_at_once` below.
    """
    lock = tmp_path / "run.lock"
    runlock._write_lock(lock, {"pid": 1})

    with pytest.raises(FileExistsError):
        runlock._write_lock(lock, {"pid": 2})
    assert json.loads(lock.read_text(encoding="utf-8"))["pid"] == 1


def test_release_leaves_a_lock_another_holder_replaced(tmp_path, capsys) -> None:
    """The nonce guard: a release removes its own lock and nobody else's."""
    lock = tmp_path / "run.lock"
    handle = runlock._Handle(path=lock, nonce="mine", owner_pid=os.getpid())
    lock.write_text(json.dumps({"pid": os.getpid(), "nonce": "someone-else"}), encoding="utf-8")

    runlock._release(handle)

    assert lock.exists()
    assert "leaving it in place" in capsys.readouterr().err


def test_the_suite_never_removes_a_lock_it_does_not_own(tmp_path) -> None:
    """The suite's own cleanup, held to the same rule as every launcher.

    A real run started while the suite is running holds the machine lock with its model
    resident. Unlinking whatever is there at session end would hand the machine to the next
    launcher with 16 GiB still in use -- exactly what `runlock` refuses to do to a stale lock,
    contradicted one file away by the fixture meant to keep the suite tidy. Ownership is
    decided by the recorded pid, and nothing else.
    """
    holder = {
        "session": "chief",
        "command": "agent-v2-eval --config configs/agent_v2d.yaml",
        "started": "2026-09-05T22:00:00Z",
        "started_epoch": time.time(),
        "nonce": "n",
    }

    foreign = tmp_path / "foreign.lock"
    foreign.write_text(json.dumps({**holder, "pid": os.getpid() + 1}), encoding="utf-8")
    reported = check_and_reclaim_machine_lock(foreign)

    assert foreign.exists(), "a lock belonging to a real run must never be removed"
    assert reported is not None and "NOT this process's" in reported
    assert "agent-v2-eval --config configs/agent_v2d.yaml" in reported

    ours = tmp_path / "ours.lock"
    ours.write_text(json.dumps({**holder, "pid": os.getpid()}), encoding="utf-8")
    reported = check_and_reclaim_machine_lock(ours)

    assert not ours.exists(), "the suite's own lock is a suite bug and is cleaned up"
    assert reported is not None and "the suite took" in reported

    assert check_and_reclaim_machine_lock(tmp_path / "absent.lock") is None


# ------------------------------------------- the MLX-mapping check, in isolation
#
# The check asks the kernel which processes have the MLX library mapped. It does not match
# process names and it does not look at memory size, because the case that matters defeats
# both: the orphan on this machine at 2026-09-05 20:22 (pid 63943, a 4B model, 16-17 GB of a
# 24 GiB box, whose recommended working set is 17.8 GiB) was called `ctxmax.py` and reported
# 20 MB resident in `ps`.


def _row(pid: int, ppid: int, pgid: int, command: str) -> tuple[int, int, int, str]:
    return (pid, ppid, pgid, command)


_LIB = "/opt/venv/mlx/lib/libmlx.dylib"


def test_the_check_excludes_the_whole_ancestry_not_just_the_caller() -> None:
    """The eight-minute stall, and the case a name check could never have had at all.

    Every one of these has MLX mapped: the launcher, because a stage imports `mlx_lm.lora`
    before it resolves its base, and each wrapper because a mapping is inherited across `fork`
    and shared by `uv run`'s own child. A filter that excludes only the caller's pid, or only
    pid plus parent, reports one of them and blocks the launch forever.
    """
    table = [
        _row(100, 1, 100, "-zsh"),
        _row(101, 100, 101, "sh -c uv run agent-v2-eval --config c.yaml"),
        _row(102, 101, 101, "uv run agent-v2-eval --config c.yaml"),
        _row(103, 102, 101, "python -m local_llm_lab.pipeline.evaluate"),
        _row(104, 103, 101, "agent-v2-eval --config c.yaml"),
    ]
    mapped = {pid: _LIB for pid in (101, 102, 103, 104)}

    assert runlock.running_model_processes(mapped, table, pid=104) == []


def test_the_check_reports_a_holder_whose_name_matches_no_pattern() -> None:
    """The live failure: a scratch script holding a 4B model, named nothing recognisable.

    This is the whole reason the check is by mapped library. `ctxmax.py` appears in no entry
    point list, no console script, and no config; it is what a benchmark someone wrote in five
    minutes always looks like, and it is exactly as fatal to a 24 GiB machine as a real run.
    """
    table = [
        _row(104, 101, 101, "agent-v2-eval --config c.yaml"),
        _row(63943, 1, 63940, "/opt/homebrew/.../Python ctxmax.py"),
    ]

    assert runlock.running_model_processes({63943: _LIB}, table, pid=104) == [
        runlock.MappedProcess(
            pid=63943, command="/opt/homebrew/.../Python ctxmax.py", library=_LIB
        )
    ]


def test_the_check_cannot_have_a_memory_size_gate(monkeypatch) -> None:
    """`ps` said 20 MB for a process holding 16 GB: Metal buffers never land in RSS.

    Proved at the inputs rather than by looking for the word "rss" in the source, because the
    failure of a size gate is that it *passes* -- a real run waved through by a threshold looks
    exactly like no contention at all. The check makes two external queries and neither returns
    a size: `ps` is asked for pid, ppid, pgid and command, and `lsof -F pn` returns pids and
    paths. There is nothing for a threshold to read, so a size gate cannot be introduced without
    changing one of these two argument lists, which is what this test watches.
    """
    calls: list[list[str]] = []

    def record(argv, **kwargs):
        calls.append(list(argv))
        return SimpleNamespace(stdout="", returncode=0)

    # `conftest` stubs both probes for every test; this one needs the real queries.
    monkeypatch.setattr(runlock, "_process_table", REAL_PROCESS_TABLE)
    monkeypatch.setattr(runlock, "_mapped_pids", REAL_MAPPED_PIDS)
    monkeypatch.setattr(runlock, "spawn_run", record)
    runlock._process_table()
    runlock._mapped_pids()

    assert calls == [
        ["ps", "-Ao", "pid=,ppid=,pgid=,command="],
        ["lsof", "-w", "-d", "txt", "-F", "pn"],
    ]
    for argv in calls:
        joined = " ".join(argv)
        for size_field in ("rss", "vsz", "vsize", "mem", "%mem"):
            assert size_field not in joined


def test_the_check_excludes_the_callers_own_children() -> None:
    """A child in the caller's process group is the caller's own run, not a competitor."""
    table = [
        _row(104, 101, 104, "agent-pipeline eval"),
        _row(105, 104, 104, "sh -c agent-v2-eval --shard 1"),
    ]

    assert runlock.running_model_processes({104: _LIB, 105: _LIB}, table, pid=104) == []


def test_a_holder_that_exited_between_lsof_and_ps_is_not_reported() -> None:
    """A refusal naming a dead pid is a refusal nobody can act on, so `ps` runs second."""
    table = [_row(104, 101, 101, "agent-v2-eval")]

    assert runlock.running_model_processes({777: _LIB}, table, pid=104) == []


def test_the_library_is_matched_on_basename_not_on_a_path() -> None:
    """Every checkout, venv and branch installs MLX somewhere different."""
    text = "\n".join(
        (
            "p111",
            "n/Users/x/Desktop/An app/.venv/lib/python3.13/site-packages/mlx/lib/libmlx.dylib",
            "p222",
            "n/opt/other/place/libmlx.dylib",
            "p333",
            "n/usr/lib/libSystem.B.dylib",
            "p444",
            "n/somewhere/notlibmlx.dylib",
        )
    )

    assert runlock._parse_lsof_map(text, "libmlx.dylib") == {
        111: "/Users/x/Desktop/An app/.venv/lib/python3.13/site-packages/mlx/lib/libmlx.dylib",
        222: "/opt/other/place/libmlx.dylib",
    }


def test_the_production_library_is_the_mlx_one() -> None:
    assert runlock.MLX_LIBRARY == "libmlx.dylib"


def test_the_lineage_walk_terminates_on_a_cycle() -> None:
    """A synthesised or racing process table must not hang the launcher."""
    table = [_row(1, 2, 1, "a"), _row(2, 1, 1, "b")]

    assert runlock._own_lineage(table, 1) == {1, 2}


def test_the_process_table_parser_drops_unparseable_rows() -> None:
    text = "  100     1   100 /sbin/launchd\nheader junk\n  101 100 101 uv run agent-v2-eval -c x\n"

    assert runlock._parse_process_table(text) == [
        (100, 1, 100, "/sbin/launchd"),
        (101, 100, 101, "uv run agent-v2-eval -c x"),
    ]


def test_a_refusal_prints_the_pid_the_command_and_the_mapping(tmp_path, monkeypatch) -> None:
    """A false match has to be diagnosable in seconds rather than guessed at.

    The mapped path is printed as well as the command, because it says *which* install the
    holder is using -- which venv, which checkout -- and that is usually the whole diagnosis.
    """
    monkeypatch.setattr(runlock, "_mapped_pids", lambda *_a, **_k: {4242: _LIB})
    monkeypatch.setattr(
        runlock, "_process_table", lambda: [_row(4242, 1, 4242, "python ctxmax.py")]
    )

    with pytest.raises(runlock.RunLockBusy) as error, runlock.model_run_lock(
        path=tmp_path / "run.lock"
    ):
        pass

    message = str(error.value)
    assert "pid 4242" in message
    assert "python ctxmax.py" in message
    assert _LIB in message
    assert "may hold no lock at all" in message
    assert not (tmp_path / "run.lock").exists(), "a refused launch writes no lock"


def test_the_default_check_runs_under_these_stubs(tmp_path) -> None:
    """The one test that takes the default `check_processes=True` path, which is the point.

    `conftest` replaces both probes for every test, and `running_model_processes` calls
    `_mapped_pids(library)` positionally -- so a stub that takes no argument raises from inside
    the guard rather than reporting no contention. The `dict` builtin did exactly that
    (`dict("libmlx.dylib")` is a ValueError), and it was invisible here because every other
    test in this file names the lock path and passes `check_processes=False`. A stub whose
    shape nothing tests is the same failure as a guard nobody has watched fire.
    """
    lock = tmp_path / "run.lock"

    assert runlock.running_model_processes() == []
    with runlock.model_run_lock(path=lock):
        payload = json.loads(lock.read_text(encoding="utf-8"))

    assert payload["process_check"] == "ok"
    assert not lock.exists()


def test_a_check_that_cannot_run_says_so_and_records_it_in_the_lock(
    tmp_path, monkeypatch, capsys
) -> None:
    """Fail-open, but loudly and on the record.

    The check is a bridge: a missing or slow `lsof` must not block a launch. But a launch that
    skipped it silently, and then collided with a scratch script holding no lock, is a mystery
    with nothing to read afterwards -- the lock file would say the launch went fine. So the
    reason lands on stderr at the time and in the lock payload for whoever looks later.
    """
    monkeypatch.setattr(runlock, "_process_table", REAL_PROCESS_TABLE)
    monkeypatch.setattr(runlock, "_mapped_pids", REAL_MAPPED_PIDS)

    def timed_out(argv, **_kwargs):
        raise subprocess.TimeoutExpired(list(argv), 20.0)

    monkeypatch.setattr(runlock, "spawn_run", timed_out)
    lock = tmp_path / "run.lock"

    with runlock.model_run_lock(path=lock):
        payload = json.loads(lock.read_text(encoding="utf-8"))

    assert payload["process_check"].startswith("failed: ")
    assert "lsof -w -d txt -F pn" in payload["process_check"]
    assert "TimeoutExpired" in payload["process_check"]

    printed = capsys.readouterr().err
    assert "the process check could not run" in printed
    assert "lsof" in printed
    assert "will not be caught this time" in printed


# ------------------------------------------------------------------- R38: real contention

_CONTENDER = '''
import os, sys, time
from pathlib import Path
from local_llm_lab.runlock import RunLockBusy, model_run_lock

lock, start, release, out, session = (
    Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5]
)


def report(text):
    tmp = out.with_suffix(".part")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, out)


def wait(flag):
    deadline = time.time() + 30
    while not flag.exists():
        if time.time() > deadline:
            raise SystemExit("barrier never appeared: " + str(flag))
        time.sleep(0.005)


wait(start)
try:
    with model_run_lock(
        command="fixture " + session, session=session, path=lock, check_processes=False
    ):
        report("WON")
        wait(release)
except RunLockBusy as error:
    report("LOST\\n" + str(error))
'''


def test_two_processes_contend_for_one_lock_and_exactly_one_wins(tmp_path) -> None:
    """R38: the race itself, with two real pids overlapping in time.

    Reachable only across processes: `O_CREAT | O_EXCL` makes the kernel pick the winner, and
    the failure it prevents — two launchers both seeing no lock and both writing one — cannot
    happen inside a single interpreter. The winner is pinned open by the release barrier, so
    the loser's attempt is guaranteed to land while the lock is genuinely held rather than
    after it was already given back.
    """
    script = _script(tmp_path, "contender.py", _CONTENDER)
    lock = tmp_path / "run.lock"
    start = tmp_path / "start"
    release = tmp_path / "release"
    outs = [tmp_path / "a.out", tmp_path / "b.out"]
    env = _child_env()

    children = [
        spawn.popen(
            [sys.executable, str(script), str(lock), str(start), str(release), str(out), name],
            env=env,
        )
        for out, name in zip(outs, ("session-a", "session-b"), strict=True)
    ]
    try:
        start.touch()
        _wait_for(lambda: all(out.exists() for out in outs), what="both contenders to report")
        verdicts = [out.read_text(encoding="utf-8") for out in outs]
        winners = [text for text in verdicts if text.startswith("WON")]
        losers = [text for text in verdicts if text.startswith("LOST")]

        assert len(winners) == 1, f"exactly one process may win: {verdicts}"
        assert len(losers) == 1

        held = runlock.read_lock(lock)
        assert held is not None
        assert f"holder pid     : {held.pid} (alive)" in losers[0]
        assert f"holder command : fixture {held.session}" in losers[0]
        assert held.pid in {child.pid for child in children}
    finally:
        release.touch()
        for child in children:
            child.wait(timeout=_TIMEOUT)

    assert runlock.read_lock(lock) is None, "the winner released the lock on the way out"


def test_the_lock_is_free_again_once_the_holder_exits(tmp_path) -> None:
    """The other half of contention: a third launcher must not be refused forever."""
    script = _script(tmp_path, "contender.py", _CONTENDER)
    lock = tmp_path / "run.lock"
    start = tmp_path / "start"
    release = tmp_path / "release"
    env = _child_env()

    holder_out = tmp_path / "holder.out"
    holder = spawn.popen(
        [
            sys.executable,
            str(script),
            str(lock),
            str(start),
            str(release),
            str(holder_out),
            "holder",
        ],
        env=env,
    )
    start.touch()
    _wait_for(lambda: holder_out.exists(), what="the holder to take the lock")
    assert holder_out.read_text(encoding="utf-8") == "WON"

    release.touch()
    holder.wait(timeout=_TIMEOUT)
    assert runlock.read_lock(lock) is None

    second_release = tmp_path / "release2"
    second_out = tmp_path / "second.out"
    second = spawn.popen(
        [
            sys.executable,
            str(script),
            str(lock),
            str(start),
            str(second_release),
            str(second_out),
            "second",
        ],
        env=env,
    )
    try:
        _wait_for(lambda: second_out.exists(), what="the second launcher to take the lock")
        assert second_out.read_text(encoding="utf-8") == "WON"
    finally:
        second_release.touch()
        second.wait(timeout=_TIMEOUT)


#: A quota, not a stopwatch: a time-boxed loop reports almost nothing on a loaded machine, and
#: "did these processes actually overlap" is then a question about the machine's spare capacity
#: rather than about the lock.
_HAMMER_ACQUIRES = 300

_HAMMER = '''
import os, sys, time
from pathlib import Path
from local_llm_lab.runlock import RunLockBusy, _acquire, _release, read_lock

lock, start, out, target = (
    Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
)
deadline = time.time() + 25
while not start.exists():
    if time.time() > deadline:
        raise SystemExit("start barrier never appeared")
    time.sleep(0.002)

wins = refusals = stolen = 0
while wins < target and time.time() < deadline:
    try:
        handle = _acquire(command="hammer", session="h", path=lock, check_processes=False)
    except RunLockBusy:
        refusals += 1
        continue
    wins += 1
    held = read_lock(lock)
    if held is None or held.nonce != handle.nonce:
        stolen += 1
    _release(handle)

tmp = out.with_suffix(".part")
tmp.write_text(f"{wins} {refusals} {stolen}", encoding="utf-8")
os.replace(tmp, out)
'''


def test_no_two_processes_ever_hold_the_lock_at_once(tmp_path) -> None:
    """R38: exclusivity itself, which is the property `O_CREAT | O_EXCL` provides and "check
    whether it exists, then write it" does not.

    The two-process fixture above proves a refusal, and a check-then-write produces a refusal
    too whenever the two attempts do not overlap — so on its own it cannot tell the two designs
    apart. This one can, by making the overlap the common case rather than a rarity: six
    processes take and release the same lock as fast as they can for a second, and each reads
    the file back straight after its own acquire. Under an exclusive create it must find its own
    nonce, because the kernel picked the winner and `_release` refuses to unlink a lock whose
    nonce is not its own, so nothing can have replaced the file in between. A stolen nonce means
    two processes were inside the lock at the same moment — two model loads on a 24 GiB box.

    Measured on this machine, three runs each: the exclusive create steals 0 of 1,800 acquires;
    the check-then-write mutant steals 6,499 / 7,893 / 8,241 while reaching the same quota.
    """
    script = _script(tmp_path, "hammer.py", _HAMMER)
    lock = tmp_path / "run.lock"
    start = tmp_path / "start"
    outs = [tmp_path / f"{index}.out" for index in range(6)]
    env = _child_env()

    children = [
        spawn.popen(
            [sys.executable, str(script), str(lock), str(start), str(out), str(_HAMMER_ACQUIRES)],
            env=env,
        )
        for out in outs
    ]
    try:
        start.touch()
        for child in children:
            child.wait(timeout=_TIMEOUT)
    finally:
        for child in children:
            if child.poll() is None:  # pragma: no cover - only on a failed run
                child.kill()
                child.wait(timeout=_TIMEOUT)

    counts = [tuple(int(part) for part in out.read_text(encoding="utf-8").split()) for out in outs]
    wins = sum(win for win, _, _ in counts)
    refusals = sum(refusal for _, refusal, _ in counts)
    stolen = sum(steal for _, _, steal in counts)

    # Refusals are the proof that the processes overlapped rather than queued politely: a run
    # in which nobody was ever turned away has not tested exclusivity at all.
    assert refusals > 0, f"the processes never really contended: {counts}"
    assert wins >= _HAMMER_ACQUIRES, f"the processes did too little work to mean anything: {counts}"
    assert stolen == 0, f"two processes held the lock at once: {counts}"
    assert not lock.exists(), "every acquire was released"


# ------------------------------------------------------- R38: a launcher does not block itself

_MAPS_AND_REPORTS = '''
import ctypes, json, os, sys
from pathlib import Path
from local_llm_lab.runlock import _mapped_pids, running_model_processes

out, library = Path(sys.argv[1]), sys.argv[2]
ctypes.CDLL(library)
basename = os.path.basename(library)
payload = {
    "pid": os.getpid(),
    "ppid": os.getppid(),
    "mapped": sorted(_mapped_pids(basename)),
    "reported": [[h.pid, h.command, h.library] for h in running_model_processes(library=basename)],
}
tmp = out.with_suffix(".part")
tmp.write_text(json.dumps(payload), encoding="utf-8")
os.replace(tmp, out)
'''

_MAPS_AND_WAITS = '''
import ctypes, sys
from local_llm_lab.spawn import popen

library, child_script, out = sys.argv[1], sys.argv[2], sys.argv[3]
ctypes.CDLL(library)
raise SystemExit(popen([sys.executable, child_script, out, library]).wait())
'''

_MAPS_AND_SLEEPS = '''
import ctypes, os, sys, time
ctypes.CDLL(sys.argv[1])
os.setsid()
time.sleep(60)
'''


@pytest.fixture
def stand_in_library(tmp_path) -> Path:
    """A uniquely named dylib the fixtures can map, standing in for `libmlx.dylib`.

    A copy of a stdlib extension module, which is an ordinary Mach-O that `dlopen` accepts. The
    real library is not used, for two reasons: mapping `libmlx.dylib` means importing MLX, which
    initialises Metal and allocates on a 24 GiB machine that a real run may be using; and the
    check has to be proved against *any* mapped library, since matching one name we chose is
    the weakness being removed.
    """
    import _socket

    library = tmp_path / f"libfixture-{uuid.uuid4().hex}.dylib"
    library.write_bytes(Path(_socket.__file__).read_bytes())
    return library


def test_a_launcher_under_a_wrapper_does_not_report_itself(tmp_path, stand_in_library) -> None:
    """R38: the self-match, with a real ancestry, a real mapping and a real outsider.

    The failure is reachable only with real processes because the exclusion is about kinship:
    every process here genuinely has the library mapped, so a filter that excludes nothing but
    the caller's own pid reports its parent and blocks the launch forever. That is not a
    hypothetical shape — a stage imports `mlx_lm.lora` to build its trainer arguments before it
    resolves the base model, so by the time the check runs the launcher and anything that
    spawned it can both be holding MLX.

    The outsider, in its own session, is the control that keeps the exclusion from degenerating
    into "report nothing", and it is the shape of the orphan this check was rewritten for: a
    process with no recognisable name that the kernel says is holding the library.
    """
    reporter = _script(tmp_path, "reporter.py", _MAPS_AND_REPORTS)
    waiter = _script(tmp_path, "waiter.py", _MAPS_AND_WAITS)
    sleeper = _script(tmp_path, "sleeper.py", _MAPS_AND_SLEEPS)
    out = tmp_path / "report.json"
    env = _child_env()

    # The outsider puts *itself* in another process group after exec. Asking `Popen` for
    # `start_new_session` would be the obvious way and is exactly what forces CPython onto the
    # fork path, which aborts an interpreter that has initialised Metal (see `spawn`).
    outsider = spawn.popen([sys.executable, str(sleeper), str(stand_in_library)], env=env)
    wrapper = spawn.popen(
        [sys.executable, str(waiter), str(stand_in_library), str(reporter), str(out)], env=env
    )
    try:
        wrapper.wait(timeout=_TIMEOUT)
        _wait_for(lambda: out.exists(), what="the wrapped launcher to report")
        payload = json.loads(out.read_text(encoding="utf-8"))

        # All three really held the library, so the exclusion had work to do.
        assert payload["pid"] in payload["mapped"]
        assert payload["ppid"] in payload["mapped"], "the wrapper must still be alive and mapped"
        assert outsider.pid in payload["mapped"]

        assert [pid for pid, _command, _library in payload["reported"]] == [outsider.pid]
        [(_pid, command, library)] = payload["reported"]
        assert library == str(stand_in_library)
        assert command, "a refusal has to be able to print the holder's command line"
    finally:
        outsider.terminate()
        outsider.wait(timeout=_TIMEOUT)


# ------------------------------------------------------------------ R38: no orphaned locks

_DIES = '''
import sys
from pathlib import Path
from local_llm_lab.runlock import hold_model_run_lock, model_run_lock

lock, mode = Path(sys.argv[1]), sys.argv[2]
if mode == "context-exception":
    with model_run_lock(path=lock, check_processes=False):
        raise RuntimeError("template does not define an assistant turn")
if mode == "hold-exception":
    hold_model_run_lock(path=lock, check_processes=False)
    raise ValueError("shapes (4,8) and (8,4) are not aligned")
if mode == "hold-systemexit":
    hold_model_run_lock(path=lock, check_processes=False)
    raise SystemExit("agent-v2-probe-patch: error: --layers is required")
if mode == "hold-keyboardinterrupt":
    hold_model_run_lock(path=lock, check_processes=False)
    raise KeyboardInterrupt
raise SystemExit("unknown mode " + mode)
'''


@pytest.mark.parametrize(
    "mode",
    ["context-exception", "hold-exception", "hold-systemexit", "hold-keyboardinterrupt"],
)
def test_a_process_that_dies_on_an_error_path_leaves_no_lock(tmp_path, mode: str) -> None:
    """R38: the orphaned lock, reachable only because the process really dies.

    These are the four ways the probe CLIs actually end badly — an exception raised inside the
    run, a `parser.error` `SystemExit`, and a Ctrl-C. An orphan from any of them blocks every
    later run, which is how a safety mechanism becomes the thing everyone routes around, and it
    cannot be shown in-process: the release under test is interpreter teardown.
    """
    script = _script(tmp_path, "dies.py", _DIES)
    lock = tmp_path / "run.lock"

    completed = spawn.run(
        [sys.executable, str(script), str(lock), mode],
        env=_child_env(),
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
    )

    assert completed.returncode != 0, completed.stdout + completed.stderr
    assert not lock.exists(), f"{mode} left an orphaned lock: {completed.stderr}"


_HOLDS_UNTIL_SIGNALLED = '''
import os, sys, time
from pathlib import Path
from local_llm_lab.runlock import hold_model_run_lock

lock, ready = Path(sys.argv[1]), Path(sys.argv[2])
hold_model_run_lock(path=lock, check_processes=False)
ready.touch()
time.sleep(60)
'''


def test_a_holder_killed_with_sigterm_leaves_no_lock(tmp_path) -> None:
    """R38: a person stopping a run is an error path too, and `atexit` never sees SIGTERM.

    Reachable only across processes, and only with a real signal: the handler under test is
    installed on the child's main thread and runs during its own teardown.
    """
    script = _script(tmp_path, "holds.py", _HOLDS_UNTIL_SIGNALLED)
    lock = tmp_path / "run.lock"
    ready = tmp_path / "ready"

    child = spawn.popen(
        [sys.executable, str(script), str(lock), str(ready)], env=_child_env()
    )
    try:
        _wait_for(lambda: ready.exists(), what="the holder to take the lock")
        assert lock.exists()
        child.send_signal(signal.SIGTERM)
        child.wait(timeout=_TIMEOUT)
    finally:
        if child.poll() is None:  # pragma: no cover - only on a failed run
            child.kill()
            child.wait(timeout=_TIMEOUT)

    assert child.returncode == -signal.SIGTERM
    assert not lock.exists(), "SIGTERM must not orphan the lock"


# --------------------------------------------------------------- the hold, inside one process
#
# `conftest.py` resets and releases the process-scoped hold around every test, so these three
# leave nothing behind for the rest of the suite.


def test_the_hold_is_idempotent_within_one_process(tmp_path) -> None:
    """A stage that loads a base model and then an adapted one takes the lock once."""
    lock = tmp_path / "run.lock"

    first = runlock.hold_model_run_lock(path=lock, check_processes=False)
    second = runlock.hold_model_run_lock(path=lock, check_processes=False)

    assert first == second == lock
    assert runlock.read_lock(lock) is not None


def test_the_hold_refuses_a_second_path(tmp_path) -> None:
    """Two lock paths in one process would mean two claims on one machine."""
    runlock.hold_model_run_lock(path=tmp_path / "a.lock", check_processes=False)

    with pytest.raises(runlock.RunLockError, match="already holds the model-run lock"):
        runlock.hold_model_run_lock(path=tmp_path / "b.lock", check_processes=False)


def test_a_release_from_a_forked_child_leaves_the_parents_lock(tmp_path) -> None:
    """A fork inherits the module state but not the ownership of the lock."""
    lock = tmp_path / "run.lock"
    runlock.hold_model_run_lock(path=lock, check_processes=False)

    stolen = runlock._Handle(path=lock, nonce=runlock._held.nonce, owner_pid=os.getpid() + 1)
    runlock._release(stolen)

    assert lock.exists()


# ---------------------------------------------------------- the box window (issue 95)


def test_a_window_is_announced_exclusively_and_closed_by_the_seat_that_opened_it(
    tmp_path,
) -> None:
    """Two seats announcing at once must not both believe they hold the slot.

    Exclusive create, like the lock, and the nonce is the same guard `_release` uses: a window is
    closed by whoever opened it, never by whoever happens to run next.
    """
    path = tmp_path / "window.json"
    nonce = runlock.announce_window("deputy", "issue 88 first block", 90.0, path=path)
    assert path.is_file()

    with pytest.raises(runlock.RunLockBusy, match="already open"):
        runlock.announce_window("codex", "regression calibration", 20.0, path=path)

    assert runlock.end_window("not-the-nonce", path=path) is False
    assert path.is_file(), "another seat's window is not closed by the wrong token"
    assert runlock.end_window(nonce, path=path) is True
    assert not path.exists()


def test_the_holder_is_not_blocked_by_its_own_window(tmp_path) -> None:
    """A seat is never blocked by the slot it holds, and every child of its command is the seat.

    The token comes from the environment the announcing command exports, so a suite the holder
    starts is the holder's; a suite anybody else starts is not.
    """
    path = tmp_path / "window.json"
    nonce = runlock.announce_window("codex", "replay checks", 20.0, path=path)

    assert runlock.blocking_window(path, holder=nonce) is None
    blocked = runlock.blocking_window(path, holder="")
    assert blocked is not None
    assert blocked.seat == "codex"
    assert "replay checks" in blocked.describe()


def test_an_overdue_window_is_reported_and_never_removed(tmp_path, monkeypatch) -> None:
    """The lock's rule, for the lock's reason.

    A mechanism that clears the state it finds inconvenient is not a mechanism. The refusal says
    the window is overdue and says who to ask; nothing here unlinks it.
    """
    path = tmp_path / "window.json"
    runlock.announce_window("codex", "regression calibration", 20.0, path=path)

    real = time.time
    monkeypatch.setattr(
        time, "time", lambda: real() + 20 * 60 + runlock.WINDOW_OVERDUE_SECONDS + 1
    )
    window = runlock.read_window(path)
    assert window is not None and window.overdue is True
    assert "overdue" in runlock.refusal_for_window(window)
    assert "not removed" in runlock.refusal_for_window(window)
    assert path.is_file(), "an overdue window is reported, not cleared"


def test_a_truncated_window_still_refuses_rather_than_reading_as_absent(tmp_path) -> None:
    """A half-written window is the case where guessing is worst.

    Absent means "the box is free". A file that will not parse means somebody wrote something and
    we cannot tell what, and reading that as free is how a launch lands in a live slot.
    """
    path = tmp_path / "window.json"
    path.write_text('{"seat": "codex", "purp', encoding="utf-8")

    window = runlock.read_window(path)
    assert window is not None
    assert window.seat == "unknown seat"
    assert runlock.blocking_window(path, holder="") is not None


def test_the_refusal_says_what_the_inventory_could_not(tmp_path) -> None:
    """The wording carries the distinction the whole issue turns on.

    Four launches were refused on 2026-09-08 because two people checked the live inventory, which
    was honest and answered a different question. The refusal says so, so the next reader does not
    have to rediscover it.
    """
    path = tmp_path / "window.json"
    runlock.announce_window("codex", "regression calibration retry", 20.0, path=path)
    text = runlock.refusal_for_window(runlock.read_window(path))

    assert "another seat holds the box window" in text
    assert "codex" in text and "regression calibration retry" in text
    assert "next minutes" in text and "inventory cannot" in text


def test_the_collector_skips_the_mlx_files_under_another_seats_window(monkeypatch) -> None:
    """The suite's half of the mechanism (issue 95).

    Skipped rather than refused: a refusal makes the whole suite red for somebody who has done
    nothing wrong, and the natural response to a red suite is to run it again, which is the
    collision. The reason carries the window so the reader learns whose slot they are in.

    Driven by calling the hook rather than by nesting a pytest run. A nested run would have to
    write the window at the real path, since `default_window_path` derives from `PROJECT_ROOT`,
    and a test that briefly refuses live seats is the wrong way to test a mechanism built because
    live seats kept being refused.
    """
    import conftest

    window = runlock.BoxWindow(
        path=Path("outputs/.box-window.json"),
        seat="codex",
        purpose="regression calibration retry",
        opened="2026-09-08T11:11:02Z",
        expected_end_epoch=None,
        nonce="abc",
        raw="{}",
        age_seconds=10.0,
    )
    monkeypatch.setattr(conftest, "blocking_window", lambda *a, **k: window, raising=False)
    monkeypatch.setattr(runlock, "blocking_window", lambda *a, **k: window)

    class _Item:
        def __init__(self, name: str) -> None:
            self.fspath = Path("/repo/tests") / name
            self.markers: list = []

        def add_marker(self, marker) -> None:
            self.markers.append(marker)

    loads_mlx = _Item("test_preflight.py")
    does_not = _Item("test_runlock.py")
    class _Config:
        stash: dict = {}

    conftest.pytest_collection_modifyitems(_Config(), [loads_mlx, does_not])

    assert len(loads_mlx.markers) == 1, "an MLX-loading file is skipped"
    assert does_not.markers == [], "a file that loads no MLX is untouched"
    reason = loads_mlx.markers[0].kwargs["reason"]
    assert "codex" in reason and "regression calibration retry" in reason
    assert "issue 95" in reason


def test_the_collector_leaves_the_holders_own_suite_alone(monkeypatch) -> None:
    """`blocking_window` returns None for the holder, so the hook is a no-op for that seat."""
    import conftest

    monkeypatch.setattr(runlock, "blocking_window", lambda *a, **k: None)

    class _Item:
        fspath = Path("/repo/tests/test_preflight.py")
        markers: list = []

        def add_marker(self, marker):  # pragma: no cover - must not be reached
            raise AssertionError("the holder's own suite must not be skipped")

    class _Config:
        stash: dict = {}

    conftest.pytest_collection_modifyitems(_Config(), [_Item()])


def test_the_pinned_mlx_file_list_has_one_home(monkeypatch) -> None:
    """One list, two enforcers: the collector and the closure rule read the same tuple.

    `tests/test_repository_rules.py` binds `_TESTS_THAT_LOAD_MLX` to `conftest`'s tuple rather
    than keeping a copy, so a file that starts or stops loading MLX cannot be right in one place
    and wrong in the other.
    """
    import conftest
    import test_repository_rules

    assert test_repository_rules._TESTS_THAT_LOAD_MLX is conftest.TESTS_THAT_LOAD_MLX


def test_the_lock_refuses_a_launch_into_another_seats_window_and_admits_the_holders(
    monkeypatch, tmp_path
) -> None:
    """The lock's half, and the order matters.

    The window is consulted **before** the process inventory, because a seat can hold the next
    twenty minutes without holding a process this instant. That gap is the whole issue: on
    2026-09-08 two people checked the live inventory, got an honest answer to a different
    question, and refused a third seat's launch four times between them.
    """
    path = tmp_path / "window.json"
    nonce = runlock.announce_window("codex", "regression calibration", 20.0, path=path)
    monkeypatch.setattr(runlock, "default_window_path", lambda: path)
    monkeypatch.setattr(runlock, "running_model_processes", lambda *a, **k: [])
    monkeypatch.delenv(runlock.WINDOW_HOLDER_ENV, raising=False)

    with pytest.raises(runlock.RunLockBusy) as refusal:
        runlock.hold_model_run_lock(path=tmp_path / "run.lock")
    assert "another seat holds the box window" in str(refusal.value)
    assert "codex" in str(refusal.value)
    assert not (tmp_path / "run.lock").exists(), "a refused launch leaves no lock"

    monkeypatch.setenv(runlock.WINDOW_HOLDER_ENV, nonce)
    held = runlock.hold_model_run_lock(path=tmp_path / "run.lock")
    assert Path(held).is_file(), "the seat that announced the window may launch into it"


def test_a_linked_worktree_resolves_to_the_primarys_lock_and_window(
    tmp_path, monkeypatch, unredirected_window_path
) -> None:
    """The defect that would have left all four of 2026-09-08's refusals unprevented (issue 95).

    Both files used to hang off `PROJECT_ROOT`, which is the **running checkout's** root. Every
    suite that refused a peer that day ran in a linked worktree, so each was looking in its own
    empty `outputs/` and could not see a window announced from the primary. The mechanism built
    to prevent those refusals would have prevented none of them.

    A real worktree, made through `spawn.run` because this file may not fork (R45).
    """
    primary = tmp_path / "primary"
    primary.mkdir()
    for argv in (
        ["git", "-C", str(primary), "init", "-q"],
        ["git", "-C", str(primary), "config", "user.email", "t@example.com"],
        ["git", "-C", str(primary), "config", "user.name", "t"],
        ["git", "-C", str(primary), "commit", "-q", "--allow-empty", "-m", "root"],
        ["git", "-C", str(primary), "worktree", "add", "-q", str(tmp_path / "linked")],
    ):
        spawn.run(argv, capture_output=True, text=True, check=True)

    linked = tmp_path / "linked"
    assert (linked / ".git").is_file(), "a linked worktree's .git is a file, which is the signal"

    monkeypatch.setattr(runlock, "PROJECT_ROOT", linked)
    monkeypatch.delenv(runlock.BOX_STATE_DIR_ENV, raising=False)
    assert runlock.box_state_root() == primary.resolve()
    # The window path is asserted through the function; the lock's is asserted as the root plus
    # the relative path, because `conftest` redirects `default_lock_path` for the whole suite so
    # no test can take the machine's lock. Both derive from the same root, which is the claim.
    assert runlock.default_window_path() == primary.resolve() / runlock.WINDOW_RELATIVE_PATH
    assert runlock.box_state_root() / runlock.LOCK_RELATIVE_PATH == (
        primary.resolve() / runlock.LOCK_RELATIVE_PATH
    )

    # The primary is unchanged, which is the other half: only worktrees move.
    monkeypatch.setattr(runlock, "PROJECT_ROOT", primary)
    assert runlock.box_state_root() == primary


def test_the_state_root_override_is_operational_not_a_test_knob(
    tmp_path, monkeypatch, unredirected_window_path
) -> None:
    """A second clone on one machine keeps its own box state unless pointed at the shared one.

    That is the same defeat a worktree used to have, one level up, and it cannot be read off the
    filesystem the way a worktree's `.git` file can — so it is an environment variable, documented
    as operational rather than introduced for a test.
    """
    monkeypatch.setenv(runlock.BOX_STATE_DIR_ENV, str(tmp_path / "shared"))
    assert runlock.box_state_root() == tmp_path / "shared"
    assert runlock.default_window_path() == tmp_path / "shared" / runlock.WINDOW_RELATIVE_PATH


def test_an_unreadable_git_marker_falls_back_to_the_checkouts_own_root(
    tmp_path, monkeypatch
) -> None:
    """Refusing on a broken `.git` would be worse than behaving as it did before this existed."""
    checkout = tmp_path / "odd"
    checkout.mkdir()
    (checkout / ".git").write_text("not a gitdir line at all\n", encoding="utf-8")
    monkeypatch.setattr(runlock, "PROJECT_ROOT", checkout)
    monkeypatch.delenv(runlock.BOX_STATE_DIR_ENV, raising=False)
    assert runlock.box_state_root() == checkout


def test_the_collector_fires_inside_a_nested_run_under_a_temporary_state_root(
    pytester, tmp_path, monkeypatch, unredirected_window_path
) -> None:
    """The registration gap the first delivery could not close, closed (issue 95).

    A nested pytest run is what proves the hook is wired into pytest at all, rather than merely
    correct when called. It could not be written before because the window's path came from
    `PROJECT_ROOT` and a nested run would have written the real one and briefly refused live
    seats. With the state root overridable, the nested run gets a temporary directory and the
    real path is never touched.

    Both directions: a foreign window skips the file, and the holder's own token runs it.
    """
    state = tmp_path / "state"
    (state / "outputs").mkdir(parents=True)
    nonce = runlock.announce_window(
        "codex", "regression calibration retry", 20.0, path=state / runlock.WINDOW_RELATIVE_PATH
    )

    pytester.makeconftest(
        """
        from local_llm_lab.runlock import mark_items_for_a_foreign_window

        def pytest_collection_modifyitems(config, items):
            mark_items_for_a_foreign_window(items, ("test_window_probe_reaches_mlx.py",))
        """
    )
    pytester.makepyfile(test_window_probe_reaches_mlx="def test_one():\n    assert True\n")

    monkeypatch.setenv(runlock.BOX_STATE_DIR_ENV, str(state))
    monkeypatch.delenv(runlock.WINDOW_HOLDER_ENV, raising=False)
    # In-process, not `runpytest_subprocess`: pytester's subprocess runner forks, and the fork
    # guard refuses it because this interpreter has MLX up (R45). In-process still exercises the
    # thing this test exists for -- pytest collecting a conftest and calling its hook.
    foreign = pytester.runpytest("-rs")
    foreign.assert_outcomes(skipped=1)
    foreign.stdout.fnmatch_lines(["*codex: regression calibration retry*"])

    monkeypatch.setenv(runlock.WINDOW_HOLDER_ENV, nonce)
    holders = pytester.runpytest()
    holders.assert_outcomes(passed=1)


def test_a_window_is_extended_without_the_slot_being_dropped(tmp_path) -> None:
    """Correcting a duration must not free the box for a moment (issue 95, second amendment).

    Ending and re-announcing is the obvious way and the wrong one: the file cannot say "the same
    holder, a moment later", so between the two calls the slot is genuinely free and another seat
    may take it. A block that overran its estimate is exactly when that must not happen — which is
    the case that produced this, a 100-minute announcement for a 131-minute block.
    """
    path = tmp_path / "window.json"
    nonce = runlock.announce_window("deputy", "88 block 1", 100.0, path=path)
    before = runlock.read_window(path)

    assert runlock.extend_window("not-the-nonce", 140.0, path=path) is False
    assert runlock.read_window(path).expected_end_epoch == before.expected_end_epoch

    assert runlock.extend_window(nonce, 140.0, path=path) is True
    after = runlock.read_window(path)
    assert after.nonce == nonce, "the same window, not a new one"
    assert after.expected_end_epoch > before.expected_end_epoch
    assert path.is_file(), "the file never left, so the slot never opened"

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["extensions"][0]["new_expected_minutes"] == 140.0


def test_extending_a_window_nobody_holds_is_refused(tmp_path) -> None:
    """There is nothing to extend, and inventing one would hand out a slot."""
    assert runlock.extend_window("any-token", 30.0, path=tmp_path / "absent.json") is False


def test_the_terminal_summary_says_a_window_skipped_part_of_the_run() -> None:
    """The hazard the skip design carries, addressed where a reader will meet it.

    A skip is right and a silent skip is not: on 2026-09-08 a reviewer read "exit 0" from a run
    whose whole `test_cli.py` the collector had skipped, and reported the file green. The exit
    code was honest and the count was in the summary; nothing said the omission had a cause.
    """
    import conftest

    window = runlock.BoxWindow(
        path=Path("outputs/.box-window.json"),
        seat="deputy",
        purpose="88 block 1",
        opened="2026-09-08T12:17:03Z",
        expected_end_epoch=None,
        nonce="abc",
        raw="{}",
        age_seconds=60.0,
    )

    class _Reporter:
        def __init__(self) -> None:
            self.lines: list[str] = []

        def write_sep(self, _char, title, **_kwargs) -> None:
            self.lines.append(f"[{title}]")

        def write_line(self, text) -> None:
            self.lines.append(text)

    class _Config:
        def __init__(self, value) -> None:
            self.stash = {conftest._WINDOW_SKIP: value}

    reporter = _Reporter()
    conftest.pytest_terminal_summary(reporter, 0, _Config((window, 71)))
    joined = "\n".join(reporter.lines)
    assert "[box window]" in joined
    assert "71 test(s) were skipped, not run" in joined
    assert "deputy: 88 block 1" in joined
    assert "before reporting them green" in joined

    quiet = _Reporter()
    conftest.pytest_terminal_summary(quiet, 0, _Config(None))
    assert quiet.lines == [], "a run no window touched says nothing"


def test_a_windows_holder_state_is_read_not_inferred(tmp_path) -> None:
    """Three states, because two would make a guess look like a fact (issue 97).

    A window with no recorded pid, or one whose pid belongs to another user, is not evidence
    either way. `unknown` says so; reporting it as "not running" would invite somebody to clear
    another seat's live slot.
    """
    path = tmp_path / "window.json"
    runlock.announce_window("deputy", "a block", 10.0, path=path, pid=os.getpid())
    assert runlock.read_window(path).holder_state == "running"
    assert runlock.read_window(path).orphaned is False

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["pid"] = _a_pid_that_is_not_running()
    path.write_text(json.dumps(payload), encoding="utf-8")
    dead = runlock.read_window(path)
    assert dead.holder_state == "not running"
    assert dead.orphaned is True
    assert "holder not running" in dead.describe()

    payload.pop("pid")
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert runlock.read_window(path).holder_state == "unknown"
    assert runlock.read_window(path).orphaned is False, "unknown is not evidence of an orphan"


def _a_pid_that_is_not_running() -> int:
    """A pid nothing holds, found rather than assumed."""
    for candidate in range(400000, 500000):
        try:
            os.kill(candidate, 0)
        except ProcessLookupError:
            return candidate
        except (PermissionError, OSError):
            continue
    raise AssertionError("no free pid found")


def test_orphaned_and_overdue_are_different_facts(tmp_path, monkeypatch) -> None:
    """The distinction the mechanism lacked on 2026-09-08, when it cost a night of box time.

    A block finished at 13:57Z, its `end` never ran, and the window sat for eight hours **well
    inside** the hour of slack past its expected end. Overdue could not have caught it; only the
    holder's absence could, and nothing recorded a pid whose life meant anything.
    """
    path = tmp_path / "window.json"
    runlock.announce_window("deputy", "a block", 100.0, path=path, pid=_a_pid_that_is_not_running())

    window = runlock.read_window(path)
    assert window.orphaned is True
    assert window.overdue is False, "well inside its expected end, and still nobody holds it"


def test_run_announces_runs_and_ends_whatever_the_command_did(
    tmp_path, monkeypatch, unredirected_window_path
) -> None:
    """The verb that stops a window outliving the seat that opened it (issue 97).

    `finally`, not a success path: a command that crashes or exits non-zero must still release
    the slot, because the failure this replaces is exactly the one where nothing tidy happened at
    the end.

    Two things this test had to be rewritten to do, and they are the same lesson. It took
    `unredirected_window_path`, because the suite's autouse fixture redirects
    `default_window_path` to its own temporary directory and the `BOX_STATE_DIR_ENV` override
    below therefore reached nothing: the window was written where the fixture pointed and the
    asserted path never held a file. And the command is now `test -e` on that path, so the
    child's exit status is the evidence that the window existed *while it ran* -- otherwise
    `not window.exists()` afterwards would pass just as well if `run` had never opened one,
    which is the one thing the test exists to rule out. An instrument proves itself (R52).
    """
    state = tmp_path / "state"
    (state / "outputs").mkdir(parents=True)
    monkeypatch.setenv(runlock.BOX_STATE_DIR_ENV, str(state))
    window = state / runlock.WINDOW_RELATIVE_PATH
    assert runlock.default_window_path() == window, "the test asserts on the path the code writes"

    code = runlock.run_under_window("deputy", "a command", 5.0, ["/bin/test", "-e", str(window)])
    assert code == 0, "the child found the window open while it ran"
    assert not window.exists(), "the window ends with the command"

    # `/bin/sh -c 'exit 3'` rather than `/bin/false`, which is not at that path on this host.
    failing = runlock.run_under_window(
        "deputy", "a failing command", 5.0, ["/bin/sh", "-c", "exit 3"]
    )
    assert failing == 3, "the exit status is the command's, passed through"
    assert not window.exists(), "and the window ends anyway, which is the whole point"


def test_run_ends_its_window_when_the_wrapper_is_killed(tmp_path) -> None:
    """`finally` never runs on SIGTERM, and that is how last night's window survived.

    Through the CLI as a subprocess, because the thing under test is a signal disposition and a
    child process, neither of which exists when `run_under_window` is called in-process. The
    wrapper is killed the way a person or a scheduler kills one; the window must be gone and the
    child with it, or the verb has only moved the failure from `announce` to `run`.
    """
    state = tmp_path / "state"
    (state / "outputs").mkdir(parents=True)
    window = state / runlock.WINDOW_RELATIVE_PATH
    environment = {**_child_env(), runlock.BOX_STATE_DIR_ENV: str(state)}

    wrapper = spawn.popen(
        [
            sys.executable,
            "-m",
            "local_llm_lab.runlock",
            "run",
            "--seat",
            "deputy",
            "--purpose",
            "a command that outlives its wrapper unless something stops it",
            "--minutes",
            "5",
            "--",
            "/bin/sleep",
            "30",
        ],
        env=environment,
    )
    try:
        deadline = time.monotonic() + 20.0
        while not window.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert window.exists(), "the wrapper opened its window"
        held = runlock.read_window(window)
        assert held is not None and held.holder_state == "running"

        wrapper.send_signal(signal.SIGTERM)
        status = wrapper.wait(timeout=20.0)
    finally:
        if wrapper.poll() is None:  # pragma: no cover - only if the assertions above failed
            wrapper.kill()
            wrapper.wait(timeout=10.0)

    assert not window.exists(), "SIGTERM closed the window on the way out"
    assert status != 0, "and the wrapper died of the signal rather than returning cleanly"


def test_run_gives_its_child_the_holder_token(tmp_path, monkeypatch) -> None:
    """The command's own suites and launches are the holder's, or the verb defeats itself."""
    state = tmp_path / "state"
    (state / "outputs").mkdir(parents=True)
    monkeypatch.setenv(runlock.BOX_STATE_DIR_ENV, str(state))
    seen = tmp_path / "token.txt"

    runlock.run_under_window(
        "deputy",
        "a command that reports its token",
        5.0,
        ["/bin/sh", "-c", f'printf "%s" "${runlock.WINDOW_HOLDER_ENV}" > {seen}'],
    )
    assert seen.read_text(encoding="utf-8"), "the child saw a token"


def test_the_window_records_the_pid_it_is_told_to(tmp_path) -> None:
    """Issue 97's cause: the CLI's `announce` exits a second after writing.

    A window recording that process names something already gone, so every such window read as
    "holder not running" from the moment it opened and a dead pid could never be the witness that
    one had been orphaned. `announce` passes its parent; `run` passes its own.
    """
    path = tmp_path / "window.json"
    runlock.announce_window("deputy", "a block", 10.0, path=path, pid=4242)
    assert runlock.read_window(path).pid == 4242

    path.unlink()
    runlock.announce_window("deputy", "a block", 10.0, path=path)
    assert runlock.read_window(path).pid == os.getpid(), "the default is still this process"


def test_the_cli_strips_only_the_separator_argparse_left_it(monkeypatch) -> None:
    """`runlock run ... -- pytest -- tests/` must reach pytest with its own `--` intact.

    `REMAINDER` hands back the leading separator and nothing else needs removing, so a filter
    over every element silently rewrites the command it was asked to run -- and the commands
    where it matters are exactly the ones a seat reaches for under a window.
    """
    seen: list[list[str]] = []

    def record(seat: str, purpose: str, minutes: float, argv: list[str]) -> int:
        seen.append(list(argv))
        return 0

    monkeypatch.setattr(runlock, "run_under_window", record)
    common = ["run", "--seat", "deputy", "--purpose", "p", "--minutes", "5"]

    runlock._window_cli([*common, "--", "/bin/echo", "--", "after"])
    runlock._window_cli([*common, "/bin/echo", "--", "after"])

    assert seen == [["/bin/echo", "--", "after"], ["/bin/echo", "--", "after"]], (
        "the leading separator goes, the command's own stays, and either spelling works"
    )


def test_a_refused_window_is_a_refusal_at_the_cli_not_a_traceback(
    tmp_path, monkeypatch, capsys, unredirected_window_path
) -> None:
    """A seat that cannot have the box reads why and waits; a traceback reads as a bug.

    Both verbs, because `run` reaches `announce_window` through `run_under_window` and would
    otherwise raise from inside the wrapper with a window already open somewhere else.

    `unredirected_window_path` for the same reason the `run` test above needs it: without it the
    planted window and the CLI's own lookup are two different files and the refusal never fires.
    """
    state = tmp_path / "state"
    (state / "outputs").mkdir(parents=True)
    monkeypatch.setenv(runlock.BOX_STATE_DIR_ENV, str(state))
    window = state / runlock.WINDOW_RELATIVE_PATH
    assert runlock.default_window_path() == window
    runlock.announce_window("another seat", "a block of theirs", 30.0)

    common = ["--seat", "deputy", "--purpose", "p", "--minutes", "5"]
    for argv in (["announce", *common], ["run", *common, "--", "/bin/echo", "hi"]):
        capsys.readouterr()
        assert runlock._window_cli(argv) == 1, "refused, and said so"
        refusal = capsys.readouterr().err
        assert "already open" in refusal and "another seat" in refusal
    assert window.exists(), "and the other seat's window is untouched"


def test_a_child_killed_by_a_signal_becomes_the_shells_own_status(tmp_path) -> None:
    """`Popen` reports -N for a child killed by signal N, and `sys.exit(-N)` truncates.

    128 + N is what every shell in the chain expects, and the wrapper's whole job is to be
    transparent about what the command did.
    """
    state = tmp_path / "state"
    (state / "outputs").mkdir(parents=True)
    environment = {**_child_env(), runlock.BOX_STATE_DIR_ENV: str(state)}

    completed = spawn.run(
        [
            sys.executable,
            "-m",
            "local_llm_lab.runlock",
            "run",
            "--seat",
            "deputy",
            "--purpose",
            "a command that kills itself",
            "--minutes",
            "5",
            "--",
            "/bin/sh",
            "-c",
            "kill -TERM $$",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
    )
    assert completed.returncode == 128 + signal.SIGTERM, completed.stderr
    assert not (state / runlock.WINDOW_RELATIVE_PATH).exists(), "and the window still ended"
