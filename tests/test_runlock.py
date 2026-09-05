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
    """
    assert runlock.LOCK_RELATIVE_PATH.as_posix() == "outputs/.model-run.lock"
    assert machine_lock_path == PROJECT_ROOT / runlock.LOCK_RELATIVE_PATH


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
