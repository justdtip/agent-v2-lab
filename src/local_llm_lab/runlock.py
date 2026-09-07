"""The model-run lock: one model-loading process on this machine at a time.

The standing lift's concurrency clause
(``design_specifications/live/STANDING-LIFT-EXPERIMENTS-2026-09-05.md``) makes concurrency the
only constraint on a research run, and this module is its whole mechanism. The machine has
24 GiB of unified memory and recommends a 17.8 GiB working set (``preflight.py:113``); two 4B
loads at once is a swap storm whose first casualty is whichever run was further along.

Four choices are load-bearing and none of them may be softened.

*Exclusive create.* The lock is taken with ``os.open(..., O_CREAT | O_EXCL)``, which fails when
the file already exists. "Check whether it exists, then write it" reopens the very race the lock
closes: two launchers can both see nothing and both write.

*Held for the process, not for the call.* The resource is the model resident in memory, and its
lifetime is the process's — from the load until the interpreter exits. A lock released when
``mlx_lm.load`` returns would let a second run load while the first is still resident, which is
the failure it was built to prevent. So ``load_weights`` takes a process-scoped hold and releases
it at exit; ``model_run_lock`` is the explicit-scope form for callers that own a narrower region.

*Released on every exit path.* The probe CLIs die on template and shape errors, and an orphaned
lock blocks every later run — which is how a safety mechanism becomes the thing everyone routes
around. The hold releases through ``atexit`` (covering a normal return, ``SystemExit``, an
uncaught exception and ``KeyboardInterrupt``) and through handlers for ``SIGTERM``/``SIGHUP``,
which ``atexit`` alone would not see. ``SIGKILL`` cannot be caught; that case is the stale lock
below.

*A stale lock is reported, never deleted by a launcher.* A launcher that clears the locks it
finds inconvenient is a launcher with no lock. Every refusal names the holder's pid, its command
line and the lock's age, so a false match is diagnosed in seconds rather than guessed at, and
leaves the decision to a person.

*The lock binds only what goes through it.* Every entry point in this package reaches weights
through ``load_weights`` or ``hold_model_run_lock``, so every one of them takes the lock. Nothing
else does: a scratch script, a notebook, a checkout of another branch, anything that imports
``mlx`` without importing this package holds 16 GB and no lock. That gap is real and it is not
closeable from inside the package, so the check beside the lock covers it by asking the kernel
which processes have the MLX library mapped, rather than by matching names we chose (see
``MLX_LIBRARY`` for why names and memory sizes both fail on the case that matters). It is
fail-open: the lock file is the mechanism, and a check that cannot run must not become a second
way to block a launch.

Environment: ``MODEL_RUN_SESSION`` names the launching session in the lock file. Everything else
in the record — command, timestamp, pid — is read from the process itself.
"""

from __future__ import annotations

import atexit
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from local_llm_lab.project import PROJECT_ROOT
from local_llm_lab.spawn import run as spawn_run

__all__ = [
    "LOCK_RELATIVE_PATH",
    "BOX_STATE_DIR_ENV",
    "WINDOW_HOLDER_ENV",
    "WINDOW_OVERDUE_SECONDS",
    "WINDOW_RELATIVE_PATH",
    "BoxWindow",
    "announce_window",
    "box_state_root",
    "mark_items_for_a_foreign_window",
    "blocking_window",
    "end_window",
    "read_window",
    "refusal_for_window",
    "MLX_LIBRARY",
    "MappedProcess",
    "STALE_AFTER_SECONDS",
    "LockHeld",
    "RunLockBusy",
    "RunLockError",
    "default_lock_path",
    "hold_model_run_lock",
    "load_weights",
    "model_run_lock",
    "read_lock",
    "refusal_for_processes",
    "running_model_processes",
]

#: Relative to the project root, as the concurrency clause of record names it.
LOCK_RELATIVE_PATH = Path("outputs/.model-run.lock")

#: Longer than the longest run in the briefing's cost table (a 52-minute 180-task evaluation, a
#: 70-minute QLoRA run, a 63-minute P2 capture), times the 2.5x allowance for the 9B model, with
#: room to spare. A lock older than this cannot belong to a run that is still going, so it is
#: reported as stale — and still not deleted, because "cannot" is a judgement a person makes.
STALE_AFTER_SECONDS = 6 * 60 * 60

#: The library every process holding an MLX model has mapped. **The check asks the kernel which
#: processes have this mapped; it does not match process names, and it does not look at memory
#: size.** Both of those were tried and both fail on the case that matters:
#:
#: * *Names.* The orphan found on this machine on 2026-09-05 (pid 63943, ~16-17 GiB of a 4B
#:   model resident on a 24 GiB box that recommends a 17.8 GiB working set) was called
#:   ``ctxmax.py`` and matched nothing in any pattern list.
#:   That is what a scratch script always looks like. A name list only ever finds the runs that
#:   were already going to be careful.
#: * *Size.* That same process reported about 25 MB resident in ``ps``, three orders of magnitude
#:   under its real footprint, because Metal buffer allocations never land in RSS. Any check that
#:   tries to tell a real load from an incidental import by looking at memory waves it straight
#:   through. Presence of the mapping is the signal; there is no size gate.
#:
#: **This is the part that catches code the repository does not own.** The lock binds only
#: callers that go through ``load_weights`` or ``hold_model_run_lock`` — anything inside the
#: package. A scratch script, a notebook, or a checkout of another branch that imports ``mlx``
#: takes no lock and never will, and can still collide with a run for the whole 24 GiB.
MLX_LIBRARY = "libmlx.dylib"

#: The two external queries, named once so the failure record cannot drift from the call.
#: Neither returns a size; see ``MLX_LIBRARY`` for why that is deliberate.
_PS_COMMAND = ("ps", "-Ao", "pid=,ppid=,pgid=,command=")
_LSOF_COMMAND = ("lsof", "-w", "-d", "txt", "-F", "pn")

_MAX_COMMAND_CHARS = 2000
_PS_TIMEOUT = 10.0
_LSOF_TIMEOUT = 20.0
_LINEAGE_HOPS = 64


class RunLockError(RuntimeError):
    """The lock could not be used as asked (a misuse, not a busy machine)."""


class RunLockBusy(RunLockError):
    """Another model-loading run holds the machine. The message names pid and command."""


def default_lock_path() -> Path:
    """The lock, under the shared box-state root rather than the running checkout's own.

    Unchanged for the primary. A worktree used to take a lock nobody else could see.
    """
    return box_state_root() / LOCK_RELATIVE_PATH


def _utc_now() -> str:
    """UTC ISO 8601 with a trailing ``Z``, matching ``runlog``."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _default_command() -> str:
    try:
        rendered = shlex.join(sys.argv)
    except (TypeError, ValueError):  # pragma: no cover - argv is always a list of str
        rendered = " ".join(str(part) for part in sys.argv)
    return rendered[:_MAX_COMMAND_CHARS]


def _default_session() -> str:
    return os.environ.get("MODEL_RUN_SESSION", "").strip() or "unnamed"


def _duration(seconds: float | None) -> str:
    """``H:MM:SS`` for a lock age; ``unknown`` when the record carries no usable time."""
    if seconds is None:
        return "unknown"
    total = int(seconds) if seconds > 0 else 0
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


# ------------------------------------------------------------------------ the box window
#
# The lock answers "is a model resident **now**". A window answers "has a seat been promised the
# next twenty minutes", and the two are not the same question. On 2026-09-08 four launches were
# refused, three of them one seat's, and two of the refusals came from people who had read the
# announcement practice and meant to follow it: each checked the live inventory, which was honest
# and answered the other question. A practice that fails two careful readers in one evening is
# the case for a mechanism, and this is it (issue 95).
#
# The heartbeat announcement stays as the human record. This file is what tooling reads.

WINDOW_RELATIVE_PATH = Path("outputs/.box-window.json")

#: Where the lock and the window live, when it is not the running checkout's own root.
#:
#: **Operational, not a test knob.** A second clone of this repository on the same machine has a
#: different root and would otherwise keep its own box state, which defeats the mechanism exactly
#: as a worktree does. Point every checkout on one machine at one directory.
BOX_STATE_DIR_ENV = "AGENT_V2_BOX_STATE_DIR"


def box_state_root() -> Path:
    """The one directory the box's lock and window live under, for every checkout on this machine.

    ``PROJECT_ROOT`` is the **running checkout's** root, and both files used to hang off it. Every
    suite that refused a peer's launch on 2026-09-08 ran in a linked worktree, so each was looking
    at its own empty ``outputs/`` and could not see a window announced from the primary; the
    mechanism built to prevent those four refusals would have prevented none of them. The lock has
    always had the same weakness, which is why one seat's runtime rebinds ``PROJECT_ROOT`` by hand
    before taking it; what has made the lock work across checkouts anyway is that the *process*
    inventory is global while the file is not.

    A linked worktree's ``.git`` is a **file** reading ``gitdir: <primary>/.git/worktrees/<name>``,
    and the primary is the parent of that ``.git`` directory. A directory ``.git`` means this
    checkout is the primary. That is read rather than shelled out for, so the answer costs no
    process and cannot fail on a machine where git is missing.

    The primary's behaviour is unchanged: it resolves to its own root, as before. Only worktrees
    move, from a private and useless file to the shared one.
    """
    override = os.environ.get(BOX_STATE_DIR_ENV)
    if override:
        return Path(override)
    marker = PROJECT_ROOT / ".git"
    try:
        if marker.is_file():
            text = marker.read_text(encoding="utf-8").strip()
            if text.startswith("gitdir:"):
                gitdir = Path(text.split(":", 1)[1].strip())
                if not gitdir.is_absolute():
                    gitdir = (PROJECT_ROOT / gitdir).resolve()
                for parent in gitdir.parents:
                    if parent.name == ".git":
                        return parent.parent
    except OSError:
        # An unreadable `.git` is not a reason to refuse; it is a reason to behave as the
        # checkout's own root, which is what happened before this function existed.
        pass
    return PROJECT_ROOT


#: The announcing command exports this with the window's nonce, and every child inherits it. That
#: is what separates "the seat that opened the window" from "everybody else" without a pid tree:
#: a suite the holder starts is theirs, a suite anybody else starts is not.
WINDOW_HOLDER_ENV = "AGENT_V2_BOX_WINDOW"

#: How far past its own expected end a window may sit before it is called overdue. Reported, never
#: removed -- the lock's rule, for the lock's reason: a mechanism that clears the state it finds
#: inconvenient is not a mechanism.
WINDOW_OVERDUE_SECONDS = 60 * 60


def default_window_path() -> Path:
    return box_state_root() / WINDOW_RELATIVE_PATH


@dataclass(frozen=True)
class BoxWindow:
    """One announced slot as it reads on disk."""

    path: Path
    seat: str
    purpose: str
    opened: str
    expected_end_epoch: float | None
    nonce: str
    raw: str
    age_seconds: float | None

    @property
    def overdue(self) -> bool:
        """Past its own expected end by :data:`WINDOW_OVERDUE_SECONDS`."""
        if self.expected_end_epoch is None:
            return False
        return time.time() > self.expected_end_epoch + WINDOW_OVERDUE_SECONDS

    def describe(self) -> str:
        late = " (overdue)" if self.overdue else ""
        return f"{self.seat}: {self.purpose}, opened {self.opened}{late}"


def read_window(path: Path | None = None, *, now: float | None = None) -> BoxWindow | None:
    """The announced window, or None. Readable even when the JSON is truncated or foreign."""
    target = path if path is not None else default_window_path()
    try:
        raw = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return None
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("not an object")
    except (json.JSONDecodeError, ValueError):
        payload = {}
    opened_epoch = payload.get("opened_epoch")
    moment = time.time() if now is None else now
    return BoxWindow(
        path=target,
        seat=str(payload.get("seat", "unknown seat")),
        purpose=str(payload.get("purpose", "unstated purpose")),
        opened=str(payload.get("opened", "an unrecorded time")),
        expected_end_epoch=(
            float(payload["expected_end_epoch"])
            if isinstance(payload.get("expected_end_epoch"), int | float)
            else None
        ),
        nonce=str(payload.get("nonce", "")),
        raw=raw,
        age_seconds=(
            moment - float(opened_epoch) if isinstance(opened_epoch, int | float) else None
        ),
    )


def announce_window(
    seat: str, purpose: str, expected_minutes: float, path: Path | None = None
) -> str:
    """Open the box window and return its nonce, which the caller exports as the holder token.

    Exclusive create, like the lock and for the same reason: two seats announcing at once must
    not both believe they hold the slot.
    """
    target = path if path is not None else default_window_path()
    nonce = uuid.uuid4().hex
    now = time.time()
    payload = {
        "seat": seat,
        "purpose": purpose,
        "opened": _utc_now(),
        "opened_epoch": now,
        "expected_end_epoch": now + expected_minutes * 60.0,
        "expected_minutes": expected_minutes,
        "nonce": nonce,
        "pid": os.getpid(),
    }
    try:
        _write_lock(target, payload)
    except FileExistsError as error:
        held = read_window(target)
        raise RunLockBusy(
            "refusing to announce a window: one is already open.\n"
            f"  {held.describe() if held else target}\n"
            "Wait for its end line, or ask that seat to close it."
        ) from error
    return nonce


def end_window(nonce: str, path: Path | None = None) -> bool:
    """Close a window this process opened. Returns False if the file is somebody else's.

    The nonce check is the same guard ``_release`` uses: a window is closed by the seat that
    opened it, never by whoever happens to run next.
    """
    target = path if path is not None else default_window_path()
    held = read_window(target)
    if held is None:
        return False
    if held.nonce != nonce:
        return False
    try:
        target.unlink()
    except OSError:
        return False
    return True


def blocking_window(path: Path | None = None, *, holder: str | None = None) -> BoxWindow | None:
    """The window that blocks **this** process, or None.

    None when no window is open, and None when the open one is ours -- the holder token comes
    from the environment the announcing command exported, so every child of that command is the
    holder too and a seat is never blocked by its own slot.
    """
    held = read_window(path)
    if held is None:
        return None
    token = os.environ.get(WINDOW_HOLDER_ENV, "") if holder is None else holder
    if token and token == held.nonce:
        return None
    return held


def mark_items_for_a_foreign_window(items: Sequence[Any], names: Sequence[str]) -> BoxWindow | None:
    """Mark every collected item from a named file as skipped while another seat holds the window.

    The decision lives here rather than in ``conftest`` so the suite's hook and any nested run
    that has to prove the hook fires are the same implementation, not two that agree today.

    Skipped rather than refused. A refusal makes the whole suite red for somebody who has done
    nothing wrong, and the natural response to a red suite is to run it again -- which is the
    collision. A skip keeps the rest of the suite honest, says whose slot this is, and leaves
    nothing to rerun.
    """
    window = blocking_window()
    if window is None:
        return None

    import pytest

    marker = pytest.mark.skip(
        reason=(
            f"the box window is held by {window.describe()}; this file can reach MLX at some "
            "scope and would refuse that seat's launch (issue 95). Run it after the end line."
        )
    )
    wanted = set(names)
    for item in items:
        if Path(str(item.fspath)).name in wanted:
            item.add_marker(marker)
    return window


def refusal_for_window(window: BoxWindow) -> str:
    """Why a launch is refused while somebody else holds the slot."""
    overdue = (
        "\nThis window is past its expected end. It is reported, not removed: clearing another\n"
        "seat's state is how a mechanism stops being one. Ask that seat to close it."
        if window.overdue
        else ""
    )
    return (
        "refusing to load a model: another seat holds the box window.\n"
        f"  {window.describe()}\n"
        "A window is a promise about the next minutes, which the live process inventory cannot\n"
        f"see. Wait for its end line.{overdue}"
    )


# --------------------------------------------------------------------------- reading a lock


@dataclass(frozen=True)
class LockHeld:
    """One lock file as it reads on disk, including one this module did not write.

    A lock is readable even when its JSON is truncated or foreign: ``raw`` keeps the bytes so a
    refusal can quote what it actually found rather than claim the file is empty.
    """

    path: Path
    session: str
    command: str
    started: str
    pid: int | None
    age_seconds: float | None
    nonce: str | None
    raw: str

    @property
    def holder_alive(self) -> bool:
        """Whether the recorded pid still exists. Unknown pid reads as not alive."""
        if self.pid is None or self.pid <= 0:
            return False
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:  # exists, owned by someone else
            return True
        except OSError:  # pragma: no cover - defensive
            return True
        return True

    @property
    def stale(self) -> bool:
        """A lock whose holder is gone, or that outlived the longest run this machine does."""
        if self.pid is not None and not self.holder_alive:
            return True
        return self.age_seconds is not None and self.age_seconds > STALE_AFTER_SECONDS

    def describe(self) -> str:
        """The holder block every refusal prints: pid and command line first."""
        alive = "alive" if self.holder_alive else "not alive"
        pid = "unknown" if self.pid is None else f"{self.pid} ({alive})"
        lines = [
            f"  holder pid     : {pid}",
            f"  holder command : {self.command}",
            f"  holder session : {self.session}",
            f"  taken          : {self.started} ({_duration(self.age_seconds)} ago)",
            f"  lock file      : {self.path}",
        ]
        return "\n".join(lines)


def read_lock(path: Path | None = None, *, now: float | None = None) -> LockHeld | None:
    """Read the lock at ``path``; ``None`` when there is none.

    Never raises on a malformed file: an unreadable holder still has to be reported, and the
    file's mtime dates it when the recorded timestamp cannot be parsed.
    """
    target = Path(path) if path is not None else default_lock_path()
    try:
        raw = target.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None
    except OSError as error:  # pragma: no cover - unreadable but present
        raw = f"<unreadable: {error}>"

    payload: dict[str, Any] = {}
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        parsed = None
    if isinstance(parsed, dict):
        payload = parsed

    pid = payload.get("pid")
    pid = pid if isinstance(pid, int) else None
    started = str(payload.get("started", "")) or "unknown"

    clock = time.time() if now is None else now
    age: float | None = None
    epoch = payload.get("started_epoch")
    if isinstance(epoch, (int, float)):
        age = max(0.0, clock - float(epoch))
    else:
        try:
            age = max(0.0, clock - target.stat().st_mtime)
        except OSError:  # pragma: no cover - vanished between read and stat
            age = None

    nonce = payload.get("nonce")
    return LockHeld(
        path=target,
        session=str(payload.get("session", "")) or "unknown",
        command=str(payload.get("command", "")) or "unknown",
        started=started,
        pid=pid,
        age_seconds=age,
        nonce=nonce if isinstance(nonce, str) else None,
        raw=raw,
    )


# ------------------------------------------------------------------ the process-list bridge


#: What the last process check managed to do, for the lock payload. Fail-open is right -- the
#: lock file is the mechanism and a check that cannot run must not block a launch -- but a
#: launch that skipped the check silently, and then collided with a scratch script holding no
#: lock, is a mystery with nothing to read. So the failure is announced on stderr and written
#: into the lock, where the next person to look already is.
#:
#: Cleared and read inside one ``_acquire``, which a launcher reaches once, on its main thread.
_probe_failures: list[str] = []


def _probe_failed(command: Sequence[str], reason: BaseException) -> None:
    """Record a probe that could not run, and say so on stderr rather than swallowing it."""
    detail = f"{shlex.join(str(part) for part in command)}: {type(reason).__name__}: {reason}"
    _probe_failures.append(detail)
    print(
        f"model-run lock: the process check could not run ({detail}). Continuing without it, "
        "because the lock file is the mechanism and a check that cannot run must not block a "
        "launch -- but a process holding MLX without a lock will not be caught this time. "
        "Recorded as process_check in the lock file.",
        file=sys.stderr,
    )


def _process_table() -> list[tuple[int, int, int, str]]:
    """``(pid, ppid, pgid, command)`` for every process, or an empty table.

    ``ps`` rather than ``pgrep``: the parent and group columns are what make the self-match
    exclusion possible at all, and ``pgrep`` does not report them. The command column is for the
    refusal message only -- it is never matched against, since a name match is exactly what the
    library check replaces.

    Through ``spawn.run``, never ``subprocess`` directly. This is the sharpest case in the
    repository for that rule: ``hold_model_run_lock`` runs inside processes that have already
    imported MLX — ``stage_train`` imports ``mlx_lm.lora`` before it resolves the base — so a
    forking ``ps`` here would abort the very run the lock exists to protect, and it would do it
    from inside the guard.

    Fail-open on any error: the lock file is the mechanism and this check is a bridge for runs
    started before it existed, so a check that cannot run must not become a second way to
    refuse a launch.
    """
    try:
        completed = spawn_run(
            list(_PS_COMMAND),
            capture_output=True,
            text=True,
            check=False,
            timeout=_PS_TIMEOUT,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        _probe_failed(_PS_COMMAND, error)
        return []
    return _parse_process_table(completed.stdout)


def _parse_process_table(text: str) -> list[tuple[int, int, int, str]]:
    """Parse ``ps -Ao pid=,ppid=,pgid=,command=`` output; unparseable rows are dropped."""
    rows: list[tuple[int, int, int, str]] = []
    for line in text.splitlines():
        parts = line.split(maxsplit=3)
        if len(parts) < 4:
            continue
        try:
            pid, ppid, pgid = (int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError:
            continue
        rows.append((pid, ppid, pgid, parts[3]))
    return rows


def _own_lineage(rows: Sequence[tuple[int, int, int, str]], pid: int) -> set[int]:
    """This process and every ancestor of it, bounded so a cycle cannot hang the walk."""
    parents = {row[0]: row[1] for row in rows}
    lineage: set[int] = set()
    current = pid
    for _ in range(_LINEAGE_HOPS):
        if current <= 0 or current in lineage:
            break
        lineage.add(current)
        current = parents.get(current, 0)
    return lineage


@dataclass(frozen=True)
class MappedProcess:
    """A process the kernel says has the MLX library mapped."""

    pid: int
    command: str
    library: str

    def describe(self) -> str:
        return f"  pid {self.pid:<7} {self.command}\n            maps {self.library}"


def _parse_lsof_map(text: str, library: str) -> dict[int, str]:
    """Parse ``lsof -F pn`` output into ``{pid: mapped path}`` for one library basename.

    ``-F`` output is one field per line, ``p<pid>`` starting a process block and ``n<path>``
    naming each of its files, so the current pid is whatever ``p`` line came last.
    """
    found: dict[int, str] = {}
    current: int | None = None
    for line in text.splitlines():
        if not line:
            continue
        tag, value = line[0], line[1:]
        if tag == "p":
            try:
                current = int(value)
            except ValueError:
                current = None
        elif tag == "n" and current is not None and value.rpartition("/")[2] == library:
            found[current] = value
    return found


def _mapped_pids(library: str = MLX_LIBRARY) -> dict[int, str]:
    """``{pid: path}`` for every process with ``library`` mapped, straight from the kernel.

    ``-d txt`` restricts the walk to mapped text files, which is where a loaded ``.dylib``
    appears and which keeps the call around 0.4 s rather than several seconds. Fail-open on any
    error, for the same reason as ``_process_table``.
    """
    try:
        completed = spawn_run(
            list(_LSOF_COMMAND),
            capture_output=True,
            text=True,
            check=False,
            timeout=_LSOF_TIMEOUT,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        _probe_failed(_LSOF_COMMAND, error)
        return {}
    return _parse_lsof_map(completed.stdout, library)


def running_model_processes(
    mapped: dict[int, str] | None = None,
    rows: Sequence[tuple[int, int, int, str]] | None = None,
    *,
    pid: int | None = None,
    library: str = MLX_LIBRARY,
) -> list[MappedProcess]:
    """Processes outside this one that hold MLX, whatever they are called.

    Excluding the caller's own pid is not enough and excluding pid plus parent is still not
    enough: the wrapping shell, ``uv run`` and ``python -m`` sit on different pids, and a stage
    that imported ``mlx_lm.lora`` before resolving its base has the library mapped in *this*
    process. Two exclusions together close that: the caller's whole ancestry, and everything in
    the caller's process group (its own children, and the siblings of a job the shell started).
    What is left is somebody else's model.

    ``lsof`` runs before ``ps`` so that a process which exits in between is dropped rather than
    reported: a refusal naming a dead pid is a refusal nobody can act on.

    Both probes are injectable so the filter is testable without a machine full of processes.
    """
    holders = dict(mapped) if mapped is not None else _mapped_pids(library)
    if not holders:
        return []
    table = list(rows) if rows is not None else _process_table()
    me = os.getpid() if pid is None else pid
    lineage = _own_lineage(table, me)
    own_group = {row[2] for row in table if row[0] == me}
    commands = {row[0]: row[3] for row in table}

    found: list[MappedProcess] = []
    for holder_pid, path in sorted(holders.items()):
        if holder_pid in lineage:
            continue
        command = commands.get(holder_pid)
        if command is None:
            # `ps` ran second and no longer lists it, so it has exited.
            continue
        if any(row[2] in own_group for row in table if row[0] == holder_pid):
            continue
        found.append(MappedProcess(pid=holder_pid, command=command, library=path))
    return found


# ------------------------------------------------------------------- acquiring and releasing


@dataclass(frozen=True)
class _Handle:
    """A lock this process took: where it is, and the nonce proving it is still ours."""

    path: Path
    nonce: str
    owner_pid: int


def _refusal_for_lock(held: LockHeld) -> str:
    if held.stale:
        why = []
        if held.pid is not None and not held.holder_alive:
            why.append(f"its pid {held.pid} is not alive")
        if held.age_seconds is not None and held.age_seconds > STALE_AFTER_SECONDS:
            why.append(
                f"its age {_duration(held.age_seconds)} is past the "
                f"{_duration(STALE_AFTER_SECONDS)} staleness threshold"
            )
        reason = " and ".join(why) if why else "it cannot belong to a live run"
        tail = (
            f"This lock looks stale because {reason}, and it is NOT removed automatically: a\n"
            "launcher that clears the locks it finds inconvenient is a launcher with no lock.\n"
            "Report it to the Chief and let a person decide whether to remove it."
        )
    else:
        tail = (
            "Wait for that run to finish. Do not delete the lock to get past it; if you believe\n"
            "the holder above is gone, report it to the Chief rather than clearing it yourself."
        )
    return (
        "refusing to load a model: another model-loading run holds this machine.\n"
        f"{held.describe()}\n{tail}"
    )


def refusal_for_processes(found: Sequence[MappedProcess]) -> str:
    """One wording for "somebody else holds MLX", shared with ``probes.guard``.

    Public because the probe CLIs refuse on the same evidence as the lock does, and two
    wordings for one condition is how two definitions of it start.
    """
    listed = "\n".join(holder.describe() for holder in found)
    return (
        "refusing to load a model: another process on this machine already holds MLX.\n"
        f"{listed}\n"
        "The kernel says these have the MLX library mapped. They may hold no lock at all -- a\n"
        "scratch script or a notebook outside this package never takes one -- so there is\n"
        "nothing to clear here. Wait for the process above to exit, or stop it deliberately."
    )


def _write_lock(path: Path, payload: dict[str, Any]) -> None:
    """Create the lock exclusively, or raise ``FileExistsError``.

    ``O_CREAT | O_EXCL`` is the whole point: the kernel decides the winner, so two launchers
    racing between a check and a write cannot both proceed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(fd, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def _acquire(
    *,
    command: str | None,
    session: str | None,
    path: Path | None,
    check_processes: bool,
) -> _Handle:
    target = Path(path) if path is not None else default_lock_path()
    process_check = "skipped"
    if check_processes:
        _probe_failures.clear()
        # The window first, because it answers the question the inventory cannot: a seat may hold
        # the next twenty minutes without holding a process this instant, and a launch into
        # somebody else's slot is the failure this whole pair exists to prevent (issue 95).
        window = blocking_window()
        if window is not None:
            raise RunLockBusy(refusal_for_window(window))
        found = running_model_processes()
        if found:
            raise RunLockBusy(refusal_for_processes(found))
        process_check = "ok" if not _probe_failures else "failed: " + "; ".join(_probe_failures)

    nonce = uuid.uuid4().hex
    now = time.time()
    payload = {
        "session": session if session is not None else _default_session(),
        "command": command if command is not None else _default_command(),
        "started": _utc_now(),
        "started_epoch": now,
        "pid": os.getpid(),
        "nonce": nonce,
        # Whether the check for processes holding MLX outside this package actually ran. A
        # holder that took no lock is invisible to the lock file otherwise, so a later
        # collision is diagnosable from here rather than guessed at.
        "process_check": process_check,
    }
    try:
        _write_lock(target, payload)
    except FileExistsError:
        held = read_lock(target, now=now)
        if held is None:
            # Removed between the failed create and the read; the machine is free again, but
            # this launch does not retry silently — a launcher that loops is a launcher that
            # queues, and the clause says refuse and report.
            raise RunLockBusy(
                "refusing to load a model: the lock at "
                f"{target} was taken and released while this launch was starting. "
                "Nothing is holding it now; run the command again."
            ) from None
        raise RunLockBusy(_refusal_for_lock(held)) from None
    return _Handle(path=target, nonce=nonce, owner_pid=os.getpid())


def _release(handle: _Handle) -> None:
    """Remove the lock, but only when it is still the one this process wrote.

    The nonce check means a release cannot delete a lock a person put there after a manual
    intervention, and cannot delete a successor's lock if this process's own was cleared.
    """
    if os.getpid() != handle.owner_pid:
        # A forked child inherits the module state but not the ownership; releasing here would
        # free the parent's lock while the parent still holds the model.
        return
    held = read_lock(handle.path)
    if held is None:
        return
    if held.nonce is not None and held.nonce != handle.nonce:
        print(
            f"model-run lock at {handle.path} was replaced by another holder; leaving it in "
            "place.\n" + held.describe(),
            file=sys.stderr,
        )
        return
    try:
        os.unlink(handle.path)
    except FileNotFoundError:
        return
    except OSError as error:  # pragma: no cover - defensive
        print(f"could not remove the model-run lock at {handle.path}: {error}", file=sys.stderr)


@contextmanager
def model_run_lock(
    *,
    command: str | None = None,
    session: str | None = None,
    path: Path | None = None,
    check_processes: bool = True,
) -> Iterator[Path]:
    """Hold the model-run lock for the body, releasing it on every exit path.

    The explicit-scope form. Use it when the caller owns a region that begins before the load
    and ends after the model is gone; ``load_weights`` uses the process-scoped form instead,
    because a model loaded through it stays resident until the process exits.
    """
    handle = _acquire(command=command, session=session, path=path, check_processes=check_processes)
    try:
        yield handle.path
    finally:
        _release(handle)


# ------------------------------------------------------------------- the process-scoped hold

_held: _Handle | None = None
_held_guard = threading.Lock()


def _release_held() -> None:
    """Release the process-scoped hold. Idempotent, and safe to call from a signal handler."""
    global _held
    handle = _held
    if handle is None:
        return
    _held = None
    _release(handle)


def _install_signal_release() -> None:
    """Release the hold on ``SIGTERM``/``SIGHUP``, which ``atexit`` never sees.

    Only signals still on their default disposition are taken, so this never displaces a
    handler the application installed; ``SIGINT`` is left alone because it already raises
    ``KeyboardInterrupt`` and unwinds into ``atexit``.
    """
    if threading.current_thread() is not threading.main_thread():
        return
    for signum in (signal.SIGTERM, signal.SIGHUP):
        try:
            previous = signal.getsignal(signum)
        except (ValueError, OSError):  # pragma: no cover - platform without the signal
            continue
        if previous is not signal.SIG_DFL:
            continue

        def handler(sig: int, _frame: Any) -> None:
            _release_held()
            signal.signal(sig, signal.SIG_DFL)
            os.kill(os.getpid(), sig)

        try:
            signal.signal(signum, handler)
        except (ValueError, OSError):  # pragma: no cover - not the main thread of a process
            continue


def hold_model_run_lock(
    *,
    command: str | None = None,
    session: str | None = None,
    path: Path | None = None,
    check_processes: bool = True,
) -> Path:
    """Take the lock for the rest of this process, and return where it was taken.

    Idempotent: a stage that loads a base model and then an adapted one takes the lock once.
    The release runs at interpreter exit — after a normal return, a ``SystemExit``, an uncaught
    exception or a ``KeyboardInterrupt`` — and on ``SIGTERM``/``SIGHUP``.
    """
    global _held
    with _held_guard:
        if _held is not None:
            target = Path(path) if path is not None else default_lock_path()
            if target != _held.path:
                raise RunLockError(
                    f"this process already holds the model-run lock at {_held.path}; "
                    f"it cannot also take one at {target}"
                )
            return _held.path
        handle = _acquire(
            command=command, session=session, path=path, check_processes=check_processes
        )
        _held = handle
        atexit.register(_release_held)
        _install_signal_release()
        return handle.path


# ---------------------------------------------------------------------------- the only door


def load_weights(hf_id: str, **kwargs: Any) -> tuple[Any, Any]:
    """Load model weights under the model-run lock. The only door to ``mlx_lm.load``.

    Wiring every entry point by hand is a guard the next one forgets, so the acquire sits at the
    seam instead: ``tests/test_repository_rules.py`` fails if anything under ``pipeline/`` or
    ``probes/`` imports ``mlx_lm``'s loader itself. The lock is taken *before* the import, so a
    refusal costs no weights and no download.
    """
    hold_model_run_lock()
    from mlx_lm import load

    return load(hf_id, **kwargs)


def _window_cli(argv: Sequence[str] | None = None) -> int:
    """``python -m local_llm_lab.runlock announce|end|status`` (issue 95).

    A shell entry point, because the seats that need windows drive the box from shells and
    wrappers rather than from Python. ``announce`` prints the export line to eval, so the token
    reaches every child of the launching command and the holder is never blocked by its own slot:

        eval "$(python -m local_llm_lab.runlock announce --seat deputy --purpose '88 block 1' \
                --minutes 100)"
        ... the launch ...
        python -m local_llm_lab.runlock end
    """
    import argparse

    parser = argparse.ArgumentParser(prog="runlock", description="the box window")
    sub = parser.add_subparsers(dest="action", required=True)
    opening = sub.add_parser("announce")
    opening.add_argument("--seat", required=True)
    opening.add_argument("--purpose", required=True)
    opening.add_argument("--minutes", type=float, required=True)
    sub.add_parser("end")
    sub.add_parser("status")
    args = parser.parse_args(argv)

    if args.action == "announce":
        nonce = announce_window(args.seat, args.purpose, args.minutes)
        print(f"export {WINDOW_HOLDER_ENV}={nonce}")
        return 0
    if args.action == "end":
        token = os.environ.get(WINDOW_HOLDER_ENV, "")
        if not token:
            print(f"{WINDOW_HOLDER_ENV} is not set; a window is closed by the seat that opened it")
            return 1
        if not end_window(token):
            held = read_window()
            print(
                "no window of this seat's to close"
                if held is None
                else f"not ours: {held.describe()}"
            )
            return 1
        print("window closed")
        return 0
    held = read_window()
    print("no window is open" if held is None else held.describe())
    return 0


if __name__ == "__main__":  # pragma: no cover - a shell entry point
    sys.exit(_window_cli())
