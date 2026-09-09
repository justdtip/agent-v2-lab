"""Gate records, so a paid hour that dies at gate five does not re-run gates one to four.

Resume is keyed on **what was measured**, never on time. A record counts only when the tree
that ran, the checkpoint, the device and the gate's own input all match the run asking for it.
Anything else is a number about a different thing, and a kit that reused it would report a gate
as passed on evidence from somewhere else.

**The key describes the tree, rather than refusing a tree it cannot describe.** An earlier
version of this refused every record whenever the working tree was modified, on the argument
that a commit does not describe a modified tree. The argument was right and the fix was wrong:
bugs are expected to be resolved on the device, so a one-line edit at minute fifty is the
normal case, and voiding every prior record for it recreates exactly the waste resume exists to
prevent. So the key is the tree's *content*: ``git stash create`` yields a commit for the
working tree without touching anything, and its tree hash is the tree that ran. A modified tree
resumes records taken on that same modified tree and nothing else. The record carries the diff
stat beside it for the human, who wants to know what was edited and not only that something
was.

**The gate's input is part of the key**, which is what lets a smoke pass and a full pass share
one store. A smoke record names its reduced input, so a full pass cannot resume past it by
accident, because the keys differ; and a gate whose smallest input *is* its full input resumes
correctly, because they are the same. That answers the question by construction, and avoids a
second store's own failure mode, the operator who forgets which one they are reading.

The refusal names every field that differs, not the first. "Not resumable" tells an operator
nothing at minute fifty of a rented hour, and fixing one field should not reveal the next.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from subprocess import SubprocessError
from typing import Any

__all__ = ["Identity", "GateRecords", "current_identity", "input_digest", "working_tree_hash"]

_GIT_TIMEOUT = 10.0

#: Fields the resume key is computed over. Everything else on the identity is for the reader.
_KEYED = ("tree_hash", "checkpoint_sha256", "device", "input_digest")


def _git(*arguments: str, cwd: Path | None = None) -> str | None:
    """``git`` through the vetted spawn path, returning None on any failure.

    ``git -C`` rather than ``subprocess``'s ``cwd``: CPython takes ``posix_spawn`` only when
    ``cwd`` is None, and a fork beside an initialised Metal context aborts the interpreter.
    """
    from local_llm_lab import spawn

    argv = ["git", *(("-C", str(cwd)) if cwd is not None else ()), *arguments]
    try:
        completed = spawn.run(
            argv, capture_output=True, text=True, check=False, timeout=_GIT_TIMEOUT
        )
    except (OSError, ValueError, spawn.UnsafeSpawnError, SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def working_tree_hash(cwd: Path | None = None) -> tuple[str, str]:
    """The tree that actually ran, and the diff stat describing it.

    Returns ``("unknown", "")`` on any failure, so a run without git degrades to "resume
    nothing" rather than to "resume everything".

    ``git stash create`` writes a commit object for the working tree and changes nothing about
    the index, the worktree or the stash list. Its tree is the tree that ran. On a clean tree it
    prints nothing, and ``HEAD``'s tree is the answer.

    Untracked files are folded in separately, because ``stash create`` does not carry them and a
    new module changes behaviour without changing a tracked byte.
    """
    created = _git("stash", "create", cwd=cwd)
    if created is None:
        return "unknown", ""
    base = created or "HEAD"
    tree = _git("rev-parse", f"{base}^{{tree}}", cwd=cwd)
    if tree is None:
        return "unknown", ""

    untracked = _git("ls-files", "--others", "--exclude-standard", cwd=cwd)
    digest = hashlib.sha256(tree.encode())
    if untracked:
        for path in sorted(untracked.splitlines()):
            blob = _git("hash-object", "--", path, cwd=cwd)
            digest.update(f"{path}:{blob or 'unreadable'}".encode())
    stat = _git("diff", "HEAD", "--stat", cwd=cwd) or ""
    if untracked:
        stat = f"{stat}\n{len(untracked.splitlines())} untracked file(s) folded into the key"
    return digest.hexdigest(), stat.strip()


def input_digest(**inputs: Any) -> str:
    """A stable digest of whatever a gate consumed: episode ids, token counts, layers.

    Passed into the identity so a smoke record and a full record are different keys rather than
    the same key with a flag on it.
    """
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Identity:
    """What a gate's number is a number about."""

    tree_hash: str
    checkpoint_sha256: str
    device: str
    input_digest: str = "none"
    #: For the reader, never for the key: two runs on the same tree agree whatever these say.
    source_commit: str = "unknown"
    diff_stat: str = ""
    smoke: bool = False

    def differences(self, other: Identity) -> list[str]:
        """Which keyed fields disagree, named, so a refusal can say what to do about it."""
        labels = {
            "tree_hash": "working tree",
            "checkpoint_sha256": "checkpoint",
            "device": "device",
            "input_digest": "gate input",
        }
        return [
            f"{labels[name]} {getattr(other, name)[:12]} against {getattr(self, name)[:12]}"
            for name in _KEYED
            if getattr(self, name) != getattr(other, name)
        ]

    @property
    def usable(self) -> tuple[bool, str]:
        """Whether a record carrying this identity may ever be reused."""
        if self.tree_hash in ("", "unknown"):
            return False, "the working tree's content could not be read, so nothing identifies it"
        return True, ""


def current_identity(
    checkpoint: Path | str | None = None, *, smoke: bool = False, **inputs: Any
) -> Identity:
    """Read this run's identity from the working tree, the checkpoint headers and the device."""
    from local_llm_lab import device as device_module
    from local_llm_lab.runlog import git_commit

    root = Path(__file__).resolve().parents[2]
    tree, stat = working_tree_hash(root)

    checkpoint_sha = "none"
    if checkpoint is not None:
        from local_llm_lab.hf_text import checkpoint_metadata

        hashes = checkpoint_metadata(checkpoint)["sha256"]
        checkpoint_sha = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()

    reading = device_module.describe()
    return Identity(
        tree_hash=tree,
        checkpoint_sha256=checkpoint_sha,
        device=json.dumps(
            {key: reading.get(key) for key in ("backend", "device", "determinism", "pinned")},
            sort_keys=True,
        ),
        input_digest=input_digest(smoke=smoke, **inputs) if inputs or smoke else "none",
        source_commit=git_commit(root),
        diff_stat=stat,
        smoke=smoke,
    )


@dataclass
class GateRecords:
    """Gate results on disk, written as each completes and reused only when they match."""

    directory: Path
    identity: Identity
    refusals: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, gate: int) -> Path:
        return self.directory / f"gate-{gate:02d}.json"

    def completed(self, gate: int) -> dict[str, Any] | None:
        """A previous result for this gate, or None with the reason recorded in ``refusals``.

        Every path out of here that returns None also writes a reason, because "no record" and
        "a record I will not use, and here is why" are different facts to an operator watching
        a rented clock. A gate with no record at all is not a refusal and gets no explanation.
        """
        path = self._path(gate)
        if not path.exists():
            return None
        try:
            stored = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            self.refusals[gate] = f"the record could not be read: {error}"
            return None

        usable, why = self.identity.usable
        if not usable:
            self.refusals[gate] = f"this run cannot resume anything: {why}"
            return None

        known = {item.name for item in fields(Identity)}
        recorded = Identity(**{k: v for k, v in stored.get("identity", {}).items() if k in known})
        was_usable, why = recorded.usable
        if not was_usable:
            self.refusals[gate] = f"the record is not reusable: {why}"
            return None

        differences = self.identity.differences(recorded)
        if differences:
            self.refusals[gate] = "the record measured something else: " + "; ".join(differences)
            return None
        return stored

    def write(self, gate: int, result: dict[str, Any]) -> Path:
        """Write one gate's record the moment it completes, and flush it."""
        path = self._path(gate)
        payload = {"gate": gate, "identity": asdict(self.identity), **result}
        with path.open("w") as stream:
            json.dump(payload, stream, indent=1, sort_keys=True)
            stream.write("\n")
            stream.flush()
        return path
