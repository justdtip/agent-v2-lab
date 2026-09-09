"""Gate records, so a paid hour that dies at gate five does not re-run gates one to four.

Resume is keyed on **what was measured**, never on time. A record counts only when the source
commit, the checkpoint and the device all match the run asking for it. Anything else is a
number about a different thing, and a kit that reused it would report a gate as passed on
evidence from another tree.

The refusal names the field that differs. "Not resumable" tells an operator nothing at minute
fifty of a rented hour; "gate 3's record was taken on commit abc1234 and you are on def5678"
tells them whether to re-run it or to check out the other commit.

**A record taken on a modified working tree is never reused.** The commit does not describe
the code in that case, so the key does not identify what ran. That is stricter than the ruling
asked for and it is the same argument: a key that can be satisfied by two different trees is
not a key.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Identity", "GateRecords", "current_identity"]


@dataclass(frozen=True)
class Identity:
    """What a gate's number is a number about."""

    source_commit: str
    tree_dirty: bool | None
    checkpoint_sha256: str
    device: str

    def differences(self, other: Identity) -> list[str]:
        """Which fields disagree, named, so a refusal can say what to do about it."""
        differences = []
        if self.source_commit != other.source_commit:
            differences.append(
                f"source commit {other.source_commit[:7]} against {self.source_commit[:7]}"
            )
        if self.checkpoint_sha256 != other.checkpoint_sha256:
            differences.append(
                f"checkpoint {other.checkpoint_sha256[:12]} against {self.checkpoint_sha256[:12]}"
            )
        if self.device != other.device:
            differences.append(f"device {other.device} against {self.device}")
        return differences

    @property
    def usable(self) -> tuple[bool, str]:
        """Whether a record carrying this identity may ever be reused."""
        if self.tree_dirty:
            return False, "the working tree was modified, so the commit does not describe it"
        if self.tree_dirty is None:
            return False, "the working tree's state could not be read, so the commit is unbacked"
        if self.source_commit in ("", "unknown"):
            return False, "the source commit could not be read"
        return True, ""


def current_identity(checkpoint: Path | str | None = None) -> Identity:
    """Read this run's identity from git, the checkpoint's headers and the device."""
    from local_llm_lab import device as device_module
    from local_llm_lab.pipeline.integrity import git_tree_dirty
    from local_llm_lab.runlog import git_commit

    checkpoint_sha = "none"
    if checkpoint is not None:
        from local_llm_lab.hf_text import checkpoint_metadata

        hashes = checkpoint_metadata(checkpoint)["sha256"]
        # One value over every file's digest, so a changed shard changes the key.
        checkpoint_sha = json.dumps(hashes, sort_keys=True)

    reading = device_module.describe()
    return Identity(
        source_commit=git_commit(),
        tree_dirty=git_tree_dirty(),
        checkpoint_sha256=checkpoint_sha,
        device=json.dumps(
            {key: reading.get(key) for key in ("backend", "device", "determinism", "pinned")},
            sort_keys=True,
        ),
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
        "a record I will not use, and here is why" are different facts to an operator.
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

        recorded = Identity(**stored["identity"])
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
