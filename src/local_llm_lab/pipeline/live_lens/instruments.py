"""Immutable, hash-identified instruments; no model runtime is imported here."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


#: The npz entry a stamped lens carries. Not a ``J<l>`` map, so ``LensMaps.load`` names it
#: explicitly rather than rejecting it as an invalid map name.
LENS_IDENTITY_KEY = "identity"


class LensIdentityError(ValueError):
    """A lens does not say which model it was fitted on, or says a different one (issue 99)."""


@dataclass(frozen=True)
class LensIdentity:
    """Which model a lens was fitted against.

    Three fields because two are not enough to separate the models this repository actually
    holds. Qwen3.5-4B and Gemma 3 4B are **both 2560-dimensional**, so hidden size cannot tell
    them apart; the Qwen lens covers layers 1 to 31 and Gemma has 34, so `1 <= 31 < 34` passes
    every existing check and the wrong lens loads in silence. The layer count catches the
    reverse direction by arithmetic and misses this one, which is worse than no guard, because
    the direction it catches is the one nobody would take.
    """

    name: str
    hf_id: str
    num_layers: int

    def as_dict(self) -> dict:
        return {"name": self.name, "hf_id": self.hf_id, "num_layers": int(self.num_layers)}

    @classmethod
    def from_dict(cls, value: object) -> LensIdentity:
        if not isinstance(value, dict):
            raise LensIdentityError("lens identity must be a mapping")
        try:
            name, hf_id, layers = value["name"], value["hf_id"], value["num_layers"]
        except (KeyError, TypeError) as error:
            raise LensIdentityError(
                "lens identity must carry name, hf_id and num_layers"
            ) from error
        if not isinstance(name, str) or not isinstance(hf_id, str):
            raise LensIdentityError("lens identity name and hf_id must be strings")
        if isinstance(layers, bool) or not isinstance(layers, int) or layers < 1:
            raise LensIdentityError("lens identity num_layers must be a positive integer")
        return cls(name, hf_id, layers)

    def describe(self) -> str:
        return f"{self.name} ({self.hf_id}, {self.num_layers} layers)"


def _stamped_identity(path: Path, archive, sha: str) -> LensIdentity:
    """The identity the file itself carries, from the archive or from its digest-bound sidecar.

    Two places, and the precedence is not a matter of taste. Inside the archive is stronger,
    because the digest the caller already checks covers it, and that is where every lens written
    from now on carries it. The sidecar exists for the two hosted lenses that were converted
    before this field did: stamping them inside the npz would change bytes whose digest is
    pinned in the pilot script and in a published record, so the stamp goes beside the file and
    is bound to it by the sidecar's own ``npz_sha256``. A sidecar that names a different file is
    not evidence about this one.
    """
    if LENS_IDENTITY_KEY in archive.files:
        raw = archive[LENS_IDENTITY_KEY]
        return LensIdentity.from_dict(json.loads(bytes(raw).decode("utf-8")))
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        raise LensIdentityError(
            f"{path.name} carries no model identity and has no sidecar at {sidecar.name}. "
            "A lens that does not say which model it was fitted on is refused, not warned "
            "about: Qwen3.5-4B and Gemma 3 4B are both 2560-dimensional and the wrong lens "
            "loads in silence. Stamp it with scripts/stamp_lens_identity.py."
        )
    meta = json.loads(sidecar.read_text())
    if not isinstance(meta, dict) or "model" not in meta:
        raise LensIdentityError(
            f"{sidecar.name} carries no 'model' block. Stamp it with "
            "scripts/stamp_lens_identity.py, which records the evidence for a retroactive stamp."
        )
    recorded = meta.get("npz_sha256")
    if recorded != sha:
        raise LensIdentityError(
            f"{sidecar.name} records npz_sha256 {recorded}, but {path.name} hashes to {sha}. "
            "The sidecar describes a different file, so its identity is not evidence about "
            "this one."
        )
    return LensIdentity.from_dict(meta["model"])


@dataclass(frozen=True)
class FrozenPopulation:
    source_sha256: str
    primary: tuple[tuple[int, int], ...]
    sensitivity: tuple[tuple[int, int], ...]
    floor: float = 0.1

    @classmethod
    def from_selection(cls, path: Path, blocks: tuple[int, ...]) -> FrozenPopulation:
        raw = path.read_bytes()
        rows = [r for r in json.loads(raw)["selected_attention"] if r["block"] in blocks]
        if any(
            r["layer_written"] != r["block"] + 1
            or not 0 <= r["head"] < 16
            or not np.isfinite(r["J_mrr"])
            for r in rows
        ):
            raise ValueError("invalid selection row")
        secondary = tuple(sorted((r["block"], r["head"]) for r in rows))
        primary = tuple(sorted((r["block"], r["head"]) for r in rows if r["J_mrr"] >= 0.1))
        if len(set(secondary)) != len(secondary):
            raise ValueError("duplicate head in selection")
        if len(primary) != 23 or len(secondary) != 26:
            raise ValueError("record does not yield the ruled 23 primary / 26 sensitivity heads")
        return cls(hashlib.sha256(raw).hexdigest(), primary, secondary)

    @property
    def sha256(self) -> str:
        return _digest(asdict(self))

    def write(self, path: Path) -> None:
        """Exclusive create: never replace membership after any results have been read."""
        payload = asdict(self) | {"population_sha256": self.sha256, "schema_version": 1}
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")

    @classmethod
    def read(cls, path: Path) -> FrozenPopulation:
        data = json.loads(path.read_text())
        population = cls(
            data["source_sha256"],
            tuple(map(tuple, data["primary"])),
            tuple(map(tuple, data["sensitivity"])),
            data["floor"],
        )
        if data["schema_version"] != 1 or population.sha256 != data["population_sha256"]:
            raise ValueError("population manifest hash mismatch")
        if (
            population.floor != 0.1
            or len(population.primary) != 23
            or len(population.sensitivity) != 26
            or not set(population.primary) <= set(population.sensitivity)
        ):
            raise ValueError("population differs from the ruled selection")
        return population


def read_band(registry_path: Path, layer_kinds: list[str]) -> tuple[tuple[int, int], ...]:
    data = yaml.safe_load(registry_path.read_text())
    pairs = tuple(tuple(pair) for pair in data["probes"]["live_lens_pairs"])
    if not pairs or len({p for pair in pairs for p in pair}) != 2 * len(pairs):
        raise ValueError("band pairs must be nonempty and disjoint")
    for pair in pairs:
        if len(pair) != 2 or any(
            type(p) is not int or not 1 <= p <= len(layer_kinds) for p in pair
        ):
            raise ValueError("invalid residual indices in registry band")
        if abs(pair[1] - pair[0]) != 1 or {layer_kinds[p - 1] for p in pair} != {
            "attention",
            "linear_attention",
        }:
            raise ValueError("band pair must join adjacent opposite layer kinds")
    return pairs


@dataclass(frozen=True)
class LensMaps:
    maps: dict[int, np.ndarray]
    sha256: str
    hidden_size: int
    num_layers: int
    identity: LensIdentity | None = None

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_sha256: str,
        hidden_size: int,
        num_layers: int,
        identity: LensIdentity,
    ) -> LensMaps:
        """Load a lens, refusing one fitted on a different model (issue 99).

        The digest check that was already here proves the file is the file the caller named. It
        cannot prove the caller named the right file, and the caller is a person or a script
        constant. ``identity`` is what the *view* says it is, and the lens has to agree.
        """
        sha = file_sha256(path)
        if sha != expected_sha256:
            raise ValueError("lens file hash mismatch")
        maps = {}
        with np.load(path, allow_pickle=False) as archive:
            stamped = _stamped_identity(path, archive, sha)
            if stamped != identity:
                raise LensIdentityError(
                    f"lens {path.name} was fitted on {stamped.describe()}, and it is being "
                    f"loaded against {identity.describe()}. Refusing: a lens of the right width "
                    "on the wrong model produces ranks that look exactly like a finding."
                )
            for name in archive.files:
                if name == LENS_IDENTITY_KEY:
                    continue
                if not name.startswith("J") or not name[1:].isdigit():
                    raise ValueError(f"invalid lens map name: {name}")
                layer = int(name[1:]) + 1
                a = np.array(archive[name], dtype=np.float32)
                if (
                    not 1 <= layer < num_layers
                    or a.shape != (hidden_size, hidden_size)
                    or not np.isfinite(a).all()
                ):
                    raise ValueError(f"invalid lens map at layer {layer}")
                a.flags.writeable = False
                maps[layer] = a
        if not maps:
            raise ValueError("lens archive is empty")
        return cls(maps, sha, hidden_size, num_layers, stamped)

    def apply(self, residual: np.ndarray, layer: int) -> np.ndarray:
        if residual.shape[-1] != self.hidden_size:
            raise ValueError("residual dimension does not match lens")
        return residual if layer == self.num_layers else residual @ self.maps[layer].T
