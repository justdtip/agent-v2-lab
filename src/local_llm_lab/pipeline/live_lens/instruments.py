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

    @classmethod
    def load(
        cls, path: Path, *, expected_sha256: str, hidden_size: int, num_layers: int
    ) -> LensMaps:
        sha = file_sha256(path)
        if sha != expected_sha256:
            raise ValueError("lens file hash mismatch")
        maps = {}
        with np.load(path, allow_pickle=False) as archive:
            for name in archive.files:
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
        return cls(maps, sha, hidden_size, num_layers)

    def apply(self, residual: np.ndarray, layer: int) -> np.ndarray:
        if residual.shape[-1] != self.hidden_size:
            raise ValueError("residual dimension does not match lens")
        return residual if layer == self.num_layers else residual @ self.maps[layer].T
