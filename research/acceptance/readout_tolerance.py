"""A readout tolerance read off a measured distribution, never chosen.

The readout gate needs a band: how far the ported readout may sit from the recorded one before
the difference is a defect rather than arithmetic. A chosen band is a number someone liked. A
measured band is the distribution of the CPU-float32-against-MLX-4-bit difference on the golden
episodes, and the remote's bfloat16 result is then checked against a projection from it.

Every band here carries the measurement it rests on, because a projection without its basis
cannot be learned from, only failed. That rule was written in this repository after a memory
projection missed by 38% and a rate projection by 2.3x, both calibrated on the shortest episode
in the set; the declaration said what it was measured on, so the miss could be named as a
method error instead of guessed at as bad luck.

The corollary is enforced by :func:`measure_band` refusing an empty sample and recording ``n``:
calibrate at the largest instance the run will reach, and say what the cost scales with.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Band:
    """A tolerance, the sample it came from, and what that sample was of."""

    value: float
    quantile: float
    n: int
    maximum: float
    median: float
    basis: str
    projected_from: Band | None = None

    @property
    def measured(self) -> bool:
        return self.projected_from is None

    def describe(self) -> str:
        kind = "measured" if self.measured else "projected"
        line = (
            f"{kind} band {self.value:.6g} at quantile {self.quantile:g} "
            f"from n={self.n}, median {self.median:.6g}, max {self.maximum:.6g}"
        )
        if self.projected_from is not None:
            return (
                f"{line}\n  projected from: {self.projected_from.describe()}\n  basis: {self.basis}"
            )
        return f"{line}\n  basis: {self.basis}"


@dataclass(frozen=True)
class BandCheck:
    """How a sample sat against a band."""

    band: Band
    n: int
    exceeded: int
    worst: float

    @property
    def passed(self) -> bool:
        return self.exceeded == 0

    def describe(self) -> str:
        verdict = "within" if self.passed else "OUTSIDE"
        return (
            f"{verdict} the band: {self.exceeded} of {self.n} exceeded {self.band.value:.6g}, "
            f"worst {self.worst:.6g}"
        )


def measure_band(deltas: list[float], *, basis: str, quantile: float = 0.999) -> Band:
    """The band is a quantile of the observed differences, not a round number near them.

    ``basis`` must say what was compared and on what, because the band is meaningless without
    it and a later reader cannot recover it.
    """
    if not deltas:
        raise ValueError("a band cannot be measured from an empty sample")
    if not basis.strip():
        raise ValueError("a band must carry the measurement it rests on")
    if not 0.0 < quantile <= 1.0:
        raise ValueError(f"quantile must lie in (0, 1], not {quantile}")
    ordered = sorted(float(delta) for delta in deltas)
    if any(math.isnan(delta) for delta in ordered):
        raise ValueError("the sample contains NaN, which no quantile of it would survive")
    index = min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])
    return Band(
        value=ordered[max(index, 0)],
        quantile=quantile,
        n=len(ordered),
        maximum=ordered[-1],
        median=median,
        basis=basis,
    )


def project_band(measured: Band, *, factor: float, reason: str) -> Band:
    """Scale a measured band to a precision it was not measured at, keeping its origin.

    The projection is deliberately a single declared factor with a stated reason rather than a
    model of quantisation error. A number this crude is honest about being crude; a fitted one
    would invite being read as a prediction.
    """
    if factor <= 0:
        raise ValueError(f"a projection factor must be positive, not {factor}")
    if not reason.strip():
        raise ValueError("a projection must say why its factor is what it is")
    return Band(
        value=measured.value * factor,
        quantile=measured.quantile,
        n=measured.n,
        maximum=measured.maximum * factor,
        median=measured.median * factor,
        basis=f"x{factor:g} on the measured band below, because {reason}",
        projected_from=measured,
    )


def check_band(band: Band, deltas: list[float]) -> BandCheck:
    """Whether a fresh sample sits inside the band, and how far the worst one is."""
    values = [float(delta) for delta in deltas]
    return BandCheck(
        band=band,
        n=len(values),
        exceeded=sum(1 for value in values if value > band.value),
        worst=max(values) if values else 0.0,
    )
