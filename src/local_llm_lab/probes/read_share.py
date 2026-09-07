"""WP4's read-share pre-check and the sizing it licenses (issue 82, parts b and c).

Two things live here, and they are separate on purpose.

**The share.** WP3's hooks give, per head, the exact read weight ``alpha[t, s]`` of position ``s``
on the state read at ``t``, with ``y_t = sum_s alpha[t, s] * v_s`` reconstructing the block's own
output to within 1e-5 absolute in float32 (the gate calibrated in the WP3 design, worst observed
9.4e-7). :func:`read_share` turns that into the number WP4 asks for: what fraction of the read at
the decision comes from the span the observation window hides.

**The size.** :func:`size_two_by_two` applies the plan's rule -- if the share is below 0.02 at
every layer, the 2 by 2 runs at the minimum size that can detect a share of 0.05, or is recorded
as not worth running -- through WO-STAT-001's method: the exact paired test, at the Holm level for
the arm family, with every "n for 80 percent" a stable crossing rather than the first n to touch
the target, because discrete tests are not monotone in n.

Nothing here loads a model. The share is a pure function of an alpha row and the value vectors it
weights, so it is testable today against a stub source, and the only thing WP3's landing adds is
the adapter that yields those two arrays for a real point.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from local_llm_lab.probes.power import (
    DEFAULT_ALPHA,
    DEFAULT_POWER_TARGET,
    holm_alpha,
    mcnemar_power,
    smallest_n_for_mcnemar_power,
)

__all__ = [
    "NOT_WORTH_RUNNING",
    "ReadShare",
    "SizingVerdict",
    "read_share",
    "read_share_by_layer",
    "size_two_by_two",
]

#: The plan's two thresholds, in the units the plan states them in (§3 WP4, ratified).
NEGLIGIBLE_SHARE = 0.02
LICENSED_SHARE = 0.05

#: The outcome when no n within ``max_points`` reaches the power target. It fires on **both**
#: branches, not only the plan's second half: a pre-check negligible everywhere that cannot reach
#: 0.05, and an observed share too small to detect, are the same answer for different reasons.
#: ``every_layer_negligible`` on the verdict tells them apart.
NOT_WORTH_RUNNING = "not worth running"


@dataclass(frozen=True)
class ReadShare:
    """One cell's share, with the two norms it came from and whether it is defined.

    Both norms are kept rather than the ratio alone, for the reason the WP3 design gives about
    the retention ratio: a ratio whose denominator is small carries no information about the
    numerator's size, and a reader who has only the ratio cannot tell a real contribution from a
    quiet read.
    """

    span_norm: float
    total_norm: float
    defined: bool

    @property
    def share(self) -> float:
        """The span's contribution as a fraction of the whole read. Undefined cells give NaN."""
        return self.span_norm / self.total_norm if self.defined else math.nan


def read_share(
    alpha_row: Sequence[float],
    values: Sequence[Sequence[float]],
    span: Sequence[int] | range,
) -> ReadShare:
    """The hidden span's contribution to one read, as a norm share, exact.

    ``alpha_row[s]`` weights ``values[s]``; ``span`` names the source positions the observation
    window hides. The share is ``|| sum_{s in span} alpha[s] v_s || / || sum_s alpha[s] v_s ||``.

    **It is a norm of a sum, not a sum of norms, and it is not bounded by one.** The span's
    partial sum can point against the remainder, and then removing the span would move the read
    further than the span's own length -- so a share above 1 is a real reading about
    cancellation, not an error, and nothing here clips it. Summing per-position magnitudes
    instead would bound it at 1 and would answer a different question: how much weight the span
    carries, rather than how much of the read it accounts for.

    **The zero rule follows the WP3 design's**: a read whose total norm is exactly zero has an
    undefined share; it is marked undefined and counted rather than silently scored, because a
    zero denominator and a zero numerator are not the same finding.
    """
    if len(alpha_row) != len(values):
        raise ValueError(
            f"alpha row has {len(alpha_row)} weights for {len(values)} value vectors; "
            "they index the same source positions"
        )
    positions = set(span)
    unknown = {index for index in positions if not 0 <= index < len(alpha_row)}
    if unknown:
        raise ValueError(f"span names positions outside the row: {sorted(unknown)}")

    width = len(values[0]) if values else 0
    total = [0.0] * width
    partial = [0.0] * width
    for index, (weight, vector) in enumerate(zip(alpha_row, values, strict=True)):
        in_span = index in positions
        for axis, component in enumerate(vector):
            term = weight * component
            total[axis] += term
            if in_span:
                partial[axis] += term

    total_norm = math.sqrt(sum(component * component for component in total))
    span_norm = math.sqrt(sum(component * component for component in partial))
    return ReadShare(span_norm=span_norm, total_norm=total_norm, defined=total_norm != 0.0)


def read_share_by_layer(
    cells: Mapping[int, Sequence[ReadShare]],
) -> dict[int, dict[str, Any]]:
    """Two figures per layer, because one of them answers each of the two questions.

    ``max_share`` is the maximum over heads. It answers **negligibility**: the plan's rule asks
    whether the share is below a threshold at every layer, and one head carrying the read is
    exactly the case the rule exists to catch, which a mean over 32 heads would hide.

    ``layer_share`` is magnitude-weighted --
    ``sqrt(sum_h span_norm^2) / sqrt(sum_h total_norm^2)``, which is the norm share of the
    concatenated per-head reads. It answers **how large an effect to size against**, and the
    maximum is wrong for that: a quiet head whose total norm is a thousandth of its neighbours'
    can carry a span share of 0.9 while contributing almost nothing to the layer, and sizing
    against its 0.9 would license a 2 by 2 that can only detect an effect nothing will produce.

    Undefined cells are counted, not dropped silently, and they contribute nothing to either.
    """
    summary: dict[int, dict[str, Any]] = {}
    for layer in sorted(cells):
        defined = [cell for cell in cells[layer] if cell.defined]
        span_energy = math.fsum(cell.span_norm * cell.span_norm for cell in defined)
        total_energy = math.fsum(cell.total_norm * cell.total_norm for cell in defined)
        summary[layer] = {
            "cells": len(cells[layer]),
            "undefined": len(cells[layer]) - len(defined),
            "max_share": max((cell.share for cell in defined), default=None),
            "layer_share": (
                math.sqrt(span_energy) / math.sqrt(total_energy) if total_energy > 0.0 else None
            ),
            "median_total_norm": _median([cell.total_norm for cell in defined]),
        }
    return summary


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


@dataclass(frozen=True)
class SizingVerdict:
    """What the pre-check licenses, with every input that produced it named (R38)."""

    points: int | None
    outcome: str
    licensed_share: float
    observed_max_share: float | None
    matched_share: float
    matched_share_source: str
    max_head_share_by_layer: dict[int, float | None]
    single_head_dominates: bool
    discordance: float
    arms: int
    alpha: float
    holm_alpha: float
    power_target: float
    power_at_points: float | None
    inputs: dict[str, Any] = field(default_factory=dict)


def size_two_by_two(
    layer_share_by_layer: Mapping[int, float | None],
    *,
    max_head_share_by_layer: Mapping[int, float | None],
    matched_share_for: Callable[[float], float],
    matched_share_source: str,
    discordance: float,
    arms: int = 4,
    target: float = DEFAULT_POWER_TARGET,
    alpha: float = DEFAULT_ALPHA,
    max_points: int = 1000,
) -> SizingVerdict:
    """Size the 2 by 2, or record that it is not worth running (plan §3 WP4).

    ``matched_share_for`` maps a read share to the matched-only share the paired test would see,
    and it is **required with no default**. There is no measured mapping from a norm share to a
    win rate, and inventing one inside a sizing function is how an assumption becomes a number
    nobody can find later. The Head of Interpretability owns that mapping; this function records
    it as an input and computes from it.

    The rule, in the plan's own units: if the observed share is below 0.02 at every layer, the
    design is sized against 0.05 instead -- the smallest share worth going to the box for -- and
    if even that cannot be reached within ``max_points``, the outcome is ``NOT_WORTH_RUNNING``.
    The second half is a real outcome and this function returns it as one.

    ``arms`` is the Holm family: the 2 by 2's four cells each read against arm C. Power is
    computed at ``alpha / arms``, which is exact for the row that would carry the family and
    conservative for the rest.

    **The layer share decides both branches, and the maximum head share decides neither.**
    That pairing is structural rather than a choice, because the two are ordered:
    ``layer_share = sqrt(sum_h span_h^2) / sqrt(sum_h total_h^2)``, and ``span_h <= m * total_h``
    for ``m`` the maximum per-head share gives ``layer_share <= m`` at every layer, always.

    So testing negligibility on the maximum is strictly stricter than on the layer share, and it
    routes the case it was meant to catch the wrong way. A loud head at a share of 0.001 beside a
    quiet head at 0.9 fails negligibility on the maximum, falls to the effect branch, and is then
    sized against a layer share of 0.00135 -- an n nobody can run -- when the plan's rule, applied
    to the quantity the plan names, sizes against 0.05 and returns something plannable. The plan's
    "share at every layer" is the hidden span's contribution to that layer's read, which is the
    layer share.

    The maximum stays, recorded and never sizing, because it answers a question the layer share
    cannot: whether one head carries the read. ``single_head_dominates`` is true when some layer's
    maximum reaches the negligible threshold while its layer share does not, so a reader meets
    that case without the sizing acting on it.

    Which statistic decides negligibility is the Head of Interpretability's to change, since issue
    82 puts admissibility with the Head; the pairing is in one place and that is a one-line change.

    ``matched_share_source`` names the mapping, so the verdict records not just the value it
    produced but where it came from -- a stand-in in a test says so, and the Head's mapping says
    its date.
    """
    if set(max_head_share_by_layer) != set(layer_share_by_layer):
        raise ValueError(
            "the two mappings describe the same layers; "
            f"layer shares cover {sorted(layer_share_by_layer)} and maximum head shares "
            f"cover {sorted(max_head_share_by_layer)}"
        )
    if not layer_share_by_layer:
        raise ValueError("the pre-check reports at least one layer")
    observed = [value for value in layer_share_by_layer.values() if value is not None]
    if not observed:
        # The zero rule again, one level up. A pre-check whose every layer came back undefined
        # has measured nothing, and sizing it as negligible would turn "no read to attribute"
        # into "the span contributes almost none of the read" -- the two findings the cell-level
        # rule is careful to keep apart.
        raise ValueError(
            "every layer's share is undefined; a pre-check that measured nothing cannot licence "
            "a size, and 'undefined' is not 'negligible'"
        )
    observed_max = max(observed)

    negligible = observed_max < NEGLIGIBLE_SHARE
    licensed = LICENSED_SHARE if negligible else observed_max
    corrected = holm_alpha(alpha, arms)
    matched = matched_share_for(licensed)

    points = smallest_n_for_mcnemar_power(discordance, matched, target, corrected, max_n=max_points)
    if points is None:
        outcome = NOT_WORTH_RUNNING
        power = None
    else:
        outcome = "sized"
        power = mcnemar_power(points, discordance, matched, corrected)

    return SizingVerdict(
        points=points,
        outcome=outcome,
        licensed_share=licensed,
        observed_max_share=observed_max,
        matched_share=matched,
        matched_share_source=matched_share_source,
        max_head_share_by_layer=dict(max_head_share_by_layer),
        single_head_dominates=any(
            maximum is not None
            and maximum >= NEGLIGIBLE_SHARE
            and (layer_share_by_layer[layer] or 0.0) < NEGLIGIBLE_SHARE
            for layer, maximum in max_head_share_by_layer.items()
        ),
        discordance=discordance,
        arms=arms,
        alpha=alpha,
        holm_alpha=corrected,
        power_target=target,
        power_at_points=power,
        inputs={
            "layer_share_by_layer": dict(layer_share_by_layer),
            "max_head_share_by_layer": dict(max_head_share_by_layer),
            "matched_share_source": matched_share_source,
            "negligible_threshold": NEGLIGIBLE_SHARE,
            "licensed_threshold": LICENSED_SHARE,
            "rule": (
                "below 0.02 at every layer -> size against 0.05, or record not worth running "
                "(plan section 3, WP4)"
            ),
            "every_layer_negligible": negligible,
            "max_points": max_points,
        },
    )
