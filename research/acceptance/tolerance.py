"""G-2(b): MLX against a ported backend, as a tolerance rather than an identity.

Free-running token-for-token reproduction across backends is struck from the acceptance
gates, and the reason is arithmetic rather than defeatism. One flipped argmax separates two
trajectories completely, ties in bfloat16 against a 262,208-entry vocabulary are common, and
over two thousand tokens the probability of zero flips is effectively zero. A gate nobody can
pass teaches nothing. Exact reproduction is therefore asked of one backend against itself
within one build and device, which is a determinism test, and the cross-backend question is
asked as four tolerance statistics instead.

**What this module measures, and what the records can support.**

``teacher_forced_agreement`` feeds the recorded prompt and the recorded emitted prefix, so a
disagreement at one position cannot cascade into the next. Every position is then an
independent comparison against the recorded argmax, which free-running comparison is not.

``confidence_violations`` applies the hard rule. Layer 34 is the model's own softmax, so the
recorded probability of an emitted token is the probability of the greedy choice. **A flip
where that probability was at least 0.99 fails the run outright**: quantisation does not move
an argmax that confident, so such a flip is a mask, position, entry or norm defect. In the
stage-two corpus 4,199 of 5,245 emissions clear that bar, so the rule covers four fifths of
the decisions rather than a corner of them.

``divergence_indices`` reports where free running first parts company, with a floor, because
the distribution of that index is informative even though its tail is not a gate.

``top_k_jaccard`` compares the recorded layer-34 top-k identity set against a fresh one. It
needs token ids only, which is the whole reason it is usable here.

**KL against the recorded distributions is not computable and this module does not offer it.**
The records store the top-k token *ids* at layer 34 and the probability of the emitted token,
and never a distribution. A KL needs both sides in full. :func:`symmetric_kl` therefore exists
for a live-against-live comparison, where both backends are loaded and both distributions are
in hand, and it refuses to pretend a recorded side exists. This is a correction to the order
rather than a gap in the implementation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

__all__ = [
    "AgreementReport",
    "ToleranceReport",
    "run_tolerance",
    "teacher_forced_run",
    "torch_forward_rows",
    "DivergenceProfile",
    "JaccardReport",
    "confidence_violations",
    "divergence_indices",
    "symmetric_kl",
    "teacher_forced_agreement",
    "top_k_jaccard",
]

#: A flip at or above this recorded probability is a defect and not a rounding difference.
HARD_CONFIDENCE = 0.99

#: Rows ranked per block when reading logits. Small enough that the float32 copy is tens of
#: megabytes rather than gigabytes; large enough that the loop is not the cost.
_RANK_ROWS = 256


@dataclass(frozen=True)
class Flip:
    """One position where the ported argmax differs from the recorded one."""

    turn: int
    position: int
    recorded: int
    produced: int
    recorded_probability: float | None

    @property
    def hard(self) -> bool:
        return (
            self.recorded_probability is not None and self.recorded_probability >= HARD_CONFIDENCE
        )

    def describe(self) -> str:
        probability = (
            "unrecorded"
            if self.recorded_probability is None
            else f"P={self.recorded_probability:.6f}"
        )
        severity = "HARD" if self.hard else "soft"
        return (
            f"[{severity}] turn {self.turn} position {self.position}: recorded "
            f"{self.recorded} ({probability}), produced {self.produced}"
        )


@dataclass
class AgreementReport:
    """Teacher-forced argmax agreement, and every flip that broke it."""

    label: str
    compared: int
    agreed: int
    flips: list[Flip] = field(default_factory=list)
    unrecorded_probability: int = 0

    @property
    def rate(self) -> float:
        return self.agreed / self.compared if self.compared else 0.0

    @property
    def hard_flips(self) -> list[Flip]:
        return [flip for flip in self.flips if flip.hard]

    @property
    def passed(self) -> bool:
        """One hard flip fails the run. Soft flips are reported, not gated."""
        return not self.hard_flips

    def describe(self) -> str:
        lines = [
            f"{self.label}: {self.agreed}/{self.compared} argmax agree "
            f"({self.rate:.6f}), {len(self.flips)} flips, "
            f"{len(self.hard_flips)} of them at P >= {HARD_CONFIDENCE}"
        ]
        if self.unrecorded_probability:
            lines.append(
                f"  {self.unrecorded_probability} compared positions carried no recorded "
                "probability, so the hard rule could not be applied to them"
            )
        # Every flip, not only the gating ones. A run that passes because no flip was confident
        # is only evidence if the reader can see where the flips actually sat; "0 at P >= 0.99"
        # on its own cannot be told apart from a threshold nobody came near.
        lines.extend(f"  {flip.describe()}" for flip in self.flips[:12])
        if len(self.flips) > 12:
            lines.append(f"  ... and {len(self.flips) - 12} more")
        return "\n".join(lines)

    @property
    def flip_confidences(self) -> list[float]:
        """The recorded probability at each flipped position, for the record."""
        return [
            flip.recorded_probability
            for flip in self.flips
            if flip.recorded_probability is not None
        ]


@dataclass
class DivergenceProfile:
    """Where free running first parts company, per episode."""

    indices: list[int]
    reproduced: int
    floor: int

    @property
    def below_floor(self) -> list[int]:
        return [index for index in self.indices if index < self.floor]

    @property
    def passed(self) -> bool:
        return not self.below_floor

    def percentile(self, fraction: float) -> int | None:
        if not self.indices:
            return None
        ordered = sorted(self.indices)
        position = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
        return ordered[position]

    def describe(self) -> str:
        if not self.indices:
            return f"no episode diverged; {self.reproduced} reproduced outright"
        return (
            f"{len(self.indices)} episodes diverged, {self.reproduced} did not; "
            f"first-divergence index min={min(self.indices)} "
            f"p50={self.percentile(0.5)} max={max(self.indices)}; "
            f"{len(self.below_floor)} below the floor of {self.floor}"
        )


@dataclass
class JaccardReport:
    """Top-k identity overlap between the recorded layer-34 set and a fresh one."""

    label: str
    scores: list[float]
    k: int

    @property
    def mean(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    def percentile(self, fraction: float) -> float | None:
        if not self.scores:
            return None
        ordered = sorted(self.scores)
        position = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
        return ordered[position]

    def describe(self) -> str:
        if not self.scores:
            return f"{self.label}: no positions compared"
        return (
            f"{self.label}: top-{self.k} Jaccard mean {self.mean:.4f}, "
            f"p01 {self.percentile(0.01):.4f}, p50 {self.percentile(0.5):.4f}, "
            f"worst {min(self.scores):.4f}, over {len(self.scores)} positions"
        )


def teacher_forced_agreement(
    episode, produced_argmax: dict[tuple[int, int], int]
) -> AgreementReport:
    """Compare a ported argmax against the recorded one at every emitted position.

    ``produced_argmax`` is keyed by ``(turn index, emitted position)`` and holds the argmax the
    ported backend produced when fed the recorded prefix up to that position. Teacher forcing is
    what makes each position an independent comparison; under free running one flip decides
    everything after it and the count stops meaning anything.
    """
    report = AgreementReport(label=episode.label, compared=0, agreed=0)
    for turn in episode.turns:
        for emission in turn.emissions:
            key = (turn.index, emission.position)
            if key not in produced_argmax:
                continue
            report.compared += 1
            produced = produced_argmax[key]
            probability = turn.emitted_confidence(emission.position)
            if probability is None:
                report.unrecorded_probability += 1
            if produced == emission.token_id:
                report.agreed += 1
            else:
                report.flips.append(
                    Flip(
                        turn=turn.index,
                        position=emission.position,
                        recorded=emission.token_id,
                        produced=produced,
                        recorded_probability=probability,
                    )
                )
    return report


def confidence_violations(report: AgreementReport) -> list[Flip]:
    """The flips that fail the run outright. One of these is enough."""
    return report.hard_flips


def divergence_indices(results: Sequence, *, floor: int) -> DivergenceProfile:
    """First-divergence index per episode from free-running reproduction.

    ``floor`` is the point below which a divergence stops being arithmetic drift. An episode
    that parts company in its first handful of tokens did not drift there.
    """
    indices: list[int] = []
    reproduced = 0
    for result in results:
        divergence = result.first_divergence
        if divergence is None:
            reproduced += 1
        else:
            indices.append(divergence.index)
    return DivergenceProfile(indices=indices, reproduced=reproduced, floor=floor)


def top_k_jaccard(
    episode, produced_top: dict[tuple[int, int], Sequence[int]], *, k: int = 5
) -> JaccardReport:
    """Identity overlap between the recorded layer-34 top-k and a produced one.

    Uses token ids only, which is why it is available at all: the records carry the top-k ids
    and never their probabilities.
    """
    scores: list[float] = []
    for turn in episode.turns:
        for emission in turn.emissions:
            key = (turn.index, emission.position)
            if key not in produced_top:
                continue
            recorded = set(turn.final_top(emission.position - 1)[:k])
            produced = set(list(produced_top[key])[:k])
            if not recorded and not produced:
                continue
            union = recorded | produced
            scores.append(len(recorded & produced) / len(union) if union else 0.0)
    return JaccardReport(label=episode.label, scores=scores, k=k)


def symmetric_kl(left: Sequence[float], right: Sequence[float]) -> float:
    """Jensen-Shannon divergence between two full distributions, in nats.

    **Live against live only.** The stage-two records store the top-k token ids at layer 34 and
    the probability of the emitted token, never a distribution, so there is no recorded side to
    take a divergence against. Passing a top-k slice here would compute a real number about a
    truncated object and that is the error this programme has already made once.
    """
    if len(left) != len(right):
        raise ValueError(f"distributions differ in support: {len(left)} against {len(right)}")
    if not left:
        raise ValueError("a divergence needs a distribution, not an empty sequence")
    total_left = sum(left)
    total_right = sum(right)
    if not (
        math.isclose(total_left, 1.0, abs_tol=1e-3) and math.isclose(total_right, 1.0, abs_tol=1e-3)
    ):
        raise ValueError(
            f"both sides must be full normalised distributions; sums are {total_left:.6f} and "
            f"{total_right:.6f}. A top-k slice is not a distribution."
        )

    def _kl(p: Sequence[float], q: Sequence[float]) -> float:
        return sum(
            value * math.log(value / other)
            for value, other in zip(p, q, strict=True)
            if value > 0.0 and other > 0.0
        )

    mean = [0.5 * (a + b) for a, b in zip(left, right, strict=True)]
    return 0.5 * _kl(left, mean) + 0.5 * _kl(right, mean)


# --- the runner -----------------------------------------------------------------------------


@dataclass
class ToleranceReport:
    """Every G-2(b) statistic for one corpus, and whether the run passed."""

    agreement: AgreementReport
    jaccard: JaccardReport
    divergence: DivergenceProfile | None

    @property
    def passed(self) -> bool:
        """Only the hard rule and the divergence floor gate. The rest is reported."""
        if not self.agreement.passed:
            return False
        return self.divergence is None or self.divergence.passed

    def describe(self) -> str:
        lines = [self.agreement.describe(), self.jaccard.describe()]
        if self.divergence is not None:
            lines.append(self.divergence.describe())
        else:
            lines.append("free-running divergence not measured in this run")
        return "\n".join(lines)


def teacher_forced_run(episode, forward, *, top_k: int = 5):
    """Read every deciding position of an episode with the recorded prefix in front of it.

    Teacher forcing here is **one forward per turn, not one per token**. Feeding the whole
    recorded sequence and reading the argmax at each position gives exactly the prediction that
    position would have made with the recorded prefix ahead of it, because attention is causal.
    Running it as *n* separate generations would cost *n* times as much for the same numbers.

    ``forward`` takes a token id sequence and returns one ``(argmax, top_k ids)`` row per input
    position. The row count is checked against the sequence length rather than assumed: an
    off-by-one there would shift every comparison by one position and still produce a plausible
    agreement rate, which is the failure this programme has already paid for once.
    """
    produced_argmax: dict[tuple[int, int], int] = {}
    produced_top: dict[tuple[int, int], tuple[int, ...]] = {}
    for turn in episode.turns:
        sequence = list(turn.prompt_ids) + list(turn.token_ids)
        rows = forward(sequence)
        if len(rows) != len(sequence):
            raise ValueError(
                f"forward returned {len(rows)} rows for {len(sequence)} positions in turn "
                f"{turn.index}; the join would be shifted and still look plausible"
            )
        for emission in turn.emissions:
            argmax, top = rows[emission.position - 1]
            produced_argmax[(turn.index, emission.position)] = int(argmax)
            produced_top[(turn.index, emission.position)] = tuple(int(t) for t in top)
    return produced_argmax, produced_top


def run_tolerance(
    episode,
    forward,
    *,
    top_k: int = 5,
    free_running: Sequence | None = None,
    floor: int = 16,
    decoding: object = "greedy",
) -> ToleranceReport:
    """The G-2(b) statistics for one episode, from a teacher-forced pass over its records.

    Greedy only: teacher forcing reads the recorded trajectory, which was made greedy, and a
    sampled comparison against it would compare a draw with an argmax. The sampled mode is
    refused by name before anything is read.
    """
    from local_llm_lab.pipeline.sampled_decode import require_greedy

    require_greedy(decoding, where="the tolerance runner (G-2b)")
    produced_argmax, produced_top = teacher_forced_run(episode, forward, top_k=top_k)
    return ToleranceReport(
        agreement=teacher_forced_agreement(episode, produced_argmax),
        jaccard=top_k_jaccard(episode, produced_top, k=top_k),
        divergence=None if free_running is None else divergence_indices(free_running, floor=floor),
    )


def torch_forward_rows(model, view, *, top_k: int = 5):
    """A ``forward`` for :func:`teacher_forced_run` backed by a loaded torch model.

    Chunked with the same partition generation uses, so a teacher-forced read and a generated
    one see the same forward shapes and a difference between them cannot be the chunking.

    UNEXECUTED: no torch architecture view exists to run this against yet.
    """
    import torch

    from local_llm_lab.forward import prefill_passes
    from local_llm_lab.pipeline.runner import _forward_logits, pin_torch_determinism

    def forward(sequence):
        pin_torch_determinism()
        cache = view.make_cache()
        rows: list[tuple[int, tuple[int, ...]]] = []
        with torch.no_grad():
            for chunk in prefill_passes(list(sequence)):
                logits = _forward_logits(model, view, chunk.input_ids, cache)
                # Rank in row blocks, never whole-tensor. A full 2,048-token chunk against a
                # 262,208-entry vocabulary is 1.07 GB in bfloat16 and 2.15 GB the instant it is
                # cast to float32, on top of 7.3 GiB of weights inside a 10.656 GiB cap. The
                # cast is still needed -- ties in bfloat16 at this vocabulary size are common
                # and the decode loop resolves them in float32 -- so it happens per block and
                # the block is dropped.
                for start in range(0, logits.shape[1], _RANK_ROWS):
                    block = logits[0, start : start + _RANK_ROWS].float()
                    top = block.topk(top_k, dim=-1).indices
                    rows.extend(
                        (int(top[local, 0]), tuple(int(value) for value in top[local]))
                        for local in range(top.shape[0])
                    )
                    del block, top
                del logits
        return rows

    return forward
