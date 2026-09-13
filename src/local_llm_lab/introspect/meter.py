"""How much an injection has damaged the forward pass, in nats, measured away from the question.

`find_edge` in the measurement harness cannot be reused here and the reason is subtle enough to
state twice. It bisects on the surviving mass of the clean top token at the DETECT answer position,
where on the untrained model that token is `NO`. On a model trained to detect, moving mass from NO
to YES *is* the thing we trained; the bisection would read it as damage, the edge would shrink as
training succeeded, and -- worse -- the noise arm would keep a genuine damage edge while the concept
arm got a detection-onset edge. The "equal damage" comparison would then be anchored on two
different physical quantities and would look clean.

So the meter reads elsewhere: short prompts with a confident, unrelated next token, none binary,
none about the model's internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..residual_patch import PatchPlan, PlannedPatch, build_masks, decoder_blocks

#: Twenty-four prompts whose next token the model is sure of and which have nothing to do with the
#: experiment. None is a yes/no question: a binary battery would move under the very thing the
#: detection arm is trying to measure.
BATTERY: tuple[str, ...] = (
    "The capital of France is",
    "2 + 2 =",
    "The opposite of hot is",
    "The largest planet in our solar system is",
    "Water freezes at zero degrees",
    "The chemical symbol for gold is",
    "Shakespeare wrote Romeo and",
    "A triangle has three",
    "The Pacific is the largest",
    "Cats have four",
    "The first month of the year is",
    "Light travels faster than",
    "A baby dog is called a",
    "The square root of nine is",
    "Bread is made from",
    "The sun rises in the",
    "Seven days make one",
    "The colour of grass is",
    "Ice is frozen",
    "A dozen means",
    "The Earth orbits the",
    "Mount Everest is the tallest",
    "Honey is made by",
    "The Atlantic separates Europe from",
)

#: Tiers taken from this repository's own records, so the numbers stay comparable across sessions.
PRISTINE = -0.01
INTACT = -0.105
HALF_MASS = -0.69


@dataclass
class MeterReading:
    damage: float
    per_prompt: list[float]


def _between(s0: float, d0: float, s1: float, d1: float, wanted: float) -> float:
    """Interpolate a scale between two swept points, in log-log where that is defined.

    Damage grows close to a power of the scale, so interpolating linearly in (scale, damage) is
    biased GENTLE -- by 12 per cent at the -0.03 rung for a square law and 27 per cent for a cube,
    always in the same direction, and worst at the gentlest tier where 48 per cent of positives
    sit. In log(scale) against log(-damage) the same anchors are exact.
    """
    if s0 > 0 and s1 > 0 and d0 < 0 and d1 < 0 and wanted < 0:
        import math
        ls0, ls1 = math.log(s0), math.log(s1)
        ld0, ld1 = math.log(-d0), math.log(-d1)
        if abs(ld1 - ld0) > 1e-12:
            t = (math.log(-wanted) - ld0) / (ld1 - ld0)
            return float(math.exp(ls0 + t * (ls1 - ls0)))
    span = (d0 - d1) or 1e-12
    return s0 + (s1 - s0) * (d0 - wanted) / span


class DamageMeter:
    """Batched: the whole battery, or a scale sweep across it, in one forward."""

    def __init__(self, model: Any, tokenizer: Any, *, device, dtype,
                 battery: tuple[str, ...] = BATTERY, system: str | None = None,
                 thinking: bool = False):
        from .render import render_prompt
        self.model, self.tokenizer = model, tokenizer
        self.device, self.dtype = device, dtype
        self.blocks = decoder_blocks(model)
        self.prompts = list(battery)
        self.ids = [render_prompt(tokenizer, p, thinking=thinking, system=system)
                    for p in self.prompts]
        self.pad = tokenizer.pad_token_id
        if self.pad is None:
            self.pad = tokenizer.eos_token_id or 0
        self._clean: list[tuple[int, float]] | None = None
        self._last_per_prompt: list[float] = []
        #: prompts whose in-batch top token differed from the cached clean one, last reading
        self._token_drift = 0

    # -- batching ---------------------------------------------------------------------------
    def _batch(self, indices: list[int]):
        rows = [self.ids[i] for i in indices]
        width = max(len(r) for r in rows)
        # RIGHT padding, so the site is the row's own last real token and differs per row. A
        # scalar site would be wrong the moment two prompts differ in length, which they do.
        input_ids = torch.full((len(rows), width), self.pad, dtype=torch.long)
        attention = torch.zeros((len(rows), width), dtype=torch.long)
        sites, lengths = [], []
        for r, row in enumerate(rows):
            input_ids[r, :len(row)] = torch.tensor(row)
            attention[r, :len(row)] = 1
            sites.append(len(row) - 1)
            lengths.append(len(row))
        return (input_ids.to(self.device), attention.to(self.device), sites, lengths, width)

    def _final_logprobs(self, input_ids, attention, sites) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits
        rows = out[torch.arange(out.shape[0], device=out.device),
                   torch.tensor(sites, device=out.device)]
        return rows.float().log_softmax(-1)

    # -- readings ---------------------------------------------------------------------------
    def clean(self) -> list[tuple[int, float]]:
        """Each prompt's own top token and its logprob, with nothing injected. Cached."""
        if self._clean is None:
            with torch.no_grad():
                ids, att, sites, _lengths, _w = self._batch(list(range(len(self.prompts))))
                lp = self._final_logprobs(ids, att, sites)
                top = lp.argmax(-1)
                self._clean = [(int(t), float(lp[i, t])) for i, t in enumerate(top)]
        return self._clean

    def reset_clean(self) -> None:
        """Call after the weights change. A percent is not a damage, and neither is a stale baseline."""
        self._clean = None

    def damage(self, vector: torch.Tensor, layer: int, scale: float,
               prompts: list[int] | None = None) -> MeterReading:
        """Mean loss, in nats, on each prompt's own clean top token under this injection.

        Routed through `curve`, so the reference is measured in the SAME forward -- see the note
        there. A separate clean pass at a different batch width is not a reference, it is a second
        experiment.
        """
        got = self.curve(vector, layer, [scale], prompts)
        return MeterReading(damage=got[0][1], per_prompt=list(self._last_per_prompt))

    def curve(self, vector: torch.Tensor, layer: int, scales: list[float],
              prompts: list[int] | None = None) -> list[tuple[float, float]]:
        """(scale, damage) at each scale, the whole sweep in ONE forward.

        The per-row plan already carries a per-row scale, so every (scale, prompt) pair is just
        another row. Nine scales over eight battery prompts is seventy-two rows and one pass; doing
        it as nine passes measures the same thing and costs nine times the card time.

        The shape of the curve is itself a result, not only a means to a scale: it is what
        distinguishes a sharp on-manifold transition from a smooth off-manifold one.

        SCALE ZERO IS ALWAYS SWEPT, and every damage is read against it rather than against the
        cached clean pass. A bf16 forward on this card is not batch-invariant, and the sweep runs at
        a different batch width from `clean()`: measured on Gemma 4 31B, an injection of EXACTLY
        ZERO reads +0.000000 nats at width 8, +0.014326 at width 24 and +0.004051 at width 96. That
        artifact is up to 1.4 times the whole -0.01 rung, in the direction that makes an injection
        look gentler than it is, so the solver answers with a scale that is too strong. Within a
        batch the offset is uniform to six decimal places, so referencing scale zero in the same
        forward cancels it exactly.
        """
        clean = self.clean()
        index = list(range(len(self.prompts))) if prompts is None else prompts
        scales = [0.0] + list(scales)
        # SCALE-MAJOR, and the reader below indexes `k * len(index)` on exactly that.
        # Prompt-major ordering puts scale s of prompt j where the reader expects prompt s
        # of scale j, which returns one prompt's damage curve as another's and raises
        # nothing. This line was flipped by a review agent working in this same worktree
        # and swept into dadc54c by `git add -A`; the tests caught it, the author did not.
        pairs = [(s, j) for s in scales for j in index]
        with torch.no_grad():
            ids, att, sites, lengths, width = self._batch([j for _s, j in pairs])
            plan = PatchPlan(layer=[layer] * len(pairs), site=sites,
                             scale=[s for s, _j in pairs], vector=[vector] * len(pairs))
            masks = build_masks(plan, width=width, hidden=vector.shape[-1],
                                device=self.device, dtype=self.dtype, lengths=lengths)
            patches = [PlannedPatch(self.blocks[l], mask=m, delta=d, layer=l)
                       for l, (m, d) in masks.items()]
            for patch in patches:
                patch.__enter__()
            try:
                lp = self._final_logprobs(ids, att, sites)
            finally:
                for patch in patches:
                    patch.__exit__()
        # the in-batch reference: row k=0 is scale zero, same width, same kernel schedule
        reference = [float(lp[i, clean[j][0]]) for i, j in enumerate(index)]
        drift = sum(1 for i, j in enumerate(index) if int(lp[i].argmax()) != clean[j][0])
        self._token_drift = drift
        out = []
        for k, s in enumerate(scales):
            if k == 0:
                continue
            start = k * len(index)
            per = [float(lp[start + i, clean[j][0]]) - reference[i]
                   for i, j in enumerate(index)]
            self._last_per_prompt = per
            out.append((s, sum(per) / len(per)))
        return out

    def scales_for_ladder(self, vector: torch.Tensor, layer: int, ladder: tuple[float, ...],
                          *, residual_norm: float, prompts: list[int] | None = None,
                          verify: bool = True,
                          percents: tuple[float, ...] = (0.008, 0.015, 0.03, 0.06, 0.125, 0.25,
                                                        0.5, 1, 2, 4, 8, 16, 32, 64, 128)
                          ) -> dict[float, tuple[float, float, dict]]:
        """Every rung of the ladder from ONE swept curve. {wanted: (scale, achieved, note)}.

        `note` carries how the scale was reached -- interpolated, or clamped to an end of the
        ladder -- and whether the curve was monotone. A clamped rung is not the strength it is
        labelled, and a caller that drops the note cannot tell.

        Sweeping per rung costs ten forwards each and measures the same curve ten times; one sweep
        and an interpolation is the same information for a tenth of the card time.
        """
        unit = float(vector.norm())
        scales = [(p / 100.0) * residual_norm / unit for p in percents]
        curve = sorted(self.curve(vector, layer, scales, prompts), key=lambda sd: sd[0])
        if not all(d == d for _s, d in curve):
            raise ValueError(f"L{layer}: non-finite damage in the swept curve: {curve}")
        monotone = all(d1 <= d0 + 1e-9 for (_a, d0), (_b, d1) in zip(curve, curve[1:]))
        # The ends of the curve are its EXTREMA, not its first and last points. 474 of 484 curves
        # on this model are non-monotone, and one that saturates and reads back up hands curve[-1]
        # a value gentler than four interior scales: the rung is then labelled clamped_high and
        # given the ladder ceiling, sixteen times the gentlest scale that actually reached it,
        # while the verified `measured` sits eight per cent off tier and trips no flag.
        gentlest = max(curve, key=lambda sd: sd[1])
        deepest = min(curve, key=lambda sd: sd[1])
        picked: dict[float, tuple[float, str]] = {}
        for wanted in ladder:
            if gentlest[1] <= wanted:
                picked[wanted] = (gentlest[0], "clamped_low")
            elif deepest[1] >= wanted:
                picked[wanted] = (deepest[0], "clamped_high")
            else:
                chosen, how = None, "interpolated"
                for (s0, d0), (s1, d1) in zip(curve, curve[1:]):
                    if d0 >= wanted >= d1:
                        chosen = _between(s0, d0, s1, d1, wanted)
                        break
                if chosen is None:
                    # The extrema straddle `wanted` with no descending pair across it, which a
                    # non-monotone curve allows. Nearest reading, and a note that says so.
                    chosen, how = min(curve, key=lambda sd: abs(sd[1] - wanted))[0], "nearest"
                picked[wanted] = (chosen, how)
        # One extra sweep for every rung at once, not one forward per rung. Measured identical to
        # the bit against per-rung verification, at a quarter of the card time.
        achieved: dict[float, float] = {}
        if verify:
            wants = list(ladder)
            got = self.curve(vector, layer, [picked[w][0] for w in wants], prompts)
            achieved = {w: d for w, (_s, d) in zip(wants, got)}
        return {w: (picked[w][0], achieved.get(w, float("nan")), {"how": picked[w][1],
                                                                 "monotone": monotone})
                for w in ladder}
