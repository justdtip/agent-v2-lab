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
        """Mean loss, in nats, on each prompt's own clean top token under this injection."""
        clean = self.clean()
        index = list(range(len(self.prompts))) if prompts is None else prompts
        with torch.no_grad():
            ids, att, sites, lengths, width = self._batch(index)
            plan = PatchPlan(layer=[layer] * len(index), site=sites,
                             scale=[scale] * len(index), vector=[vector] * len(index))
            masks = build_masks(plan, width=width, hidden=vector.shape[-1],
                                device=self.device, dtype=self.dtype, lengths=lengths)
            patches = [PlannedPatch(self.blocks[l], mask=m, delta=d, layer=l)
                       for l, (m, d) in masks.items()]
            for p in patches:
                p.__enter__()
            try:
                lp = self._final_logprobs(ids, att, sites)
            finally:
                for p in patches:
                    p.__exit__()
        per = [float(lp[i, clean[j][0]]) - clean[j][1] for i, j in enumerate(index)]
        return MeterReading(damage=sum(per) / len(per), per_prompt=per)

    def curve(self, vector: torch.Tensor, layer: int, scales: list[float],
              prompts: list[int] | None = None) -> list[tuple[float, float]]:
        """(scale, damage) at each scale. Cheaper than bisection and the shape is itself a result:
        it is what distinguishes a sharp on-manifold transition from a smooth off-manifold one."""
        return [(s, self.damage(vector, layer, s, prompts).damage) for s in scales]

    def scale_for_damage(self, vector: torch.Tensor, layer: int, wanted: float,
                         *, residual_norm: float, prompts: list[int] | None = None,
                         percents: tuple[float, ...] = (0.5, 1, 2, 4, 8, 16, 32, 64, 128)
                         ) -> tuple[float, float]:
        """The scale that costs `wanted` nats, by interpolating a swept curve. Returns (scale, achieved).

        Interpolated on the ladder rather than bisected: bisection costs a dozen sequential
        evaluations per vector and layer, and the curve is wanted anyway.
        """
        unit = float(vector.norm())
        scales = [(p / 100.0) * residual_norm / unit for p in percents]
        curve = self.curve(vector, layer, scales, prompts)
        curve = sorted(curve, key=lambda sd: sd[0])
        for (s0, d0), (s1, d1) in zip(curve, curve[1:]):
            if d0 >= wanted >= d1:
                span = (d0 - d1) or 1e-12
                chosen = s0 + (s1 - s0) * (d0 - wanted) / span
                return chosen, self.damage(vector, layer, chosen, prompts).damage
        # never reached the wanted damage inside the ladder; say so with the strongest tried
        return curve[-1][0], curve[-1][1]
