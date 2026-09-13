# The bench and the harness meant different things by "strength", by about twenty times

2026-09-14, WS-D. Gemma 4 31B, concept vectors from the 240-word bank, damage read on the
eight-prompt battery with the batch-matched meter, six concepts averaged per cell.

The steering bench sets strength as a per cent of the residual norm at the injection site. The
measurement harness sets it in nats of damage on an unrelated battery. Nobody had ever put the two
on the same axis, so a bench session and a grid run could use the same word and differ by an order
of magnitude.

| layer | 10% | 20% | 40% | 80% | 160% |
|---|---|---|---|---|---|
| 20 | −0.232 | −0.368 | −0.470 | −5.032 | −13.740 |
| 32 | −0.224 | −0.885 | −4.452 | −7.124 | −20.850 |
| 40 | −0.236 | −0.340 | −1.342 | −3.138 | −9.079 |
| 48 | −0.181 | −0.184 | −0.637 | −5.535 | −17.453 |

## What this changes

**The bench's default is past the top of our ladder.** Our strongest training rung is −1.2 nats.
Forty per cent at layer 40 is −1.34, and at layer 32 it is −4.45. A user turning the desk to its
default 40 per cent is steering harder than anything the adapter was ever trained on, and well past
the half-mass tier.

**Our tiers are single-digit percentages.** The −0.08 rung, where most of the evaluation lives, sits
below 10 per cent of the residual norm at every layer. Ten per cent already costs −0.18 to −0.24.

**Damage is wildly non-linear in per cent, and not monotone in layer.** Layer 32 at 40 per cent
costs −4.45 while layer 48 at the same per cent costs −0.64, a factor of seven. Between 40 and 80
per cent, layer 20 jumps from −0.47 to −5.03. Any experiment that holds per cent fixed across
layers is holding nothing fixed.

## The immediate consequence

The choice experiment's first run, which found no steering of a forced choice, ran at −0.08 nats.
That is roughly sixteen times gentler than the setting that visibly bent an ordinary greeting into
a literary image in a bench session the same evening. The null is therefore a statement about a
whisper, not about steering, and the experiment is being re-run as a dose-response sweep across the
whole ladder. A single strength is a weak test of a steering claim in either direction; an effect
that grows with strength is much harder to argue with than one point.
