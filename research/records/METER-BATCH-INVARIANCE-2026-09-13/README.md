# The damage meter was measuring its own batch schedule

2026-09-13, WS-D. Measured on the rented RTX PRO 6000 against google/gemma-4-31b-it in bf16.

## The measurement

An injection of exactly zero must cost exactly zero nats. It does not, if the reading and its
reference are taken in forwards of different widths.

| rows in the forward | measured damage of a zero injection |
|---|---|
| 8 | +0.000000 |
| 24 | +0.014326 |
| 96 | +0.004051 |

Within each batch the figure is uniform across all rows to six decimal places, so this is the
kernel's tiling and not sampling noise. It is not monotone in width either, which is why nobody
would have guessed the size of it from first principles. The same vector at the same real scale
reads −7.149586 at width 8 and −7.147171 at width 96, a difference of the same order.

This is the same phenomenon as [bf16 forward not batch-invariant](../../..) recorded on 2026-09-10,
where widths 1 and 64 differed by about five per cent at the final residual. What is new is that it
had been sitting inside the instrument that defines this programme's units.

## Why it mattered here and not elsewhere

`DamageMeter.clean()` batched the 8 battery prompts. `DamageMeter.curve()` batched every (scale,
prompt) pair in one forward, so a twelve-rung sweep was 96 rows. Damage was `curve reading minus
cached clean`, which is a difference across two different batch schedules.

The tiers this experiment runs at are −0.01, −0.08 and −0.20 nats. An artifact of +0.004 is 40 per
cent of the pristine rung; the +0.014 seen at width 24 is 143 per cent of it. And the sign is the
damaging one: an injection reads **gentler** than it is, so the scale solver answers a request for
a whisper with a scale that is too strong.

## What it produced downstream

The base-model evaluation of 2026-09-13, run with a correct prompt and a correct scorer, printed:

- per-row scatter at the pristine tier of 373 per cent of the mean being measured
- 252 of 484 rows more than 30 per cent off the tier they were labelled with
- 479 of 484 damage curves not monotone in scale
- 45 per cent of pristine-tier rows clamped to an end of the ladder

A curve that wiggles by more than the quantity being interpolated is not a curve you can
interpolate. Every "matched damage" verdict this project has issued at the fine tiers was matching
on means while the rows underneath disagreed.

## The repair

`curve()` now always sweeps scale zero and reads every damage against that block, in the same
forward. The offset is uniform within a batch, so referencing inside the batch cancels it exactly
rather than correcting for it. `damage()` routes through `curve()` so there is one path and no
second way to get it wrong.

The clean token *identity* still comes from the cached `clean()` pass, so the target token is
stable across the whole experiment rather than drifting per call; prompts whose in-batch argmax
disagrees with the cached token are counted and exposed.

The test does not wait for hardware. It injects a width-dependent constant into `_final_logprobs`
and asserts the reading does not move, then asserts the old form *would* have moved on the same
fixture — per method entry 35, an agreement is only evidence if the fixture could have disagreed.

## What has to be rebuilt

Everything calibrated through the old meter, which is every scale in `scales.pt`. Those were being
rebuilt anyway: a system prompt was added to the protocol on the same day and it moves the residual
norm at the injection site by up to a quarter at layer 48.

## What this does not touch

The base model's detection result. It answered NO on 484 of 484 rows with zero unparsed replies,
and the arms' *means* were close even under the old meter. Nothing about the batch artifact makes a
NO into a YES. What it invalidates is the claim that any two arms were at equal damage, which is
the claim the whole content-versus-disturbance comparison rests on.
