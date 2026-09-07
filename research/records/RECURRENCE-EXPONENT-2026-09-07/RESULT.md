# Result: the chunkwise recurrence does not share the unrolled path's exponent

Pre-registered in `PRE-REGISTRATION.md` before the run. Sweep `sweep.json`, log `run.log`, as-run
script `recurrence_exponent_sweep.as-run.py.txt` (`.py.txt`, unimportable). Issue #90's second item.

## An R47 breach, declared

**The last row — the unrolled backward at 8,192 tokens — peaked at 16.78 GiB, 0.94 of the 17.76 GiB
working set, against R47's 10.66 GiB threshold, with no declared window.**

The pre-registration promised the sweep would stop *before* such a size and ask. It did not: the
harness checks the peak **after** measuring, because that is when `mx.get_peak_memory` can report
it, so the stop recorded the breach instead of preventing it. The design was wrong against its own
pre-registration.

**It was projectable.** The unrolled peaks were 2.50, 4.56, 5.85 and 8.69 GiB at 1,024 to 4,096;
extrapolating the last ratio gives about 16.5 GiB at 8,192, which is what happened. A projection
from rows already in hand would have stopped before running. The machine survived, the lock
released cleanly, and nothing else was on the box — an account, not a defence. **A peak guard must
project from the rows it has, not report the row it just ran.**

## Backward exponent, log-log over 512 to 8,192, medians of three after a discarded warm-up

| form | overall | segments |
| --- | --- | --- |
| chunkwise, chunk 256 | **1.070** | 1.02, 1.08, 1.27, 0.94, 1.10 |
| chunkwise, chunk 128 | **1.202** | 0.97, 1.19, 1.21, 1.30, 1.38 |
| unrolled, five clean sizes | **1.713** | 1.65, 1.70, 1.76, 1.82 |
| unrolled, including the breach row | 1.763 | …, 1.93 — **contaminated, see below** |

Forward: chunk 256 **1.02**, chunk 128 **0.97**, unrolled **1.03** over the five clean sizes.

## The breach row is contaminated, and it is excluded

The 8,192-token unrolled row ran at 16.78 GiB, 0.94 of the working set. **Its forward time gives a
segment exponent of 3.11 after five segments at 1.05, 1.01, 1.02 and 1.06** — the forward is linear
in every clean segment and then jumps by a factor of eight in one doubling. That is memory pressure,
not scaling, and it condemns the whole row.

Two figures I published from it are withdrawn. **The unrolled forward is 1.03, not 1.43**: the 1.43
was entirely the contaminated row, and the clean forward is linear exactly as the algebra says.
**The backward's last segment, 1.93, is not clean either**; the clean segments are 1.65, 1.70, 1.76
and 1.82.

**The conclusion does not rest on that row.** The unrolled backward fits **1.713** over the five
clean sizes alone, against chunkwise's 1.070 and 1.202 — all six of whose sizes are clean, since the
chunkwise rows at 8,192 peaked at 1.92 and 2.37 GiB and ran before the pressure. And the withdrawal
**sharpens** the finding rather than weakening it: with a linear forward and a 1.71 backward, the
anomaly is located in the backward alone, which is what the algebra makes surprising.

**The exponent does not transfer.** At chunk 256 the chunkwise backward is 1.07 across four
octaves, which is the algebra's 1.0 within the spread. **Chunk 128 is 1.20 and rises to 1.38 in the
last segment** — a real difference between the two chunk lengths, and consistent with R55: the
smaller chunk pays more per token as rows grow.

## The premise is restated, as pre-registered

**The unrolled path does not reproduce 1.57 here.** It fits **1.713** over the five clean sizes and
its segments rise monotonically, **1.65 to 1.82**. The original figure came from 125 to 1,000 tokens, four
points over one octave. So **1.57 is a description of that range and the exponent is not constant:
it steepens with length.** The anomaly is real and larger than reported, not an artefact — the
fourth outcome of the pre-registered table.

## What matters more than the exponents

| tokens | unrolled backward | chunkwise-128 | ratio |
| --- | --- | --- | --- |
| 4,096 | 7.27 s | 0.155 s | **47×** |
| 8,192 | 27.63 s | 0.402 s | **69×** |

The ratio grows because the exponents differ. Peak memory at 8,192: **16.78 GiB unrolled against
1.92 GiB chunkwise-128**. So under the chunkwise form **the recurrence is not what limits row
length** — it is 2 GiB at twice the production cap.

## Consequence for #85

The first row of the pre-registered table: the envelope's **linear recurrence term stands** for the
form we run, at chunk 256. The TRAIN-COST whole-step rise from 1.07 to 1.49 is therefore the
attention term, which #85 already carries explicitly, **not** the recurrence. The envelope is left
alone.

**One caveat the numbers earn**: this is the operator at batch 1, one layer, float32, with no
adapter and no optimizer. It isolates the recurrence, which is what the question asked, and it is
not a training step.
