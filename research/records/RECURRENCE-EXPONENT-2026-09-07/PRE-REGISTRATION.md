# Pre-registration: does the chunkwise recurrence share the unrolled path's superlinear backward?

**Written before the run**, per the Chief's four conditions (18:45). Issue #90's second item. Author:
Deputy. Nothing has been run; the harness maps MLX and starts only after the Chief says the box is
free.

## The question, and why the existing evidence cannot answer it

`research/records/TRAIN-EFFICIENCY-2026-09-05` measured the **unrolled** recurrence
(`gated_delta_ops`) and fitted a backward exponent of **1.57** in sequence length where the algebra
says 1.0, over 125 to 1,000 tokens — four points across one octave. Nothing has measured the
**chunkwise** form (`gated_delta_chunkwise_ops`), which is what training actually runs and what
arm A ran on for its whole life.

The TRAIN-COST variant-A rows *were* measured under chunkwise, and the whole training step there
goes as **T^1.18** overall with segment exponents **1.07, 1.10, 1.34, 1.49**. That cannot settle it,
because a training step contains the attention term as well as the recurrence, and the attention
term is quadratic. Two readings fit those numbers equally: the recurrence is near-linear and the
rise is attention coming up underneath it, or the recurrence is superlinear and the short rows have
not reached the regime. **Measuring the operator alone removes the attention term by construction**,
which is the whole point of doing it this way.

## Design, fixed before the run

**Sizes.** 512, 1,024, 2,048, 2,688, 4,096, **8,192** tokens. The last is beyond the production cap
deliberately: an exponent fitted only inside the range the model runs at cannot say what happens
when the cap is raised, which is the decision #85's envelope feeds.

**Shapes.** The model's own, as `scripts/recurrence_forms_check.py` uses them: **16 key heads, 32
value heads, head dims 128 and 128**, batch 1, float32.

**Chunks.** 128 and 256. Both, because the chunk length is the lever R55 already identifies and an
exponent that differs between them is itself the finding.

**Both forms in one harness, on the same tensors and the same clock.** `gated_delta_chunkwise_ops`
against `gated_delta_ops`. This is not decoration: the 1.57 figure comes from a different range in a
different harness, so **the sweep re-fits the unrolled path over these sizes too**. If the unrolled
path does not reproduce 1.57 here, the figure is range-dependent and the framing of #90 changes
rather than the answer.

**Forward and backward timed separately.** The question is the backward's exponent. Backward is
`mx.grad` of a scalar loss through each form, `mx.eval`-ed before the clock stops, so nothing is
left lazy and counted in the next measurement.

**Fit rule.** Log-log least squares over all sizes, **and** per adjacent segment. Medians of at
least three repeats after a discarded warm-up, because the first call at any shape pays kernel
compilation — that contamination cost the CTX-EFFICIENCY sweep its 4,096 row and was only caught
because the resulting curve was not monotone.

## What each outcome means, stated now

| chunkwise backward exponent | reading | consequence for #85 |
| --- | --- | --- |
| ≈ 1.0 | does not transfer; the algebra holds for the form we run | the envelope's linear recurrence term stands; the TRAIN-COST rise is attention, which #85 already carries explicitly |
| ≈ 1.57, matching the unrolled path | transfers | the envelope is **wrong in shape, not merely in coefficient**; long rows are worse than projected and the 6k ceiling is optimistic |
| between, ~1.2–1.4 | partial, or range-dependent | the exponent is a description of the measured range and **must be reported with its range**; #85 carries it as an interval, not a slope |
| unrolled ≠ 1.57 here | the original figure is range-dependent | #90's premise is restated before its conclusion is |

**No outcome is "inconclusive" by default.** If the repeats do not separate the forms beyond their
own spread, that is reported as the forms being indistinguishable at these sizes, with the spread
quoted, not as a failure to measure.

## Memory and the concurrency rule

Per size, one layer, batch 1, float32 — the terms that scale:

| tokens | q+k+v | chunk-local at 128 | chunk-local at 256 |
| --- | --- | --- | --- |
| 2,688 | 84 MiB | 42 MiB | 88 MiB |
| 4,096 | 128 MiB | 64 MiB | 128 MiB |
| 8,192 | 256 MiB | 128 MiB | 256 MiB |

The forward's resident set is well under a gibibyte at every size; the backward holds more, and
**8,192 is the one to watch**. R47's window threshold is 0.6 × 17.76 GiB = **10.66 GiB**. The peak is
recorded per size with `mx.get_peak_memory`, and **if any size projects above 10.66 GiB the sweep
stops before it and asks for a declared window** rather than running and reporting afterwards.

R55(b)'s invocation arithmetic, one layer and no accumulation, against the 499,000 cap: 8,192 tokens
is 64 chunk-invocations at chunk 128 and 32 at chunk 256 — 7,796 and 15,593 buffers each before the
cap, so the buffer count is not the binding constraint at this scale and bytes are.

## Conditions on the run

- **`check && run`**: the harness maps MLX. It starts only after the Chief says the box is free, with
  the library check in the launching command and the launch conditional on it.
- Heartbeat entries at launch and at exit.
- Record here: this file, the as-run script as `.py.txt`, the raw timings, the fits.
