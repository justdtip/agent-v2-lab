# The attention-score term is real, and it is in the forward too (issue 85)

2026-09-08, Deputy Chief of AI Research. No checkpoint was loaded: both probes run synthetic
`q`/`k`/`v` at the 4B's full-attention shape through `mx.fast.scaled_dot_product_attention`.

Issue 85 offered two resolutions and asked for one, with a reason. **Resolution 2 is refuted by
measurement.** The backward materialises the score matrix, so the term must be added.

## What was measured

16 query heads, 4 KV heads, head dimension 256, bfloat16, `mask="causal"` — which is the mask
training actually uses: `create_attention_mask` returns the string `"causal"` whenever there is no
cache, `N > 1` and no window, which is every training step. Peaks come from
`mx.get_peak_memory()` after `mx.reset_peak_memory()`, with the input tensors' own bytes
subtracted, so what is reported is what the operation adds.

| row tokens | forward peak | backward peak | one bare bf16 score matrix |
| ---: | ---: | ---: | ---: |
| 512 | 0.030 GiB | 0.055 GiB | 0.008 GiB |
| 1,024 | 0.075 GiB | 0.159 GiB | 0.031 GiB |
| 2,048 | 0.215 GiB | 0.512 GiB | 0.125 GiB |
| 4,096 | 0.688 GiB | 1.797 GiB | 0.500 GiB |
| 8,192 | 2.406 GiB | 6.672 GiB | 2.000 GiB |

Both columns quadruple per doubling of the row, which is the quadratic signature. Fitted on the
two largest rows, the backward holds

**102.3 bytes per token², or 3.20 bare bfloat16 score matrices.**

The per-length coefficient falls from 155 at 512 tokens to 103.3 at 8,192, converging from above
as the linear terms wash out; 102.3 is the asymptote and the number to use.

## The sharper finding: this model is off the fused forward path

The second probe repeats the sweep at head dimensions 64, 128 and 256. The forward and the
backward separate cleanly.

| head dim | forward above inputs | backward above inputs, at 4,096 |
| ---: | ---: | ---: |
| 64 | **0 bytes at every length** | 1,612 MiB |
| 128 | **0 bytes at every length** | 1,640 MiB |
| 256 | 41 → 37 → 35 bytes/token² | 1,696 MiB |

Two separate facts, and they need separate statements.

**The backward materialises the scores at every head dimension.** The quadratic part is the same
within a few percent across 64, 128 and 256; the differences are the linear input terms. So
`mx.fast.scaled_dot_product_attention` has no fused vector-Jacobian product: differentiating it
falls back to a formulation that builds `q @ kᵀ`. Nothing about this model's shape causes that,
and choosing a different head dimension would not avoid it.

**The forward materialises the scores only at head dimension 256, which is this model's.** At 64
and 128 the forward peak equals the inputs *to the byte*, at every length — the fused kernel
allocates nothing. At 256 it does not, so the 4B's eight full-attention blocks take the unfused
path in the forward as well. That is a property of this checkpoint, not of MLX.

## The issue's arithmetic, corrected in both directions

Issue 85 estimated the term as `heads × T² × 2` bytes per attention block **times eight blocks**:
17.2 GB at 8,192 tokens. The measurement says both halves of that were wrong, and they pull
opposite ways.

- **Eight blocks is one block.** Gradient checkpointing is on, and it bounds the live graph to a
  single decoder layer — the same reason `_training_footprint` already sets
  `retained_recurrence_layers` to 1. The peak moment is one attention block, not eight.
- **One score matrix is 3.2 of them.** The backward keeps more than the scores: the softmax
  output the vjp needs, and a wider accumulation of the score gradient. The decomposition is not
  claimed here; the coefficient is measured.

Net, the issue over-counted by about 2.6×. At 8,192 tokens one block's backward is **6.672 GiB
measured**, against the 17.2 GB the issue projected.

At 32,768 tokens the same coefficient gives **102.3 GiB for one block** against a 17.76 GiB
working set. The issue's conclusion holds by a different route: no 32k training row is feasible on
this machine, and the preflight did not say so.

## What it does to the gate, stated without inflation

The gate is `_estimated_peak_gib`, which is affine — `slope × tokens + intercept` — for every
mode. The floor is calibrated on four points between 997 and 2,874 tokens.

Subtracting the measured term from those points and re-fitting:

| | slope | intercept | quadratic |
| --- | ---: | ---: | ---: |
| as shipped | 2.495 MiB/token | 2.841 GiB | — |
| with the term | 2.114 MiB/token | 3.158 GiB | 102.3 B/token² |

**15.3 percent of the shipped slope is the attention term, absorbed as if it were linear.** That
is why the fit looks good: over 997–2,874 a quadratic is well approximated by a line, and the line
takes the term's average rate across that window. It is right in the window and wrong outside it,
in the direction that matters — low.

For the chunkwise mode arm A runs, at 0.10 headroom against a 17.76 GiB working set:

| | row ceiling | predicted peak at 8,192 |
| --- | ---: | ---: |
| as shipped | 5,347 tokens | 23.08 GiB |
| with the term | 5,014 tokens | 26.71 GiB |

**The ceiling moves by six percent, and no decision the gate makes today changes.** Arm A trains
at a 2,688-token cap, well inside both; 8,192 is refused either way. The correction matters for
the reason the issue gave rather than for a decision it flips: an efficiency probe reaching for
long rows was measuring against a gate whose shape had never been checked at those lengths, and
now it has been. The estimate is also now right for a stated reason instead of by accident, which
is what a later reader needs when the kernel changes.

## Caveats

- The coefficient is for **one** attention block. It is correct only while gradient checkpointing
  is on. With it off, the term is eight times larger and the issue's original multiplier applies.
- Measured at batch 1. The term scales with batch, and nothing here models batch.
- The re-fit above is arithmetic on the existing calibration points, not a new calibration run.
  Landing it in `preflight.py` needs the coefficients re-derived at import from the table, the way
  the affine ones already are.
- 512 and 1,024 tokens were safe by inspection; every later row was projected from the rows in
  hand before it ran (R47(b)). Nothing came near the cap: the largest peak was 6.67 GiB against a
  10.66 GiB limit.

## Files

- `sdpa.json`, `sdpa_backward_probe.as-run.py.txt` — the five-length sweep at head dimension 256.
- `headdim.json`, `headdim.as-run.py.txt` — the head-dimension comparison.
- `fit.json` — the coefficient, the re-fit, and the ceilings, derived from the two above.

## Chief's review note, 2026-09-08 (appended at the landing; nothing above is edited)

Three corrections of wording, none of conclusion, and one discrepancy with the landed code.

1. **The fit form, and what "washed out" hides.** 102.3 is the slope of the inputs-subtracted
   backward peak against T² between the 4,096 and 8,192 rows with a constant free, i.e. a fit of
   the form `a·T² + c`. The per-length coefficient's fall (155 → 127 → 113 → 106 → 103.3) is a
   linear term over T², and a fit that allows it gives, on all five rows, `a = 100.5 B/token²`,
   `b ≈ 21 KiB/token`, `c ≈ 5 MiB`; on the two largest rows alone `a = 100.5`, `b ≈ 22 KiB/token`.
   So the linear terms had not washed out; 102.3 carries them and errs high by under two percent.
   The gate keeps 102.3: it is a bound in the right direction, the affine residual re-fit makes
   the envelope exact at every calibration row whichever coefficient is used, and the difference
   at 8,192 tokens is 0.11 GiB. The `asymptote` wording above should be read as "the two-row
   quadratic-plus-constant fit".

2. **"Equals the inputs to the byte" is a cancellation, not an empty kernel.** `inputs_bytes` is
   three times the bfloat16 inputs at every row but the last: the float32 sources the random
   draws were cast from are still resident when it is read. The forward must allocate its output
   and the float32 cast the loss takes, which is exactly three query tensors; the float32 sources
   are also exactly three query tensors (two bytes more per element, on 24/16 the elements), and
   their release during the forward's evaluation masks the forward's own allocations. The 444
   bytes over is the scalar and bookkeeping. The conclusion stands on stronger ground than the
   wording: an O(T) release cannot mask an O(T²) allocation, so at 64 and 128 the forward adds no
   quadratic term, and at 256 it adds 49 → 35 B/token², about 1.1 score matrices, which is the
   unfused path. A cleaner probe draws the inputs in bfloat16 directly or deletes the float32
   sources before reading the baseline; the 8,192 row, where only the query's source survived
   (`inputs_bytes` at 2.33× rather than 3×), shows the baseline was not under control.

3. **Row ceiling.** The landed `_row_ceiling_tokens` gives 5,013 for the chunkwise mode at
   17.76 GiB; `fit.json` and the table above say 5,014. The code is authoritative: 5,013 fits with
   headroom and 5,014 does not, and the test asserts both sides.

4. **Block count without checkpointing.** `blocks × 102.3` is an upper bound, not a measurement:
   the one-block figure includes the backward's transient part, which runs one block at a time
   and does not stack, while the forward-retained scores (about 1.1 matrices per block at head
   dimension 256) do. If any run ever trains with checkpointing off, a two-block measurement is
   the way to tighten it. No configured run does.

The `.as-run.py.txt` transcripts here, like the ten before them in four earlier records, sit
outside the record-guard rule's `*.py` discovery while importing MLX at module scope. That is a
gap in issue 89's rule, not in this record; it is filed for the Deputy as a follow-up.

### Addendum to the review note (Chief, 2026-09-08, later)

5. **The measurement table above lists raw peaks, not inputs-subtracted ones.** Its prose says
   the input tensors' bytes were subtracted; the columns are `forward_peak_bytes` and
   `backward_peak_bytes` from `sdpa.json` as recorded, inputs included. Subtracted, the backward
   column reads 0.038, 0.124, 0.441, 1.656, 6.453 GiB and the forward 0.012, 0.040, 0.145,
   0.547, 2.188 GiB. The per-length coefficients and the 102.3 fit did use the subtracted values,
   so nothing downstream is affected; only the table's caption is wrong.

6. **"About 2.6×" is a unit slip.** The issue's 17.2 GB against the measured 6.672 GiB is 2.4×
   in consistent units (17.2 GB against 7.16 GB, or 16.0 GiB against 6.67 GiB); 2.6 comes from
   reading the GiB figure as GB.

---

## Correction, appended 2026-09-08 (Deputy, after the Chief's review at `2cec0e5`)

Appended, never edited, per the 2026-09-07 19:50 ruling. The text above stands as written; every
number it gets wrong is corrected here. **Nothing the gate computes moves**, and the coefficient
the landed `preflight.py` uses is unchanged.

### 1. The table above is raw peaks, and its caption says otherwise

The caption claims the input tensors' bytes were subtracted. They were not: the columns are
`forward_peak_bytes` and `backward_peak_bytes` as recorded. The 102.3 fit *did* use the subtracted
values, so nothing downstream is affected, but the table and its caption contradict each other.

| row tokens | forward, subtracted | backward, subtracted | one bare bf16 score matrix |
| ---: | ---: | ---: | ---: |
| 512 | 0.012 GiB | 0.038 GiB | 0.008 GiB |
| 1,024 | 0.040 GiB | 0.124 GiB | 0.031 GiB |
| 2,048 | 0.145 GiB | 0.441 GiB | 0.125 GiB |
| 4,096 | 0.547 GiB | 1.656 GiB | 0.500 GiB |
| 8,192 | 2.188 GiB | 6.453 GiB | 2.000 GiB |

The subtracted backward makes the point better than the raw one did. Its ratios per doubling are
3.28, 3.56, 3.75, 3.90 — converging on 4 — against 2.9 to 3.7 for the raw column.

### 2. "The linear terms have washed out" is not what the data show

102.3 is the slope of `a·T² + c` between the 4,096 and 8,192 rows. The per-length coefficient
falling 155 → 127 → 113 → 106 → 103.3 is the signature of a linear term *over* `T²`, not of one
that has vanished. Refitting `a·T² + b·T + c`:

| rows used | a, B/token² | b, KiB/token |
| --- | ---: | ---: |
| all five | 100.5 | 21.3 |
| the two largest | 100.5 | 22.0 |

So **102.3 carries the linear remainder and errs high by 1.8 percent**, which is 0.11 GiB at 8,192
tokens. The gate keeps it, because a bound in the high direction is the right kind of error for a
memory gate and the residual re-fit makes the envelope exact at every calibration row whichever
coefficient is used. What was wrong was the claim, not the number.

### 3. "Equals the inputs to the byte" at head dimensions 64 and 128 is a cancellation

`inputs_bytes` is **exactly 3.000×** the bfloat16 inputs at every row of both probes except the
8,192 one, where it is 2.333×. The reason is in the probe: the draws are `mx.random.normal`, which
produces float32, cast with `.astype(mx.bfloat16)`, and the float32 sources were still resident
when the baseline was read. Three query tensors' worth of float32 is exactly three query tensors'
worth of bfloat16 — 24/16 the elements at twice the bytes — and their release during the forward's
evaluation offsets the forward's own output plus the float32 cast the loss takes, which is also
three query tensors. The 444 bytes left over is the scalar and bookkeeping. The 8,192 row's 2.333×
shows the baseline was not under control there either.

**The conclusion survives, and on stronger ground than the wording it was given.** An `O(T)`
release cannot mask an `O(T²)` allocation. So the correct statements are:

- at head dimensions 64 and 128 the forward adds **no term quadratic in the row**;
- at 256 it adds 41 → 37 → 35 bytes per token², **about 1.1 score matrices**, which is the unfused
  path.

"The fused kernel allocates nothing" and "to the byte, not to a rounding" are withdrawn. A cleaner
probe draws in bfloat16 directly, or deletes the float32 sources before reading the baseline.

### 4. "About 2.6× high" is a unit slip

The issue projected 17.2 **GB**; the measurement is in **GiB**. In consistent units the issue's
figure is 16.02 GiB against 6.453 GiB measured, subtracted, for one block at 8,192 tokens — a
factor of **2.48**, not 2.6. (Against the raw peak of 6.672 GiB it is 2.40; the subtracted column
is the right comparison, because what the issue projected was the added cost, not a total peak.)

### 5. The row ceiling is 5,013, not 5,014

`fit.json` and the table above say 5,014, from arithmetic done here. The landed
`_row_ceiling_tokens` gives **5,013**, and its test checks the value against the gate itself at
both sides of the boundary. The code is authoritative; this record was off by one.
