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
