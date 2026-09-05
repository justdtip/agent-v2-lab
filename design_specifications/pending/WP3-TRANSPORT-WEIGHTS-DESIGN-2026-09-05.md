# WP3 design: transport weights on both channels (Qwen3.5-4B)

Author: Head of Interpretability, 2026-09-05. Status: DESIGN, for the Chief before dispatch.
Nothing dispatched, nothing run. Basis: `pending/03-REVISED-PLAN-2026-09-05.md` WP3;
`under_review/CHIEF-PROBE-REFINEMENTS-2-DERIVATIONS-2026-09-05.md` sections 9 and 10;
`under_review/CHIEF-JSPACE-PAPER-READING-2026-09-05.md` section 6. Every claim below that is
about the model or the library was checked against the installed code and the cached config on
2026-09-05; the file and line are given so the Chief can check them the same way.

## 1. Question and scope

For each channel that can move content from position `s` to a later position `t`, what fraction
of the read at `t` comes from each distance? The recurrent channel's answer is exact from the
forward pass (section 6.1 of the paper-reading memo, verified to 1.2e-6 in section 10); the
attention channel's is the attention probability. This is the measurement that licenses or
refutes F4, "long-range transport in this model is attention-only".

**Scope limit, to be carried into every reading.** The read weight `alpha` is the Jacobian of
the DeltaNet output with respect to the *values* only. `q`, `k`, `g` and `beta` at position `s`
also depend on `h_s`, so the total sensitivity of `y_t` to `h_s` is larger than `|alpha_{t,s}|`.
`alpha` is the content path with routing held fixed, which is the exact analogue of attention
mass and the right quantity for a transport claim. A null on `alpha` licenses "the recurrent
block does not carry content from that distance"; it does **not** license "the recurrent block
is insensitive to that position". The paper's frozen-QK variant is the same restriction.

## 2. Four corrections to the drafted brief

**C1. The attention indices in the drafted reading are block indices, and reading them as
repository layers selects five blocks of the wrong kind.** The plan's reading names "attention
layers 11, 15, 19, 23, 27". The repository convention is fixed in code: layer `L` is the residual
after block `L - 1` (`pipeline/jlens.py:519`, `probe_layer_kind`, docstring and body). The cached
config's `layer_types` puts `full_attention` at block indices 3, 7, 11, 15, 19, 23, 27, 31, which
is `is_linear = (layer_idx + 1) % 4 != 0` (`mlx_lm/models/qwen3_5.py:207`). So those blocks write
layers 4, 8, 12, 16, 20, 24, 28, 32. An implementer who hooks "layers 11, 15, 19, 23, 27" in the
repository convention attaches to blocks 10, 14, 18, 22, 26, and every one of those is a linear
block. The measurement would silently return DeltaNet statistics where attention was intended and
would not raise.

  - In-band (13 to 29) attention blocks are **15, 19, 23, 27**, writing layers 16, 20, 24, 28.
    Four, not five: block 11 writes layer 12, one layer below the band's onset.
  - Head count for the "top 1 percent of workspace-layer heads" argument becomes 4 x 16 = **64**
    query heads, not 80. The argument that a 1 percent selection is not meaningful survives.
  - Every layer set in this design is written as `block b -> layer L` so the pair cannot be lost.

  Second-order consequence for R40b, raised here because this design is the first to depend on
  it: the ratified probe layers 13, 16, 20, 24, 28 have writer kinds linear, attention,
  attention, attention, attention. Four of the five are the residual immediately after a full
  attention block. That is not wrong, but EXP-001 kind-matched its layers precisely so that a
  layer effect could not be read as a block-kind effect, and this set does not. It is the Chief's
  ruling and I am not reopening it; I am recording that any WP3 or WP5 result reported at these
  layers is confounded with block kind unless the partner rule supplies the matched set, and
  asking that the partners be named in the registry when WP2 lands.

**C2. The pre-registered threshold is not scale-free and is satisfied at gap 1, so it cannot
discriminate.** The drafted reading is "median `|alpha|` below 0.05 beyond 256 tokens on every
recurrent head". `alpha_{t,s} = beta_s k_s^T [prod] q_t`, and the library fixes both norms:
`k = inv_scale * rms_norm(k)` and `q = inv_scale**2 * rms_norm(q)` with `inv_scale = Dk^-0.5`
(`mlx_lm/models/qwen3_5.py:178-180`). Measured on the real constants, `||k|| = 1.000` and
`||q|| = 0.0884 = 1/sqrt(128)`. The product of contractions has operator norm at most 1, so

    |alpha_{t,s}| <= beta_s * ||k|| * ||q|| = 0.0884 * beta_s   at every gap, gap 1 included,

and with a typical cosine of about `1/sqrt(Dk)` between an unrelated key and query the median at
gap 1 is near 0.008. The threshold is roughly an order of magnitude above the value it is meant
to exclude at the *shortest* distance. As drafted the reading passes whatever the model does.

  Replace absolutes with two scale-free statistics, defined identically on both channels so the
  cross-channel comparison is licensed (R35):

  - **Mass share** `m_B(t,h) = sum_{s in B} |w_{t,s}| / sum_{s<=t} |w_{t,s}|`, with `w = alpha`
    for a value head and `w = p` (attention probability) for a query head. For attention this
    reduces to the drafted "attention mass per bin", so nothing is lost.
  - **Retention ratio** `r_B(t,h) = mean_{s in B} |w_{t,s}| / mean_{s in bin 1-4} |w_{t,s}|`.

  The mass share is the comparable quantity; the retention ratio is the horizon quantity. Both are
  needed because **the mass share is confounded with bin width**: at `t = 1547` the bin 257 to
  1024 holds 768 of 1547 eligible positions, so a head with uniform weights has a mass share of
  0.50 there and 0.003 in bin 1 to 4. A 0.05 criterion on the mass share would be failed by an
  almost perfectly local head. Every mass share is therefore reported against its uniform null
  `|B| / t`, and every bin's width is reported as `n of N` (R38). The retention ratio has no
  bin-width term and is the one the reading turns on.

  This also makes the statistic robust to the block's own output processing, which absolutes are
  not: the DeltaNet output passes through `RMSNormGated` over the value-head dimension and then
  `out_proj` (`qwen3_5.py:198-199`), and attention carries `attn_output_gate: true`. A per-head
  RMS norm removes absolute magnitude and preserves the mixture; shares survive it, `|alpha|` does
  not.

**C3. The key-to-value head pairing is `h // 2`, and a fixture with `Hk == Hv` cannot detect a
wrong rule.** The config has `linear_num_key_heads: 16` and `linear_num_value_heads: 32`, while
`g` and `beta` are per value head (`beta = sigmoid(b)`, `b = in_proj_b(x) -> [B, T, 32]`;
`g = compute_g(A_log, a, dt_bias)`; `gated_delta.py:274-275`, `:9`). The library reconciles them
with `repeat_factor = Hv // Hk; q = mx.repeat(q, repeat_factor, -2); k = mx.repeat(k, ...)`
(`gated_delta.py:242-244`). `mx.repeat` on the head axis yields `[k0, k0, k1, k1, ...]`, verified:
value head `h` uses key head **`h // 2`**. `mx.tile` would give `h % 16`, which agrees with the
correct rule for exactly one of the 32 heads. The branch at `gated_delta.py:242` is guarded by
`repeat_factor > 1`, so a random-tensor fixture built with `Hk == Hv` never executes it and any
pairing rule passes. Section 10's verification used `Dk = Dv = 128` and reports no head count; if
it ran at `Hk == Hv` it cannot have exercised this. **The R31 fixture must use `Hv / Hk = 2` with
distinguishable per-head keys, or it cannot show the defect.** This is the same shape as the
EXP-002 mask fixture that could not show the mask defect.

**C4. The gap bins stop at 1024 and the experiments they underwrite reach 2688.** WP3 exists in
part to supply per-kind predictions for WP5 (EXP-003), whose levels run to 2048, and for EXP-002's
persistent cache; `train.max_seq_length` in `configs/models/qwen35-4b.yaml` is 2688. Bins become
**1-4, 5-16, 17-64, 65-256, 257-1024, 1025-2688**, with the last two reported only where `t`
admits them and the eligible `n of N` stated per bin.

## 3. Method

**Contexts.** EXP-001's 42 probe points on the `jsweep` split regenerated from the recorded seed
(20260902, probe step 3), rendered as EXP-002 section 3.2 constructs the persistent unwindowed
text, so the text is the one the transport question is about; plus ten agent transcripts; plus the
EXP-002 decision contexts. Median length about 1547 tokens on the probe points.

**Hook points, exactly.** For each linear block, capture the tensors *as passed to*
`gated_delta_update`: `q`, `k`, `v` after the causal depthwise conv and the RMS norm and scaling
(`qwen3_5.py:169-180`), and `a`, `b`, `A_log`, `dt_bias`, from which `beta = sigmoid(b)` and
`g = exp(-exp(A_log) * softplus(a + dt_bias))` are recomputed exactly as the library does.
Capturing `in_proj_qkv` output instead would be pre-convolution and the reconstruction in section
4 would fail, which is how that error is caught rather than assumed. For each attention block,
capture the probabilities per query head. No residual capture, so R18b's capture dtype does not
apply and the block runs natively; the `alpha` arithmetic is float32.

**Computing `alpha` cheaply and exactly.** `alpha_{t,s} = beta_s k_s^T [prod_{r=s+1..t} A_r] q_t`
with `A_r = g_r (I - beta_r k_r k_r^T)`, `s + 1` leftmost. Do not form the products. Define
`w_t = q_t` and scan backwards with `w_{s-1} = A_s w_s`, which is a rank-1 update,
`A_r w = g_r (w - beta_r k_r (k_r^T w))`, costing `O(Dk)` per step. One full `alpha` row at one
`t` is then `O(T * Dk)` per head, not `O(T * Dk^2)`: about 152 Mflop for all 768 value heads at
`T = 1547`, seconds rather than the hours an explicit-product reading of "T x T weight matrices
per head" would cost, and numerically better because no 128x128 product is accumulated. Section
10's explicit reference stays as the fixture's oracle at `T = 96`, not as the production path.

**Positions.** Full `alpha` rows at the decision position of every context, plus 32 positions per
context spread geometrically over the context for the distance distribution. Not every `t`: the
full lower triangle is `O(T^2 Dk)` per head and buys nothing the sample does not.

**Underflow accounting.** With a median gate constant of 2.4 tokens the backward scan will
underflow to exactly zero at long gaps on most heads. That is the finding, not a fault, but it
must be visible: report the fraction of `(head, bin)` cells in which `w` reached exactly zero, as
`n of N`, so that a null is never silently a float artefact.

## 4. Verification, before any statistic is read

1. **Reconstruction.** `y_t = sum_s alpha_{t,s} v_s` against the block's actual
   `gated_delta_update` output at every sampled position, not only the decision. Report the worst
   absolute and relative error over all `(block, head, position)` cells as `n of N`, with the
   tolerance stated in advance: worst absolute error above 1e-4 in float32 fails the run.
2. **Pairing.** The fixture of C3, at `Hv / Hk = 2` with per-head-distinguishable keys, against
   the library's own `gated_delta_ops` (R31: the real class, not a stand-in).
3. **Bound.** Measured retention against the section 10 prediction
   `sqrt(1 - (2 beta - beta^2) rho^2)` per token at three gaps, as section 9 asked.
4. **Attention rows** sum to 1 per query head to 1e-6.

A failure of 1 or 2 stops the run and is itself the report.

## 5. Pre-registered reading

Primary, on the retention ratio, over in-band blocks only (linear blocks writing layers 13 to 29;
attention blocks 15, 19, 23, 27 writing layers 16, 20, 24, 28):

- **R1.** Median over the 576 in-band value heads of `r_B` at bin 257-1024 is below 0.05, and at
  bin 1025-2688 below 0.01, while the median over the 64 in-band query heads at those bins is
  above 0.05. Read: the recurrent channel has no long-range transport and attention does.
- **R2.** If the two medians are not separated at 257-1024, F4 is not supported and every
  prediction the plan derives from it (WP5's per-kind predictions, EXP-002's arm A null) is
  withdrawn pending WP3's actual curve.
- **R3.** Named exceptions are not a failure of R1. *Amended 2026-09-05 after the calibration
  preview, which showed the retention ratio's own tail is not safe to name heads from.* A value
  head is a named exception only if **both** its retention ratio `r_B` at 257-1024 exceeds 0.05
  **and** its mass share at that bin exceeds the bin's uniform null. The retention ratio alone is
  a quotient whose denominator is the head's own short-gap weight, so a head that simply ignores
  gaps 1 to 4 scores an arbitrarily large ratio with no long-range transport at all. The
  calibration preview's p90 retention curve, 1.00, 1.54, 1.10, 0.78, 0.52, 0.52 across the six
  bins, does not decay while its median curve does, and exceeds 1 at two bins; that is the
  signature of a varying denominator, not of a second population of long-horizon heads. Named
  exceptions are reported with block, head, `n of N`, and both statistics side by side. The
  plan's universal quantifier ("every recurrent head") is dropped: over 576 heads it converts one
  tail head into a refutation of a distributional claim.
- **R3a.** Every quantile reported for the retention ratio is reported for the mass share at the
  same cells, and the handling of a zero or underflowed denominator is stated in the artifact.
  Zeros are a third of the long-bin cells in the preview, so the rule that governs them governs
  the top decile.
- **R3b.** *Added 2026-09-05 after the joint-criterion re-run.* A named head is reported with the
  **distribution** of its write strength `beta`, not its median, and with the concentration of its
  long-bin `alpha` mass over source positions. The re-run's long-gate population is the model's
  low-write population: over the 1,536 (context, block, head) entries in the calibration artifact,
  heads whose gate product over 1,024 tokens exceeds 0.5 have a median `beta` of 0.060 against
  0.388 for the rest, and the rank correlation between the gate product and `beta` is -0.832. The
  gate product is computed from `g` alone, so that correlation is not an artefact of `tau_head`'s
  definition, which carries a `beta` term. A head with `g` near one and `beta` near zero has a
  nearly frozen state: nothing decays because nothing is written, and `alpha` is proportional to
  `beta_s` at the source. Two very different objects produce this signature and the median cannot
  separate them. A **write-once memory head** has `beta` near zero at most positions and a spike
  at a few, and its long-bin mass concentrates on those source positions. A **stale smear** has
  `beta` uniformly small and its mass spread flat, which shows as a mass share sitting *at* the
  uniform null rather than above it. The re-run's long bins sit at the null, 0.55 against 0.55 and
  0.27 against 0.26, which is the smear signature; the discriminating measurement is `beta`'s
  upper quantiles per head and the fraction of each joint cell's bin mass carried by its top ten
  source positions. Until that is reported, a named head is a candidate and not a memory head.
- **R3c.** *Added 2026-09-05 after the concentration re-run.* Neither the write spike nor the
  source concentration is read against a uniform null, because uniform is not what a
  non-selective head produces. For the long-gate heads the operator product is approximately the
  identity, since `g` is near one and `beta` is small, so `alpha_{t,s}` reduces analytically to
  `beta_s (k_s . q_t)`, a product of the head's own write strength and a random high-dimensional
  alignment. That form alone concentrates. Fitting `beta = sigmoid(N(mu, sigma^2))` to the
  re-run's own reported joint-cell statistics (median 0.057, 99th percentile 0.61, 2.5 percent of
  positions above 0.5) and drawing the alignment at random gives a top-ten source share of 0.138
  at 768 sources and 0.216 at 400, against the measured 0.144 and 0.167. The measured values are
  1.05 times this null at 257-1024 and 0.77 times it at 1025-2688, where the measurement falls
  *below* the null and 99 percent of null draws exceed it. The 11-fold and 6-fold ratios against
  uniform are therefore not evidence of selective writing. By the same token, "strong writes above
  0.5 at 2.5 percent of positions" is exactly the upper tail of a smooth sigmoid-Gaussian with
  that median and is not a spike; a spike requires a demonstrated departure from unimodality.
  **The discriminating test is source-set overlap.** A write-once memory head draws every later
  read from the same fixed source positions, so the top-ten source sets at two read positions
  coincide; a head whose weights are write strength times random alignment draws a fresh set each
  time. Report the overlap of top-ten source sets across read positions within a head against the
  hypergeometric chance overlap. That test is content-independent, available from the captured
  tensors, and it is the one that separates memory from noise.
- **R3d.** *Added 2026-09-05 after the overlap run.* Overlap alone does not separate retrieval
  from a static filter, and the measured value sits where that alternative is live. Simulating
  the same generative form, top-ten far-source overlap runs 6.96 of 10 for a frozen query, 4.29
  at query cosine 0.9, 2.02 at 0.6 and 1.5 at independence; the run's write-strength-only null of
  1.68 lands at independence, which corroborates the simulation, and its measured 3.24
  corresponds to a query cosine near 0.8. A head whose query direction is 80 percent constant
  retrieves the same far sources at every read position without any content specificity. **The
  completing measurement is the direct one: the median pairwise cosine between `q_t` at different
  read positions, per head.** Permuting real queries across read positions is a weaker substitute,
  because pairwise overlap is nearly invariant to relabelling a fixed set of queries and the
  permutation's only effect comes through the changing far-source eligibility window, which
  confounds it. A named head is a memory head when its overlap exceeds the write-strength null
  *and* its queries move.
- **R4.** Secondary, reported but not decisive: mass shares on both channels against their
  uniform nulls, and `E[log g]`, `tau_head` and the interference constant per head on real text.
  These are the bound; `alpha` is the measurement, and where they disagree the measurement wins.

Gate statistics are additionally reported stratified at the decision positions and at the hidden
span, not only pooled: retention from `s` to `t` depends on the gates *between* them, so a head
with a short pooled `tau` can still retain one span whose interval happened to be open, and a
pooled average hides exactly that.

## 6. What the result cannot say

- Nothing about the routing path (section 1). A recurrent null is a content-transport null.
- Nothing about whether transported content is J-space content. WP3 needs no lens and gets no
  lens; "the workspace is carried by attention" needs WP3 composed with the hosted lens, which is
  WP5's job, not this one.
- Nothing about the 9B, whose band is a placeholder under R40b.

## 7. Cost and dependencies

No lens, no Jacobian, no training. One forward per context plus the scans: minutes on the 42
probe points once implemented. Runs while WP1 is under review, as the plan says. Depends on
nothing except the hook points; C1's layer table should be settled by the Chief before an
implementer starts, because it is the one error here that produces a plausible-looking wrong
answer rather than a crash.

## 8. Live calibration of the reconstruction gate (Chief, 2026-09-05 evening; R41a addendum)

Run under the Director's direct authorisation, no other model process alive, on the cached post-trained checkpoint: hooks on `gated_delta_update` as the block calls it, alpha rows by the backward scan with the h // 2 pairing, two EXP-001 contexts (1550 and 1546 tokens), seven positions each from 92 to the decision, all 24 recurrent blocks, all 32 value heads: 10,752 (context, block, position, head) cells. Script and log: `outputs/probes/jlens-hosted-qwen35-4b-2026-09-05/calibrate_alpha.{py,log,json}`. Wall time 17.6 s including the model load.

- **The block runs in float32 on the kernel path** (q, k, v, a, b and the output are all float32; `use_kernel` true), so the gate's dtype assumption holds on the real model.
- **Reconstruction.** Worst absolute error of y_t = sum_s alpha[t, s] v_s against the kernel's own output: 9.4e-7 (relative 2.3e-6); against the library's float32 ops path from the same inputs: 8.8e-7. Median 5.1e-10, 99th percentile 4.1e-8. Zero cells above 1e-4, 1e-3 or 1e-2.
- **Location (the Deputy's field).** The worst cell is context 0, block 0, head 5, position 1161. Block 0 is about ten times the rest on both statistics (median 6.0e-9 against 9e-11 to 8.9e-10 elsewhere; maximum 9.4e-7 against 6.5e-8 for the next block), which is a mild concentration and not none (corrected on the Head's reading of the artifact); it takes embedding input, which is an unremarkable reason, and it is irrelevant to the gate since nothing comes within two orders of 1e-5. Within every other block the errors are ordinary float32 accumulation.
- **Growth with position (the Deputy's second field).** Per-position maxima rise from 5.8e-8 at position 92 to 6e-7 at 1549 while medians stay at 4e-10 to 8e-10: mild accumulation, about tenfold over a seventeenfold length. Extrapolated to 2688 the worst stays below 2e-6.
- **Gate as ruled.** Absolute reconstruction error above **1e-5** in float32 fails a WP3 run, position-independent, with the measured curve above as the reference; 1e-4 in section 4 is superseded. A future run whose worst error rises above 1e-6 at these lengths is reported as a change even while it passes.
- **Gate constants on real text.** Median gate time constant over blocks 6.2 tokens (neutral-input figure was 2.4), maximum 24.4.
- **Transport preview, pooled over all 24 blocks and 14 positions (not the in-band, per-pair, per-head statistic WP3 proper reports; a calibration of the statistic, not its result).** Median mass share and median retention ratio to bin 1-4: bin 5-16: 0.096 and 0.36; 17-64: 0.109 and 0.106; 65-256: 0.076 and 0.023; 257-1024: 0.018 and 0.0011 (uniform null 0.55); 1025-2688: 5e-9 and 1e-9. The recurrent half of R1 is met in preview by two orders of magnitude at 257-1024. The tail R3 anticipates is real: the 90th percentile of the retention ratio is 0.52 at 257-1024 and 0.52 at 1025-2688, so about a tenth of (head, position) cells keep half their near-field weight at long range; 1042 of 7680 cells at 257-1024 and 1360 of 4608 at 1025-2688 underflowed to exactly zero. WP3 proper names those heads by block and head index, per R3, and stratifies by position and by the hidden span, per section 5.

## 9. The long-gap tail is real, and it is a head population (Chief, 2026-09-05 late evening; two further calibration runs, same artifact files)

The Head's objection to reading the retention ratio alone was correct: its denominator is the head's own near-field weight, a head that ignores its neighbours scores an arbitrary ratio, and the 90th percentile above one at bins 5-16 and 17-64 is that artifact. R3 is amended to the joint criterion (retention above 0.05 and mass share above the bin's uniform null) and every retention quantile is reported beside the same quantile of mass share, with the zero rule: a cell whose bin-1-4 mean is exactly zero has an undefined ratio and is excluded and counted (none occurred).

The check the objection asked for was run, and it does not come out the way the objection predicted. Per cell, at gaps 257-1024 the mass share's 90th percentile is 0.55 against a uniform null of 0.55 and 779 of 7,680 cells meet the joint criterion; at 1025-2688 the 90th percentile is 0.27 against 0.26 and 428 of 4,608 cells meet it. The tail is not a division artifact.

Why the gate bound did not forbid it: the bound used the block-level constant, -1 / mean(log g) over heads and positions, whose maximum was 24.4 tokens. A mean of log g over a heavy-tailed head population is dominated by its fast heads and says nothing about its slow ones. Computed per value head over each context, the constants are:

| percentile over 1,536 (context, block, head) entries | 1 | 10 | 25 | 50 | 75 | 90 | 99 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gate constant, tokens | 0.4 | 1.6 | 6.2 | 46 | 496 | 1,531 | 5,167 | 12,926 |

233 of 1,536 entries (15 percent) exceed 1,000 tokens and 507 (33 percent) exceed 256, and they sit in every recurrent block (per-block counts of heads above 1,000 range from 0 in block 4 to 11 in block 15, with blocks 6, 7, 9, 12, 15, 16 and 18 holding 7 to 11 each). The joint cells belong to these heads: their median gate constant is 1,023 tokens at 257-1024 and 1,823 at 1025-2688 against 28 to 30 for the other cells, their maximal gate product over a 1,024-token stretch has a median of 0.71 and 0.81 against a few billionths, and 402 of the 779 and 340 of the 428 joint cells come from heads above 1,000 tokens. 236 distinct (block, head) pairs at 257-1024 and 144 at 1025-2688 are joint at some position.

Consequences.
- **F4 is restated.** "Long-range transport in this model is attention-only" is false as a universal. Most recurrent heads have horizons of tens of tokens; a minority of about fifteen percent, present in every block including the band, keep their gate near one and read across the whole context with a mass share at or above uniform. The derivations memo's "gate constants median 2.4, max 51.7 on neutral input" was a pooled constant and is superseded by the per-head distribution above.
- **R1 stands on the median and R3's named exceptions are a population, not a tail.** WP3 proper names every joint head per block, head and R41b pair, with its gate constant and maximal gate product as fields (issue 81), and reports the fraction of in-band heads that are long-gate.
- **EXP-002 and EXP-003 predictions change.** The recurrent state can carry content from more than a thousand tokens back in these heads, so EXP-002's 2 by 2 is worth running at full size and its arm-A prediction is no longer a null; EXP-003's per-kind prediction becomes a per-head one, and WP5 must compose WP3's head list with the hosted lens to ask whether the long-gate heads carry J-space content.
- **Instrument rule (R38 form).** A time constant is reported per head, never pooled over heads; a pooled mean over a heavy-tailed population is the wrong-axis maximum in another dress.

## 10. Write-once memory heads, not stale smears: the R3b measurements (Chief, 2026-09-05 late evening; artifact re-run, same files, `calibrate_alpha5.log`)

The Head's mechanism question (section 9's population has low write strength, so is its state frozen and stale, or does it write rarely and retrieve?) was put to the two measurements R3b names, computed from the captured tensors with no new forward pass.

- **Write strength.** Rank correlation between a head's maximal gate product over 1,024 tokens and its median write strength beta: -0.84 (the Head's -0.83 confirmed). Joint cells' heads have median beta 0.057 at gaps 257-1024 and 0.047 at 1025-2688, against 0.31 and 0.30 for other cells. But their upper quantiles are not small: median over joint cells of the head's 99th-percentile beta is 0.61 and 0.64, of its maximum beta 0.97 and 0.97, and strong writes (beta above 0.5) occur at 2.5 percent of positions against 29 percent for other heads. Low typical write, rare hard writes.
- **Concentration.** Fraction of a joint cell's long-bin read mass carried by its ten largest source positions: median 0.144 at 257-1024 (uniform 0.013, eleven times) and 0.167 at 1025-2688 (uniform 0.027, six times); 90th percentiles 0.26 and 0.27. Every joint cell at 257-1024 and 425 of 428 at 1025-2688 exceed three times uniform. (Non-joint cells show high concentration too, 0.24 and 0.18, because their long-bin mass is negligible and a few residual positions dominate it; the comparison that matters is joint cells against their own uniform.)
- **Both together.** Joint cells with a write spike (99th-percentile beta above 0.5) and concentration above three times uniform: 564 of 779 at 257-1024, from 186 distinct (block, head) pairs; 333 of 428 at 1025-2688, from 113 pairs.

Reading. Because the read weight from a source is proportional to the write strength at that source, a head that writes hard at a few positions and holds its gate open reads those positions from far away with its mass concentrated on them; that is the write-once signature, and it is what the population shows. The smear alternative (uniformly small beta, flat mass) describes a minority: about a quarter of joint cells at the short long bin and a fifth at the long one lack the spike. The named heads are therefore candidates for write-once memory heads, not stale smears, pending the content question. Consequence held from section 9 and now released on the Head's own criterion: EXP-002's 2 by 2 goes to full size. What remains open, and is WP3 proper's job with WP5: which source positions the spikes fall on (tool outputs, file names, plan statements), whether those sources coincide with content the model later uses, and whether what these heads carry is J-space content. R3b's fields (beta's upper quantiles and the top-ten concentration, beside the gate constant and gate product) are on every named head.

## 9a. Per-head, not per-entry: the cross-context agreement (Chief, 2026-09-05 late evening, on the Deputy's correction)

Section 9's "233 of 1,536 entries" counts (context, block, head) cells over two contexts, and the Deputy is right that a per-head fraction follows only if the two contexts name the same heads; a gate is data-dependent and could be slow on one context and fast on another. Computed from the artifact: heads above 1,000 tokens number 116 in context 0 and 117 in context 1, 115 in both and 118 in either, Jaccard 0.97; above 256 tokens, 254 and 253 with 253 in both. The Spearman correlation of per-head constants across the two contexts is 1.00 and the absolute difference in log10 of the constant has median 0.00 and 90th percentile 0.01. Per block, the sets agree head for head (for example block 15: 11, 11 and 11; block 18: 10, 10 and 10).

So the population is a per-head property on these contexts: **115 of 768 value heads (15.0 percent) above 1,000 tokens in both, 253 (32.9 percent) above 256 in both**, denominators named. The stability is strong enough to suggest the constant is set mostly by each head's parameters (the per-head decay scale) with the input-dependent term second-order, but both contexts are agent transcripts from one task family; WP3 proper reports the same agreement across the sweep corpus and the ten transcripts, and a head is called long-gate only if it is above threshold on every context class measured. Every head fraction in this design is stated against 768 with the denominator in the same line (the Deputy's reporting requirement, issue 81).

## 10a. Section 10 withdrawn in part: the concentration null was wrong, the release is held (Chief, 2026-09-05 late evening, on the Head's R3c)

For a long-gate head the operator product is near the identity and the read weight reduces to the write strength at the source times the alignment of that source's key with the query. That form concentrates its mass on a few sources by itself: with beta fitted as a sigmoid-Gaussian to the joint cells' own statistics (median 0.057, 99th percentile 0.61, 2.5 percent above 0.5, all three reproduced by one smooth fit), a random alignment gives a top-ten source share of 0.138 at 768 sources and 0.216 at 400. Measured: 0.144 and 0.167, which is 1.05 times that null at 257-1024 and 0.77 times it at 1025-2688. The elevenfold and sixfold ratios in section 10 were against uniform, which is not what a non-selective head produces. The spike leg fails the same way: 2.5 percent of positions above 0.5 is the upper tail of the smooth fit, so a spike claim needs a demonstrated departure from unimodality, not a tail count. Section 10's reading is withdrawn and **EXP-002's full-size release is pulled back to held**.

Test adopted (R3c): source-set overlap. A write-once memory head draws every later read from the same stored positions, so the top-ten source sets at different read positions within one head coincide; a head whose weights are write strength times random alignment draws a fresh set each time. The overlap is content-independent and needs no forward pass. Two nulls, both reported: the hypergeometric floor, and a stronger one that keeps the head's real write strengths and keys and randomises only the query, because the shared write-strength factor induces overlap across read positions even with no memory and the hypergeometric floor alone would over-call memory. Memory is claimed only if the overlap exceeds the second null.

Standing rule, recorded as R44 in the wiring map at the Head's proposal: every threshold gets the null its own generative form implies. Three statistics today had a uniform or zero null where the natural variation was not: the 0.05 retention threshold, the linear-span share, and the top-ten concentration.

## 9b. Order is parametric, scale is set by the input: section 9a corrected (Chief, 2026-09-05 late evening, on the Deputy's reading of the derivations memo)

Section 9a called the neutral-input constants of the derivations memo (line 145: median 2.4 tokens, maximum 51.7, over the same 768 heads) a pooled quantity superseded by the per-head distribution. They are per head. The comparison is therefore real text against neutral input on the same heads: median 46 against 2.4, maximum 12,926 against 51.7, and at neutral input not one head is above 1,000 tokens. The input term of the gate, softplus(a + dt_bias) with a = in_proj_a(x), moves every constant by one to two and a half orders of magnitude on agent transcripts; that is first-order, and the long-gate population is created by it.

The two results fit: the constant is 1 / (exp(A_log_h) * E[softplus(a_h + dt_bias_h)]). If the input contribution is close to common across heads within a context it scales every constant by one factor and leaves the ranking intact, which is exactly what Spearman 1.00 and a median absolute log10 difference of 0.00 across two contexts of one task family look like. The head parameter sets the order; the text sets the scale; the threshold cuts the scaled values. Section 9a's agreement statistics are evidence about the order and none about the scale, and "mostly parametric" is withdrawn as a statement about the count.

Fields added to issue 81 by the Deputy and adopted here: each head's measured constant as a ratio to its neutral-input constant, per context, with the spread of that ratio across heads (near-common spread means the count is one number per context); the long-gate count at neutral input, zero, printed beside the count on every context class; and the naming rule: long-gate only if the head clears the threshold on every context class measured. The sweep corpus, being wikitext-like rather than agent transcript, is the first context class that can move the scale, and its count is the number that decides whether F4's restatement is about this model or about this model on agent transcripts.

## 10b. The overlap test: stable far-source sets beyond what write strength alone induces (Chief, 2026-09-05 late evening; `overlap_test.py`, `overlap_test.json`)

R3c's test, run on the 115 heads above 1,000 tokens on both contexts (230 head-context entries), at read positions 50, 75, 90 and 100 percent of each 1,550-token transcript, top-ten source sets among sources at gap 257 or more, overlap restricted to sources eligible for both read positions. Two nulls, as section 10a required.

| statistic | value |
| --- | --- |
| mean overlap of top-ten far-source sets across read positions, real | 3.24 of 10 |
| hypergeometric chance | 0.10 |
| query-randomised null (real write strengths and keys, random query of matched norm, 12 draws) | 1.68 |
| entries with real overlap above the query null | 219 of 230 |
| entries above twice the query null | 113 of 230 |
| median ratio, real to query null | 1.98 |

Reading. The query-randomised null at 1.68 confirms the Head's point that shared write strength alone induces overlap far above hypergeometric chance; the real overlap sits about twice that null in half the entries and above it in 95 percent. So the far sources these heads read are more stable across read positions than write strength times random alignment predicts, which is the signature of retrieval from a fixed set of stored positions rather than a fresh draw at each read. One null remains, and it is an acceptance item for WP3 proper: a head whose queries barely change across positions would show stable sets without content-specific retrieval, so the head's real queries are permuted across read positions as a third null; memory in the content sense is claimed only if the real overlap exceeds that null too. On the two nulls specified in R3c the result is high, and **EXP-002's 2 by 2 is released to full size on that basis, with the read-share pre-check first and the third null reported before the 2 by 2's reading is written.**

What the overlap does not say: which positions the stable sources are (tool outputs, file names, plan statements) and whether their content is what the model later uses; both are WP3 proper's job with WP5.

## 10c. Release held again: the overlap is between the hypotheses and the query cosine decides (Chief, 2026-09-05 late evening, on the Head's R3d)

Calibration of the overlap statistic on the same generative form (the Head): top-ten far-source overlap is 6.96 of 10 for a frozen query, 4.29 at a query cosine of 0.9, 2.02 at 0.6, about 1.5 at independence. The measured write-strength-only null of 1.68 lands at independence, so the model and the measurement describe the same object; the measured 3.24 corresponds to a query cosine near 0.8. A head whose query direction is that stable reads the same far sources at every position with no content specificity, a static filter, and nothing measured so far excludes it; nor confirms it, since 3.24 is well below the frozen-query 6.96. The third null changes form: permuting real queries across read positions is nearly invariant to relabelling and confounded with the eligibility window, so the direct measurement is the median pairwise cosine between a head's queries at different read positions. Near 0.8: static filters, overlap explained. Low: retrieval, the finding stands. **The full-size release of section 10b is held pending that number**; the read-share pre-check runs first regardless.

## 10d. The query cosine: static filters, and the release stays held (Chief, 2026-09-05 late evening; `query_cosine.py`, `query_cosine.json`)

Median pairwise cosine between a head's queries across positions, long-gate heads (the 115, both contexts, 230 entries) against the other 1,306 entries, queries taken as passed to the update after the convolution, normalisation and scaling:

| statistic | long-gate heads (p10 / p50 / p90) | other heads (p10 / p50 / p90) |
| --- | --- | --- |
| query cosine across the four read positions | 0.68 / 0.83 / 0.99 | 0.54 / 0.82 / 0.97 |
| query cosine across 60 positions spread over the context | 0.41 / 0.68 / 0.97 | 0.31 / 0.61 / 0.89 |
| key cosine across the same 60 positions | 0.19 / 0.51 / 0.80 | 0.14 / 0.34 / 0.83 |
| long-gate entries with read-position query cosine above 0.6 | 219 of 230 | |
| below 0.3 | 2 of 230 | |

Reading. The Head's calibration put the measured overlap of 3.24 at a query cosine near 0.8; the measured cosine at the read positions is 0.82 at the median. The overlap is explained by query stability: these heads point a nearly fixed query at a nearly fixed key direction (their keys are also more stable than other heads', 0.51 against 0.34) and read the same far sources at every position because the query barely moves, not because they retrieve content. They are **static filters**, long-range channels that carry a write-strength-weighted running summary of value content along a fixed direction across the whole context. That is a real long-range recurrent channel, and it is not addressable memory. F4's restatement stands in this form: fifteen percent of value heads carry long-range aggregate content on agent transcripts; none of them has been shown to retrieve specific positions.

Two further points. Query stability is high across all heads in this model: at the four read positions the other heads' median cosine is 0.82, indistinguishable from the long-gate heads' 0.83, and over 60 positions it is 0.61 against 0.68. So the fixed-query property is general to these blocks and the long-gate heads differ from the rest by the gate, and somewhat by key stability, not by the query; content-varying queries are the exception in these blocks; and the overlap re-run at 200 vectorised null draws keeps the headline (3.24 against 1.68) and adds the per-pair view the Deputy asked for: 550 of 1,380 read-position pairs above the query null's 90th percentile (chance about 138) and 274 above its 99th (chance about 14), with provenance now inside the file.

Consequences. **The full-size release of EXP-002's 2 by 2 stays held; the 2 by 2 runs at the size the read-share pre-check licenses**, and the pre-check is now the right instrument, since it measures directly how much of the hidden span these channels carry into the decision read. WP3 proper reports the query cosine and the key cosine as fields on every named head, beside the gate constant, the gate product, the write-strength quantiles and the overlap against the query null; a head is called a memory candidate only if its overlap exceeds the query null with a query cosine below 0.3, and no head met that tonight.

## 10e. Per head, not per median: overlap crossed with query cosine (Chief, 2026-09-05 late evening, on the Deputy's correction; `overlap_by_cosine.json`)

Section 10d classified the population by its median query cosine, which is the pooled-constant error one statistic further on: the overlap percentiles (1.83, 2.50, 3.17, 4.00, 4.83 at the 10th to 90th) read through the Head's calibration span cosines from about 0.5 to above 0.9. So the cross is per head-context entry, overlap ratio to the query null against the read-position query cosine, over the 230 entries.

| criterion | retrieval-like (high overlap, low cosine) | static (high overlap, high cosine) | at null, low cosine | at null, high cosine |
| --- | --- | --- | --- | --- |
| ratio > 1.5, cosine < 0.6 | 4 | 166 | 7 | 53 |
| ratio > 1.5, cosine < 0.5 | 0 | 170 | 5 | 55 |
| ratio > 2.0, cosine < 0.6 | 2 | 112 | 9 | 107 |
| ratio > 2.0, cosine < 0.5 | 0 | 114 | 5 | 111 |

The five entries with cosine below 0.5 have overlap ratios of 0.13 to 1.25, at or below the null; the two entries above ratio 2 with cosine below 0.6 sit at 0.58 and 0.60 (context 1 block 7 head 31; context 0 block 9 head 13) and are borderline on both axes. Spearman correlation of overlap with read-position cosine across entries is 0.25, so the population's overlap spread is real and mostly comes from other head properties (key stability, write-strength distribution, eligible window), and none of it lands at the retrieval end.

Consequence, now on a per-head basis: **the retrieval quadrant is empty at every threshold tried; the 2 by 2 runs at the size the read-share pre-check licenses.** WP3 proper reports this cross for every named head on every context class, and the memory criterion of section 10d (overlap above the query null and query cosine below 0.3) stands, with the two borderline heads named for inspection of their source positions.
