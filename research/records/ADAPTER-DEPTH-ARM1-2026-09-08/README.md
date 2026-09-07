# Issue 88, arm 1 (top 8): the lever is real, the memory saving is four times what was fitted, and the run trained past its own best

2026-09-08, Deputy Chief of AI Research. 1,200 of 1,200 iterations in **100.7 minutes**, status ok,
verdict warnings. `configs/agent_v2e_qwen35_4b_top8.yaml`, which differs from arm A's recipe by two
lines: the output directory and `train.lora_layers: 8`.

## Throughput: the probe's ratio holds

| | it/s |
| --- | ---: |
| arm 1, top 8, median over 120 reports | **0.2070** |
| arm A, all 32, 780 rows in 8,030 s | 0.0971 |
| **speedup** | **2.13×** |
| the probe's figure, 276/126 | 2.19× |

**So the probe's ratio essentially holds on the real dataset**, and my mid-run claim that it did not
was drawn from three warm-up reports and is withdrawn. The correction cost nothing but it is the
second time in this run that a number from a partial window read as a finding, which is why the
rule now is to publish none.

## Memory: the depth term is about four times what issue 88 fitted

| | GiB |
| --- | ---: |
| arm A, all 32 layers, long run | 9.726 |
| arm 1, top 8, long run | **8.335** |
| raw difference | 1.391 |

Both are long runs on the same dataset at the same accumulation, so the difference is depth plus
whatever their allocator growth differs by. Correcting for that (below): **1.216 GiB**, which at
2,688 tokens over 24 removed layers is **0.0193 MiB per token per layer**.

**Issue 88 fitted 0.0054.** The measurement is **3.6× that**, and the issue's headline — "84 percent
of the backward's cost at full depth does not come from the adapted layers" — understates the
lever's memory value by the same factor. It is still true that depth is not the main term; it is
not true that restricting depth "cannot buy meaningful memory". At this recipe it buys 1.2 GiB, a
seventh of the peak.

My own R47(b) projection of 9.39 GiB used the fitted term and was **1.05 GiB high**. Conservative,
which is the right direction for a gate, and wrong by enough to matter for planning.

## The allocator correction, and its own uncertainty

Arm 1's peak rose in six steps: **four early** (iterations 20, 30, 40, 70; +0.432, +0.443, +0.569,
+0.741 GiB) and **two late and tiny** (230 and 300; +0.001 and +0.045). Issue 94 attributed arm A's
late steps — everything past the widest buildable row — to allocator growth totalling 0.221 GiB.
Arm 1's late total is **0.046 GiB**.

So arm 1 accumulated **0.175 GiB less allocator growth than arm A**, and that much of the raw
difference is not depth. The corrected figure above subtracts it. The correction assumes arm A's
0.221 and arm 1's 0.046 are comparable measures of the same phenomenon at two depths, which is a
reading rather than a measurement; the raw 1.391 and the adjusted 1.216 bracket the answer, and
both are three to four times the fitted term.

That fewer adapted layers also means less allocator growth is a second observation worth a line: it
is consistent with fragmentation across live gradient blocks, and arm 2 at 16 layers gives it a
third point.

## The validation curve, and what it does and does not say

| iteration | val loss |
| ---: | ---: |
| 1 | 0.721 |
| 400 | 0.194 |
| 800 | **0.011** |
| 1,200 | 0.140 |

**A 12.7× rise after the best point.** Read with the 29 loss-spike flags, the first at iteration 310
and rising in frequency after, the shape is ordinary overfitting: the run reached its best at 800
and the last 400 iterations made it worse.

Three things stop that being a finding yet.

**The validation set is 48 rows** (`valid` 24 plus `valid2` 24), so a single reported figure is
noisy and 0.011 against 0.140 is four points on a small sample, not a curve.

**0.011 is low enough to be suspicious in itself.** On 48 rows that is nearer memorisation than
generalisation, and the honest reading of the pair is that the 800 point may be as untrustworthy as
the 1,200 one.

**Arm A cannot answer it.** It stopped at 780 of 1,200 and reported validation twice, at iterations
1 and 400 (0.152 and 0.092). **It has no 800 or 1,200 point**, so the comparison the question asks
for cannot be made against it. This is the same reason arm A cannot serve as the full-depth arm of
the ablation, now confirmed from its record rather than argued from its status: the three arms must
be identical but for depth, and arm A differs in iterations run.

What the curve does say, and it matters for the remaining arms: **selection should take the 800
checkpoint, not the last**, and if the same shape appears at 16 and 32 layers then 1,200 iterations
is past the useful length for this recipe and the ablation should say so.

## Loss spikes

| | flags | per 100 iterations |
| --- | ---: | ---: |
| arm A, 780 iterations | 15 | 1.9 |
| arm 1, 1,200 iterations | 29 | 2.4 |

Comparable rates, arm 1's slightly higher, and its first at iteration 310 against arm A's 140. Not
enough to separate depth from run length with two arms.

## Files

- `arm1-health.json`, `arm1-provenance.json`, `arm1-train-stdout.txt` — the run's own artefacts.
- `trace.json` — the derived figures above, with every input, so each number can be recomputed.

---

## Correction and hypothesis, appended 2026-09-08 after the Chief's review

### The 2.13× mixed two estimators

Arm 1's 0.2070 is a **median of the reported per-report rates**; arm A's 0.0971 is a **wall-clock
average**. Like for like, both ways:

| estimator | arm A | arm 1 | ratio |
| --- | ---: | ---: | ---: |
| median of reported it/s | 0.1010 (78 reports) | 0.2070 (120 reports) | **2.05×** |
| wall clock, iterations over elapsed | 780 / 8,030 s = 0.0971 | 1,200 / 6,044 s = 0.1985 | **2.04×** |
| the mixed figure published above | — | — | 2.13× |

**2.04× is the number.** The conclusion is unchanged and slightly weaker: the probe's 2.19× holds,
with the measured ratio a little under it rather than a little over.

### Why the depth term might be 3.6× the fitted one, and what would show it

**The hypothesis.** With no trainable parameter below the lowest adapted layer, the backward stops
there, so the frozen layers below need retain nothing. Then the per-layer cost is a cost **per
retained layer**, and removing 24 layers from the retained set removes 24 layers' worth of retained
activation — a whole-peak quantity, not a marginal slope.

**What the probe measured, which is not the same thing.** The 0.0054 comes from
`TRAIN-COST-2026-09-05`, variants M and N: adapters on the top 8 and 16, **one row, one optimiser
step**, at the probe's fixed row lengths, and the fitted quantity is the **backward's addition per
token** — 0.91, 0.96 and 1.04 MiB/token at 8, 16 and 32 layers — decomposed as 0.87 independent of
depth plus 0.0054 per layer. That is a per-layer contribution to **one component's marginal slope**.

Mine is the difference in **total peak** between two long runs at the same 2,688-token cap, at
accumulation 4, on the real dataset. **The two quantities are not the same measurement**, so the
3.6× is a discrepancy between two things rather than one correcting the other, and the record
should not be read as saying the probe was wrong about what it measured.

### The test arm 2 performs

Under the per-retained-layer reading, top 16 retains 8 more layers than top 8:

    8.335 + 8 × 0.0193 × 2,688 / 1,024 = 8.74 GiB, before allocator growth

**Near 8.74 says the saving is per retained layer.** Near 8.34 says it is something else — a fixed
cost of adapting at all, paid once whatever the depth. Arm A's 9.726 is the third point and it sits
where the linear reading predicts (8.335 + 24 × 0.0193 × 2.625 = 9.55, before allocator growth,
against 9.726 with it), which is consistent but is the point the term was fitted to and therefore
proves nothing on its own.

**Second thing arm 2 tests**: allocator growth against depth. Arm 1's late steps total 0.046 GiB
against arm A's 0.221. If it scales with retained layers, arm 2 lands near 0.10; if it is a
property of the run rather than the depth, near 0.22.

Both projections go in arm 2's announcement per R47(b), before it runs.
