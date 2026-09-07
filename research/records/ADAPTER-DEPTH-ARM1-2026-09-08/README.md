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
