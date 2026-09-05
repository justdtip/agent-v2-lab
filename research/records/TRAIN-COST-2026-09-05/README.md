# TRAIN-COST-2026-09-05: one training step per row length, the trainer's own recipe (R48d, issue 85)

**Question.** Is a 32k-token training row feasible on this machine, and where does the training
memory slope come from? (Director, 2026-09-05 21:50; R48c standing order: training efficiency first.)

**Answer.** No, by a factor of about five. As the trainer stands (variant A) one step costs
2.40 MiB of peak memory per token of row (least squares over 1,024 to 4,096 tokens; segments
2.1 to 2.5), crossing the 17.76 GiB working set at about 6,400 tokens and the 85 percent launch
cap at about 5,300. A 32k row projects to about 78 GiB. The chunked cross-entropy variant (B),
which never materialises the full logits, saves 0.24 MiB per token, a tenth of the slope, at no
time cost; the arithmetic that predicted 1.89 MiB per token for the vocabulary term is refuted.
About 2 MiB per token is unexplained by the logits, the layer-boundary residuals (0.16) or the
attention scores (quadratic; too small below 4k). Step throughput is 126 tokens per second at the
recipe's 2,688-token row and 102 at 4,096, compute-bound at the expected training-to-inference ratio.

**Provenance.** Model `mlx-community/Qwen3.5-4B-MLX-4bit` via `evaluate.load_policy` (model-run
lock taken per process, 3bc0842); recipe from `configs/agent_v2b_qwen35_4b.yaml` (LoRA rank 16,
scale 32, all layers, chunkwise gated-delta recurrence chunk 64, gradient checkpointing, AdamW
3e-5, `mlx_lm` 0.31.3 `default_loss`); MLX 0.32.2; Apple M4 Pro 24 GiB, working set 17.76 GiB.
One process per point; two steps per point, the second reported; peak from `mx.get_peak_memory`
after reset following load and LoRA setup. Launcher: projection cap 0.85 of the working set from
the last two points; guard on `kern.memorystatus_vm_pressure_level` critical on two consecutive
10 s samples (never fired; pressure stayed at warning or below). Run 22:24 to 22:31 by the Chief
under the standing lift, the Director's request for this probe being the R47 window.
Validation at 256 tokens: both variants take a real step (loss 14.47 then 13.53); B matches A's
loss to three decimals.

**Files.** `rows.jsonl` (every point, including the two projection stops), `row-*.json`,
`summary.json` (fits, ceilings), `launcher.log` (progress and guard lines), the three scripts as
they ran, `training-row-cost.html` (the chart page; also published as a private artifact).

**Next.** The same step with gradient checkpointing off, and with the recurrence's backward
removed, to the recipe's row length, to locate the 2 MiB per token. Preflight's chunkwise envelope
(one point at 2,874 tokens) is replaced by these five points under issue 85.

## Addendum, 22:36 to 22:43: probes 2 and 3 (variants C, D, E, F); rewritten 22:56 after the first copy vanished from the tree

Same method, capped at 0.6 of the working set, 256 or 512 to 2,688 tokens, one process per point
through `load_policy` and the lock. Rows in `rows-probe2.jsonl` (C, D) and `rows-probe3.jsonl`
(E, F); logs in `launcher-probe2.log`, `launcher-probe3.log`; per-row files `row-C-*`, `row-D-*`,
`row-E-*`, `row-F-*`. The scripts as they ran are `*.as-run.py`; `train_cost_probe.py` and
`run_train_cost_probe.py` here carry the Deputy's record guard.

| variant | change against A | slope, MiB per token | step at 2,688 tokens |
| --- | --- | --- | --- |
| A | the trainer as it stands | 2.40 | 21.4 s |
| B | chunked cross-entropy, full logits never resident | 2.16 | 21.0 s |
| C | gradient checkpointing off | 13.9 (256 and 512 only; stopped by projection) | faster by a quarter at 512 |
| D | recurrence's backward removed (`stop_gradient` at the gated-delta output) | peaks identical to A to the hundredth | 15.8 s |
| E | no LoRA on gate, up or down | 2.19 | 20.9 s |
| F | all 496 adapter tensors cast to bfloat16 | 2.22 | 20.5 s |

**The deltas are not additive.** Each is a single change against A. E and F act on the same
MLP adapter path (E removes it, F changes its dtype), so their 0.21 and 0.18 are largely the same
fifth of a MiB, not 0.39. B (the loss) and D (the recurrence) are independent of that path and
of each other. Subtracting all four from A gives 1.63 and means nothing.

**What the rows say.** Checkpointing is called and works: off, the step costs 13.9 MiB per
token, so the hook discards about 11.5 of it. The recurrence's backward is 26 percent of step
time and none of the memory: stopping the gradient through 24 of 32 blocks freed nothing to the
hundredth of a GiB. The logits, the MLP adapter path and the adapter dtype are each worth about
a tenth. The float32-adapter hypothesis (predicted 2.41 for the MLP's gate and up outputs
promoted to float32) is refuted: measured 0.21. What survives all four removals is about 2.2 MiB
per token; the Head's bf16 accounting of what every layer's backward needs is 2.50, or 1.94 once
D removes the recurrence's 0.56 from it, so the fit is close but carries a quarter-MiB gap, the
same size as every intervention measured here, and is not claimed as a match.

**What is open, and under test.** Whether the 2.2 is held for all thirty-two layers at once
(the Head's model-free SwiGLU stack shows checkpointing does *not* accumulate across depth, peak
above parameters flat from 8 to 64 blocks) or is depth-independent and forward-time. Variants G
(staged gradient evaluation), J (forward only, training mode, no gradient), K (forward only,
inference mode) and L (sixteen of thirty-two layers checkpointed) run at 22:56; rows in
`rows-probe4.jsonl` when they land. The convexity on A's rows is one attention block's scores and
probabilities under recompute, two to three T-squared matrices, not eight blocks; it goes into
issue 85 as a named term with the linear residual fitted around it.
