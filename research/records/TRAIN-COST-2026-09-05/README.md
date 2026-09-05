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
