# QUANT-GAP-2026-09-05: the hosted lens on the bfloat16 checkpoint against the 4-bit one

**Question.** Every hosted-lens result on the record (HOSTED-JLENS-QWEN35-4B-RESULTS-2026-09-05, band
signatures, the 42-case tables, the reaction contrast) was read on `mlx-community/Qwen3.5-4B-MLX-4bit`,
while the lens was fitted on the unquantised post-trained model. How much of what was seen is the model
and how much is the quantisation? The Director approved the 8.48 GiB pull of
`mlx-community/Qwen3.5-4B-MLX-bf16` at 23:04 on 2026-09-05.

**Method.** `run_hosted_lens.py` (the scratchpad copy that produced the 4-bit results; guarded record copy
in `research/records/jlens-hosted-qwen35-4b-2026-09-05/`) run on the bfloat16 snapshot
(`475632ded9a95863da4e4b235ab9ccbc5d3cc6bf`) with the same n1000 lens, the same nine readout layers
(5, 11, 12, 16, 20, 21, 27, 28, 32), the same 42 regenerated cases (seeded; identical task ids), the same
16-context corpus, through `evaluate.load_policy` and the model-run lock; 557 s. `quant_gap.py` pairs the
two outputs case by case, layer by layer, and self-tests to correlation 1 and difference 0 on the 4-bit
output against itself.

**Result: the workspace readings are the model's, not the quantisation's.**
- Layer similarity (CKA over 32 x 32 layers): max |difference| 0.004, mean 0.001.
- Corpus statistics per layer (top-k hit rate, kurtosis, top-1 persistence and its null, effective
  dimensionality): agree within a few percent at every layer; the persistence-above-null band and the
  dimensionality curve coincide (bfloat16 crosses the null one layer earlier, at 5 rather than 6).
- The 42 cases, matched condition, hosted readouts in band: correlation of log P(true) across cases 0.79
  to 0.98 (weakest at layer 16, 0.79 Pearson and 0.53 Spearman), sign agreement of the true-versus-false
  margin 0.86 to 0.95, mean |difference| in log10 P(true) 0.08 to 0.32 (largest at layers 11, 27, 28, about
  a factor of two in probability). The matched-wins counts move by at most three of 42.
- **The model's own output disagrees more than its workspace does**: at the final layer and in the
  model output the correlation is 0.66 and the sign agreement 0.81, the lowest of any readout. The 4-bit
  quantisation perturbs the answer distribution more than it perturbs the mid-layer content the lens
  reads.
- Sanity prompts: `spider`, `rhyme` and `france` are fixed and compared by the first layer at which the
  lens surfaces each target (table in the log of this record); `arith` is regenerated with fresh numbers
  each run and is not paired, which `quant_gap.py` states rather than compares.

**Consequences.** The hosted-lens memo's findings stand as findings about the model. Where a result
rests on the final-layer probabilities rather than on the lens readings, the quantisation is a
factor-of-two effect and should be stated. The §6 reaction contrast (Base against post-trained) was
not re-run here; it compares two 4-bit models and the gap applies to both sides equally.

**Files.** `out-bf16-n1000/` (the run's eight outputs), `quant_gap.py`, `quant_gap.json`,
`quant_gap.md`, `lens-bf16.log`. The training-step cost on the bfloat16 base, the second use of the
pull under R48, is appended below when its rows land.

## Addendum, 23:31: the training step on the bfloat16 base (R48, training efficiency)

`train_cost_probe.py` variant A (the trainer's own recipe: LoRA rank 16 on all layers, chunkwise
recurrence, gradient checkpointing, AdamW, `default_loss`) with `--hf-id` pointing at the bfloat16
snapshot, one process per point through `load_policy` and the lock, 0.6 working-set cap. Rows in
`rows-train-cost-bf16.jsonl`, `row-A-bf16-*.json`, `launcher-train-cost-bf16.log`.

| tokens | bfloat16 base peak (GiB) | 4-bit base peak (GiB) | bfloat16 step (s) | 4-bit step (s) | bfloat16 throughput (tok/s) |
| --- | --- | --- | --- | --- | --- |
| 256 | 9.01 | 3.41 | 1.6 | 1.6 | 161 |
| 512 | 9.63 | 4.17 | 3.1 | 3.3 | 166 |
| 768 | 10.23 | | 4.6 | | 166 |
| 1,024 | not run: projected 10.82 GiB, 0.61 of the working set | 5.30 | | 7.0 (147 tok/s) | |

**Reading.** The per-token slope is the same on both bases, 2.44 MiB per token on bfloat16 against
2.40 on 4-bit, so the training memory slope is not a property of the quantised weights; what
changes is the intercept, about 5.6 GiB more for the unquantised weights (8.48 GiB on disk against
about 2.3). The bfloat16 base trains 8 to 13 percent faster per token (166 against 147 to 154),
so the 4-bit base is the smaller trainer and not the faster one: quantised matrix multiplies cost
time in the step. Row ceiling on bfloat16 at the working set: about 3,900 tokens, against about
6,400 on 4-bit; at the recipe's 2,688-token row the bfloat16 step projects to about 14.6 GiB,
0.82 of the working set, feasible for an announced training run under R47(c) but above the
usability cap during the Director's hours. Recorded as the second use of the pull; the trade is
about 10 percent throughput for about 5.6 GiB of headroom, and the choice belongs to the recipe.
