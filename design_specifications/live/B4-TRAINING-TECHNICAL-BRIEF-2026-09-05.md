# Technical brief: the B4 training arm (Qwen3.5-4B on run B's data)

For the Research Director, from the Chief. 2026-09-05. Every number below was read from the
files named, not from the checklist; the token statistics were measured with the Qwen3.5
tokenizer (a tokenizer load, no weights).

## 1. What the arm is

| | |
| --- | --- |
| Hypothesis | H2 (SPEC-003 §0): a stronger base on **unchanged** data beats the 3B. One variable changed from the reference run B. |
| Pre-registered criterion (config `criteria:`) | total higher than run B with McNemar p < 0.05 on paired task ids, or a family-level gain ≥ 5 in one long-horizon family with no loss elsewhere. Recorded, never used for selection. |
| Reference to beat | run B on the 3B coder base: 145/180 on the 180-task test split; 47/60 on the 60-task split this config evaluates. |
| Command | `uv run agent-pipeline --config configs/agent_v2b_qwen35_4b.yaml train` |
| Output | `outputs/agent-v2b-qwen35-4b/` (adapters, checkpoints every 100, `train.log`, `metrics.jsonl`, `run.log`, `events.jsonl`, `health.json`, `provenance.json`) |

## 2. The model

| | Qwen3.5-4B (`mlx-community/Qwen3.5-4B-MLX-4bit`) | 3B reference (run B) |
| --- | --- | --- |
| Layers | 32: 24 Gated DeltaNet (linear attention) + 8 full attention, one in four | 36 dense |
| Hidden / vocab | 2,560 / 248,320, tied embeddings | 2,048 / 151,936 |
| Weights | 4-bit, group size 64, about 2.9 GB on disk | 4-bit |
| Thinking | supported; **off** for this arm: every prompt ends `<|im_start|>assistant\n<think>\n\n</think>\n\n` | unsupported |
| Preflight | passed under R18a at schema 2 (native-dtype view loop exact; float32 deviation 5.5% reported, not gated) | passed |
| Cross-turn cache | `none` at evaluation (DeltaNet cache is not trimmable; snapshot strategy not yet equivalence-verified). Training is unaffected; evaluation is slower per task than the 3B's trimmed cache | trim |

## 3. LoRA and optimiser (from `configs/models/qwen35-4b.yaml`, the arm config, and the preflight artifact)

| | |
| --- | --- |
| Targets (policy `auto` → `all-linear` on a hybrid) | attention blocks: q, k, v, o; MLP: gate, up, down; DeltaNet blocks: `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj` |
| Rank / scale / dropout | 16 / 32.0 / 0.0 (unchanged from the 3B recipe) |
| Trainable parameters | 32,464,896 (about 0.8% of the model; the 3B had 29.9M) |
| Layers adapted | all 32 (taken from the resolved view; the `num_layers: 36` in the YAML is overridden) |
| Optimiser | AdamW, learning rate 3e-5, no schedule |
| Batch | 2 sequences × 2 accumulation steps = effective 4 |
| Iterations | 400 → 1,600 rows seen ≈ 0.84 of an epoch |
| Sequence limit | 2,688 tokens |
| Gradient checkpointing | on |
| Validation | every 100 iterations on 24 batches; checkpoint saved every 100 |
| Memory estimate | 3.85 GiB weights against a 22 GiB budget; activations at batch 2 × 2.7k tokens with checkpointing are the real cost, expected well inside budget (the 3B peaked at 9.4 GB) |

## 4. The data (`data/agent_v2b-qwen35-4b`, rendered from `data/agent_v2b` by the R21 render stage; task content byte-identical to run B's)

Composition of `train.jsonl` (1,915 rows):

| Source | Rows | Notes |
| --- | --- | --- |
| Expert trajectories | 1,675 | 240 tasks; 374 clean-variant rows; the rest carry one injected fault (transient 380, wrong_path 382, failed_edit 240, unknown_tool 155, stale_path 144) |
| of which recovery targets | 367 | the corrective decision after a failed call, oversampled ×2 (wrong_path, unknown_tool) and ×6 (stale_path, failed_edit) |
| Chat replay | 240 | generic assistant prompts answered by the **old 3B agent adapter** as teacher (see §6) |

By family (expert rows): aggregate_report 286, ledger_reconcile 246, conditional_update 206,
batch_update 196, cross_reference 153, pointer_chain 133, update 116, synthesis 93, list 73,
search 73, read 50, calculate 50. Validation 284 rows; test 494 rows (60 tasks at difficulty 2).

## 5. Context window and what is actually supervised

Measured under the Qwen3.5 tokenizer on the rendered rows:

| | Train | Valid |
| --- | --- | --- |
| Prompt tokens: mean / median / p95 / max | 879 / 824 / 1,592 / 2,042 | 966 / 1,015 / 1,824 / 2,421 |
| Completion tokens: mean / p95 / max | 77 / 197 / 215 | 83 / 207 / 214 |
| Prompt + completion max | 2,085 | 2,464 |
| Rows over the 2,688 limit | 0 | 0 |
| Supervised share of all tokens | **8.0%** | 7.9% |

So the 2,688 limit is never hit, and the loss is computed on about 147,000 completion tokens
across the whole run: the note plus the fenced JSON call, roughly 60–80 tokens per expert row.
The 800–1,600 prompt tokens per row are the system prompt (about 360 tokens), the task, the
earlier notes, and the last two tool observations verbatim; older observations are one-line
stubs. That is the paradigm: the prompt is context, the note is the target.

Rows that exceeded the limit would have their completion truncated by the loader
(`tuner_data.py:78-80`); none do.

## 6. Representative rows (verbatim from the rendered file)

**Expert, `ledger_reconcile`, clean, step 3.** Prompt tail is the second invoice's contents
(eleven distractor lines, then `amount=67` / `status=held`), then the generation suffix.
Completion:

```
Invoices read: 2 of 4. approved: 37; held (skip): 67. Reading invoice-2-227.txt; pending: invoice-3-885.txt.
```json
{"name": "read_file", "arguments": {"path": "lab/train/0223/ledger/invoice-2-227.txt"}}
```<|im_end|>
```

The note carries the running list, the count against the total, the file being read, and the
enumerated remainder; the listing that named those files is already hidden.

**Expert, `conditional_update`, `stale_path` recovery target, step 5.** The previous
(unsupervised, deliberately failing) turn read `service-8.ini` and got `file not found`.
Completion:

```
That path does not exist and the earlier listing is no longer visible, so list the directory again for the exact names instead of guessing.
```json
{"name": "list_files", "arguments": {"directory": "lab/train/0201/services"}}
```<|im_end|>
```

**Chat replay, request "Write a small typed Python function that returns the even integers from
zero through 14 ..."** Completion:

```
!DOCTYPE html
<html>
<head>
  <title>Train-0160: Even Integers from 0 to 14</title>
...
  <p>Output: [0, 2, 4, 6, 8, 10, 12, 14]</p>
</body>
</html<|im_end|>
```

## 7. Findings you will want to act on

1. **The chat-replay rows are largely junk, and run B trained on them too.** Of the 240 rows,
   127 completions are HTML pages regardless of the request, and all 61 code-style requests
   have no code in their completion. These were generated by the earlier 3B agent adapter as
   "teacher" for conversational retention. For **B4 they must stay**: the arm is the
   controlled comparison and its data is run B's byte for byte; changing them would add a
   second variable. For the **D arms** (SPEC-003) the decision is yours: regenerate the 240
   rows from a competent source (the Qwen3.5 base itself with thinking off is the obvious
   one), or drop chat replay and measure retention directly. Either is a spec amendment.
2. **Eight percent supervised share** is inherent to the paradigm and matches the 3B runs, but
   it means each iteration's gradient comes from about 300 target tokens. Ideas if you want
   more signal per step: pack short rows, or raise the batch to 4 with accumulation 1 if
   memory allows on the 4B (preflight suggests it does). Not for B4; a D-arm variable.
3. **No learning-rate schedule** and a run shorter than an epoch: the 3B's validation loss
   bottomed at step 300 and rose by 400, which is why selection now screens every checkpoint
   behaviourally with validation loss as a tie-break. Expect the same shape here.
4. **Thinking is off by rendering**, so the model is trained never to think before the note.
   The evaluation-only `D4-think` arm (thinking on at inference, same adapter) is how to test
   whether intra-turn reasoning helps without retraining.
5. **Evaluation on this config is the 60-task split**, matching run B's 47/60, not the
   180-task split. If you want the 180-task comparison (145/180) for B4, set `eval.limit: 180`
   before the evaluation lift; it costs about 70 more minutes and is the better-powered test.

## 8. What the run will show and write (R26)

A start line naming model, config, data-manifest hash and git commit; a progress line every
10 iterations with loss, learning rate, tokens/s, peak memory against budget and ETA; a `val`
line every 100 with best-so-far; health flags on stderr (loss spike over 2× the trailing
median, validation rising three reports in a row, throughput under half the trailing median,
memory over budget); a non-finite loss aborts with no provenance; a run that stops short or
leaves no fresh final checkpoint is `incomplete` and ineligible. `health.json` is written on
every exit path and copied into provenance.

## 9. Cost

About 100 minutes of training at the measured 1.4× of the 3B's 70, then the 48-task selection
screen (about 15 minutes) and the 60-task evaluation (about 25 minutes at the slower `none`
cache; 70 minutes for 180 tasks). One GPU, one execution claim: it waits for P6 to finish.
