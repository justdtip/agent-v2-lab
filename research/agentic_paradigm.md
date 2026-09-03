# Making a 3B model agentic: the v2 paradigm

Date: 2026-09-02. Supersedes `architecture_review.md` as the working plan.

## 1. What the evidence says about the first attempt

The step-250 QLoRA adapter reached 40/40 on the original short tasks and 0/12 on the
long-horizon tasks. Reading the traces in `outputs/complex-agent/` (not the aggregate numbers)
shows five concrete behaviours, none of which is a capacity problem:

| Behaviour observed | Example trace | Root cause |
| --- | --- | --- |
| Stops iterating early | ledger: reads 2 of 5 invoices, then sums | Never trained on a loop longer than 4 steps |
| Guesses instead of looking | aggregate_report: reads `metrics/0001.json` five times, never lists | Expert data for that family also skipped the listing |
| Invents tools | batch_update: `apply_mode_replacement` ×4 | No example of an unknown-tool error and its correction |
| Cannot recover from an error | cross_reference: `file not found` → immediately `finish "Resolution"` | No error appears anywhere in the training data |
| Loses facts in a long context | expanded ckpt-50 ledger: adds held amounts and noise numbers into `72 * 138 + …` | Needle-in-haystack over 5k tokens of deliberate noise |

Two further problems were in the pipeline itself rather than the model:

- The evaluator required `set_plan` and `update_plan` calls, so both pointer-chain tasks were
  scored as failures **with the correct answer**. Compliance and success were conflated.
- The data generator passed tool arguments to the chat template as a JSON *string*, so the
  model learned to emit `"arguments": "{\"path\":…}"` (escaped) instead of Qwen's native
  `"arguments": {"path": …}`. Every call cost extra tokens and diverged from the pretrained format.
- The native `<tool_call>` tokens are untrained in this checkpoint (see §2.4), which the lenient
  parser hid: the "�" characters in every saved response are that token failing to be produced.

The response to these failures was depth expansion: four full-precision copied blocks
(308M trainable parameters, 10 GB peak memory, 5-14 tokens/s training). Its probes produced a NaN
test loss and a perplexity of 1.9 million before a stable recipe was found, and the stable run
scored 0/12 at checkpoint 50 and 0/6 at checkpoint 100. This is the wrong lever: the model has
the capacity, it lacks the behaviours and is being asked to do attention-heavy retrieval that no
3B model does reliably. **Depth expansion is shelved**; its code stays for ablations.

## 2. The paradigm

Make the *problem* small enough for a 3B model, then make the model reliable at it.

### 2.1 State-carrying notes (the model's working memory is text it writes itself)

Every assistant turn is a short progress note followed by exactly one tool call. The note is
not free-form reasoning; it is a state record: what the plan is, which items are done, which
values have been collected, what comes next. Examples from the generator:

```
Invoices read: 3 of 5. approved: 86, 61; held (skip): 43. Reading invoice-3-670.txt.
Manifest: worker-0 safe->observe; worker-1 audit->strict. Applied 1 of 2; replacing mode=audit with mode=strict in worker-1.ini.
That path does not exist; use the exact path from the earlier tool result instead of guessing: lab/…/summary.txt.
```

Because the note at step *k* is a function of the note at *k-1* and the latest observation only,
the model never has to attend back across many noisy files. Loop position, running totals, and the
answer format are always in the most recent few hundred tokens.

### 2.2 Observation windowing (the harness enforces the memory discipline)

Only the last two tool results stay in context verbatim. Older ones are replaced by a one-line
stub (`[earlier read_file result hidden: 27 line(s). Use your notes.]`). The system prompt tells
the model this, so writing complete notes is the only way to succeed. Training rows are built with
the same windowing applied at each step, so train and inference contexts match exactly.

Side effects: contexts drop from ~5k tokens to ~1.1k on average (max 2.2k), training and rollouts
run several times faster, and the model cannot "cheat" by re-reading a stale observation.

### 2.3 Recovery is in the data, and the mistake is not

A quarter of training tasks are clean; the rest carry one of three perturbations:

- **transient**: a call returns `ERROR: temporary failure…` once; the note says "retry unchanged".
- **wrong_path**: a guessed or typo'd path returns `file not found`; the note says "use the exact
  path from the listing" or "list first"; the correct call follows.
- **unknown_tool**: a plausible but nonexistent tool (`update_file`) is called; the error names the
  available tools; the note switches to `replace_text`.

The mistaken step is executed so the real error text lands in the context, but it is **excluded
from the loss** (`Step.supervise = False`). The model sees mistakes and learns only the recoveries.

### 2.4 A call format made of ordinary tokens (and no plan tools)

The first v2 training run exposed a checkpoint defect. In `Qwen2.5-Coder-3B-Instruct-4bit` the
embedding rows for `<tool_call>` and `</tool_call>` (ids 151657/151658, tied with the output head)
are identical to the untrained padding rows (cosine similarity 1.000, norm 0.40 versus ~0.98 for
ordinary tokens). At the position where the marker should be emitted, the policy's next-token
distribution is flat: p(`<tool_call>`) = 0.0004 and junk tokens tie with it. Every adapter trained on
the native template therefore emits garbage there; ChatGPT's adapter only "worked" because its
parser ignored the marker and regex-matched the JSON. The aborted run is kept in
`outputs/agent-v2-aborted-native-format/` as evidence.

The protocol therefore renders each call as plain text the base model already produces on its own:

```
<note>
```json
{"name": "read_file", "arguments": {"path": "lab/…/invoice-0.txt"}}
```
```

The tool list is written into the system prompt by us (the chat template's `tools=` path is not
used, because it appends an instruction to use the broken tags). Parsing takes the first JSON
object and ignores anything after the closing fence; generation stops at that fence.

There are no `set_plan` or `update_plan` tools: the plan is the first note, progress lives in every
subsequent note, and the evaluator scores only what the task asked for.

### 2.7 Cross-turn KV cache reuse

Consecutive prompts in a trajectory share almost everything. The system message and task never
change, assistant notes are never rewritten, and an observation is replaced by its stub exactly
once and then stays fixed, so only the turn falling out of the window differs. Measured across
two full trajectories, the longest common token prefix covers 56-58% of all prompt tokens,
against 23-26% for caching only the fixed system-and-task prefix.

`TurnCache` exploits that: each turn it trims the KV cache back to the longest prefix shared with
the new prompt and encodes only the remainder. Prefill was measured at 56% of turn wall-clock
(2,502 ms of a 4,466 ms turn at 2,034 prompt tokens), so the saving is real rather than nominal:
**37% faster end to end** on a six-task sample.

This is a speed-only change, and a stale cache would produce silently wrong attention rather than
an error, so correctness is enforced rather than assumed. The cache offset is checked against the
computed prefix length after every trim and the cache is rebuilt on any mismatch or failed trim,
and `research/cache_equivalence.py` runs the same tasks with and without the cache and requires
bit-identical raw text, actions, and verdicts. It currently reports 6/6 identical. `--no-cache`
disables it.

### 2.5 Verified self-improvement instead of reward hacking

Once the SFT policy works, the environment verifier becomes the teacher:

1. **Rollout**: sample the policy at temperature 0.7, N times per task, on a fresh split
   (`iter1`, `iter2`, …; the split name seeds the generator, so none of these tasks are in
   valid/test).
2. **Verify**: keep trajectories whose final answer, file state, and required tools all check out.
   Steps that produced an error (other than injected transient faults) are dropped from
   supervision; recovered trajectories are kept.
3. **Retrain** on expert + kept on-policy rows. This is expert iteration (STaR/ReST): the notes in
   the kept rows are the model's own, so the policy converges to its own most reliable phrasing.
4. **Preference stage (next)**: branch a verified trajectory at step *k*, resample *M* actions, and
   continue each with the policy. Actions whose continuations fail become "rejected", successful
   ones "chosen", against an identical prefix. This yields exact step-level DPO pairs without a
   multi-turn RL loop. Mining is `agent-pipeline branch` (writes
   `data/preferences/<split>/pairs.jsonl` as `{prompt, chosen, rejected}`); training is
   `agent-pipeline prefer`, which continues the SFT adapter with mlx-tune's `DPOTrainer`, the
   frozen SFT policy as reference, and writes an mlx-lm-loadable adapter to
   `outputs/agent-v2/prefer/adapters`. Both stages are untested against a trained policy until
   run A finishes; the code review found and worked around four mlx-tune 0.6.0 traps (an
   unused `load_in_4bit`, `load_adapter` not freezing the base, `save_pretrained` copying the
   input adapter, and a synthesised `adapter_config.json` with the wrong LoRA scale).

GRPO on final reward is deliberately not the first RL step: mlx-tune's GRPO is single-turn
(prompt → completion → scalar), long rollouts make the group-normalised advantage sparse, and a
verifier that already produces exact success labels is better used for filtering and pairs.

### 2.6 Everything produces a transcript

Every stage that runs the model (checkpoint screening, evaluation, rollouts) streams a
per-step transcript to the terminal, note / call / observation / verdict, and writes the same
to `outputs/agent-v2/transcripts/<stage>/<task>.md` plus a `transcripts.jsonl`. The
transcript is the primary debugging artifact; aggregate metrics come second.

## 3. Task suite

Twelve families, deterministic per `(seed, split, index)`, so the whole dataset is reproducible
and split isolation is by construction:

- Short (2-4 steps): read, search, calculate, synthesis, update (now with verify), list.
- Long (7-14 steps): pointer_chain, ledger_reconcile, cross_reference, conditional_update,
  batch_update, aggregate_report (now lists before reading).

Difficulty: `train` = level 0, `valid` = 1, `test` = 2 (more items per task, more noise lines).
The test split therefore requires **longer horizons than anything trained on**, which is the
generalisation axis that matters most for agents. Prompts are paraphrased (2-3 phrasings per
family) so the model keys on the instruction, not the template.

## 4. Evaluation protocol

`success` = exact answer + expected file state + required tools used. `clean` = success with zero
tool errors. Reported separately: valid-action rate, schema-validity rate (calls whose arguments
pass validation against the tool schema before execution), executable-call rate, tool errors,
recovered errors, mean steps, generated tokens per successful task, p50/p95 wall-clock per task,
per-family and per-variant rates, and a histogram of first failure reasons (`parse error`,
`repetition loop`, `step budget exhausted`, `wrong answer`, `no finish call`,
`file state wrong`, …). The simulator is validator-first: an invalid call never executes; it
returns `ERROR: invalid call: …` and the policy has to recover.

- Checkpoint selection uses the **validation split behaviourally** (18 tasks per checkpoint),
  not validation loss. The first run showed loss and behaviour can disagree (step 100 had good
  loss and emitted EOS immediately).
- `--stress` injects a transient error on every task's second call at evaluation time, which
  measures recovery on tasks that were not perturbed in training.
- The test split is touched once per pipeline iteration.

## 5. Training recipe (run A of v2)

- Base: `mlx-community/Qwen2.5-Coder-3B-Instruct-4bit`, frozen 4-bit.
- QLoRA rank 16 / scale 32 on q,k,v,o,gate,up,down across all 36 layers (~30M parameters).
- 240 train tasks → 1,460 expert rows (+240 chat-replay rows for conversational retention),
  prompt-masked loss on note + call only (~60 target tokens per row).
- AdamW 3e-5, batch 2 × accumulation 2, 400 iterations (about 10 s each on the M4 Pro), max
  sequence 2,688, checkpoints every 100; extend with mlx-lm's `--resume-adapter-file` if the
  validation screen is still improving at the last checkpoint.
- Gradient checkpointing on (`train.grad_checkpoint`): turning it off at batch 2 exhausted Metal
  memory (the activations of 36 layers at 2.7k tokens exceed the 24 GB budget); with it on, peak
  memory is about 9.4 GB.


### 5.1 What the step-100 screen of run A taught (2026-09-02, 14:50)

Short families passed cleanly with well-formed notes and calls. Two long families failed in
ways that trace back to the expert notes, not the model:

- **ledger_reconcile**: the note recorded "Invoices read: 2 of 4" but not the remaining file
  names. After two reads the listing is hidden, and invoice names carry a random suffix, so the
  data teaches the model to guess (`invoice-2-888.txt`, which did not exist). Fix: every read
  note in the listing-driven families carries a `pending:` list of short file names.
- **cross_reference**: `search_files` for a Next-Key matches both the previous record (which
  mentions the key) and the target record. The expert note said "Search matched <path>" as if
  there were one match, so the rule "the file already read is not the new one" was never stated.
  Fix: the note names both matches and says which is already read.

Both fixes are implemented in `tasks.py`, guarded by two invariant tests: every read of a file
whose listing is hidden must be named in the previous note or the prompt, and every multi-match
search note must name the already-read file. Run A continues unchanged so that run B (same
recipe, fixed notes) is a data-only ablation.


### 5.2 Run C: asserted completion does not extrapolate (2026-09-03)

Run C changed three note templates so that lists were consumed as queues rather than indexed
(incremental half-buckets for aggregate_report, a named queue head with explicit phase
transitions for batch_update, a running maximum for conditional_update). On the 180-task split it
regressed from run B's 145/180 to 118/180 (McNemar p < 0.0001, 29 tasks broken, 2 fixed), and
the two targeted families did not move (0/15 and 1/15).

Every failure has the same shape: a completion marker fires at the training-length position
rather than at the true total. The test split deliberately uses longer instances than training.

| Family | Marker | Fired at | True point | Cases |
| --- | --- | --- | --- | --- |
| aggregate_report | `(full)` on the first bucket | 2 values | 3 of 6 | 15/15 |
| conditional_update | `(final)` on the running max | 2-3 reads | 5 of 5 | 15/15 |
| batch_update | `phase verify begins` | after 1 apply | after 4 | 12/15 |
| cross_reference (untouched) | never advances past the first key | - | - | 15/15 |

Training totals are 4 metrics (first half 2), 3 services and 2 workers; test totals are 6 (3), 5
and 4. The model learned each marker as a habit of sequence position, not as a function of the
remaining list. Run B's format survived the same extrapolation because progress was stated as
`k of N` with the remaining items enumerated and the only transition condition was the literal
`pending: none`. The regression in the untouched cross_reference family is consistent with
interference: the recovery oversampling repeats rows from exactly the three rewritten families
up to six times, so the new style dominated the corrective decisions the model saw most often.
That is a hypothesis; probes P5 and P2 in `representation_probes.md` are the way to test it.

Design rule learned: **a note may enumerate what remains and count against a stated total; it
must never assert completion.** Completion has to be a consequence the reader can check
(`pending: none`), never a token the writer emits. Run C's adapter is kept as a negative
control for the probes.

Run D should build on run B's templates, keep the incremental buckets for aggregate_report but
state the split point as a number derived from the total in every note (`split after 3 of 6`)
with no `(full)` marker, drop phase-completion assertions, and train on a mixture of lengths
(difficulty 0 and 1) so that no completion condition can be learned as a position.

## 6. Roadmap

1. **Now**: run A, screen checkpoints, evaluate base vs. best on test (clean and stress).
2. **Iteration 1**: `agent-pipeline rollout --split iter1`, then `agent-pipeline data --extra
   data/rollouts/iter1` and retrain. Expect gains on the families with the lowest pass@k.
3. **Step-level DPO** from branched rollouts (§2.5 item 4): `agent-pipeline branch --split pref1`
   then `agent-pipeline prefer`, then `agent-pipeline eval --adapter outputs/agent-v2/prefer/adapters`.
   First real run pending.
4. **Novel-family hold-out**: add two families that appear only in test to measure transfer of
   the note discipline rather than the families.
5. **Real tools**: the same protocol and harness with a sandboxed file/shell backend, plus the
   safety layer that the original model card lists (argument validation, timeouts, audit log,
   approval for consequential actions). The synthetic environment remains the regression suite.
6. **Ablations to justify each component**: no windowing, no notes, no recovery variants,
   single-phrasing prompts. Depth expansion can be re-tested as one ablation on top of v2.

## 7. Runbook

```bash
uv run agent-pipeline data            # deterministic data + manifest
uv run agent-pipeline train           # QLoRA; log in outputs/agent-v2/train.log
uv run agent-pipeline train --resume-from outputs/agent-v2/adapters/0000400_adapters.safetensors --iters 200
uv run agent-pipeline select          # behavioural screen of every checkpoint, live transcripts
uv run agent-pipeline eval --base     # baseline on test
uv run agent-pipeline eval            # best adapter on test
uv run agent-pipeline eval --stress   # recovery under injected faults
uv run agent-pipeline rollout --split iter1
uv run agent-pipeline data --extra data/rollouts/iter1 && uv run agent-pipeline train
uv run agent-pipeline branch --split pref1   # step-level preference pairs
uv run agent-pipeline prefer                 # DPO on those pairs, from best-adapter
uv run agent-pipeline eval --adapter outputs/agent-v2/prefer/adapters
uv run agent-pipeline report          # every evaluation summary in one table
uv run agent-pipeline all             # the whole thing
```

Add `--quiet` to any stage to print only pass/fail lines; transcripts are still written to disk.
Single-policy tools: `uv run agent-v2-eval --adapter <dir> --limit 5` and `uv run agent-v2-rollout`.

## 8. Literature check (sources supplied 2026-09-02)

| Source | What it is | What it changes here |
| --- | --- | --- |
| Sharma & Mehta, *Small Language Models for Agentic Systems* (arXiv 2510.03847) | Survey of 1-12B models as agents; argues tool use is schema- and API-constrained accuracy, not open-ended generation | Strongest match to this project. Adopt its **metrics** (schema validity, executable-call rate, cost per successful task, p50/p95 latency, escalation rate) alongside task success; adopt **validator-first execution** (validate arguments against the tool schema *before* executing; a validation failure becomes an observation the policy must recover from) and **constrained decoding** as an inference-time safety net; its "practical recipe" (10k-50k successful traces, LoRA per task cluster, INT4 serving, periodic refresh from logged validator failures) is the scaled version of `rollout` → `data --extra` → `train`. Its SLM-default / LLM-fallback router with uncertainty thresholds is the deployment shape once real tools exist. |
| Han et al., *Blueprints and Prompt Template Search* (arXiv 2506.08669, Microsoft) | A large model writes per-task-family "blueprints" (step-by-step reasoning guides, 12 styles); a successive-halving search over 32 prompt templates picks the best; no training; +5-20 points for 3-7B models on BBH/MBPP/GSM8K | Our first note is a blueprint the model writes for itself, learned from the expert trajectories. Two transferable ideas: (1) hold out a family and test whether a blueprint *in the prompt* recovers performance without training (a cheap generalisation probe); (2) the prompt-template search is exactly the right way to tune the system prompt and `keep_last` on the validation split instead of by hand. |
| Srivastava, Cao & Wang, *Towards Reasoning Ability of Small Language Models* (arXiv 2502.11569) | ThinkSLM benchmark: 72 SLMs on 17 reasoning tasks | Two findings matter: training method and data quality dominate parameter count for reasoning, and 4-bit **quantisation preserves reasoning while pruning harms it**. Both support the frozen-4-bit-plus-adapter design and the rejection of depth expansion. |
| Kim et al., *Meerkat* (npj Digital Medicine, 2025) | 7-8B medical models fine-tuned on 441k GPT-4-extracted chain-of-thought examples grounded in textbooks; a 7B model passes USMLE | Evidence for the data recipe rather than the architecture: grounded, verifiable step-by-step traces at scale beat continued pre-training. Our verifier plays the role their textbooks play. Their multi-metric reasoning evaluation (completeness, factuality, consistency) maps onto our per-step transcript review. |
| Dikici, *Small Language Models: A Systematic Review* (Expert Systems, 2026) | PRISMA review of 68 studies on SLM trade-offs, privacy, deployment | Confirms LoRA/PEFT retains 95-99% of full fine-tuning accuracy with ~90% fewer trainable parameters; nothing to change, but it is the citation for keeping the base frozen. |

Concrete additions to the roadmap from this pass:

7. **Validator-first execution** (done): the simulator validates argument names and types against the tool schema before executing; invalid calls return a structured error and count toward schema validity.
8. **The survey's minimum metrics** (done): schema validity, executable-call rate, p50/p95 wall-clock per task, tokens per successful task, plus `repetition loop` and `step budget exhausted` as explicit failure reasons.
9. **Prompt/template search on the validation split** (blueprints paper): treat the system prompt wording and `keep_last` ∈ {1, 2, 3} as searchable with successive halving, once a trained policy exists.
10. **Blueprint-in-prompt generalisation probe**: a family that never appears in training, evaluated with and without a written blueprint in the user prompt.
