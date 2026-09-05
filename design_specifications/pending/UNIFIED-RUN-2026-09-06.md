# The unified run: the evidence composed into one training run, the coherence-length standard, and the comparison across runs

Status: pending, for the Head's review (fast) and the Deputy's implementation. Written by the Chief
under the Director's standing order of 2026-09-06 (record time about 08:45): once the WP12 thread is
closed, (1) compose the evidence so far into a unified training run, (2) obtain the best-effort model
under the current capability standard, how long the model maintains coherence on a suite of agentic
tasks, one task per evaluation point, tasks varied in scope and complexity, with substantial
investment in the dataset, and (3) compare across training runs. Standing authority for all three as
a group; implementation and review delegated across the sessions as has been the practice.

## 1. What the evidence says, and what the run does about it

| evidence (record) | decision in this run |
|---|---|
| Training memory grows at 2.4 MiB per token on this box, the same on 4-bit and bf16; the residual is bounded; checkpointing works; the row ceiling is about 6,400 tokens (TRAIN-COST-2026-09-05) | 4-bit QLoRA base, batch 1 with accumulation 4, gradient checkpointing on, gated-delta chunk 64; the row cap rises from 2,688 to 3,584 tokens to admit difficulty-2 training rows, inside the ceiling with margin |
| Step throughput 126 tokens per second at 2,688-token rows on the full adapter set, 102 at 4,096; the top 16 layers train at 206, the top 8 at 276 (TRAIN-COST, R50) | Arm A, the deliverable, trains the full adapter set (R50(a): its adapters may be read by the lens later). Arm B, the same data on the top 16 layers, is the R50(d) measurement of the throughput lever as an R35 named difference and the second point of the cross-run comparison; it runs after A if the night allows |
| The band is the R41e pairs 12/13, 16/17, 19/20, 23/24, 27/28; the lens converges on the output from layer 20 (F10); the relays sit in the band and are favoured over the rest by 3.5 (WP12) | Nothing in the training recipe changes for this yet: the full adapter set covers every pair; a layer-restricted arm would be a third arm and is not in scope tonight |
| Long-context retrieval lives in attention at layers 20 to 24 by softmax competition; the recurrent channel carries a running summary, not addresses (RETRIEVAL-CHANNELS) | The v2 protocol already externalises state into notes and windows old tool results; the finding supports it and argues against relying on the recurrent state for retrieval. Windowing (`keep_last` 2) is kept |
| Inference context to 64k is not the constraint; training row length is (CTX-EFFICIENCY, R48) | The evaluation keeps `max_steps` 24 and `max_tokens` 200 per step; the coherence standard is measured in steps, not context |
| Run D's recipe (SPEC-003): difficulty-0 and difficulty-1 training splits, family-balanced screens at difficulty 1 and 2, test at 2, test3 at 3, chat replay, recovery repeats | Kept as the base recipe, extended below; the pre-registered selection rule (family-balanced screen) is unchanged so that runs stay comparable |
| Prior runs: agent-v2c (Qwen2.5-Coder-3B, trained and evaluated on 180 test tasks); agent-v2b-qwen35-4b failed at iteration 0 (OOM, later fixed by R32); agent-v2d and agent-v2d-qwen35-4b designed and never trained | The comparison set is the base Qwen3.5-4B with no adapter, arm A, arm B, and v2c's best adapter as the cross-model reference on byte-identical task content (the R21 render pattern) |

## 2. The capability standard: coherence length

One task per evaluation point, as the evaluation already does (each task is one trajectory). Scope
and complexity vary across the twelve families (six short-horizon, six long-horizon) and the
difficulty levels 0 to 3, whose horizons are strictly increasing by construction.

Coherence length of a trajectory: the number of assistant steps completed before the first
incoherence event, where an incoherence event is the first of: a note-integrity violation
(`integrity.first_violation`, which the checker already records with its kind), a detected loop
(`detect_loop`), an invalid or non-executable action, or an unrecovered tool error; a trajectory
that reaches success with none of these has coherence length equal to its step count and is
censored at completion. Reported per family and per difficulty as the median coherence length, the
fraction of trajectories coherent to completion, and the survival curve (fraction still coherent
after s steps) with Wilson intervals on the fractions; success and tokens per success stay beside
it. The implementer confirms which of these signals carry a step index today and adds the index
where one is missing; the Head reviews the definition before it is computed on any adapter.

## 3. The dataset investment

Extending Run D's recipe, byte-identical where unchanged so B and C comparisons stay paired:
- a `train2` split of 120 tasks at difficulty 2, perturbed, role train: the first training exposure at the test horizon;
- `train` doubled to 480 at difficulty 0 and `train1` to 240 at difficulty 1, family-balanced;
- recovery repeats rebalanced so every family sees each variant at least twice in training;
- every generated row passes the integrity checker before it is written (rows that fail are regenerated with the next seed and counted in the manifest);
- the Research Division's scope extension: at least two new long-horizon variants within the generator's tool world (multi-file plans and cross-tool reconciliation are the candidates), specified as an addendum to SPEC-003 section 3 within the hour and implemented by the Deputy if the addendum lands before the data stage runs; otherwise the run proceeds on the four items above and the extension goes to the next run;
- chat replay unchanged (240 rows); test and test3 unchanged so the comparison is paired.

## 4. Run parameters (arm A; arm B differs only in the adapter set)

Config `configs/agent_v2e_qwen35_4b.yaml` from `agent_v2d_qwen35_4b.yaml`: model qwen35-4b;
rank 16, scale 32, dropout 0; iters 400; batch 1, accumulation 4; learning rate 3e-5;
max_seq_length 3,584; gated_delta_chunk 64; grad_checkpoint true; val_batches 24; eval every 100,
save every 100; seed 20260902. Selection: the SPEC-002 family-balanced screen on valid (difficulty
1) and valid2 (difficulty 2), unchanged. Evaluation: test (180, difficulty 2) and test3 (60,
difficulty 3), with the coherence summary. Rollout stage: not run tonight. Pre-registered criterion
for A against the base model: reported with intervals, no threshold, as for D4 against D3; for B
against A: the R35 named difference on the same table.

Projected time on this box: at about 1,200 tokens per row, 4 rows per step, 126 tokens per second,
a step is about 40 seconds and 400 steps about 4.5 hours for arm A; arm B about 2.8 hours. The
selection screens (four checkpoints, 48 tasks each) and the test evaluations (240 tasks) add about
two hours per arm. The base-model evaluation runs first, while the data stage builds.

## 5. Comparison across runs

One report table from `report`: base Qwen3.5-4B, arm A best adapter, arm B best adapter, v2c best
adapter (3B), on test and test3, with success, coherence length statistics, loop failures,
exhaustion, tokens per success, and the survival curves as one figure. Paired McNemar on task ids
where the model is the same; the 3B reference is unpaired across models and reported with its
intervals. Every number carries its provenance (issue 84).

## 6. Schedule, the lift, and who does what

- The Deputy, with Codex: the coherence summary in `pipeline/evaluate.py` and `report`; the config;
  the generator and data-stage changes of section 3; the base-model evaluation; then the stages
  data, render, train, select, eval for arm A, then arm B. Training runs are exempt from R47 and are
  announced in the heartbeat log at launch with the projected time; one model load at a time, so the
  base evaluation waits for the WP12 injection test to release the model and training waits for the
  base evaluation. The Deputy commits records within the hour (R49).
- The Research Division: the SPEC-003 section 3 addendum (scope extension and dataset quality) within
  the hour; the review of the generated manifest against it before training starts.
- The Head: review of this memo and of the coherence-length definition, fast; the R36 verdict on
  the comparison table when it exists.
- The Chief: gates every patch by reading the hunks; closes WP12 in parallel; writes the comparison
  reading and the Director's report.
- The Director: informed at launch and at the table; nothing waits on the Director tonight.

## 7. What this run does not do

It does not change the protocol, the selection rule or the test splits; it does not use the
throughput lever on the deliverable arm; it does not train on test3's horizon; it does not run the
rollout stage. Those are the next run's questions, and the comparison table is what decides them.

## 2 (revised, 10:05, after the Head's review; supersedes section 2 above, which is not to be computed on any adapter)

The Head found five faults, two of which would move the headline the wrong way: a trajectory that
succeeds in eight steps has a coherence length of eight and one that succeeds in twelve has
twelve, so a faster adapter reports a shorter length; and completion is correlated with capability,
so censoring at completion is informative and the survival machinery's assumption is violated by
construction. The standard is therefore restated:

1. **Headline: the fraction of trajectories coherent to completion**, per task, with Wilson
   intervals, per family and per difficulty and overall. One task per evaluation point, unchanged.
2. **Competing risks, never pooled into a first-of-any length.** The four causes, note-integrity
   violation, detected loop, invalid or non-executable action, unrecovered tool error, are reported
   separately as the cause-specific hazard per step (events of that cause at step s over the
   trajectories still at risk at s) and the cumulative incidence by cause. An adapter that halves
   loops while worsening tool errors then shows as exactly that.
3. **Tool errors get a window.** An error at step s is an event if it is not recovered, by the rule
   evaluate.py already uses for recovered_errors, by step s + k with k = 3; the last k steps of every
   trajectory are censored for that cause. The event time then depends only on the past.
4. **No median coherence length.** If a length is reported at all it is the Kaplan-Meier median
   with "not reached" when fewer than half of trajectories fail, as a descriptive only, with the
   informative-censoring caveat stated beside it; the per-step hazard against the number actually at
   risk is the only step-indexed summary the report carries.
5. **Cross-cell comparisons only on scale-free quantities.** Horizons increase by construction
   across families and difficulties, so comparisons across cells use the per-step hazard or steps as
   a fraction of the task's own required steps (the generator knows each task's horizon); the report
   says which quantities are comparable across cells and which are not.

The four signals and the precondition on step indices stand. The Deputy's reading of the code
(10:00): violations carry a step; parse errors carry one on the step; detect_loop returns a bool
and must return the index of the last step of the first window in which the repetition closed, None
for no loop, with the flag derived from it; "unrecovered" is defined through the existing
recovered_errors rule, cited by line in the docstring.

**Corrections and rulings, 10:12.** (a) The memo pointed at `recovered_errors` by name as the
recovery rule; the Deputy read it: `env.py:281` sets recovered_errors to the error count on success
and zero otherwise, so it is not a recovery rule and would make the tool-error term a function of
success, reintroducing the outcome the standard is meant to be separate from. Ruling: recovery is
per error, a later successful call within the three-step window with the same tool and the same
primary argument, the primary argument extracted per tool from the tool schema and named in the
docstring. An `unknown_tool` error has no later call with the same tool by construction; ruling: it
is recovered by a later successful call with any valid tool carrying the same primary argument
within the window, which is what the data's `unknown_tool` recovery variant teaches, and otherwise
it is an event at its own step. (b) The Research Division's addendum
(`SPEC-003-ADDENDUM-SCOPE-2026-09-06.md`) is accepted for this run: two families,
`multi_file_plan` (horizon 10 + level) and `cross_tool_reconcile` (11 + level), whose notes carry a
monotonically shrinking outstanding set rather than an accumulating prefix, with the
note-consistency test (the outstanding set equals the sources not yet read at every step of a clean
row and shrinks by exactly one per read) and the structural-variant assertion at level 0 required
before rows are written. (c) The screens are resized so the new families can be screened: `valid`
and `valid2` become 28 rows each, two per family over fourteen families, family-balanced as before;
the selection rule is unchanged in form. This is a change to SPEC-003 section 3 proper and is made
here by the Chief under the standing order; test and test3 remain byte-identical. (d) The report
gains a per-family and per-variant breakdown of loop and exhaustion counts, pre-registered as the
measurement that could refute the addendum's mechanism (that the current note form narrates rather
than tracks); without it the next run changes the data and nobody can attribute the result.

**Projection checked against the base evaluation, 11:35.** The base-model evaluation of 180 test
tasks at max_steps 24 and max_tokens 200 has run 67 minutes and reached about task 140, so a task
costs about 27 seconds on the 4-bit 4B and the 180-task split about 80 minutes. The memo's "two
hours of evaluation per arm" was low: per arm, test and test3 (240 tasks) are about 1.8 hours and
the four selection screens (4 × 56 tasks at the resized screen) about 1.7 hours, so about 3.5 hours
of evaluation beside 4.5 hours of training, eight hours for arm A. The night holds arm A and its
comparison; arm B (top 16) goes to the day, and the selection screens can be cut to two checkpoints
(steps 200 and 400) if the morning is needed for the table, a change the Chief makes at launch and
records in the heartbeat. The training half remains projected, not measured, until the first ten
steps report tokens per second.

**The training projection's basis, 11:45 (the Deputy's question).** The 4.5 hours was computed on
actual row lengths, not on the cap: the v2d Qwen rendering has 2,826 training rows averaging 1,082
tokens (median 1,090, p90 1,785, maximum 2,874; nine rows over 2,688 and none over 3,584), so a
step of four rows is about 4,300 tokens, 34 seconds at the 126 tokens per second measured on
2,688-token rows and 42 seconds at the 102 measured on 4,096-token rows, which brackets 400 steps
between 3.8 and 4.7 hours. Raising the cap to 3,584 keeps nine rows whole that were truncated before
and changes throughput by nothing measurable; the new difficulty-2 training rows will lift the mean
toward the valid split's 1,219, so the projection is 4 to 5 hours and the heartbeat announcement
carries this basis (mean row tokens, tokens per step, the throughput figure used and the row length
it was measured at) so the number can be checked against the first ten steps' tokens per second.
Noted for the next run rather than changed tonight: 400 steps at batch 4 is 0.57 of an epoch of the
v2d rows and less of the enlarged data; whether more steps help is a comparison the table can ask
for, not a change to make blind.

**11:50 (the Deputy).** The step is four sequential single-row passes (batch 1, accumulation 4),
each at its own length's throughput, and the probe's row nearest the real length, 1,024 tokens,
ran at 147 tokens per second, which would put 400 steps at 3.3 hours, below the bracketed range;
the bracket from the two measured lengths stays as the announced projection because it rests on
direct measurements, and the first ten steps settle it. Both arms train the same number of steps
on the same data and so see the same fraction of it, which is what makes the comparison between
them mean what it says; a run that sees just over half its data once is a different object from
one trained to convergence, and the table is read with that in view.

**The base-model evaluation, 12:05 (the Deputy; committed 64d916a).** Qwen3.5-4B, 4-bit, no
adapter, test split: 123 of 180 (0.683), clean rate identical; valid actions 0.997, schema
validity 1.0, executable calls 0.840; loops 20 and exhausted 32, so 52 of 57 failures are the two
mechanisms the SPEC-003 addendum was accepted to attack; 74,025 generated tokens, no think tokens;
wall time 1:19:22, 26.5 seconds a task, which settles the evaluation half of the projection at about
80 minutes a split. And the field `recovered_errors` reads exactly 0 against 273 tool errors: the
empirical form of the 10:12 ruling. That field is the error count on success and zero otherwise,
so on a run where no successful trajectory errored it reports nothing, and had the tool-error cause
been defined by that name it would have fired at the first error of every failed trajectory and
nowhere else, taking the whole cumulative incidence with it under competing risks. The per-error
rule is not a tidier definition of the same thing; it is the only one of the two that measures
anything, and the next reader who sees a field in the codebase that looks like it already does the
job should read this line first.
