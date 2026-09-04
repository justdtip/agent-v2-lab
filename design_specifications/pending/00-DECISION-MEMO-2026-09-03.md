# Decision memo: where the agent-v2 series stands and what happens next

Author: Chief AI Research Scientist (Claude). Date: 2026-09-03. Read-only assessment; no source,
data, adapter, or probe output was modified. Companion specifications: SPEC-001 to SPEC-004 in
this directory. Execution of any GPU stage is gated by the standing prohibition on deploying
agents; every specification separates code work (allowed now) from model runs (gated).

## 1. Orientation: how the agent-v2 adapters were produced

The pipeline (`src/local_llm_lab/pipeline/`) is a closed loop around a deterministic six-tool file
workspace. Twelve task families are generated per `(seed, split, index)`; expert trajectories are
replayed through the simulator; every assistant turn is a short state-carrying note plus one
fenced-JSON tool call; the harness hides all but the last two tool observations, so the note is
the only cross-turn memory. Recovery variants execute a wrong step unsupervised and supervise the
correction. Training is QLoRA (rank 16, scale 32, seven projections, all 36 layers) on the frozen
4-bit `Qwen2.5-Coder-3B-Instruct` for 400 iterations; checkpoints are chosen by an 18-task
behavioural screen on the validation split, not by loss.

Three runs share this recipe and differ only in data:

| Run | Data change | 180-task test | Role now |
| --- | --- | --- | --- |
| A | first v2 notes | 47/60 on the 60-task set | superseded |
| B | `pending:` lists, multi-match search notes | 145/180 (80.6%) | best policy |
| C | queue-style notes for three families | 118/180 (65.6%) | negative control |

Only C regenerates byte-for-byte from the current source tree; A and B data are hash-verified but
their generator revisions are unrecoverable (no Git). SPEC-001 §9 adopts the provenance report's
preservation recommendation so this never recurs.

## 2. What the evidence says, probe by probe

### 2.1 J-space probe (complete, valid)

World A: without the note, the model does not hold the hidden filename (correct digit at
p = 0.10, near uniform); with the note it copies it at p = 1.000. The J-lens positive control at
layer 24 detects an in-context filename at p = 2e-6, so the null on the hidden one is
substantive. The notes create the state; windowing makes writing them load-bearing.

### 2.2 P2 hardened run `state-corrected-hardened-20260903T172549` (complete, base, notes intact)

The manifest and the task brief both describe this run as partial. It is not: the final
`state-base-mix.{npz,json,md}` landed at 18:30 (360 tasks, 2,367 rows, 63 minutes of capture).
The manifest is stale on this point.

Reading it with the mandatory position controls:

| Target | Raw margin over position baseline (best layer) | Within-position, leakage-free | Reading |
| --- | --- | --- | --- |
| pending_count | +0.087 R² (L12) | R² 0.56-0.62 at L6-18, 0.39-0.41 at L24-35 | state decodable beyond position, strongest early-mid |
| phase | +0.105 acc (L30) | acc 0.72-0.83 rising with depth | decodable |
| prev_error | +0.275 AUC | AUC 1.00 everywhere | trivial: the `ERROR` text is a visible observation |
| next_tool | +0.194 acc (L30) | acc 0.81-0.92 rising with depth | the action about to be emitted; a positive control, not state |
| running_max | +0.074 R² (L35) | R² about 0 except L35 (0.40); 8 cells, 57 rows | underpowered and ill-posed (numeric magnitude regression) |
| first_bucket_count | +0.179 R² (L6) | R² 0.91-0.97 | strongly decodable |

Conclusion available from this one cell of the design: with the state written in the note, the
untrained base model already carries a linear encoding of pending count, phase, and bucket
count at the pre-note position. The representational substrate for reading state from notes
pre-exists; the adapter does not have to create it. Combined with §2.1 this is the "both high"
row of the scorecard for the read side: regimen is sufficient and cheap.

What this run cannot say, and why the remaining seven cells should not be run as designed:

1. The mixed dataset's `train-` rows (737 of 2,367; corrected 2026-09-03 22:10 from an earlier
   "1,050": the `p2mix` difficulty-0 rows are different tasks) are byte-identical to the
   adapters' SFT training rows. Any base-versus-adapter margin on them is confounded with
   memorisation, and excluding them shifts the difficulty mix, so adapter comparisons must be
   made within difficulty (ruling R7).
2. `--strip` rewrites assistant notes only; the last two verbatim observations and the user
   prompt stay, so a stripped run under-tests rather than tests.
3. No cross-policy comparison, paired test, confidence interval, or repeated split exists. Every
   number is a point estimate from one 70/30 task split at one seed.
4. Two targets are mis-specified: `running_max` should be the comparison the policy actually
   makes (is the value just read the new maximum), and `prev_error` should be asked about an
   error that has scrolled out of the window.


**Precision caveat (added 2026-09-04, R18):** every P2 and J-space number above was captured
through a float32 block path on a bfloat16 model; on the 3B the divergence from the native
forward is 0.3% max relative, on Qwen3.5-4B 5.5%. The offline bfloat16-rounding refit (SPEC-004
§1 C7) reported on 2026-09-05: the supported set is unchanged across all 96 cells and the
largest margin change is 0.005, so the precision gap is immaterial to these conclusions.

### 2.3 P5 adapter delta (items 1-4 complete; block ablation absent)

All three adapters use 12-13 of 16 singular values to hold 90% of the energy, spread evenly over
layers with the largest relative updates in the last three layers' q/k projections. B and C share
an update subspace (mean principal-angle cosine 0.40 against 0.10 random); A is further from
both (0.24). Flat LoRA spectra are the norm at this learning rate, so this is not evidence of
saturation. It is weakly informative either way and should not move the decision.

### 2.4 P1 assistant axis (void twice)

The first build failed the PC1 check. The corrected build with multi-sentence in-character
prompts and one-shot exemplars produced a full 288-record rollout corpus and then stopped at the
fail-closed role gate. Inspecting that corpus: the best role (pirate) expresses persona in 3 of 8
responses; ghost, oracle, prophet, hermit, monk, chef, detective, and eleven others express it in
0 of 8. The coder-instruct 3B base has no usable persona space. P1 is closed for this base and is
reopened only as a 20-minute check on the Qwen3.5 base (SPEC-004 §4).

### 2.5 v2c failure analysis from saved transcripts (no model run)

All 62 v2c failures are note-writing failures with perfect action validity:

| Shape | Count | Families |
| --- | --- | --- |
| Note copied forward verbatim, Next-Key never captured, 23 identical reads | 15 | cross_reference (15/15) |
| One or two values dropped from the note, then runaway `calculate` | 15 | aggregate_report 14, ledger 1 |
| Queue dropped, then completion asserted, `finish` with files untouched | 14 | batch_update |
| `(final)` emitted after 2-3 of 5 reads, wrong service throttled | 12 | conditional_update |
| One approved amount silently dropped from the running list | 6 | ledger_reconcile |

cross_reference and ledger_reconcile notes are byte-identical between B and C, yet they fell
14 to 0 and 13 to 9. That is negative transfer from training three families on compressive
notes. Two pipeline defects amplified it: the 18-task screen gives 12 of 18 slots to families
that never fail (a 16/18 screen concealed two families at 0/15), and validation loss bottoms at
step 300 in both B and C while the screen picked 400.

## 3. Verdict on the regimen-versus-parameters question

For the failures that exist, the answer is regimen-bound, with high confidence:

- the state is readable when written (P2), not held when unwritten (J-space);
- the failures are in writing the note, and a data change caused them (§2.5);
- run B's templates survive length extrapolation on the same base; run C's do not.

The parameters question is nonetheless about to be answered by the directive to move to
`Qwen/Qwen3.5-4B` (and possibly 9B). The right way to answer it is as a controlled arm: same data
(run B's), new base, everything else fixed. SPEC-003 lays out that matrix.

## 4. Decisions

1. Do not complete the P2 grid as designed. Redesign it (SPEC-004 §2) and run it only after
   SPEC-001 lands, on SFT-disjoint splits, with both notes and observations strippable, and with
   bootstrap intervals over task resamples.
2. Make the code base model-agnostic before any further run (SPEC-001). Qwen3.5 is a hybrid
   Gated DeltaNet plus full-attention stack (32 layers; one full-attention layer in four; tied
   embeddings on 4B, untied on 9B; a 248k vocabulary; thinking on by default; a recurrent cache
   that cannot be trimmed). Load-bearing modules that hard-code the dense Qwen2 layout are
   `probes/capture.py`, `pipeline/jlens.py`, `probes/adapter_delta.py`, `pipeline/runner.py`
   (`TurnCache`), `pipeline/protocol.py` (prompt rendering and thinking), and `pipeline/cli.py`
   (LoRA target list, training config).
3. Treat thinking as a first-class mode with three settings (off, inference-only, trained). The
   note stays mandatory in every mode: thinking is intra-turn compute, the note is inter-turn
   memory, and the chat template strips prior reasoning, so thinking cannot replace the note.
4. Fix evaluation and selection before training again (SPEC-002): a family-balanced screen at two
   difficulties, note-integrity diagnostics computed from generator ground truth, a hardened
   verdict, and confidence intervals in every table.
5. Run D (SPEC-003) reverts to B's templates, bans completion assertions by generator invariant,
   mixes lengths in training, and trains the {3B-Coder, Qwen3.5-4B} × {B data, D data} matrix.

## 5. Prioritised patch list

| Priority | Item | Spec | GPU | Rationale |
| --- | --- | --- | --- | --- |
| P0 | Architecture view, model registry, thinking modes, cache strategy, LoRA target discovery, provenance | SPEC-001 | preflight only | every later run depends on it |
| P0 | Note-integrity diagnostics, retroactive over saved B and C evals | SPEC-002 §2 | none | turns §2.5 into an automated per-run metric now |
| P1 | Selection screen and verdict hardening, CIs, known bugs | SPEC-002 §1, §3-5 | none | selection is currently uninformed |
| P1 | P2 offline re-analysis of the saved npz (CIs, seeds, surface baseline, ordinal targets) | SPEC-004 §1 | none | extracts the remaining value from 63 minutes of capture |
| P2 | Run D data recipe and generator invariants | SPEC-003 §1-3 | none for data | the change with the largest expected effect |
| P2 | Training matrix (3 to 4 runs) | SPEC-003 §4 | about 2 h per run | answers regimen and parameters together |
| P3 | P5 block ablation, P6 patching, P2 redesign runs, P1 on Qwen3.5 | SPEC-004 | 0.3 to 6 h each | after run D exists |
| P4 | P3, P4, rollout and DPO stages | SPEC-004 §6 | large | need a policy worth sampling from |

## 6. Process note

Four read-only exploration subagents were used during this assessment to read source and saved
outputs in parallel; none wrote, ran a model, or touched the GPU. If the prohibition on agent
deployment was meant to cover those as well, say so and the review workflow will proceed without
them.
