# Representation probes: is the ceiling parameters or regimen?

Date: 2026-09-03. Design by Claude; implementation delegated. Companion to `jspace_probe.md`.

## 1. The question, and the hypothesis being tested

Three runs of the same 3B base with different data have moved held-out success from 78% to
80.6%, with the remaining failures concentrated in two families that fail in a specific way
(indexing into a list the note already holds). Before spending another cycle on data, we want
to know which of two worlds we are in:

- **Regimen-bound.** The representations needed to do these tasks exist or can be created by
  adapter training; failures come from what the data asks the model to do and how. More precise,
  more varied data will keep paying.
- **Parameter-bound.** The model cannot form or use the needed representation at 3B regardless of
  data; the residual failures are a capacity floor and the next lever is architectural.

The hypothesis Daniel wants to pursue, synthesised from the sources below: the global workspace
(J-space) is where the assistant manipulates the concepts a gating mechanism judges relevant to
the task; post-training is what gives that workspace the *assistant's* point of view; and doing a
task well requires (a) being on the assistant axis, (b) holding a model of the relationship between
self and environment, and (c) some sense of being on track without ground truth, which is where
introspective awareness would enter, because a model that can consult its internal environment
model is more resilient to uncertainty.

Each probe below tests one link in that chain and states in advance what each outcome would
license. The scorecard in §9 combines them into a regimen-versus-parameters verdict.

## 2. Sources and what each contributes

| Source | Contribution to this design |
| --- | --- |
| Lu, Gallagher, Michala, Fish, Lindsey, *The Assistant Axis* (2601.10387) | The axis recipe: default-assistant minus mean role-playing activations at every layer, measured as cosine of mean response-token activations at a middle layer; bounded task requests sit high, meta-reflection and vulnerability drive drift; activation capping stabilises. Built on open 27-70B models, so replicable at 3B. |
| Gurnee et al., *Verbalizable Representations Form a Global Workspace* (2607.15495) | J-lens readout (already implemented in `pipeline/jlens.py`); post-training gives J-space the assistant's viewpoint; protocols for directed modulation (hold X while doing Y), intermediate-swap, and broadcast tests. |
| Lindsey, *Emergent Introspective Awareness* (2511.21399) | Concept-injection protocol: vector = activation for "Tell me about {word}" minus mean over other words, injected at about two-thirds depth from the pre-response token at strengths 2-4 relative to residual norm; success = affirmative detection naming the concept before saying it; false-positive control at strength 0. |
| Fonseca Rivera, *Training Introspective Behavior* (2601.01828) | Introspective detection is trainable with LoRA on a 7B model (0.4% to 85% on held-out concepts, zero false positives). Tells us a null result at 3B is a *training* gap, not a hard floor, and gives the recipe if we want it. |
| Jiralerspong & Bricken, *Cross-Architecture Model Diffing with Crosscoders* (2602.11729) | Unsupervised model diffing. Full crosscoders need substantial training; our adapters share a base, so the LoRA delta is an exact, tiny model diff we can analyse directly. |
| Anthropic, *On the Biology of a Large Language Model* (attribution graphs) | Causal validation through intervention. Cross-layer transcoders (30M features) are out of reach here; the intervention half of the method, activation patching between a failing and a correct run, is not. |

## 3. Assets and constraints

- Base: `mlx-community/Qwen2.5-Coder-3B-Instruct-4bit`, 36 layers, hidden 2048, tied unembedding.
- Adapters: `outputs/agent-v2/best-adapter` (run A), `outputs/agent-v2b/best-adapter` (run B),
  `outputs/agent-v2c/best-adapter` (run C). LoRA rank 16, scale 32, on q/k/v/o/gate/up/down of
  all 36 layers; stored as `lora_a` (in × 16) and `lora_b` (16 × out) per module.
- Ground-truth task state at every step, from the generator: pending files, phase, bucket
  contents, running maximum, whether the last observation was an error, the next expert action.
- Existing infrastructure: `residual_at` (residual stream at any layer with the correct causal
  mask), `jacobian_vector_product`, `jlens_map`, `readout`, `TurnCache`, the runner and evaluator.
- Compute: one JVP through half the network is 144 ms; a full forward on a 2k-token prompt is
  about 2.5 s; a rollout at temperature 0.7 is 5-60 s per task. Everything below fits in hours,
  not days, on the M4 Pro. Nothing requires training a transcoder or crosscoder.
- Numerics: float32 for all captured activations and all vector arithmetic (float16 overflowed
  in the J-lens work). Every activation capture must use the model's own causal mask; the
  equivalence check in `jlens.py` is the template.

## 4. Probe P1: the assistant axis, before and after agentic post-training

**Question.** Where does each adapter put the model on the assistant axis, on chat and on agent
tasks, and does position or drift along it predict failure?

**Method.** Build the axis on this model following the paper, scaled down:

1. Default-assistant rollouts: the 240 `data/chat_replay` prompts (six categories) with the
   plain assistant system prompt, greedy, 192 tokens.
2. Role rollouts: 24 roles spanning the paper's high end (consultant, reviewer, analyst,
   librarian, tutor, engineer) and low end (ghost, hermit, leviathan, oracle, trickster, pirate,
   prophet, wanderer) plus neutral ones (child, poet, drill sergeant, historian, chef, detective,
   monk, gambler, nurse, astronaut), each with 5 of the same prompts. A role vector is the mean
   post-block residual over response tokens, per layer.
3. Axis per layer = mean(default-assistant vectors) − mean(role vectors). Report cosine with
   PC1 of the standardised role vectors as the paper's sanity check (they saw > 0.6).
4. Position of a response = cosine between its mean response-token activation and the axis at
   the middle layer (18), with all layers reported.

Then measure, for base and each adapter:

- Position on chat prompts (does agentic post-training move the default persona?).
- Position per turn on the 60-task test trajectories, using the note+call as the response.
  Report the trajectory mean, the slope across turns (drift), and the per-turn minimum.
- Split by outcome: successful versus failed trajectories, and by failure reason. The specific
  prediction from the paper is that repetition loops and budget exhaustion should show lower or
  falling projections *before* the loop begins.

**Causal test.** Steer along the axis at layer 18 at every position with coefficients ±{1, 2, 4}
times the mean residual norm, and re-run the two broken families (15 tasks each). Also apply the
paper's capping rule at the 25th-percentile threshold.

**Controls.** A random unit vector of the same norm, steered identically. Axis built from a
disjoint half of the roles, to check the axis is stable (cosine between halves).

**First build on the base model (2026-09-03), 60 default prompts against 24 roles × 5 prompts.**
The axis is reproducible: split-half cosine 0.98 at every layer, and its norm is about 12.5% of
the mean residual norm at layers 18-24, so it is a real direction rather than noise. But it fails
the paper's sanity check: |cos(axis, PC1)| is 0.25-0.31 at middle layers against their > 0.6,
and the role ordering is not theirs (consultant, analyst and reviewer sit at the *low* end,
engineer, detective and child at the high end, with a total spread of only 0.07). The suspicion is
that a 3B coder model largely ignores a one-line character prompt, so the "role" vectors differ
from the assistant vectors mainly by the presence of a system prompt, not by persona. The paper
filtered rollouts with a judge for role expression; ours were unfiltered and their texts were not
saved. Until role expression is verified and filtered, P1 positions measure prompt-format
difference and must not be read as persona.

**Direct check (same day).** Greedy generations for the same question under the pirate, ghost and
consultant system prompts and under the default assistant prompt are near-identical textbook
paragraphs; none contains a trace of the persona. The base model ignores a one-line character
prompt entirely. The first axis build is therefore void and the run-B build was stopped. Whether
stronger elicitation (multi-sentence in-character prompt, explicit "never break character",
a one-shot exemplar) produces any persona expression decides what P1 can mean on this model: if
it does, the builder must persist rollout texts and filter by role expression before building the
axis; if it does not, the finding is that this coder-instruct model has no usable persona space,
which bears directly on the hypothesis that task competence rides on position along that axis.

**Decision rule.** If adapters sit higher than base on agent tasks and steering toward the
assistant end raises success in the broken families, persona position is part of the mechanism
and is a regimen lever (it can be trained or capped). If position does not separate success from
failure and steering does nothing beyond the random control, the axis is not where the residual
failures live, which is itself useful: it removes one hypothesis.

## 5. Probe P2: the internal environment model (linear state probes)

**Question.** Does the residual stream encode the task state the note carries, and did the
adapter create that encoding or merely learn to read it out?

**Method.** For every supervised row in the training and test splits, capture the residual
stream at the last prompt token (the position from which the note will be generated) at layers
{6, 12, 18, 24, 30, 35}, for base and each adapter. Train ridge or logistic probes, split by
task id so no task appears in both train and test, to decode:

| Target | Type | Ground truth from |
| --- | --- | --- |
| Number of files still pending | ordinal 0-6 | generator |
| Phase (inspect / apply / verify / other) | categorical | generator |
| Previous observation was an error | binary | simulator replay |
| Next expert action's tool | categorical | expert trajectory |
| Running maximum so far (conditional_update) | regression | generator |
| Number of values in the first bucket (aggregate_report) | ordinal | generator |

Report held-out accuracy or R² per layer per model, with a shuffled-label control.

**The decisive contrast.** Run every probe twice: once on the normal context, and once on a
context whose notes have had their state fields stripped (pending lists, bucket contents, running
maximum removed), exactly as the J-space A/B did. If probe accuracy collapses when the note is
stripped, the state lives in the note and the model reads it; if accuracy survives, the model
holds it internally and the note is a readout.

**Confound found on first run (2026-09-03) and the control it forces.** On the train split with
notes intact, a lookup that knows only (family, step index) predicted every target as well as or
better than the activation probe (pending count R² 0.91 vs 0.87; first-bucket count 1.00 vs 0.97;
phase 0.92 vs 0.97; previous-error 0.89 vs 1.00). The generator makes state an almost exact
function of position within a family at a fixed difficulty, so a probe that decodes "where am I in
the trajectory" from the residual stream, which is easy, scores as if it decoded state. The
stripped-context contrast inherits the confound: position stays decodable after stripping, so
accuracy would not collapse and the result would read falsely as "held internally".

Two corrections are therefore mandatory before any P2 number is reported: (1) every table carries
a **position-only baseline** column fitted on the same task split, and the reportable quantity is
the probe's margin over it; (2) the probe dataset mixes difficulties (levels 0, 1 and 2 across
splits) and includes the recovery variants, so that the same (family, step) occurs with different
true states and position no longer determines the label. The decision rules in §5 apply to the
margin over the position baseline, not to raw accuracy.

**Decision rule.** Adapter high, base low → the adapter *created* the representation: regimen
works, and more of it will keep working. Both high → the representation pre-exists and the
adapter only shaped the output; regimen is sufficient and cheap. Both low with the fact visible in
context → the model cannot form the representation from what it reads, which is the first result
that would point at capacity. Stripped-context collapse in every model → World A confirmed at the
representational level, consistent with the J-space probe.

## 6. Probe P3: the sense of being on track

**Question.** Before any verdict exists, do the activations carry a signal that predicts eventual
success or failure, and can the model verbalise it?

**Method.**

1. Generate on-policy rollouts with `agent-pipeline rollout` at temperature 0.7, 4 samples per
   task, on a fresh split (`probe1`, 120 tasks), for each adapter. This yields hundreds of
   trajectories with a verifier label and a full record of every turn.
2. At turns 1, 2, 3, and at the midpoint of each trajectory, capture the last-prompt-token
   residual at the same six layers. Train a logistic probe (split by task) to predict *eventual*
   success. Report AUC per turn per layer. Chance is the base rate.
3. Restrict to trajectories that eventually loop or exhaust the budget, and ask whether the probe
   fires before the first repeated action: report the earliest turn at which AUC exceeds 0.7.
4. Verbalisation: at the same turns, interrupt with a user message, "Before continuing: are you on
   track to complete this task? Answer yes or no, then one sentence of reasoning," and record the
   answer. Compare stated confidence to the probe and to ground truth (accuracy, calibration, and
   agreement with the probe). This is the counterfactual-reflection interruption from the J-space
   paper, used here as a measurement rather than a training target.

**Controls.** Probe trained on shuffled outcome labels. Probe trained only on features of the
*text* so far (turn index, number of errors observed) to check the activations add anything over
what the transcript already says.

**Decision rule.** Signal present and verbalisable → the model has a usable sense of being on
track; nothing to train, and it can be exposed as a stop-and-replan trigger. Signal present but
not verbalised → the gap is between workspace and report, exactly the case counterfactual
reflection training targets; that becomes the next training experiment. No signal at any layer
beyond the text baseline → the model has no internal foreknowledge, and resilience has to come
from environment feedback, which is what the recovery variants already do.

## 7. Probe P4: introspective awareness under concept injection

**Question.** Can this model, base or adapted, notice a manipulation of its own activations, and
is a task-state manipulation *used* even when it is not noticed?

**Method, part A (the paper's protocol).** Build concept vectors for 40 words as the activation
at the pre-response token for "Tell me about {word}" minus the mean over the other words, at
layer 24 (two-thirds depth). Prompt: "I'm going to inject a thought into your activations. On
each trial, tell me whether you detect an injected thought. If you do, describe it." Inject from
the pre-response token through the response at strengths {0, 2, 4, 8} scaled to the mean residual
norm. Score with the paper's four criteria; use an LLM judge only for the "named before said"
criterion, and report raw transcripts. Report detection rate, identification rate, false-positive
rate at strength 0, and coherence. Expectation for 3B: near zero, and Rivera's result says that
would be a training gap rather than a floor.

**Method, part B (task-grounded, the one that matters here).** Take the state directions learned
in P2 (for example "previous observation was an error", "pending count high") and inject them
mid-trajectory at layer 24 into a *clean* context. Measure whether the next action changes in the
direction the state implies (re-observe instead of proceed; keep reading instead of calculating).
This is the intermediate-swap test from the J-space paper applied to task state: it asks whether
the representation is causally load-bearing, not merely decodable.

**Controls.** Random directions of matched norm. Injection at the wrong layer.

**Decision rule.** Part B positive → the state representation is used by the policy, so improving
it (via data) improves behaviour; strong support for regimen. Part B null while P2 decodes the
state → the representation exists but is inert, a much more worrying result, pointing at a
routing limitation. Part A is reported for completeness and as a baseline for any future
introspection training.

## 8. Probe P5: what the adapters wrote (exact model diff)

**Question.** Where in the network did each adapter make its changes, how much of the rank-16
budget did it use, and did runs A, B and C write to the same places?

**Method.** For each adapter and each of the 252 LoRA modules, form ΔW = scale · (lora_a lora_b)ᵀ
(verify orientation against `mlx_lm.tuner.lora.LoRALinear`). Report:

1. Relative update size ‖ΔW‖_F / ‖W‖_F per layer and module type (dequantise W once).
2. Effective rank: singular value spectrum of ΔW, and the number of singular values holding 90%
   of the energy. Rank utilisation well below 16 means the adapter is not capacity-limited.
3. Cross-run agreement: principal-angle cosines between the column spaces of ΔW_A, ΔW_B, ΔW_C per
   module. High agreement with different data says the runs converge on a shared subspace.
4. What the update writes: push the top singular output directions of the down_proj and o_proj
   updates through the J-lens readout at that layer, and report the top tokens. This is the
   crosscoder question ("what is model-specific") answered with the exact delta instead of a
   trained crosscoder.
5. Layer-block ablation: zero the adapter in blocks of six layers, evaluate on the 18-task
   validation screen, and report which blocks carry the behaviour.

**Decision rule.** Low rank utilisation and behaviour concentrated in a few blocks → the adapter
has headroom and the residual failures are not adapter capacity. Full-rank updates spread across
every layer → the adapter is saturated and either a larger adapter or a different target is the
next step. This probe is the cheapest of the six and needs no generation at all.

## 9. Probe P6: causal tracing on a failing decision

**Question.** When the model drops a value while splitting a list, is the error in *reading* the
note or in *computing* on it?

**Method.** Activation patching between a failing and a correct run of the same decision. Use the
run B adapter on a test `aggregate_report` task at the first calculate step, where it produces a
wrong partition, and the run C adapter (or the same adapter with the queue-style note substituted)
where it produces the right one. Patch the residual stream from the correct run into the failing
run at each (layer, position) block: positions are grouped into system prompt, task prompt, the
note tokens listing the values, the last two observations, and the final token. Report which
patches flip the output to the correct partition, as a heat map over layer × group.

**Controls.** Patching from a run of an unrelated task. Patching random positions.

**Decision rule.** Flips from patching the note-value positions at early or middle layers → the
error is in reading and representing the list, which data changes reach (and the queue notes
already target). Flips only from late layers at the final position → the error is in the
computation itself, which is where a capacity argument would first become credible. This is the
intervention half of the attribution-graph method and the most direct test of the parameters
question in the whole suite.

## 10. Scorecard: reading the results together

| Pattern of results | Verdict |
| --- | --- |
| P2 decodes state in adapters but not base; P4B injections move behaviour; P5 shows low rank use; P6 flips from note positions | **Regimen-bound.** Representations are created by training, used by the policy, the adapter has headroom, and errors are in reading rather than computing. Keep investing in data and targets. |
| P2 decodes state in both base and adapters; P3 signal present | Regimen-bound and cheap: the base already has the representations, training only shapes output. |
| P2 fails in every model even with the fact visible; P6 flips only at late layers on the final token; P5 shows saturated updates | **Parameter-bound** for the specific computation. The next lever is architectural or a task decomposition that removes the computation. |
| P1 separates outcomes and steering rescues failures | The persona axis is a live lever; add it to the training or capping regime. |
| P3 signal present but not verbalised | Counterfactual reflection training is the next experiment, with P3 rerun as its success metric. |

## 11. Order of execution and cost

1. **P5** first, no generation, under an hour; sets the capacity prior.
2. **P2**, activation capture over existing rows, a few hours including the stripped-context
   contrast.
3. **P1**, needs the role rollouts (about 20 minutes per model) plus trajectory projections
   from existing evaluation transcripts; steering runs on 30 tasks per condition.
4. **P3**, needs fresh rollouts (the longest generation cost, roughly 2-3 hours per adapter at 4
   samples × 120 tasks), then probing and the interruption protocol.
5. **P6**, one decision, a few hundred forward passes.
6. **P4**, last, because part A is expected to be near zero at this scale and part B depends on
   P2's directions.

Every probe writes JSON under `outputs/probes/<probe>/` and a Markdown summary, and every probe
reports its control alongside its result; a result without its control is not reported.

## 12. What this suite cannot settle

None of this trains a transcoder or crosscoder, so feature-level circuit diagrams of the kind in
the attribution-graph paper are out of scope; P6 gives the intervention half only. The assistant
axis is built from 24 roles rather than 275, so its PC1 agreement is the check that it is real.
The persona and introspection results are on a 3B coder model with a 4-bit base and a LoRA
attached; the source papers worked with 27-70B open models and frontier Claude, and effect sizes
should be expected to be smaller here. That is not a reason not to measure them; it is the
reason the decision rules are stated relative to controls rather than to the papers' numbers.
