# Codex work order, in sequence

**Issued by the Chief on the Director's instruction, 2026-09-08.** Three tasks in order. Task 3's
first half is blocked on nothing and should fill any wait in tasks 1 and 2 rather than sitting idle.

Everything here is subject to the rules landed today: R57 and R60 on identity and lens ontology,
R58 on how a patch is made, R56 on what makes a gate evidence.

---

## Task 1 — build the agent-transcript lens fitter

**Why.** Both lenses we hold were fitted on WikiText prose at 128 tokens: the hosted one by
Neuronpedia, yours by design to match it, which was correct for a comparison and for nothing else.
Of the 205,824 positions in your fit, **205,823 are plain prose**; not one is a tool call, a note,
an observation, or a token the model generated. The map reads agent transcripts. A lens has never
seen the distribution it is being asked to read.

**What to build.** The fitting path you already have, taking its corpus from **Gemma's own agent
rollouts** instead of prose.

1. **Rollouts.** Gemma over the *training-split* agent tasks through the evaluation stage, exactly
   as the Qwen rollouts were produced, at the evaluation's step ceiling and not the pilot's.
2. **Capture.** Residuals during those rollouts, through the model's own forward. Your
   `native_residuals` path already does this and is unaffected by the hand-run loop's history.
3. **Fit.** The same regression estimator, at **2,048 tokens**.

**The context length is not a preference.** It must exceed Gemma's 1,024 sliding window. Below the
window the windowed and globally-attending layers are the same layer, which is why every claim
about that contrast has been void for both existing lenses. Fitting above it makes that comparison
possible for the first time in this programme.

**The corpus must span the spans.** Calls, notes, observations and generated tokens, in the
proportions the model actually produces them. Record the per-span position counts in the manifest;
the prose fit's 205,823-of-205,824 is the number that made this task necessary and its successor
must be inspectable the same way.

### One hard dependency, and it is not optional

**Do not generate the rollouts until the observation-rendering fix has landed and been shown to
work.** Today's stage one rendered tool results as bare user turns, which taught the model that
when it executes a tool the person answers and its next output is a reply. Rollouts generated under
that rendering would bake the handicap into the lens permanently, and no later fix would reach it.
The Deputy is running the confirming test now. Wait for it.

### Fit both precisions, and treat the difference as a result

Fit on **`gemma3-4b-bf16`** and again on **`gemma3-4b`** (4-bit), same corpus, same estimator.

The bf16 fit is the reference: it matches the hosted lens's precision so the comparison stays like
for like, and the SAE bridge in task 3 **requires** it. The 4-bit fit matches the checkpoint the map
actually runs, removing one mismatch from the map's primary reading.

Rather than choose, **measure**: the difference between the two lenses is the first number anyone
here will have on what quantisation does to a fitted lens, and the bridge's precondition makes us
want it regardless. Report it per layer, relative and not absolute, with the map's own norm beside
it per R56(d).

### Acceptance

- Per-span position counts in the manifest, and at least one span other than prose above ten per
  cent.
- Context length above 1,024 recorded, with the count of positions that actually exceed it.
- Both precisions fitted, their per-layer difference reported relative.
- The identity block per R57: base checkpoint, training applied (none), depth. It must load under
  the lineage rules without a special case.
- A declared window with a **projected peak and its basis** before the run, per R60(c). Your prose
  fit peaked at 12.44 GiB on 128-token windows; 2,048-token windows are a different shape and the
  projection should say so and say from what.

---

## Task 2 — run it

The fit itself, both precisions, under one announced window.

**What it produces:** the first lens in this programme fitted on the model doing the work it will be
read on, above the window that makes its layer structure visible.

**What it does not settle.** It is a regression lens: a corpus-fitted predictor of the final
residual. It is not a Jacobian and the two estimate different things, so a comparison against the
hosted lens remains a *measurement of where a fitted predictor and a local linearisation coincide*
and never a validation of either. That ruling stands from this morning and you were right to make
it.

When it lands, the map's pre-registration gets a dated amendment naming which lens it reads and why,
and the secondary comparison — globally against window-attending layers — becomes attemptable for
the first time.

---

## Task 3 — the SAE-to-J-lens bridge

Specified in full at `SAE-J-BRIDGE-ORDER-2026-09-08.md`. Two stages, and read the "do not build"
section before starting: the causal-abstraction programme in that derivation needs about 2,400
independent episodes per arm against our measured fifteen per hour, and is not ordered.

**Stage A1 is blocked on nothing and should start now.** It needs three matrices already on disk, no
model forward, no box time, about 150 seconds per layer on the processor. For each sparse feature it
computes what that feature pushes the output toward, read from a given layer, to set beside the
feature's published label. Run it against the bf16 entry: the dictionaries were trained on
unquantised weights and for this bridge that is a **failed precondition rather than a disclosure**.

Never form the full transfer matrix; it is 17 GB per layer. Compute the lens-times-decoder product
first, then take the unembedding column by column keeping only the top tokens.

**Stage A2** follows once a lens fitted on transcripts exists, because reading a run's activations
through a prose lens would inherit the problem task 1 exists to remove.

**Stage B only if A justifies it.** Its dictionary is 2.7 GB per layer and 89 GB across the model
and must be built on demand and never stored.

---

## Two cautions that apply to all three, from the derivation and already in the map's pre-registration

**An averaged Jacobian can cancel a causal sensitivity.** Writing the context-specific Jacobian as
`K` and its average as `J`, the identity `E[K'K] = J'J + E[(K-J)'(K-J)]` gives `||J d|| <= E||K d||`
for every direction. Two contexts with opposite Jacobians average to zero while each preserves the
norm. **So a small reading through a lens is never evidence of a small causal effect**, and any null
must be stated in those terms.

**Mutual information between a feature at two times does not identify persistent memory.** If a
transcript retains the fact and the model recomputes it at each event, the information is positive
while nothing is carried. Our agent protocol retains the transcript by construction, so this trap is
directly under us. Persistence claims need interventions with carrier controls, not correlations.

---

## What all of this may and may not be said to show

These are instruments. A contribution to a lens score is a contribution to that score, at that
layer, under that averaging convention. It is not a token probability, not a causal derivative, and
not evidence of a workspace, a maintained state or flexible access. Write the records so that the
finding, the technique and our implementation are separable, per the Director's standard: the
technique is the part that transfers and it is the part usually left implicit.
