# Plan progress as the state programme's second state variable, and the 12B capture that would test it

**Draft, D-CRO, 2026-09-09, on the Chief's instruction. Cites plan §16.18.** Nothing here has run.
This is a design to be ruled on, not a result, and it is written before any 12B residual has been
captured so that the pre-registration cannot be shaped by looking. Where a choice is the Director's
or the Chief's it is marked **open** rather than settled quietly.

## 1. The hypothesis, stated as one

The Director's, in his terms: **the 12B carries a richer representation of the task than the 4B,
possibly a maintained sequence of intermediate-step representations, possibly only under
fine-tuning.**

Three claims are folded together there and they fail differently, so the draft separates them:

| | claim | how it can be false |
|---|---|---|
| H1 | the 12B's residual stream is **more predictive** of task state than the 4B's | both equally predictive once probe capacity is matched |
| H2 | that state is a **maintained sequence** of intermediate steps, not one recomputed each turn | the state is recomputed from the transcript at every decision |
| H3 | H1 or H2 hold **only after fine-tuning** | they hold in the base models, or in neither |

H2 is the one the derivation is built for and the one the programme has never been able to test,
because the agent protocol retains the transcript by construction (`§12`, and the bridge order's
second standing caution). H1 is a confound problem before it is a measurement. H3 is an arm.

## 2. The state variable: plan progress

**File existence** is the first state variable and is ordered
(`STATE-PROGRAMME-RUN-ORDER-2026-09-10`). This is the second.

Every expert trajectory in `pipeline/tasks.py` is a sequence of steps with a known index: the maker
writes `steps`, the runner records one action per turn, and `Task.horizon` is the expert's step
count. So for every episode and every decision position there is a ground truth **plan progress**
pair: the index of the step being taken, and the number of steps remaining. It needs no annotation
and no model to produce it, which is the whole reason to use it: the state is a property of the
task, observable to the experimenter and never told to the model.

That is also its limit, and the draft says so rather than leaving it: the model's own plan need not
be the expert's. An episode that reaches the same answer by a different route has a true step index
under its own plan and a misleading one under the expert's. **Open:** whether the estimands are
computed on all episodes, or only on those whose action sequence matches the expert's prefix. The
second is cleaner and smaller; the first is the population the programme cares about.

## 3. The estimands, the derivation's, on both models over the same episodes

The corpus is one set of episodes rendered once. The 4B and the 12B share a tokenizer and a
template byte for byte (`configs/models/gemma3-12b-cuda-bf16.yaml`), so **the same rendered episode
is both models' input**, and "matched positions" means the same token index in the same episode.

**E1, predictive sufficiency.** Does a feature state read from the residual predict the next action
and the steps remaining, *beyond what the transcript alone predicts*? The baseline is not chance: it
is a predictor given the visible transcript, because a transcript that says "read three files,
two remain" makes the answer readable without any internal state. The estimand is the gain over
that baseline, per model, at matched positions.

**E2, update closure.** As a step completes, does the state move to the successor the task's own
dynamics imply? The derivation's closure condition is that the abstraction commutes with the
transition; here that is: the state read after step *k* completes is the state the model would be
in had it started at step *k+1*. Measured as a distance between the read state and the predicted
successor, against the tolerance the pilot derives, exactly as the first state variable's is.

**E3, retention under carrier controls.** The one that needs the intervention. Replay identical
tokens so the generated-text path is fixed, then reset or exchange the retained activations, and
see whether the state survives. Without that control, a feature that appears "maintained" across
turns is equally explained by the model recomputing it from a transcript that never changed.
**WS-B's cache exchange is the separating control**, and it is unported and refused today
(SWE-1, this morning), so E3 is ordered after it and not before.

**The fine-tuning arm** repeats E1–E3 on a fine-tuned checkpoint, and is H3. It is last: it costs a
training run and it means nothing until the base measurement exists.

## 4. The two rules this draft is required to carry

**Richer means more predictive at matched probe capacity.** The 12B's stream is 3,840 wide and 48
deep against the 4B's 2,560 and 34. A probe fitted on the 12B has more parameters and more layers to
choose from, so it will read more of *anything* — including noise — and an unmatched comparison
measures the architecture rather than the representation. Three ways to match, and the choice is
**open**:

- **fixed rank**: constrain the probe to *r* directions on both, *r* chosen below the 4B's width;
- **fixed parameter count**: give both probes the same number of free parameters, so the 12B's probe
  is proportionally lower rank;
- **matched control features**: the map's own discipline, a control set matched for layer,
  activation prevalence, norm and selection opportunity, per the derivation's §14.

Whichever is chosen, the **same** capacity applies to both models and to the controls, and a result
reported without naming its capacity is refused. This is the same shape as the golden test's
comparability gate: a difference between two things measured differently is not a measurement of
either.

**The pre-registration is written before any 12B capture is read.** Diagnostics, tolerances,
the capacity rule, the episode set and the analysis are sealed first, by the same mechanism the run
order already uses: `preregistration.json`, refusing to start without it and refusing to re-derive
once a main row exists. The captures may be *made* first — they must be, since they are made on the
card in the same pass as the lenses — but they are not looked at.

## 5. The capture plan, so the data exists when the pre-registration does

Residual captures of both models over the same episodes, **at decision positions, every layer**,
made on the card in the same pass as the lens fits.

**Why decision positions and not every position**, in numbers rather than in principle. Per position
a capture is `(num_layers + 1) × hidden × bytes`:

| | layers captured | hidden | per position, bf16 |
|---|---:|---:|---:|
| 4B | 35 | 2,560 | 175.0 KiB |
| 12B | 49 | 3,840 | 367.5 KiB |

At an assumed 300 episodes and 12 decision positions each:

| | decision positions only | every position (≈2,000 tokens/episode) |
|---|---:|---:|
| 4B | **0.60 GiB** | 100 GiB |
| 12B | **1.26 GiB** | 210 GiB |
| both | **1.86 GiB** | 310 GiB |

The card has 236 GiB free. Decision positions fit with about two orders of magnitude to spare;
every position does not fit, and would not fit even alone for the 12B once the checkpoints and the
lens artefacts are on the same disk.

*The right-hand column was 1,024 times larger in the first draft of this table, TiB written where
GiB was meant, which would have made "every position" look absurd rather than merely impossible.
Caught by computing it rather than by reading it back, which is the only way this class of error is
ever caught.*

**The row counts above are assumptions and are marked as such**: the
episode count follows the state programme's pilot and the decisions-per-episode figure is a laptop
observation, so both are re-measured from the first captured episode and the manifest carries the
measured value beside the assumed one.

**Stored in the model's own precision.** bf16 on disk, because the capture must not change the
arithmetic it is capturing — the promoted-float32 finding, at 69.4% on this card at 1,400 tokens, is
the reason that sentence is here rather than assumed. `capture_dtype: native` in every manifest.

**What each capture carries**, so a reader can tell what it is: the checkpoint digest, the episode
id and the step index, the token index, the layer, the decoding mode, the device reading, and the
`basis` field per cell. The ground-truth plan progress is stored beside it and comes from the task,
never from the model.

## 6. What this draft does not do

It does not choose the capacity rule, the episode population, or the tolerances; those are §4's
open items and the pilot's. It does not order the fine-tuning arm. It does not assume E3 can run:
it cannot until the cache exchange is ported, and saying so is the point of listing it.

And it does not claim the 12B is richer. The programme's own record of this week is that the
instrument fails before the model does — a comparison across two arithmetic paths, a check that
could not pass, a gate whose premise was wrong rather than whose port was. **A first result showing
the 12B more predictive than the 4B is, until the capacity rule is applied and the controls run,
a measurement of the 12B being bigger.**

## 7. Where this sits

Cites plan §16.18. The first state variable's order is
`design_specifications/pending/STATE-PROGRAMME-RUN-ORDER-2026-09-10.md`; this second one reuses its
machinery unchanged — the same diagnostics-as-code discipline, the same derivation table, the same
seal, the same `measured / untestable / not measured` three-state reporting — and adds only the
state variable, the capacity rule and the capture plan.
