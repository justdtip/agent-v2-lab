# Gemma 3's J-space: the map, and the instrument that carries it

**2026-09-08, open. Stage one has run; stage two has not. This record exists now because the
instrument evidence belongs beside every number the map will publish, not in a message.**

## The layer-34 check, and what it does **not** prove

**Corrected 2026-09-08 after the CRO read the capture code. The claim that stood here was too
strong and the record now says what the check actually establishes.**

| | |
|---|---|
| horizon-1 reads at layer 34 | 5,316 |
| returned at rank 1 | 100.00% |
| agentic episodes covered | 11 |

The number is right. The reading of it was not. `session.py` substitutes the model's own softmax
over the native logits at `layer == num_layers` and never calls the readout there:

```python
np.array(mx.softmax(logits[0, local].astype(mx.float32)))
if layer == self.view.num_layers
else self.readout(h[0, local], layer)
```

So there is **no lens map at layer 34, no residual tap and no readout orientation** in that path.
1.000 across 5,316 reads establishes that decoding is greedy, that the horizon is subtracted
correctly, and that the record's logits line up with the tokens actually emitted. Those are worth
having and they are bookkeeping, not instrument validation. This record previously said that
readout orientation, lens application and position bookkeeping could not all be wrong and produce
that figure. **Two of those three are not exercised.**

It still earns its keep in one place: it is the check that caught the reading-row join error
described in `DIAGNOSTIC-RERUN.md`, because a top-1 that must equal the emitted token fails loudly
the moment a join is wrong. A bookkeeping check that catches bookkeeping errors is doing its job.

**The version that would prove what we wanted** is one line: call the real readout at layer 34 as
well, and assert it agrees with the native logits. The final layer's map should be the identity, so
disagreement there is a defect in the readout path itself. That is the check we thought we had, it
is cheap, and it is a gate on stage two rather than a footnote to it.

`identity_layer_check.py` recomputes the figure from the committed records with no model and no
box; `identity_layer_check.json` is stage one's answer. Read it as the bookkeeping statement it is.

## The depth gradient behind it, which is not yet a result

| layer | span | median horizon-1 rank | rank 1 |
|---:|---|---:|---:|
| 11 | sliding | 1331 | 0.40% |
| 12 | global | 1282 | 0.23% |
| 17 | sliding | 4128 | 0.19% |
| 18 | global | 2699 | 0.47% |
| 23 | sliding | 249 | 8.84% |
| 24 | global | 3 | 36.47% |
| 30 | global | 1 | 73.61% |
| 34 | sliding | 1 | 100.00% |

The shape is what a reader expects and is worth nothing on its own: eleven episodes of one model
under one rendering. **Nothing here reads the 23-against-24 or the 30 difference.** Those are
layer pairs whose contrast is the map's actual question, and answering it from this table would
be answering it from the pilot's convenience sample.

## What the map compares, and what it does not

The primary comparison is **calls against notes, by span and by depth**. It reads *attempts*, not
completions: what the residual carries when the model is about to act, whether or not the action
turns out to be right.

**No facet on success.** Stage one passed 2 of 11 agentic episodes. A facet on an outcome with
two positives is not a comparison, and it stays out at any rate the corrected runs produce. If the
step ceiling and the rendering lift the pass rate, that is a fact for this record and still not a
facet.

## Provenance of the numbers above

Stage one, 15 episodes across 12 families, 8 layers, Gemma 3 4B at 4-bit against the hosted
bf16-fitted Jacobian lens under R60, `--max-steps 12`, seed 20260902. Its records are the control
for every later comparison and are not modified.

**The runs after stage one are not comparable to it in one respect and the map must say so.** Two
corrections landed on 2026-09-08: the step ceiling moved from 12 to 24, which is the height that
produced every Qwen number the map is read against, and the observation rendering changed from a
bare re-role to the workspace answering the model's own call. Both are independently right and
neither was made to move a number. They landed together, so a run that carries both tests whether
Gemma can do these tasks when the harness and the rendering are fair to it, and does **not**
attribute the difference between them. A reader should not take it as doing so.

## Decision: stage two runs under the fixed environment, and stage one stops being an outcome control

**Mine to make, recorded before the run because the pre-registration bars amendment once a record
is read.**

Two changes now separate stage two from stage one. The rendering: observations present as the
workspace answering the model's own call, with the convention declared in the system prompt. The
environment: `list_files` raises on a directory it cannot satisfy instead of answering `(none)`,
landed by the Chief at `020aa89`.

**Both go in, and stage one is no longer a control for outcomes.** The reasoning is the same on
each and it is not a close call. A corpus generated under a rendering that misdescribes the episode,
or under an environment that answers a question falsely, bakes that defect into everything fitted or
read on it, permanently and invisibly. Stage one's pass rate was measured against a simulator that
told the model its workspace was empty; preserving comparability with that number means preserving
the lie in order to keep a column aligned.

**What stage one remains a control for.** Instrument behaviour, which is what it was mostly worth:
the identity's bookkeeping check, the shape of the depth gradient, the per-layer read volumes and
the memory envelope. None of those depend on the simulator telling the truth. Its trajectories stay
readable as what this repository's environment produced before the fix, which is the only honest
description of them.

**What no reader may do with it.** Compare a stage-two pass rate against stage one's and call the
difference a model effect, a rendering effect or an environment effect. Three things moved and the
run was not designed to separate them. If that separation is wanted it is a deliberate follow-up
with one variable at a time, and it is cheap compared with getting it wrong here.
