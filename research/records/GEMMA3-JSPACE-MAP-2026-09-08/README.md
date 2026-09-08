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

## Where the map reads, against where the lens was fitted

**Measured off the records already on disk, no model and no box, before stage two's figures are
read. The lens was fitted at absolute positions 16 to 126: `SKIP_FIRST_N_POSITIONS = 16` in the
upstream fitter, which excludes attention sinks as having atypical residual statistics, and the
final position is dropped for having no next-token target.**

Every scored position in the corrected run and stage one, by where it falls relative to that range:

| | rank rows | share |
|---|---:|---:|
| below 16, the excluded attention-sink regime | 0 | 0.00% |
| inside 16 to 126, where the lens was fitted | 1,144 | 2.48% |
| above 126 | 44,936 | 97.52% |

**The attention-sink worry does not reach this instrument at all.** Rank rows exist only for
generated tokens, generation begins after the prompt, and the smallest opening prompt in the set is
420 tokens. No scored position is below 16 and none can be. The 1.3% of *reading* rows that do sit
there are prompt positions, which carry no rank rows and enter no comparison.

**The extrapolation is not partial, it is total.** Every agentic read — the whole primary
comparison — is outside the range the lens was fitted on, by a factor of at least 3.3 and up to 18.
The only in-range reads in the entire corpus are 1,144 from two short chat episodes.

### What those two episodes let us measure

They are the only place the lens is used as fitted, and they cross the boundary within a single
episode, so they give a comparison rather than a caveat:

| layer | in-range median | in-range rank 1 | out median | out rank 1 |
|---:|---:|---:|---:|---:|
| 11 | 886 | 1.4% | 715 | 1.0% |
| 17 | 2315 | 0.7% | 1760 | 1.9% |
| 23 | 34 | 23.1% | 22 | 13.3% |
| 24 | 5 | 32.2% | 5 | 31.4% |
| 30 | 1 | 68.5% | 1 | 68.6% |
| 34 | 1 | 100.0% | 1 | 100.0% |

**No systematic degradation is visible at the boundary.** Medians and rank-1 shares track each
other at every layer, with layer 23 slightly better outside and layer 17 slightly worse, on 143
against 105 reads.

**And that is a much weaker reassurance than it looks.** The out-of-range side here reaches about
position 250 — an extrapolation of roughly **2x**. The agentic episodes run to 2,298, which is
**18x**. This measures the near boundary and says nothing about the far tail, and there is no
in-range comparison available at any depth of context because the corpus contains none. A map read
at 2,000 tokens is resting on an instrument validated to 126 and spot-checked to 250.

**The honest statement for anything stage two publishes:** the lens is used outside its fitted
position range for 97.5% of reads and for 100% of the primary comparison; the one available
boundary check finds no degradation at 2x and nothing tests 18x.

### The boundary check corrected, and the differential test that replaces it

**The Chief's three objections to the check above are correct and it is downgraded to a spot-check
with its limits stated.** In particular one I should have seen in my own printed table:

**It confounds position with context.** Out-of-range reads sit later in the sequence and have more
context, which makes next-token prediction easier independently of the lens. Degradation and an
easier task push opposite ways and a level comparison cannot separate them.

**Its two statistics disagree where it matters.** At layer 11 the median *improves* out of range,
715 against 886, while the rank-1 share *falls*, 1.0% against 1.4%. At layer 23 the same split, 22
against 34 and 13.3% against 23.1%. Median and top-1 measure different parts of the distribution
and they point opposite ways at exactly the shallow depths the map's interesting claims live at.
They agree only at 24 and 30, where both are near saturation. **A check whose two statistics
disagree is unresolved, not null.** I printed both and read them as agreement.

**And 2x is not 18x.** No in-range comparison exists at any real depth of context, and this run
cannot produce one.

**The differential test survives the confound**, because both arms sit at the same positions: if
extrapolation damages the transported representation, a longer-range prediction should suffer more
than a next-token one, which leans on the identity. Same two chat episodes, split at 126, by
horizon, as log2 of the out-of-range median over the in-range median — positive means out-of-range
is worse:

| layer | horizon 1 | horizon 8 |
|---:|---:|---:|
| 11 | −0.31 | +0.03 |
| 18 | +0.35 | **+1.44** |
| 23 | −0.63 | +0.53 |
| 24 | +0.00 | **+1.01** |
| 30 | +0.00 | **+1.28** |

**The sign flips in the predicted direction and all five layers agree at horizon 8**, where
out-of-range medians run one to two and a half times worse. At horizon 1 the directions are mixed,
which is what the context advantage would produce.

**It is suggestive and it is not established, and the distinction is the whole point.** Permutation
tests on the median ratio give p = 0.026 at layer 30 and 0.071 at layer 18; the rest are far from
significance. Ten tests were run, so nothing survives a Holm correction — 0.026 × 10 is 0.26. The
five agreeing signs are worth something and less than they look, because the layers share positions,
episodes and a residual stream and are not independent draws.

**So the honest statement is that the only evidence bearing on whether the extrapolation matters
points at "it does, at long horizons and at depth", and is too weak to act on.** It is recorded
because it is the sole measurement that exists on the question, and because writing it up as a
result would be the day's tenth instance of the pattern in `METHOD-2026-09-08`.

## The stage-two declaration against its measurements, so far

**Recorded mid-run, 6 of 15 episodes, because the method error is visible now and the final peak is
not. The numbers close out when the run does.**

| | declared | observed at 6 of 15 |
|---|---:|---:|
| peak memory | 3.4 GiB | 4.70 GiB |
| rate | 1.80 s/token | 3.68 s/token overall; 5.01 on the long episode, 2.15 on the other five |
| window | 5.5 h, then 7 h | extended to 9 h; naive projection 5.2 h |

Neither overrun is near a threshold, R47's stop sits at 10.66 GiB, and the window has headroom. **The
projection method is the finding.** Both figures were measured on `calculate-0158`, two turns and
103 tokens, the shortest episode in the set, because it was the cheapest to run. Memory and per-token
cost both scale with context, and the episodes that follow run to 2,749 positions. Projecting a
context-scaling quantity from the shortest instance understates it by construction, and the 38%
memory miss and the 2.3x rate miss are the same mistake once each.

The rule, for the next declaration: **calibrate at the longest context the run will reach, not the
shortest**, and treat episode length rather than episode count as the driver. The Chief caught it
because the declaration carried its measurement, which is what R60(c) is for.

Composition at 6 of 15: `update-0028` is 1,050 of 1,965 generated tokens, 53.4%, against the
pre-run projection of 34.1% of rank rows. The equal-weight-per-episode rule was fixed before the run
and handles it; the projection was the optimistic one.
