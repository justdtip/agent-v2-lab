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

## Whether the reading-side readout is an instrument or a frequency counter

**Measured because the Chief proposed abandoning it — "a top-k over a prompt the model has just
been shown is dominated by what is in that prompt, and may not be a good instrument for this
question at all." Both of us had had a comprehension-side claim dissolve, and the shared cause
looked like context frequency.**

Every reading position in the final turn of the eight complete stage-two episodes, at layer 34,
which is the model's own distribution and involves no lens. Bucketed by how many times the true
next token had already appeared earlier in that prompt:

| times the true token appeared before | n | in the top ten | is top-1 |
|---|---:|---:|---:|
| 0 | 2,238 | 51.5% | 22.6% |
| 1 to 2 | 1,843 | 82.4% | 59.3% |
| 3 to 9 | 2,576 | 94.6% | 80.2% |
| 10 or more | 3,918 | 96.8% | 83.1% |
| all | 10,575 | 84.2% | 65.4% |

**The Chief is right about the comparison and wrong about the instrument, and the numbers separate
the two cleanly.**

Right about the comparison: the gradient is enormous. A token seen ten times is in the top ten 97%
of the time; one never seen, 52%. So contrasting a position whose true token has appeared 28 times
against one whose true token has appeared **zero** times — which is exactly `context` against
`Result` at the fork we both wrote up — is confounded before any schema is invoked. Both of our
comprehension-side claims died on this and it is the same death.

Wrong about the instrument: **51.5% in the top ten for a token that has never occurred, out of a
vocabulary of 262,208, is not a frequency counter.** Chance is 0.004%. The readout carries real
predictive content on unseen tokens; what it cannot do is support an uncontrolled comparison
across positions of unequal token history.

**The rule this earns, and it is a filter rather than an abandonment.** Any comparison of the
reading-side readout between positions must match on the true token's prior occurrence count in
that prompt, or the frequency gradient dominates the result. That is also the exact selection both
of us independently proposed for the matched pass/fail pair — now with the reason quantified rather
than suspected, and with a threshold: below about three prior occurrences the readout is
substantially weaker, and that is where both of our claims sat.

## The map: the pre-registered primary comparison

**Stage two complete. Twelve of twelve families, fifteen episodes, nothing missing, 4.5 hours,
534,990 rank rows, peak 4.70 GiB. Spans joined through `live_lens.spans`, whose assertions run
before any faceted number exists.**

**Call arguments against call skeleton, horizon 1, per layer** — the comparison the pre-registration
was restated to, holding format constant and varying only whether the content is task-dependent:

| layer | argument median rank | skeleton median rank | log2 ratio |
|---:|---:|---:|---:|
| 5 | 1,792 | 507 | +1.82 |
| 10 | 1,186 | 474 | +1.33 |
| 16 | 9,954 | 1,194 | **+3.06** |
| 19 | 20,018 | 5,402 | +1.89 |
| 21 | 4,636 | 570 | **+3.03** |
| 23 | 785 | 115 | +2.77 |
| **24** | **2** | **2** | **0.00** |
| 25 | 1 | 2 | −1.00 |

**Through the whole pre-collapse stack the task-dependent content is held one to two orders of
magnitude further from the output than the convention that carries it, and the difference closes
exactly at layer 24.** At 24 the arguments are also *more* often already decided than the skeleton —
45.7% at rank 1 against 32.3% — having been eight times worse one layer earlier.

### It survives both controls, and one of them strengthens it

**Deduplicated** to one entry per distinct decision per episode, so a repeated call counts once:

| layer | raw | deduplicated |
|---:|---:|---:|
| 16 | +3.06 | +2.73 |
| 21 | +3.03 | **+3.82** |
| 23 | +2.77 | **+4.14** |
| 24 | 0.00 | 0.00 |

The effect **grows** at the layers where it is largest. Repeated decisions were diluting it, not
producing it — which is the opposite of what deduplication did to every other candidate finding in
this programme.

**Per episode**, 11 of 12 positive at layer 16, 10 of 12 at 21, 11 of 12 at 23, medians +3.17 to
+4.46. It is not carried by one episode.

**The single exception is `update-0028`**, negative at every layer, and its cause is known: its
twenty-four arguments are one character each, `"`, `/` and `.`, so its "task-dependent content" is
more predictable than its scaffolding. The exception is the degenerate case rather than a
counterexample.

### What it is not

Every reading sits above token position 126 and the lens was fitted at 16 to 126, so the whole table
is an extrapolation of three to eighteen times, disclosed in full above. The comparison is a
*correlation with depth*, not a causal claim: nothing here shows that layer 24 is where the model
decides, only that it is where a readout trained toward the output stops distinguishing the two
spans. And horizon 1 under greedy decoding saturates at the final layer by construction, which is
why the null moved to horizons 4 and 8.

### Layer 24, measured as a distribution, under a stated deduplication rule

**The deduplication rule, because two seats computed this from the same rows and differed by seven
points.** Within an episode, each turn's emitted call token sequence is collected, and a turn is
kept only if that exact sequence has not appeared in an earlier turn. **Deduplication is per turn,
keyed on the whole call.** A turn whose call differs by one character is kept in full. 65 turns
kept, 23 dropped across the twelve agentic episodes.

My first version keyed on the *decision* — token plus its predecessor, collapsed across turns — and
it was wrong for this comparison in a way that flattered it. Skeleton tokens repeat identically
across turns while arguments vary, so a token-level key deduplicates skeletons far harder than
arguments and discards independent observations from different turns. The Chief diagnosed it from
the asymmetry alone: the argument rows agreed and only the skeleton's rank-1 share and far tail
moved. Recomputed under the per-turn rule, my numbers reproduce theirs exactly.

**Layer 24** (n = 1,241 argument, 1,312 skeleton)

| span | rank 1 | 2–10 | 11–100 | 101–1k | 1k–10k | >10k |
|---|---:|---:|---:|---:|---:|---:|
| call argument | **46.2%** | 19.7% | 16.0% | 6.9% | 6.3% | 4.9% |
| call skeleton | 31.8% | **36.9%** | 13.7% | 8.5% | 3.0% | 6.1% |

**Layer 25**

| span | rank 1 | 2–10 | 11–100 | 101–1k | 1k–10k | >10k |
|---|---:|---:|---:|---:|---:|---:|
| call argument | **59.7%** | 21.8% | 9.3% | 5.0% | 2.4% | 1.9% |
| call skeleton | 46.5% | **34.9%** | 7.9% | 7.3% | 2.9% | 0.5% |

**What holds, and it is large under either rule.** Arguments lead on being exactly right, 46.2%
against 31.8%; skeletons lead on being *nearly* right, 36.9% in ranks 2 to 10 against 19.7%. That
gap is the finding: **the argument is more often decided, the skeleton more often almost-decided.**

**What weakens under the correct rule, and I had overstated it.** I wrote that the argument is
"decided or nowhere". The far tail is 11.2% beyond rank 1,000 against the skeleton's 9.1% — the
same direction, but a two-point difference rather than the four I reported from the over-deduplicated
version. The bimodality is real at the top of the distribution and slight at the bottom.

The primary comparison's log2 median ratios under this rule are **+3.80 at layer 21** and **+3.22 at
layer 23**, against raw values of +3.03 and +2.77. The result that deduplication *strengthens* the
effect survives the change of rule; only its size moves.


## Closing the stage-two analyses

**Three readings that need no model and no GPU, done before the CUDA migration starts, because
they are the evidence base every extension in the migration plan is motivated by.**

### 1. The harness's own verdict reasons, replacing my reading of the trajectories

Every episode's recorded actions replayed through a fresh simulator. The replay reproduces all four
passes exactly, so the reasons below are the harness's, not an interpretation:

| episode | reasons |
|---|---|
| `read-0108`, `calculate-0158`, `synthesis-0039`, `cross_reference-0032` | passed |
| `search-0061` | wrong answer |
| `pointer_chain-0018` | wrong answer |
| `list-0149` | wrong answer; **unexpected file change** |
| `ledger_reconcile-0163` | wrong answer; file state wrong |
| `conditional_update-0093` | wrong answer; file state wrong; unexpected file change |
| `batch_update-0166` | **file state wrong ×3, and no wrong-answer reason at all** |
| `update-0028` | no finish call; file state wrong; required tools unused |
| `aggregate_report-0167` | no finish call; file state wrong; required tools unused |

| reason | episodes |
|---|---:|
| file state wrong | **7** |
| wrong answer | 5 |
| unexpected file change | 2 |
| no finish call | 2 |
| required tools unused | 2 |

**The most common failure is leaving the workspace wrong, not answering wrongly.** And
`batch_update-0166` carries no wrong-answer reason, which confirms from the harness what I had
inferred from the answer strings: it said the right thing and did the wrong thing. **The class I
called "wrong-answer termination" is named after its second most common reason.**

The taxonomy is also finer than my trajectory reading. `list-0149`'s gratuitous write is penalised
explicitly as an unexpected file change; `conditional_update-0093` carries all three of wrong
answer, the file it should have changed unchanged, and the file it should not have changed changed.
That is the incomplete survey stated by the harness rather than by me.

### 2. The token cap reaches exactly one episode

| | |
|---|---:|
| turns across all fifteen episodes | 94 |
| turns reaching the 200-token cap | **1** |
| episodes affected | `aggregate_report-0167` only |

So the cap is not a systemic confound. It is one episode, scored as a model failure, whose
terminating turn was truncated mid-call — and that one episode also failed the same way in stage
one. A single instance, and it should be disclosed rather than counted.

### 3. The primary comparison by position band

Deduplicated per turn, layers 21 and 23, as log2 of the argument median over the skeleton median:

| band | layer 21 | layer 23 | n argument | episodes |
|---|---:|---:|---:|---:|
| 500–800 | +4.79 | +4.19 | 277 | 12 |
| 800–1200 | +5.38 | +4.77 | 195 | 9 |
| 1200+ | **+2.22** | **+2.26** | 769 | 6 |

**The effect is present in every band and roughly halves above position 1,200.**

**And it is not the loop episode.** Excluding `update-0028` changes the first two bands by at most
0.31 and the 1200+ band **not at all** — identical to three decimals, same n. Its twenty-four turns
deduplicate to two, both early, so it contributes nothing above position 800. The deduplication rule
is doing exactly what it was written for.

**What the halving means cannot be settled here.** Greater extrapolation degrading both arms toward
noise would shrink the gap; so would a composition difference between the six episodes long enough
to reach those positions. This corpus cannot separate them, which is the amendment-9 obstacle
arriving from a third direction — and it is the most direct motivation yet for a lens fitted at
transcript length, where the question becomes measurable instead of confounded.
