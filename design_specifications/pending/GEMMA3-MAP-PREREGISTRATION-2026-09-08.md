# Pre-registration: the Gemma 3 representation map

**Written before any record exists, as the house rule requires.** The Chief's, on the Director's
order to map internal representations across a variety of tasks. Nothing below may be changed once
a record has been read; a change after that is an amendment, dated, with the reason.

## What is being run

The live-lens pilot on Gemma 3 4B, base model, no adapter, reading residuals through the hosted
Jacobian lens **while the model executes tasks**. Two stages: an instrument check at eight layers,
then the map at all thirty-four. Fifteen episodes — one per task family across all twelve, plus
three chat episodes including one whose context exceeds the sliding window.

The instrument-check stage is **not** a result. It is read only for whether the path worked, and
its numbers are not reported as findings.

## The rules, stated so they transfer

Every rule below is written in terms of the model rather than of our code or this checkpoint,
because the same rules must apply unchanged to the 12B repeat and to the next family. **A rule that
names a layer index is a rule that does not transfer**, so depth is a fraction and a layer's role is
its attention span, not its index.

### Primary reading

**Foreknowledge**: the share of emitted tokens whose competition rank in the lens distribution,
read `h` positions earlier, is within ten. Reported as a grid of **depth by horizon**, faceted by
span — note, call, observation, chat prose — with the **final layer's value as the base rate on
every row**, since the final layer's lens is the identity and its share is what the readout gives
with no transport at all.

The primary comparison, fixed here: **within agentic episodes, the share at horizon one in *call*
spans against *note* spans, per layer.** Calls are where the model commits to an action; notes are
where it deliberates. If commitment is visible earlier in depth than deliberation, that is the
first thing this instrument can say about how the model completes a task.

### Secondary reading, and the condition without which it is void

**Globally-attending layers against window-attending layers, at horizon four.**

A block's span is read from the model's own dispatch, not from a period rederived by us. For this
checkpoint that makes the globally-written residual layers **6, 12, 18, 24 and 30** in the
repository's one-based convention, and the rest window-written. This must be recorded per layer in
the manifest, because the layer-family machinery currently reports this backbone as dense — true of
its module kinds and false of the model — and a reader of the artifact must not inherit that.

**This comparison is restricted to positions beyond the sliding window and is void without them.**
A windowed causal mask is vacuous when the sequence is shorter than the window, so below it the two
kinds of layer are the same layer and any difference is noise. Report the count of qualifying
positions beside the comparison; if it is small, say so and draw no conclusion.

**A further caveat that cannot be removed by any amount of data**: the lens itself was fitted at a
sequence length of 128 against a window of 1,024, so every window-attending layer was fully causal
throughout its fit. This comparison therefore reads long-context behaviour through a lens that never
saw it. It is a measurement worth making and it is not evidence about the model until a lens fitted
above the window agrees with it.

### Secondary reading, unconditioned

**Lens-against-next-token top-one agreement by span and depth**: where along an agent's prompt the
lens reads at all. This is descriptive and carries no comparison.

## What is not being done

**No hypothesis test.** Fifteen episodes at one seed on one checkpoint is a map, not an experiment.
Every number is descriptive, intervals where they are meaningful, and no p-value is computed or
quoted. The Qwen pilot's corresponding figure is shown beside this one on identical axes where the
layers coincide; that juxtaposition is a comparison of two maps and not a test of a difference.

**No claim from a single layer or a single episode.** A pattern is reportable when it holds across
at least two task families or across a run of adjacent layers. One cell is an observation.

**No claim about the trained agent.** This is the base model. Every adapter this programme owns is
Qwen's, so nothing here speaks to what training does to Gemma.

## What would make this a null

Stated in advance so it cannot be reinterpreted afterwards. If foreknowledge shares at every
non-final layer sit within the final layer's base rate, the lens reads nothing this instrument can
distinguish from the readout itself, and the map is a null. That is a real possible outcome, it is
reportable, and on this checkpoint it would most likely indict the lens's 128-token fit rather than
the model.

## Recorded before the fact

The globally-attending layers, the span facets, the horizons, the primary comparison, the window
condition and the null are all fixed by this document. The layer set for the map is all
thirty-four. The episode set is the twelve family-stratified task ids and three chat episodes named
in `GEMMA3-WORK-ORDERS-2026-09-08.md`.

---

## Amendment: three cautions from the SAE/J-lens derivation, 2026-09-08

From a formal derivation supplied by the Director. Each was checked numerically before being
adopted; each bears on a number this map will publish.

**A low lens reading does not imply a low causal effect.** The hosted lens is an *averaged*
Jacobian. Writing the context-specific Jacobian as `K(w)` and the average as `J`, the deviation
`D = K - J` gives `E[K'K] = J'J + E[D'D]`, hence `||J.d|| <= E||K(w).d||` for every direction `d`.
Averaging can cancel a causal sensitivity entirely: with `K = I` and `K = -I` equally likely, `J`
is zero while every individual `K` preserves the norm. **Verified.** So a low foreknowledge share
at a layer is evidence about the averaged geometry and **not** evidence that the residual there
has little causal influence on the output. Any null this map reports must be stated in those terms.

**Positions within an episode are not independent samples.** Any interval or test quoted from this
map must be computed over episodes, not over the 106,322 reading rows. Token-level resampling
inside an episode does not create independent draws, and an interval that treats it as if it did
is overconfident by roughly the square root of the positions per episode. The map has fifteen
episodes; that is the sample size for every claim it makes.

**"Unexplained" is not "outside the workspace".** If any J-space occupancy figure is ever reported
beside these readings, it carries this caution: the residual left by a sparse fit at budget `k` can
itself lie inside the cone at that budget. With dictionary `[e1, e2]`, `k = 1` and `h = (2, 1)`,
the fit is `(2, 0)` and the residual `(0, 1)` is itself a cone member. **Verified.** A residual
means unexplained *by this fit at this budget*, and nothing more.

### And a correction to the Chief's own proposal of this morning

The Chief proposed composing the lens with Gemma Scope 2's decoder directions, so that each sparse
feature acquires a token-level readout. That composition is **exact** and it stands: applying a
linear readout to the exact sparse identity `h = b + Dz + e` gives `Lh = Lb + LDz + Le`, so the
signed quantity `z_i * (L D)_{v,i}` is that feature's contribution to that lens score, with the
whole error carried by `L.e`.

**What it does not give, and the Chief was not careful to separate, is a decomposition of the
residual's occupancy of J-space.** Two reasons, both verified. Decomposing each decoder direction
independently into J directions and summing does not yield the sparse fit of their sum: with
dictionary and decoder both the identity in two dimensions and `k = 1`, feature-wise decomposition
gives `(2, 1)` where the sparse fit of the same vector is `(2, 0)`. And energy is not additive over
a non-orthogonal dictionary, so a per-feature share of `||h||^2` is not a partition and can exceed
one. **A feature's contribution to a score and a feature's share of the geometry are different
quantities and must not be reported in the same units.**

### A precondition we do not currently meet

The derivation requires the sparse dictionary and the J dictionary to refer to the **same residual
hook, layer, coordinate scaling and model checkpoint** before their vectors may be combined. Layer
and hook align: Gemma Scope's residual site is the output of block N, which is this repository's
probe layer N+1, and the lens carries maps on the same index. **The checkpoint does not.** Gemma
Scope was trained on Google's unquantised weights; the map runs our 4-bit conversion. So the
quantisation mismatch already disclosed for the lens is, for the SAE bridge, not a disclosure but a
**failed precondition**. Any composition work must be done on the bf16 entry.

---

## Amendment, Chief, 2026-09-08, before stage two runs and before any stage-two record exists

Seven audited surfaces found five things in this document that would not have measured what it says
they measure. The house rule bars amendment once a record is read, so all of it is settled here.
Every change below is forced by evidence gathered since the document was written, and each names it.

### 1. The base rate is degenerate at the primary horizon, so the null moves

**What was written.** Foreknowledge is reported "with the final layer's value as the base rate on
every row", and the map is a null if "shares at every non-final layer sit within the final layer's
base rate".

**Why it fails.** Sampling is greedy. The final layer's lens is the identity, so at horizon one the
emitted token *is* that layer's argmax by construction: rank 1 for 5,760 of 5,760 emitted tokens,
exactly 100.00%. A base rate of 100% cannot be exceeded, so every possible outcome sits "within" it
and the null is true whatever the model does. **At horizon one the pre-registered null is not a
falsifiable statement.**

**Fixed.** The null is evaluated at **horizons four and eight only**, where the final layer's base
rates are 19.1% and 10.2%. At horizon one the final layer is recorded as the identity check it
actually is, not as a base rate, and no null is claimed there. The primary comparison stays at
horizon one — it compares two spans against each other, not against the base rate — but it may not
be described as exceeding or falling within any base rate.

### 2. The span facets do not exist in the records, so they are defined here

**What was written.** Results are "faceted by span — note, call, observation, chat prose".

**Why it fails.** No record row carries a span label. Rank rows hold position, layer, horizon, token
id, rank and probability, and nothing else, and no note/call segmentation exists in this document or
in the code. The boundary would therefore have been chosen after the records were read, which is the
exact thing a pre-registration exists to prevent. The observation facet is worse than undefined: it
is **empty by construction**, because rank rows are emitted only for generated tokens and
observations are never generated.

**Fixed.** The observation facet is struck; it can never be populated by this instrument. The
remaining spans are defined now, in terms of the model's output and not of our parser:

- **note** — generated tokens from the start of the turn up to and excluding the opening fence.
- **call skeleton** — the fence, the key names, the punctuation and the tool name: every generated
  token inside the fence whose value is fixed by the calling convention rather than by the task.
- **call argument** — the generated tokens inside the fence that carry a task-dependent value: the
  path, the query, the expression, the replacement text.
- **chat prose** — all generated tokens in a chat episode.

The segmentation is emitted into the record at write time as a per-position label, so that the facet
is a property of the data rather than of a later analysis.

### 3. The primary comparison is restated, because the original is not identifiable

**What was written.** "Within agentic episodes, the share at horizon one in *call* spans against
*note* spans, per layer. Calls are where the model commits to an action; notes are where it
deliberates. If commitment is visible earlier in depth than deliberation, that is the first thing
this instrument can say about how the model completes a task."

**Why it fails.** Three reasons, each sufficient.

*Format entropy.* Call spans are stereotyped fenced JSON, 17 of a median 36 tokens fixed
scaffolding, against free English in notes. "Calls are read earlier than notes" cannot be separated
from "JSON is more predictable than English". The lens compounds it in the same direction: it was
fitted on prose, so it is nearer in-domain on notes than on calls.

*The sign is already known and it is the wrong way round.* The one existing run has **note above
call at every non-final layer** — 0.4822 against 0.3815 at layer 24. The hypothesis as written is
contradicted by the only data we have, in the direction the confound predicts.

*Pooling destroys the effect.* The D-CRO's argument-start control, built across every episode:

| token class | n | at layer 24 | at layer 30 | median |
|---|---:|---:|---:|---:|
| argument-start, excluding `update-0028` | 128 | 1.6% | 79% | 30 |
| all other tokens, excluding `update-0028` | 5,442 | 29% | 44% | 30 |

Argument-start tokens commit **later** than ordinary tokens, not earlier, and they sit at the
opposite end of the depth range from the scaffolding they are pooled with. A single "call" number
averages two populations that behave in opposite directions.

**Fixed.** The primary comparison is **call argument spans against call skeleton spans, at horizon
one, per layer** — a contrast between two token classes inside the same syntactic object, produced
in the same turn, under the same format. It holds format, position and lens domain roughly fixed and
varies what the pre-registration actually cares about: whether the task-dependent content of an
action is represented at a different depth from the convention that carries it.

Note against call is retained as a **secondary, descriptive** reading, reported with the skeleton
and argument shares printed separately beside it, and it carries no claim about deliberation or
commitment.

### 4. Composition is bounded before the run, not described after it

**Why.** Projected at the corrected rendering, `update-0028` alone contributes 34.1% of stage two's
rank rows, the top three episodes 61.0%, and the top four 70.9%. That episode is 24 steps with two
distinct calls, one of them repeated 23 times. An unweighted pooled statistic over positions would
be a statement about that loop.

**Fixed.** Every reported figure is computed **per episode first and then aggregated across episodes
with equal weight**, so no episode's contribution scales with its length. The per-episode values are
published beside the aggregate. Where a figure cannot be computed per episode, it is not reported.

**And a rule the D-CRO's own record earned the hard way.** A decision the model repeats is one
observation, not many. Twenty-three identical calls are twenty-three copies of one commitment, and
pooling them produced a clean, false 24-against-30 result that dissolved on deduplication to n=5
against n=13. **No count over positions may be quoted as a sample size without deduplicating
byte-identical repeated decisions within an episode.** This applies to the fixed point in
`update-0028` above all, precisely because it is the most persuasive thing in the corpus.

### 5. The window-conditioned secondary is live, and was wrongly declared void

**What was written.** The comparison "is restricted to positions beyond the sliding window and is
void without them."

**Why it failed.** It was not void. `live_lens_pilot.py:176-181` hardcodes that only one episode puts
any position past 1,024, a figure derived from opening prompt lengths. Measured from the records:
**8 of 14 episodes** qualify, covering 26,336 of 106,322 reading positions (24.77%) and 3,140 of
5,760 emitted tokens (54.5%), with a maximum position of 2,298. Under the corrected rendering the
share rises further.

**Fixed.** The qualifying count is **computed from the run's own positions and recorded in the
manifest**, never asserted in advance. The comparison proceeds. The caveat that the lens was fitted
at 128 tokens against a 1,024 window stands unchanged and is the binding limitation on it.

### 6. The episode set is eleven families, not twelve

`agentic-d2-read-0108` was skipped as complete though its own record footer reads
`"status": "aborted"`, and it now sits in `pilot/quarantine/` and is absent from the manifest. Stage
one covers **11 of 12 families** and no reader of the artefact could have told. Stage two runs into
a clean directory, and the manifest records the family count it actually achieved rather than the
one it intended.

### 7. What the environment fix does to comparability, stated rather than discovered

`list_files` answered an unsatisfiable directory with `FILES: (none)` rather than an error, which
manufactured `update-0028`'s failure outright and is fixed at `020aa89`. Stage two therefore runs
against a **different environment** from stage one, on top of the different rendering. Outcomes are
not comparable between the two stages and no figure may be carried across them. This is the same
trade already accepted for the rendering fix and for the same reason: a corpus gathered under an
instrument known to be wrong cannot be repaired afterwards.

**One consequence for the fixed point.** Under the fixed simulator the model's opening
`list_files("/")` now returns an error rather than a false empty, so the belief that produced the
23-repeat loop may not form at all. The fixed point may simply not be there in stage two. That is
the correct outcome and not a loss, but it means **any validation that depends on that fork must be
performed against the stage-one and corrected-rendering records already on disk**, which are frozen,
and not assumed to reappear.

### 8. The position-range disclosure, which is now the binding limitation on this map

Added while stage two is generating, because it constrains **publication** rather than measurement and
so does not change what the run computes. Measured by the D-CRO off records already on disk at
`880e720`.

| where the lens reads | rank rows | share |
|---|---:|---:|
| below position 16 — the attention-sink regime the fit excludes | **0** | 0.00% |
| positions 16 to 126 — where the lens was actually fitted | 1,144 | **2.48%** |
| above position 126 | 44,936 | **97.52%** |

**My sub-16 concern is withdrawn and the premise it rested on was false.** I reasoned that with no
cache across turns every turn re-encodes from zero, so scored positions would re-enter the excluded
sink regime at each turn boundary. Rank rows exist only for generated tokens, generation begins after
the prompt, and the smallest opening prompt in the set is 420 tokens. Turn-relative offsets restart at
the prompt length, not near zero. No scored position is below 16 and none can be.

**The other half is worse than I stated it.** The extrapolation is not partial but total: **every
agentic read is outside the fitted range, and so is 100% of the primary comparison.** The only
in-range reads in the entire corpus are 1,144 from two short chat episodes.

**Every published per-layer figure from this map carries this sentence:** *the lens is used outside
its fitted position range for 97.5% of reads and for 100% of the primary comparison; the one available
boundary check finds no degradation at roughly 2x extrapolation and nothing tests the 18x the agentic
episodes reach.*

**The boundary check, and what it is worth.** Two chat episodes cross position 126 inside a single
episode, giving 143 in-range against 105 out-of-range reads:

| layer | in-range median | in-range rank 1 | out median | out rank 1 |
|---:|---:|---:|---:|---:|
| 11 | 886 | 1.4% | 715 | 1.0% |
| 23 | 34 | 23.1% | 22 | 13.3% |
| 24 | 5 | 32.2% | 5 | 31.4% |
| 30 | 1 | 68.5% | 1 | 68.6% |

**It is a spot-check and must be reported as one, for three reasons.** It confounds position with
context: out-of-range reads sit later in the sequence and therefore have more context, which makes
next-token prediction easier on its own, so "no degradation" may be degradation cancelled by an easier
task. Its two statistics disagree at the shallowest layer, where the lens does the most work — median
rank improves out of range at layer 11 while rank-1 share falls. And its out-of-range side reaches
only about position 250, so it measures a 2x extrapolation and is silent about 18x.

**No in-range comparison exists at any real depth of context, because the corpus contains none.** That
is not a gap this run can close.

**The fix is a fit at agent-transcript context lengths**, which is Codex's task 1 and would replace an
18x extrapolation with none. Stage two is not blocked on it — the map is worth having with the
limitation stated — and no per-layer figure is published without the sentence.

### 9. The extrapolation-cost test, pre-registered before stage two's records exist

**Declared conflict.** The Chief proposed this hypothesis and predicted its direction. The pilot
result agrees with that prediction. Everything below is therefore fixed **now**, while stage two is
still generating and no data on which to tune it exists, because a Chief reframing the statistics of
his own confirmed prediction after seeing the numbers is the tenth instance of this programme's own
documented failure pattern.

**The hypothesis, one sentence, directional, and already on the record in a message sent before the
pilot test was run:** if extrapolating the lens beyond its fitted position range costs accuracy, the
cost is larger at long horizons than at horizon one, because a long-range prediction leans on the
transported representation while a next-token prediction leans on the identity.

**Why it is one test and not ten.** The claim is about the **horizon-by-extrapolation interaction**,
not about any cell. Testing ten cells and correcting for ten answers a different question — *which
cell degrades* — that nobody asked. The pilot's Holm correction across ten was the right correction
for the wrong hypothesis, and it under-reports evidence for the pre-specified one.

**The statistic.** Per layer, `r_h = log2(median out-of-range rank / median in-range rank)` at horizons
1 and 8, then the paired difference `d = r_8 - r_1`. One number per layer. The test is on the
distribution of `d`.

**Pilot values, for reference and not as evidence:**

| layer | 11 | 18 | 23 | 24 | 30 | mean |
|---|---:|---:|---:|---:|---:|---:|
| `d` | +0.34 | +1.09 | +1.16 | +1.01 | +1.28 | **+0.98** |

All five positive; the mean is a factor of 1.97 on the median rank.

**The unit of replication is the episode, not the layer, and this is the binding constraint.** Five
layers of one residual stream over two episodes are not five draws: they share positions, episodes and
the stream itself. A layer-level sign test gives 1/32 and **that is an upper bound on the evidence,
not the evidence.** With two episodes the honest n is two. This is the same error as claim 6 in
`research/records/METHOD-2026-09-08/` — counting correlated observations as independent ones — and it
is named here so it cannot be committed later by either seat.

**So, fixed now:**

1. The test is computed **per episode**, giving one `d` per (episode, layer). Episodes are the
   replication unit and are aggregated with equal weight, per amendment 4.
2. One pre-specified directional test on the episode-level mean of `d`, one-sided, no correction,
   because there is one hypothesis.
3. The per-layer table is published as description, never as five tests.
4. **Confirming** is a positive episode-level mean `d` with an interval excluding zero. **Refuting**
   is an interval containing zero or a negative mean. **Both outcomes are reportable and neither
   blocks anything**; this test decides how loudly the extrapolation limitation is stated, not
   whether the map is published.
5. If it confirms, the deep-layer long-horizon readings are the least trustworthy in the map — and
   layers 24 and 30 at horizon 8 are exactly where the fixed point commits and where the depth
   gradient turns, so the penalty would be concentrated on every interesting result we have.

**Nothing here changes what stage two computes.** It fixes how one already-planned analysis is read.

### 10. The failure classes, fixed before stage two lands, because the ordered one addresses a minority

Amendment 8's strategy-switching measurement — whether `search_files` was ever called, and whether an
episode entered repeated failure without it — was ordered on the strength of one episode. Counted
across the 18 completed agentic episode-runs we hold, it addresses **3**. The modal failure is a
different shape entirely:

| failure shape | episode-runs |
|---|---:|
| **terminated on its own with a wrong answer** | **9** |
| passed | 4 |
| loop detected | 3 |
| exhausted without the loop flag | 2 |

**Nine of eighteen call `finish` and stop while wrong.** That is not a model failing to switch
strategy. It is a model that believes it is done, and the ordered measurement does not reach it in any
run.

**And the retrospective count is not trustworthy either**, which the D-CRO found while computing it.
`repeated_failure_step` counts consecutive observations beginning `ERROR`. Under the old simulator
`list_files` answered an unsatisfiable directory with `FILES: (none)` — a falsehood, not an error — so
the detector was blind to exactly the failure the pre-fix runs contain most of:

| run | steps | `ERROR` observations | `FILES: (none)` |
|---|---:|---:|---:|
| stage one | 89 | 16 | 6 |
| corrected rendering | 40 | 27 | 1 |
| stage two | 15 | 3 | 0 |

So the 3-of-18 pools two incompatible definitions. **Only stage two's rows measure what the field was
designed to measure.** Pre-fix and post-fix values of `repeated_failure_step` and
`repeated_failure_without_search` are not comparable and are never tabulated together.

**Fixed.** Every episode is assigned exactly one of four classes, reported as counts beside every
aggregate:

1. **passed**
2. **self-terminated wrong** — `not success and not exhausted and not loop_detected`
3. **looped** — `loop_detected`
4. **exhausted** — `exhausted and not loop_detected`

Class 2 needs no code, no rerun and no error text: it turns on the verdict alone, so it is
**simulator-independent and comparable across every run we hold**, which is the property the
strategy-switching field lacks.

**A hypothesis, recorded as one and not as a finding.** The D-CRO proposes that unwarranted certainty
about a path and unwarranted certainty about being finished are the same phenomenon at different
scales. The Chief tested it on the four completed stage-two episodes: answer-token median probability
was 1.000000 for a wrong self-terminated episode and 1.000000 for a passing one, which appeared to
confirm it. **The control refuted the test, not the hypothesis.** Median probability across *every*
emitted token is ~0.999999 in every episode regardless of outcome, so the statistic is saturated and
carries no information about answers specifically.

A test with power needs the probability the model assigns to the **correct** answer while emitting a
wrong one, which is not saturated. The obstacle is that only the top ten per layer are stored, so
where the correct token falls outside that the result is a bound — the regime that already bit at the
fork, where `workspace` was outside the top ten at every read layer. **What a bound-only result is
allowed to say is decided before that test runs, not after.**
