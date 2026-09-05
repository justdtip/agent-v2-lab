# EXP-003: how strongly is a concept represented as a function of its token distance back, inside the window?

> Read first: `01-IMPLEMENTER-BRIEFING.md`, `02-INTERFACE-AND-WIRING-MAP.md` (§7 R18a/b, R25,
> R26, R31, R34, R35, R38), `EXP-001-JSPACE-QWEN35-4B.md` and its readings,
> `under_review/WO-INTERP-002-DISTANCE-AND-RELEVANCE-2026-09-05.md` §6 and
> `under_review/LIT-REVIEW-WO-INTERP-002-AXIS-AND-RESOLUTION-2026-09-05.md`.
> Owner and author: Head of Interpretability. Status: pending.

## 1. Question, and why it is the live one

EXP-001 asked whether a concept survives leaving the window. The Director's critique of that
framing is accepted and recorded: no claim in the J-space literature asserts that it can, and our
own runner guarantees it cannot, since `TrimCache` discards what it cannot trim, `SnapshotCache`
reuses only the immutable prefix, and `ArraysCache` is untrimmable so a hybrid rebuilds. The state
at that decision never saw the filename.

**Inside the window is where the literature makes claims and where the note contract is actually
decided.** This experiment measures representation strength as a graded function of token distance
back, and its output is a **curve, not a verdict**.

**Why it changes a design decision either way.** If representation decays gradually, the window is
a tunable and D4's notes need carry less for recent content. If there is a cliff at distance d,
notes must carry anything older than d and d is a design constant. If it is flat, distance is not
the axis and EXP-004's relevance question inherits the whole weight.

## 2. Pre-registered predictions

**Decisive measurement:** the model's own next-token distribution over the target's first token,
paired per concept against the **never-mentioned** control, with the **mismatched** null beside
it. Both controls, per the Chief's amendment. The decomposition of EXP-001 §3.3 applies in full:
P(target) and P(distractor) reported separately, absolute probabilities beside the rates, uniform
over the candidate set stated.

**The statistic per level is declared here, before any knee is looked for.** Second-stage
re-placement (§3.6) is an interim look, so WO-STAT-001 §4's pooling rule governs it: a pooled null
may be read directly, a pooled positive must be confirmed on the second-stage points alone.

| Outcome | Reading | Licenses |
| --- | --- | --- |
| Monotone decay, no level separating sharply from its neighbours | Graded falloff; the window is a tunable | relax D4's note contract for content within the half-strength distance, and cost the saving |
| Strength high to level d then falling sharply within one or two levels | A cliff at d | notes carry everything older than d; **d is a design constant and is recorded as one** |
| Flat across the grid, above the never-mentioned control throughout | Distance is not the axis inside the window | note contract is not decided by distance; EXP-004's relevance arm inherits the question |
| Flat and **indistinguishable from never-mentioned** at every level | Instrument failure, not a result | fix before any reading; see §5's gate |
| **Still descending at 2048** | Our operating window sits **inside** the decay rather than past its knee | a finding about the window choice, worth more than the curve; the window itself goes to the Chief |
| Recurrent and attention layers diverge at matched depth | The recurrent path has its own horizon | EXP-002's question answered from the observational side; report per kind and do not pool |

## 3. Method

### 3.1 Axis: tokens, with turns recorded alongside

Ruled in WO-INTERP-002 §6.1 on the literature review. The mechanism integrates over tokens, the
mechanistic literature is unanimous on tokens, no precedent exists for a recurrent memory horizon
measured in turns, and our own per-message lengths span 21x to 27x depending on split, so "five
turns back" is not one distance. The source paper for this instrument (Gurnee et al.,
arXiv:2607.15495) measures persistence across **positions** — §4.1, Figure 28 panel (c),
autocorrelation of the top-1 lens token at position t against t+Δ versus a shuffled-position null.

**Recorded per point:** token distance (the axis), the turn index, and the containing message's
token length, so the note contract, which is stated in turns, can be read off the curve. The
review's honest gap is stated here rather than buried: no paper plots one dataset on both axes and
shows the conclusions diverge, so recording both supplies that demonstration at no cost.

### 3.2 Grid: seven levels, geometric ratio 2, 32 to 2048 tokens

Five levels identify a decay and cannot discriminate its shape; a p-parameter model needs at least
p levels and discrimination designs want p_max plus two to three (Box and Lucas 1959; Dette et al.
2010; de la Garza 1954). Spacing is geometric because exponential cannot be separated from
power-law except on a log grid spanning one to two decades (Clauset, Shalizi and Newman 2009).
32 / 64 / 128 / 256 / 512 / 1024 / 2048 spans about 1.8 decades and brackets Khandelwal et al.
(arXiv:1805.04623) 50-to-200 horizon with three levels.

**Which window, stated so the curve is not over-read.** The grid characterises the **deployment**
window, `max_seq_length: 2688` (`configs/models/qwen35-4b.yaml:15`,
`configs/agent_v2d_qwen35_4b.yaml:40`), because this experiment exists to decide D4's note
contract and that contract operates there. It does **not** characterise the architecture, whose
native `max_position_embeddings` is **262144**, a factor of 97.5. A curve still descending at 2048
is row five of §2.

### 3.3 Layers: EXP-001's kind-matched family, reported per kind and never pooled

The registry's six fractions resolve on this model to 5, 11, 16, 21, 27, 32, and **16 and 32 are
attention outputs** — verified from EXP-001's own conformance block, which records both the
convention (layer L is the residual after block L-1 and takes the kind of the block that wrote it)
and the kinds the instrument used. A full-attention block reaches the whole window regardless of
distance, so a decay curve read there confounds recurrent decay with attention's own distance
profile, which is this programme's central question measured through the wrong mechanism.

EXP-003 therefore inherits EXP-001's derived kind-matched family and reports the curve **per
kind**. The difference between kinds at matched depth is the recurrent contribution.

**Ruled: the family rule is generalised so layer 16 draws a partner (research division, 2026-09-05).**
As derived for EXP-001 the rule pairs each in-band *linear* layer with its nearest *attention*
neighbour, so 16 — itself an attention output — takes none. The consequence is that the curve has
kind contrast at 11/12, 20/21 and 27/28 **and none at 16**, which is the middle of the band and
the depth where Gurnee et al. put the workspace content. At exactly the layer most likely to carry
the concept, the curve would report an attention reading with nothing to read it against. That is
the confound §3.3 exists to remove, surviving at the one depth it matters most.

**Generalised rule:** each in-band layer draws its nearest neighbour **of the other kind**, in
either direction, smaller depth difference first. Checked against the existing family: 11 still
draws 12, 21 still draws 20, 27 still draws 28, and nothing else moves. 16 draws **17**. The
family becomes **5, 11, 12, 16, 17, 20, 21, 27, 28, 32**.

**The tie-break is not an index comparison, and its reason is recorded here because the next
person to extend the family inherits the rule and not the reason** (research division,
2026-09-05). Both 15 and 17 are linear and equidistant from 16, and the choice between them
silently fixes three asymmetries. Measured:

| partner | mean depth, attention / linear | attention is the deeper member | pairs bracketing an attention block |
| --- | --- | --- | --- |
| 15 | 19.0 / 18.5 | 3 of 4 | 3 of 4 |
| **17** | **19.0 / 19.0** | **2 of 4** | **2 of 4** |

Under 17 the two kind groups sit at the same mean depth, so a linear depth trend cancels in any
pooled kind contrast; under 15 a residual half-layer of depth difference lies in the same
direction as the kind difference. The third row is the one to think hardest about: each pair is
separated by one block, and under 15 three quarters of the family measures what an **attention**
block contributed while one quarter measures what a **linear** block did — and the recurrent
contribution this experiment wants is the second.

**Tie-break, stated generally:** on a tie, take the neighbour that leaves the two kind groups
closest in mean depth; on a remaining tie, the lower index. The resolved choice and its reason go
in the conformance block.

**Reporting, which is what makes the tie matter.** The **primary** reading is per pair, each at
its own depth, never pooled — under which 15 and 17 would be equivalent. The pooled per-kind
curves are reported as the **summary**, and it is the summary that the balance above protects.

This is a small code change to `kind_matched_layer_family` rather than a spec-only statement, and
it must be tested on the real block classes like the rest of the family derivation.

**Bound, so this is not over-read:** the decisive measurement is the output distribution and is
**layer-independent**. This confound touches the corroborating lens rows, not the verdict.

### 3.4 The construction (B7)

An implementer cannot invent this, so it is stated rather than implied.

**The concept is EXP-001's**, and deliberately: the random three-digit suffix of the third
invoice, revealed once in the directory-listing observation of a `ledger_reconcile` task on the
`jsweep` split. The probe stem is EXP-001's, the context forced to end mid-note at
`Reading invoice-2-`, and the candidate pair is EXP-001's, the true suffix's first token against
the already-read suffix's first token. Carrying all three makes the two experiments comparable
under R35 and reuses machinery that is built and tested. It also gives a consistency check:
**EXP-001's condition is this curve's limit past the window edge**, so a curve that has not
descended to EXP-001's null as it approaches the window edge is a discrepancy to explain.

**The intervening material is real task turns from the same family, not generator filler.** The
note contract this experiment exists to inform operates on real trajectories, so synthetic filler
would answer a different question — the Chief's point, adopted. Distance is therefore the token
count of **real trajectory** between the listing observation and the scored decision. The cost is
that intervening content differs between levels as well as distance; that is controlled by drawing
**many tasks per level** so content varies at random with respect to distance, and by recording
the realised content length per point.

**Hitting a geometric grid with material that does not come in powers of two.** Each level is a
target distance with a **geometric tolerance band**: level `d` admits realised distances in
`[d/√2, d·√2)`. On a ratio-2 grid these bands tile the axis without gap or overlap, so every point
belongs to exactly one level by construction. **The realised distance is recorded per point and
the curve is plotted against realised distance, not against the nominal level.** A level that
cannot be filled within its band is reported short, with its count, rather than filled by
stretching the band.

**The render is unwindowed up to the deployment window.** No observation stubbing: the placement
must stay in attention's reach, which is the entire point of measuring inside the window. Rendered
through `protocol.build_prompt` with the registry's template kwargs, capped at
`max_seq_length: 2688`. The grid's top level of 2048 therefore sits inside the window with room,
and every level is genuinely in-window.

**The candidate set, with the uniform baseline as a number.** The scored comparison is between two
tokens, but the absolute scale is read against the ten single-digit tokens a three-digit suffix can
begin with, so **uniform is 0.1**. That is the number EXP-001's verdict turned on and it is stated
here so it is not re-derived.

**The mismatched null is EXP-001's, matched on distance.** The same candidate pair scored against
**another concept's context at the same grid level**. Distance matching is not optional here as it
was in EXP-001: an unmatched null would differ from its treatment by distance as well as by
concept, which is the axis under test.

**The notes are stripped (B8, Chief).** In an unwindowed render every later note repeats the
filenames in its `pending:` list, so the suffix would appear again after its placement and the
leakage count would reject the whole cohort. `strip_pending` applies to the notes exactly as in
EXP-001 and EXP-002's B6, so the suffix enters the context **once**, in the listing observation,
and the count is what proves it rather than an assumption that it did.

**Fill per band, measured before the lift rather than promised (B8, Chief), and the grid does not
fit `jsweep` as declared.** Measured over all 60 `jsweep` `ledger_reconcile` tasks with the real
tokenizer and no model, the material available between the listing observation and the end of the
trajectory has median **1139** tokens and maximum **1153**:

| level | band lower edge | tasks with enough material |
| --- | --- | --- |
| 32 – 1024 | 22.6 – 724.1 | **60 of 60** |
| **2048** | **1448.2** | **0 of 60** |

So the top level is **unfillable from `jsweep` as declared**, and "reported short at run time"
would have been discovered after a lift. Difficulty is the lever: the same family gives median
available distance 999 at difficulty 0, 1140 at 1, 1316 at 2, **1506 at 3** and **1700 at 4**,
so difficulty 4 fills every band including 2048's `[1448, 2896)`.

**Ruled: the whole grid is drawn from one higher-difficulty split, not mixed.** Filling only the
top level from a longer source would make difficulty covary with distance at exactly one level,
which is a systematic content confound where §3.4 otherwise has only a random one. EXP-003
therefore declares **its own split at difficulty 4**, named in the spec with its seed, and
**R28's fingerprint disjointness test is run on it** — `jsweep`'s own declaration pins difficulty
1 (`JSPACE_SPLITS`), so a different difficulty is different content and inherits no disjointness
from it. The dry-run fill table is regenerated on the declared split and reported as `n of N`
before the lift.

**The leakage count applies to this construction.** Per point, counted in the rendered text as B6
counts and not inferred from flags: the target suffix appears **exactly once**, at its placement,
and nowhere the model could read it from later. A point failing that is dropped and tallied with
its reason, as `n of N` under R38. Without it the curve measures leakage rather than memory.

### 3.5 Controls, both required

- **Never-mentioned:** the identical probe with the concept never placed. This is the floor the
  curve is read against, not zero.
- **Mismatched (EXP-001's null):** the same candidate pair scored against another concept's
  context, which separates tracking the concept from tracking its tokens.

### 3.6 Two stages, and the second is an interim look

Stage one sweeps the seven-level grid. Stage two re-places levels around the knee once located.
**Because stage one is read before stage two is designed, stage two is an interim look**: a pooled
null may be read directly, a pooled positive must be confirmed on stage-two points alone
(WO-STAT-001 §4).

### 3.7 Replicates, unequal and pre-registered

Allocation is **unequal**: the extremes are near-certain outcomes and a proportion near 0 or 1
needs fewer points for the same interval, while the knee carries the shape discrimination. Equal
allocation is not what a discrimination design asks for. Pre-registered here as numbers rather
than as a promise, from WO-STAT-001's tables (exact sign test, two-sided, α = 0.05):

| level (tokens) | n | power vs 0.70 | vs 0.75 | vs 0.80 | role |
| --- | --- | --- | --- | --- | --- |
| 32 | 20 | 0.42 | 0.62 | 0.80 | anchor, near-certain |
| 64 | 42 | 0.74 | 0.92 | 0.99 | knee bracket |
| 128 | 54 | 0.84 | 0.97 | 1.00 | knee |
| 256 | 54 | 0.84 | 0.97 | 1.00 | knee |
| 512 | 42 | 0.74 | 0.92 | 0.99 | knee bracket |
| 1024 | 20 | 0.42 | 0.62 | 0.80 | anchor |
| 2048 | 20 | 0.42 | 0.62 | 0.80 | anchor, window edge |

**252 points in stage one.** The knee bracket follows Khandelwal's 50-to-200 horizon
(arXiv:1805.04623), which is the only prior estimate we have. The anchors are powered for a large
effect only, which is what an anchor is for; if an anchor's result is equivocal the second stage
re-places there rather than the spec pretending 20 points settle it. **54 is WO-STAT-001's stable
crossing rather than its first crossing:** power at this test is non-monotonic in n, 49 gives 0.81
and 50 gives 0.78, and 54 is the first n beyond which it does not dip.

## 4. Cost, from a measured rate and not from the lens rate

**The decisive measurement needs no Jacobians.** EXP-001 established that the output distribution
carries the verdict and the lens corroborates; its measured 133 s per probe point is the **lens**
rate, dominated by nine layers times sixteen contexts times three sources of finite differences.
The full grid is swept on the output measurement, which is prefill-bound. Lens readouts are taken
at **three levels only** — the two endpoints and the knee — for corroboration.

**The per-point prefill figure the budget rests on, and its source, are stated in the spec before
the lift is requested.** EXP-002's run supplies it. The budget is not written from the lens rate.

## 5. Prerequisites and gates

- **Instrument gate.** The near levels must separate from the never-mentioned control, or nothing
  is readable: a curve that is flat *and* at the control's level is §2 row four, not a finding.
- Preflight for `qwen35-4b` under R18a (passes).
- Code: placement at a token distance, the leakage count, the two controls, per-kind reporting,
  the per-level statistic; fake-only tests, **R31 at the tokenizer seam as well as the cache
  seam** — a fake tokenizer that accepts what the real chat template refuses is what stopped
  EXP-002 three seconds in.
- **R38 as amended: every count in the artifact and the reading states n of N**, with N the rows
  the source holds.
- One Director or Proxy lift, costed from §4's measured rate.

## 6. Acceptance

The instrument gate passing; the curve reported per kind with both controls and the decomposition;
the leakage tally recorded; the per-level statistic as declared in §2 before any knee is located;
the reading written against §2 by the Head of Interpretability and ratified by the Chief; and
EXP-004's arms parameterised from the curve rather than from a guess.

— Head of Interpretability
