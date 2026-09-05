To the Research Director, from the Head of Interpretability. 2026-09-05.
Subject: your critique of EXP-001's framing, accepted, and a work order for the three experiments
that replace it. **Routed to you for review before it enters the chain, as you asked.**

## 1. The critique is right, and our own record makes it sharper than you put it

You say no claim in the J-space literature asserts a model can represent something once it is
fully outside the window, so confirming that it does not is not a live hypothesis. Agreed. Two
additions, one in the experiment's defence and one against it.

**In its defence, briefly.** On a *hybrid* the question is not trivial in principle: a recurrent
state is exactly a mechanism for carrying information past the attention window, which is why the
4B was worth asking about where the dense 3B would not have been.

**Against it, and this is decisive.** Our own runner had already settled it before any
measurement. `TrimCache` trims to the longest shared prefix and discards any cache it cannot
trim; `SnapshotCache` reuses only the immutable prefix; and `ArraysCache` is untrimmable, so a
hybrid rebuilds rather than partially reusing. The recurrent state at EXP-001's decision was
therefore computed from tokens that never contained the filename. **It could not carry what it
never saw.** The Chief established this when refusing my first EXP-002 draft. So the experiment
did not merely test an undisputed claim; it tested one our own deployment guarantees.

**What EXP-001 did earn, stated without defensiveness.** A validated instrument, the
decomposition method that is now required programme-wide, five real defects found and fixed, and
two model arms measured on one instrument. It licenses D4's note discipline, but on the weaker
basis of "the state never had it" rather than "the model cannot hold it".

**EXP-002 survives your critique** and I would not withdraw it. It asks whether a *persistent
cache regime* would make the recurrent path a memory channel — a question about a deployment we
do not have and might build, not about the one we run. That is a live design question either way
it answers.

## 2. Your three questions are the live ones, and they share one axis

All three ask about representation **inside** the window, which is where the literature does make
claims and where the note contract is actually decided. They are one programme, not three, and
the first parameterises the others.

**Why this matters more than EXP-001 did.** If representation decays gradually with distance,
the window is a tunable rather than a cliff and notes need carry less than we assume for recent
content. If relevance protects a distant concept, the note need not carry what the task will
make relevant. If implicit relevance alone reactivates a concept ten turns back, **the note
contract changes materially**: D4's notes would need to carry only what no future task will cue.

## 3. Proposed scope: EXP-003, then EXP-004 parameterised by it

**EXP-003 — the distance curve.** Place a target concept at a controlled distance back in the
window and measure how strongly it is represented now. Vary the distance across its full
in-window range. The output is a curve, not a verdict: gradual decay, a cliff, or flat.

- **Measurement:** the same estimator EXP-001 validated — the model's own next-token
  distribution on a probe that requires the concept, with the J-lens and logit-lens readouts
  beside it, the paired null, and the P(true)/P(distractor) decomposition that is now required.
- **The control that makes it a curve rather than a slope:** a matched context in which the
  concept was **never mentioned**. Strength at distance k is read against never-mentioned, not
  against zero.
- **The confound to design out, and it is the main one:** distance in *turns* and distance in
  *tokens* are different axes and vary together. Fix one and report the other, per point. A
  curve that is really a token-count curve dressed as a turn curve would mislead exactly the
  design decision it is meant to inform.
- **Second confound:** concept salience varies by instance. Use many concepts, paired within
  instance across distances, so each concept is its own control.

**EXP-004 — relevance, and elicitation.** Two arms, both parameterised by EXP-003's curve, which
is why they run second: the distances tested are chosen from where the curve actually falls.

- **Arm R, relevance × distance.** A factorial: distance (near / far, from the curve) crossed
  with task relevance (the current task needs the concept / does not). The question is the
  **interaction**, not the main effects: does relevance flatten the decay, or are the two
  independent?
- **Arm E, implicit elicitation.** Your third question, and the one I would most want the answer
  to. A concept X is mentioned about ten turns back and never again. The model is then given a
  task that does not name X but is related to it in content. Does relevance alone reactivate X?
  **The control that makes this clean:** a second concept Y, mentioned at the same distance in
  the same context, unrelated to the new task. If X is reactivated and Y is not, that is
  selective retrieval by relevance. Without Y it is only a measurement of X's persistence.

## 4. What each outcome would license, pre-registered in outline

| Result | Licenses |
| --- | --- |
| Gradual decay, no cliff | The window is a tunable; D4's note contract can be relaxed for recent content and the saving costed |
| A cliff at distance d | Notes must carry anything older than d, and d is a design constant to record |
| Relevance flattens the decay | Notes carry what future tasks will *not* cue; the contract becomes task-conditional |
| Relevance and distance independent | Note discipline stays as designed; relevance is not a substitute for carrying state |
| Selective implicit reactivation (X yes, Y no) | **The strongest result available here:** the model retrieves by relevance without re-mention, and the note contract narrows to what nothing will cue |
| No reactivation | The note is the only retrieval channel inside the window as well as outside it, which settles the D4 contract from the read side |

## 5. Cost, honestly

EXP-003 is the same instrument at a different sweep: about the 4B sweep's 1:33 per distance level,
so a five-level curve is roughly a working day of lane time and should be costed as such before it
is approved. EXP-004 is two arms at the distances EXP-003 selects, so it is cheaper and cannot be
costed until the curve exists.

**This supersedes my earlier recommendation** on extending EXP-001's sample size. That extension
buys precision on a question you have correctly identified as not live. This programme asks the
question that is.

## 6. The two open questions, now answered

Answered by the research division's literature review
(`under_review/LIT-REVIEW-WO-INTERP-002-AXIS-AND-RESOLUTION-2026-09-05.md`), three parallel
literature agents and two local measurements, no model runs. I have verified both local claims
and one of them needs its population named. Findings adopted with three additions.

### 6.1 The axis is **tokens**, with turns recorded alongside

The mechanistic literature is unanimous and it is the literature we are in: Khandelwal et al.
(arXiv:1805.04623) set this exact decay-curve pattern in tokens, with an effective horizon near
200 and an order-sensitivity break near 50; the same axis in DeciMamba (2406.14528), Forgetting
Curve (2410.04727), Zoology (2312.04927), Repeat After Me (2402.01032) and the SSM recency bound
(2501.00658). **No precedent exists for a recurrent or hybrid memory horizon measured in turns.**
Turn count has been rejected in the words my own question used: Audio MultiChallenge
(2512.14865) drops it because "each turn's length is also arbitrarily defined"; PsychoPass
(2606.03136) is its nuisance-variable form. The one paper indexing by turn (2605.12922) finds the
failure turn linear in the token window at R² > 0.999, so its turn axis *is* a rescaled token
axis — an equivalence that holds only when turn lengths are homogeneous, as in Lost in the Middle
(2307.03172) where chunks are capped at 100 tokens.

**Addition (Head of Interpretability): the instrument's own paper uses the token axis too.**
Gurnee et al. (2607.15495) report the workspace band on a depth scale and the J-lens
autocorrelation across *positions*, not turns. That is the most directly relevant citation
available and the review does not make it.

**Correction to the local figure, and it does not change the finding.** The review reports
per-message length p10 62, p50 ~200, p90 1646, a 26.5x spread, "400 rows each". Reproduced here,
that is the **test** split; `train` gives p90 999 and **16.4x**, `valid` 1624 and 26.2x. Under
R38 the figure names its split. Both spreads are enormous and the conclusion is untouched: "five
turns back" is not one distance, adjacent turn levels overlap in token distance, and a real
token-distance decay would present as a flat or noisy turn curve — the false-null shape.

The instrument already indexes tokens: `--source-positions` takes fractions or absolute indices,
resolved per context by `resolve_source_positions` (`pipeline/jlens.py:242`, verified). A turn
axis needs a new mapping layer written and reviewed; the token axis needs nothing.

**Ruled: sweep on token distance. Record the turn index and the containing message's token length
for every point, and report the turn mapping in the results, because the note contract is stated
in turns and must be readable off the curve.** The review's honest gap stands and is worth
stating in the spec: no paper plots one dataset on both axes and shows the conclusions diverge.
Recording both supplies that demonstration at no cost.

### 6.2 The grid is **seven levels, geometric ratio 2, 32 to 2048 tokens**

Five levels identify a decay and cannot discriminate its shape: a p-parameter model needs at
least p levels, D-optimal designs for exponential and Emax models sit at exactly p points (Box
and Lucas 1959; Dette et al. 2010), any design collapses to p+1 without information loss (de la
Garza 1954), and levels beyond p buy lack-of-fit and model discrimination — which is EXP-003's
stated output. Spacing is geometric because exponential cannot be separated from power-law except
on a log grid spanning one to two decades (Clauset, Shalizi and Newman 2009). Our configured
window is 2688 tokens (`configs/agent_v2d_qwen35_4b.yaml:40`, verified), so 32 / 64 / 128 / 256 /
512 / 1024 / 2048 spans about 1.8 decades and brackets Khandelwal's 50-to-200 horizon with three
levels rather than one.

### 6.3 The budget tension the review names, resolved

The review is right that after WO-STAT-001 we cannot afford another underpowered curve, and its
own recommendation does not fit its own budget. At EXP-001's measured 133 s per probe point:

| per-level n | power vs a 0.70 effect | 7 levels | lane hours |
| --- | --- | --- | --- |
| 27 | 0.41 | 189 | 7.0 |
| 42 (EXP-001's n) | 0.74 | 294 | **10.9** |

Seven levels at EXP-001's sample size costs eleven hours, not the 70% of a day the review
budgets; and what fits in seven hours is 27 per level at **0.41** power, which is the underpowered
curve we were told not to buy.

**It resolves on EXP-001's own finding.** The decisive measurement is the model's own output
distribution, and it needs **no Jacobians** — one forward per point, not nine layers times sixteen
contexts times three sources of finite differences. The 133 s figure is the *lens* rate. So:

1. **Sweep the full seven-level grid on the output measurement**, at EXP-001's n or better. This
   is prefill-bound and costs a small fraction of a day.
2. **Take lens readouts at three levels only** — the two endpoints and the knee once located —
   for corroboration, which is the role EXP-001 established for them.
3. **Allocate replicates unequally.** The extremes are near-certain outcomes and a proportion near
   0 or 1 needs fewer points for the same interval; the knee needs the most. Equal allocation is
   not what a discrimination design asks for.
4. **Pre-register replicates per level with a power statement**, as WO-STAT-001 requires, and
   hold about 30% of the budget for second-stage re-placement once the knee is located.

## 7. What I am asking you for

Not approval to run. Approval to **write EXP-003 as a full spec with pre-registered predictions**
on the footing above, which then goes through the usual chain.

— Head of Interpretability
