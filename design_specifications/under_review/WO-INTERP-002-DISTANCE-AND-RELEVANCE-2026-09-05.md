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

## 6. What I am asking you for

Not approval to run. Approval to **write EXP-003 as a full spec with pre-registered predictions**,
which then goes through the usual chain. Two things I would want your view on first: whether the
turn-versus-token axis should be fixed on turns (which matches the note contract) or tokens (which
matches the mechanism), and whether a five-level curve is the right resolution to start at.

— Head of Interpretability
