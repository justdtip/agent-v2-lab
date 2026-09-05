# EXP-002: Is the Qwen3.5-4B's recurrent state a usable memory channel under a persistent cache?

> Read first: `01-IMPLEMENTER-BRIEFING.md`, `02-INTERFACE-AND-WIRING-MAP.md` (§7 R18a/b, R26,
> R31, R34, R35), `EXP-001-JSPACE-QWEN35-4B.md` and its readings under `under_review/`.
> Owner and author: Head of Interpretability. Status: pending. Requested by the Director,
> 2026-09-05; premise reformulated on the Chief's correction the same day.

## 1. The premise, corrected

**This spec's first draft was wrong and the correction matters more than the method.** It argued
that EXP-001's null might be a limit of the lens, since a residual-stream lens cannot observe a
recurrent state. That is true in general and false for the condition EXP-001 measured, for a
reason that needs no interpretability at all: **the state cannot carry what it never saw.**

In the stripped condition the prompt contains the suffix nowhere — the leakage guard drops any
case where it does (`jspace_sweep.py:223`). And the deployed runner never carries a recurrent
state across the eviction that hides the listing: `TrimCache` trims to the longest shared prefix
and **discards any cache it cannot trim** (`runner.py:86-146`), `SnapshotCache` reuses only the
verified immutable prefix (`:149`), and `ArraysCache` defines neither `is_trimmable` nor `trim`,
so it inherits `_BaseCache.is_trimmable() == False` and a trimming strategy on a hybrid
degenerates to a rebuild. So the recurrent state at EXP-001's decision was computed from tokens
that never included the filename. Its null is a fact about the model's information, not about the
lens's reach.

The first draft's with-list swap would have injected a state computed from a context whose notes
*do* carry the filename. It would almost certainly have raised P(true), and that would have been
a capability fact about the DeltaNet path — can a state that saw X deliver X later — presented as
a fact about EXP-001's condition. Outcome 1 would have licensed reconsidering `keep_last` on a
misattribution.

**The question that survives is better, and is the Director's real one.** The recurrent path can
only be a memory channel under a regime where the state persists while the text it saw is gone.
That regime is not hypothetical: it is available *because* `ArraysCache` is untrimmable. A
persistent-cache inference regime would retain in the recurrent state every observation the
window later hides, while the attention KV for those hidden spans can be masked or evicted. So
the experiment is not a swap between contexts but a **within-run dissociation** between the two
paths under one persistent cache. Either answer changes a design decision, which the first draft
could not claim.

## 2. Pre-registered predictions

**Decisive measurement:** the model's own next-token distribution over the suffix's first token,
per arm, paired per probe point. EXP-001 §3.3's decomposition applies in full — report P(true
suffix) and P(already-read suffix) separately, since an ordering statistic cannot say which
probability moved — and absolute probabilities are reported alongside, uniform over ten digits
being 0.1.

| Outcome | Reading | Licenses |
| --- | --- | --- |
| Arm A raises P(true) materially above arm C, above the foreign-state control, and the identity gate passes | The recurrent path **is** a memory channel: state outlives the text that wrote it and remains readable | a persistent-cache regime is a real alternative to state-carrying notes for the D4 line; specify and cost it before D4's note contract is fixed |
| Arm A indistinguishable from arm C, with arm B above both | The note is the only channel on **both** paths; EXP-001's World A confirmed causally rather than observationally | D4 unchanged; note discipline settled on the read and write sides; close the question |
| Identity gate fails, or arm B fails to raise P(true) | The instrument cannot deliver a filename the model can plainly see; **no conclusion** | fix before any reading |
| Arm A and the foreign-state control rise together | A generic perturbation of the recurrent state, not retrieval | no design change; report as such |

Arm B is the gate on the instrument as well as an upper bound: if a run whose attention can still
see the listing does not put the suffix at the top, nothing about arms A or C is readable.

## 3. Method

1. **Probe points.** EXP-001's family unchanged, so the two are comparable under R35: split
   `jsweep`, `ledger_reconcile`, step 3, the first read after the listing leaves the window,
   leakage guard, coinciding first tokens skipped. Expect ~42 points.
2. **One persistent cache per point, over unstripped observations and stripped notes (B6,
   Chief, 2026-09-05).** Prefill every turn of the history through a single cache built as the
   model builds it, `ArraysCache(size=2)` for the 24 recurrent blocks and `KVCache` for the 8
   attention blocks, truncated at the same decision, mid-note at `Reading invoice-2-`.
   **Observations are unstripped, so the listing is present; notes have `strip_pending` applied,
   as in EXP-001.** This is load-bearing rather than tidy: an unstripped note still carries the
   filename in its `pending:` list, and a note is an assistant message that the observation
   window never hides, so under arm A attention would read the filename off a visible note with
   the listing's span masked and arm A would rise for a reason that has nothing to do with the
   recurrent state. With notes stripped, **the filename enters the model exactly once, through
   the listing observation**, which is the span arm A masks.
3. **Arms, each scored at that decision on the same points.**
   - **A, recurrent only**: mask the attention KV over the hidden-observation spans, so attention
     is blind to exactly what the window hides while the recurrent state is untouched. With B6's
     stripped notes the filename's only route into attention is the masked listing span, so a
     rise here is attributable to the recurrent state alone. This is the experiment.
     **The masked spans are the observation messages `keep_last = 2` would hide**, located from
     message boundaries in the rendered token sequence and recorded per point in the conformance
     block as start, end and token count, so a reader can verify the mask covered exactly the
     hidden text and nothing else.
   - **B, everything**: no masking. Attention can see the listing. Upper bound and instrument
     gate. Notes are stripped here too, so B and A differ by the mask alone.
   - **C, EXP-001's condition**: a fresh prefill of the windowed, stripped text with no
     persistent cache. The baseline the other arms are read against, and the arm that reproduces
     EXP-001 — it must match that artifact's decisive row, or the reading stops.
   - **Control, foreign persistent state**: another point's persistent cache under A's masking.
     Separates retrieval from generic perturbation.
4. **Identity gate, a gate and not a control.** Capture a run's own cache entries and re-inject
   them into the same run unmasked; the output distribution must be unchanged. **Tolerance: the
   maximum absolute difference over the scored candidates' probabilities must not exceed 1e-6**,
   and **the observed maximum is recorded whether it passes or fails**, so the threshold can be
   re-calibrated from evidence rather than argument if it proves too tight on this hardware.
   Recorded in the artifact before any arm. If it fails, no arm is readable.
5. **Statistic.** Per point and per arm, P(true) and P(already-read), each paired against arm C
   over the same points, exact two-sided sign test, with absolute probabilities reported.
   Multiplicity: Holm across the arms.

## 4. Logging and artifacts (R26, R34, R35)

New CLI `agent-v2-state-swap` with `--model`, `--policy`, `--split`, `--count`, `--data-seed`,
`--generator-version`, `--arms`, `--output`; no `--skip-preflight-check` in normal use. `RunLog`
with EXP-001's identity block and one progress line per probe point. The artifact records the
identity gate first, then per arm and per point the two probabilities and their pairing. The R34
conformance block names which cache class was used per block kind, which spans were masked and
how they were located, and the position at which each arm was scored. The R35 comparability block
is populated as EXP-001's is, including `fp32_manual_vs_native`.

## 5. Prerequisites and gates

- Code: a per-layer position mask handed to the attention blocks through `ArchitectureView`,
  which already builds per-kind masks and runs blocks with a per-block cache entry; the recurrent
  blocks take no mask, which is the point. Masking a middle span leaves later positions'
  indices unchanged, so RoPE is untouched. The four arms, the identity gate, fake-only tests.
- **R31 applies to the cache seam specifically.** `ArraysCache` and `KVCache` are library classes
  and a stub will not expose their behaviour; that a plain object looks trimmable while
  `ArraysCache` inherits `is_trimmable() == False` is the kind of fact only the real class shows.
  This is the same trap that produced the hybrid-period defect.
- Preflight for `qwen35-4b` passes under R18a (it does).
- One Director lift. No Jacobians, so the cost is prefills and short generations rather than tail
  passes; expect well under the 4B sweep's 1:33. The implementer states a measured rate after two
  points, as EXP-001 did.

## 6. Acceptance

The identity gate passing, arm B raising P(true), arm C reproducing EXP-001's decisive row, the
four arms reported against §2, the reading written by the Head of Interpretability and ratified
by the Chief, and the decision memo's EXP-001 entry amended to say whether World A on the hybrid
is observational or causal — and, if arm A rises, a costed persistent-cache option filed before
D4's note contract is fixed.

— Head of Interpretability
