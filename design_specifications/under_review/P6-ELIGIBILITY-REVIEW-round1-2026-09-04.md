# P6 provenance and eligibility (issue #22, three rounds), review round 1, ratified (2026-09-04 21:30)

Reviewer: Claude (Chief). Evidence: issue #26 and the Director's ratification on #25. Diff not
re-read; the round-2 reviewer's pre-registered prediction (5 cases, named ledger tasks, decision
steps 7/7/7/7/6, all counterfactual notes transcript-sourced), reproduced by the implementation
and pinned against the real files by a test, is adopted as the verification.

## Verdict: APPROVED TO COMMIT with one small condition. No fourth review round.

The Deputy's judgement not to order a fourth round is upheld: a pre-registered prediction
confirmed by implementation and pinned by a real-files test is stronger evidence than a
post-hoc read.

## Ratified

- The eligibility recomputation as the R22 extension; the verdict filter as the reading of
  SPEC-004 §5's case universe; fail-closed version binding per R23.

## Condition (K1, ruling R24)

The remaining HEAD seam in flip scoring (`_is_flip`, `patch.py:516-521`) is acceptable only
when per case the dropped value and decision step are identical under v1 and HEAD. The tool
already computes both; record `scoring_version_stable` per case, include only stable cases in
the headline, list unstable ones separately. For the five selected cases this is a recorded
fact, not a behaviour change. The Deputy verifies directly; no new round.

## Noted for the experiment's interpretation

- **Five cases.** The universe is small because `aggregate_report` fails under both B and C,
  so only `ledger_reconcile` supplies pass/fail pairs with an empirically passing note. The
  heat map will be indicative, not decisive; intervals over five tasks will be wide. Report it
  as such.
- **Bound follow-up (not a condition):** a secondary P6 condition for the `aggregate_report`
  failures using the generator-v4 note as the counterfactual (no passing run exists). It tests
  a designed-correct note rather than an empirically passing one and must be labelled so; it
  is the only way to put the family that matters most under the patching lens.

## On commit

The P6 command is final with `--data-seed 20260902 --generator-version 1`; the lift request
follows in the evidence format.
