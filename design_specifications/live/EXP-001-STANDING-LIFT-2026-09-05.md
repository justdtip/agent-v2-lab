# EXP-001 standing lift, recorded by the Chief on the Director's verification, 2026-09-05

## What the Director verified

The Deputy reported (cross-session message, evening of 2026-09-05) that the Director had given
it standing authority to run EXP-001 once the preflight artifacts are rebuilt. The Chief did not
act on the peer's report. The Director then verified it directly to the Chief in the Chief's
own session: "I record it as I verified it here to you." This file is that record.

## Scope, as the Chief reads it

1. **Preflight regeneration** for both registered models (`qwen25-coder-3b`, `qwen35-4b`) under
   the calibrated schema-3 preflight slice (#51). These runs are the stated precondition of the
   authority, so they are covered by it. Under R37 they land in the same commit as the
   calibration code; the Chief's gate on that code stands and is not waived by this lift.
2. **EXP-001's runs** (the 4B base sweep and the 3B comparator, and the arms EXP-001 §5 names)
   under the Deputy's authority, on the single execution lane, one run at a time.

Not covered: training of any kind (B4 attempt 4 keeps its own per-run lift), the R32 stage-2
memory and step-time probe (its own lift), and the P6 secondary condition (its own lift).

## Conditions that still gate the EXP-001 runs

- C1: the sweep's null is read from a cache rather than recomputed (halves the run), landed and
  gated.
- C2: the Head of Interpretability has answered sum versus per-sample mean for the `future` and
  `all` readouts, recorded in EXP-001 §3.2 and the conformance block, before the first run.
- The preflight artifacts on disk are schema 3 for the model being run, and the probe CLI's own
  preflight gate passes without `--skip-preflight-check`.
- Every run writes `run.log`, `events.jsonl` and provenance (R26), and cites its commit.

If the Director's intent was narrower than reading 1 above (the regeneration runs excluded),
the Director says so and reading 1 is struck; nothing else in this file changes.

## Amendment, later on 2026-09-05: the Director's standing instruction for the night

Given by the Director to the Chief directly, in the Chief's session:

> "Get through as much of the interpretability as you can. Once conditions are green for
> EXP-001, send a message to interpretability and have them run it. Lift granted. Basically,
> all interpretability lifts granted."

Effect, as the Chief records it:

1. **Every interpretability lift for the night of 2026-09-05 is granted**: the preflight
   regeneration for both registered models (the precondition), EXP-001's runs (4B base sweep,
   3B comparator, and the arms §5 names), the P6 secondary condition on the aggregate-report
   family, and any further probe run the Chief judges ready. Runs are one at a time on the
   single execution lane, each with R26 logging and provenance, each citing its commit.
2. **Not covered**: training of any kind, and the R32 stage-2 memory and step-time probe
   (training-side; its own lift stands as before).
3. **Who runs EXP-001**: the Head of Interpretability, on the Chief's message that the
   conditions are green (C1 landed and gated; C2's sentence and conformance-block naming in;
   schema-3 preflight artifacts on disk for the model being run). The Deputy keeps the lane
   and the board; the Chief keeps the gate on code.

## Direct assent in the Head of Interpretability's session, 2026-09-05

The Chief's amendment above reached this session as a cross-session message, which is a peer's
report and not the Director's word to me. I told the Chief I would take EXP-001 on their green,
because the Director had asked me directly earlier in my own session to run it, and would leave
the wider scope until the Director confirmed it here. The Director then did:

> "Standing lift assent granted."

Recorded so the authority for each arm is traceable to the session it was given in. On this
basis the wider scope of the amendment, the preflight regeneration for both models and the
further probe runs it names, is authorised in this session as well. Training of any kind and the
R32 stage-2 probe remain outside it, as the amendment states.

What the assent does **not** move: the Chief's gate on code is not waived by any lift. The
preflight artifacts must not be regenerated until the calibration slice (#51) passes that gate,
because artifacts stamped from uncommitted code carry no usable provenance. As of this entry the
slice is still at the gate, both artifacts on disk are schema 2, and `pipeline/preflight.py`
carries 374 uncommitted lines. Nothing runs until the Chief sends the green.

— Head of Interpretability
