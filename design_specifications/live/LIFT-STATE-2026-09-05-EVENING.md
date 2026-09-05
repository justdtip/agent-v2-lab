# Lift state as of 2026-09-05 evening (Head of Interpretability's record)

Recorded by the Head of Interpretability from the Proxy's relay of the Director, 2026-09-05.
This note exists because two live records disagree and a run was queued against one of them.

## What is settled

**The veto is lifted.** Daniel confirmed it directly through the Proxy. `live/HOLD-2026-09-05.md`
is discharged. Ratification, implementation, commits and dispatch proceed.

## What is not settled

**The standing lift of R40c is not confirmed.** The Proxy, relaying Daniel directly: he did not
explicitly confirm the removal of the per-run gate, that part is unconfirmed by him directly, and
it is to be treated as open until he addresses it specifically.

`live/STANDING-LIFT-EXPERIMENTS-2026-09-05.md` and R40c in `pending/02-INTERFACE-AND-WIRING-MAP.md`
both read as settled and would authorise a model-loading run on their face. The Chief wrote that
record as verbatim intent from the Director in chat, so the disagreement is not about anyone's
good faith; it is that the record which would settle the question is itself the one in dispute.
Correcting or annotating it is the Chief's, not mine. The Chief was told at the time of this note.

## SETTLED (2026-09-05 late, the Director direct through the Proxy)

This supersedes both the narrow reading I wrote earlier and the Chief's operational definition.
Neither survives; the Director has answered the question they were arguing about.

**The Director's word: pursue research freely. The concurrency rule is the actual constraint, and
was always the intent behind it, not gatekeeping test or research runs generally.** Confirmed in
scope as a test run: a performance or hardware benchmark that loads the model but tests no
hypothesis about model behaviour.

**Operating rule.** Research runs proceed. Test runs, calibrations, instrument pre-checks,
validation runs, benchmarks and the experiments themselves do not wait on a per-run lift. My
earlier distinction, that a run producing a number nobody has is not a test run, is withdrawn: it
was the right caution while the Director's intent was unknown and it is wrong now that it is
known. The record keeps it because the reasoning was checkable and it should be visible that it
was overtaken rather than quietly dropped.

**What still binds, and why each is not a lift question.**

- **Concurrency.** One model-loading run at a time on this machine, checked immediately before
  launch with `pgrep` on the probe and training entry points, refusing if one is alive. This is
  now the operative constraint rather than one rule among several.
- **Training lifts.** Not covered by any of this and keeping their own gate.
- **Pre-registered readings and WO-STAT-001 power.** These are the programme's own scientific
  standard, not an authorisation step: they are what makes a result interpretable, not what makes
  it permitted. A run that starts without a reading is not blocked, it is uninterpretable. Nothing
  in the lift touches them and I would keep them exactly as they are.
- **Results to the Director**, and visualisations as a deliverable.

**Cleared for my queue.** WP3's transport statistics, WP1's validation run, the source-position
inspection under R3f, EXP-002's read-share pre-check, and wave 2, subject only to concurrency and
to their slices being built.

## Standing constraints unchanged

The Proxy carries the Director's authority. Here the Proxy narrowed authority rather than granting
it, which needs no gate. The Director's direct instruction governs over any relay. I do not commit
or push. No implementing subagents.
