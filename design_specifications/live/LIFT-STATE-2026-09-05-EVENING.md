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

## Amendment (2026-09-05 late, Proxy relaying Daniel directly)

**A standing lift now covers test runs.** Test runs proceed without a per-run lift from Daniel or
the Proxy. Anything beyond that, full experiment ratification and lifts, still follows the
existing rules, so the per-run gate on experiments stands unchanged.

**My reading of the boundary, stated so it can be corrected rather than assumed.** The contrast
drawn is test runs against full experiments, so "test run" is read as a run whose purpose is
verification of a known answer, not discovery of a new one. That clears:

- the unit fixtures, which needed nothing anyway;
- **WP1's validation run** against the recorded 42-case table, whose whole purpose is to
  reproduce a result already in hand;
- dry runs and instrument fixtures that load the model only to check they reproduce.

It does **not** clear anything that produces a result nobody has: WP3's run, the source-position
inspection of the three candidate units, EXP-002's read-share pre-check and its 2 by 2, and all
of wave 2. Those are experiments and they keep the per-run gate.

The narrow reading is deliberate. The lesson of this evening was that the gate distinction
collapses in the direction everyone wants, and a run that discovers something is not made a test
by being called one. If the intended boundary is wider, it is one line from Daniel or the Proxy to
say so, and this note is the thing to correct.

## Operating rule until Daniel addresses it

The stricter reading governs. Anything that loads the model needs a per-run lift.

- **Needs a lift:** WP3's run (it hooks real GatedDeltaNet and attention blocks over 42 probe
  points and ten transcripts, so "needs no lens" is true of the lens and false of the model);
  WP1's validation run against the 42-case table; all of wave 2.
- **Does not need a lift:** every design; WP5's re-specification; WP7(b)'s stimulus set; the
  shared projector and clamp brief; writing WP1's module and its unit fixture.

Concurrency rule of record is unaffected: no model-loading run starts while another is alive.
Checked at the time of this note with `pgrep`; nothing was running.

## Standing constraints unchanged

The Proxy carries the Director's authority. Here the Proxy narrowed authority rather than granting
it, which needs no gate. The Director's direct instruction governs over any relay. I do not commit
or push. No implementing subagents.
