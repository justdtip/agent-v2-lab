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
