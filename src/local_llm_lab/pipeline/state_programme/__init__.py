"""The state programme's run script, as ordered in `STATE-PROGRAMME-RUN-ORDER-2026-09-10.md`.

Built and tested on fixtures on the laptop; run on the device. Nothing in this package loads a
model. The pieces, in the order the script uses them:

- :mod:`family` — the existence task family, in matched pairs, and the reliability instrument.
- :mod:`diagnostics` — the ten diagnostics, fixed functions of a transcript, never re-scored.
- :mod:`tolerances` — the derivation table: bootstrap lower bounds, the drop rule, ``n``, the
  budget rule that writes both numbers.
- :mod:`seal` — the pre-registration and its two refusals.
- :mod:`run` — stages 0–4 with a policy seam, so a scripted policy drives the fixture run and a
  model drives the device run through the same code.
- :mod:`record` — the record directory, and a README written from its files and nothing else.
"""
