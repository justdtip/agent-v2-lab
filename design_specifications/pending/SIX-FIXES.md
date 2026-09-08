# The six without the port: the fixture fixes, verified on the box

Patch: `design_specifications/pending/SIX-FIXES.patch`. **Apply after the six** — `WINDOW-RUN`,
`PILOT-PATH`, `LENS-IDENTITY`, `PILOT-ARGUMENTS`, `NATIVE-RESIDUALS`, `BAND-PAIRS` — and **not**
with `ARCH-VIEW-PORT`, which lands on its own with its own fixes. Five files, 62 insertions.

**Verified, on the box, nothing skipped for a window:**

```
1970 passed, 14 skipped, 1 warning in 101.95s
```

The Chief's split is built and it is green. `STACK-FIXES.patch` remains the version for the seven
including the port, and it was verified the same way at 1,976.

## What is in it

Only what the six need. Every fix the port needs — the three doubles' forwards, the per-block mask
translations, the traversal assertion — is deliberately **left out** and travels with the port,
because nothing in the six requires a callable model.

- `generation_prefix` on the fake chat specs in `test_patch` and `test_jlens` (PILOT-PATH).
- `probes.partner_tie_breaks` on the fake specs the sweep reads, and `probe_partner_tie_breaks` in
  the spec field set (BAND-PAIRS).
- `write_lens(..., identity=)` at the remaining caller, and the archive's `identity` entry in the
  file-set assertion (LENS-IDENTITY).
- `counts["residual_source"]` asserted (NATIVE-RESIDUALS).
- Two of my own tests, which were wrong rather than the code: the rotating-cache test handed KV
  caches to a hybrid's recurrent blocks, and the `(8, 7)` band case asserted a partner for the
  final layer, which `in_band_layers` reports rather than decides on.

## Why the split is right, in one line

`NATIVE-RESIDUALS` is what unblocks the Gemma regression fit, and it needs `PILOT-PATH` and nothing
else of the port. The port's own landing then has one job and one acceptance, and its acceptance
number is the residual-source disagreement `NATIVE-RESIDUALS` already measures.
