# Issue 86 — the family generalises, and the tie-break stops pretending to be a rule

Patch: `design_specifications/pending/BAND-PAIRS-FROM-THE-FUNCTION.patch`, against `41fc725`.
Five files, 262 insertions.

**Not yet green, and I am not reporting it as such.** `test_jlens.py` imports `mlx.core` at module
scope, so every test in it is skipped while my evaluation window is open, and the collector says so
in its own terminal summary. Everything the box allows was run: **869 passed, 1,098 skipped,
exit 0**, which includes `test_models.py` on the new registry field and the repository rules. The
two acceptance derivations were checked another way, below. `test_jlens.py` and
`test_jspace_sweep.py` run at the first gap and I will report the number before this lands.

## Both acceptance derivations, verified without MLX

`local_llm_lab.pipeline.jlens` imports no MLX and the derivation is pure Python over layer
indices, so the claims can be checked beside another seat's window even when the test file cannot.
`design_specifications/pending/BAND-PAIRS-derivation-check.py` does it, nine assertions, all pass,
`mlx` absent from `sys.modules` at the end:

- **EXP-003's family**: `(5, 11, 12, 16, 17, 20, 21, 27, 28, 32)`, pairs `{11: 12, 16: 17, 21: 20,
  27: 28}`, no unrecorded tie.
- **The band's five pairs, from the same function**: `((12, 13), (16, 17), (19, 20), (23, 24),
  (27, 28))`, identical to `probes.live_lens_pairs`.
- **Without the ruling** the same call returns 15 for layer 16 — the old wrong answer — and reports
  it as a tie it broke without one.

## What changed

**The no-partner clause is gone.** Every in-band layer takes a partner of the opposite kind, in
either direction. The old clause was true of EXP-001 §3.5's fractions, where the in-band attention
layers happened to need none, and false of everything derived after EXP-003 generalised it at 19:38
on 2026-09-05. Under the band four of five primaries are attention-written, so the old code left
16, 20, 24 and 28 unpaired and could not produce the band at all.

**The tie-break is data with a source, not code.** `kind_matched_layer_family` takes `tie_breaks`
and consults it when two candidates sit at equal distance. `configs/models/qwen35-4b.yaml` records
one entry, `partner_tie_breaks: {16: 17}`, with EXP-003's file, line range and timestamp in the
comment beside it, reaching the function through a declaration-only `ModelSpec` field in the shape
`live_lens_pairs` already uses. Both call sites pass it.

**An unrecorded tie is reported, not hidden.** It resolves to the lower index — a sweep must not
die of a tie — and the layer and both candidates go into a new `unrecorded_ties` field, into
`as_dict`, and into the `reason` string. So an artifact carrying a fallback says it is one.

**`LAYER_FAMILY_RULE` is the generalised rule**, since R34 quotes it into every artifact, and a
test pins the string in both directions: the new clauses present, `"lower index on a tie"` and
`"takes no partner"` absent.

## Two findings you should have before ruling

**The band needs no tie-break at all, and I nearly recorded three that nothing ruled on.** My first
draft wrote `{16: 17, 20: 19, 24: 23, 28: 27}` into the registry, reading the last three off R41b's
band. They are not needed: give the function the band's five *recurrent* members and each has a
unique nearest attention layer one step away, so all five pairs come out with no tie. The three
extra entries would have been answers to ties nobody ruled on, written down as rulings — the exact
failure this field exists to prevent, committed while building the field to prevent it. The
registry records one entry, the one EXP-003 measured.

**This bears directly on issue 80.** The band is now derivable from the five recurrent members plus
the block-kind rule, so 80 does not need a second band literal; `live_lens_pairs` can be *checked*
against this function instead of duplicated, which is what my comment on 80 argued for from the
other direction. Worth deciding before 80 is implemented.

**And a mismatch worth naming.** R40b's set `(13, 16, 20, 24, 28)` does not reproduce the band
through this function, because `in_band_layers` puts the ceiling at `round(32 × 5/6) = 27` and 28
falls outside it. EXP-001 §2's in-band range and the band are different sets. Nothing here depends
on resolving that, but a reader who assumes they are the same will get four pairs and not know why.

## What is unaffected

**EXP-001's recorded family stands**, and the issue was right about why: under the rule it was
recorded with, layer 16 took no partner, so the tie never arose. Only families derived after the
generalisation are exposed, which is EXP-003 and the band.
