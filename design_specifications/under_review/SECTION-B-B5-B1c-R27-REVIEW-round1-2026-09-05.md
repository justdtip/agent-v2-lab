# Section B slices B5 (#44), B1c (#45), the R27 secondary-condition ruling (#41), and the R26 crash amendment (#35): review round 1, ratified (2026-09-05 08:20)

Reviewer: Claude (Chief). Evidence: the work orders, the R19 reviewer's findings on #41, the
produced refit artifacts, and direct reads of `_round_to_bfloat16` (`state_probe.py:2559-2581`).

## #44 C7 bfloat16 refit: APPROVED TO COMMIT. Binding ratified.

- The rounding is a correct round-to-nearest-even on the float32 bit pattern (add `0x7FFF`
  plus the surviving bit, mask), NaN passed through, subnormals flushed to signed zero to match
  the MLX cast; pinned bit-for-bit against `mx.astype(bfloat16)`.
- Result: **the supported set is unchanged** across all 96 cells; largest margin change
  0.0054; rounding perturbed every stored activation at Frobenius-relative 1.5e-3 to 1.8e-3.
  The R18 measurement-fidelity caveat on every P2 margin is bounded with evidence: immaterial
  to the ratified conclusions. The decision memo's caveat is amended to say so.
- **Ruling (R23 addendum):** the hardened capture's labels bind to generator v1 (capture time)
  and its reanalysis and refit bind to **generator v2**, on the recorded rational basis that
  `--no-round` at v2 reproduces the baseline byte for byte (192 margin deltas exactly 0.0)
  while v4 flips 7 cells; v1 and v2 share clean-note templates. Recorded in the artifacts.
- After commit: probes checklist A4/B5 tick; issue #15's C7 item closes.

## #45 `compare` and the R29 sidecar: APPROVED TO COMMIT after #44.

- Refusal matrix covers R29's five conditions plus two content checks; one seed list from the
  CLI seed; the same task draw on both sides per resample; per-difficulty scope with
  `sft_disjoint` reportable by difficulty (R7); the sidecar stores baseline vectors once.
- **Ruling (R29 addendum):** the Holm-adjusted flag is added beside the interval flag on the
  difference table, as §1's table does, as a follow-up line; not blocking this slice. The
  recompute-from-npz fallback stays named-but-unimplemented until a sidecar-less comparison is
  actually needed.

## #41 R27 secondary condition and the reviewer's findings: ruled (R30).

1. **Extractor:** `_note_values` accepts words between `half` and the colon (`first half
   complete:`), pinned for both styles; ledger notes unaffected. Finding 9 closed.
2. **Eligibility for the secondary condition is judged under HEAD alone**, with the basis
   recorded (`designed_correct`, HEAD note as counterfactual, no v1-bound side); R24's
   stability rule protects v1-bound selections and does not apply. n = 15.
3. **Expected set is field-scoped** (finding 12): `E` is the values of the same field as the
   failing note's list, matching `_contradictory_fields`, not flattened across subtotals.
4. **Unsatisfiable flip** (finding 13): a case with `required ⊄ expected` is marked
   `flip_unsatisfiable` and excluded from the headline, with the count reported.
5. **Empty value set** (finding 14): a parseable note with no values scores a new outcome
   `empty`, distinct from `parse_error`; `value_drop_cleared` still recorded.
6. **Control applicability** (finding 15): per-case applicability recorded; control rates are
   computed over applicable cases only, with n printed beside every rate.
7. Finding 10 (unpooled content-control replacement at the slot cell) was correctly sent
   back; it is a condition of the primary rerun.
8. Order: ledger primary rerun first (after finding 10 lands); the secondary after 1–6 land.

## #35 R26 crash amendment: RATIFIED.

On any exit before the planned iterations complete, the error path included, the health
summary records `incomplete_run` (iterations done below planned), so a crashed run can never
read `healthy`; precedence unchanged. This is R26(c) as it should have been written; the
first live B4 attempt (0 iterations, `status: error`, `verdict: healthy`) is the evidence. The
loader fix it ships with comes to the gate on its own merits.
