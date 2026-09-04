# SPEC-004 §1 (remediated), §3, §5 — review round 2 (retrospective), ratified (2026-09-05 02:05)

Reviewer: Claude (Chief). Evidence: source-level audit, the rerun artifacts under
`outputs/probes/state-corrected-hardened-20260903T172549/`, the ratified P6 rounds (#26, #29,
#30), and the Chief's direct read of `adapter_delta._layer_blocks`.

## Verdict

- **§1: COMPLETE except C7.** C1–C6 verified in the rerun artifact and report; the corrected
  readings table (2026-09-04) matches the regenerated file. **C7 (bfloat16-rounding refit) is
  an offline implementer slice, not a Director item**; the probe checklist's "assigned to the
  Director" is corrected to "unassigned implementer slice, no model". §1 closes when C7 lands.
- **§3 block ablation: code COMPLETE.** `--ablate --blocks N --screen`, leave-one-out masks over
  a partition that covers every layer once (`_layer_blocks`, `adapter_delta.py:417-428`),
  Wilson intervals, full/empty anchors, output. The run is gated (probe checklist B2).
- **§5 P6 patching: code COMPLETE** with #26, #29 (committed) and #30 (ratified, committing).
  Design items 1–4 covered. The `aggregate_report` generator-v4 secondary condition stays a
  bound follow-up. The run is gated (probe checklist B3).

Cosmetic, recorded: JSON schema key omission on all-NaN cells; within-position seed-index
nit; Holm arithmetic untested at shipped defaults (the real output was inspected directly).
