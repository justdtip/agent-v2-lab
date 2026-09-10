# Follow-up: accepted corrections are present, but the conclusion still contradicts them

**Review of `547b7a0`, against the Chief's ruling at `6c2f7d0`.** The audit `ca396fb` was accepted in full and merged at `ff7cd93`. The D-CRO correctly relabelled the map error, marked the repeat gate unexecuted, withdrew the amplification ratios, and narrowed the random-direction claim. The newly added min/median/max table reproduces at its stated precision from all 108 scalar rows. No new experiment or scientific result is present in this update.

**One material documentation correction remains:** the producer's current concluding caution still says that reading a float32 lens against a native capture is “benign at layer 33”, supported by 1.4× and 539×. That reinstates the withdrawn numerical argument and an unmeasured compatibility claim. The Chief's latest correction is authoritative; this surviving paragraph is not evidence for safe cross-precision reading at any layer.

## Where the correction did not propagate

`check.json` gives exact line numbers in the frozen [producer document](producer.md). These are current explanatory/concluding passages, not excerpts deliberately labelled as historical errors.

| Passage | Remaining problem | Correction |
|---|---|---|
| Final caution in §11, beginning “is benign at layer 33” | Uses the withdrawn 1.4×/539× ratios to distinguish safe and unsafe pairings | Replace the paragraph with the policy wording below |
| §8.2 explanation, beginning “A map whose derivative changes by 100% under a 0.19% move” | Treats the selected-token movement as the size of the whole intervention again | Describe the measured changes without the 0.19% whole-input interpretation; their connection to an FD operating interval remains an inference |
| §9 opening, “every layer is refuted — the two minima are a factor of 64 apart” | Contradicts the explicit §11 correction: normalized minima differ by 16 and a common adequate scale was not refuted | Refer to the two tested layer-specific steps without claiming a universal common-scale refutation |
| §11 item 5, “width-stable at 12B scale at every depth” | Omits the accepted three-sampled-layers limitation | Say “at the three sampled layers”; retain the smoke row's one-layer, one-row, 128-token scope |

Suggested replacement for the concluding caution:

> The sampled native and random displacements produced substantial changes in float32 directional derivatives at the early layers. These measurements do not establish a whole-map condition number or compatibility between a fitted lens and captures from another precision or path. Cross-path compatibility remains unmeasured, including at layer 33. The ruled restriction on early-layer cross-path readings remains in force until the intended pairing is measured under its own reduction, population and precision.

The already-corrected sections should remain. Older explanatory passages can be narrowed or explicitly marked superseded; their unrevised form should not be treated as a fresh finding. No new device work is requested, and the existing queue remains unchanged. The genuine exact-repeat comparison and complete P1–P6 preregistration amendment are still queued, not reviewed as executed.

## Technique and implementation

This check asks whether an accepted correction propagated into the document's active conclusions. It distinguishes that question from whether the original finding changed. The technique transfers: verify the corrected summary against raw data, then inspect every still-active use of the withdrawn claim, especially concluding recommendations.

`check.py` uses only the standard library. It verifies the four frozen source hashes, recomputes the six new range rows, and inventories four specifically identified passages. This is a pinned passage inventory, not a general semantic linter. `verify.py` checks that each passage is detected independently and that removing one does not hide the others; it also checks refusal of corrupted source bytes and corrupted scalar summaries. Run `python3 check.py` to reproduce `check.json`, and `python3 verify.py` with `ruff` on PATH for the verification record.

All evidence is pinned in `sources.json`. Producer files are copied unchanged for evidence and have not been modified at their source. No model, native test suite, remote command, or device access was used. This commit is ready for documentation review.
