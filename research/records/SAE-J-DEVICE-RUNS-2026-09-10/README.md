# SAE–J bridge: the device runs — pairings registered, A1 and A2 on the card

**Chief, 2026-09-10 (card clock 13:09Z–). Standing order `SAE-J-BRIDGE-ORDER-2026-09-08.md`, T5 and the review of T1–T4: "what the Chief now supplies, in this order: the positions file … the pairing registrations … the shipped-token example files … then A1 through both admitted lenses with the corrupted-layer control, and A2 on the 4B capture."** This record holds what was supplied and what came back. Part 1 is complete; Part 3 is filled in as the runs end.

## 1. The pairing registration for the 4B (complete)

### What a pairing is, and why a reading needs one

The lens was fitted along one computational path and the capture was written along another. Both are float32 and both are the same weights, but the fit ran the model at forward batch 32 (the width the finite-difference Jacobians were taken at) and the capture ran it at width 1. On this card the float32 forward is *not* batch-invariant at the bit level: the kernels chosen for a batch of 32 accumulate in a different order from the kernels chosen for a batch of 1, so the residual `h` at a given layer and position differs between the two paths by a small amount. A lens fitted on one path and read against a residual from the other is reading a vector it was never fitted to, by that amount. The bridge order therefore made the rule: **no reading is licensed until the cross-path term at the exact cell is measured and registered**, and the registration is keyed to the whole identity of the pair (eleven fields, plus the cell receipt), so a term measured at one layer never transfers to its neighbour and a term measured on one prompt never transfers to another.

### The mathematics of the term

For a cell `(row i, position p, layer ℓ)` let `h_fit ∈ ℝ^d` be the residual at the block output produced by the fit's path (width 32, the prompt replicated 32 times so the batch schedule is the fit's) and `h_cap ∈ ℝ^d` the residual the capture stored (width 1). The registered scalar is the relative Euclidean displacement

    relative(i, p, ℓ) = ‖h_fit − h_cap‖₂ / ‖h_cap‖₂ ,

taken as the **maximum** over the lens's fit-width chunks (the 4B lens has one chunk, width 32; the 12B's merged lens has three, 16/16/8, each measured at its own width). It is a ratio of norms with no cancellation across cells and no averaging: the number beside a cell is that cell's. Why this quantity and not a lens-score difference: the lens map `J_ℓ` is linear, so for any reading `J_ℓ h` the score-space error is bounded by the operator norm times this term, `‖J_ℓ h_fit − J_ℓ h_cap‖ ≤ ‖J_ℓ‖ · ‖h_fit − h_cap‖`; registering the residual-space term bounds every linear reading at once, whatever the map's gain, and the score-space share is then reported separately by the runner (T3's two named errors) on the actual reading.

Before a term is written, the tool must reproduce the capture: it re-runs the width-1 path with the capture's own forward (the outer language model, attentions where the sample kept them, the two kept logits) and refuses unless the recomputed residual equals the stored bytes exactly. That check is what makes the term a property of the *capture's* path rather than of some path that merely shares its flags. It passed on every one of the 19,800 cells.

### The measurement

`scripts/measure_pairings.py` (T5, Codex, `5ce2635`; sha256 `1fbd37b53f6b…`), run by the Chief on the card, 12:31Z–13:09Z, untimed, alongside no measured run. Inputs: the 4B checkpoint (hash manifest checked), the admitted float32 lens `1c8d2bd7…` with ν `9f64bcf3…` (fit width 32 read from the admission, never typed), the 4B capture at width 1 float32 (33 repository layers), the positions file of 600 cells (300 decisions × {P_note, P_act}, `positions-sample.json`, whose cells the runner validated), and the rendered corpus (digest `790ceff…`). Output: `pairings-sample.json`, 19,800 registrations (600 cells × 33 layers), sha256 `988a1318b2ed0c7f0921eb7ffe867d16f117210f729159678a558b36305d9c59`, 49.5 MB, kept on the card at `/workspace/chief/captures/4b/` with a laptop copy; its per-layer summary is [pairings-4b-summary.json](pairings-4b-summary.json), produced by [summarise_pairings.py](summarise_pairings.py).

| layer | median | 90th pct | max | cells above 1e-3 |
|---:|---:|---:|---:|---:|
| 1 | 5.3e-07 | 9.7e-07 | 1.9e-06 | 0 |
| 2 | 5.8e-07 | 9.7e-07 | 2.2e-06 | 0 |
| 3 | 4.6e-07 | 7.2e-07 | 1.3e-06 | 0 |
| 4 | 5.0e-07 | 8.7e-07 | 2.4e-06 | 0 |
| 5 | 4.3e-07 | 8.7e-07 | 1.7e-06 | 0 |
| 6 | 4.6e-07 | 1.1e-06 | 2.9e-06 | 0 |
| 7 | 4.0e-07 | 7.7e-07 | 1.4e-06 | 0 |
| 8 | 4.0e-07 | 8.7e-07 | 1.8e-06 | 0 |
| 9 | 3.4e-07 | 7.2e-07 | 1.6e-06 | 0 |
| 10 | 3.5e-07 | 7.8e-07 | 1.7e-06 | 0 |
| 11 | 2.8e-07 | 6.6e-07 | 1.6e-06 | 0 |
| 12 | 3.4e-07 | 8.5e-07 | 1.9e-06 | 0 |
| 13 | 3.4e-07 | 8.1e-07 | 1.7e-06 | 0 |
| 14 | 3.6e-07 | 8.6e-07 | 1.9e-06 | 0 |
| 15 | 3.7e-07 | 8.0e-07 | 2.0e-06 | 0 |
| 16 | 3.8e-07 | 8.7e-07 | 1.5e-05 | 0 |
| 17 | 4.2e-07 | 8.3e-07 | 1.0e-05 | 0 |
| 18 | 4.8e-07 | 9.0e-07 | 5.9e-06 | 0 |
| 19 | 5.0e-07 | 9.2e-07 | 5.5e-06 | 0 |
| 20 | 5.5e-07 | 9.3e-07 | 5.8e-06 | 0 |
| 21 | 6.2e-07 | 1.0e-06 | 6.2e-06 | 0 |
| 22 | 6.8e-07 | 1.1e-06 | 7.5e-06 | 0 |
| 23 | 7.8e-07 | 1.2e-06 | 1.1e-05 | 0 |
| 24 | 8.9e-07 | 1.4e-06 | 1.4e-05 | 0 |
| 25 | 9.5e-07 | 1.5e-06 | 2.2e-05 | 0 |
| 26 | 9.9e-07 | 1.5e-06 | 3.0e-05 | 0 |
| 27 | 1.0e-06 | 1.6e-06 | 2.8e-05 | 0 |
| 28 | 1.0e-06 | 1.6e-06 | 3.0e-05 | 0 |
| 29 | 1.0e-06 | 1.7e-06 | 2.7e-05 | 0 |
| 30 | 1.1e-06 | 1.7e-06 | 2.6e-05 | 0 |
| 31 | 1.2e-06 | 1.8e-06 | 2.6e-05 | 0 |
| 32 | 1.2e-06 | 1.9e-06 | 3.1e-05 | 0 |
| 33 | 1.3e-06 | 2.0e-06 | 3.4e-05 | 0 |

**Reading the table.** Every one of the 19,800 terms is below 3.4 × 10⁻⁵, and the medians sit at 3–13 × 10⁻⁷, which is float32 rounding (unit roundoff 6 × 10⁻⁸) accumulated over a few hundred additions. The term grows slowly with depth, as an accumulated rounding difference should, and the layer-33 maximum is one cell in six hundred. The A1/A2 thresholds declared below are three to five orders of magnitude above these numbers, so on this card the fit-vs-capture path difference is not what will decide any reading; the registration exists so that "the pairing is measured" is a measured sentence with a number beside each cell, as the order required, and so that a future capture on a different kernel or precision has a table to be compared against.

### The domain-of-validity statement

Every cell lies outside the lens's fit domain (positions 16–126 in 128-token prose prompts): the cells sit at positions 400–4,300 in agent transcripts of up to 4,300 tokens. The positions file therefore carries the lens owner's statement, which the runner checks field by field against the lens and ν digests and the cells' position and context sets. The statement licenses the lens as a readout of the six tool logits at the action position **from repository layer 24** on these contexts, on the strength of W-5's band-resolved agreement (0.99/0.99/0.98 from layer 24, no band dependence), **and nothing else**: an A2 reading below layer 24, and any reading at the note position, is outside measured validity and is reported as such; A1 crosses no capture path. The full text is in the dry-run admission's `provenance.capture.domain_of_validity`.

## 2. The declared configuration, before any numeric run

Two configs, written by [bridge_configs.py](bridge_configs.py) on the card from the captures' hash manifests and the admission sidecars (no digest typed), and carried verbatim into every result: [bridge/config-4b.json](bridge/config-4b.json) (dictionary `resid_post_all/layer_17_width_16k_l0_small` of `google/gemma-scope-2-4b-it`, repository layer 18) and [bridge/config-12b.json](bridge/config-12b.json) (`layer_23…` of `google/gemma-scope-2-12b-it`, repository layer 24). Each carries a `declared_basis` block stating why each threshold has the value it has; in brief: `k = 10` and `chunk = 256` as the laptop A1; the control layer a strong control from the laptop control curve (layer 2 for the 4B, 0.047 overlap on the hosted lens; layer 3 for the 12B by analogy), with `maximum_control_overlap = 0.5` because the curve shows a neighbouring layer shares 0.7–0.8 of top-10 sets; the eight named features of the laptop A1 for the two-products check with an absolute tolerance of 1e-3 on scores of order 1–40; the convention discriminator at 0.5, the second amendment's table; and the A2 error budget at 0.5 for both the raw reconstruction share and the lens-score error share, denominator floor 1e-6, near-zero refusing, with the identity tolerance 1e-3 relative. None of these admits by default; each is a refusal line. A third config, `config-4b-l24.json`, is the 4B one with the dictionary at `layer_23` (repository layer 24), the first layer the domain statement covers at the action position, so that one A2 reading is inside measured validity.

**Admission dry-runs, both passed** ([bridge/admission-4b-dry-run.json](bridge/admission-4b-dry-run.json): config, checkpoint, lens, dictionary, examples, capture, positions, pairings — the 600 cells' registrations matched through the runner's `_cell_pairing`; [bridge/admission-12b-dry-run.json](bridge/admission-12b-dry-run.json): config, checkpoint, lens, dictionary, examples). No labels: the runner records that no Neuronpedia source maps to `resid_post_all`, so no feature in this record carries a published label, and none is invented.

## 3. The runs

*Filled in as they end: 4B at repository layer 18 (A1 + A2), 12B at repository layer 24 (A1), 4B at repository layer 24 (A1 + A2).*
