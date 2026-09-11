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

### The 12B registration (complete, 16:29Z)

The same tool on the 12B: 600 cells (300 sampled decisions × {P_note, P_act}, positions 415–1,552 in contexts up to 1,554 tokens) × 47 repository layers, the merged float32 lens `49001ae6…` with ν `17a4a9e5…` whose three chunks were fitted at widths 16, 16 and 8 — so each cell's fit path was run three times, once per chunk width, and the registered scalar is the maximum of the three chunk terms — against the capture at width 1. 28,200 registrations, 2 h 27 min on the GPU, sha256 `f7b604a130c281aec9e86524d0e3f2ecc036a10d6ba81b4399c3374b4f84baf3`, 80.2 MB on the card at `/workspace/chief/captures/12b/`; summary [pairings-12b-summary.json](pairings-12b-summary.json); the 12B domain statement is [domain-12b.txt](domain-12b.txt) and is discussed under §3.4.

| layer | median | 90th pct | max | cells above 1e-3 |
|---:|---:|---:|---:|---:|
| 1 | 5.9e-07 | 1.2e-06 | 2.6e-06 | 0 |
| 5 | 5.8e-07 | 1.1e-06 | 2.5e-06 | 0 |
| 10 | 4.5e-07 | 8.4e-07 | 2.0e-06 | 0 |
| 15 | 3.0e-07 | 5.9e-07 | 1.3e-06 | 0 |
| 20 | 4.5e-07 | 8.9e-07 | 1.5e-05 | 0 |
| 25 | 4.7e-07 | 8.5e-07 | 1.7e-05 | 0 |
| 30 | 7.5e-07 | 1.2e-06 | 1.9e-05 | 0 |
| 35 | 9.7e-07 | 1.6e-06 | 1.9e-05 | 0 |
| 40 | 1.2e-06 | 1.9e-06 | 3.2e-05 | 0 |
| 45 | 1.4e-06 | 2.1e-06 | 4.7e-05 | 0 |
| 46 | 1.5e-06 | 2.4e-06 | 5.1e-05 | 0 |
| 47 | 1.4e-06 | 2.5e-06 | 5.0e-05 | 0 |

Every term below 5.1 × 10⁻⁵ (one cell at layer 46), medians 4 × 10⁻⁷ to 1.5 × 10⁻⁶, rising with depth as on the 4B. The width-1 replay reproduced the stored capture bit for bit on all 600 cells at all 47 layers.

## 2. The declared configuration, before any numeric run

Two configs, written by [bridge_configs.py](bridge_configs.py) on the card from the captures' hash manifests and the admission sidecars (no digest typed), and carried verbatim into every result: [bridge/config-4b.json](bridge/config-4b.json) (dictionary `resid_post_all/layer_17_width_16k_l0_small` of `google/gemma-scope-2-4b-it`, repository layer 18) and [bridge/config-12b.json](bridge/config-12b.json) (`layer_23…` of `google/gemma-scope-2-12b-it`, repository layer 24). Each carries a `declared_basis` block stating why each threshold has the value it has; in brief: `k = 10` and `chunk = 256` as the laptop A1; the control layer a strong control from the laptop control curve (layer 2 for the 4B, 0.047 overlap on the hosted lens; layer 3 for the 12B by analogy), with `maximum_control_overlap = 0.5` because the curve shows a neighbouring layer shares 0.7–0.8 of top-10 sets; the eight named features of the laptop A1 for the two-products check with an absolute tolerance of 1e-3 on scores of order 1–40; the convention discriminator at 0.5, the second amendment's table; and the A2 error budget at 0.5 for both the raw reconstruction share and the lens-score error share, denominator floor 1e-6, near-zero refusing, with the identity tolerance 1e-3 relative. None of these admits by default; each is a refusal line. A third config, `config-4b-l24.json`, is the 4B one with the dictionary at `layer_23` (repository layer 24), the first layer the domain statement covers at the action position, so that one A2 reading is inside measured validity.

**Admission dry-runs, both passed** ([bridge/admission-4b-dry-run.json](bridge/admission-4b-dry-run.json): config, checkpoint, lens, dictionary, examples, capture, positions, pairings — the 600 cells' registrations matched through the runner's `_cell_pairing`; [bridge/admission-12b-dry-run.json](bridge/admission-12b-dry-run.json): config, checkpoint, lens, dictionary, examples). No labels: the runner records that no Neuronpedia source maps to `resid_post_all`, so no feature in this record carries a published label, and none is invented.

## 3. The runs

### 3.1 The 12B A1 at repository layer 24 — complete ([bridge/result-12b-a1.json](bridge/result-12b-a1.json))

The first A1 through a 12B lens (the admitted float32 merge of the three chunk fits, `49001ae6…`), dictionary `layer_23_width_16k_l0_small` of `google/gemma-scope-2-12b-it`, 16,384 features × top-10, CPU, 13:28Z–13:35Z.

- **Convention, raw arm recorded first:** overlap@10 with the shipped `top_tokens` **0.134** without the gain, **0.981** with it. The discriminator's verdict: the shipped file applies the final gain; convention settled. The 12B suite's file was made the same way as the 4B's (laptop: 0.245 / 0.995), and the raw arm was written down before the gain arm was looked at, as the rule requires.
- **Two products, float32:** the eight named features through the composed path and by a direct product each: top-10 sets identical, worst absolute score gap **1.9 × 10⁻⁶** against the 1e-3 line.
- **Negative control:** the same eight features read through the layer-3 map share **0.000** of their top-10 with the layer-24 reading (max 0.0), below the 0.5 line; the control distinguishes.
- **The readout's shape:** top-10 scores from 0.57 to 39.9 (median 4.0); 10,635 distinct top-1 tokens across 16,384 features, the most common top-1 covering 0.35% of features — no single token dominates. No labels (no published source maps to `resid_post_all`), and these are gain-only linear scores, not logits.

### 3.2 The 4B at repository layer 18 — A1 passed, A2 refused by the identity check, and what the refusal was

The run (13:22Z–13:28Z, dictionary `layer_17…`) passed A1 and stopped in A2 on the first cell: "the score decomposition is not exact at token 30423: gap 3.418e-03 against a score of 1.177e-01". The runner asserts, before it ranks anything, the identity

    L h = L b + Σ_i z_i (L d_i) + L e ,   with e = h − b − Σ_i z_i d_i ,

on the emitted token, and refuses when the gap exceeds `1e-3 × max(1, |L h|)`; the identity is exact in exact arithmetic for *any* linear `L`, so a gap can only come from an inconsistency between the two ways the code forms the terms (an orientation or convention error) or from rounding. The refusal cannot say which, so a diagnostic ([diag_identity.py](diag_identity.py), output [bridge/diag-4b-l18.json](bridge/diag-4b-l18.json)) recomputed every term for all 600 cells twice from the same float32 `h` and `z`: once along the runner's float32 path and once in float64.

| | median | max |
|---|---:|---:|
| gap, float32 (the runner's path) | 1.2 × 10⁻³ | 5.0 × 10⁻³ |
| gap, float64, same inputs | 1.8 × 10⁻¹² | 9.1 × 10⁻¹² |
| Σ\|terms\| (16,384 summands + bias + residual) | 4,300 | 7,829 |
| float32 gap / Σ\|terms\| | 2.5 × 10⁻⁷ | 1.5 × 10⁻⁶ |

**Rounding, not orientation.** In float64 the identity holds to one part in 10¹²; in float32 the same terms miss it by a few parts in 10⁷ of their absolute sum — the expected size of accumulated rounding (unit roundoff 6 × 10⁻⁸) in a sum of 16,384 terms of order 1–40 whose net is order 1. The declared tolerance compared that rounding to the *net* score, which at this cell happened to be 0.12, and so refused. Only one of the 600 cells crosses the line under the runner's rule (the others have larger net scores), which is why the run stopped where it did rather than everywhere. The orientation is right.

**The fix, on main at `468756c` (Chief; file-only review requested from Codex in the bridge order).** `decompose_position` now asserts the identity in float64 from the same float32 `h` and `z`, reports the float32 gap and Σ|terms| beside it, and adds a second guard that the float32 score path (which still produces the norms, the shares and the top-k) agrees with the float64 identity at the emitted token — so a mismatched orientation between the two paths is still refused (a test breaks the score path's orientation and sees the refusal; 104 tests in the three bridge suites pass). The fixed runs at layers 18 and 24 run from a fresh checkout of `468756c` on the card, never from the checkout a running process is using.

**What the diagnostic also showed, ahead of the A2 result.** On these 600 out-of-domain cells at layer 18 the dictionary reconstructs the residual well in the raw sense — raw share `‖e‖/‖h‖` median **0.066**, max 0.087, inside the 0.5 line — but the *lens-score* error share `‖L e‖/‖L h‖` is median **0.63**, max 0.82, outside it: the 6.6% of the residual the dictionary leaves unexplained carries most of the readout-relevant direction at this layer. That is exactly the "rankable by the raw share, refused by the score share" case T3 was built to name, and it says that at layer 18 an A2 ranking would describe the dictionary's failure at these positions, not the position. Median active features 20. The A2 result files will carry this per cell.

### 3.3 The 4B at repository layers 18 and 24 — A1 passed at both, A2 ranked nothing at either, and what that measures

Both runs from a fresh checkout of the fixed code (`aa2daa4`: the identity in float64, the float32-path guard measured against the dot product's term scale; the first fix `468756c` had the guard against the net score and refused rounding once more — recorded in the order), 13:48Z–14:04Z, CPU: [bridge/result-4b-l18.json](bridge/result-4b-l18.json), [bridge/result-4b-l24.json](bridge/result-4b-l24.json), summaries beside them from [summarise_a2.py](summarise_a2.py). The layer-24 run that had completed under the earlier code gives the same shares to every printed digit; only the identity terms changed.

**A1, both layers, all checks passed.**

| | layer 18 (dictionary `layer_17`) | layer 24 (dictionary `layer_23`) |
|---|---:|---:|
| convention: raw arm, recorded first / gain arm | 0.245 / 0.995 — gain, settled | 0.575 / 0.997 — both above 0.5, uninformative, not a stop |
| two products, worst absolute gap (line 1e-3) | 4.3 × 10⁻⁶ | 3.8 × 10⁻⁶ |
| control, layer 2, mean overlap@10 (line 0.5) | 0.000 | 0.000 |
| top-10 scores, min / median / max | 0.54 / 5.9 / 71 | 0.44 / 3.1 / 28 |
| distinct top-1 tokens of 16,384; most common share | 10,016; 1.6% | 11,510; 0.9% |

The layer-18 convention numbers reproduce the laptop's to the third decimal (0.245 / 0.995), as they must — the discriminator reads the dictionary against the unembedding with no lens and does not depend on the device. At layer 24 the raw arm already agrees with the shipped file above the line, so the check has no power there and says so; the gain arm's 0.997 is consistent with the settled convention, and nothing is concluded from it. The control at layer 2 through the float32 device lens shares nothing at all (the laptop's bf16 lens shared 0.047 on 512 features; this is 8 features), and the two-products identity holds to a few parts in 10⁶.

**A2, both layers, both positions: raw reconstruction admissible everywhere, lens-score ranking refused everywhere.**

| position, layer | raw share ‖e‖/‖h‖, median [min, max] | lens-score share ‖L e‖/‖L h‖ | ratio | active features | identity gap, float64 (float32) |
|---|---:|---:|---:|---:|---:|
| P_note, 18 | 0.064 [0.056, 0.074] | 0.70 [0.55, 0.82] | 11.1 | 21 [11, 27] | 9 × 10⁻¹² (4.8 × 10⁻³) |
| P_act, 18 | 0.070 [0.062, 0.087] | 0.59 [0.53, 0.71] | 8.4 | 20 [12, 27] | 6 × 10⁻¹² (4.9 × 10⁻³) |
| P_note, 24 | 0.123 [0.096, 0.175] | 0.67 [0.54, 0.82] | 5.5 | 19 [7, 31] | 1.5 × 10⁻¹¹ (7.8 × 10⁻³) |
| P_act, 24 | 0.135 [0.109, 0.159] | 0.71 [0.63, 0.82] | 5.4 | 16 [10, 25] | 1.1 × 10⁻¹¹ (9.8 × 10⁻³) |

Ranked: **0 of 300** in every row; raw-admissible: 300 of 300 in every row; every refusal reason the same: "lens score error exceeds its declared threshold: the decomposition describes the dictionary's failure at this position; features are not ranked under it".

**What the two shares measure, and why they disagree.** The dictionary writes the residual as `h = b + Σ_i z_i d_i + e`, with `e` the part it does not reconstruct. The *raw* share is `‖e‖ / ‖h‖` in the residual stream's own metric: here 6–7% at layer 18 and 12–14% at layer 24 — the dictionary reconstructs these out-of-domain, agent-transcript activations about as well as such dictionaries reconstruct anything, with 16–21 features active of 16,384. The *lens-score* share is the same ratio after the linear readout `L = W · diag(g) · J_ℓ` has been applied: `‖L e‖ / ‖L h‖`, over all 262,144 vocabulary scores. The ratio of the two shares is

    (‖L e‖ / ‖L h‖) / (‖e‖ / ‖h‖)  =  (‖L e‖ / ‖e‖) / (‖L h‖ / ‖h‖) ,

the gain of the readout on the unexplained part relative to its gain on the whole activation. That ratio is **8–11 at layer 18 and 5–7 at layer 24**, in every one of the 600 cells (the minimum over cells is 3.7). A ratio of one would mean the residual `e` points in directions the readout treats like any other; a ratio of ten means it points, disproportionately, along the directions the readout amplifies most. So the 6% of the activation the dictionary leaves behind carries 60–70% of the norm of what the vocabulary would read from this layer, and the 94% the dictionary does explain carries the rest. The A2 stage, as declared in the config before any run, therefore does not rank features by their contribution to the emitted token's score: any such ranking would be a ranking of contributions to a third of the score, presented as if it were the score. This is the case T3 was built to name — "rankable by the raw share, refused by the score share" — on real activations, at both layers, both positions, all 600 cells.

**What it does and does not say.** It does not say the features are meaningless, nor that the dictionary is wrong about the residual stream: in its own metric it is good. It says that the readout-relevant part of these activations, at these out-of-domain positions, lives mostly in the dictionary's error term, so a feature-level account of *what the model is about to read out* cannot be built from these dictionaries here. It also does not say the lens is at fault: the same amplification appears at layer 24, which the domain statement covers at the action position (W-5 agreement 0.99), and at layer 18, which it does not. Whether the ratio falls where the dictionary is in its own domain (prose, short contexts), whether a wider or lower-sparsity dictionary explains the readout-relevant directions, and whether the ratio is a property of these dictionaries' training objective (the residual metric) rather than of the model, are three measurements this record does not make and does not guess at. Under the two interpretation limits carried in every result: a contribution to a lens score is a contribution to that score at that layer under this averaging convention, not a probability and not a cause; and no description here is a published label.

**The identity, now asserted in float64.** The float64 gap is at most 1.5 × 10⁻¹¹ across 1,200 cells, while the same terms summed in float32 miss by up to 9.8 × 10⁻³ — the rounding the first run refused. Both numbers are in every A2 row so a reader can see what the float32 path did.

### 3.4 The 12B at repository layer 47 — inside the domain statement; A1 passed, A2 ranks 90 of 300 at the action position

Dictionary `layer_46_width_16k_l0_small` of `google/gemma-scope-2-12b-it`, the merged float32 lens, the 12B pairings of §1, config [bridge/config-12b-l47.json](bridge/config-12b-l47.json), dry-run admitted on all eight checks, run 16:34Z–16:51Z: [bridge/result-12b-l47.json](bridge/result-12b-l47.json), [bridge/summary-12b-l47.json](bridge/summary-12b-l47.json). Layer 47 is the layer the 12B domain statement licenses at the action position without qualification (W-5 resolves every cell there, agreement 0.96–1.00 in every band); the note position stays outside measured validity and is reported as such.

**A1:** raw arm 0.674 recorded first, gain arm 0.987 — both above the line, so the discriminator is uninformative here, as at 4B layer 24, and nothing is concluded from it; two products to 1.2 × 10⁻⁶; the layer-3 control shares nothing; 13,122 distinct top-1 tokens, the most common covering 0.2% of features; top-10 scores 0.15–6.2.

| position | raw share, median [min, max] | lens-score share | ratio | active | ranked |
|---|---:|---:|---:|---:|---:|
| P_note (outside validity) | 0.200 [0.150, 0.262] | 0.64 [0.54, 0.73] | 3.2 [2.8, 3.8] | 32 | 0 of 300 |
| P_act (inside validity) | 0.159 [0.126, 0.207] | 0.53 [0.44, 0.62] | 3.3 [2.7, 3.9] | 14 | 90 of 300 |

The amplification is smaller at the 12B's last stored layer — about 3× rather than 5–11× — and at the action position the lens-score share falls below the declared 0.5 line in 90 cells, which are therefore ranked (their top features are in the result file; no labels exist for them, and none are given). The identity holds to 4 × 10⁻¹¹ in float64 against float32 gaps up to 3.1 × 10⁻².

### 3.5 The in-domain control: the amplification belongs to the dictionaries and the readout, not to the positions

The question §3.3 left open first: is the 5–11× amplification a property of the out-of-domain, agent-transcript positions, or of these dictionaries and this readout wherever they are applied? [indomain_control.py](indomain_control.py) answers it with the same dictionaries, the same admitted lens, the same error budget and the **same `decompose_position`** as the A2 runs, applied to residuals from the lens's *own* fit domain: the fit split's 201 prose prompts, 128 tokens, at five fixed positions (16, 44, 71, 99, 126), captured at width 1 in float32 with the capture tool's conventions (bf16 checkpoint cast to the coherent float32 model, eager attention, block output at repository layer L). 1,005 cells per layer; 17 minutes on the card, 16:35Z–16:52Z; [bridge/indomain-4b.json](bridge/indomain-4b.json). This is a control, not an admission run: these cells have no registered pairing (they *are* the fit domain), and the runner's capture-format admission does not apply to them.

| layer, in domain | raw share, median [min, max] | lens-score share | ratio | active | ranked |
|---|---:|---:|---:|---:|---:|
| 18 | 0.092 [0.027, 0.521] | 0.62 [0.32, 2.17] | 6.6 [2.0, 16.4] | 19 | 155 of 1005 |
| 24 | 0.148 [0.028, 0.266] | 0.65 [0.39, 1.14] | 4.4 [2.6, 22.3] | 17 | 63 of 1005 |

By position (median ratio): | position | layer 18 | layer 24 |
|---|---:|---:|
| 16 | 6.7 | 4.3 |
| 44 | 6.6 | 4.5 |
| 71 | 6.6 | 4.4 |
| 99 | 6.8 | 4.4 |
| 126 | 6.3 | 4.5 |

**Reading it against §3.3.** In the dictionaries' own domain the raw share is *higher* than on the agent transcripts (0.09 against 0.06–0.07 at layer 18; 0.15 against 0.12–0.14 at layer 24 — these dictionaries reconstruct the structured transcripts slightly better than prose), and the lens-score share is the same 0.6–0.65. The ratio is 6.6 at layer 18 and 4.4 at layer 24 in domain, against 8–11 and 5.4 out of domain: the same order, the same ranking of the two layers, at every one of the five positions. The amplification is therefore not the positions'. It is a property of the pair (dictionary, readout): whatever these dictionaries leave unexplained lies, disproportionately, along the directions the vocabulary readout amplifies, in prose at the fit length as much as in agent transcripts thirty times longer. About one in seven in-domain cells at layer 18 (155 of 1,005) and one in sixteen at layer 24 (63) fall under the 0.5 score-share line and would be ranked; on the agent transcripts none did at either layer, and at the 12B's last layer 90 of 300 did at the action position.

**What licenses the reading, in one line of algebra.** With `L` the linear readout, the ratio `(‖L e‖/‖L h‖)/(‖e‖/‖h‖)` equals `(‖L e‖/‖e‖)/(‖L h‖/‖h‖)`: the readout's gain on the residual `e` divided by its gain on the activation `h`. A dictionary trained to make `‖e‖` small in the residual stream's own metric is not trained to make `‖L e‖` small; if the directions `L` amplifies carry little of the activation's variance, the dictionary has no reason to spend features on them, and its error lands there. That is consistent with the published observation that sparse-autoencoder reconstruction errors move the next-token distribution more than random perturbations of the same norm (Gurnee, 2024, "SAE reconstruction errors are (empirically) pathological"); this record measures it on the lens-readout metric at two layers of the 4B and one of the 12B and does not go further than that.

**The same control on the 12B** ([indomain_control_any.py](indomain_control_any.py), the 4B script with its paths as arguments; the 12B lens was fitted on the same prose manifest, which the Gemma 3 tokenizer shared across sizes makes the same token ids; 21 minutes, 16:55Z–17:16Z; [bridge/indomain-12b.json](bridge/indomain-12b.json)):

| layer, in domain | raw share, median [min, max] | lens-score share | ratio | active | ranked |
|---|---:|---:|---:|---:|---:|
| 24 | 0.088 [0.021, 0.383] | 0.77 [0.49, 1.46] | 8.5 [3.2, 30.5] | 21 | 1 of 1005 |
| 47 | 0.182 [0.135, 0.337] | 0.71 [0.48, 0.97] | 3.9 [2.4, 5.9] | 7 | 3 of 1005 |

At 12B layer 47 the in-domain ratio is 3.9 against 3.2–3.3 on the agent transcripts, and at layer 24 it is 8.5 — the 12B's middle layer amplifies the residual as strongly as the 4B's, and a few in-domain cells there have a lens-score share above one (the residual's readout is larger than the whole activation's, which happens when `L e` and `L h` are partly opposed). The ranking of layers and the flatness across positions repeat. Four models-and-layers, in and out of domain, say the same thing.

**What remains not measured.** Whether a wider or denser dictionary (the suites ship `l0_medium` and 65k/262k widths) lowers the ratio; whether a dictionary trained on the readout metric would; and whether the 90 ranked 12B cells' top features say anything about the action — the last requires labels these dictionaries do not have, and the interpretation limits forbid inventing them.

## 4. The review of 2026-09-11 and the model's own answer

A review received on the morning of 2026-09-11 (Daniel's relay; the text is in the order) made the point that the finding above rests on a metric — the lens-score share over the whole vocabulary — that is not a measurement of the model's decision, and proposed six experiments to separate "the dictionary misses useful information" from "the metric overstates how much that matters". Three have run; this section carries the decisive one first.

### 4.1 Substituting the reconstruction into the model (the review's experiment 3), corrected

A second review (same source, later on 2026-09-11; filed at the end of REVIEW-2026-09-11.md) found three faults in the first run of this test: the angle-matched control matched the projection of `+e` where the intervention moves the residual by `−e`; the lens prediction held the final RMS fixed while the exact derivative let it respond, so the two were not comparable; and KL computed in float32 went negative at the 10⁻⁷ scale, and a floor of 10⁻¹² in the ratio could manufacture huge values. All three are corrected in [substitution_test_v2.py](substitution_test_v2.py), whose output [bridge/subst2-4b.json](bridge/subst2-4b.json) replaces the first run's ([bridge/subst-4b.json](bridge/subst-4b.json), kept). The revision also runs the intervention at a quarter, a half and full strength; adds a second sample, the 32 lowest-margin decisions among the 300 action cells (selected by one unpatched forward each; the margin does not depend on the layer); and reports three predictions of the top-10 logit-gap changes — the exact derivative through the remaining model, the lens's residual change `J δ` through the exact derivative of the model's own final readout at the original final residual (normalisation responding), and the fixed-RMS lens — plus a comparison before any readout: the exact derivative of the final pre-norm residual along `δ = ĥ − h` against `J δ`. 8 minutes on the card, 00:32Z–00:40Z. The inert same-state control changes nothing (max logit difference 0.0) and the unpatched forward reproduces the capture bit for bit, in all 128 cell-layers.

**What was tested, exactly.** The next token under a teacher-forced prefix at the action position, the six-tool decision's first token. Complete tool names, arguments and outcomes were not tested here. In the first sample 30 of 32 baselines emit the expert's token; in the low-margin sample 23 of 32. Margins over all 300 action cells: median 15.3 logits, tenth percentile 9.6, minimum 1.2.

| 4B | L18, first 32 | L18, lowest-margin 32 | L24, first 32 | L24, lowest-margin 32 |
|---|---:|---:|---:|---:|
| original margin, median [min] | 16.3 [4.4] | 7.2 [1.2] | 16.3 [4.4] | 7.2 [1.2] |
| **argmax flips, reconstruction (random / angle-matched, of 128 draws)** | **0 of 32 (random 2 of 128; angle 0 of 128)** | **9 of 32 (random 15 of 128; angle 13 of 128)** | **0 of 32 (random 0 of 128; angle 0 of 128)** | **0 of 32 (random 15 of 128; angle 12 of 128)** |
| flips at ¼ / ½ / full strength | 0 / 0 / 0 | 1 / 2 / 9 | 0 / 0 / 0 | 0 / 0 / 0 |
| margin after, median [min] | 14.3 [6.6] | 7.7 [0.05] | 11.9 [1.8] | 9.1 [1.2] |
| top-1/top-2 gap change, median [min, max] | −0.7 [−6.4, +5.9] | +0.7 [−11.8, +5.9] | −0.6 [−11.1, +7.7] | +3.5 [−5.0, +8.2] |
| KL (float64), reconstruction: median / max | 3 × 10⁻⁷ / 0.06 | 1.1 × 10⁻³ / 6.4 | 1.9 × 10⁻⁵ / 0.11 | 2.5 × 10⁻³ / 0.47 |
| KL, random of the same norm: median / max | 3 × 10⁻⁷ / 3.6 | 1.5 × 10⁻³ / 13.8 | 8 × 10⁻⁷ / 0.11 | 3.2 × 10⁻³ / 15.6 |
| KL, angle-matched (signed projection matched): median / max | 4 × 10⁻⁷ / 0.22 | 1.1 × 10⁻³ / 12.6 | 4 × 10⁻⁷ / 0.23 | 3.0 × 10⁻³ / 4.4 |
| KL paired difference, reconstruction − random median: median / p90 | +5 × 10⁻⁹ / +2 × 10⁻⁵ | −1 × 10⁻⁴ / +1.7 | +1.5 × 10⁻⁵ / +1.4 × 10⁻³ | −6 × 10⁻⁴ / +4.7 × 10⁻³ |
| KL ratio where the denominator resolves, median | 1.3 | 0.8 | 7.4 | 0.6 |
| before the readout: cos(exact ∂h_final along δ, J δ), median [min, max] | −0.04 [−0.43, 0.60] | −0.05 [−0.50, 0.36] | 0.19 [−0.47, 0.56] | 0.46 [−0.19, 0.84] |
| corr with the actual gap changes at full strength: derivative / lens through the head / fixed-RMS lens | 0.82 / 0.27 / 0.22 | 0.83 / 0.22 / 0.23 | 0.84 / 0.23 / 0.25 | 0.86 / 0.34 / 0.30 |
| relative error of the derivative at ¼ / ½ / full | 0.16 / 0.32 / 0.62 | 0.21 / 0.43 / 0.66 | 0.14 / 0.28 / 0.58 | 0.14 / 0.27 / 0.56 |
| relative error of the lens through the head at ¼ / ½ / full | 1.49 / 1.50 / 1.50 | 1.42 / 1.44 / 1.24 | 1.39 / 1.34 / 1.30 | 1.64 / 1.55 / 1.46 |

**The decision moves where the margin is small, and at layer 18 the reconstruction moves it more often than a random error of the same size.** On the first sample the argmax never changes at either layer. On the lowest-margin sample at layer 18 the reconstruction flips 9 of 32 next tokens at full strength (1 at a quarter, 2 at a half) — 28% — while the four random draws per cell flip 15 of 128 (12%) and the angle-matched draws 13 of 128 (10%). By that count the dictionary's error is about twice as likely as a matched random error to cross a close decision boundary at layer 18. The KL tells a different part of the story: its paired difference against the cell's own random median is negative at the median (−10⁻⁴, ratio 0.8) with a p90 of +1.7, so on most low-margin cells the reconstruction disturbs the distribution less than a random error and on a few it disturbs it far more — the effect is uneven across cells, as the review said, not uniformly worse. At layer 24 the picture reverses: the low-margin sample flips nothing under the reconstruction while random draws flip 15 of 128, and the reconstruction's KL is below random's (ratio 0.6); on the first sample the reconstruction's KL is 7.4× the random median but the paired difference is 1.5 × 10⁻⁵ at the median and 10⁻³ at the ninetieth percentile, small against a 12-logit margin. So: not "the decision never moves" — it moves on close decisions at layer 18, at twice the random rate; and not "the dictionary's error is broadly pathological" — at layer 24 it is gentler than random. A complete-call test on the low-margin cells is the next measurement, since a flipped first token there may or may not be a different call.

**Where the intervention is not linear.** The exact derivative's relative error grows with strength in proportion — 0.15 at a quarter, 0.3 at a half, 0.6 at full, at both layers and both samples — which is the signature of second-order terms at the reconstruction's actual size, not of a wrong derivative (row four of the review's table). At a quarter strength the derivative predicts the gap changes to 15%.

**The lens map does not propagate this perturbation, and normalisation is not the reason.** Passing the lens's residual change through the exact derivative of the model's own final readout, so the normalisation responds, leaves its relative error at 1.3–1.6 at every strength (a prediction worse than predicting zero), against 0.14–0.62 for the derivative; and before any readout the cosine between `J δ` and the exact change of the final residual is −0.04 at layer 18 and 0.19–0.46 at layer 24. The averaged, fit-domain map `J` does not describe how this model carries a perturbation from these positions to its output, even though it reads the six tool logits at the action position to 0.99 agreement (W-5). Reading a state and predicting a perturbation are different uses, and the second is not licensed by the first.

**Consequence for §3.** The A2 refusals stand as what they are — the declared vocabulary-wide budget was not met — but the sentence "the readout-relevant part of these activations lives mostly in the dictionary's error term" is withdrawn as a statement about the decision: where the decision has a margin it does not change; where it is close, the reconstruction at layer 18 changes it about twice as often as a random error of the same size, and at layer 24 less often. What §3 measured, and still shows, is that the dictionaries' error is amplified by the gain-only vocabulary readout in directions that do not bear on this decision. A decision-focused fidelity measure (the review's experiment 1, running as this is written) is the instrument to add beside the budget, not in place of it.
