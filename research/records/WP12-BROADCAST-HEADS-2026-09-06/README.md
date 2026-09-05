# WP12-BROADCAST-HEADS-2026-09-06: which attention mediates entry to the J-space (stage 1, from weights)

**Order.** The Director, 2026-09-06 02:15: "If there is a J space, which attention mediates entry to
it? Anthropic claim that they identified a particular type of attention head that is related to the
J space. Read the full paper for specifics." The paper's claim (§4.3.2, read verbatim) is the
broadcast head: an attention head whose OV circuit selectively relays J-space content between token
positions, found from weights by two metrics against the J-lens vectors and confirmed by ablation.

**Design, reviewed by the Head (WP12 and its amendment in 03-REVISED-PLAN-2026-09-05.md).** Stage 1
is weight arithmetic with no forward pass: for each of the 128 attention heads the weight-only OV
map (W_O composed with W_V per query head, GQA over four kv heads), and for each of the 768
recurrent value paths the weight-only composition out_proj composed with in_proj_v, which omits the
causal conv1d and its silu, the gated-delta state and the gated RMSNorm, all input-dependent, so the
two channels' columns are different objects and are reported apart. Populations at the block's input
layer, from the hosted lens (file key J{b-1} for block b) and the tied embedding: 2,000 sampled J-lens
directions J^T u_w; the same under one fixed random orthogonal rotation; 2,000 MLP output rows of the
preceding block; isotropic random directions. Metrics as in the paper: gain (mean image norm over the
population divided by the median over random directions) and label preservation (mean reciprocal
rank of cos(M v_i, v_i) among all j), with the random-direction value beside them. Selection rule
fixed before the run: no percentile; a head broadcasts J when its gain on J exceeds both controls and
its label preservation on J exceeds both controls; the count is n of N per channel and the whole
ordered list is in `out/broadcast_heads.json`.

**Predictions, written before the results.** (1) The broadcast heads, if any, concentrate in the
first half of the band (the paper: where the J-space's effective rank is lowest), which here means
layers 16 and 20 over 24 and 28. Whether they are the retrieval heads of blocks 19 and 23 (heads 2,
12, 15 and 9 in the RETRIEVAL-CHANNELS run) or different heads at the same layers is the question
this stage answers first; the Head's view is that this is the more interesting of the two. (2) The
recurrent value paths show lower alignment with the lens directions than the attention paths at
matched layers, on both metrics; the recurrent-sink finding is motivation for this and not evidence,
since it measured which positions long-gate heads read and not what directions they write.

**Stage 2 (not run here).** Ablation under the lift of the named heads at every position, against
layer-matched random heads, measured by recall@25 of the hosted-lens readout at the band layers
with the next-token change rate beside it, a recall drop counting only if the change rate stays
within twice the control's; the behavioural arm waits on EXP-002's injection arm and does not hold
stage 2.

**Files.** `broadcast_heads.as-run.py.txt` (unimportable by name), `out/` and `run-stage1.log` when
the run lands.

## Result, 2026-09-06 03:05 (stage 1; 26 s; 896 heads; 2,000 directions per population)

**Attention: 24 of 128 heads broadcast the lens population beyond both controls on both metrics;
in the band, 12 of 64.** By layer written (broadcasting n of 16; median gain on J; median label
preservation on J): layer 4: 3, 0.96, 0.11; 8: 2, 0.82, 0.63; 12: 0, 0.92, 0.65; 16: 0, 0.86, 0.35;
20: 8, 1.13, 0.45; 24: 4, 1.00, 0.22; 28: 0, 0.85, 0.10; 32: 7, 1.21, 0.02. Across all attention
heads the median label preservation on J is 0.178 against 0.011 for rotated J, 0.076 for MLP rows
and 0.013 for random directions, so attention's OV maps preserve the lens directions as labels
sixteen times more faithfully than the same directions rotated; the gain medians sit at 0.93, 1.00
and 0.96, so the selectivity is in preservation, not amplification. The strongest head by the worse
of its two margins is block 19 (layer 20) head 0: gain 1.42 against 0.99 rotated and 0.67 MLP, label
preservation 0.664 against 0.006 and 0.037. The full ordered list is in `out/broadcast_heads.json`;
the broadcasting heads are blocks 3 (heads 4, 12, 13), 7 (4, 5), 19 (0, 1, 3, 10, 12, 13, 14, 15),
23 (0, 1, 9, 11) and 31 (4, 5, 6, 11, 12, 13, 14). Layers 8 and 12 preserve labels for every
population (medians 0.63 to 0.65 on J and 0.66 on rotated J at the block 7 head shown), near-identity
copy maps whose margins are therefore small; the rule handles them.

**Prediction 1.** The broadcasting heads concentrate at layer 20 (8 of 16) over 24 (4 of 16), with
none at 16 or 28, and a second group at the final block, layer 32, next to the unembedding, which the
paper's "workspace-layer" restriction would exclude. So the first half of the band holds them, as
the paper found, with the qualification that the first in-band attention layer (16) holds none. The
retrieval heads of the RETRIEVAL-CHANNELS run rank among the 128 attention heads at 11 (block 19
head 12), 12 (block 19 head 15), 17 (block 23 head 9), 28 (block 19 head 2), 114 (block 27 head 1)
and 128 (block 3 head 15); the first three broadcast J and the last three do not, and the strongest
broadcaster, block 19 head 0, was not a strong retrieval head. Entry to the workspace and retrieval
of far content therefore share heads at layers 20 and 24 without being one operation: half of the
strongest retrieval heads are lens relays, and the strongest lens relay is not a retrieval head.

**Prediction 2, confirmed.** The recurrent value paths' weight-only compositions preserve lens
labels barely above chance: median 0.0061 on J against 0.0043 rotated, 0.0095 MLP rows and 0.0045
random, thirty times below attention's 0.178, with gain medians at 0.98, 1.00 and 1.02. The rule
still counts 112 of 768 as exceeding both controls (36 of 128 in band), because tiny margins on tiny
numbers pass a sign test; the strongest recurrent path, block 16 head 10, reaches label preservation
0.048, a fourteenth of the strongest attention head's. With the stated omission of the conv, the
state and the gated norm, the weight-only recurrent value path is not a J-space relay. The two
channels' columns are not commensurable and are not read as if they were.

**What follows.** Stage 2's ablation set is the 24 attention heads, with the 12 in-band heads as the
primary set and layer-matched random heads as control, measured by recall@25 of the hosted-lens
readout and the next-token change rate under the threshold stated in the design. The Head's review
of this reading precedes it.

## Corrections on review (the Head and the Deputy, 03:30; adopted). The selection rule changes and the result above is amended by this section.

**Layer 32 is out, on our own ruling.** R43a puts layer 32 outside the lens family: the lens files
cover blocks 0 to 30 and the lens is the identity at layer 32 by construction, so label preservation
against J there measures whether a head preserves directions the unembedding can read, trivially
true next to the output. Its seven passing heads are that degeneracy and are excluded; N is 112.

**The gain criterion does not transfer, so the set is selected on preservation with gain reported
beside it.** Among the 112, 17 heads pass both metrics against both controls, 41 pass preservation
against both controls and fail only gain, and 14 pass gain only. The gain medians are 0.93 on J,
1.00 rotated and 0.96 MLP rows, flat across populations, so the paper's worse-of-two rule drops
strong relays for failing the half that does not discriminate here. **Selectivity lives in label
preservation, not amplification: a genuine difference from the paper, stated as its own line.** The
rule, with the Deputy's margin so that it carries the claim: a head relays the lens when its label
preservation on J is at least double the best control (`mrr_margin >= 1`); gain is reported beside
it. Under that rule: **28 of 112 attention heads, 23 of 64 in the band; by layer written, 4: 3, 8:
0, 12: 2, 16: 2, 20: 7, 24: 5, 28: 9.** The earlier "none at 16 or 28" was the gain criterion's
artefact; layer 28 holds the most. Among the 28 the gain median is 0.90 against 1.00 for the best
control and a quarter have gain above both controls. The ordered set with every number is in
`out/selection_preservation_rule.json`.

**The near-identity copy maps at layers 8 and 12 are caught by the margin, which is the control
working.** Rotated-J label preservation medians are 0.24 at layer 8 and 0.18 at layer 12 against
0.03 or less at every other layer: those heads preserve any direction set, and the margin against
the rotated control keeps them out (layer 8: 0 of 16 selected; layer 12: 2). The metric is not
rewarding direction-preserving maps as such.

**The rotated control is matched by construction.** It is one fixed random orthogonal rotation of
the whole residual space applied to the J population, which preserves the population's Gram matrix
and therefore its spectrum, pairwise geometry and effective dimensionality; it is the paper's
control exactly, not a rotation within a lower-dimensional subspace.

**The recurrent paths, stated as magnitude.** Under the sign rule 112 of 768 pass, which is what a
bare inequality on tiny numbers admits and is not a finding; under the margin rule 20 of 768; with an
absolute floor at the attention median, none. The recurrent median on J, 0.0061, loses to its own
MLP-row control, 0.0095, and the strongest recurrent path, 0.071, is below the attention median of
0.178: not one of the 768 reaches even the median attention head, let alone a relay. With the
stated omission of the conv, the state and the gated norm, the weight-only recurrent value path is
not a J-space relay.

**The retrieval heads, as six heads and not a rate.** Under the margin rule, block 19 head 12 (rank
6 by preservation margin, 0.366 against 0.039) and block 23 head 9 (rank 13, 0.575 against 0.136)
are selective lens relays; block 19 head 15 preserves at 0.040 against 0.024 (margin 0.68, below
the double) and is not; block 19 head 2 preserves at 0.988 but so does its rotated control at 0.994,
a copy map that relays everything and is not selective; block 27 head 1 and block 3 head 15 do not
relay at all. So of the six strongest retrieval heads two are selective relays, one is a copy head,
and three are not relays; the strongest relay, block 19 head 0, was not a strong retrieval head.
That supports the claim that entry and retrieval are not one operation and share heads; it does
not support a rate.

**Stage 2 set.** The 28, with the 23 in-band heads primary, layer-matched random controls,
recall@25 of the hosted-lens readout and the next-token change rate under the stated threshold.


## Re-scoring on the Head's third review, 03:50 (adopted; a null-distribution rerun follows and re-scores again)

**Prediction 1 changes verdict and the record says so.** Scored on the gain-and-preservation set
(24 heads, layer 32 in): confirmed, with the relays at layer 20 over 24 and none at 16 or 28. Scored
on the preservation-margin set (28 heads, layer 32 out): **refuted**. In band by layer written, 16
has two, 20 has seven, 24 has five and 28 has nine; the first half of the band holds nine and the
second half fourteen, and the single largest layer is the last in the band. The paper's finding that
broadcast heads concentrate in the first half of the workspace layers does not reproduce on this
model; that is a difference from the paper and a finding. R38(h) is recorded from this: a corrected
selection rule re-scores every prediction written against the old one, both scorings shown.

**The rotated control becomes a distribution.** One fixed orthogonal rotation is the right
construction but one realisation, so "at least double the best control" compared against a single
draw. Stage 1 is rerun with twenty independent rotations and twenty independent MLP-row samples per
block; the criterion becomes label preservation on J beyond every one of the forty draws and at
least double their medians, with layer 32 excluded and gain reported beside. The set, the layer
counts, the retrieval-head statuses and both predictions are re-scored on that run below.

**Stage 2 is graded, or it answers nothing.** Twenty-eight of 112 is a quarter of the attention
heads; ablating a quarter of attention and seeing the lens readout fall shows that attention matters,
which is not in doubt. The paper's finding was that a very small set carries the effect. Stage 2
therefore ablates the top k heads by preservation margin for k of 2, 4, 8, 16 and the full set, each
against a layer-matched random set of the same size (five seeds), and reports recall@25 of the
hosted-lens readout at the band layers and the next-token change rate as functions of k. If a small k
takes most of the recall drop while the random set of the same size does not, the paper's claim
reproduces; if the curve is flat in k and tracks the random control, the set is not special and the
ablation is measuring damage.

**The recurrent result, stated positively first.** The recurrent value paths carry lens-readable
content less well than they carry a control direction set: the median J preservation, 0.0061, is
below the median MLP-row control, 0.0095. Separately, the strongest of the 768, 0.071, is below the
median attention head, 0.178. Both statements stand on their own.

## Final stage-1 selection under null distributions, 04:20 (`broadcast_heads_nulldist.as-run.py.txt`, `make_selection.py` (renamed from `select.py`, which shadowed the standard library module and broke the model import chain in the first composition run), `out/selection_nulldist.json`; 241 s)

Twenty independent orthogonal rotations of the lens population and twenty independent samples of
2,000 MLP-row directions per block; the rule: label preservation on J beyond every one of the forty
draws and at least double their medians, layer 32 excluded, gain reported beside.

**29 of 112 attention heads; 24 of 64 in the band. By layer written: 4: 3, 8: 0, 12: 2, 16: 2, 20: 8,
24: 5, 28: 9.** The set differs from the single-draw set by one head (block 19 head 15 enters, at a
margin of 1.07 and beyond all forty draws); every other member is common. Among the 29 the gain
median is 0.91 and 28 percent have gain above both controls; the strongest heads by preservation are
block 11 head 10 (0.938 against a worst draw of 0.349), block 15 head 6 (0.733 against 0.248), block
19 head 0 (0.664 against 0.047), block 23 head 1 (0.655 against 0.289) and block 23 head 9 (0.575
against 0.134). Copy maps at layers 8 and 12 (rotated-J medians 0.26 and 0.19) stay out.

**Prediction 1, re-scored on the final set: refuted, as on the corrected single-draw set.** First half
of the band (16, 20): 10; second half (24, 28): 14; largest single layer 28. Scored three ways for the
record: confirmed on the gain-contaminated set, refuted on the preservation set, refuted on the
null-distribution set. The paper's first-half concentration does not reproduce on this model.

**Prediction 2, re-scored: confirmed.** Recurrent value paths: 23 of 768 pass the same rule on tiny
numbers; the median J preservation, 0.0061, is below the median MLP-row control, 0.0095, so the
paths carry lens-readable content less well than a control direction set; the strongest of 768,
0.071, is below the attention median, now 0.254 with layer 32 out, and none of the 768 reaches it.

**The six retrieval heads under the final rule.** Block 19 head 12 (rank 7; 0.366 against a worst
draw of 0.043) and block 23 head 9 (rank 13; 0.575 against 0.134) are selective relays; block 19 head
15 enters at the margin (rank 29; 0.040 against 0.026, double its medians and beyond all draws);
block 19 head 2 preserves at 0.988 with its rotated draws at 0.995 to 0.997, a copy map; block 27
head 1 and block 3 head 15 do not relay. Three of six are relays, one marginal; one is a copy head;
two are not. Six heads, no rate.

**What is fixed for stage 2.** The ordered set of 29 with the 24 in-band primary, ablated at k = 2,
4, 8, 16 and 29 against layer-matched random sets of the same size, five seeds, recall@25 and the
next-token change rate as functions of k, the drop counting only if the change rate stays within
twice the control's. The script `ablation.py` is written and reads this selection; it runs on the
Head's word.

## Stage 1b and stage 2 as reviewed, running from 04:50 (`composition.py`, `separation.py`, `ablation.py`; one chain, one model load at a time)

**Stage 1b, the Director's hypothesis (04:00).** Layers 4 and 8 are outside the band, so relays
there "must be coupling heads that couple to heads in later layers": composition in the
transformer-circuits sense. For each of the three layer-4 relays (block 3 heads 4, 10, 12) and each
of the 64 band heads, the Q-, K- and V-composition scores of Elhage et al., ‖W_X^B W_OV^A‖_F over the
factors' norms, against twenty rotations of the relay's OV map (Q W_OV Q^T, same spectrum, scrambled
directions) and against the population of every layer-4 and layer-8 head paired with every band
head. Predictions written before the rows: (1) the layer-4 relays K-compose with the band relays
above the null, with the retrieval heads of blocks 19 and 23 among the partners; (2) they preserve
token identity at least as well as lens directions, so their relay status is address-writing rather
than workspace content. The Head's correction before the rows: token identity and lens directions
share the unembedding, so a head preserving one tends to preserve the other; the separation control
therefore adds the lens direction with its projection on the same token's identity direction
removed, and reports the mean absolute cosine between the two, so the prediction is about the margin
over the shared component and the size of that component is visible.

**Stage 2, the Head's graded design (03:50) with two changes (04:35) and a calibrated gate
(04:45).** Ablate the top k heads by preservation margin for k = 2, 4, 8, 16 and all 29, zeroing each
head's attention output at every position before the output gate, against layer-matched random sets
of the same size (five seeds), on eight documents of 512 tokens with 64 sampled positions each.
Measurements: recall@25 of the hosted-lens readout, reported only at readout layers with at least
one ablated head writing at or below them, with the number of upstream ablated heads beside each
layer; the next-token change rate; both at every k with the control beside each, and the specificity
rule (a recall drop counts only if the change rate stays within twice the control's) applied at
interpretation, never to the data. Hard control: readout layers with no ablated head at or below
them cannot be affected, so their recall@25 must equal the reproducibility floor, measured first by
running the unablated readout twice; exact equality if that pass is bit-identical, the measured floor
otherwise, stated on the quantity it gates. If the curve in k is steep and the random control flat,
the paper's small-set claim reproduces; if it is flat in k and tracks the control, the 29 are not a
special set and the ablation measures damage.

## Stage 1b result, 05:05: the layer-4 relays do not compose with the band by weight, and they carry lens content, not addresses

**Composition (192 pairs, three layer-4 relays × 64 band heads; 38 s).** Q-, K- and V-composition
scores sit at the rotation null: median ratios to the null 1.00, 0.98 and 0.95; the raw scores
(0.019 to 0.021) are what a random map of the same spectrum gives; no pair exceeds the 99th
percentile of the population of every layer-4 and layer-8 head paired with the band (0 of 192 on
each of Q, K, V against about 1.9 by chance). The strongest single ratios are 1.06 (Q) and 1.01 (K,
V). Per relay, the share of band heads above the rotation null's maximum is 0 to 12 percent on K
and 0 to 6 percent on V; block 3 head 10 reaches 59 percent on Q, but at ratios of 1.05, which is
the noise floor of a twenty-draw maximum and not composition. **Prediction 1 of stage 1b is
refuted:** by weight-only composition scores, the layer-4 relays do not address the band relays or
the retrieval heads above what a random map does. Limitation stated: the Frobenius composition
score is a weight-only, direction-agnostic statistic and is known to be blunt; the causal test is
to patch a layer-4 relay's output on real text and read the band relays' attention and the lens
readout downstream, which stage 2's machinery can do and which is filed as the follow-up.

**Separation (the orthogonalised control the Head asked for).** For the three layer-4 relays, label
preservation is 0.100, 0.121 and 0.164 on the lens directions, 0.004, 0.007 and 0.017 on token
identity (at the random level of 0.005, 0.008, 0.023), and 0.099, 0.112 and 0.155 on the lens
directions with the identity component removed. The shared component is small at layer 4: mean
absolute cosine between a lens direction and its token's identity direction 0.101. **Prediction 2
of stage 1b is refuted the other way:** the layer-4 relays do not write addresses; they relay the
part of the lens direction that is not the embedding, which is workspace content in the lens's
sense, at a layer before the band. The thirteen layer-4 non-relays preserve the lens at a median of
0.107 too, with identity at 0.063, so at layer 4 the relay status is a margin over the null draws
and not a categorical difference. Two further numbers bear on the whole distinction: the shared
component grows with depth, 0.41 at layer 20 and 0.69 at layer 28, so by the top of the band lens
directions and token identity are largely the same directions and address-versus-content cannot be
drawn there from these populations, as the Head anticipated; and the in-band relays' preservation
survives the orthogonalisation (block 19 head 0: 0.664 on the lens, 0.663 with the identity
component removed), so they too relay content and not identity.

**Reading.** The out-of-band relays are lens-content relays at layer 4 whose coupling to the band,
if any, is not visible in weight composition. The Director's hypothesis stands as a causal question
for a patching run; it is not supported as a weight-space fact.

## Selection on the orthogonalised population, 06:10: layer 28's nine survive; prediction 1 refuted a third time, now robust to the cosine ramp

The same forty-draw nulls and the same rule on the lens directions with each token's identity
component removed, at every layer (`broadcast_heads_orth.as-run.py.txt`, `out/selection_orth.json`).
**29 of 112 attention heads; 24 of 64 in band; by layer 4: 3, 8: 0, 12: 2, 16: 2, 20: 7, 24: 6, 28: 9.**
The set differs from the lens-population set by one head (block 19 head 15 leaves at a margin of 0.88,
block 23 head 0 enters); the nine at block 27, layer 28, are the same nine heads (4, 5, 6, 8, 11,
12, 13, 14, 15). Median preservation by layer barely moves under orthogonalisation (layer 28: 0.098
to 0.105; layer 20: 0.445 to 0.402), so the layer-28 relays preserve the part of the lens direction
that is not the unembedding and were not an artefact of the ramp. **Prediction 1 scored a third
time: refuted**, first half of the band 9 against second half 15; the three scorings in the record
are confirmed on the gain-contaminated set, refuted on the lens-population set, refuted on the
orthogonalised set. The paper's first-half concentration does not reproduce on this model, on a
population from which the output direction has been removed.

**The cosine ramp in full (F10), by the block's input layer:** 0.09 at layer 1, 0.10 at 3, 0.17 at
8, 0.23 at 12, 0.29 at 16, 0.48 at 20, 0.63 at 24, 0.69 at 27, 0.71 at 28, 0.75 at 31: a smooth
ramp with no cliff, already 0.48 in the middle of the band.

**Layer 28 is saturated, and that is a finding to explain rather than a set to ablate (the Head).**
Nine of sixteen heads at one layer passing a selectivity rule that survives orthogonalisation is not
a small special set; it says a majority of the attention heads writing layer 28 relay workspace
content selectively. The retrieval heads under the orthogonalised rule: block 19 head 12 (rank 7,
0.39) and block 23 head 9 (rank 10, 0.67) selective relays; block 19 head 15 below the double at a
margin of 0.88; block 19 head 2 a copy map (0.989 against draws at the same level); block 27 head 1
and block 3 head 15 not relays.

**The recurrent channel on the orthogonalised population, with one block excluded.** Block 0's
input layer is the embedding itself, so the lens population there is the identity population and
orthogonalising it against the same identity directions leaves a degenerate population that scores
0.53 on every map, real or rotated, at a margin of zero; its 32 paths are excluded from the recurrent
summary and none was selected. With block 0 out: the median J preservation on the orthogonalised
population remains below the median MLP-row control, the strongest recurrent path remains below the
attention median, and the count at or above the attention median is as recorded in
`out/selection_orth.json` under `recurrent_excluding_block0`. The recurrent conclusion is unchanged
by orthogonalisation.

## Addendum, 2026-09-06 06:50: the band against the rest, layer 28's nulls, and an absolute floor

**The band is favoured over the rest of the model even though the shallow half is not favoured
within it (the Deputy's point).** On the orthogonalised rule, 24 of the 64 heads writing the band
pass (37.5%) against 5 of the 48 writing outside it (10.4%; layer 32 excluded from both), a ratio of
3.6. Prediction 1's refutation is about where inside the band the relays sit; the band itself is
where they sit.

**The Head's hypothesis that layer 28's nine were the layer-32 degeneracy in weaker form is
refuted**, as plainly as the prediction refutations: the same nine heads pass on the population with
the output direction removed, and their median preservation moves from 0.098 to 0.105. They relay
the part of the lens direction that is not the unembedding.

**Layer 28's nulls do not collapse (the Head's diagnostic, `out/layer28_diagnostic.json`).**
Medians over the sixteen heads writing each layer, on the orthogonalised population: real
preservation, rotated-null preservation, MLP-row-null preservation, and the per-head ratio of the
real to the better null:

| layer written | real | rotated null | MLP null | ratio | pass | pass, floor 0.1 |
|---|---|---|---|---|---|---|
| 4 | 0.106 | 0.025 | 0.092 | 0.84 | 3 | 2 |
| 8 | 0.601 | 0.254 | 0.446 | 0.94 | 0 | 0 |
| 12 | 0.605 | 0.182 | 0.403 | 1.00 | 2 | 2 |
| 16 | 0.290 | 0.034 | 0.148 | 0.99 | 2 | 2 |
| 20 | 0.402 | 0.006 | 0.042 | 1.86 | 7 | 6 |
| 24 | 0.244 | 0.012 | 0.093 | 1.82 | 6 | 6 |
| 28 | 0.105 | 0.012 | 0.036 | 2.70 | 9 | 7 |

Layer 28's nulls are not lower than layer 20's, so the margin rule is not easier there; its median
ratio is the highest in the model, and the ratio rises through the band while the absolute
preservation falls. The saturation is a property of the heads: a majority of the heads writing the
last in-band attention block relay workspace content weakly but selectively. Layer 8 is the mirror
case: the highest real median in the model and zero passes, because random MLP rows preserve that
population's labels almost as well (0.45), which is block 0's degeneracy in weaker form and is what
the null-distribution rule is for.

**The margin rule admits tiny-number passes, and an absolute floor is the fix.** Block 27 head 13
passes at a preservation of 0.002 over a null of 0.001; block 27 head 15 at 0.090, block 3 head 4 at
0.099, block 19 head 4 at 0.048. This is the recurrent channel's tiny-number problem in attention.
With a floor of 0.1 on absolute preservation, in addition to the null-distribution and double-median
clauses, the set is 25 of 112 (layer 4: 2, 12: 2, 16: 2, 20: 6, 24: 6, 28: 7). **Prediction 1
scored a fourth time under the floor: refuted**, first half 8 against second half 13; at floors of
0.05 and 0.2 it is 8 against 14 and 8 against 11, refuted at every floor. Layer 28 stays saturated
at 7 of 16.

**The ordering for the graded ablation must be by absolute preservation, not by margin.** Ordered
by margin, the first layer-28 head enters at k = 1 (block 27 head 14, preservation 0.12 over a null
of 0.006); ordered by absolute preservation it enters at k = 4 (block 27 head 6, 0.698), behind
block 23 head 0 (0.905), block 23 head 1 (0.726) and block 11 head 10 (0.903, out of band). The
small-k arms, which carry the paper's claim, then hold the strongest relays rather than the smallest
nulls. `ablation.py` takes `FLOOR` and `ORDER=abs` for this. With the floor the in-band set is 21
(16: 2, 20: 6, 24: 6, 28: 7) and the exclude control draws from 14, 10, 10 and 9 unselected heads
per layer: exact at every layer, no forced overlap. The proposed run is
`SET=inband SEL_FILE=selection_orth.json FLOOR=0.1 ORDER=abs CONTROL=exclude K_LIST=2,4,8,16 NSEED=5`,
which ablates k = 2, 4, 8, 16, 21. It waits on the Head.

**What layer 28's nine share that its seven do not.** The nine relays (heads 4, 5, 6, 8, 11, 12,
13, 14, 15) and the seven non-relays (0, 1, 2, 3, 7, 9, 10) have identical gain (medians 0.847
against 0.845), similar retrieval behaviour on the retrieval-channels run (answer-span ratio medians
7.3 against 7.0, hit@10 0.69 against 0.60; head 3, a non-relay, is the layer's strongest retriever at
a ratio of 40 and hit@10 1.0), and the seven sink slightly more (first-5% mass 0.338 against 0.277).
They differ in lens preservation (0.317 against 0.010, the selection) and in key-value group: with
sixteen query heads over four key-value heads, the nine belong to groups 1, 1, 1, 2, 2, 3, 3, 3, 3
and the seven to 0, 0, 0, 0, 1, 2, 2. All four query heads of key-value group 0 are non-relays and
all four of group 3 are relays. Since the OV map is the product of the head's output slice and its
group's value projection, the clustering is structural: group 3's value projection carries the lens
directions and group 0's does not. The Head's reading, that block 27 is the last attention block
inside the band and the last chance to write into the workspace before the motor layers, is
consistent with the ratio gradient and is untested.

The gain medians above (0.847 against 0.845) are on the lens population; on the orthogonalised
population they are 0.962 against 0.990. On neither do the nine differ from the seven, and on both
they sit below the rotated null near 1.0, which is the recorded failure of the gain criterion on this
model.

**Correction, 06:58 (the Deputy): the band's attention members are five layers, not four.** Under
R41e the band is the pairs 12/13, 16/17, 19/20, 23/24, 27/28, whose attention members are layers
12, 16, 20, 24 and 28, written by blocks 11, 15, 19, 23 and 27: eighty in-band query heads. The
06:50 addendum computed "in band" on four layers and put layer 12 outside, the fourth time tonight a
list built by extending the obvious rule dropped pair 12/13. Recomputed on the recorded band: 26 of
80 in-band heads pass (32.5%) against 3 of 32 outside (9.4%), a ratio of 3.5; the conclusion does
not move. The ablation script carried the same four-layer list into its in-band set and its readout
layers, and was corrected before the run: it now asserts the R41e pairs as the recorded ruling
(R38(f)), derives the attention members from the model's layer kinds, ablates from all five blocks,
and reads recall@25 at all ten band layers.

**Prediction 1 scored a fifth time, on the five-layer band, as rates per head** because the halves
are uneven (three attention layers against two): first half, layers 12, 16 and 20, 48 heads; second
half, 24 and 28, 32 heads. Passing heads first against second, as counts and as rates: no floor 11
of 48 (0.23) against 15 of 32 (0.47); floor 0.05, 10 (0.21) against 14 (0.44); floor 0.1, 10 (0.21)
against 13 (0.41); floor 0.2, 10 (0.21) against 11 (0.34). Refuted at every floor by rate; at floor
0.2 the raw counts nearly tie, which is the uneven halves and not the heads. With the middle layer
20 left out of both halves, 12 and 16 against 24 and 28 give 4 of 32 against 11 to 15 of 32 at
every floor.

**The floor is post-hoc, and the exact control follows from it (the Head, 06:55).** The floor of
0.1 was chosen after seeing the data, and the set it produces is one where the exclude control needs
no forced overlap; a convenience criterion and a defensible one arrive at the same place. The
sensitivity across floors is therefore the primary result. On the five-layer band the in-band set is
21, 23, 24 and 26 heads at floors 0.2, 0.1, 0.05 and none; the exclude control is exact at floors 0.2
and 0.1, exact but deterministic at 0.05 (layer 28 has eight selected and eight unselected, so every
seed draws the same eight), and forced at no floor (layer 28 has nine selected and seven unselected,
overlap two). Because the floor only trims the tail of an ordering by absolute preservation, the
arms k = 2, 4, 8 and 16 are the same at every floor and the four floor sets are the arms k = 21, 23,
24 and 26 of one run; that run is `SET=inband SEL_FILE=selection_orth.json FLOOR=0 ORDER=abs
CONTROL=exclude K_LIST=2,4,8,16,21,23,24 NSEED=5`, launched at 07:00 under the standing lift, one
model load. The first layer-28 head enters at k = 4 (block 27 head 6, 0.698), behind block 23 head
0 (0.905), block 11 head 10 (0.903) and block 23 head 1 (0.726). The reading will say which claim the
curve answers: the gate (margin rule and floor) identifies the relays, the ordering by absolute
preservation puts the heads carrying the most lens content first, so the small-k arms test whether
the heads that carry the most lens content carry the effect, which is a different claim from whether
every head the rule identified carries it; the k = 21 to 26 arms answer the second.

**Pre-registration, 07:20, before the key-value group test runs (the Head's statistic).** The
membership pattern is no evidence either way: relays per group of four at layers 20, 24 and 28 are
1, 4, 0, 2; 2, 3, 1, 0; 0, 3, 2, 4 (orthogonalised rule, no floor), with exact enumeration giving
p = 0.055, 0.33 and 0.055 for the dispersion of the four counts and a Fisher combination of 0.032;
under the floor of 0.1 the same combination is 0.34. Evidence that moves by a factor of ten under a
rule change made for unrelated reasons is not weak evidence, it is none, and the weight test is the
only test there has been of the group hypothesis. Its primary statistic, fixed before it runs: per
in-band block, over the sixteen real heads, the ratio of between-group to within-group variance of
lens preservation (one-way F over four groups of four), against the same ratio computed on each of
the forty null populations, twenty rotated-lens and twenty MLP-row draws, under the same real heads.
The property lives in the shared value projection at a block if F on the lens population exceeds F on
every one of the forty draws; the result is the number of such blocks among five. The null is the
group structure the real heads impose on an arbitrary population, so passing means the structure is
specific to the lens directions. Secondary and reported beside it: A, each value projection alone as
the row-space map through its transpose, and the energy it captures from the lens population against
the draws; B, the 16 by 4 cross table of output slices against value projections scored by the
stage-1 rule, in which the diagonal is real and the other 48 cells are maps the model never
computes, a factorial probe of where the property sits and not 64 head measurements. The script is
`kvgroup.py`; it runs when the ablation releases the model. Two notes carried from the Head on the
ablation's reading: at floor 0.05 the layer-28 control has exactly one possible set, so that arm's
control variance is stated as zero rather than as a small number; and the floor sensitivity
collapsing into three extra arms of one run, because the floor only trims the tail of an absolute
ordering, is the design that turned a cost into arms.

## Stage 2 result, 07:35: the top two relays do the damage of nine random heads, and the factor falls with k

Run as pre-registered above (`ablation.as-run.py.txt`, `run-ablation-floors.log`,
`out/ablation_inband_floors.json`): the 26 in-band relays of the orthogonalised rule on the five-layer
band, ordered by absolute lens preservation, zeroed k at a time; five layer-matched random sets of
the same size from the unselected heads at the same layers; readout at all ten band layers on eight
512-token documents at 64 positions; bit-identical reproducibility of the unablated readout.

**The metric is top-25 overlap with the unablated readout**, the share of the unablated lens
readout's top-25 tokens that survive at readout layers with an ablated head at or below them. It is
1.0 at k = 0 by construction and measures damage to the model's prior output, not recall of anything
external. The paper's recall at 25 scored recovery of an externally known target and starts at 0.86;
it is not the same quantity and the two are not compared.

| k | arm | overlap, relays | overlap, random (range over 5 seeds) | change rate, relays / random | specificity (within 2×) | random heads for the same damage |
|---|---|---|---|---|---|---|
| 0 | unablated | 1.000 | 1.000 | 0 / 0 | | |
| 2 | | 0.832 | 0.893 (0.841 to 0.928) | 0.059 / 0.037 | holds | 9.0 (4.5×), between the arms at 8 and 16 |
| 4 | | 0.824 | 0.874 (0.825 to 0.923) | 0.084 / 0.054 | holds | 10.1 (2.5×), arms 8 to 16 |
| 8 | | 0.803 | 0.840 (0.824 to 0.864) | 0.098 / 0.073 | holds | 13.3 (1.7×), arms 8 to 16 |
| 16 | | 0.722 | 0.789 (0.762 to 0.809) | 0.156 / 0.100 | holds | 23.5 (1.5×), arms 23 to 24 |
| 21 | floor 0.2 | 0.686 | 0.752 (0.725 to 0.792) | 0.178 / 0.118 | holds | beyond 26 |
| 23 | floor 0.1 | 0.678 | 0.727 (0.683 to 0.761) | 0.209 / 0.127 | holds | beyond 26 |
| 24 | floor 0.05 | 0.675 | 0.717 (0.691 to 0.740) | 0.223 / 0.134 | holds | beyond 26 |
| 26 | no floor | 0.666 | 0.727 (0.673 to 0.790) | 0.271 / 0.133 | fails, 2.04× | beyond 26 |

Equivalent random heads are interpolated linearly in log2(k + 1) between the two measured random
arms that bracket the selected value, named beside each figure because the transform does real
work. "Beyond 26" means the random curve's lowest measured value, 0.727 at 26 heads, is still above
the selected value, so the equivalent is a bound: the twenty-one relays at floor 0.2 do more damage
than twenty-six arbitrary heads at the same layers, and so do the sixteen strongest (0.722 against
0.727 at 26, a bound by 0.005). The control at k = 24 has one possible set at layer 28 (eight
selected, eight unselected), so its variance there is zero; the k = 26 arm has a forced overlap of
two at layer 28 in every seed and is the contaminated arm.

**Reading.** (1) The paper's claim reproduces at the top of the ordering: two heads, block 23 head 0
and block 11 head 10, do the damage of nine random heads at the same layers, a concentration of 4.5.
(2) Random ablation of any two heads already costs ten points, and the selection adds six on top;
the ten points of general damage is a fact about this measurement and stands beside the increment.
(3) The concentration lives in the top two to four heads, not in the set the rule produced: the
factor runs 4.5, 2.5, 1.7, 1.5 over k = 2, 4, 8, 16, and the tail arms from 21 to 26 add 0.020 of
damage where random adds 0.025. The rule identifies a population within which a few heads matter;
it does not identify twenty-six heads that matter (the Head). (4) The falling factor is not decay:
the ordering is by absolute preservation, so a factor that falls monotonically with k is evidence
that preservation ranks heads by causal importance, a validated ranking beneath the concentration,
which means the stage-1 weight arithmetic predicts what ablation does without running the model.
(5) Per step, selected drop against random drop: 2: 0.168 against 0.107; 4: 0.009 against 0.019;
8: 0.020 against 0.034; 16: 0.082 against 0.051; 21: 0.036 against 0.037; 23: 0.008 against 0.025;
24: 0.003 against 0.010; 26: 0.009 against a control gain of 0.010. Heads three and four (block 23
head 1, block 27 head 6) did less than two random heads, so the gap narrowed at k = 4 because the
control caught up; heads nine to sixteen, which include block 27 heads 5, 8 and 12, did more than
random again; below rank 21 the selection is ordinary or weaker. The one-head datum on block 27
head 6 at k = 4 does not generalise to the layer-28 members as a class. (6) Specificity holds at
every floored arm: at k = 21 the selection changes the next token at 0.178 of positions against the
random sets' 0.118 while damaging the readout more; breaking the model would move the output far
more than a control that damages the readout less, so the overlap damage is not the model being
broken. It fails only at the no-floor arm (2.04 times), which is also the arm with forced overlaps.
(7) By readout layer at k = 26, relays against random: 12: 0.903 against 0.894; 13: 0.899 against
0.885; 16: 0.811 against 0.770; 17: 0.799 against 0.766; 19: 0.755 against 0.736; 20: 0.572 against
0.681; 23: 0.581 against 0.696; 24: 0.482 against 0.641; 27: 0.482 against 0.643; 28: 0.373 against
0.560. The damage the relays do beyond random grows with depth and is largest at the deepest band
layers, which is where the lens converges on the output (F10). (8) The hard upstream gate had no
layer to fire on in any arm, because block 11 head 10 enters at k = 2 and layer 12 is the lowest
readout; a single-head arm on block 23 head 0, with layers 12 to 20 upstream and required
bit-identical, restores it and is running as the next load (the Head). (9) A process note: the
first attempt to write this section chained the record path behind a page build that failed, so the
copies went to the root and were refused and the append never reached the file; nothing was lost,
and it is the R49(f) pattern, caught by its own failure.

**Correction, 07:50 (the Head, withdrawing a claim made at 07:30; the Chief had adopted it in
reading (4) above, which is withdrawn with it).** The cumulative equivalence factor falls whenever
later heads add less than the running average, so 4.5, 2.5, 1.7, 1.5 is arithmetic and not a test
of the ordering. The per-step contributions are the test. Per head ablated, the selection adds
0.084, 0.0043, 0.0051, 0.0102, 0.0072, 0.0041, 0.0031, 0.0044 over the steps to k = 2, 4, 8, 16,
21, 23, 24, 26 (the selected curve is deterministic, no seeds); ranks three to eight add about half
of what ranks nine to sixteen add, so the ordering is not monotone in effect. Against random per
head: 0.053, 0.0095, 0.0085, 0.0064, 0.0074, 0.0127, 0.0098, and a gain of 0.005, ratios 1.57,
0.45, 0.60, 1.59, 0.97, 0.32, 0.31 and negative. The random per-step values are differences of
independent five-seed means; the per-head standard deviation of the random step is 0.019, 0.026,
0.010, 0.003, 0.006, 0.021, 0.041, 0.025 at those steps, so the ratios at the 2-to-4 step and past
21 are within their own noise, 4-to-8 is marginal, and 8-to-16 (1.59 at a step deviation of 0.003)
is solid. The supportable statement, adopted in place of readings (3) and (4): a small number of
heads carry a disproportionate share; they are not contiguous in the preservation ordering; the
metric identifies a population containing them without ordering them correctly within it. By name,
ranks one and two are block 23 head 0 and block 11 head 10; ranks three to eight are block 23 heads
1 and 9, block 27 heads 6 and 4, block 19 head 0 and block 15 head 6; ranks nine to sixteen are
block 27 heads 5, 8 and 12, block 19 heads 7, 12 and 6, block 11 head 15 and block 23 head 6. Layer
28 has two heads in the weak stretch and three in the strong one, so neither of the two layer-28
data points generalises to that layer's members as a class. The question is settled by single-head
ablation of every in-band head (`singles.py`, 80 arms, one load, the next run): per head the overlap
at every readout layer, at layer 28 which is downstream of all of them, and the change rate; the
rank correlation of preservation with damage over the 80 and within the 26; selected against
unselected at each layer.

**The depth gradient normalised (the Head).** At k = 16 the gap between random and relays per
readout layer is 0.031, 0.034, 0.041, 0.099, 0.100, 0.120, 0.101, 0.151 at layers 16, 17, 19, 20,
23, 24, 27, 28, with 3, 3, 3, 7, 7, 11, 11, 16 ablated heads upstream: per upstream ablated head
0.010, 0.011, 0.014, 0.014, 0.014, 0.011, 0.009, 0.009, flat with depth; at layers 12 and 13, with
only the two layer-12 relays upstream, the gap is zero (0.006 in random's favour and 0.005). The
gradient is arithmetic of how many ablated heads lie upstream; the per-head excess is about 0.01 at
every depth past 13. At k = 26 the gap is in random's favour at every readout from 12 to 19 (0.009
to 0.041), because the tail heads at layers 12, 16 and 20 do less than random at the shallow
readouts, and in the relays' favour from 20 on (0.109 to 0.187, 0.007 to 0.010 per upstream head).
The five-seed range of the random sets is in the table above at every arm.

## The key-value group test, 07:58: the group structure is lens-specific at three of five blocks, and the property lives in the head's own output slice paired with its group's value projection

Pre-registered statistic (`kvgroup.as-run.py.txt`, `run-kvgroup.log`, `out/kvgroup.json`; 104
seconds, weights only): per in-band block, over the sixteen real heads, the ratio of between-group
to within-group variance of lens preservation, against the same ratio on each of forty null
populations under the same heads.

| layer written | F, lens | null median (rotated, MLP) | null max | draws at or above | beyond all draws | group means of preservation, 0 to 3 |
|---|---|---|---|---|---|---|
| 12 | 1.14 | 0.65 (0.70, 0.61) | 0.78 | 0 of 40 | yes | 0.68, 0.74, 0.45, 0.22 |
| 16 | 0.65 | 0.74 (0.82, 0.65) | 0.85 | 31 of 40 | no | 0.50, 0.58, 0.49, 0.17 |
| 20 | 2.45 | 2.50 (2.52, 2.46) | 2.74 | 31 of 40 | no | 0.91, 0.26, 0.49, 0.41 |
| 24 | 3.12 | 2.00 (1.79, 2.30) | 2.35 | 0 of 40 | yes | 0.66, 0.25, 0.56, 0.09 |
| 28 | 2.83 | 1.31 (1.41, 1.27) | 1.81 | 0 of 40 | yes | 0.07, 0.49, 0.21, 0.15 |

**Result: 3 of 5 blocks.** At layers 12, 24 and 28 the heads sharing a value projection preserve
the lens alike beyond what the same group structure does to arbitrary populations; at layer 28,
where the membership pattern was first seen, F on the lens is 2.83 against a null maximum of 1.81,
and group 0's four heads preserve at a mean of 0.07 against group 1's 0.49. At layer 20 the group
structure is as strong on the nulls as on the lens (2.45 against a median of 2.50): group 0 there
holds three near-identity copy maps (0.99) that preserve any population, which is group structure
without lens specificity. At layer 16 there is no group structure on either.

**Where the property lives, from the cross table (secondary, B).** Of the 320 cells, 64 per block,
only diagonal cells pass the stage-1 rule: every passing head passes with its own group's value
projection and with no other, and the off-diagonal cells preserve at a median of 0.004 at every
block. An output slice composed with a foreign value projection relays nothing, so the property is
not in the value projection alone; a value projection composed with a foreign output slice relays
nothing, so it is not in the output slice alone. It is in the trained pair: the shared value
projection constrains which lens directions a group's heads can read, which is the group structure
the primary statistic detects, and the head's own output slice determines whether they are written
back into the lens directions, which is why membership alone could not settle it. The 48
off-diagonal cells per block are maps the model never computes.

**Test A is degenerate as designed and is recorded as such.** A value projection alone as the
row-space map through its transpose preserves labels at 1.0 for the lens and for every null draw,
because a symmetric projection onto any 128-dimensional subspace keeps each vector nearer its own
image than any other's. Its energy half is reported beside: the share of a population's energy inside
the projection's row space (isotropic expectation 0.05) exceeds every null draw for group 0 at layer
20 (0.099 against 0.054, the copy-map group), for groups 2 and 3 at 20, group 2 at 24, and groups 0
and 3 at 28; at layer 28 group 1 captures 0.150 of the lens energy but rotated draws reach 0.166
there, so it is not beyond the null. The energy measure is not the pre-registered statistic and is
not read as one.

**The gate arm and the replicate (`out/ablation_k1.json`, `run-ablation-k1.log`).** Block 23 head
0 alone, writing layer 24: the readouts at layers 12, 13, 16, 17, 19, 20 and 23 are exactly 1.0,
bit-identical to the unablated readout, so the intervention does not reach where it cannot reach.
Downstream, at layers 24, 27 and 28, the overlap is 0.917 against 0.942 for a random layer-24 head
(0.929 to 0.945 over five seeds), change rate 0.016 against 0.021; this arm reads over three layers
where the curve reads over ten, so it is reported beside the curve and not on it. The run's final
arm replicated the 26-head arm under the same seeds: relays 0.666 and 0.666, random sets 0.727 and
0.727, change 0.271 and 0.271, identical to the last digit. Under the same seeds that is a check of
determinism, not a variance estimate; the variance across seeds is the range in the stage-2 table.

**Next load, launched 08:00: the single-head sweep** (`singles.as-run.py.txt`, 80 arms), which
tests whether preservation orders heads by effect and gives the per-head table.

**Pre-registered before the single-head sweep lands, 08:05 (the Head).** (a) The primary evidence
on whether preservation orders heads by effect is the sweep, not the curve: eighty single-head
damages with no differencing of means and no cumulative confound; the statistic is the Spearman
rank correlation of preservation with single-head damage over the eighty in-band heads, and within
the 26 relays. (b) Redundancy against synergy: for k = 2, 4, 8 and 16, the sum of the top-k heads'
single-head damages, each taken as one minus the mean overlap over all ten readout layers so that
every head is scored on the same layers, is compared with the measured k-arm's damage on the same
ten layers. If the sum exceeds the arm, the heads are redundant, carrying the same content by
different routes, and the dip at ranks three to eight in the cumulative curve is explained without
any fault in the ordering: ranks one and two already removed what those heads carry. If the sum
falls short, the heads are synergistic. The two experiments together distinguish a bad ordering
from redundancy where neither can alone. (c) The energy half of test A at layer 20, group 0 (0.099
against a null maximum of 0.054): a near-identity output map on a population requires the value
projection to capture that population's energy, so the energy excess and the copy-map property of
that group's heads are one fact seen twice, confounded by construction; the energy reading cannot
separate them and is not read as evidence about relaying at that block.

**Corrections and a pre-registration, 08:12 (the Head).** (a) Attribution: the null that made
layer 20 a clean negative was the Chief's refinement, not the Head's specification. The Head asked
for the between-group over within-group variance ratio against the forty draws; the Chief set it as
the F on the lens against the same F computed on each null population under the same real heads,
because a shared projection imposes group structure on any population, and the Head said at the
time it was the better version. (b) The asymmetric reading above, that the value projection
constrains and the output slice determines, is not a reading of the cross table alone and is
re-attributed: the cross table is symmetric in what it shows, swapping either component destroys
the property, which says both matter and not what each does; the asymmetry comes from the
combination of the F test, where heads sharing a value projection behave alike at three blocks,
giving the projection a group-level effect, and the diagonal, which gives the output slice a
head-level one. Two tests, one inference. (c) The off-diagonal half of the cross table is weaker
evidence than it looks: an output slice composed with a value projection it was never trained with
composes to noise generically, so a median of 0.004 is near what any network would give and says
nothing specific about the lens. The finding rests on the diagonal, and the diagonal needs a control
that preserves scale and structure while destroying only the pairing. Pre-registered, to run when the
sweep releases the model: each head's output slice scored against twenty rotated versions of its own
group's value projection, the rotation applied in the 128-dimensional head space between them, so
that the directions read (the projection's row space) and the directions written (the slice's column
space) are kept and only their correspondence is scrambled. Reading: the trained pairing is the
property at a head if its lens preservation exceeds all twenty pairing rotations; the result is the
count among the 26 relays. If the relays do not beat it, the table shows only that trained matrices
work with their own partners and not with strangers, which is true of any network. (d) The stage-2
table's caption: the selected arm has no seed variance by construction, the heads and documents being
fixed, so the range column is the whole variance estimate, and a deterministic row is not a
converged one. (e) The Head's synthesis, recorded as R53 in the wiring map: near-identity copy maps
confounded four measurements tonight, and every statistic built on preservation carries a copy-map
control by default.

**Pre-registration amended before the sweep lands, 08:16 (the Head): the redundancy null is
multiplicative, not additive.** Damage is one minus an overlap and is bounded by one, but a sum of k
damages is not: at the 0.084 per head the curve implies, the sum reaches 1.34 at k = 16 while the
arm cannot exceed 1.0, so the additive rule would report redundancy from arithmetic at the k where
the question matters. Overlap is a surviving fraction, so the independence null is the product of the
top-k single-head overlaps, each over all ten readout layers; the measured arm's ten-layer overlap
above the product reads as redundant (the heads damage the same content), below it as synergistic.
At k = 2 the two constructions agree to a thousandth, which is why the defect was invisible where it
could be checked. The Spearman is reported over all eighty, within the 26 relays and within the 54
unselected heads separately: over the eighty it is high partly because the selection separates
high-damage from low-damage heads, which is already known; if preservation orders the unselected
too, it is a general ranking of causal importance; if only the selected, the threshold is doing the
work.

**Correction, 08:22 (the Head's check of test D's dimension).** The attention head dimension in this
model is 256, not 128; 128 is the recurrent path's key and value head dimension. `kvgroup.py` reads
the dimension from the model (`a.head_dim`) and its rotations were always 256 by 256, so the code
was right and the description above ("the 128-dimensional head space") was the slip; the script now
asserts the projection shapes against the model's head count and dimension so a wrong dimension
raises rather than rotating a subspace silently. The same slip reached one number in the energy
reading: a d-dimensional row space captures d/H of an isotropic direction, which is 256/2560 = 0.10,
not 0.05. The energy comparisons above are against the rotated and MLP-row draws and stand as
written; the phrase "isotropic expectation 0.05" does not, and group 0's 0.099 at layer 20 sits at
the isotropic level rather than twice it, which does not change the confound reading. For the
pairing control, each head passes by chance at one in twenty-one, so about 1.2 of the 26 relays
pass by chance; the count is reported against that expectation.

**Correction, 08:32 (the Head's check of the energy nulls against isotropy).** The draws' median
energy sits well below 0.10 at every block (rotated medians 0.033 to 0.074 at layers 12 to 24 and
0.080 to 0.161 at 28, varying by group), and the reason is the measure, not the null. The quantity
the first run computed is the quadratic form v^T Wv^T Wv v over unit vectors, and its isotropic
expectation is the squared Frobenius norm of the value projection over the width, not d over H,
because the projection's rows are not orthonormal; the four groups' projections differ in scale,
which is why the rotated medians differ by group. The rotated draws, which scatter isotropically in
the residual space, estimate exactly that expectation, so the null was neutral and "isotropic 0.10"
was the wrong expectation for that form. On the corrected expectation, group 0 at layer 20 (0.099
against its rotated median of 0.051) is a genuine excess of about two, so the confound reading
stands; at layer 28 group 1's 0.150 sits below its own rotated median of 0.161 and is no excess.
The rerun adds the scale-free form, the share of energy inside the projection's row space through
the orthogonal projector, whose isotropic expectation is d over H exactly (0.10 here), and reports
both forms with both expectations beside the draws; a self-test on a random projection returned the
share at d over H and the quadratic form at the Frobenius expectation.

**Which energy form carries which claim, 08:40 (the Head, conceding the null was neutral).** For a
unit vector uniform on the sphere the expectation of the quadratic form is the trace of the map's
Gram matrix over the width, the squared Frobenius norm over the width; d over H is the special case
of an orthogonal projector, which the value projection is not. The group hypothesis is an alignment
claim, and the quadratic form conflates alignment with the map's scale, which differs across the
four groups, so any cross-group statement ("group 0 captures more than group 1") rests on the
scale-free share through the orthogonal projector, whose isotropic expectation is d over H, and no
such comparison has yet been made on a statistic that supports it. The quadratic form is read only
within a group against its own rotated null, where scale cancels: group 0 at layer 20 (0.099
against 0.051) stands as an excess of about two, and group 1 at layer 28 (0.150 against 0.161) as
a clean null. The rerun reports both forms with both expectations per block and group.

## The single-head sweep, 08:45: preservation does not predict a head's causal effect, the relays are not individually stronger than the rest at any layer, and the relay set is strongly redundant

Every one of the 80 in-band heads zeroed alone (`singles.as-run.py.txt`, `run-singles.log`,
`out/singles.json`; 21 minutes, one load; bit-identical reproducibility; the readouts upstream of
every head exactly 1.0 for all 80, so the gate held eighty times). Damage is one minus the top-25
overlap with the unablated readout at layer 28, the one readout downstream of every head; the
ten-layer mean is used for the products.

**Pre-registered Spearman of preservation against single-head damage at layer 28:** over the 80,
rho = −0.17 (p = 0.13); within the 26 relays, 0.09 (p = 0.65); within the 54 unselected heads,
−0.31 (p = 0.022). Preservation does not order heads by effect anywhere, and among the unselected
it runs the wrong way. **Relays against the rest:** median damage 0.061 against 0.075, one-sided
Mann-Whitney p = 0.97; by layer (relays against unselected, medians): 12: 0.109 against 0.091;
16: 0.061 against 0.091; 20: 0.063 against 0.061; 24: 0.052 against 0.055; 28: 0.067 against 0.061.
At no layer are the relays stronger as a class. The ten strongest single heads in the band by this
measure are block 11 head 8 (0.218, preservation 0.899, unselected because it also preserves the
null draws, a copy map), block 11 head 10 (0.161, relay rank 2), block 23 head 14 (0.159,
preservation 0.001), block 11 heads 3, 9, 4 and 11 (0.14 to 0.12, preservation 0.001), block 15
head 5 (0.138, 0.001), block 27 head 5 (0.129, relay rank 9) and block 19 head 8 (0.122, 0.001).
Seven of the ten preserve no lens directions at all.

**Pre-registered redundancy test, multiplicative null:** measured k-arm overlap against the
product of the top-k single-head overlaps, ten layers: k = 2, 0.832 against 0.817; k = 4, 0.824
against 0.796; k = 8, 0.803 against 0.713; k = 16, 0.722 against 0.538. Measured above the product
at every k and increasingly so: **redundant**, the heads damage the same content, and the dip at
ranks three to eight in the cumulative curve is redundancy, not an ordering that puts weak heads
early; rank 4 (block 27 head 6) does 0.082 alone at layer 28, rank 8 (block 27 head 4) 0.112, rank
9 (block 27 head 5) 0.129, while adding almost nothing to the cumulative curve.

**The reading of stage 2 changes.** (1) The set-level result stands as measured: the top relays
zeroed together damage the readout more than layer-matched random sets, and the sets are redundant.
(2) But the set effect is not lens-specific transport by individually special heads. Rank 2, block
11 head 10, does 0.162 alone over ten layers, which is above the median of its layer but below block
11 head 8 and level with heads there that preserve nothing; the cumulative curve's small-k advantage
is that one strong layer-12 head, counted at all ten readouts, plus redundancy among the rest. (3)
Single-head damage to the lens readout is dominated by a head's general influence on the residual,
not by whether its output map preserves lens directions: the heads that preserve nothing damage the
readout as much as the relays. The Head's warning stands in its strongest form: this metric is
damage to the model's prior output, and it is not the paper's recall of an injected concept. (4)
So the paper's causal claim, that the heads picked out by lens preservation are the ones that carry
workspace content, is not supported at single-head resolution on this model, and the weight
arithmetic of stage 1 does not predict what ablation does. What stage 1 finds is real (selective
preservation, band-favoured, group-structured, surviving orthogonalisation), and what it finds is
not causal importance for the readout. (5) The lens-specific causal test the paper actually ran,
injection of a concept along a lens direction at one layer with its recall measured downstream with
and without the relays, is the one experiment that could still separate "relays carry workspace
content" from "relays preserve directions that anything can carry"; it is filed as the follow-up
and not run tonight, because the Director's overnight order takes the model once this thread is
closed.

## The group rerun, 08:50: the pairing control is beaten by nearly every head, relay or not, and the value projections do not carry the lens directions differentially

Rerun with the pairing control and both energy forms (`kvgroup.as-run.py.txt`, `run-kvgroup-2.log`,
`out/kvgroup.json`, 131 seconds). The pre-registered F values are unchanged (12: 1.14 beyond a
maximum of 0.78; 16: 0.65 against 0.85, no; 20: 2.45 against 2.74, no; 24: 3.12 against 2.37;
28: 2.83 against 1.81; three of five).

**Pairing control.** 25 of the 26 relays beat all twenty pairing rotations, against a chance
expectation of 1.2; so do 36 of the 54 non-relays (12: 8 of 14; 16: 9 of 14; 20: 8 of 9; 24: 8 of
10; 28: 3 of 7). The scrambled maps preserve at 0.004 to 0.006, chance for a population of 2,000,
and the real maps of most heads preserve far above it. So the control is passed by trained heads
generally and does not single out the relays: what it shows is that a head's output slice works
with its own value projection and not with a scrambled correspondence, which the Head said in
advance is true of any network and not about the lens. The reading "the property lives in the
trained pair" is correct and empty; the relay-specific evidence remains stage 1's selectivity
against rotated and MLP-row draws, and the group structure of preservation at three of five blocks.

**Energy, scale-free share (isotropic 0.10 exactly, the rotated medians at 0.099 to 0.101
everywhere, so the null is neutral by construction).** Lens share by group 0 to 3: layer 12:
0.081, 0.084, 0.092, 0.069; 16: 0.074, 0.076, 0.066, 0.082; 20: 0.217, 0.090, 0.110, 0.122; 24:
0.104, 0.101, 0.114, 0.093; 28: 0.105, 0.091, 0.093, 0.115. The one cross-group excess is layer
20's group 0 at 2.2 times isotropic, the copy-map group, which is the confound seen twice. At layer
28 the group holding all four relays (group 3, 0.115) and the group holding none (group 0, 0.105)
are indistinguishable, so the value projections there do not carry the lens directions
differentially, and the group structure of preservation at layer 28 (F 2.83 beyond every draw) has
no mechanism at the level of value-projection alignment with the lens. At layers 12 and 16 every
group captures less than isotropic (0.07 to 0.09): the lens directions are slightly
under-represented in the value subspaces there. The quadratic form agrees with its Frobenius
expectation within a few percent at every group except layer 20's group 0, as the corrected
account said it would.

## Pre-registered before it runs, 09:00: the injection test, ruled necessary to close (the Head)

Strong redundancy is the signature of a redundantly encoded quantity: heads that damage the same
content carry the same content, removing any one changes little because the others still write
it, and the single-head null is what that signature predicts. So the sweep and the redundancy
result are ambiguous between two opposite conclusions, the relays not being special against the
relays redundantly carrying the same workspace content, and nothing measured so far separates
them. The injection test does (`inject.py`). A lens direction for a concept token, the direction
the layer-12 lens reads as that token, is added to the residual stream at layer 12, after block 11
so that the in-band relays at blocks 15, 19, 23 and 27 are downstream, at eight positions per
document spaced 56 apart, sixteen concept tokens over 64 events; the lens is read downstream for
recall@25 of the injected concept at the same position at every deeper band layer and at later
positions (+1, +2, +4, +8, +16) at layers 16, 20, 24 and 28, the broadcast across tokens. Magnitude
alpha times the median residual norm at layer 12, alpha the smaller of 0.5 and 1.0 giving
same-position recall at layer 16 of at least 0.9 on the unablated arm, chosen before any ablation
arm is read; a no-injection baseline gives the chance rate. Arms: the 24 downstream relays zeroed
by absolute preservation at k = 4, 8, 16, 24, against layer-matched random sets from the unselected
heads at the same blocks, five seeds. Reading: the broadcast hypothesis survives redundancy if, at
k = 8 or 16, the relays' recall of the injected concept at layer 28, at the same position or four
positions later, falls below the minimum of the five random sets; it closes negative if the relays
sit within the random range at both. Every k, layer and offset is reported. Two corrections to the
sweep's account carried with it: the within-layer comparison is the clean evidence and carries the
argument; the pooled Spearman and the top-ten list are confounded by source depth (six of the ten
strongest single heads are at the shallowest band block, whose perturbations pass through sixteen
more blocks) and come out of the reasoning; every damage figure in the sweep section is at the
layer-28 readout unless it says ten-layer; and the gate held exactly at 1.0 eighty times, the
strongest form that check has taken.

## The coupling test, 09:02: the layer-4 relays do not couple to the band relays

The last causal test of the Director's hypothesis that the out-of-band relays at layer 4 couple
to heads in later layers (`coupling.as-run.py.txt`, `run-coupling.log`, `out/coupling.json`; 160
seconds, bit-identical reproducibility). Each of the sixteen heads writing layer 4 was zeroed alone
on the same documents and positions, and per head: the damage at the ten band readouts, the
relative change it causes in the 26 in-band relays' own attention outputs at the sampled positions,
and the same for the 54 unselected in-band heads. The three layer-4 relays (block 3 heads 4, 10,
12) change the band relays' outputs by a median of 0.072 against 0.058 for the other thirteen
heads (one-sided Mann-Whitney p = 0.12), and their general damage is likewise a little larger
(0.115 against 0.092 over ten layers): a slightly larger general influence, not a coupling. The
selectivity ratio, the change they cause in the relays over the change they cause in the unselected
band heads, is 0.89, 0.77 and 0.91 for the three relays against 0.74 to 0.92 for the others: every
layer-4 head, relay or not, perturbs the unselected band heads more than it perturbs the relays,
and the relays are not more selective than the rest. With the weight-composition result of stage
1b (composition at the rotation null, content not addresses), the coupling hypothesis is not
supported by either the weights or the causal test. What the layer-4 relays are remains what
stage 1 found: heads whose output maps preserve lens content selectively, out of band, with
nothing measured tonight that distinguishes their downstream effect from that of their neighbours.

**Pre-registration amended, 09:12, before the k = 16 arm is read (the Head, urgent): one primary
comparison and a permutation statistic, not a minimum over five seeds.** "Below the minimum of
five random sets" is a one-in-six event under exchangeability, and allowing two values of k and
two positions gives four chances, so the original criterion returned "survives" by chance with
probability 0.52, the threshold that could not fail arriving at the last experiment. Primary
comparison: k = 16, same position, layer 28. Statistic: the 64 injection events are shared across
the six arms at that k (the relays and five random sets), so under exchangeability the relay arm is
one of six labels per event; the relay arm's mean recall over the 64 events is compared with the
distribution of the same statistic when, independently per event, one of the six arms is relabelled
as the relay arm, 10,000 draws. Reading: p < 0.05 one-sided (relays lower) means the hypothesis
survives redundancy; otherwise it closes negative. The other k values and the +4 offset are
secondary and descriptive; the script's own "survives" flag encodes the superseded criterion and is
not read. Two design notes carried with it: the pairing control's difference between relays (25 of
26) and non-relays (36 of 54), one-sided p = 0.0023, is confounded with the selection by
construction, since a head chosen for preserving well has more room above its own scrambled floor,
and is reported as a confounded difference rather than as a control that singles out no relay; and
the injection set is the 24 relays downstream of layer 12 and excludes block 11 head 10, the head
that drove the small-k advantage in the ablation curve, so a negative here is a negative about the
downstream relays and not about that effect.

**Pre-registration amended a third time, 09:25, before any arm of the corrected run is read (the
Head).** The per-event permutation of 09:12 is withdrawn as anti-conservative: the unit of
randomisation is the head set, not the injection event, and relabelling arms per event averages
away the arm-level variance (the five-seed control range at k = 2 in the ablation curve implies a
per-arm deviation of about 0.022, which 64 events do not reduce), so its null is too narrow by
half again. Resolution on the comparison between arms is bought only with more arms. Primary
comparison as before, k = 16, same position, layer 28; statistic: the relay arm's rank among
twenty arms, the relays and nineteen layer-matched random head sets, p = rank over 20, ties split;
the hypothesis survives redundancy only if the relays are the lowest of the twenty (p = 0.05). The
other k values keep five random arms and are descriptive, with the relay arm's rank reported the
same way. Sharing the 64 events across arms is kept for precision on each arm's mean.

**The first injection run is a failed instrument, recorded as such (`run-inject.log`,
`out/inject.json`).** It injected the gradient direction J^T u_c, which raises the concept's logit
fastest per unit norm but does not make the lens read the concept: with the magnitude at the
residual's own norm, same-position recall at layer 13, one recurrent block after the injection,
was 0.27 and at layer 28 zero on every arm, relays and random alike; nothing reached the readout
and the arms compare zeros. The readout is unembed(norm(J h)), so the vector that makes the lens
read c is the one whose image under J is u_c, the pseudo-inverse direction J^+ u_c, which the
corrected run (`inject_events.as-run.py.txt`) uses, with a direction check (the cosine of J v_c
with u_c) printed before anything else, a no-injection baseline, and a magnitude grid of 0.25,
0.5, 1 and 2 times the residual norm with the smallest reaching 0.9 same-position recall at layer
16 chosen before any ablation arm is read. This is the instrument proving it injects before its
readings are read (R52).

**Pre-registration amended a fourth time, 09:32, before any ablation arm of the corrected run is
read (the Head, on F10).** The lens-to-identity cosine is 0.63 at layer 24 and 0.71 at 28, so a
layer-28 lens readout is about two thirds an output readout, and "the injected concept is still
recalled at 28" is largely "the injection still influences what the model would say", a weaker and
different claim from workspace transport. Primary comparison: **k = 16, same position, layer 24**,
downstream of the relays at blocks 15, 19 and 23 but not block 27, three quarters of the set against
a readout 0.63 output-aligned; layer 28 reported beside it, all four blocks against a readout 0.71
aligned, the trade-off stated rather than resolved: a result at 24 and not at 28 means the transport
is workspace-specific; at 28 and not at 24 means block 27 is doing it or it is output influence. The
rank statistic and the nineteen random arms are unchanged. Calibration rules: the grid point
selected is reported; if no grid point reaches 0.9 same-position recall at layer 16, the instrument
has failed and the arms are not read, since a pre-image direction needing more than twice the
residual norm to register says the concept is not writable at that layer. The pseudo-inverse uses
numpy's default relative cutoff on singular values; the cutoff, the condition number of the
layer-12 lens and the number of singular values it truncates are recorded beside the direction
check (the cosine of J v_c with u_c came back 1.0 at minimum and median, so nothing was truncated
at that tolerance).

**The second injection run is a failed instrument too, and the third form is pre-registered
before it runs, 09:40.** The exact pre-image J^+ u_c read at zero at every layer and every
magnitude on the grid, before any arm was read: the layer-12 lens has a condition number of
636,000 (singular values from 13.2 down to 2e-5; 1,419 of 2,560 below a hundredth of the largest,
none truncated at numpy's default cutoff of 5.7e-13), so the pre-image lives in the near-null
space, its image under J has the right direction (cosine 1.0) and negligible size at any unit
norm, and the injection is invisible to the readout. The run was stopped during its arms, which
were comparing zeros. Third form: the regularised pre-image (J^T J + lambda I)^-1 J^T u_c, the
gradient on the small singular directions and the pre-image on the large ones, lambda relative to
the largest singular value squared on a grid of 0.01, 0.1 and 1, magnitude on a grid of 0.5, 1 and
2 times the residual norm; the calibration selects the smallest magnitude reaching 0.9
same-position recall at the first readout with an attention block between it and the injection
(layer 16 for injection at 12), then the lambda reading best there, before any ablation arm is
read; the direction check now prints the cosine and the size of J v_c per unit norm for each
lambda. Failure rule as pre-registered: no grid point at 0.9 means the concept is not writable at
that layer by this instrument and the arms are not run. Fallback, pre-registered now: if layer 12
is not writable, inject at layer 16 (lens-to-identity cosine 0.29), with the relays downstream of
it (blocks 19, 23, 27; 21 heads) as the set, the primary readout at layer 24 (downstream of blocks
19 and 23) and layer 28 beside, the same rank statistic among twenty arms at k = 16, and the
calibration readout at layer 20. A negative at 16 is then a negative about the 21 downstream
relays.

**Added before the run, 09:45 (the Head): room at the primary readout.** The calibration guards
layer 16; the comparison happens at 24. Rule: the unablated arm's same-position recall at layer 24
is stated before any ablation arm is read and must be at least 0.5; if the grid point chosen at
16 leaves it lower, the grid moves up among the points that register at 16; if none gives room,
the finding is that the injected content does not survive to layer 24 with every relay intact,
which is itself the answer, and the arms are not run. The cosine of 1.0 in the direction check is
the expected value when J has full row rank, confirming the algebra and that nothing was truncated,
and it says nothing about whether the direction is injectable at a sane magnitude; the grid is the
check that can fail informatively, and the condition number and the count of small singular values
explain a failed grid.

## The band lenses' spectra, 09:55: the lens at the start of the band is a narrow filter, and where injection is feasible was knowable without a model (the Head)

Computed from the lens file alone (`out/lens_spectra.json`; the reachability and merit columns use
200 random unit directions as a proxy for unembedding rows, because the bf16 embedding could not be
read into numpy in the time; the spectra themselves are exact):

| layer | largest singular value | condition number | participation ratio | singular values at or above 1% of the largest | share of a random direction reachable in that subspace |
|---|---|---|---|---|---|
| 12 | 13.2 | 636,586 | 517 | 1,141 | 0.45 |
| 16 | 16.0 | 1,008,853 | 742 | 1,558 | 0.61 |
| 20 | 10.8 | 87,557 | 1,410 | 2,395 | 0.94 |
| 24 | 8.4 | 72,563 | 1,877 | 2,519 | 0.98 |
| 28 | 6.7 | 9,605 | 2,050 | 2,539 | 0.99 |

**F11 (instrument).** The lens at the start of the band is a heavily filtered low-dimensional
projection: at layer 12 fewer than half of the residual's directions reach the readout at a
hundredth of the largest gain and the participation ratio is 517 of 2,560, so a claim of the form
"the lens shows X at layer 12" inherits a narrow filter on the residual. The conditioning improves
monotonically through the band, with the effective dimensionality rising from a fifth at 12 to
four fifths at 28, which is the geometry the hosted-lens work saw from the other side as the lens
vectors' effective dimensionality rising from about a tenth at layer 12 to nearly half at 20. F11
sits beside F10 as the second thing learned about the lens by trying to use it rather than by
reading it, and both bear on WP5 and WP7: the lens is a sharp instrument at 24 and 28, where it is
also two thirds an output readout, and a blunt one at 12 and 16, where it is not.

**The merit and why both instruments failed.** The logit gain per unit injection norm is the
product of the size of J v and its cosine with the readout direction; per unit norm it is maximised
exactly by the gradient direction (Cauchy-Schwarz), and the regularised pre-image trades gain for
alignment: at layer 12 the medians are 0.70, 0.92, 0.98 and 0.99 for lambda 0.01, 0.1, 1 and the
gradient, against 0.0008 for the exact pre-image. So the gradient was the best direction by gain,
and it read at 0.27 one block downstream because gain is not the quantity: the readout is a softmax
over 248,320 tokens after normalisation, and the injected direction must lift the concept over
every competitor, which the gradient does not at any magnitude that keeps the residual sane. The
calibration grid is the only test of that, and the spectra say where it can pass: at layer 12 the
concept's own readout direction is mostly outside the reachable subspace, and at 20 it is almost
entirely inside.

**Pre-registered before the primary arm lands, 10:18 (the Head): the third outcome, and the
magnitude caveat.** The descriptive arms at k = 4 and 8 are not at the null but at the opposite
extreme: the relays remove less of the injected content than every random set at layer 24 (0.938
against five controls at 0.922, rank six of six). The criterion covers survival (rank one of
twenty) and closure (inside the range) and says nothing about a rank of twenty of twenty, which is
a third outcome. Ruling now, so that it is not constructed after the number: a reversal at k = 16,
the relays highest of the twenty at layer 24, is reported as a mechanism and not as
uninterpretable: if removing the relays preserves injected content better than removing random
heads, the relays are the heads that would otherwise remove or overwrite it, suppression of
off-manifold material by the heads most engaged with the workspace. It is stated as a hypothesis
with the caveat below, it is not the broadcast hypothesis appearing, and the contrast-direction
instrument (injection along J^T of the concept's row minus the mean of the rows the lens already
reads at that position, built for the margin rather than the gain) becomes the test of whether the
suppression reading survives a better-constructed injection, run as the next use of the instrument.
The caveat: the calibration passed only at the top of the grid, twice the residual norm, so the
injected vector is larger than the entire residual at its position and the state read is nothing
like a natural activation; every conclusion from this instrument inherits that. It bears hardest on
the zero-broadcast result, which is stated in these words and no wider: an injected lens direction
of twice the residual norm at layer 12 does not appear in the lens's top 25 at any later position
at layers 16 to 28, on any arm; that is not the claim that the workspace does not broadcast. That
the calibration passed only at the top of the grid is the third independent sign of the geometry
F11 describes: the layer-12 lens is barely writable even after regularisation.

## The injection test, 10:25: the broadcast hypothesis does not survive redundancy, and the thread closes

Third instrument (`inject_events.as-run.py.txt`, `run-inject-events.log`,
`out/inject_events_L12.json`; the two failed instruments in `run-inject-failed-gradient.log`,
`out/inject_failed_gradient.json` and `run-inject-failed-preimage.log`): regularised pre-image,
lambda 0.01 of the largest singular value squared, magnitude twice the residual norm, the only grid
point passing both the calibration and the room rule (unablated same-position recall 0.953 at 13,
0.984 at 16, 0.953 at 20, 0.922 at 24, 0.828 at 28; no injection gives zero everywhere). Sixteen
concept tokens, multilingual word pieces from the vocabulary filter, 64 events.

**Primary, pre-registered: k = 16, same position, layer 24, the relay arm's rank among twenty.**
Relays 0.938; all nineteen random sets 0.922, which is the unablated value to the event. Rank 20 of
20. The hypothesis required rank 1; **it does not survive.** The comparison has no dynamic range:
no sixteen-head set of any kind, relay or random, removed a single event of the injected content at
layer 24, and the relays' rank rests on one event of 64 in the other direction. Descriptive arms
at 24: k = 4 and 8, relays 0.938 against five controls at 0.922 (rank 6 of 6); k = 24 the same
pattern. At layer 28, where there is range (controls 0.766 to 0.906 around the unablated 0.828),
the relays' ablation raised recall to 0.891, rank 19 of 20; at k = 4 and 8, 0.828 and 0.859, ranks
4.5 of 6. At layer 20 (upstream of blocks 23 and 27, downstream of 15 and 19) nothing moves. At
every later position, on every arm, zero.

**Reading, under the pre-registrations of 09:12 to 10:18.** (1) The broadcast hypothesis closes
negative for the 24 downstream relays: their removal as a set does not remove injected workspace
content where random sets of the same size at the same layers do not. Combined with the sweep
(relays level with the rest at every layer alone) and the ablation curve's redundancy (now read as
non-specific), the paper's claim that the heads picked out by lens preservation carry workspace
content is not supported on this model at single-head, set-level or injection resolution. What
stage 1 finds is real and is not that. (2) The pre-registered third outcome appears at layer 28
and not at 24: removing the relays makes the injected concept more readable at the output-aligned
readout, which is recorded as the suppression hypothesis, the relays being heads that would
otherwise act on an off-manifold perturbation in the workspace, stated as a hypothesis under the
magnitude caveat (the injected vector is twice the residual at its position), with the
contrast-direction instrument as its test. That the effect appears at 28 and not at 24 is, by the
pre-registered trade-off, either block 27 doing it or output influence, and cannot be separated
here. (3) The zero-broadcast result is stated no wider than the instrument: an injected lens
direction of twice the residual norm at layer 12 does not appear in the lens's top 25 at any later
position at layers 16 to 28. (4) The instrument's limits are the instrument's, not the model's:
the layer-12 lens is barely writable (F11; calibration passed only at the top of the grid), and a
test at layer 20, where reachability is high, with the fifteen relays downstream and the contrast
direction, is the next use of it.

**F11 corrected with the injected concepts' own rows (`out/lens_spectra_concepts.json`).** The
random-direction proxy was optimistic, not pessimistic: the sixteen concepts' unembedding rows lie
less inside the reachable subspace than random directions at every layer (medians 0.42 against
0.45 at 12; 0.56 against 0.61 at 16; 0.82 against 0.94 at 20; 0.87 against 0.98 at 24; 0.88 against
0.99 at 28; the least reachable concept at 0.23, 0.29, 0.43, 0.52, 0.55). The lens's image is not
trained toward unembedding rows more than toward arbitrary directions on this fit; if anything the
opposite. The Head's expectation on this point is refuted by the sixteen rows, and the layer-20
reachability of 0.82 for real tokens still makes 20 the writable layer.

**WP12 closes.** Stage 1 stands (selective relays, band-favoured, group-structured, surviving
orthogonalisation; prediction 1 refuted five times; the Director's coupling hypothesis unsupported
by weights and by intervention; the Head's layer-32 hypothesis refuted; F10 and F11 on the
instrument; R53 on copy maps). Stage 2's causal claim does not reproduce. Filed for the next use of
the model on this thread: the contrast-direction injection at layer 20; the paper's own injected-
thought report protocol as the behavioural version; the lens-specific recall of the model's own
final token under ablation as the paper's other metric. The model goes to the Director's overnight
order.

**Correction, 10:35 (the Deputy): the primary comparison was uninformative, not negative.** On 64
events, 0.938 is 60 and 0.922 is 59: the relays scored 60 and all nineteen random sets scored 59,
so the rank of twenty among nineteen tied values separated by one event measures the tie-breaking
and not an effect. "The broadcast hypothesis does not survive redundancy" in the section above
reads as a tested refutation and is withdrawn. The honest statement: the pre-registered primary
comparison had no dynamic range, nothing moved the metric, so it cannot distinguish the hypothesis
from its negation; the hypothesis is untested at layer 24 by this instrument, and the one-event
difference is reported and not interpreted. The finding the test does carry is the secondary at
layer 28, where there is range: four events of 64 in the direction opposite to the hypothesis,
random sets scattering around the unablated value, the pre-registered third outcome. Reading (1)
of the closure is restated accordingly: the paper's causal claim is not supported on this model,
by the single-head sweep (relays level with the rest within every layer), by the set-level
redundancy (non-specific), and by the injection test's secondary (the unpredicted direction at
28); it is not refuted by the injection test's primary, which was uninformative. A worse headline
and a better paper, as the Deputy put it; the same shape as the tiny-denominator ratio and the
head that appeared in both contexts by chance: a well-defined quantity answering a different
question from the one asked of it.
