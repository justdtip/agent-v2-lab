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
