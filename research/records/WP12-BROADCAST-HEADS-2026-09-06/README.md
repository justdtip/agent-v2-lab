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
