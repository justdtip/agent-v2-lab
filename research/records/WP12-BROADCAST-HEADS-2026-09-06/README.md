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
