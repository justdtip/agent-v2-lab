To the Research Director, from the research division session `an-app-cd`. 2026-09-05.
Subject: literature and precedent on the two questions you were asked to rule in
WO-INTERP-002 — the distance axis, and the starting resolution of the curve.
Requested via Proxy. Read-only; three parallel literature agents plus two local measurements.

> **HOLD IMPOSED AND LIFTED, 2026-09-05 evening.** The Director vetoed ratification of the
> experiments in the pipeline, EXP-002 and its rerun and WO-INTERP-002 among them, with nothing
> to be ratified, lifted or run pending discussion with the Chief; he then confirmed the veto
> lifted. Both relayed to this session by the Proxy. Recorded because the pause is part of the
> history of these recommendations, not because it still binds.
>
> **Still open, and narrower than the lift:** the Director has not directly confirmed the
> standing lift that would remove the per-run gate. Until he addresses it specifically, a run
> still needs its own lift; the veto being lifted restores the pipeline to individually gated
> runs, not to ungated ones.

## 1. Distance axis: tokens primary, turns recorded alongside

**The mechanistic literature is effectively unanimous on tokens, and it is the literature we
are in.** Khandelwal et al. (ACL 2018, arXiv:1805.04623) set the pattern for exactly this kind
of decay curve and measured in tokens, reporting an effective horizon near 200 tokens with an
order-sensitivity break near 50. Every memory-horizon result on recurrent or state-space models
uses tokens and nothing else: DeciMamba's effective receptive field (arXiv:2406.14528), the
Forgetting Curve (arXiv:2410.04727), Zoology's interaction distance i-j (arXiv:2312.04927),
Repeat After Me (arXiv:2402.01032), and the SSM recency-decay bound exp(-k(t-s))
(arXiv:2501.00658). **No precedent was found for any recurrent or hybrid memory horizon
measured in turns or messages.** Our target is a hybrid whose recurrence integrates over
tokens, so the mechanism axis and the literature axis agree.

**The strongest citation is our own instrument's paper, and the first version of this memo
missed it.** Gurnee et al., "Verbalizable Representations Form a Global Workspace in Language
Models" (arXiv:2607.15495, 2026), is where the Jacobian lens this programme runs on is defined.
Figure 28 panel (c) is an autocorrelation of the lens's top-1 token across **positions**: the
rate at which the same concept remains at the top of the readout at position t and at position
t plus delta, against a position-shuffled null. It is near the null in early layers, rises
sharply at the band onset, peaks across the middle, and falls back in the final layers. So the
paper that defines our instrument measures concept persistence across the token stream, and
never across turns. Verified here against the local copy of the PDF, not relayed: the figure
caption and the body text both read as quoted, and the identifier, authorship and depth-band
claim were confirmed independently from the arXiv record.

**Turn count has been named as a bad axis, in these words.** Audio MultiChallenge
(arXiv:2512.14865) rejects it because "each turn's length is also arbitrarily defined" and
adding turns "does not mechanically increase difficulty"; they moved the axis to duration.
PsychoPass (arXiv:2606.03136) found near-perfect classification "largely explained by the
inclusion of number of turns as a feature", the nuisance-variable form of the same problem.

**The one paper that does use a turn axis proves the point.** Dongre et al. (arXiv:2605.12922)
index goal accessibility by turn and find the failure turn is a linear function of the token
window, R-squared above 0.999. Their turn axis is a rescaled token axis, which is what a turn
axis becomes when turn lengths are homogeneous. Liu et al. (arXiv:2307.03172) get away with a
document-index axis for the same reason: their chunks are capped at 100 tokens.

**Ours are not homogeneous, and this is the local finding that decides it.** Per-message length
in characters, `agent_v2d-qwen35-4b`, named by split as R38 requires:

| split | rows read | p10 | p50 | p90 | p90/p10 |
|---|---|---|---|---|---|
| train, first 400 rows | 400 of 2826 | 61 | 198 | 999 | 16.4x |
| train, full split | 2826 | 61 | 204 | 1298 | 21.3x |
| valid, whole split | 381 of 381 | 62 | 207 | 1624 | 26.2x |
| test, first 400 rows | 400 of 1841 | 62 | 211 | 1646 | 26.5x |
| test, full split | 1841 | 62 | 217 | 1652 | 26.6x |

`agent_v2` and `agent_v2c` test splits match the `agent_v2d-qwen35-4b` test figures to within a
character. The first version of this memo reported only the test-split row and described it as
"400 rows each" without naming the population; the Head of Interpretability caught that under
R38 and reproduced the numbers independently. Two things the correction surfaced that the
original did not: train is the mildest split at 16.4x, so a reader recomputing on train would
have thought they had found a discrepancy, and the valid split holds 381 rows, so a request for
400 silently read the entire population rather than a sample of it.

A 16x to 27x spread means "five turns back" is not one distance. On a five-level turn grid the
token distance of a level overlaps its neighbours, and a genuine token-distance decay would
appear as a flat or noisy turn curve. That is the false-null shape, and it is the same failure
mode the S3 mask review just caught in a different guise.

**Second local fact.** The instrument already indexes tokens. `--source-positions` takes
fractions of each context's length or absolute token indices, resolved per context by
`resolve_source_positions` (src/local_llm_lab/pipeline/jlens.py:242). A turn axis needs a new
turn-boundary-to-token mapping layer written and reviewed; the token axis needs nothing.

**Recommendation.** Sweep on token distance. Record the turn index and the containing span's
token length for every point, and report the turn mapping in the results so the note contract,
which is stated in turns, can be read off the curve. Do not sweep on turns.

**Honest gap.** No paper was found that plots one dataset on both axes and shows the
conclusions diverge; the evidence above is convergent, not that direct demonstration. Recording
both axes costs us nothing and would supply it.

## 2. Starting resolution: five is a floor, not a design; seven doublings fit our window

**Five identifies a decay but does not discriminate its shape.** A p-parameter model needs at
least p distinct levels (Seber and Wild 1989); D-optimal designs for exponential and Emax
models are minimally supported at exactly p points (Box and Lucas, Biometrika 1959; Dette et
al., Biometrika 2010), and any design collapses to p+1 support points without information loss
(de la Garza, Ann. Math. Statist. 1954). Levels beyond p buy exactly one thing: lack-of-fit and
model discrimination. That is precisely EXP-003's stated output, a shape rather than a verdict,
so the extra levels are the point and not a luxury. Practical target from the discrimination-
design literature is p_max plus two to three.

**Spacing should be geometric.** Log or geometric spacing is standard when the effect is
multiplicative or the range spans orders of magnitude, and exponential cannot be separated from
power-law except on a log grid spanning one to two decades (Clauset, Shalizi and Newman, SIAM
Review 2009). Field practice agrees by habit rather than by argument: context length is swept
on powers of two almost everywhere (RULER 4K to 128K, NoLiMa 250 to 32K), while needle depth is
usually linear, and the only non-uniform depth grid in common use is the NIAH sigmoid option,
a heuristic with no stated derivation.

**Our window makes the grid obvious.** The configured operating window is 2688 tokens
(configs/agent_v2d_qwen35_4b.yaml:40). A ratio-2 geometric grid from 32 to 2048 gives seven
levels — 32, 64, 128, 256, 512, 1024, 2048 — spanning about 1.8 decades, which clears the
discrimination threshold. It also brackets the only prior horizon estimate we have, Khandelwal's
50 to 200 tokens, with three levels rather than one.

**Which window, settled, because the grid depends on it.** The 4B's native
`max_position_embeddings` is 262144, read from the model's own `text_config` in the MLX
snapshot; our configured operating window is 2688, in both the registry entry
(configs/models/qwen35-4b.yaml) and the run-D config. The factor is 97.5. The Head of
Interpretability has ruled that EXP-003 characterises the **deployment** window rather than the
architecture, because the experiment exists to decide D4's note contract and that contract
operates at 2688. The 32-to-2048 grid therefore stands. Their spec will say the choice is
deliberate, say which window it is so nobody later reads the curve as an architectural result,
and pre-register the reading if the curve is still descending at 2048: that our operating window
sits inside the decay rather than past its knee, which would be a finding about the window
choice worth more than the curve that produced it.

**The budget arithmetic, which is the real constraint.** At fixed total N, going from five
levels to seven inflates each level's standard error by a factor of about 1.18. Field practice
buys replicates over levels heavily; RULER runs 500 replicates across six levels. Given
WO-STAT-001, we cannot afford another underpowered curve, so replicates per level must be
pre-registered with a power statement rather than left to fill the day.

**Recommendation, and where it has since been overtaken.** Seven levels, geometric ratio 2, from
32 to 2048 tokens. My budget split of roughly 70 percent on the grid and 30 percent held for
second-stage re-placement does not survive contact with the measured rate: the Head of
Interpretability costs seven levels at EXP-001's sample size at about eleven hours, and what fits
in seven hours is 27 per level at 0.41 power, which is exactly the underpowered curve WO-STAT-001
forbids. Their resolution, adopted in the work order, is better than mine and does not touch the
grid: the decisive measurement is the model's own output distribution and needs no Jacobians, so
the 133 seconds per point is the lens rate and not the measurement rate. Sweep the full grid on
the output measurement, take lens readouts at three levels for corroboration, and allocate
replicates unequally, since the extremes are near-certain and the knee is not. If the Director prefers a smaller first pass, drop to five levels by
removing 64 and 1024, keeping the endpoints and the knee bracket, and state in the spec that
the five-level version can report decay but cannot claim a shape.

## 3. One thing the window check turned up, which is not a literature question

The 4B is not uniformly recurrent. Its `text_config` gives `full_attention_interval` 4 and a
`layer_types` list of 32 entries in which every fourth is `full_attention`: eight full-attention
blocks at one-based positions 4, 8, 12, 16, 20, 24, 28 and 32, with linear attention everywhere
else. A distance-decay curve on this model therefore mixes two mechanisms, one that decays with
token distance and one that does not decay the same way at all.

It bears directly on the probe grid. `resolve_layers` maps a fraction to
`max(1, round(fraction * num_layers))` (src/local_llm_lab/probes/policies.py:113), so the
registry's `[0.167, 0.333, 0.5, 0.667, 0.833, 1.0]` resolves at 32 layers to 5, 11, 16, 21, 27,
32. Two of those six are full-attention blocks, and they are the two that matter most: the
mid-band layer 16, which is where Gurnee et al. put the workspace content, and the final layer
32. A flat curve at layer 16 would then say little about the recurrent state, because that layer
can reach the whole window by attention regardless.

**The indexing flag is answered from the instrument's own record, and the concern is ruled.**
EXP-001's 4B artifact states the convention in its conformance block: layer L is the residual
after block L-1, and a probe layer's kind is the kind of the block that wrote it. It records the
kinds it actually used — 5 linear, 11 linear, 12 attention, 16 attention, 20 attention, 21
linear, 27 linear, 28 attention, 32 attention — so 16 and 32 are attention outputs as derived
above, and the assumed indexing is the indexing the probe used. The period is derived
structurally from `view.blocks[3].is_linear` and only cross-checked against configuration, which
is the R31 preference the right way round.

The Head of Interpretability has ruled that EXP-003 inherits EXP-001's kind-matched family
rather than the bare registry fractions, and reports the curve **per kind, never pooled**. That
is better than dropping to linear-only layers, which would remove the confound and the
comparison with it: each in-band linear layer already draws an attention partner at adjacent
depth, pairs 11 with 12, 21 with 20, 27 with 28, so the difference between kinds at one depth
*is* the recurrent contribution. The bound on the whole concern belongs in the spec too: the
decisive measurement is the model's own output distribution, which is layer-independent, so the
kind confound touches the corroborating lens rows and not the verdict.

**The tie-break is arbitrary in the rule and not arbitrary in its effect.** The generalised rule
adopted for the family takes each in-band layer's nearest neighbour of the other kind, breaking
a tie on the lower index, which gives 16 the partner 15. Both 15 and 17 are linear and
equidistant, so the rule picks by index alone; but the two choices give different designs.
With 15, attention is the deeper member of three pairs out of four and the kind groups have mean
depths 19.0 and 18.5. With 17 it is the deeper member of two out of four and both groups sit at
exactly 19.0. A depth trend in the decay therefore partly aliases onto the kind contrast under
15 and cancels under 17. The same choice also decides which block separates the pair: under 15
three of the four pairs bracket an attention block and one brackets a linear block, while 17
splits them two and two. The recommendation was 17 wherever kind contrasts are compared or
averaged across depth, and it has been ruled that way. The condition is met: the Head of
Interpretability has made the reporting explicit rather than leaving it to be read, with the
primary reported per pair at its own depth and never pooled, and the pooled per-kind curves kept
as the summary. It is the summary the balance protects. The tie-break is now stated generally —
on a tie take the neighbour leaving the kind groups closest in mean depth, lower index only on a
remaining tie — with the measured table beside it and the resolved choice in the conformance
block, so the rule carries the reason that decided it rather than an index comparison.

Of the three asymmetries the decisive one is the third. Depth balance and orientation are both
about a nuisance trend cancelling; the bracketing count is about what the contrast measures,
since under the other choice three quarters of the family would measure what an attention block
contributed and one quarter what a linear block did, and the recurrent contribution is the
second. That is not a nuisance, it is the quantity.

**The gap that produced that family, offered rather than ruled.** The artifact's `layer_roles` marks
16 as a primary with no partner, because the family pairs each in-band *linear* layer with an
attention neighbour and 16 is itself an attention output. So there is kind contrast at depths
11/12, 20/21 and 27/28 and none at 16 — the middle of the band, where Gurnee et al. put the
workspace content. The same structural rule run in the other direction supplies one: layer 15
and layer 17 are both linear-attention outputs under the artifact's own convention, either
serving as 16's linear partner at adjacent depth.

## 4. Two verification notes

Do not cite MemDelta (arXiv:2606.29914) for the axis question; a first automated pass suggested
it addressed the turn-versus-token confound and a re-read of the abstract shows it does not.
The de Vocht et al. (2014) author list came from a secondary source, the article being
paywalled; the claim it supports, that cumulative exposure obscures intensity and duration and
both should be modelled, is not load-bearing here.

— research division session `an-app-cd`. The Head of Interpretability holds the seat; this
memo is research support for their work order, not a ruling.
