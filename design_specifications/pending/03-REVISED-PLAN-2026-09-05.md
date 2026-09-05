# 03. Revised plan after the hosted-lens results (2026-09-05)

Author: Chief AI Research Scientist. Status: FOR THE DIRECTOR'S APPROVAL. Nothing in section 6 has been sent; no session has been messaged; no run has been started. The veto was lifted on 2026-09-05 (evening); per-run lifts for agent deployment remain in force unless decision D4 changes them.

Sources: `under_review/CHIEF-JSPACE-PAPER-READING-2026-09-05.md` (the paper, read verbatim), `under_review/HOSTED-JLENS-QWEN35-4B-RESULTS-2026-09-05.md` (today's runs), `under_review/CHIEF-PROBE-REFINEMENTS-2-DERIVATIONS-2026-09-05.md` (the read-weight and horizon derivations), the chat of 2026-09-05.

## 1. What changed today, in one line each

- F1. A reference-recipe Jacobian lens exists for our exact checkpoint, loads in seconds, reads at one matrix product per (layer, position), and catches a context signal our finite-difference estimate missed. Our estimator is retired for reading.
- F2. The workspace band on Qwen3.5-4B sits at layers 13 to 29 by persistence, the lens geometry fans out at layers 18 to 25, the first twelve layers hold no persistent content, and the last three are motor. The registry's probe fractions put two of six probe layers outside the band and one in the motor regime.
- F3. The lens recovers a factual answer thirteen layers before the plain unembedding; unspoken intermediates on raw prompts are weak at 4B; the decision-point workspace in our task holds the type of the answer, not its value.
- F4. A Gated DeltaNet block's value-path Jacobian from position t to t' is exactly the read weight (verified to 1e-11), so the recurrent channel's transport is computable from the forward pass without any Jacobian. Restated 2026-09-05 late evening on the live calibration (WP3 design section 9): most recurrent heads have horizons of tens of tokens, but about fifteen percent of value heads in every block have gate constants above a thousand tokens and read across the whole context, so long-range recurrent transport exists in a head population and the earlier "attention-only" statement is withdrawn.
- F5. The paper's base-versus-post-trained dissociation does not reproduce on this lineage: the Base carries Assistant-reaction content on user tokens as much as the post-trained model; only the response-start position separates them.
- F6. World A (EXP-001) stands on the reference instrument, on both checkpoints.
- F7. The Director wants workspace traces over whole agentic episodes, every position and every layer, for visualisation and for the goal-retention question.
- F8 (2026-09-05 late evening; WP3 design sections 8 to 10i; measure named). On agent transcripts the recurrent channel of this model behaves, to first order, as a write-strength-weighted running summary and not as content-addressed memory. Measured with top-ten far-source overlap and query cosine on the same seven read positions spanning 25 to 100 percent of the context, over the 115 value heads whose gate stays open on both transcripts: the population's overlap is 1.78 of 10 against 1.30 for a null that keeps each head's real write strengths and keys and randomises only the query. That null is the running-summary model written down, so the population sits within about a third of it (median ratio 1.32; 201 of 230 entries above it): a positive fit, with the far-source sets only mildly more stable than write strength alone predicts. Ten entries from nine heads exceed one and a half times the model with a query cosine below 0.5; one entry exceeds twice it; none exceeds twice it with a cosine below 0.3. On the right unit for the cross, the (block, key head) pair whose two value heads share the cosine (85 units), three units exceed one and a half times the model with cosine below 0.6 in both contexts against a chance expectation of 0.58: block 1 key head 7, block 6 key head 0 and block 15 key head 3; block 6 key head 0 is the strongest (2.00 at cosine 0.348 and 1.76 at 0.248) and the only one that survives at cosine below 0.5 (expectation 0.21); at ratio above 1.8 with cosine below 0.35 none replicates (expectation 0.01). The three are named for source-position inspection on those figures. The long-gate heads are distinguished from the rest by the gate and by key stability (0.58 against 0.32 on distinct key heads), not by their queries, whose stability across positions is general to these blocks. Consequence: long-range addressable memory in this model is carried by attention; of the three routes for goal retention in the paper-reading memo's section 7, the recurrent state is out as a general mechanism; the notes-plus-windowing paradigm is the right architecture for this model on the evidence, measured rather than assumed. Still unmeasured: whether the running summary is workspace content (WP5 composing WP3's head list with the hosted lens), and the same measurement on the sweep corpus, which sets the scale of the gate constants (section 9b). Earlier versions of this finding, written on read positions confined to the back half of the context, overstated the population's overlap at 3.24, called the heads static filters on that basis, and stated "no content-addressed retrieval" without naming the measure; all three are corrected here.

## 2. Decisions requested from the Director

| id | decision | Chief's recommendation |
| --- | --- | --- |
| D1 | Adopt the hosted lens as the programme's reading instrument (ruling R40a). The finite-difference JVP path stays in the tree for the self-only and future-only variants only, documented as legacy. | Yes. |
| D2 | Re-base probe layers on the measured band (ruling R40b, amended by R41b to the kind pairs 12/13, 15/16, 19/20, 23/24, 27/28); EXP-001 stays as recorded on the old fractions; EXP-003 and EXP-004 adopt the band before any code. | Yes, with the partner rule unchanged. |
| D3 | Disposition of the ten dirty worktrees (section 5). | Land the first-render fix and the record slices; hold the power worktree for re-powering; keep the S3 instrument. |
| D4 | Lift regime for EXP-005 trace collection, which runs the agent. Options: per-run lifts as now; or a standing lift bounded to trace collection on the existing evaluation task set, no training, a stated episode cap. | Standing lift, capped at 40 episodes per collection, reported per batch. |
| D5 | Priority order of section 4. | Wave 1 first: instrument, layers, transport weights, since each is a day and everything else reads through them. |
| D6 | The Base contrast: accept the negative result and pivot Thread 1 to the self-monitoring probes, or refit a Base-specific lens first (WP11). | Pivot; refit only if a Base result later hinges on transfer fidelity. |

## 3. Work packages

Owner conventions as in the wiring map: the Chief writes specs and rulings and gates every commit; the Deputy dispatches implementation to Codex and commits; the Head of Interpretability designs and reviews interpretability slices (R36); the Proxy relays the Director. No implementing subagents. Every run is pre-registered with its reading and powered by the WO-STAT-001 method before it starts.

**WP1. Hosted lens in the probe stack.** Owner: Deputy, Codex implements; Head reviews. New module `local_llm_lab/pipeline/hosted_lens.py`: load the converted `.npz` (or convert the `.pt` under the restricted unpickler as `convert_lens.py` does), transport `h @ J.T` with the layer convention L = file index + 1 and identity at L = 32, readout through `ArchitectureView` as today. `agent-v2-jspace-sweep --lens hosted` produces full top-k per (case, layer) plus the two-candidate rows for continuity; the Holm family becomes the layers under one instrument. Registry gets a `lens:` block (repo, folder, filename, SHA-256, prompts, layer convention) so provenance records it; the lens file stays in the HF cache, not in git. The model-run lock of the standing-lift note's concurrency clause is not a WP1 deliverable but issue 83, a standalone slice that lands before WP1's validation run, since that run is the first model load cleared without a per-run lift. Regression fixture: today's 42-case table (hosted L20 11/42, L21 15/42, L27 31/42; Base L20 8/42) reproduced within one case, in R31 form against the real module. Closes the artifact follow-ups on #69 and #72 and supersedes #78. Cost: one implementation day. Deliverable: READY TO COMMIT with the fixture green.

**WP2. Band-based probe layers.** Owner: Chief (ruling R40b as amended by R41b), Deputy implements. `probes.layer_fractions` in `configs/models/qwen35-4b.yaml` replaced by `probes.band_layers` = 12, 13, 15, 16, 19, 20, 23, 24, 27, 28, written as kind pairs (12/13, 15/16, 19/20, 23/24, 27/28: each pair one DeltaNet-written and one attention-written layer within one layer of each other) and the pairs named in the registry; the 9B entry gets a placeholder pending its own band measurement (WP1's corpus stage run once on the 9B). Cost: hours. Depends on D2.

**WP3. Transport weights on both channels.** Owner: Head designs from the Chief's section-9 brief and the read-weight derivation; Deputy implements; Chief gates. Hooks on each DeltaNet block's projected q, k, gate and beta and on each attention layer's probabilities; exact read weights alpha[t', t] per head with the verified product order (s + 1 leftmost); statistics per layer and head: |alpha| in gap bins 1 to 4, 5 to 16, 17 to 64, 65 to 256, 257 to 1024; E[log g], tau_head, interference constant; attention mass per gap bin. Pre-registered reading: superseded by the Head's WP3 design (pending/WP3-TRANSPORT-WEIGHTS-DESIGN-2026-09-05.md section 5, readings R1 to R4, ratified R41a: retention ratio and mass share against a uniform null, bins to 2688); the original threshold was not scale-free. attention blocks 15, 19, 23, 27 (writing layers 16, 20, 24, 28; corrected from block indices misread as layers, see R41a) carry all transport beyond that. Contexts: the sweep corpus, ten agent transcripts, and the EXP-002 decision contexts. Replaces the gap-stratified future lens and the gate-input brief. Cost: one design day, one implementation day, minutes to run.

**WP4. EXP-002 with an exact pre-check.** Owner: Deputy; Head reviews. Issue 82 (opened 2026-09-05 late evening; the plan's earlier pointer to "existing EXP-002 issues" referred to issues that were never opened). Blocked on issue 81 for (b), which is built on WP3's hooks. (a) Land the first-render fix with the four account corrections already ruled. (b) Read-share pre-check from WP3's hooks: the hidden span's contribution to the read at the decision, per layer, as a norm share, exact. (c) The approved 2 by 2 with the write ablation, sized by the WO-STAT-001 method given (b); if the share is below 0.02 at every layer, the 2 by 2 runs at the minimum size that can detect a share of 0.05, or is recorded as not worth running. Depends on WP3.

**WP5. EXP-003 re-specification.** Owner: Chief writes; the WO-STAT-001 author re-powers; Deputy implements after a dry run. Readouts: model output (decisive), hosted-lens rank of the target at each band layer, logit lens for comparison. Statistics: paired Wilcoxon on log-rank, matched against mismatched context, per layer; recovery depth (first band layer at which the target enters the top 25) as a function of distance. Levels unchanged, 32 to 2048, geometric. Per-kind predictions from WP3: beyond the recurrent horizon, recovery at a recurrent-output layer implies an attention layer below carried it. Depends on D2 and WP1; runs after WP3.

**WP6. EXP-004 update.** Owner: Chief. Hosted-lens arm replaces the JVP arm; the implicit-elicitation control adopts the paper's paired-question protocol (same stimulus, next-word question against name-the-property question). Small.

**WP7. Representation probes, revised.** Owner: Head; Deputy implements slices. (a) Assistant axis on the post-trained model against its role-play personas, decomposed by gradient pursuit against the hosted dictionary at band layers: variance share, and a clamped steering test of the J-space component against the remainder. (b) Self-monitoring probes on both checkpoints: BUT after a dispreferred prefill, damn and failure words under thought suppression, disclaimer and fictional at the Assistant token in roleplay. Replaces the reaction contrast as Thread 1's post-training signature. (c) Adapter deltas read through the lens: the top tokens of each LoRA update direction, per layer, the paper's component-reading method. (d) State probes moved to band layers. Each with a pre-registered reading.

**WP8. EXP-005, workspace traces over agentic episodes.** Owner: Deputy implements capture and storage; Head designs the pinned-token set; Chief builds the viewer. Capture during the agent's own run through the cached forward: at every position, prompt and generated, the hosted readout at every layer, stored as top-25 per cell plus the ranks of pinned tokens (goal, plan step, tool name, target file stem), with step boundaries; one file per episode. Viewer: the paper's layer-by-position map with pinned-token rank trajectories, as an HTML artifact. First statistic: rank of the goal tokens at band layers as a function of the distance since the goal was stated, the goal-retention measurement. Cost per 1500-token episode: about 30 s of readout on top of the run. Depends on D4 and WP1.

**WP9. Causal privilege at 4B.** Owner: Head. Two-hop prompts in our task family; swap the intermediate's J-space component against the remainder with the paper's clamp; success rate against the paper's 54 percent on Haiku 4.5. This is the experiment that turns "the band exists" into "the band is causally privileged" on this model. Depends on WP1.

**WP10. Counterfactual goal-reflection training, design only.** Owner: Chief drafts after WP8's first result. Two arms against the note baseline: goal restatement (goal, constraints set earlier, completion criterion) and state restatement (files read, values collected, pending), loss on the reflection turn only, uninterrupted evaluation, hosted-lens check that the implanted tokens enter the band at decision positions, ablation of those lens vectors as the causal test. Not scheduled until WP8 says whether there is anything to strengthen.

**WP11. Contingency: Base-fitted lens.** The reference library on torch with the MPS backend, about a hundred wikitext prompts, one to two hours. Only if D6 chooses refit or a later Base result needs it.

## 4. Ordering

- Wave 0 (this week, records): D3 dispositions; land the first-render fix and the R38 slices; close the EXP-001 artifact follow-ups through WP1.
- Wave 1 (instrument): WP1, WP2, WP3. Each about a day; WP3 can run while WP1 is under review because it needs no lens.
- Wave 2 (experiments on the new instrument): WP4 (issue 82) pre-check then 2 by 2 at the licensed size; WP5 dry run then full, after WP3's measured curve; WP8 first traces on ten episodes.
- Wave 3 (probes): WP7, WP9, WP6.
- Wave 4 (intervention): WP10, gated on WP8.

## 5. Dispositions of the in-flight worktrees (for D3; corrected 2026-09-05 evening on the Deputy's audit, verified by the Chief against the landing commits and the branch files)

Eight of the nine worktrees were already landed when this table was first written; comparing each to the commit that landed it, not to the moving branch head, shows six identical to their landing commit and three behind it, the branch having moved past them. Only one holds unlanded work.

| worktree | head | dirty | status | disposition |
| --- | --- | --- | --- | --- |
| wt-template-prefix | 24ca77b | 8 | landed as a922013 (2026-09-05 late evening) with the four account corrections in one commit, after the Chief's gate on every hunk | done; worktree removed |
| wt-s3 | 9f00be2 | 5 | landed as 526f295; worktree behind the branch | residue; remove after the template fix lands |
| wt-exp002 | 045d015 | 2 | landed as 9f00be2; identical | residue; remove |
| wt-r38-slice1 | 9ec46d9 | 5 | landed as 990b43c; identical | residue; remove |
| wt-r38-slice2 | 990b43c | 10 | landed as efcfb78; identical | residue; remove |
| wt-r38-slice3 | efcfb78 | 13 | landed as 8420542; identical | residue; remove |
| wt-r38-slice4 | 8420542 | 9 | landed as ba6435b; worktree lacks the SUPPORTED_VERSIONS check the branch has | residue; remove |
| wt-power | 045d015 | 3 | landed; the three files are byte-identical to the branch (POWER-ANALYSIS memo, probes/power.py, tests/test_power.py) | residue; remove. Re-powering for WP5 is new work under WO-STAT-002, not this worktree. |
| wt-p6-scorer | fe5f94f | 2 | landed as b080e41; worktree lacks the widened value pattern and the moved fixture | residue; remove |
| wt-hybrid-period | c349739 | 8 | landed as 2bb2761; worktree behind the branch | residue; remove |

Rule drawn from this (R38 form): a disposition table names the landing commit for every worktree it lists, and "land" is written only after a diff against that commit shows work the branch lacks. Corollary from the cleanup (the Deputy, R38(e)): a difference from the branch tip is not evidence of anything, since a landed change diverges from the tip whenever a later commit touched the file; and absence from the tip needs the history checked before it means unlanded, since a superseded line is absent for a reason. All ten scratchpad worktrees were removed on 2026-09-05 late evening after that check; one paragraph absent from the tip was traced to the fix that made it obsolete.

## 6. Draft messages, held until the Director approves

Each is written for the GitHub issue channel; the Deputy opens the issues named "new" below. Nothing has been sent.

### 6.1 To the Deputy Chief

Subject: Programme revision after the hosted-lens results; four implementation lanes and three dispositions.

What changed: read `under_review/HOSTED-JLENS-QWEN35-4B-RESULTS-2026-09-05.md` first, then `pending/03-REVISED-PLAN-2026-09-05.md` sections 1 to 4. The reference-recipe lens for our checkpoint is on this machine (HF cache, `neuronpedia/jacobian-lens`, `qwen3.5-4b`) and is the reading instrument from now on; our finite-difference path is legacy for reading. The band is layers 13 to 29. World A stands on the new instrument.

Lanes, in order, each a separate issue and a separate worktree, Codex implements, no implementing subagents, READY TO COMMIT with the fixture named:
1. WP1 (new issue): `pipeline/hosted_lens.py`, `--lens hosted` on the sweep, full top-k in the artifact, Holm family per instrument, registry `lens:` block, fixture = today's 42-case table within one case. Supersedes #78; closes the artifact parts of #69 and #72.
2. WP2 (new issue, after ruling R40b lands in the wiring map): `probes.band_layers` in the registry with the partner derivation unchanged.
3. WP3 (new issue, design from the Head first): the transport-weight hooks and statistics; reference implementation of the read weight is the verified numpy in the derivations memo.
4. WP4 (issue 82): the first-render fix is landed (a922013); the read-share pre-check from WP3's hooks; then the 2 by 2 at the size the pre-check licenses, or recorded as not worth running if the share is below the stated floor.
Dispositions (D3, once approved): land wt-template-prefix, wt-r38-slice1 to 4, wt-p6-scorer, wt-hybrid-period through the normal gate; hold wt-power; keep wt-s3.
Constraints unchanged: the Chief gates every commit reading every hunk; per-run lifts for anything that runs the agent; pre-registered readings before runs.

### 6.2 To the Head of Interpretability

Subject: Instrument switch and four probe revisions; two designs requested first.

What changed: `under_review/HOSTED-JLENS-QWEN35-4B-RESULTS-2026-09-05.md` and `under_review/CHIEF-JSPACE-PAPER-READING-2026-09-05.md`. Three points bear on your suite directly. The hosted lens replaces the JVP readout and gives full-vocabulary ranks at every layer for one matrix product. The band is layers 13 to 29, so the registry fractions move (R40b pending). The Base carries the Assistant-reaction content on user tokens as much as the post-trained model, so the base-versus-post-trained contrast on reaction words is not a usable Thread 1 signature on this lineage.

Requested, in order:
1. WP3 design (first): the transport-weight measurement on both channels from the Chief's section-9 brief plus the read-weight derivation (derivations memo, section 6 of the paper-reading memo); pre-registered reading as drafted in the plan or amended with reasons.
2. WP7: the assistant axis decomposed against the hosted dictionary with a clamped steering test; the three self-monitoring probes on both checkpoints; adapter deltas read through the lens; state probes moved to band layers. One pre-registered reading each.
3. WP9: the causal-privilege test on our two-hop prompts with the clamp.
4. Review chains under R36: WP1 and WP5 are interpretability slices; your verdict on them.
Constraints: designs to the Chief before dispatch; runs under per-run lifts; no implementing subagents.

### 6.3 To the Proxy

Subject: Programme revision; nothing to implement; two relays.

The plan at `pending/03-REVISED-PLAN-2026-09-05.md` supersedes the queue in the derivations memo. Two things to carry: the Director's decisions D1 to D6 once given, into the wiring map as R40; and, if D4 is granted, the standing lift for EXP-005 trace collection with its cap. The Proxy standing order is unchanged: lifts relayed through you carry the Director's approval; substantive changes still gate on the Chief.

## 7. Constraints restated

No implementing subagents. The Deputy commits; the Chief never commits or pushes. Every commit gated on a read of every hunk. Per-run lifts for agent deployment until D4 says otherwise. Pre-registered readings and WO-STAT-001 power before any run. R38 and R39 discipline on every record. The Director's direct instruction governs over any relay.

## F9 (2026-09-06, 01:30): where attention reads is an inverted U on layers 20 to 24, converging with the lens fan-out

On retrieval-heavy text (RETRIEVAL-CHANNELS-2026-09-06, 72 items, both channels under one span statistic), attention's concentration on the answer's known span by layer is 1.8, 2.6, 2.6, 3.5 at layers 4 to 16, then 15.2 at 20, 19.5 at 24, 9.1 at 28 and 6.3 at 32 (medians, as multiples of the random-query null). The peak coincides with the hosted-lens fan-out region at layers 18 to 25 and sits on the attention members of R41e's pairs 19/20 and 23/24. Two instruments with no shared machinery land on the same place. The recurrent heads that could physically have retrieved at that range read the span at the uniform share on every kind (F8 confirmed where its author said it had not been tested); the mechanism is the read operation, softmax competition against a linear sum over a superposed store, not query mobility. Bears on WP5: the distance curve reads at exactly those pairs and should be most sensitive there.

## WP12 (2026-09-06, 02:20; the Director's flag): which attention mediates entry to the J-space — the paper's broadcast heads on Qwen3.5-4B

**The paper's claim, read verbatim (§4.3.2, "Broadcast Across Tokens").** For an attention head H and a
population P of unit directions, two weight-based metrics on the head's OV circuit: *gain*, the mean
of ‖W_OV v‖ over v ∈ P normalised by the head's gain on isotropic random directions; and *label
preservation*, the mean reciprocal rank of cos(W_OV v_i, v_i) among {cos(W_OV v_i, v_j)}_j,
contrasted against the same statistic on random directions so that a head that copies everything
indiscriminately scores zero. Broadcast heads for P are the top 1 percent of workspace-layer heads
by the worse of the two ranks. Populations: the J-lens vectors J; J under a fixed random orthogonal
rotation (same spectrum and pairwise geometry); SAE decoder directions in three kurtosis strata; MLP
output-weight rows. The heads selected for J separate cleanly from every control on both metrics and
concentrate in the first half of the workspace layers, where the J-space's effective rank is lowest.
Ablating them (zeroing their outputs at every position) drops recall@25 of the J-lens readout at
mid-workspace layers to 0.67 against 0.86 for layer-matched random heads, changes the top-1 next
token at 5 percent of positions against 2, cuts the injected-thought report rate from 0.54 to 0.09,
and reproduces about a third of the experiential-language drop of full J-space ablation. The
appendix (A.20) adds the aggregate composition of J-space directions with attention weight matrices,
and A.24 shows individual heads whose Q, K, V and O projections have interpretable J-lens readouts.

**On our model.** The population J at band layer L is the set of residual directions the hosted lens
maps to unembedding directions, J_L^T u_w for a sample of tokens w, from the 31 fitted matrices we
already hold. The candidates are the 128 attention heads (8 blocks × 16 query heads; GQA over 4 kv
heads) and, because this model is a hybrid, the 768 recurrent value paths (out_proj ∘ in_proj_v per
head), which the paper had no occasion to test: does the recurrent channel broadcast J-space content
at all? Stage 1 is weight arithmetic with no forward pass: dequantise W_V and W_O, form each head's
OV map, compute gain and label preservation against J, J-rotated and the MLP output rows, rank, and
name the broadcast heads per channel with their layers. Stage 2 is the ablation under the lift:
zero the named heads at every position on a pretraining-style corpus and the EXP-001 cases, and
measure recall@25 of the hosted-lens readout at the band layers against layer-matched random heads,
plus the top-1 change rate; the injected-thought analogue is EXP-002's injection arm. Two links to
tonight's record are the predictions to write before running: F9 put attention's reading of far
content in layers 20 to 24, and the strongest retrieval heads sit in blocks 19 and 23; if the
broadcast heads are the same heads, entry to the workspace and retrieval are one operation, and if
they are different heads at the same layers the two are separable. The recurrent-sink finding
predicts that recurrent value paths in long-gate heads carry the opening of the context rather than
J-space content.

**Capacity.** Stage 1 needs no model run and can be computed while issue 81 and 87 are implemented;
stage 2 is one model load. The Head reviews the design before stage 1 runs (R38(g)); the Director's
instruction is to pursue two threads in parallel, implementing one while running the other, and
this is the second thread beside training efficiency.


**WP12, amended on the Head's review (02:40; adopted).** (1) *No percentile selection.* The paper's
top one percent is a few hundred heads at their scale and about five at ours, hostage to which metric
is worse on the day; every head exceeding its control distribution is reported, as n of N, with the
whole ordered list in the artifact (the named-exception rule of WP3, for the same reason: a quantile
over a small population manufactures a selection). (2) *The two channels' output maps are different
objects.* Attention's OV is a fixed linear map from residual to residual, W_O composed with W_V per
head. The recurrent path has the state between its value projection and its output projection, and
after the state an input-dependent gated normalisation; a weight-only composition out_proj ∘ in_proj_v
omits the gate and the normalisation, which are exactly the parts that are not weights. Stage 1
computes that composition and says so in every table; the two columns are not commensurable and are
not read as if they were. (3) *Stage 2 carries the paper's specificity control.* The next-token change
rate is reported beside recall@25 under the same ablations, with the threshold stated in advance: a
recall drop counts only if the next-token change rate stays within twice the layer-matched random
control's. (4) *The second prediction restated in testable form.* The recurrent value paths should
show lower alignment with the lens directions than the attention paths at matched layers; the sink
finding is motivation, not evidence, since it measured which positions long-gate heads read and not
what directions they write. (5) *Dependency stated.* EXP-002's injection arm is not built and is held
behind WP3 and its pre-check; stage 2 as a whole does not wait on it: layer-matched random-head
ablation with recall@25 and the next-token change rate is a complete stage 2 on its own, and the
behavioural arm follows when the arm exists, with its mapping onto the paper's injected-thought
report stated then rather than asserted now. Stage 1 runs on the fixed design; the first prediction,
whether the broadcast heads are the retrieval heads of blocks 19 and 23 or different heads at the
same layers, is the one that matters.

**WP12 stage 1b (2026-09-06, 03:55; the Director's hypothesis): the out-of-band relays as composition partners of the band relays.** The Director's flag: layers 4 and 8 are outside the J-space band, so relays there "must be coupling heads that couple to heads in later layers". That is composition in the transformer-circuits sense (Elhage et al. 2021: an early head's OV output read by a later head's Q, K or V matrix; the induction circuit is a previous-token head K-composing with an induction head). The hypothesis in testable form: the early relays write, at each position, labels that the band relays' keys or values later read, which is also what the retrieval result requires, since the band heads read the answer span at question time and something wrote a matchable label at the fact's position. Test, weight arithmetic like stage 1: for every pair (A early relay, B band head), the Q-, K- and V-composition scores ‖W_X^B W_OV^A‖_F normalised by the factors' norms, against the distribution over random (A, B) pairs at the same layers and over twenty rotations of W_OV^A; report which band heads the early relays compose with, whether the retrieval heads (blocks 19 and 23) are among them, and which composition kind carries it. Separation control, because the lens at layer 4 sits close to the embedding: measure what the layer-4 relays' OV preserves on token-identity directions (embedding rows) against lens directions, so a previous-token or duplicate-token copy of token identity is not read as workspace content. Predictions written before the run: (1) the layer-4 relays K-compose with the band relays above the random-pair null, and the strongest partners include the retrieval heads of blocks 19 and 23; (2) the layer-4 relays preserve token-identity directions at least as well as lens directions, so their "relay" status is address-writing rather than workspace content. Runs after the null-distribution rerun of stage 1 fixes the sets; the Head reviews (R38(g)).

## F10 (2026-09-06, 05:15; from WP12 stage 1b): the lens converges on the unembedding through the band, so a lens reading near the top of the band is partly a logit-lens reading

The mean absolute cosine between a hosted-lens direction (J_L^T u_w) and the same token's identity direction (u_w) is 0.10 at layer 4, 0.41 at layer 20 and 0.69 at layer 28 (WP12-BROADCAST-HEADS-2026-09-06, separation run; the full profile by layer is written by the orthogonalised rerun). The degeneracy R43a invoked to exclude layer 32 (the lens is the identity there) is a ramp, not a cliff: by layer 28 lens directions and unembedding directions are largely the same directions, so preserving workspace content and preserving output-readable content are close to one question at the top of the band. This is the convergence of the two lenses the hosted-lens memo found at the top of the stack, measured directly rather than inferred from their agreement. Consequences, each to be carried by the work package it touches: WP5's rank statistics at layers 24 and 28 are partly logit-lens ranks; WP7's decompositions at those layers partly decompose output-readable content; WP9's clamp at the top of the band clamps partly the logit lens. Any result at layers 24 to 28 states the cosine at its layer beside it. In WP12 the consequence is immediate: layer 28's nine relays, which refuted prediction 1, were selected against a lens two-thirds the unembedding, and the selection is rerun on the orthogonalised population at every layer before that verdict stands.
