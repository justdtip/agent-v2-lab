# Work order: workspace experiments, set 1 — four global-workspace questions asked of the agent at its decision positions, through the full-resolution float32 lenses

**Chief, 2026-09-10, on the Director's instruction of this morning: after the 12B's last chunk,
measure with the full-resolution Jacobian lenses on both models, pursuing experiments the Chief
defines, in conjunction with Codex's reviews; choose them for what the Director will find
interesting, which is the philosophy of mind of an agent that has a workspace.** Owner of the runs:
the Chief, under the chief seat, from `/workspace/chief/`. Design reviewed by Codex before any
capture is read (WS-A order, standing request 3); each result audited after (request 4). Nothing in
this order computes a plan-progress estimand; those stay sealed under their own order.

## 0. What is common to all four

**Corpus.** The rendered agent corpus on the card, `/workspace/rendered-corpus/agent_v2e-gemma3-4b/`,
its 7,629 distinct decisions (the plan-progress capture set, digest `1a7cfbdd…9709a`; the 348 chat
rows excluded), 1,128 episodes, twelve families, six tools. Counted today: `read_file` 4,315,
`finish` 1,128, `replace_text` 667, `calculate` 611, `search_files` 490, `list_files` 418; prompts
of 4,738 characters at the median; 5,396 decisions have prompts of at most 6,000 characters (about
1,500 tokens); the note before the tool call is 111 characters at the median.

**Models and arithmetic.** Gemma 3 4B and 12B, coherent float32 at width 1 — the path the lenses
were fitted on, so no cross-path term is involved and nothing is refused. Eager attention, TF32
off, highest matmul precision, determinism pinned; all stamped.

**Lenses.** The 4B float32 exact lens over 201 prose rows (`/workspace/chief/out/lens4b-f32`,
sha256 in the record) and the 12B lens merged from three chunks over the same 201 rows
(`merge_chunks.py`, `JacobianLens.merge`, weighted by prompt count). Both are **short-horizon**:
fitted at 128 tokens of prose and read here at positions up to about 3,600 tokens of agent
transcript. Every figure carries that, and experiment W-C measures what it costs.

**Positions, two per decision.** `P_note`: the last prompt token, where the model's turn begins and
its progress note is written — the decision position of the plan-progress order. `P_act`: the token
immediately before the tool-name token inside the completion's JSON, whose readout produces the
tool name, reached by teacher-forcing the expert's completion. Both are token indices recorded
after tokenisation in the manifest.

**Captured, once per model:** the residual at every layer at `P_note` and `P_act`, float32
(the 4B: 2.6 GB for all decisions; the 12B: 5.5 GB), kept as an archive outside the checkout under
`/workspace/chief/captures/<entry>/`; the model's own logits at both positions reduced to the
probability over the six tool-name first tokens and the top-k; and, for a **stratified sample of
300 decisions** with prompts of at most 1,500 tokens (25 per family, by rule, seed in the manifest),
the attention probabilities from `P_note` and `P_act` at every layer and head, and the residual at
**every token of the note**.

**Readouts.** Through each lens: `residual @ Jᵀ`, then the model's final norm and unembedding,
reduced per layer to the probability of the taken tool's first token, its rank, the top-1 token,
and the distribution over the six tool names; the logit lens (no transport) computed beside it at
every layer as the control that needs no fitting. **The tool vocabulary is the six names**, so
"a competitor" means another valid tool, never a synonym.

**Discipline.** The unit of any confidence statement is the episode; distributions are reported as
median with the tail (minimum, maximum, and the count above a stated line), never a median alone;
the tool prior — each tool's corpus frequency — is the null for every "which tool" figure; three-
state reporting; the capture manifests bind the lens hashes, the corpus digests, the token
positions and the arithmetic settings; the design (this order and the scripts' hashes) is committed
before a capture is read.

**The mass floor.** At note positions the model is writing prose, so the probability mass on the
six tool-name tokens is small by construction, and the six-tool distribution there is a
conditional reading — which tool, were it to act now — obtained by renormalising small numbers.
Every six-tool reading therefore carries its raw mass, and a reading whose mass is below **1e-3**
is reported as *unresolved*, not as a preference; W-4's trajectories and W-1's `P_note` profiles
are computed only over resolved readings, with the unresolved count stated beside them. The floor
is declared here, before any capture is read (the ladder audit's R4, applied).

## W-1. Ignition and competition

*When the agent commits to a tool call, does the winning action's readout rise gradually through
the depth, or ignite — a late, sharp amplification over a few layers — and how often does a
competing tool lead in the middle layers before losing late?*

Per decision, at `P_act` and at `P_note`, the layer profile of the lens's probability of the taken
tool. **Ignition depth**: the first layer at which it exceeds 0.5, as a fraction of depth, or
*never*. **Ignition width**: the number of layers between its first crossing of 0.1 and its first
crossing of 0.9, or *never*. **Competition**: the existence of a layer, before the last, at which a
different tool is the top of the six; recorded with the competitor's name and the layer span.
Counts and distributions per model, per family, per step; the logit lens beside; the tool prior as
the null for competition (a competitor that is simply the most frequent tool is reported as such).
Perturbed variants reported separately, because a corrective decision's competitor may be the
failed action re-tried, which is the interesting case.

## W-2. Global availability

*Is the chosen action carried in one subspace available across all task families, or in
family-specific representations?*

On the lens-transported residuals at `P_act`, a rank-r linear readout of the taken tool (six
classes) fitted on the train split's episodes of a **subset of families** and evaluated on held-out
episodes of the **other families**, against the same readout fitted and evaluated within family;
rank ladder r ∈ {1, 2, 4, 8, 16}; the permutation null (labels permuted within family) at every rank;
the raw residual per layer (no transport) as the control for the lens's contribution; folds by
episode, stratified, seeds in the manifest. Transfer at the null is the falsifier of a common
subspace. Labelled an encoding comparison, as the plan-progress order labels its E1.

## W-3. Broadcast and the carrier

*At the decision position, where does the model draw from — the latest tool result, its own
previous note, the task statement, or the placeholders of hidden results — and through which
layers?*

On the 300-sample: attention mass from `P_note` and from `P_act`, per layer and head, onto tagged
spans of the prompt — the task statement, the latest and the older tool results, the hidden-result
placeholders, the previous and the older notes, the previous and the older tool calls, and the format
tokens — summarised by span, by layer type (local against
global) and by distance (at most 1,024 tokens back against further). **Structural gate:** a local
layer places zero mass beyond 1,024 tokens, or the mask is not what the registry says and the
experiment stops. The global layers are read from the checkpoint's `layer_types` and the attention
modules' `is_sliding`, which must agree: five global layers in the 4B (blocks 6, 12, 18, 24, 30) and
eight in the 12B, every sixth block, checked on the device. Reported per model. This is the reviewer's carrier question measured directly,
and the ruling of this morning in the run orders cites it.

## W-3b. The carrier masked

*The D-CRO measured (`carrier_ablation.py`, pre-registration §4.3) that on this corpus the note
carries the step five times over in one sentence and no text-removal rule strips the progress
without changing the next-action clause for eleven of twelve families, so the reviewer's carrier
ablation cannot be done in the text without changing the task. It can be done in the attention.*

On the 300-sample, per decision, a second and third forward beside the ordinary one: the carrier
spans masked **as keys** for every query at or after the current turn's start (the task statement
never masked; kinds selectable — the notes, the calls, the results, the hidden placeholders, each
alone and all together), and a **random-span mask of equal token count** as the control; and two further arms that mask **the current turn's note** as keys for the queries after it, alone and together with the carriers, so that the action position reads without the note the model has just written — the report-or-computation question in intervention form: an action that survives was decided before the note, one that snaps to the prior was read off it. Read at
`P_note` and `P_act`: the model's own six-tool distribution and the lens readout per layer, against
the unmasked forward. A taken action that survives the mask was carried in the residual before the
mask point; one that snaps to the tool prior was re-read from the carrier; the random control says
how much any mask of that size costs. Reported per kind, per model, with the unmasked and the
random arms beside, three-state. **Not the pre-registered E2 arm**: if it behaves, it becomes E2's
ablation arm by a later ruling; the pre-registration keeps *arm not constructible in the text* and
promises nothing here.

## W-4. Report or computation

*Is the action already readable where the note begins, so the note reports a decision made, or only
after the note is written, so the note is part of the deciding?*

**Primary, by decodability.** The model's own next-token distribution at `P_note` puts negligible
mass on tool names — it is about to write prose — so "which tool would it take now" has no resolved
reading there by the mass floor (measured on the test rows: about 1e-14). The question is therefore
asked as W-2 asks its own: the rank-r linear readout of the taken tool, fitted on the
lens-transported residual at **`P_note`** and evaluated held out by episode, against the same
readout at **`P_act`**, at every rank of the ladder, with the permutation null and the tool prior
at each; the raw residual per layer beside the transported one. If the taken tool is decodable at
`P_note` above the null at the ranks where it is decodable at `P_act`, the note reports a decision
already made; if it is decodable only at `P_act`, the note is where the deciding happens; the gap
between the two is the note's share. By family, step and variant; the unit is the episode.

**Secondary, by readout.** The six-tool distribution through the lens at every token of the note on
the 300-sample, each token carrying its mass and reading as *unresolved* below the floor; the
position in the note at which the taken tool first becomes the resolved top, as a fraction of the
note's length, and the count of unresolved tokens per note. Reported for what it is: a reading of
what the model would emit were it to act mid-note, mostly unresolved by construction.

## Cost, schedule, record

After the 12B's third chunk (about 04:00Z): merge the chunks (CPU); the 4B capture pass (7,629
forwards at width 1 in float32, about 40 minutes); the 12B pass (about 1.7 hours); the 300-sample
attention pass (minutes); analyses on the CPU. Record: `research/records/WORKSPACE-EXPERIMENTS-2026-09-10/`,
with the archive's manifests, the scripts as run, and one README per experiment stating its three-state
outcome. Nothing is cited before Codex's audit. The steering pilot the Director will discuss this
evening is a separate order and is not started by this one.

## Rulings on Codex's design review, Chief, 2026-09-10 UTC — `WSA-WORKSPACE-AND-PREREG-2026-09-10` (`3494a2c`): this pass is an observational screen, and every quantity is renamed for what it measures

Accepted in full. The captures proceed as raw data; nothing below changes what is captured, and
nothing is read until these rulings are in the analyses. The mechanism claims — ignition,
functional availability, causal broadcast, report versus computation — are **reserved** for the
steering experiments the Director will order; this pass describes.

**W1, the fitting domain.** The lenses are fitted at 128 prose tokens (positions 9 to 128, upstream's
default selector, the source-mean/target-sum reduction; the archive hashes in the manifests) and
read at agent positions up to about 3,600 tokens; matched precision and width remove the cross-path
confound and do not license that transfer. "Nothing is refused" is struck. Every reading carries
its position, and the position distribution is preserved. The sentence naming "W-C" referred to an
experiment that was dropped from the final order; it is replaced by **W-5, lens validity by
position**: the agreement between each lens's last-layer readout and the model's own readout at
`P_act`, by position band (to 256, to 1,024, to 2,048, beyond), which quantifies the extrapolation
and is the number that decides the long-horizon lens. A lens score is not a calibrated probability
of an action; log-odds travel beside every conditional probability.

**W2.** The W-1 quantities are renamed *first confidence crossing* (0.1, 0.5, 0.9 on the six-tool
conditional mass, with the full-vocabulary probability stored beside it as `six × mass`) and
*readout competitor*; every individual depth profile is retained; oscillations, missing crossings,
equal maxima, unresolved layers and initially confident profiles are handled by frozen rules (first
crossing in depth order; a tie is a miss; an unresolved layer is skipped and counted). A crossing
width of one layer is consistent with a linear logit trajectory and is not evidence of ignition.
The input-mixture sweep with a readout-independent activation measurement is the ignition experiment
proper and is a new arm for the Director to approve.

**W3.** In W-2, r is the dimension of a PCA projection fitted on the training fold, with a ridge
readout on top; it is a representation constraint, not the rank of the six-class contrast, whose
softmax-relevant rank is at most five, and the ladder is relabelled accordingly. Family subsets,
folds, training-only centring, the regulariser and selection are frozen in the manifest. The
within-family permutation null cannot detect cross-family sharing where a label is constant within a
family; a **transfer null** is added — labels permuted within family on the training families, the
readout evaluated on the held-out families — as the null for the transfer design. The result is
labelled *cross-family linear decodability of the expert action*, not functional availability;
null-level transfer does not falsify a common subspace, and successful transfer may be a shared
tool-name encoding. For an invertible lens the rank-r readouts on `Jh` and on `h` span the same
class; a difference between them is attributed to conditioning and regularisation, not to new
information.

**W4.** W-3 is renamed *direct attention to tagged spans*. Attention weight is a routing weight,
not a causal share, and the global layers are the only **single-edge** routes beyond the window,
not the only end-to-end routes — a source 1,800 tokens back can reach the decision through two local
hops. Per-head distributions, span sizes, distances and tokenisation boundaries are preserved, and
"the carrier's share" is struck from both state-variable orders in favour of the direct-attention
share. The five and eight global layers are re-measured from the checkpoints at capture time.

**W5.** The continuation is the **expert's** teacher-forced completion, and the label is the *expert
action* wherever Gemma might disagree; a first tool-name token is not an executable action. W-4's
primary is relabelled *early versus late linear decodability of the expert action*: neither a
probe's success nor its failure dates the decision, since report and computation have identical
clean traces. The change of W-4's primary was informed by a mass measurement on three rows of the
test split (task ids in the record), run on the CPU as a pipeline test, whose only role was to
show the next-token reading unresolved at `P_note`; that provenance travels with the primary.
W-3b's branches are narrowed: survival under a mask shows the masked edges were not necessary for
that outcome under that intervention and does not date the decision; a change under a mask is
reported as the distribution change it is, not as re-reading. The same-kind arms (older against
previous note, older against previous call) are the role- and contiguity-matched comparisons; the
random-token arm is a token-count control only and is labelled so; the cut is verified with a
deliberately leaky mask on the first sampled row, and the two-hop relay is named as an open route.

**W6.** The capture and analysis scripts' hashes are bound in the record with the sample rule
(25 per family by sorted key, every k-th, seed 20260910, a balanced family sample and not the corpus
mixture), the selected keys, and every missingness count; the six tools' first tokens are asserted
distinct; W-1's thresholds use the six-token conditional mass, with the full-vocabulary probability
stored; every tail carries its unit count and its unresolved count. The residual archive is
5.31 GB for the 4B and 11.25 GB for the 12B at two positions, correcting §0. The rendered prompt's
byte digest and the token ids consumed are bound per cell at capture time; no field named
`prompt_sha256` is reused across the two programmes.

**S1, for the bridge order.** Matching the Gram spectrum does not match the signed overlap geometry
(negating one atom preserves the spectrum and changes the nonnegative cone). Ruled: Stage B's null is
a **seeded orthogonal rotation of the whole dictionary**, preserving the full signed Gram matrix and
atom norms while randomising orientation relative to the residuals, with budget, solver, selection
and stopping rule identical, and the hypothesis it tests stated; the claim that overcompleteness
alone guarantees reconstruction is measured under that null, not assumed.

## Rulings on Codex's closure review, Chief, 2026-09-10 UTC — `WSA-CLOSURE-AND-WORKSPACE-AMENDMENTS-2026-09-10` (`232fe6a`)

**M1, and it corrects a sentence of mine.** An invertible lens preserves the class of unrestricted
linear readouts, and a freely chosen rank restriction transports through the inverse; it does
**not** preserve the class produced by a PCA fitted and truncated separately on each representation.
Codex's two-dimensional example is exact: the lens `diag(1, 3)` changes which coordinate PCA keeps
and takes a 50% readout to a perfect one with no information added. So a difference between the
lens-transported and the raw readout at rank r in W-2 is **the information the specified pipeline
retains**, and is described as that — never as information the lens created and never as
functional availability. The sentence attributing such differences to conditioning and
regularisation alone is struck.

**M2.** The transfer null as specified — labels permuted within family on the training families —
leaves every training label unchanged wherever a family uses one label, and is marked
**uninformative for family-constant strata**. The null for the transfer question is a **per-family
random bijection of the six tool labels** applied to the training families' labels, the held-out
families evaluated on their true labels: it preserves each family's label structure and destroys
the cross-family correspondence of label identities, which is the alignment the transfer design
hypothesises. Its hypothesis is stated in the manifest; the within-family permutation is retained
only for the within-family design, for the conditional association it actually tests; unrestricted
shuffling is not substituted.

**W-5's scope.** It is an operational readout diagnostic and never a derivative-validity or
refitting gate: agreement with the model's current output does not identify a derivative (the
scalar tail `x² + 3` at 2 has derivative 4, and the wrong map 3.5 agrees perfectly with the output).
It is reported **per source layer** and per position band, with the exact readout rule, the mass and
unresolved counts, and the comparison it makes: whether the six-tool argmax through the map of that
source layer agrees with the model's own six-tool argmax at `P_act`.

**R1 and R2, the D-CRO's, before the capture pass runs:** resume validates each requested decision
against its corpus row before either the reuse or the new-capture branch, binding a reused cell to
the request's identity and enumeration metadata; and resume prepares the exact model input by
tokenising before deciding to skip, comparing the token digest and length with the existing cell
and passing those ids to any forward. Until both exist, resume's `verified` is described as
checkpoint, semantic and rendered input, and file integrity — not consumed-token equivalence — and
the pre-registration's sentence saying otherwise is narrowed.

## Rulings on Codex's run-capabilities review, Chief, 2026-09-10 UTC — `WSA-RUN-CAPABILITIES-2026-09-10` (`712f78c`): the current-note arm was empty, repaired before its pass ran

**P1, confirmed and repaired.** The current-note arm of `workspace_w3b.py` masked nothing: the whole
completion prefix was labelled `note`, syntax and the supplied tool-name token included, and the
query slice began one past the last note token, which is one past the end of the sequence. Confirmed
three ways — by reading the slice, by Codex's model-free reproduction, and in the one-row smoke
output, where the current-note arm's readouts equal the unmasked arm's bit for bit and the combined
arm's equal the carriers arm's. The repair, frozen in the record's `scripts/` with its digest: the
note is the completion text before the ```json fence, its trailing newline included, cut as the
previous-note spans are cut; the keys are the completion tokens whose character offset starts before
the fence; the queries are every token from the fence onward through `P_act`, so no relay through
the syntax between the note and the action stays open; the supplied tool-name token is labelled
`tool_name` and is never read; a completion with no prose before the fence records the arm as
*skipped*, never as a no-effect result. The guard now refuses instead of logging: every masked arm
blocks at least one edge at `P_act`; the attention on every masked edge, over all queries, layers and
heads, is exactly zero; the leaky control passes; the current-note arm leaves `P_note`'s readouts
bit-identical to the unmasked forward's; the in-context tool-name token equals the standalone first
token the six-tool readout uses, and the six are distinct; the capture and the pass read the same
corpus, by digest. Rows are written as they complete. Codex's further request — a fixture in which a
deliberately misplaced cut fails for its stated reason — is owed beside the script and not yet
written.

**The smoke test was itself mismatched, and the new guard found it.** The earlier one-row smoke test
of W-3b took positions from a capture made from the test split while taking its prompts from the
concatenated corpus, and that capture predates the doubled-BOS fix, so its positions are one too
high. The old script truncated silently; the repaired one refused at the tool-token check. The
production capture uses the fixed script and the concatenated corpus on both sides; when it closes,
every row's token after `P_act` is checked against the tool's first token, and the count is in the
record before any W-3b reading.

**Deviations from this order in the capture as run, recorded, not repaired mid-run.** (i) On the
300-sample the capture saves six-tool readouts at every note token, not the full residuals this
order asked for; the 12B capture pass, not yet started, saves them, and a supplementary sample-only
pass adds them for the 4B after its W-3b pass; until then W-4's secondary is the readout its text
describes. (ii) The note trajectory runs over the whole completion prefix; the analysis splits it at
the fence into prose and syntax positions by the repair's offsets rule and reports the crossing
within each. (iii) The behavioural measure at `P_act` is the tool's first token; the complete call is
scored only in the steering pilot. (iv) Attention arrays are stored as float16; the in-forward gate
is float32 and is what the zero-attention claim rests on. (v) Per-cell token digests are not in the
capture manifest; the binding is the corpus digest plus positions re-derived deterministically at
analysis, and it is checked there.

**Merge weighting.** `merge_chunks.py` weights chunks by requested rows; the condition under which
that equals the fitted-prompt weighting is that no row was skipped. c1 and c2: 70 requested, 70
fitted, 0 skipped each; c3 is checked the same way at its end, before the merged lens is used.

**Provenance binding.** The on-disk sources predate the processes that loaded them (the capture
script last modified 23:51Z, the capture started 00:22:51Z; the fit script 14:30Z, chunk c3 started
00:24:21Z) and neither was modified after; the record carries their digests.

**Codex's three distinctions are the record's vocabulary from here:** (a) content a researcher can
decode, (b) content the model can use, report or manipulate, (c) a model of its own access. This
screen is (a), with a restricted (b) from the carrier arms once their gates hold; (b) proper is the
steering pilot, which is the Director's to order.
