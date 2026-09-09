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
