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
spans of the prompt — the task statement, the most recent tool result, the previous note, the
hidden-result placeholders, everything else — summarised by span, by layer type (local against
global) and by distance (at most 1,024 tokens back against further). **Structural gate:** a local
layer places zero mass beyond 1,024 tokens, or the mask is not what the registry says and the
experiment stops. The global layers are read from the checkpoint's `layer_types` and the attention
modules' `is_sliding`, which must agree: five global layers in the 4B (blocks 6, 12, 18, 24, 30) and
eight in the 12B, every sixth block, checked on the device. Reported per model. This is the reviewer's carrier question measured directly,
and the ruling of this morning in the run orders cites it.

## W-4. Report or computation

*Is the action already readable where the note begins, so the note reports a decision made, or only
after the note is written, so the note is part of the deciding?*

Per decision: the lens's distribution over the six tools at `P_note` against `P_act`, at the last
layer and at each model's median ignition depth from W-1; the **decided-before-the-note fraction**
— decisions whose taken tool is already top of the six at `P_note` — per family and step, against
the tool prior. On the 300-sample: the same readout at **every token of the note**, giving the
trajectory of the decision through the note, and the note position at which the taken tool first
becomes top, as a fraction of the note's length. The falsifier of "the note is a report" is the
taken tool becoming top only during or after the note beyond what the prior explains.

## Cost, schedule, record

After the 12B's third chunk (about 04:00Z): merge the chunks (CPU); the 4B capture pass (7,629
forwards at width 1 in float32, about 40 minutes); the 12B pass (about 1.7 hours); the 300-sample
attention pass (minutes); analyses on the CPU. Record: `research/records/WORKSPACE-EXPERIMENTS-2026-09-10/`,
with the archive's manifests, the scripts as run, and one README per experiment stating its three-state
outcome. Nothing is cited before Codex's audit. The steering pilot the Director will discuss this
evening is a separate order and is not started by this one.
