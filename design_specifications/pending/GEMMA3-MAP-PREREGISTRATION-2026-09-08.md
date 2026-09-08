# Pre-registration: the Gemma 3 representation map

**Written before any record exists, as the house rule requires.** The Chief's, on the Director's
order to map internal representations across a variety of tasks. Nothing below may be changed once
a record has been read; a change after that is an amendment, dated, with the reason.

## What is being run

The live-lens pilot on Gemma 3 4B, base model, no adapter, reading residuals through the hosted
Jacobian lens **while the model executes tasks**. Two stages: an instrument check at eight layers,
then the map at all thirty-four. Fifteen episodes — one per task family across all twelve, plus
three chat episodes including one whose context exceeds the sliding window.

The instrument-check stage is **not** a result. It is read only for whether the path worked, and
its numbers are not reported as findings.

## The rules, stated so they transfer

Every rule below is written in terms of the model rather than of our code or this checkpoint,
because the same rules must apply unchanged to the 12B repeat and to the next family. **A rule that
names a layer index is a rule that does not transfer**, so depth is a fraction and a layer's role is
its attention span, not its index.

### Primary reading

**Foreknowledge**: the share of emitted tokens whose competition rank in the lens distribution,
read `h` positions earlier, is within ten. Reported as a grid of **depth by horizon**, faceted by
span — note, call, observation, chat prose — with the **final layer's value as the base rate on
every row**, since the final layer's lens is the identity and its share is what the readout gives
with no transport at all.

The primary comparison, fixed here: **within agentic episodes, the share at horizon one in *call*
spans against *note* spans, per layer.** Calls are where the model commits to an action; notes are
where it deliberates. If commitment is visible earlier in depth than deliberation, that is the
first thing this instrument can say about how the model completes a task.

### Secondary reading, and the condition without which it is void

**Globally-attending layers against window-attending layers, at horizon four.**

A block's span is read from the model's own dispatch, not from a period rederived by us. For this
checkpoint that makes the globally-written residual layers **6, 12, 18, 24 and 30** in the
repository's one-based convention, and the rest window-written. This must be recorded per layer in
the manifest, because the layer-family machinery currently reports this backbone as dense — true of
its module kinds and false of the model — and a reader of the artifact must not inherit that.

**This comparison is restricted to positions beyond the sliding window and is void without them.**
A windowed causal mask is vacuous when the sequence is shorter than the window, so below it the two
kinds of layer are the same layer and any difference is noise. Report the count of qualifying
positions beside the comparison; if it is small, say so and draw no conclusion.

**A further caveat that cannot be removed by any amount of data**: the lens itself was fitted at a
sequence length of 128 against a window of 1,024, so every window-attending layer was fully causal
throughout its fit. This comparison therefore reads long-context behaviour through a lens that never
saw it. It is a measurement worth making and it is not evidence about the model until a lens fitted
above the window agrees with it.

### Secondary reading, unconditioned

**Lens-against-next-token top-one agreement by span and depth**: where along an agent's prompt the
lens reads at all. This is descriptive and carries no comparison.

## What is not being done

**No hypothesis test.** Fifteen episodes at one seed on one checkpoint is a map, not an experiment.
Every number is descriptive, intervals where they are meaningful, and no p-value is computed or
quoted. The Qwen pilot's corresponding figure is shown beside this one on identical axes where the
layers coincide; that juxtaposition is a comparison of two maps and not a test of a difference.

**No claim from a single layer or a single episode.** A pattern is reportable when it holds across
at least two task families or across a run of adjacent layers. One cell is an observation.

**No claim about the trained agent.** This is the base model. Every adapter this programme owns is
Qwen's, so nothing here speaks to what training does to Gemma.

## What would make this a null

Stated in advance so it cannot be reinterpreted afterwards. If foreknowledge shares at every
non-final layer sit within the final layer's base rate, the lens reads nothing this instrument can
distinguish from the readout itself, and the map is a null. That is a real possible outcome, it is
reportable, and on this checkpoint it would most likely indict the lens's 128-token fit rather than
the model.

## Recorded before the fact

The globally-attending layers, the span facets, the horizons, the primary comparison, the window
condition and the null are all fixed by this document. The layer set for the map is all
thirty-four. The episode set is the twelve family-stratified task ids and three chat episodes named
in `GEMMA3-WORK-ORDERS-2026-09-08.md`.
