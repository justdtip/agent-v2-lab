# The primary comparison reverses with token history

**First result from the completed representation map. Chief, 2026-09-09.** Fifteen episodes, all 34
layers, 534,990 rank rows, computed under the pre-registration fixed before the run existed.

## The join, validated before any number was read

| assertion | result |
|---|---|
| `emitted x layers x horizons == rank rows`, every episode | **passed** |
| `token_id` agreement on every joined row | **0 mismatches of 489,702** |
| unmatched rows | **0** |

Both were required by the amendment because the span facet is derived rather than recorded, and
because a join across two position conventions is what produced this programme's worst error.

## The headline number, which is a mixture of two opposite effects

Foreknowledge at horizon one — the share of emitted tokens whose rank, read one position earlier, is
within ten — deduplicated for byte-identical repeated calls, computed per episode and averaged with
equal weight across all twelve agentic episodes:

| layer | call argument | call skeleton | difference |
|---:|---:|---:|---:|
| 18 | 0.001 | 0.012 | −0.011 |
| **23** | 0.123 | 0.275 | **−0.151** |
| **24** | 0.632 | 0.689 | −0.056 |
| 29 | 0.936 | 0.989 | −0.053 |
| 33 | 1.000 | 1.000 | 0.000 |

Read alone this says **the convention is available earlier in depth than the content**, with the gap
widest at layer 23 and closed by 33. It also puts the corpus-wide transition at **layer 23 to 24**,
where foreknowledge jumps from 0.12 to 0.63 on arguments — the same layer step at which the D-CRO's
fixed point commits.

**That single number should not be reported on its own, and the pre-registration said so before it
was computed.**

## Stratified by token history, it reverses

Prior occurrences of the token in its own turn's prompt, at layer 24:

| prior occurrences | argument | skeleton | difference |
|---|---:|---:|---:|
| 0 | 0.741 | 0.846 | −0.105 |
| 1–2 | 0.761 | 0.858 | −0.097 |
| 3–9 | 0.675 | 0.821 | −0.145 |
| **10+** | **0.629** | **0.360** | **+0.269** |

**The sign flips in the top bucket.** For tokens the model has already seen ten or more times,
arguments are read *earlier* in depth than skeletons — the opposite of the headline.

**It is not a composition artefact, which was the first thing checked.** One episode supplies 270 of
784 argument tokens in that cell, so the pooled figure could have been one episode's giant arithmetic
span. Per episode, among the ten with both spans present, **nine show arguments ahead of skeletons**:

| episode | argument | skeleton |
|---|---:|---:|
| `aggregate_report-0167` | 0.744 | 0.250 |
| `cross_reference-0032` | 0.714 | 0.515 |
| `pointer_chain-0018` | 0.641 | 0.417 |
| `conditional_update-0093` | 0.562 | 0.188 |
| `ledger_reconcile-0163` | 0.526 | 0.391 |
| `search-0061` | 0.400 | 0.273 |
| `read-0108` | 0.400 | 0.143 |
| `list-0149` | 0.263 | 0.000 |
| `synthesis-0039` | 0.667 | 0.000 |
| `batch_update-0166` | 0.310 | **0.385** |

## What it means, stated no more strongly than it supports

**The headline is a mixture.** Skeleton-ahead in the three low-history buckets and argument-ahead in
the high-history one, averaged into a single number that reports only the majority. The unstratified
comparison the pre-registration originally specified would have been read as a fact about convention
versus content, and it is substantially a fact about how often the model has already seen the token.

**The plausible reading**, and it is a reading: an argument token seen ten or more times is being
*copied* from context, and copying resolves early. A novel argument — a value the model must produce
rather than repeat — resolves late, later than the convention that frames it. On that account the
depth at which a call token becomes predictable tracks whether the model is **retrieving or
composing**, not whether the token is structure or content.

**I named a test for it, ran it, and my prediction failed.** The contrast: argument tokens at layer
24 and equal prior occurrence, split by whether the surrounding call errored. I predicted that once
prior occurrence is held fixed, call outcome would add nothing.

| prior occurrences | succeeded | n | errored | n | difference |
|---|---:|---:|---:|---:|---:|
| 0 | 0.955 | 22 | 0.519 | 27 | **+0.436** |
| 1–2 | — | 111 | — | 18 | cell too small |
| 3–9 | 0.676 | 207 | 0.778 | 36 | −0.101 |
| 10+ | 0.587 | 407 | 0.665 | 182 | −0.078 |

**Call outcome does add something, and the sign is inconsistent.** At zero prior occurrence
successful calls resolve far earlier; at three or more, errored calls resolve slightly earlier.

**The test does not settle the reading in either direction, for two reasons I would rather state than
argue past.** The large effect sits on the two smallest cells — 22 and 27 observations — which is the
day's most familiar shape: the biggest number on the least data. And the small, consistent effect in
the two well-populated buckets runs the wrong way for any account I can defend. A plausible
explanation is residual repetition: near-identical repeated calls survive a deduplication keyed on
byte-identical ones, and a repeated wrong argument is trivially predictable. That is a confound, not
a finding.

**So the retrieving-versus-composing reading stands as a reading and is untested.** What it needs is
a contrast that separates repetition from composition directly, which this one does not.

## Limits carried from the pre-registration

Every read here sits **outside the lens's fitted position range**, which is positions 16 to 126; the
map reads to 2,749. The one available boundary check finds no degradation at roughly 2x and nothing
tests the 18x these episodes reach. The layer-34 row is the model's own distribution and is exact;
every other row is a lens read. **Four passes in twelve agentic episodes**, so this is a map of
attempts, and the equal-weight rule keeps the 24-turn looping episode at one twelfth rather than the
20.0% of rank rows it supplies.

---

## The two statistics disagree at layer 24, and both are right

The D-CRO computed the same comparison as a median-rank ratio; I computed it as a foreknowledge
share. They agree everywhere except the layer that matters, so here they are on the same
deduplicated rows:

| layer | rank = 1, argument | rank = 1, skeleton | rank ≤ 10, argument | rank ≤ 10, skeleton | median argument | median skeleton |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.000 | 0.000 | 0.001 | 0.037 | 10,275 | 1,194 |
| 21 | 0.009 | 0.018 | 0.051 | 0.107 | 4,659 | 336 |
| 23 | 0.024 | 0.090 | 0.101 | 0.269 | 797 | 86 |
| **24** | **0.462** | **0.318** | **0.658** | **0.687** | 2 | 2 |
| 25 | 0.597 | 0.465 | 0.815 | 0.814 | 1 | 2 |
| 29 | 0.774 | 0.790 | 0.940 | 0.984 | 1 | 1 |

**At layer 24 arguments are more often exactly right and less often within ten.** That is not a
contradiction and neither statistic is a summary of the other: **the argument distribution at 24 is
bimodal** — decided, or still far away — while the skeleton distribution is concentrated in the
middle. Before 24 the skeleton leads on every statistic by one to two orders of magnitude.

**Report both.** This is the same median-against-top-one disagreement that made an earlier boundary
check unresolved rather than null, and it arrived independently in two seats' analyses of the same
rows. A single headline statistic would have hidden the structure that is the finding.

## The pre-registered extrapolation test cannot be run on this corpus

Amendment 9 fixed a test of whether reading outside the lens's fitted position range costs accuracy,
with the episode as the replication unit and a declared conflict, because I had predicted the
direction. **It is uncomputable here, and structurally so.**

| | episodes |
|---|---:|
| with any reading at positions 16–126 | **2 of 15** |
| with both arms at n ≥ 30 | **2 of 15** — both chat |
| agentic episodes with any in-range reading | **0 of 12** |

Agentic prompts run to 500 tokens and beyond, and generation begins after the prompt, so **no agentic
position is ever in range**. Every agentic reading in the map — all 325,000 of them — is an
extrapolation, and the only in-range comparisons available are two short chat episodes, which is the
n the pilot already had.

**So the question of what the 3x-to-18x extrapolation costs is not answerable from this corpus at
any scale.** It is not a matter of waiting for more episodes; more episodes of this kind add only
out-of-range rows. **The only route is a lens fitted at agent-transcript context lengths**, which is
Codex's task 1 and which replaces the extrapolation rather than measuring it. That moves task 1 from
the critical path to the only path for this particular question.

---

## The window-conditioned secondary comparison: null, and uninformatively so

This comparison was declared void in the pre-registration on a hardcoded assumption, shown this
morning to be live, and has never been computed. It now can be: **7 of 12 agentic episodes qualify,
supplying 99,042 rank rows beyond the 1,024-token sliding window at horizon 4.**

Globally-attending residual layers against their immediate window-attending neighbours, equal weight
per episode:

| global layer | global | neighbours | difference |
|---:|---:|---:|---:|
| 6 | 0.035 | 0.051 | −0.016 |
| 12 | 0.033 | 0.028 | +0.005 |
| 18 | 0.016 | 0.016 | −0.001 |
| 24 | 0.033 | 0.039 | −0.006 |
| 30 | 0.120 | 0.113 | +0.007 |

**No difference at any depth, inconsistent in sign, on seven episodes and ninety-nine thousand rows.**

**And the pre-registration said in advance that this null would be uninformative**, in terms worth
quoting because they were written before any of it existed:

> *the lens itself was fitted at a sequence length of 128 against a window of 1,024, so every
> window-attending layer was fully causal throughout its fit. This comparison therefore reads
> long-context behaviour through a lens that never saw it. It is a measurement worth making and it is
> not evidence about the model until a lens fitted above the window agrees with it.*

A lens for which the two layer kinds were identical during fitting is a lens that cannot distinguish
them at readout. **The null is exactly what an instrument blind to the distinction produces, so it
cannot separate "the model does not differ here" from "the lens cannot see it."** That was called
correctly in advance and it is the reason the comparison is reported and not interpreted.

**One thing worth noting beside it.** Foreknowledge beyond position 1,024 runs 0.016 to 0.120 at
horizon 4, against a base rate of 19.1% computed across all positions. Long-context positions are
much harder to read. That is consistent with extrapolation degrading the lens and equally consistent
with long contexts being genuinely harder, and nothing in this corpus separates them — the same
confound, for the third time.

## Three pre-registered questions, one answer

| question | outcome |
|---|---|
| what the 3x–18x extrapolation costs | **uncomputable**: no agentic reading is in range |
| global against window-attending layers | **null, uninterpretable**: the lens was fitted where the distinction does not exist |
| why long-context foreknowledge is low | **unseparable**: instrument or task, not distinguishable here |

All three want the same thing and it is not more episodes. **A lens fitted at agent-transcript context
lengths, above the sliding window, is the only instrument that makes any of these three answerable**,
and it is Codex's task 1. That is now three independent routes to the same conclusion, and none of
them was the argument for it this morning.

---

## The deduplication rule, stated because two seats computing it differently nearly published a finding

Two independent computations of the layer-24 distribution agreed on arguments throughout and
differed on skeletons by seven points at rank 1. The asymmetry was the diagnosis: a sampling
difference moves both rows, a **definitional** difference moves the row whose tokens repeat.

**The rule, as used everywhere in this record.** Within an episode, take each turn's emitted call
token sequence, and keep a turn only if that exact sequence has not appeared in an earlier turn.
Deduplication is **per turn, keyed on the whole call**. A turn whose call differs by one character is
kept in full.

**The rejected alternative** keyed on the decision — a token and its predecessor, collapsed across
turns. Skeleton tokens repeat identically across turns while arguments vary, so a token-level key
deduplicates skeletons much harder and discards independent observations. Two different turns
emitting the same punctuation after the same fence are two decisions in two contexts; only a turn
**re-issuing an identical call** is the repetition the rule exists to remove. Recomputed under the
per-turn rule, both seats' numbers agree exactly: 65 turns kept, 23 dropped.

**Third time today a definitional difference nearly became a finding**, after the pooled-against-
deduplicated commitment table and the flat-against-per-turn reading join. The common feature is that
none was visible in the number itself.

## The corrected characterisation

"The argument is decided or nowhere" overstated the lower half. Under the settled rule:

| | rank 1 | ranks 2–10 | beyond rank 1,000 |
|---|---:|---:|---:|
| call argument | **46.2%** | 19.7% | **11.2%** |
| call skeleton | 31.8% | **36.9%** | 9.1% |

**The bimodality is strong at the top of the distribution and slight at the bottom** — two points of
separation in the far tail, not four. What stands is that **arguments are more often decided and
skeletons more often almost-decided**: arguments lead on exactly right by 14.4 points, skeletons lead
on nearly right by 17.2.

**And the headline survives the rule change**, which is what matters most. Median-rank log2 ratios
under the settled rule are **+3.80 at layer 21** and **+3.22 at layer 23**, against +3.03 and +2.77
undeduplicated. Deduplication still strengthens the effect; only its size moved. That was the
load-bearing claim and it did not depend on which key was used.
