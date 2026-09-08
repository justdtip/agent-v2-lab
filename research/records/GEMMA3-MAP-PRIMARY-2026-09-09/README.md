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
