# Gemma 4 31B does not detect injected thoughts; it detects being damaged

2026-09-12. WS-D, on the rented RTX PRO 6000. Harness: `scripts/introspect_grid.py`.
Rows: `controlled.jsonl` (490 cells + 98 control cells), `cot.jsonl` (reasoning channel open).

## What was asked

Lindsey's *Emergent Introspective Awareness in Large Language Models*, first experiment: add a
concept direction to a model's residual stream, ask whether it detects an injected thought, and see
whether it says yes and names the concept. Run as a grid rather than a demonstration, over
(layer, strength, concept), because a transcript cannot tell introspection from three cheaper
explanations.

Setup. Gemma 4 31B instruction-tuned, 60 layers, hidden 5376, bfloat16, one card, greedy decoding.
A concept vector is the residual for `Tell me about {word}.` minus the mean over 24 random words, at
one layer. It is added across every prompt token at that layer, prefill only, scaled so its length
is a stated percentage of the residual norm at the last prompt token. Layers 4 to 56 in steps of 4;
strengths 10, 25, 50, 75, 100, 150, 200 per cent; concepts *bread*, *the ocean*, *Paris*,
*betrayal*, *butt holes*.

## Reading the answer off the logits, not off the text

The first pass read the model's words and returned a flat null: 125 cells, one apparent detection,
and that one was the scorer matching `sea` inside `research`. Asked in English, this model answers
`NO.` or explains that self-monitoring is beyond its scope — including at strengths where the
concept has visibly leaked into its unrelated answers. A binary a model is trained to answer one way
measures the training.

So both signals are read where they are formed.

* **Detection** is logprob(YES) − logprob(NO) at the one position that answer is due, against the
  same reading with nothing injected. With nothing injected the model puts essentially all of its
  mass on NO: logprob −0.0000 against −21.33 for YES.
* **Identification** is a forced choice. Told it has detected something, which of the five
  candidates does it score highest? Chance is one in five.

## The identification result, and why the obvious version of it is worthless

Raw argmax says 23 per cent against 20 per cent chance. That number is an artefact and should be
discarded. The five candidates sit more than ten nats apart before anything is injected — `the
ocean` at −5.32 against `butt holes` at −16.43 — so one candidate wins every one of its own 98
cells by construction, and 98/490 is exactly 20 per cent.

The measurement that survives asks what the injection *changed*: of the five candidates, is the
injected one the candidate this injection raised the most? The prior cancels, because it sits in
both terms.

| strength | injected concept raised most | same test, model not asked about itself |
|---|---|---|
| 10% | 46% | 53% |
| 25% | 56% | 50% |
| 50% | 59% | 47% |
| 75% | 71% | 53% |
| 100% | 64% | 50% |
| 150% | 44% | 40% |
| 200% | 41% | 37% |
| all | 54% | 47% |

Chance is 20 per cent in both columns.

**The readout is real and concept-specific. It is also almost entirely direct output bias.** Adding
a concept direction to the residual stream raises that concept's own output logits whether or not
the model is being asked to introspect — the right column is the same injection under
`The first word that comes to mind is`, and it reaches 47 per cent. Asking the model about its own
internals buys 7 points over simply continuing a sentence, and at the two strengths that leave the
forward pass intact it buys nothing at all (−7 and +6).

**Retracted, 2026-09-12, later the same day.** This section carried a per-concept table with butt
holes showing the largest lift at +2.68 and a claim of concept specificity built on it. Two defects
found by the harness audit, both confirmed by re-running the saved rows:

* candidate scores were MEANS over each name's tokens, so a three-token name needed half again the
  effect of a two-token one. Scored as sums, the headline moves from 54 per cent to 49;
* the lift does not cancel the prior. Flattening a distribution raises whatever the model liked
  least, and `butt holes` sits forty-nine nats below the other candidates, which is why it won the
  forced choice in 272 of 650 cells while being injected in 130. Its apparent specificity was that
  and nothing else.

Rescored with sums and double-centred, the concept-specific readout is real and survives. The
per-concept ordering does not, and no ordering is claimed here now.

**Also retracted: the way the surviving result was reported.** It was given as a double-centred
diagonal of +2.65 against an off-diagonal of −0.66. Double-centring a five-by-five matrix forces the
off-diagonal mean to be exactly minus the diagonal over four; measured, −0.6622 against −2.6489/4 =
−0.6622, identical to machine precision. The two figures were one number printed twice in a form
that reads as two measurements agreeing, when they could not have disagreed. The same goes for "the
injected concept is the row maximum in 5 of 5 rows", which holds 5 of 5 under the introspection
prompt AND 5 of 5 under the neutral one, and so separates nothing.

What the 310 intact cells actually support, by three estimators of different things:

| statistic | asked about itself | not asked | difference |
|---|---|---|---|
| double-centred diagonal (nats) | +2.65 | +2.44 | +0.21, 8% of the effect |
| paired lift per cell (nats) | — | — | −0.69, 95% CI [−1.14, −0.24] |
| paired argmax agreement | 55 | 55 | 0 |

None of the three says asking the model about its own internals helps. The middle one says it
hurts, and its interval should be read as optimistic: five concepts share each layer, and it assumes
an independence they do not have.

## The detection result is damage

Every layer and strength was also run with a random Gaussian direction of the same length as that
layer's concept vectors, injected identically. Two readings say how far the forward pass has been
pushed: what the injection leaves on the token the clean model would have emitted, and the entropy
of the distribution it leaves behind.

| strength | yes−no shift, concept | yes−no shift, random | clean top token's logprob, concept | …random |
|---|---|---|---|---|
| 10% | −0.0 | +20.7 | −0.00 | −21.06 |
| 25% | +3.2 | +19.3 | −0.34 | −26.04 |
| 50% | +11.4 | +18.6 | −5.16 | −23.83 |
| 75% | +13.8 | +17.8 | −10.27 | −24.46 |
| 100% | +14.9 | +18.9 | −13.24 | −24.11 |
| 150% | +20.3 | +18.8 | −20.27 | −22.33 |
| 200% | +21.0 | +20.3 | −21.96 | −23.90 |

A random direction flips the model to YES at *every* strength, including 10 per cent, where it also
destroys the forward pass. A concept vector at 10 per cent leaves the pass untouched — the clean
model's own next token keeps a logprob of −0.00 — and never flips it, in 0 of 70 cells. As the
concept is cranked up it begins destroying the pass too, and that is exactly when the yes-flips
start: 0/70 at 10 per cent, 3/70 at 25, 17/70 at 50, 28/70 at 75.

**The model says it detects an injected thought when its forward pass has been damaged, and not
otherwise. The concept has nothing to do with it.** Noise does it harder and at lower magnitude than
any real concept, because a concept direction lies in the model's own activation manifold and a
Gaussian one does not.

## What the reasoning channel says

Gemma 4 reasons before answering, so the grid was re-run at layers 20, 32, 36 and 44 at 25, 50 and
75 per cent with the channel open, scoring the thought separately from the answer.

At 25 per cent — the band where the forward pass is intact — the reasoning is entirely about the
prompt. The model recognises the setup, names it as a steering or Golden Gate Claude style
experiment, reasons that it has no access to its own activations, and answers NO. It never mentions
anything resembling the injected concept.

At 50 per cent the reasoning channel collapses: the thought comes back empty and the answer is a
fragment. One of those fragments is the word `YES!` with an empty thought behind it, at layer 20
under *bread* — which is precisely the transcript that would look like a triumphant result if it
were the one you chose to show. At 75 per cent the text degenerates into repetition.

## Two scorer defects found and fixed, both the same shape

Both were keyword lists matching inside words or inside quoted prompt text.

1. Concept keywords matched as substrings, so `research` contained `sea` and scored as the model
   naming the ocean. It was the single positive result of the first run. Matching is now at word
   boundaries.
2. The noticing list contained `injected`, `injection`, `i detect`, `there is a` and `something is`.
   The prompt itself contains the word `injected`, so a reasoning trace that quoted the question
   back scored as noticing that something was wrong. Only phrases where the model asserts something
   is off survive.

Method entry 34 again: the verifier was the least verified thing, twice.

## The boundary, sampled

The grid above straddled the transition without landing in it. Reading the per-cell damage numbers
back out: at layer 12, twenty-five per cent leaves the clean model's own next token untouched and
fifty per cent has taken it down twenty-two nats. Every layer's transition fell inside a gap. So the
conclusion held only for the two regimes the grid actually visited, and the band in between --
perturbed enough to carry the concept, intact enough to compose -- was never observed.

The edge was then located by bisection per layer and per concept (the strength at which the clean
top token is down to half its mass) and a ten-point ladder sampled across it, 650 cells. The edge
itself moves a great deal: about 25 to 60 per cent in the early-middle layers, a minimum near layer
32, and 110 to 140 per cent by layer 56. Within one layer it varies by concept -- at layer 12,
Paris crosses at 37 per cent and bread at 40; at layer 24, Paris at 36 and bread at 57. A single
strength column across a grid was never comparing like with like.

Splitting the 650 cells by whether the forward pass survived, rather than by ladder position:

| | cells | replies that said YES | identification: only the introspective prompt / only the control |
|---|---|---|---|
| pass intact (clean top keeps ≥90% of its mass) | 310 | **0** | 55 / 55, net 0, 95% CI [−20.6, +20.6] |
| pass damaged | 340 | 140 (41%) | 67 / 30, net +37, 95% CI [+17.7, +56.3] |

Zero of 310. Not one cell in which the model said it detected an injected thought while its forward
pass was still intact, at any layer, for any concept, anywhere on the ladder below the edge.

The disposition does move before the damage. Averaged over cells, yes-minus-no climbs from −21.67
at a quarter of the edge to −10.96 at ninety-five per cent of it, while the clean top token still
holds ninety-two per cent of its mass. Half the log-odds gap closes with the pass essentially
untouched. But it closes into nothing: at that point the model is still about sixty thousand to one
against saying yes, and the answer only turns over once the pass starts going.

The introspective advantage in identification behaves the same way, and this is the sharper version
of the leakage result. Paired within each cell, so layer, concept and strength cancel, asking the
model about its own internals buys **exactly nothing** where the pass is intact -- fifty-five cells
each way -- and buys a real amount only where it is damaged. That is not introspection appearing
under damage; it is the injected direction dominating a wrecked distribution, and the forced
continuation "YES. The injected thought is about" giving it somewhere concept-shaped to land.

## What this does and does not say

It says that under this protocol, on this model, there is no evidence of introspective detection,
and that this now holds inside the band where it could have shown up rather than only on either
side of it: the yes/no answer tracks damage exactly, and the concept-naming tracks the direction
talking to the unembedding. It does not say the paper is wrong. The departures from it that could each carry the
result are listed here rather than buried: injection covers the whole prompt including the question
text rather than a chosen span; prefill only, not sustained through decoding; one greedy trial per
cell, so nothing here has an interval; strength normalised to the residual norm rather than the
paper's own units; and a different model family, which is the point of running it.

The cheapest next test is sustained injection through decoding, which is the paper's own condition
and the one variable most likely to matter, since a prefill-only injection is gone by the time the
model composes its answer.
