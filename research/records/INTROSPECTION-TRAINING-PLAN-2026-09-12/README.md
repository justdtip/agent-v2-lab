# Training a model to report injected concepts, and deciding whether the report means anything

2026-09-12, WS-D. Written after four independent strategies were each attacked, and after the
audit that found six silent defects in the measurement code we would have trained against.

## The decision, first

**Train the 4B, not the 12B.** The number is 116 GB against 67 GB free.

Daniel asked for the 12B on the grounds that we have a fitted lens for it. We do —
`lens12b-admitted.npz` — but we also have `lens4b-admitted.npz`, so the lens is not a reason to
prefer it. Everything else is a deficit. The 12B weights were deleted from the card this afternoon
and cost 25 GB to re-fetch, which fits. Its SAE dictionary has never been on this box and does not
fit: the 4B's `resid_post_all` suite is 61 GB over 38 layer directories at width 2560, and a 12B
suite scales by width (3840/2560) and depth (48/38) to roughly 116 GB. That does not fit after
deleting the 4B dictionary, and does not fit after deleting the 27B as well. It is a disk fact, not
a scheduling one.

The dictionary is not decoration here, and this is the part worth arguing. Every concept vector in
every arm of this work, ours and Rivera and Africa's alike, comes out of one generator: the
residual for `Tell me about X.` minus the mean over random baseline words. A detector trained on
vectors from that generator only ever has to learn to recognise *the generator*. Holding out
concepts does not close that hole, because a held-out concept is still built the same way. Holding
out the **generator** closes it, and the SAE decoder rows are the only source of held-out
generators on this machine: sixteen thousand directions per layer, with known semantics, that no
training distribution of ours ever contained.

What choosing the 4B gives up is the scale objection. Every number we have is from Gemma 4 31B, and
Rivera and Africa's result lands on models of 9B and up. Nobody has checked that the 4B base
behaves like the 31B base at all. That check is the first stage below and costs under an hour. If
the 4B's clean model is not pinned on NO the way the 31B's is — logprob −0.0000 against −21.33 for
YES — there is no baseline to improve on and the decision reopens.

One pre-flight gate before anything rests on the dictionary: gemma-scope is usually fitted on
*base* activations and the card holds the instruction-tuned checkpoint. Run the identity check. If
it refuses, the mechanistic arm is dropped and the 4B recommendation stands on cost and
weights-on-disk alone, which is still enough.

## Before any training at all

The report has two channels and our own data already gives them different answers. Whether the
model says yes tracks damage. Whether it can name the concept tracks the injected direction talking
to the unembedding. Each needs its own decisive test, and all of them run on the **base** model with
no LoRA and no training.

**Stage 0 costs no GPU time and could end the programme.** The per-cell introspective lift is
already in the saved rows. Read it back paired, per cell, on the 310 cells where the forward pass
survived. Done today: the mean is −0.69 nats with a 95 percent interval of [−1.14, −0.24]. The
introspection prompt does not make the concept more readable than a plain sentence continuation. It
makes it less so. Everything downstream is therefore about what a *trained* readout can learn, not
about a capacity the untrained model has.

**The naming channel has a closed form.** Injecting a vector at layer L changes the final residual
by the vector itself, carried additively, plus whatever the remaining blocks compute differently.
Only the second term could be content. The first is exactly calculable with no extra forward pass,
because the unembedding is tied to the embedding: predict each cell's diagonal lift from the carried
term alone and compare against what was measured. If the prediction accounts for the measurement,
the identification result is arithmetic on a vector we added, and no amount of training changes what
that number is a fact about. Two traps to avoid producing a plausible wrong answer: Gemma's RMSNorm
applies one-plus-weight rather than weight, and the final logit softcapping must be checked rather
than assumed absent.

**The yes/no channel needs a disturbance that is not an injection.** Every control anyone has run,
ours and theirs, varies *which vector is added*. A model that has learned "a vector of the kind this
protocol adds is present, and roughly how big" passes all of them. The test nobody has run reaches
the same damage without adding a vector at all: scale one attention head's output, or one MLP's, or
simply multiply the residual by one-plus-epsilon, bisecting each to the same surviving mass on the
clean model's own next token. Then compare the yes-minus-no shift. If a concept moves it no further
than pure magnitude does at matched damage, the answer for that channel is disturbance, established
on the base model, with no dictionary and no training. That is the most likely outcome and it should
be pre-committed to as a result rather than treated as a failure.

Two cheaper tests run alongside. Take the gradient direction at layer L that most efficiently raises
the concept name's logprob, and inject that instead: if the model names the concept under it, the
naming channel is bias readout and nothing else matters. Note what that can and cannot do — it can
kill the content claim, it can never establish it, and it must not be used as a projection, because
projecting the concept off it would remove content only if introspective readout were orthogonal to
steepest ascent, which is a coincidence of measure zero. And inject the difference of two concept
vectors: same recipe, same subspace, same norm statistics, corresponding to no concept. A confident
yes with the forced choice at chance is a construction detector. That one needs a strength sweep
rather than a single point, since normalising a difference of two near-orthogonal vectors puts each
component at about seventy per cent and our own edges are sharp enough that it would otherwise fall
below both and return nothing.

## The training, if the tests leave something to train

LoRA rank 32, alpha 64, attention and MLP projections, with instruction-following replay at about
half the data to stop the adapter forgetting how to talk. That much is Rivera and Africa's and there
is no reason to differ.

Where we differ is the strength axis. They sample alpha as a plain multiplier. We know from today
that the damage edge moves from 18 to 140 per cent of the residual norm across layers, and varies by
concept within a single layer, so a fixed multiplier trains on wildly different amounts of damage
under one label. Sample strength in units of each cell's own edge instead. This is the one place our
instrumentation buys something the prior work did not have.

The training data needs four kinds of negative, not one: clean prompts, matched-norm noise, matched
*damage* noise, and mismatched concepts. The third is ours and is strictly harder than theirs, since
noise is more destructive per unit of norm than a concept direction is — which means their
norm-matched control was easier than it looked.

The mechanical problem to solve is that the injection happens during the forward pass, so every
training example needs its hook installed inside the training loop, interacting with gradient
accumulation, the adapter, and gradient checkpointing. That is where a silent defect would live.

Guard explicitly against the adapter learning to answer yes always, learning magnitude rather than
direction, and learning the statistical signature of how our vectors were built rather than what
they mean.

## Cost

Sequenced so it can be stopped between stages. Stage 0 is free and already done. The base-model
tests are hours, not days, and need no training. The 4B fine-tune is small. The dictionary work is
the long pole and only matters if the earlier stages leave a content claim standing.

## What this would still not settle

Prefill-only injection is not holding a thought: the concept is present while the model reads the
question and gone while it writes the answer, so nothing here can distinguish a reportable
perturbation to how a prompt was encoded from a state the model is currently in. Sustained
injection is implemented and unused.

Strength is still not one quantity unless `--local-scale` is on, since a delta sized from the last
prompt token's norm is applied at every position, and residual norms vary several-fold across a
sequence.

And the forced choice is five names from one list. A model cannot be recorded as naming something
that is not a candidate, and the free-text route is scored by keyword lists whose vocabulary decides
the answer — which is the same fact as three of the defects found today.
