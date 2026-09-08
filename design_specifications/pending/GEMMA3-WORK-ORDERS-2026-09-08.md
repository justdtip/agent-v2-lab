# Gemma 3 pivot: work orders

**Issued by the Chief on the Director's approval, 2026-09-08.** The plan is
`GEMMA3-PIVOT-2026-09-08.md` with its section 11 corrections; this document says who does what,
in what order, and what each person must not do. Read section 11 before section 6, because the
ordering changed: **the first reading no longer waits on the architecture port.**

The Director's approved shape: **do the port and the first reading on 4B, then repeat on 12B.**
Every port defect is identical on both, 4B's weights and lens are already downloaded and
verified, and the evaluation comparison against the Qwen record is like-for-like only at 4B. The
12B repeat then answers whether a depth profile is a property of Gemma 3 or of one size, and it
is the size whose final block is global-attention.

---

## Rulings made here, so nobody waits on them

**The pilot runs on 4-bit weights.** The Qwen pilot ran on a 4-bit checkpoint, so a two-panel
comparison is like-for-like only at 4-bit. The hosted lens was fitted on bfloat16 weights, and
that mismatch is disclosed in the record rather than avoided; it is the same class the Qwen work
already carries. Lenses we fit ourselves are fitted on bfloat16, where a finite-difference
derivative is not taken through a quantisation.

**Both precisions are converted here from the official weights**, which are on disk and
authenticated, rather than taken from a third party's conversion. The provenance then names a
source we can name and a conversion we performed.

**Provisional ruling on the tool role, for the pilot only: render observations as user turns.**
Gemma's template enforces user and model alternation, has no tool role, and raises on our shape.
The pilot is a base-model reading with no training, so the only thing at stake is prompt
rendering, and re-roling keeps us inside the model's own distribution. **This does not pre-empt
the Director's ruling**, which governs anything that trains, because that is where the choice
changes the supervised target rather than the prompt. Record it in the pilot's manifest as
provisional and name it in the reading rules.

---

## The Deputy

Engineering in this repository. Order is strict; each item's reason is why it sits where it does.

1. **Land the registry entry** (`GEMMA3-REGISTRY-ENTRY-2026-09-08.md`) with the covering test in
   `test_models.py`, or it is the first registry file the suite ignores. Unruled fields stay
   marked. Both covering tests are free of the model library, so this lands in any window.

2. **The pilot path.** Four small items, none of which needs the box.
   - **The cache guard, narrowly.** Accept a rotating cache when head capture is off; keep
     refusing it when head capture is requested, naming the column arithmetic as the reason.
     Residual capture emits against the monotone offset and is safe; head capture maps a weight
     column to an absolute source position by identity and is not. Do not widen the type gate.
   - **The generation prefix into the chat specification, with the assertion kept.** The same
     value is stamped into training manifests and into the lens-corpus tokenizer identity, which
     the prose stage re-derives and hash-compares, so deleting the guard would produce a
     self-consistently falsified corpus identity on the J-space artefact.
   - **Issue 99, the lens identity.** Both lenses are now on disk and the Qwen one loads onto a
     Gemma view without complaint; the demonstration in the issue is the acceptance test.
   - **The pilot script's pinned constants become arguments.** The lens path, its digest and the
     registry are module constants today, so `--model` moves nothing.

3. **The architecture-view port**, as designed in `ARCH-VIEW-PORT-2026-09-08.md`, with the hot-path
   ruling already given: pay the observation forward, never cache it, never drive it from a
   mid-network residual. One change, with the gate above the sliding window and the negative
   control. This gates lens *fitting*, the Jacobian estimator, and every claim resting on the
   sliding-against-global contrast. It no longer gates the first reading.

4. **The single-token and stale-reference sweep**, extended as proposed to every
   `convert_tokens_to_ids`, every place a project string is assumed to be one token in the
   model's vocabulary, every token id compared without a round-trip, and the two unaudited
   subsystems on the J-space path, the capture probe and the state probe.

5. **Rotating-cache bookkeeping for head capture**, then issue 98 in its old place.

**Do not**: run any further Qwen training or evaluation; wire the stop-token field; widen a cache
type gate; or ship any part of item 3 separately.

---

## Codex, through the Director

**The Qwen Jacobian line is retired, and that should be said plainly rather than left to be
inferred.** The step-size ruling, the plateau sweep that found the corrected constant, the
original-scale diagnostic and the six-comparison self-check were correct work that answered a
question we have stopped asking. They were hybrid-specific: they existed because gated-delta
state made the finite-difference reference ambiguous, and on an all-attention model the cached
and uncached responses compute the same function by construction. What carries is the method,
not the artefacts: registration before measurement, the instrument proving itself before it is
read, and every mismatch disclosed in the record. That method is why this pivot has a plan
instead of a guess.

The new work, in order:

1. **Validate our fitting path against the hosted lens.** This is something we could never do on
   Qwen and it is now free. A pre-fitted Jacobian lens for `google/gemma-3-4b-it` is downloaded,
   converted and verified at `research/records/GEMMA3-LENS-2026-09-08/`. Fit our own regression
   lens at every layer on Gemma and compare the two: agreement is evidence our path is sound,
   and disagreement localises to a layer. An independent ground truth is worth more than either
   lens alone.

2. **The refit above the sliding window, on RunPod.** The hosted lens was fitted at a sequence
   length of 128 against a window of 1,024, so every windowed layer was fully causal throughout
   its fit. It therefore cannot support any claim about the sliding-against-global contrast,
   which is the new thing the pivot buys. Refit with the same reference implementation at a
   length beyond the window, on our own corpus. Estimated at a few dollars of the Director's
   credits and the best single use of them.

3. **The prose and corpus work carries**, with the corpus rebuilt on Gemma. Two Gemma-specific
   cautions: the template applies a trim to message content, and the corpus locates spans by
   verbatim substring, so audit the prompts for leading and trailing whitespace first; and set
   the corpus length beyond 1,024 for anything resting on the layer contrast.

**Do not**: fit or measure anything further on Qwen3.5; carry the hybrid step rule into a Gemma
fit without re-deriving it; or treat the hosted lens as neutral with respect to the window.

---

## The Chief

1. One window in the next gap for the suite that lands the window verb and the band-pairs
   function, with the two lens test files inside it.
2. One short box job to convert the official weights at both precisions, recorded.
3. Review of every item above, and the pre-registration of the pilot's reading rules **before any
   record is read**, as the house rule requires.
4. The 12B repeat, scoped once 4B has produced a reading: the artifact writer's one-gigabyte cap
   refuses a 12B lens at 1.39 GB, and the memory envelope is calibrated at 4B.

---

## What the Director still owes

**The tool role, for anything that trains.** The provisional ruling above unblocks the pilot and
deliberately does not settle this. Re-roling observations as user turns preserves alternation and
keeps us inside the model's distribution; a custom template keeps our conversation shape and
leaves the model's. The choice changes what a supervised target looks like, which is why it is
his.

**Whether Qwen3.5 stays a live comparator.** It decides whether every recorded artefact must stay
backward-readable and what becomes of the adapter-depth result.

**Whether a trained Gemma agent is on the near path.** Every adapter we own is Qwen, and the
programme's question is about a trained agent. If it is near, several deferred items become
blockers.

---

## Amendment, on the Director's ruling of 2026-09-08: map the representations first

**"I would like to map out internal representations first across a variety of tasks."** That is a
ruling on the third open question and it settles two things.

**No trained Gemma agent is on the near path.** Every deferred training item stays deferred: the
floor recalibration, the launch envelope, the LoRA rank, the training preflight. And the tool-role
question loses its training half for now, so **the provisional ruling becomes the operative one**:
observations render as user turns. It reverts to provisional the day a supervised target is built.

**The pilot's design changes, because "a variety of tasks" is a specification and the Qwen pilot's
episode set was a convenience sample.** It covered nine of twelve families, chosen for length
rather than for coverage.

### The map has two axes and the first design under-resolved both

**Depth: all 34 layers, not six.** The layer set is fixed at capture time — the records persist
readouts and ranks, not residuals, so a different depth means another run of the model. Six layers
were right for a comparison against the Qwen pilot and are wrong for a map, because the question a
map answers is *where in depth* something happens. The lens carries maps for layers 1 to 33 and
the final layer is the identity, so all 34 are available.

**Variety: one episode per family, all twelve.** Chosen from the test split at difficulty 2, with
task horizons from 2 to 14 steps, so the set spans short lookups and long multi-step
reconciliations rather than clustering:

```
("test-read-0108-clean", 2)                ("test-search-0061-clean", 2)
("test-calculate-0158-clean", 2)           ("test-list-0149-clean", 2)
("test-synthesis-0039-clean", 2)           ("test-update-0028-clean", 2)
("test-pointer_chain-0018-clean", 2)       ("test-cross_reference-0032-clean", 2)
("test-conditional_update-0093-clean", 2)  ("test-batch_update-0166-clean", 2)
("test-ledger_reconcile-0163-clean", 2)    ("test-aggregate_report-0167-clean", 2)
```

Keep the three chat episodes as the non-agentic contrast, including the long summary, because it
is the only episode that exceeds the 1,024 sliding window and therefore the only place the
sliding-against-global contrast is live at all.

### Run it in two stages, and the first stage is not the map

**Stage one, about fifty minutes: the same fifteen episodes at eight layers.** Not a result. It
proves the whole path end to end — the registry entry, the cache guard, the generation prefix, the
lens identity check, the rendering of observations as user turns, the record chain — on the real
task set, at a cost we can afford to throw away. Read it only for whether the instrument worked.

**Stage two, about three and a half hours: the same fifteen episodes at all 34 layers.** This is
the map. Estimated from the Qwen pilot's measured rate of roughly twenty-four seconds per episode
and layer; Gemma's vocabulary is six per cent larger and its hidden size identical, so the rate
should carry. Peak memory about five gigabytes: the model at two and a half, the lens at
eight-hundred-odd megabytes once expanded to float32, and the rank buffer at roughly six hundred.
Comfortably under the declaration threshold, but announce the window for its duration.

### The deliverable

A heatmap of layer against task family, one panel per horizon, showing the share of emitted tokens
whose competition rank is within ten in the lens distribution read that many positions earlier,
with the final layer's value as the base rate. Thirty-four rows, twelve columns, three panels. The
span facet stays as the second figure, and the Qwen pilot's six-layer profile remains comparable by
restricting to its layers.

**Pre-register the reading rules before any record is read**, as the house rule requires, and
record which layers are global-attention (6, 12, 18, 24 and 30) so that contrast is stated in
advance rather than found afterwards.
