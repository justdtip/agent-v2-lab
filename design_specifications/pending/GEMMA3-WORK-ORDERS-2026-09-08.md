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

---

## Gemma Scope 2 and the lens: compose them, do not merely label with them

The Director's observation that Gemma Scope gives pre-understood features to map to is right, and
there is a sharper composition available than labelling. Verified against the repository, not
researched.

**`google/gemma-scope-2-4b-it` is our exact model.** Its config names
`model_name: google/gemma-3-4b-it`. Three sites at every one of the 34 layers, in the `*_all`
directories: the residual stream after each block, the attention output, and the MLP output.
JumpReLU dictionaries at widths from 16,384 to a million, at three sparsity levels. One
16k residual dictionary is 336 MB; the whole repository is 4.65 TB, so downloads are per layer and
per site, never wholesale.

**The layer conventions align, which is the first thing that had to be true.** The residual
dictionary's hook is `model.layers.N.output`, the output of block N, which under this project's
writer convention is probe layer N+1 — the same residual the lens reads through its `J{N}` key.
So a dictionary and a lens map at the same index describe the same vector space.

### The composition

A dictionary's decoder rows are feature directions in the residual space at a layer. The lens is a
map from that space into the final space, read through the model's own unembedding. **Apply the
lens to the feature directions.** Every feature then acquires a token-level readout: what this
feature, seen from this layer, pushes the output toward. Set that beside the feature's published
auto-interpretation label and each dictionary entry carries both a human description and a
verbalisation, derived independently.

During a task the reading changes in kind. Instead of *the residual at layer 18 predicts this
token*, it becomes *layer 18 is running these labelled features, and these two are what push
toward that token.* That is the map the Director asked for, with the axis of interpretation the
lens alone cannot supply.

**It is exact where it matters.** The lens applied to a feature direction is not an approximation;
it is the lens's own linear action on that vector. The only approximation is the dictionary's
decomposition of the residual into features, and that one is measurable as reconstruction error
rather than assumed.

**It needs no model.** It is a product of two matrices we hold, so it runs off-box, in parallel
with the mapping run, and competes for nothing.

### Caveats to record, not to discover later

- The dictionaries were trained on Google's unquantised weights and we run a 4-bit conversion.
  The reconstruction error under quantisation is unmeasured, and it matters more for a dictionary
  than for a lens, because sparsity is a threshold on activations and a threshold is exactly what
  quantisation noise moves. Measure it at one layer before any claim rests on a feature being
  inactive.
- Our lens was fitted at 128 tokens, so the feature readouts inherit that fit's accuracy.
- The attention-output and MLP-output dictionaries answer which sublayer *wrote* a feature. That
  is a circuit question, and a lens is structurally incapable of it. Worth having, after the map.

### Order

The map first, because it says which layers are worth 336 MB and a matmul. Then the composition at
those layers. Owner: Codex, alongside the lens validation, since it is the same matrices and the
same skill, and it blocks on nothing.

---

## What a finding must record, on the Director's statement of the deliverable

**"The value here is not necessarily the code. It is the knowledge encoded by the findings, and
the techniques used to elicit them. These transfer to any model with some tweaks, which is what I
want to test."**

That is a specification, and three things follow.

**Every record separates three layers.** What the model does, which is the finding. How it was
elicited, which is the technique, stated so that someone on a different stack could reproduce it
without our code. And what our implementation happens to be, which is the least durable of the
three. Our records currently interleave all three, and the technique is usually the part left
implicit. Write it explicitly, in its own section, in terms of the model rather than of our
modules.

**The port is the test, not the cost.** The audit's central finding is which of our instruments
described one architecture and which asked any model. The Deputy's design for the port is that
principle in code: observe the model rather than describe it. **So the list of what broke in the
move from a hybrid to a dense model is itself a deliverable**, and it should be written up as one
rather than left in a work order. It is the most direct evidence we will get about which
interpretability techniques are architecture-bound.

**Retired work is reframed, not mourned.** The Qwen Jacobian step rule, the plateau sweep and the
six-comparison self-check are gone. The finding underneath them is not: *recurrent state makes a
finite-difference reference ambiguous, because the state is a lossy summary whose value depends on
the schedule that produced it, so an all-attention model admits Jacobian methods that a hybrid does
not.* That is a transferable statement about which architectures admit which lens methods, it cost
two days to learn, and it belongs in the record as a finding rather than in a changelog as a
deletion.

**A consequence for the pre-registration.** The pilot's reading rules must be stated in
model-agnostic terms — depth as a fraction, layer kind by its role rather than its library flag,
spans by their function in the protocol — so the same rules apply unchanged to 12B and to the next
family. A rule that names a layer index is a rule that does not transfer.

---

## Two corrections from Codex, and the first is to the Chief's own transferable claim

Codex's preparation record (`GEMMA3-FITTING-2026-09-08/FINDINGS-AND-TECHNIQUE.md`, branch
`codex/lens-fitting`, 5eab827) corrects two statements above. Both are accepted, and the first
matters most because it is the very statement this document held up as the model of a transferable
finding.

### 1. What the hybrid actually showed, stated correctly

**The Chief wrote:** *recurrent state makes a finite-difference reference ambiguous, so an
all-attention model admits Jacobian methods that a hybrid does not.*

**That is too strong, and Codex is right to refuse it.** What the Qwen work observed was that *our
cached shortcut* disagreed with *our uncached reference*, and that the disagreement could not be
separated from step-size and numerical-scale effects. Two distinct causes were being run together:
the dependence of a perturbation's effect on the *intervention history* that produced the cached
state, and the numerics of the step rule. Neither says a Jacobian is unavailable on a recurrent
model. An uncached full-sequence Jacobian is perfectly well defined there; it is only more
expensive.

**The transferable statement, corrected:** *where a model's per-position state is a lossless record
of its prefix, the cached and the uncached finite-difference references coincide by construction,
so the cheap cached path needs no validation. Where the state is a lossy fixed-size summary, they
do not, so the cached path must be validated against the uncached one or abandoned.* That is a
claim about when a cost-saving shortcut is safe, which is the useful and portable form. The
earlier version was a claim about which architectures admit an entire method, which is not what
was measured.

The correction is worth more than the original claim, and it is a good illustration of the rule
this document sets: the technique section is where over-generalisation shows up, because a finding
stated at the wrong scope reads as a bigger result and transfers as a wrong one.

### 2. The lens validation was ill-posed as ordered

**The Chief ordered:** fit our own regression lens at every layer and compare it with the hosted
Jacobian lens; *agreement is evidence our path is sound, and disagreement localises to a layer.*

**That is not a validation, because the two estimate different things.** A regression lens is
fitted to minimise prediction error on a corpus. A Jacobian lens is the local linearisation of the
tail. They can disagree while both are correct, so disagreement cannot diagnose a fitting defect
and agreement is weaker evidence than it looks.

**Corrected order.** The validation is like against like: **our Jacobian fit against the hosted
Jacobian lens**, same estimand, same corpus family, differences attributable to fitting length and
precision rather than to method. The regression-against-Jacobian comparison stays, but as a
**measurement about the model** — where a fitted linear predictor and a local linearisation of the
tail coincide, and where they part — which is a finding worth having and is not a check on our
implementation.

### Also from that record, and worth naming as the transfer test working

The corpus sampling rule was re-run under Gemma's tokenizer and yields **50 held windows of 1,024
tokens against Qwen's 51**. The rule transferred unchanged; only the count moved. That is exactly
the shape this programme is testing for, and it is the first instance of it measured rather than
asserted.

Codex's sequencing is correct and stands: no Gemma fit until the architecture port has produced
conformance evidence and the official bf16 conversion exists. The conversion is the Chief's, and it
needs the machine.
