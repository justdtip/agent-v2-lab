# The Gemma 3 pivot: what it buys, what it costs, and the order of work

**Status:** the Director's order, 2026-09-08. Qwen3.5 is retired as the base family because its
architecture does not suit the instruments. The programme moves to Gemma 3, the existing
evaluation suite runs on it, and **understanding Gemma 3's J-space is the priority.**

Written by the Chief from a 33-agent audit of every model-specific assumption in the tree,
each finding adversarially checked, plus ground-truth research. Every claim below carries a
file and line, or says it could not be verified. The audit ran read-only beside the Deputy's
live evaluation; nothing was loaded.

---

## 1. Ground truth, read from the official config on this box

`google/gemma-3-4b-it` is authenticated and downloading. Its `config.json` is on disk and
settles what the audit could only research:

| | Gemma 3 4B | Qwen3.5 4B (for comparison) |
|---|---|---|
| decoder layers | 34 | 32 |
| hidden size | 2560 | 2560 |
| intermediate size | 10240 | 9216 |
| vocabulary | 262,144 | 248,320 |
| periodicity | `sliding_window_pattern` (absent, MLX defaults to 6) | `full_attention_interval: 4` |
| sliding window | 1024 | n/a |
| terminators | `eos_token_id: [1, 106]` | a single id |
| wrapper | `model_type: gemma3`, text nested under `text_config` | `qwen3_5`, same nesting |

Three consequences of the table.

**The hidden size is identical.** That is a coincidence with teeth; see §5.

**The vocabularies are close, not distant.** The audit reasoned from a 151k Qwen vocabulary
and concluded that token-denominated constants would be badly rescaled and that lens ranking
would cost 1.7x more. The real Qwen3.5-4B vocabulary is 248,320. The change is about six per
cent. Every estimate resting on the larger figure is wrong in the safe direction and should be
restated.

**`sliding_window_pattern` is absent from the config and MLX supplies 6 by default**
(`mlx_lm/models/gemma3_text.py` `ModelArgs`). So does it supply the head counts, which the
config also omits: `gemma3.py` injects 8 attention heads and 4 key-value heads. Those defaults
are correct for 4B and wrong for other sizes. A registry entry must not inherit them silently.

---

## 2. What the pivot buys

### 2.1 A pre-fitted Jacobian lens, for free, at every layer

`neuronpedia/jacobian-lens` hosts lenses for the entire Gemma family: Gemma 2, all five Gemma 3
sizes, and Gemma 4, pretrained **and instruction-tuned**. For our model:

- `gemma-3-4b-it/jlens/Salesforce-wikitext/gemma-3-4b-it_jacobian_lens.pt`, 432.5 MB
- 33 maps of 2560 x 2560, which is exactly `set(range(1, 34))` for a 34-layer model

`scripts/convert_jlens.py` already converts that format and names no architecture.
`LensMaps.load` accepts it with no code change: the layer bound `1 <= layer < num_layers`
is satisfied, and `write_lens`'s 1 GB cap passes at 865 MB.

The instruction-tuned lens matters. The audit assumed only a pretrained lens existed and
conceded a pretrained-against-instruction-tuned mismatch as an unavoidable caveat. It is
avoidable: use the `-it` lens against the `-it` model.

Codex has spent two days fitting and validating a Jacobian lens on Qwen. On Gemma the artefact
is a download.

### 2.2 The Jacobian instrument becomes trustworthy

This is the deeper gain and it is epistemic, not a saving of lines.

On the hybrid, the finite-difference reference was ambiguous. Gated-delta state is a lossy
fixed-size summary whose value depends on the schedule that produced it, and this project's own
measurement is that the bfloat16 forward is shape-dependent by 0.4 to 0.9 nats between
schedules (178b71b). Dividing a shape-dependent discrepancy by a small step amplifies it, which
is why the cached path had to be litigated before any fit could be read at all.

On Gemma the tail from layer L is a stateless, position-wise-causal function of the residual
matrix. The cached and uncached responses compute the same function **by construction**: a
key-value cache is a lossless per-position record of the prefix, and a perturbation at a later
row cannot reach it.

What that retires: `jacobian.self_check` and `require_self_check`, six comparisons whose whole
purpose was establishing the cached path; the restore/broadcast dual mode; the finite-difference
JVP fallback and the epsilon-resolution ceremony around it, since forward-mode `mx.jvp` is exact
with no recurrence in the tail; and the special status of the `future` readout, which was
described as "the readout that can see a recurrent channel" and now simply is the broadcast
component.

The step-size ruling of section 15 of the lens-fitting requirements, and the diagnostic Codex
ran against it, are hybrid-specific. They do not carry over and they do not need to.

### 2.3 A cleaner layer contrast than the hybrid gave us

The two predicates are the same formula:

- Qwen: `is_linear = (layer_idx + 1) % full_attention_interval != 0`
- Gemma: `is_sliding = (layer_idx + 1) % sliding_window_pattern != 0`

So the writer convention holds verbatim with period 6, and the band machinery landed for issue
86 generalises with a field rename and a predicate, not a rewrite. For 4B: global-written layers
are 6, 12, 18, 24 and 30; the registry's default fractions give in-band layers 11, 17, 23 and 28,
none of which is global-written, so every one takes a partner and the kind-matched family is
**6, 11, 12, 17, 18, 23, 24, 28, 30, 34** — four adjacent sliding-global pairs, better coverage
of the contrast than Qwen gave.

It is also better science. On Qwen an attention block and a gated-delta block differ in
everything: parameters, state, kernel, numerics. A readability difference between them means
nothing in particular. On Gemma a sliding block and a global block are the same module class
with the same parameter shapes, differing in exactly two declared things, the mask window and
the RoPE base. That is a near-clean single-variable contrast on receptive field.

**One caveat with teeth: the contrast is null by construction below the window.** A causal mask
with a window is vacuous when the sequence is shorter than the window. The default corpus length
is 128 against a window of 1024, and the hosted lens was itself fitted at 128 tokens. Any claim
resting on the sliding-global contrast needs contexts longer than 1024 and must record the
window in its conformance block. A contrast run at 128 tokens is a null by construction, not a
finding.

---

## 3. What the pivot costs

Gemma 3 is not a drop-in. The **hand-run** forward loop in `arch.py` is wrong for it in three
ways, and that loop is what the J-lens estimator and all lens fitting use.

1. **The embedding scale is missing.** `arch.embed` returns the embedding directly; Gemma
   multiplies by the square root of the hidden size, about 50.6x at 2560, and does it by
   rounding the scalar through bfloat16 first. The same omission sits in the native-final-residual
   diagnostic. Reproduce the bfloat16 rounding, not a float32 square root.
2. **One mask where Gemma needs two.** `arch.masks` returns a single attention mask and
   `run_block` selects by kind. Gemma builds a global mask and a windowed mask and dispatches per
   layer index.
3. **`layer_kind` cannot see the distinction.** It keys on a block-level flag; Gemma's is one
   level down on the attention module. Every Gemma block currently reports the same kind.

**These must land as one change, with a fourth item, and nothing may ship piecemeal.** The
residual-equivalence gate is the only detector, and its prompt is padded to 64 tokens. Both Gemma
masks are identical below the window, so **fixing the embedding scale alone turns the gate green
while leaving the mask defect silently wrong on every real sequence.** The fix and a preflight
prompt longer than the sliding window are one unit.

Design constraint: do not widen `layer_kind`'s return vocabulary in place. It feeds
`ResolvedSpec.layer_types` in every recorded manifest, and attention capture is refused for any
block whose kind is not `attention`, which would be five of every six Gemma blocks. Put the
sliding-global selector beside `layer_kind`, not inside it.

Separately, the rotating key-value cache is refused in three places, two of which fail closed
and one of which does not actually catch it. Do not simply widen the type gates: the capture path
reads column j as absolute source position j, which a rotating cache breaks. Either carry the
rotation bookkeeping verbatim or restrict capture to the global-attention layers.

**The live-reading path is the exception and this decides the schedule.** `NativeCapture` wraps
the model's own forward, so it gets Gemma's two masks and its embedding scale for free. The live
lens needs only the cache guard widened. The hand-run loop needs all of it.

---

## 4. Do not spend time on these

The adversarial pass knocked down five items that reading alone would have promoted to blockers.

- **`chat.extra_stop_tokens` being unwired is not a blocker and must not be wired as it stands.**
  Gemma's config declares both terminators, MLX passes the whole set into the tokenizer, and the
  generation loop stops on membership. Turns will end correctly with no change from us. Worse,
  the obvious fix crashes: adding a token absent from the vocabulary raises, and it would raise
  after the weights are loaded. Delete the field or leave it. Issue 98 stands as a record-honesty
  defect, not a blocker. **The Chief called this a blocker earlier today on the first half of the
  reasoning without checking the generation path; that was wrong and is corrected here.**
- **The frozen module-level end-of-turn constant** is inert cleanup: turn completion fires through
  the model-agnostic fenced-block route.
- **The training preflight envelope** is opt-in per registry entry; omit the launch block and the
  recurrence clauses never execute. The real hazard is the inverse, an author who silences the
  error by declaring a recurrence mode and gets a passing run whose log claims a schedule that
  never ran.
- **The Qwen-fitted memory envelope** gates nothing today: the footprint rejection runs only under
  the training consumer, and the training path does not pass one. It is a misleading published
  number, not a refused run.
- **`cache.strategy: auto` misresolving** is inert, because the trim cache re-tests trimmability
  every turn and degrades to none. Declare the strategy explicitly anyway.

The Qwen-specific attention import does not fail on a Gemma-only run; the class exists and the
real gate is an isinstance reached only when head capture is requested. Do not spend time on it.

---

## 5. The landmine: a lens has no identity

`LensMaps.load` validates three things: the file hash, that each map is hidden-size square, and
that the layer index is below the layer count. **It does not record which model the lens was
fitted on.**

Qwen3.5-4B and Gemma 3 4B are both 2560-dimensional. The Qwen lens carries layers 1 to 31;
Gemma's layer count is 34; every check passes. **The Qwen lens will load onto a Gemma view
without a complaint and produce plausible foreknowledge ranks.**

Today the only thing preventing it is an accident: the pilot script pins the lens, its hash and
the registry as module constants, so it dies elsewhere first. The moment that script is ported,
the accident is gone.

A lens must carry a model identity, checked at load. This is filed and it lands before any Gemma
lens is loaded.

---

## 6. The order of work

**Phase 0, no box time.** Pin the checkpoint and record its four config files. Rule on the tool
role, below. Author the registry entry. Note that the training batch size and sequence length are
indexed bare in preflight, so an empty train block loads the model and then dies on a bare key
error, wasting the slot.

**Phase 1, no box time, runs alongside whatever holds the box.**
1. The architecture-view port as one change: embedding scale, dual masks, the kind predicate, the
   first-cache selection, together with a preflight prompt longer than the window. Multi-day.
2. The evaluation chat blockers. The generation prefix is hard-coded to a ChatML marker and
   asserted, so every generation render raises on Gemma. Move it into the chat spec as a declared
   field; **do not delete the assertion**, because the same value is stamped into manifests and
   into the lens corpus identity, and deleting the guard converts a loud failure into a
   self-consistently falsified corpus identity on the J-space artefact.
3. The stop-set defect: the closing tool tag is looked up as a single token and returns the
   unknown-token id on a tokenizer that lacks it, making the unknown token a live terminator.
   Round-trip the id before adding it.
4. Lens identity, section 5.
5. Rotating-cache support, with the position bookkeeping, not a widened type gate.

**Phase 2, box time, in order.** Preflight on Gemma, expected to fail the residual gate before
the port lands and worth running anyway because the artefact records the real per-layer cache
interleave and the error magnitude confirms the embedding diagnosis. Then a single-trajectory
smoke test. Then the full evaluation suite. Then the J-space pilot.

**Deferred:** training and LoRA on Gemma; the launch envelope; the residual-band definition;
Gemma Scope 2; the span-masked attention experiment, whose stated purpose was the
attention-against-recurrent asymmetry and has no meaning on a dense model.

---

## 7. The first real reading: the Gemma 3 live-lens pilot

The live-lens pilot is the only instrument in the tree that reads intermediate representations
**while the model completes tasks**, which is the Director's priority verbatim. It has a completed
Qwen counterpart, so the figure is a two-panel comparison rather than an orphan, and it needs no
fitting.

Prerequisites are the registry entry, the cache guard, the generation-prefix field, the lens
identity check, and the lens download and conversion. The pilot script itself needs its pinned
constants replaced by arguments, which the audit's critic correctly flagged as unlisted work.

Six layers, chosen so the layer choice **is** the experiment: 11 and 12, and 23 and 24, two
adjacent sliding-global pairs; 30, a late global-written layer; 34 as the base rate. Nine
episodes, keeping the long ones, because contexts above 1024 tokens are the only place the
contrast is live. Base model, greedy, no adapter, capture forcing no cache reuse, head capture
off. Estimated well under an hour and around 5 GiB, so no declared window is needed.

The figure: share of emitted tokens whose competition rank is within ten in the lens distribution
read one, four and eight positions earlier, by layer, faceted by span, with the final layer as the
base rate on every row. It renders as two panels, Qwen beside Gemma, on identical axes. The new
column the pivot buys, at no extra cost, is that layers 12, 24 and 30 are global-written and 11
and 23 are sliding-written. A higher share at the global-written layers would be the first
reading of where Gemma 3's long-range routing lives.

Pre-registration before any record is read, as the house rule requires.

---

## 8. Our lens and Gemma Scope 2

They are complements with a strict ordering. **Our lens is primary; Gemma Scope 2 is the
interpreter of second resort.**

A lens is a change of coordinates into the final space, read through the model's own unembedding;
it is total, and it answers the headline measurement, which is the competition rank of the token
the model actually emitted. A sparse autoencoder is a change of basis into labelled sparse
features; it is lossy, and a latent has no rank against an emitted token. Nothing in this tree
consumes a sparse autoencoder, and the whole live path is built on a matrix.

Gemma Scope 2 earns its place on the question the lens cannot answer: when a layer reads badly
through the lens but a state probe says the task state is there, the dictionary says what is
there instead. Its transcoders, released at every layer, are the right tool for the circuit
question of why a call token was already ranked third at layer 18. Fit or download the lens
first, read the profile, then point the dictionaries at the two or three layers the profile made
interesting. Do not download all-layer dictionaries speculatively.

Two risks to measure before any claim rests on them: the dictionaries were trained on
unquantised activations at Google's checkpoints, and our runtime is a 4-bit conversion. The
reconstruction error under quantisation is unmeasured.

---

## 9. Rulings the Director owes

1. **Is Qwen3.5 retired or does it stay a live comparator?** This decides whether the layer-kind
   vocabulary must stay backward-readable against every existing Qwen record, and what becomes of
   the adapter-depth ablation, the cache acceptance, the frozen head population and every
   calibration point. Three plans assume a port; none asks what happens to the record the port
   invalidates.
2. **The tool role.** Every conversation here is system, user, assistant, tool, assistant, tool.
   Gemma's template has branches for user, assistant and system only, and enforces alternation
   with an explicit exception. A tool message either renders empty or raises, and the alternation
   check rejects our shape regardless. Re-role observations as user turns, which preserves
   alternation and changes the training distribution, or ship a custom template. This is a
   research-design decision, and it changes what supervised targets look like.
3. **What a head's written vector means on Gemma**, given that the post-attention norm is applied
   to the sublayer output before the residual add, so per-head contributions no longer sum to what
   enters the stream. It decides whether head-capture work is worth doing at all, and every read
   share depends on it.
4. **The checkpoint and its precision.** Four-bit for evaluation, to stay comparable with the Qwen
   baseline; bf16 for lens work, because a finite-difference derivative through quantised weights
   puts the quantisation error in the numerator. The Director's account is authenticated, so our
   own conversion from the official weights is available and gives provenance we can name.
5. **Whether an instruction-tuned Gemma run is on the near path**, since every adapter we own is
   Qwen and the programme's question is about a trained agent. If it is, several deferred items
   become near-term.

---

## 10. Which size: 4B now, 12B as the planned step up, 27B only off this laptop

The Director's order raised model size explicitly ("a larger parameter model with an
architecture more friendly to the lens may be superior"). The first plan answered the
architecture half and not the size half. Figures below are read from the official configs and
the hub, not estimated.

| | 4B | 12B | 27B |
|---|---|---|---|
| decoder layers | 34 | 48 | 62 |
| hidden size | 2560 | 3840 | 5376 |
| sliding window | 1024 | 1024 | 1024 |
| terminators | `[1, 106]` | `[1, 106]` | `[1, 106]` |
| MLX 4-bit text-only | 2.56 GB | 7.23 GB | 16.03 GB |
| hosted lens, instruction-tuned | 432.5 MB | 1,386.1 MB | 3,526.0 MB |
| lens resident as float32 | 0.81 GiB | 2.58 GiB | 6.6 GiB |

**27B is out on this host.** Its weights alone are 14.9 GiB against a 17.76 GiB working set,
before a lens. It is a RunPod question or nothing.

**12B fits and is the meaningful step up.** Model and lens together are about 9.3 GiB resident
before activations, which needs a declared window under R47 but is not close to the ceiling. Two
things must change first, and both are small: `artifacts.py`'s one-gigabyte cap refuses a 12B
lens at 1.39 GB, and the memory envelope's calibration is fitted at 4B.

**There is a real scientific reason to prefer 12B for the lens work specifically, and it is not
parameter count.** With a period of six, a block is global-attention when its index plus one
divides by six. 48 divides by six and 34 does not. So **12B's final block is global-attention,
and 4B's final four blocks are all sliding.** On 4B the top of the model, which is where the
lens is most accurate and where the readout happens, has a hard-bounded receptive field with no
global layer above it. That is a structural asymmetry of exactly the kind this project has
named before as a structural zero rather than a finding. On 12B it does not exist.

**Recommendation: run the port and the first pilot on 4B, then repeat on 12B.** Every port
defect is identical on both, and proving it on the smaller model is faster and cheaper; 4B's
weights and lens are already downloaded and verified; and the evaluation comparison against the
Qwen record is like-for-like only at 4B. Then repeat the pilot on 12B, where the layer contrast
is cleaner and the top of the model is not truncated, and the two together answer whether a
depth profile is a property of Gemma 3 or of one size.

Gemma Scope 2 covers both sizes at every layer, so nothing about that choice is foreclosed.

---

## 11. Corrections, and the reordering they force

Appended rather than edited in, because three of these correct the sections above and one
corrects a contradiction inside this document. The Deputy read the checkpoint's weight shapes
rather than its config; the Chief verified each against the same files.

**The vocabulary is 262,208, not the 262,144 in section 1.** The embedding on disk is
`[262208, 2560]`. The smaller figure is MLX's dataclass default, which the multimodal wrapper
overwrites before the text arguments are built. The size of the correction is trivial and its
kind is not: anything token-denominated must come from the loaded view, never from a config
field or a default.

**The readout is tied, and the flag is set at load rather than at construction.** The checkpoint
carries no `lm_head`. The text model's constructor sets the tie flag false and builds one; the
weight sanitiser sets it true and removes the layer when the weights carry none. Anything that
inspects the model before weights are loaded reads it backwards, and both `unembed` and
`native_readout` branch on it.

**The `text_config` is sparser than section 1 implies, and the defaults are load-bearing.** Head
counts, head dimension, vocabulary, both RoPE bases, the norm epsilon, the pre-attention scalar
and the window pattern are all absent and all supplied by MLX. The Deputy confirmed the injected
head counts against the projection shapes: a query projection of `[2048, 2560]` over head
dimension 256 gives eight query heads, and `[1024, 2560]` gives four key-value heads. Correct
for this checkpoint, silently wrong for any other size.

**A contradiction in this document, and section 6 is the wrong half.** Section 4 refutes the
memory-envelope blocker on the ground that the footprint gate runs only under the training
consumer and no caller in `src/` passes one. Section 6 then warns that the bare training indexes
in preflight would crash a first Gemma run after the weights loaded. Both cannot be true, and the
verified one is section 4: every `require_preflight` call site passes only a spec and a skip
flag. **The key error cannot fire and no slot is at risk.** The registry entry still comes first,
for the better reason the Deputy gives: without one, every Gemma run takes the fallback spec,
which hands the model a ChatML turn ending and declares a cache strategy and a thinking mode with
no record of why.

### The reordering, and it is the useful part

**Rotating caches are built on every run, not on long ones.** `make_cache` returns an ordinary
cache only where the block index plus one divides by the pattern, which for 34 layers is five
blocks, and a rotating cache for the other twenty-nine. So the capture guard's
`type(c) is not KVCache` refuses on the first forward of the first turn. (Twenty-nine and five,
not twenty-eight and six.)

That looks like bad news and is good news, because of what it forces us to check. The refusal
protects two different things and only one of them is real:

- **Residual capture is safe under rotation.** It emits against `_offset`, the monotone
  total-token counter, which a rotating cache maintains identically to an ordinary one. Verified
  at the emit sites and at the offset-agreement check above the guard.
- **Head capture is not.** The attention path maps a weight column index to an absolute source
  position by identity. Under rotation a column is not its absolute position, and at most
  `max_size` columns exist. That arithmetic is wrong and no widening of a type gate fixes it.

So the correct change is narrow and provable: **accept a rotating cache when head capture is off,
and keep refusing it when head capture is requested, with the column arithmetic named as the
reason.** The first pilot runs with head capture off in any case.

**Therefore the first reading does not need the multi-day port.** Section 3 already says the
live-reading path is the exception, because capture wraps the model's own forward and inherits
its masks and its entry transform; the ordering in section 6 did not follow its own finding. The
corrected order:

1. **The registry entry**, delivered, with a covering test so it is not the first registry file
   the suite ignores.
2. **The pilot path**, all small: the narrow cache guard; the generation prefix moved into the
   chat specification with its assertion kept; the lens identity check; and the pilot script's
   pinned constants replaced by arguments.
3. **The pilot.** The first real reading of Gemma 3's intermediate representations during task
   execution, which is the Director's stated priority, and it arrives days earlier this way.
4. **The architecture-view port**, which the hand-run loop needs and which therefore gates lens
   *fitting*, the Jacobian estimator and every claim resting on the sliding-against-global
   contrast. Still the largest item, still landing as one change with its longer gate and its
   negative control.
5. Then the single-token sweep, the stop set, rotating-cache bookkeeping for head capture, and
   issue 98 in its old place.

The port has not become less important. It has stopped being in front of the thing the Director
asked for.
