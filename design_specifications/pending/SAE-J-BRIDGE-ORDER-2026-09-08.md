# Codex order: build the SAE-to-J-lens bridge, in two stages, and not the rest

**Issued by the Chief on the Director's instruction, 2026-09-08**, against the derivation
*SAE–J-lens geometry and causal state abstraction*. The derivation is sound where it was checked
numerically: its two counterexamples, its pair-weight factor and its averaging-cancellation result
all reproduce. What follows takes the part of it that is buildable and useful now and says plainly
what is not, so that nobody points at the whole document and starts.

## The one-line summary

**Build the exact score decomposition first. It is exact, it needs no model forward, no sparse
solver, and no box time. Build the sparse cone machinery only if the first stage shows it is
needed. Do not build the causal-abstraction programme at all yet; our throughput cannot reach its
acceptance rule by three orders of magnitude.**

---

## Stage A — the exact score decomposition (build this)

The derivation's §5.1 applied to our objects. For a residual layer `L`, a lens map `J_L`, the
model's readout `W`, and a sparse decoder `D`:

    L  = W J_L                       the lens score map
    Lh = L b + (L D) z + L e         exact, for every activation h = b + Dz + e

The signed quantity `z_i * (LD)_{v,i}` is feature `i`'s contribution to the score on token `v`,
and the entire approximation is carried by the single term `L e`. **This is an identity, not a
model.** It is the composition the Chief proposed and the derivation confirms it is exact.

**A1 — the dictionary readout, no activations required.** For each sparse feature `i`, compute
`L d_i` and keep its top tokens. This answers *what does this feature push the output toward, read
from this layer*, and sets that beside the feature's published auto-interpretation label. It needs
only three matrices already on disk and **no model forward at all**.

**A2 — the run-resolved decomposition.** Take activations already captured by a pilot run, encode
them, and report per position: the top features by `|z_i (LD)_{v,i}|` for the emitted token `v`,
and the residual share `|L e| / |L h|`. This is what turns *the residual at layer 18 predicts this
token* into *these labelled features are active and these two are what push toward it*.

### Costs, computed at Gemma 3 4B's real dimensions

| object | shape | size |
|---|---|---|
| one lens map | 2560 x 2560 | 0.03 GB |
| unembedding | 262,208 x 2560 | 2.69 GB |
| one 16k decoder | 2560 x 16,384 | 0.17 GB |
| **full score transfer `LD`** | 262,208 x 16,384 | **17.2 GB — do not materialise** |

So `LD` is never formed. Compute `JD = J_L D` first, which is 215 GFLOP and trivial, then take
`W (JD)` column by column keeping only top-k. That is 22 TFLOP per layer: about **150 seconds per
layer on CPU BLAS**, 80 minutes for all 33, and no model is ever loaded. On the GPU it is
11 seconds per layer, but that needs the box and is not worth contending for.

### Preconditions, and one is currently unmet

The derivation requires the sparse dictionary and the lens to share **hook, layer, coordinate
scaling and checkpoint**. Hook and layer align and this was verified: Gemma Scope's residual site
is the output of block N, which is this repository's probe layer N+1, and the lens carries maps on
the same index.

**The checkpoint does not align.** Gemma Scope was trained on Google's unquantised weights; our
evaluation entry is a 4-bit conversion. For the bridge this is not a disclosure, it is a **failed
precondition**. Build and run stage A against `gemma3-4b-bf16`. If a 4-bit number is ever wanted,
it is a separate measurement of what quantisation does to the bridge, and must be labelled as one.

Download sparse dictionaries **one layer at a time**, residual site, 16k width. One is 336 MB; the
repository is 4.65 TB. Take the layers the map makes interesting, not the set.

---

## Stage B — the sparse cone fit (build only if A justifies it)

The derivation's §§4 and 5.2. The J dictionary `B` has columns `normalise(J_L' w_v)`, one per
vocabulary entry, so it is **2560 x 262,208 = 2.69 GB per layer, 88.6 GB across all layers**. It
must be built per layer on demand and not stored.

The fit is nonnegative and sparse at budget `k`, and the derivation is explicit that the top scores
are *not* the answer, because the Gram matrix `G = B'B` carries the overlap between dictionary
directions. Three things must be reported with any number from it, and each is a separate error:
the SAE reconstruction error, the lens estimation error, and the solver's own optimisation gap,
which a solver **cannot certify from its own residual**.

**Carry these two cautions in the code and in every artifact, not only in a paper.** A residual left
by the fit can itself lie inside the cone at that budget — verified with a two-dimensional example —
so *unexplained by this fit* never means *outside the workspace*. And energy is not additive over a
non-orthogonal dictionary, so a per-feature share of `||h||^2` is not a partition and can exceed
one. **A feature's contribution to a score and its share of the geometry are different quantities in
different units and must never appear in one table.**

---

## Do not build: the causal-abstraction programme

Sections 8 to 16 define predictive sufficiency, update closure, causal state abstraction and an
acceptance rule over declared tolerances. The mathematics is correct. **It is out of reach at our
throughput and pointing at it now would waste months.**

The derivation's own bound gives the cost. To bound the diagnostic distance below 0.05 across ten
diagnostics at 95% confidence needs **2,397 independent episodes per arm**. Our measured rate is
fifteen episodes in fifty-eight minutes, so that is **6.4 days of continuous box time per arm**, and
an acceptance rule has many arms. Positions inside an episode are not independent draws and cannot
be substituted.

Two of its cautions apply to us **now** regardless, and are already in the map's pre-registration: an
averaged Jacobian can cancel a causal sensitivity, so a low lens reading is not evidence of low
causal influence; and mutual information between a feature at two times does not identify persistent
memory when a transcript retains the fact and the model recomputes it. That second one is a trap our
agent protocol walks straight into, because the transcript is retained by construction.

---

## Acceptance

- **A1** reproduces `L d_i` for a named feature two ways: through the composed matrices, and by a
  direct product for that single column. They agree to float32 tolerance.
- **A1** carries a negative control that bites: a lens from a *different layer* against the same
  decoder must give a measurably different token set, and the check must be shown to fail when the
  layers are deliberately mismatched. A bridge that cannot tell layers apart is not reading one.
- **A2** reports `|L e| / |L h|` per position. If the residual term dominates, the decomposition is
  describing the SAE's failure and not the model, and the artifact must say so rather than ranking
  features under it.
- Every artifact records the checkpoint, the lens digest, the dictionary layer, width and sparsity,
  and the residual convention, per R57 and R60.
- No box time for stage A. If stage B needs the machine, announce with a projected peak and its
  basis, per R60(c).

## What this may and may not be said to show

It is a **measurement instrument**. A large feature contribution to a lens score is a contribution
to that score, at that layer, under that averaging convention. It is not a token probability, not a
causal derivative, and not evidence of a workspace, a maintained state, or flexible access. The
derivation is unusually careful about this and the artifacts should inherit its care rather than its
vocabulary.

---

## Amendment: which dictionary, on the Director's authorisation to download

**Authorised: one sparse dictionary, downloaded. And take the every-layer suite, not the one whose
configuration is already on disk.**

Codex reports finding a layer-18 configuration cached with no weights beside it. That file is an
**artefact of the Chief's exploratory inspection this morning** — a single 248-byte `config.json`
pulled while sizing the repository — and not a considered choice of dictionary. Downloading its
weights would be following a footprint.

The repository holds two different things at the residual site:

| path | layers available | width | one file |
|---|---|---|---|
| `resid_post/` — a single-layer deep dive | **only 9, 17, 22, 29** | 16k and others, 52 variants | — |
| `resid_post_all/` — the every-layer suite | **0 to 33, all 34** | 16k | **335.7 MB** |
| `resid_post_all/` at the wide setting | 0 to 33 | 262k | 5,370.8 MB |

**Take `resid_post_all`, 16k width.** The deep dive exists at four layers only and therefore cannot
support a readout across the map's layer set at all; and mixing a deep-dive dictionary at one layer
with the all-layer suite at others would confound every across-layer comparison the bridge is for.
Declare the sparsity variant once and use **the same one at every layer**, because sparsity changes
every quantity downstream and a mixed set is not a series.

The 262k width is 5.4 GB per layer and is not a first pass.

**Start at Gemma Scope layer 17, which is this repository's probe layer 18.** It sits inside the
map's layer set, and it is also one of the four deep-dive layers, so a later cross-check between the
two suites is possible there and nowhere else. That is worth having at no cost.

**And Codex is right about R61**, which the Chief failed to find because it is written inside a
section rather than under its own heading. Heavy work announces a window whether or not it loads a
model. The download is network and disk and needs none; the small numerical checks need none; the
full matrix readout at roughly 150 seconds per layer does, and it must wait for a free box and be
announced with a projected peak per R60(c). The Deputy holds the box at the time of writing.

---

## Second amendment: the labels. Chief, on the Director's source audit, 2026-09-08

The dictionary choice is unchanged. What changes is everything this order promised to print beside a
feature. The Director's audit found that Neuronpedia's layer-17 labels describe the deep dive rather
than the ordered every-layer suite, and ruled that labels are unavailable unless an exact mapping is
found, since matching feature numbers alone would produce false pairings. **That is confirmed, and
the limitation is broader than it was stated.**

### What is true, all of it fetched or read

**Neuronpedia says so itself, in a machine-readable field.** The feature endpoint for
`gemma-3-4b-it/17-gemmascope-2-res-16k` returns, inside its own `source` object,
`hfFolderId: "resid_post/layer_17_width_16k_l0_medium"`. The provenance pointer is published; the
audit is confirmed by the source rather than inferred from layer numbers.

**There are no published labels for `resid_post_all` at any layer at all.** Three independent
confirmations: SAELens's registry covers `resid_post_all` for layers 0–33 with **zero** Neuronpedia
bindings, while all twelve residual bindings sit in the deep-dive release at 9, 17, 22 and 29;
probing the API returns 200 at exactly those four layers and 404 at every other layer of 0–33; and
Neuronpedia serves a 65k residual source, a width that exists **only** under `resid_post/`. So the
chosen dictionary is unlabelled at all thirty-four of its layers, not merely at layer 17.

**The sparsity does not match and cannot be made to.** Neuronpedia labels `l0_medium`, declared
`l0: 60`. `resid_post_all` ships only `l0_small` at 20 and `l0_big` at 120, and there is **no
`l0_medium` anywhere in `resid_post_all`, at any layer or width**. The labelled configuration is not
a different run of the same setting; it is a setting the chosen suite never published.

**File size will not tell the two apart, and this is a trap worth naming.** The controlled pair —
same declared hook `model.layers.17.output`, same width 16384, same `jump_relu`, same `l0: 20` —
are both exactly **335,686,016 bytes** with byte-identical 384-byte headers, and first differ at
byte 385, the opening of `b_dec`. Identical size is forced by shared shape and dtype. **A
mis-targeted download lands a file of exactly the right size.** Verify by hash, never by length.

**Where a label does exist it is an unscored machine guess.** `explanationModelName` is
`gemini-2.5-flash-lite`, `scores` is empty, and coverage is partial at roughly 92.5% on a 40-index
sample. A caution on top: the base and instruction-tuned label *texts* were identical at 14 of 14
sampled indices despite being records on SAEs trained on different models. That is unexplained.
**Do not cite base/instruct label agreement as evidence of anything.**

**A labelled every-layer suite does exist for this model, at the wrong site.** `transcoder_all`
covers layers 0–33 complete, with labels, at `blocks.N.hook_mlp_out`. It reads the MLP output, so
the identity `Lh = Lb + (LD)z + Le` at the residual does not apply and it is not a substitute. It is
recorded here so that "no labelled every-layer dictionary exists" is not entered as a finding, since
that is false.

### The one measured result that looked like a route, and why it is not

Reading decoder rows by HTTP byte range, 768 features sampled across the index range, same-index
decoder cosine between the deep-dive and every-layer 16k dictionaries at layer 17:

| comparison | mean cosine | notes |
|---|---:|---|
| deep dive vs every-layer, same index | **0.672** | median 0.850, 71% above 0.5, 43% above 0.9 |
| the repository's own different-seed control | **0.013** | 0 of 256 nearest-neighbour hits |

So the indices are **not** arbitrary relative to each other, and the alignment survives across layers.
The likely reason is shared initialisation lineage rather than a shared run.

**It still does not open a label route, for two reasons that are each sufficient.** Eighteen per cent
of features sit below 0.2 cosine, so index-matching would silently fabricate roughly **one pairing in
five** — exactly the false pairings the Director's rule forbids, and silent because the other four in
five would look right. And the aligned pair measured is `small` against `small`, **neither of which
is labelled**: the labels are on `l0_medium`, which has no counterpart in the chosen suite at all.

### What survives, and it is the better instrument

**The dictionary ships its own interpretability evidence.** `examples.safetensors`, read by HTTP range
without downloading it, carries `top_tokens` and `top_logits` at [16384, 10], `bottom_tokens` and
`bottom_logits` at [16384, 10], `activations`, `seq_ids` and `positions` at [16384, 1000],
`feature_frequencies` at [16384], and a token corpus `tokens` at [236963, 512]. **Right dictionary,
every layer, no mapping, no third party.** This file, not Neuronpedia, is A1's companion.

**And it hands A1 a free correctness check nobody asked for.** A1 computes `L d_i = W J_L d_i` and
keeps its top tokens. Replace `J_L` with the identity and A1 becomes `top_k(W d_i)`, which is what
the shipped `top_logits` already is. Compare the two and report the overlap. That catches a wrong
decoder orientation, a wrong unembedding, or a tokenizer indexing error — none of which the layer
mismatch would reveal, and all of which would otherwise be invisible. **The gap between
`top_k(W d_i)` and `top_k(W J_L d_i)` is then the quantity the bridge exists to measure: what the
Jacobian adds over the direct path.**

### The rule, stated so it cannot be misapplied

**Labels are unavailable for this dictionary. The label field ships as `unlabelled`, with the reason
recorded verbatim in the artifact:** *no Neuronpedia source maps to `resid_post_all`; the residual
labels index `resid_post/layer_17_width_16k_l0_medium`, a different training run at a sparsity this
suite never published.*

**A description we generate is not a published label and no artifact may present it as one.** Where
A1's own top-token readout is printed, it is headed as ours, and it records the layer it was read
from. This distinction is the whole of the Director's rule and it survives even a perfect mapping.

### What changes in the order's own text

- **Stage A1, "and sets that beside the feature's published auto-interpretation label"** — struck.
- **Stage A2, "these *labelled* features are active"** — the word is withdrawn. It becomes: *these
  features are active, these two are what push toward it, and here is what each pushes toward read
  from this layer*, each feature named by suite path and index.
- **First amendment, "a later cross-check between the two suites is possible there and nowhere else.
  That is worth having at no cost"** — both halves false. The deep dive exists at hook layers 9, 17,
  22 and 29, all inside `resid_post_all`'s range, so the cross-check is available at four layers,
  this repository's probe layers 10, 18, 23 and 30. And it costs a second 335.7 MB download of a
  dictionary the bridge does not otherwise use.

### Cost correction, which is 3.4x

A folder-level or `snapshot_download` pull takes `params.safetensors` at 335,686,016 B **and**
`examples.safetensors` at 815,733,608 B, so **1.15 GB per layer** against the 335.7 MB this order
budgeted. Fetch `params.safetensors` and `config.json` **by name**; pull `examples.safetensors` only
at layers actually interpreted. Read its header by HTTP range and confirm the schema for the variant
actually chosen before committing to it.

### Correction to the A1 validation check, on Codex's caveat

**Codex is right and the defect is mine.** The second amendment says A1 with `J_L` replaced by the
identity "is precisely what the shipped `top_logits` is". That asserted a convention instead of
establishing one. Google's tutorial applies the final normalisation in its direct feature readout,
and whether the shipped `examples.safetensors` follows that convention is unknown. The sentence is
struck and replaced by what follows. Codex's handling — keep the raw comparison, record the overlap,
treat a mismatch as a stop for investigation and **not** as proof the decoder orientation is wrong —
is correct and is adopted.

**Codex also found the Gemma-specific trap independently, and it is the one that matters.** Gemma's
RMSNorm applies `1 + weight`, not `weight`:

```
def __call__(self, x):
    return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)
```

The stored vector is centred near zero, so applying it raw as a gain would produce near-noise. Codex
has saved both the stored vector and the effective gain.

### There are two conventions to distinguish, not three

For a **single** decoder direction `d_i`, applying the full final RMSNorm and applying only the gain
differ by a positive scalar, because the normaliser `c = sqrt(mean(d_i²) + eps)` depends only on
`d_i`:

    W · rmsnorm(d_i) = (1/c) · W · (d_i ⊙ (1 + w))

Verified numerically at Gemma's real width: max absolute deviation `9.2e-14`, exact proportionality,
**top-10 identical and the full ordering identical.**

So for a top-k or overlap comparison, "full RMSNorm" and "gain only" are the **same test**. The only
distinction that exists is **raw `W d_i`** against **gain-applied `W (d_i ⊙ (1 + w))`**. Do not run
three arms and do not report full-norm and gain-only as separate conditions; they cannot disagree.

*The scalar does matter if logit **values** are compared rather than ranks. Compare ranks and
overlap, and the question disappears.*

### The check is a discriminator, so design it to discriminate

A single failed comparison has three causes that look identical: a convention mismatch, a transposed
decoder, and a tokenizer indexing error. Running both arms turns an ambiguous stop into a decision:

| raw arm | gain arm | conclusion |
|---|---|---|
| matches | does not | shipped file is raw; convention settled, bridge proceeds |
| does not | matches | shipped file applies the final gain; convention settled, bridge proceeds |
| **both match** | | uninformative — the two arms agree, so the check has no power here; report it and rely on other evidence |
| **neither matches** | | **not a convention problem.** Orientation or tokenizer indexing. Stop and investigate. |

Only the fourth row is a stop. The first two are the check succeeding, and the point of running both
arms is that one of them is expected to fail.

**And Codex's ordering is right**: run the raw arm first and record its overlap before looking at the
other, so the first number is not chosen after seeing which convention agrees.

---

## Third amendment: three premises have moved, and what is true now. D-CRO, as lens owner, on the Chief's instruction, 2026-09-10

The Director has asked whether the derivation has gaps and would work. This amendment says what is
true of its preconditions today, each claim checked against a source rather than against a message,
and it corrects two sentences of the derivation and one of this order.

### 1. The checkpoint precondition is met. The dictionary is not on disk.

**Met.** The bridge needs the dictionary and the lens on one checkpoint, and this order failed it
because the evaluation entry was the 4-bit conversion. The bf16 entries now exist in the registry:
`gemma3-4b-bf16` (`models/gemma-3-4b-it-bf16`, base `google/gemma-3-4b-it`) here, and
`gemma3-4b-cuda-bf16` (`hf_id: google/gemma-3-4b-it`, the official unquantised snapshot, loaded
text-only) on the device. Gemma Scope was trained on that snapshot. Stage A runs against either.

**Not on disk, and the earlier sentence saying otherwise is withdrawn before it is repeated.** Both
caches were inventoried. They hold **two 248-byte `config.json` files and no weights**: the global
cache has `resid_post_all/layer_17_width_16k_l0_big` (`l0: 120`) and
`resid_post/layer_17_width_16k_l0_big` (`l0: 150`); the project cache has the second only. Zero
`params.safetensors` anywhere. This is the first amendment's footprint, unchanged. A1 needs no
forward, which is true; it needs the 335.7 MB `params.safetensors` at each layer it reads, which is
a download, by name, verified by hash and never by length, exactly as the second amendment says.

One fact the inventory adds: **even the `l0_big` setting is not the same sparsity across the two
suites** — 120 in the every-layer suite, 150 in the deep dive at the same hook and width. The second
amendment's rule that a mixed set is not a series applies to the label `big` as well as to `medium`.

### 2. "Do not build the causal-abstraction programme" rested on throughput, and throughput is the device's. Rewritten as the device-phase programme, with its preconditions named and their state.

The mathematics was never the objection; the count was. The count stands: with ten fixed
diagnostics at 95% confidence and tolerance 0.05, the derivation's Hoeffding bound (§10, the
inequality at its line 984) gives `n ≥ log(2M/α)/ε² = 2,397` episodes per arm, recomputed. At this
order's quoted laptop rate that is 6.4 days per arm, recomputed.

**What that rate is, so the projection is honest.** Fifteen episodes in fifty-eight minutes is a
lens-capture run — Gemma 3 4B at 4-bit, eight layers read at every forward, on a laptop under the
R47 cap — not a decoding rate. No manifest reproducing that figure was found; the thirteen-episode
pilot of 2026-09-07 took 30.9 minutes with six capture layers. **The device's episode rate is
unmeasured.** For 2,397 episodes to take eight hours the card must decode at 19 times the laptop's
capture rate; for two hours, 77 times. Both are plausible for a card decoding a 4B model without
capture, and neither is a number. The first device run that generates episodes records its rate,
and the corpus time is computed from that record, not from this paragraph.

**The programme, and the four preconditions each with its state today:**

| precondition | why | state |
|---|---|---|
| **a sampled decoding path** | the estimands of §§10–11 are total-variation distances between distributions over outcomes; a greedy loop yields point masses, and the derivation says so at its line 951 | **unmet on the device.** The MLX rollout samples (`rollout.py --temperature`, default 0.7; every training config sets 0.7). The torch path is greedy by construction: `runner.py:623 torch_greedy_stream`, and `evaluate.py:111` says it "cannot honour a temperature" and refuses one. A torch sampler is a small addition and it is a **change to what the device's runs measure**, so it lands as its own commit with the temperature in every manifest. |
| **retained-state exchange for the carrier tests** | §12: replay identical tokens while resetting or exchanging retained state per layer, so the generated-text path and the retained-value path are tested separately | **seam present, intervention ordered after the migration.** Plan §6.7 names `intervene(layer, position, fn)` — present on `cuda-migration` in `torch_capture.py` — and the HF cache object, with `branch.py` (419 lines, kept) for branched continuations. Plan §13.2 is explicit that the **cache strategies themselves are not deferrable** and are WS-B's; what is deferred is the exchange intervention built on them. |
| **the episode corpus generated on the device** | 2,397 per arm, independent episodes; token-level resampling is not a substitute (derivation line 1392) | **not started.** Depends on the sampler above. The 12B entry's note that the rendered corpus is byte-identical across sizes means one corpus serves 4B and 12B. |
| **vector-level statistics, not coefficient-level** | the vocabulary directions are too correlated for the restricted Gram solves to be readable (§5.2, and the Gram caution already in Stage B) | **a rule of the analysis, met by writing it down here.** Every device-phase artifact reports contributions and distances at the level of J-space vectors; coefficient tables from a sparse solve are diagnostic only and carry the second amendment's two cautions. |

**The order's "do not build" is therefore replaced by "build in this order, on the device":** the
sampler; one episode corpus at a recorded rate; the discovery/alignment/evaluation split on separate
task families (derivation line 1392); then §11's counterfactual substitution estimand, then §12's
carrier tests, each against the tolerance the Director sets in advance. Nothing in this list runs on
the laptop, and nothing in it is blocked by the mathematics.

### 3. The 12B: the dictionary exists, and the question is alignment, not training

The registry entry `gemma3-12b-cuda-bf16` landed 2026-09-10: `google/gemma-3-12b-it`, **48 blocks,
hidden 3,840**, same tokenizer and template as the 4B. The dictionary question is answered by the hub
rather than posed as a step:

| | 4B | 12B |
|---|---|---|
| `google/gemma-scope-2-*-it` | exists | **exists** |
| `resid_post_all` layers | 0–33, all 34 | **0–47, all 48** |
| widths / sparsity | 16k, 262k / small, big | **16k, 262k / small, big** |
| `params.safetensors`, 16k small | 335.7 MB | **503.5 MB** |
| `examples.safetensors` | 815.7 MB | 815.4 MB |
| deep dive layers | 9, 17, 22, 29 | 12, 24, 31, 41 |

So **Gemma Scope 2 covers the 12B residual site at every layer**, in the same structure as the 4B,
and the Director's route of training a dictionary from activation data on the device is **not
needed for this site**. It stays available for a site or width the release does not publish, with
the acceptance A2 already carries: a declared token budget, layer set, and an ε budget per site.

What is *not* answered by the hub, and is a step:

- **Hook alignment at 12B**, to be verified as it was for the 4B: the release's `model.layers.N.output`
  is block N's output, which is this repository's probe layer N+1. Read the 12B `config.json` at one
  layer and confirm the hook string before any readout; do not inherit the 4B's verification.
- **No 12B lens exists.** The bridge needs the lens on the same checkpoint, and every lens this
  programme holds is 4B. WS-D fits the 12B lens on the device through the same torch stage and the
  same ν; until it exists there is no `J_L` to compose, and A1 at 12B is `top_k(W d_i)` only, which
  is the shipped `top_logits` check and not the bridge.
- **Cost at 12B.** `LD` would be `vocab × 16,384` per layer as before; `JD = J_L D` is
  `3,840² × 16,384 ≈ 242 GFLOP` per layer, trivial; `W(JD)` column by column is
  `262,208 × 3,840 × 16,384 ≈ 16.5 TFLOP` per layer, about the 4B's 22 TFLOP scaled by width.
  Still CPU-feasible per layer and still not worth contending for the card.

### 4. Two corrections verified against the sources, one to the derivation and one that confirms it

**The derivation's endpoint sentence is unsupported and is struck.** Line 154 reads: *"Its default
Sonnet implementation targets the penultimate residual; the main exposition uses a final-residual
schematic."* Against the sources: the released code's default is `target = n_layers - 1`
(`jlens/fitting.py:79`), and its recorder hooks the block's output, so the target is the **final
block's output before the final norm**, with `unembed` supplying the norm. The Chief's verbatim
reading of the paper (`CHIEF-JSPACE-PAPER-READING-2026-09-05.md`, line 16) records that the
reference fit *"targets the final layer by default"*, and lists a **penultimate-layer target only as
an A.7 robustness variant** (line 18). So the paper does mention a penultimate target — as a variant,
alongside frozen-QK, self-only and future-only — and says nothing about it being Sonnet's default.
The corrected sentence for the derivation: *the released code and the paper's reference fit target
the final block's pre-norm output; a penultimate-layer target is one of the paper's A.7 variants.*
The derivation's conclusion — endpoint, pair weights and output normalisation must be explicit —
survives unchanged, and is what ν declares.

**The derivation's pair-weight factor is confirmed by upstream's own words.** The paper defines the
Jacobian as an expectation; the released README states it as `J_l = E[∂h_final/∂h_l]` over
*"prompts, source positions, and all current-and-future target positions"*, and then names the
precise estimator: *"cotangents summed over target positions, then averaged over source
positions"*. The fitting docstring adds that this is *"the reduction used in the paper"*. That is
exactly the derivation's `J^sum = ((T₀+1)/2) · E[K]` over uniform eligible pairs — a positive global
factor that leaves directions and cones unchanged and that variable lengths turn into a prompt
reweighting. Our ν declares it as `target_reduction: sum, not normalised` and
`source_reduction: mean`, and the WS-D golden harness refuses a comparison whose ν differs in it.

### What changes in the order's own text

- **The one-line summary**, "*Do not build the causal-abstraction programme at all yet; our
  throughput cannot reach its acceptance rule by three orders of magnitude*" — withdrawn. It reads:
  *build the causal-abstraction programme on the device, in the order of §2 above, after the sampler
  and one recorded-rate corpus exist.*
- **"Preconditions, and one is currently unmet"** — the checkpoint clause is met; the section's
  instruction to run against `gemma3-4b-bf16` stands, and `gemma3-4b-cuda-bf16` is its equivalent on
  the device.
- **"Do not build: the causal-abstraction programme"** — the section's two cautions that apply now
  (averaging can cancel a causal sensitivity; retained transcripts confound mutual information) are
  kept verbatim in the device-phase programme's analysis rules. Its cost paragraph is replaced by §2
  above. Its 6.4-day figure was a capture rate and is not to be cited as a decoding rate.
- **Stage A's status.** SWE-2 is building Stage A under this order as amended; no Stage A code is on
  any branch yet, and the only "bridge" on a branch is Codex's transcript-lens lane, which is a
  different instrument and is not this.
