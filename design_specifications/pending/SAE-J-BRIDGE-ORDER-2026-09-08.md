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

**Correction to §1 of the third amendment, dated 2026-09-10.** The inventory that found zero
`params.safetensors` was measured before the commit and was true when made. Three minutes before
`b21b2a5` was committed, SWE-2 fetched layer 17 `l0_small` of the 4B dictionary into the primary
cache (335,686,016 bytes, verified against the LFS digest; provenance in their A1 record). Plan
§16.16 records both times. Every other layer remains a config file, and the sentence "A1 needs a
download per layer it reads" stands for all of them but that one.

## Review of Stage A, Chief, 2026-09-10 — `e2d2bc2` on `cuda-ws-c`, merged into `cuda-migration` at `5d32db8`

**Verdict: passes; merged.** Checked against the merge result, not the branch: the full suite 2,564
passed, 10 skipped at nine sites, every skip for untracked data or saved evaluations absent from a
scratch worktree, under the Chief's own window at 05:49Z; the torch set 154 passed; the
real-artefact alignment test ran against the primary checkout's lens, checkpoint and dictionary
rather than skipping; the rules suite clean; no deletion on either side since the base. The module,
the tests, both record scripts and the record were read in full.

What is right, in the order's terms. **The identity** is computed term by term and asserted on the
emitted token before anything is ranked, and the per-feature term is taken through row `v` of `L`
so no vocabulary-by-width block is ever formed; the spy at the matmul enforces that rather than a
reading of the code. **The gain-only linearisation** is the one choice made and is declared, with
the per-vector scalar dropped and score values marked as not logits. **Orientation** is read from
the stored shapes. **The readout** is read from the safetensors container by tensor name with
bfloat16 widened by bit shift, tested against known bit patterns, after the first real run died on
a float32 fixture; the fixture now carries the real dtype, which is the fixture rule turned on its
author and recorded as such. **Hook alignment** is a test on the real hook string and the real
archive keys. **The discriminator** ran the raw arm first and recorded it before the gain arm was
looked at, which is what makes 0.245 against 0.995 a measurement and settles the convention: the
shipped tokens were made through the final gain, which is the answer to Codex's caveat. **Two
ways** agree to 1.7e-6. **The negative control** bites at every layer tried and is shown to fail on
the same layer. **Labels** ship as `unlabelled` with the second amendment's reason verbatim, and
the readout sample is headed as ours. **A2** asserts the identity, refuses to rank under a dominant
residual, and is unexecuted on real activations because none exist on disk, which is the device's.

**The finding against its author is the best thing in the record.** Two predictions of the control
ordering failed, each from a matrix aggregate; the engineer measured the curve at every layer
rather than fit a third model, and the curve is the instrument: single-peaked at the hooked layer,
slow rise from below, fast fall from above, and a map near the identity is far from the readout.
The rule it yields transfers: choose controls from the measured curve, twelve or more layers away
for a strong one, and never from a distance to the identity. The one-line laptop rate claim in
this order's cost table is corrected by measurement: 39 seconds per layer on this BLAS, not 150.

**One gap, and it is the first item of the next assignment.** Nothing refuses a dictionary trained
on a different model of the same width: `hook_alignment` checks the layer and the hidden size, and
a `google/gemma-3-4b-pt` dictionary read through the `-it` lens passes both. The config carries
`model_name`; it is compared with the lens identity's model and refused on mismatch, with a test
on the pt-against-it case, and the registry entry's `base` is the third party to the same check
when the bridge runs from a registry name.

**Next, SWE-2, all on fixtures:** that check; the torch adapter for the decoder intervention
wrapper on `cuda-migration` (an encoder callable computing the JumpReLU in a declared dtype and
returning the residual's dtype and device, the decoder as the `[residual, feature]` matrix, which
is `w_dec` transposed, and `b_dec`), tested against the numpy `encode` and `decode` and through
the wrapper's residual-preservation identity; and the reconstruction-budget gate, the per-site
distribution of `|e| / |h|` and the active-feature count with the dominance threshold as a
declared input, whose real inputs are the device's after A2's first forward. **Stage B stays
unbuilt** on this order's own condition: A2's residual share on real activations is what justifies
it. A1 at layers other than 18 waits for the map to name them.

## Review of the three additions, Chief, 2026-09-10 — `9af0abf` on `cuda-ws-c`, merged into `cuda-migration` at `628a6b4`

**Verdict: passes; merged.** Against the merge result: full suite 2,574 passed, 10 skipped at nine
sites for absent worktree artefacts, with user warnings promoted to errors, under the Chief's
window at 06:10Z; the torch set 164 passed; no deletion on either side since the base. The
identity check is three-way and refuses when it cannot compare, which is the non-inert form: a
config naming no model and a lens carrying no identity are refused, not passed; the registry
round-trip test pins that every entry's name, hf id and base resolve to one string, and the real
config's `google/gemma-3-4b-it` is checked against the real lens identity in the gated test. The
adapter hands the wrapper the transposed decoder, the bias apart, and an encoder computing in the
declared precision and casting the code to the residual's dtype, with the read-only views of the
raw reader copied first and the warning that found that promoted to an error in the test; the
wrapper's residual identity holds through it on the A2 fixture. The reconstruction-budget gate
echoes its threshold and returns the mask `decompose_position` accepts. The record's after-review
section, the exploratory heading on the sweep, and the shipped-file wording are as asked; the
`-qq` note is the honest form of a count read from progress marks.

**Next, SWE-2**, a delegation from WS-E, device-free: `lab-device preflight` gains dictionary rows
and a render row. For each dictionary layer the registry entry or the run names: present in the
primary cache, verified against the hub's declared digest for its exact path (the `fetch-dictionary`
mechanism, reused, with an offline form that reads a digest recorded at fetch time), and its config's
`model_name` equal to the entry's `base` through `base_of_artifact`, each a row with a basis. And a
render row: the corpus rendered under the entry reproduces the split digests the manifest of the
laptop render records, so the 12B re-render assertion in the runbook is a command, not a sentence.
Tests with stubs and a fixture corpus; pathspec commit on `cuda-ws-c`; review and merge as before.
Stage B and A1 at other layers wait for the device's numbers as recorded.

## Review of the preflight rows, Chief, 2026-09-10 — `743b044` on `cuda-ws-c`, merged into `cuda-migration` at `8be822d`

**Verdict: passes; merged.** One `fetch-dictionary` remains in the merge result, the Chief's, and
`lab-device` parses. Against the merge result: the device-setup, rules, import-tree, bridge and
runlock sets, 163 passed, 0 skipped, with user warnings promoted to errors, under the Chief's
window at 06:44Z; and a real offline preflight on this machine naming the cached layer 17: present
with its 335,686,016 bytes, digest **undecided** because no fetch has recorded one and the hub was
not asked, model `google/gemma-3-4b-it` against the entry's base through `base_of_artifact`. That
undecided row is the row doing its job. The render row refuses a different source before any render
and names the differing split after one. The missing-file row names the fetch command with the
layer read off the folder's layout. Nothing further for SWE-2 on the laptop; Stage B and A1 at
other layers wait for the device's numbers.

## Ruling, Chief, 2026-09-10 — Codex's bridge-pairing audit (`WSA-BRIDGE-PAIRING-AUDIT-2026-09-10`, `4df1225`): a measured layer is not a measured pairing

The guard at `c4f2f9f` (merged 80857a2) is correct today because its pairing table is empty: every
declared cross-precision or cross-width reading is refused, at every sampled layer, and the
same-path and no-capture cases are allowed. The defect is in how the refusal will lift. `path_pairing`
keys on model and layer only, and any non-empty entry — even `{'relative': 0.004}` with no scope —
lifts the refusal for every crossing at that layer: a different capture precision, a different
width, the fit/capture direction reversed. The existing test inserts exactly such a bare number and
expects acceptance, so it encodes the defect.

**Ruled, for SWE-2, before any pairing is registered.** (B1) A registered pairing carries the
identity of the pair it measured — fit precision, capture precision, fit width, capture width, which
side is the lens's path and which the capture's, the lens's ν digest, the positions or reduction
and endpoint, and the context length it was measured at — and the guard compares **every** field to
the reading proposed; a measurement missing any field is unmeasured; the no-capture path stays
allowed. Reuse the existing ν, artefact-identity and position fields rather than a new naming.
Tests mutate each bound field independently: the matching fixture is accepted, and changing either
precision, either width, the direction, the lens identity, the positions or the context leaves that
pairing unmeasured; a measurement at one layer still does not transfer to its neighbour. (B2) The
"no amplification ratio anywhere" test validates the table's schema recursively and corrupts each
supported nesting location, checking the quantity's meaning and location rather than banning
numbers. (B3) One copied median (layer 33, native, the equal-norm random arm) reads 0.313000 against
0.312953 recomputed; write the declared precision consistently. Laptop only; the technique is the
audit's — test non-transfer as well as lookup.

**Correction to B1, Chief, 2026-09-10 — Codex's fix review (`WSA-BRIDGE-FIX-REVIEW-2026-09-10`,
`c55752f`): the ν digest identifies the fitting declaration, not the fitted matrix.** The six prior
findings are verified closed, each of the ten pairing fields changed and removed independently and
refused. The new finding is in this ruling: `nu_sha256` hashes the declaration, so two different
fitted archives produced from one declaration would hash identically and a pairing measured
against one would license a reading against the other. **Ruled: the pairing key carries the lens
artefact's own sha256 beside the declaration digest** — declaration identity and artefact identity
are different things and a pairing is measured against a particular matrix. A pairing missing
either is unmeasured. Tests: the same declaration with two artefacts is two pairings. SWE-2, laptop,
before any pairing is registered.

**Ruling for Stage B, Chief, 2026-09-10, fixed before any Stage B number exists (from an external
review of the programme description):** the readout dictionary is about a hundred-fold overcomplete
(vocabulary ~262k against a 2,560-wide residual) and non-orthogonal, so a nonnegative sparse fit at
any moderate budget explains a large share of almost any vector's energy — a property of
overcompleteness, not of J-space. The matched control for Δρ therefore matches **the Gram spectrum
of B** — the same number of atoms and the same distribution of pairwise overlaps — and not merely
its size; a control that matches size alone measures overcompleteness and is refused. The
construction is declared and sealed the way the tolerances are, before the number exists.

**Correction to the Stage B control, Chief, 2026-09-11 UTC (Codex, `3494a2c`, S1):** a matched
Gram spectrum does not match the signed overlap geometry — negating one atom keeps the spectrum
and the norms and changes the nonnegative cone. The control is a **seeded orthogonal rotation of
the whole dictionary**, preserving the full signed Gram matrix and atom norms; everything else
identical; the hypothesis stated; overcompleteness's own contribution measured under it.

## Next instruction, Chief, 2026-09-10 (09:55Z) — make the device lenses admissible and build the runner: four file-only tasks for Codex

From an audit of the bridge against the device artefacts (verified on the code before this instruction was
written): the fit writer (`fit_lens_f32.py`) and the chunk merger (`merge_chunks.py`) store maps as `J{repository
layer}` (1 … 33 on the 4B, 1 … 47 on the 12B) and write `nu.json` (the fitter) or promise it and do not write it
(the merger); the bridge loader `LensMaps.load` reads `J{block}` and adds one (`layer = int(name[1:]) + 1`, valid
only below `num_layers`), so a device archive loads shifted by one layer or refuses at its last map; the loader
also requires an identity block inside the archive or a digest-bound sidecar, which the device archives lack;
`hook_alignment` refuses a layer without a stored matrix although `LensMaps.apply` already treats the final layer
as the identity; and the reconstruction-budget helper measures `‖e‖/‖h‖` while A2's guard measures `‖Le‖/‖Lh‖`,
two different quantities under one name (a reproduction: 9.95% raw against 99.5% in score space). None of this
touches the fitted matrices; all of it is engineering between the artefacts and the bridge.

**T1 — an admission path for device lens archives.** A script `admit_device_lens.py` that takes a device fit
directory (`exact-maps.npz`, `nu.json`, `manifest.json`) and writes, beside the untouched originals, a loader-
form archive and its digest-bound identity sidecar: keys renumbered from repository layer to the loader's
convention, the identity carrying the base model, `num_layers`, the fit declaration (corpus, positions 16–126,
context 128, precision, forward and dimension batch, anchor batch, the chunk table for a merged archive) and
the source archive's sha256. It refuses an archive whose map count or shapes disagree with the declaration, a
sidecar naming a different file, and a declaration lacking the width fields. The merger gains the `nu.json` it
promises: every chunk's declaration and archive digest, the merged population (70/70/61) and the weights.
Tests: a fixture archive with three maps round-trips with the right layer assignment; the shifted-by-one and
wrong-identity fixtures refuse; loading the admitted 4B fixture through the real `LensMaps.load` and reading a
known residual through layer k gives the fit's matrix at repository layer k, not k+1.

**T2 — the final-layer identity endpoint.** `hook_alignment` accepts the final layer when the lens identity
declares its endpoint as the identity map, and still refuses any non-final layer without a stored matrix.
Tests: the 34th (48th) dictionary aligns; a missing interior map refuses by name.

**T3 — two reconstruction errors, two names.** `raw_reconstruction_share` (`‖e‖/‖h‖`) and
`lens_score_error_share` (`‖Le‖/‖Lh‖`) reported separately everywhere the budget appears; the budget gate
renamed for what it measures and never used as A2's admission mask; thresholds and the near-zero-denominator
rule declared in the config before any interpretation; the 9.95%/99.5% reproduction is a fixture that must
show "rankable by the raw share, refused by the score share". No default threshold may admit.

**T4 — a general device A1/A2 runner.** Explicit inputs only (model snapshot, dictionary directory, admitted
lens archive, capture directory, positions, output directory); before any numeric helper runs it checks the
checkpoint's hash manifest, the lens identity and archive sha256, the dictionary's digest receipts, the
capture manifest and the pairing registration, and refuses by name on any mismatch; a `--dry-run` that
performs every check on fixtures and stops; no hardcoded path. Its output carries the provenance block and the
interpretation limits of the second amendment verbatim.

Laptop only; tests on the CPU; no device job, download or model load; commit on each test's own exit code;
report by commit with the checked set named beside each verdict. **What stays with the lens owner (the Chief):**
the pairing registrations (4B fitted at dimension batch 32 and read at width 1; 12B at 16/16/8 and read at 1) with
their measured terms, and the domain-of-validity statement for readings beyond position 126; the shipped-token
example files for the chosen dictionaries, fetched to the card; and the runs (A1 through the new lenses with a
corrupted-layer control, A2 on the 4B captures, then the 12B). Stage B remains conditional on real A2 evidence
under the sealed rotation control above.

## Review of T1–T4, Chief, 2026-09-10 (~11:15Z) — `e5898e0`, `6ae9648`, `434bb55`, `c73f4e2` on `codex/sae-j-bridge-admission`

Read: `device_lens_admission.py`, the loader and bridge diffs, the runner's structure and its capture admission, the
handoff record. Executed on the laptop: the admission, runner, bridge, intervention and live-lens suites (157
collected, all passing; the large real-artefact test deselected as Codex did; worktree untouched). Executed on the
card's CPU against the real artefacts, from a copy of the branch's source under `/workspace/chief/saej-admit/`:

- **The real 4B fit admits and loads correctly.** `admit_device_lens` on `out/lens4b-f32` (source archive
  56c7b49e…, the one the 4B capture recorded) wrote an admitted archive (1c8d2bd7…) and its sidecar; the real
  `LensMaps.load` returns layers 1 … 33 of 34, every admitted matrix at repository layer k equal to the fit's
  `J{k}` (no shift), `apply` at 24 equal to `x·J24ᵀ`, and the final layer the identity — the T1 acceptance on
  the actual instrument, not a fixture.
- **The new merger reproduces the as-run 12B merge to the bit.** Re-merging c1/c2/c3 into a new directory gives an
  archive with sha256 e7942f1d6a73…, identical to `out/lens12b-f32/exact-maps.npz` in all 47 matrices, now with
  the `nu.json` the record promised (chunk declarations, populations 70/70/61, weights 70/201, 70/201, 61/201,
  widths 16/16/8 as ordered lists). That identity is what lets the 12B capture, whose `loaded` event recorded
  e7942f1d…, be paired with an admitted lens at all. The re-merged directory then admitted (rc 0) against the 12B
  checkpoint's hash manifest.

**One finding, provenance, before merge (F1).** The new merger is committed at the record's as-run script path,
`research/records/WORKSPACE-EXPERIMENTS-2026-09-10/scripts/merge_chunks.py`. On `cuda-migration` that path did not
exist; on main it holds the script that produced the 12B archive (5c39d626e129, in Provenance). A merge into main
would overwrite an as-run record script with its successor. Ruled: the new entry point moves to
`scripts/merge_device_lens_chunks.py` (the admission module already holds the logic), the record's script stays
as run, and the record's Provenance gains one line naming the successor and the bitwise identity above. Also
note, not a defect: the admitted artefacts of this review live in a scratch directory on the card; the canonical
admissions are to be produced by the merged code beside the originals (`out/lens4b-f32/admitted-maps.npz`,
`out/lens12b-f32-remerged/…`), and the record will cite those.

**Verdict.** T1–T4 accepted; after F1, merge in Codex's order (`e5898e0`, `6ae9648`, `434bb55`, `c73f4e2`) into
`cuda-migration` and then into main. T2's identity endpoint, T3's two named errors with an explicit budget and no
admitting default, and T4's admission-before-numerics with a dry run are as ordered. What the Chief now supplies,
in this order: the positions file for the 4B capture (cells at P_note and P_act, the capture-file hashes, the
corpus and tokenizer digests, the domain-of-validity statement for positions beyond 126 with the measured
band figures from W-5); the pairing registrations for the 4B (fit width 32, capture width 1) and the 12B
(16/16/8 against 1) with their measured terms; the shipped-token example files fetched to the card; then A1
through both admitted lenses with the corrupted-layer control, and A2 on the 4B capture, on the card after the
12B W-3b pass.

## T5, Chief, 2026-09-10 (~12:00Z) — the pairing-measurement tool, file-only for Codex; the runs are the Chief's

No tool in the tree measures a pairing; the bridge only validates and consumes registrations. A pairing is the
cross-path term the WS-D order defined: the same prompt run through the **fit's path** (float32, the forward
batch the lens was fitted at — dimension batch 32 for the 4B; 16, 16 and 8 for the 12B's chunks, each chunk's
term measured at its own width and the merged lens carrying all three) and through the **capture's path**
(float32, width 1, the capture's kernel and precision flags), and the relative displacement of the residual at
the read layer and position, `‖h_fit − h_capture‖ / ‖h_capture‖`, on the exact cell the reading will use.

`scripts/measure_pairings.py`, explicit inputs only: the model snapshot (its hash manifest checked against the
capture's `load_report_sha256`), the admitted lens archive and sidecar (for `lens_sha256`, `nu_sha256` and the
fit widths — read from the admission, never typed), the capture directory and the positions file (the cells,
already validated by the runner's rules: row, position, token_index, token_id, context_tokens), the layers to
measure, and an output path that must not exist. For each cell and layer it runs both paths on the card (one
prompt, two batch schedules; the width-1 path must reproduce the capture's stored residual to the bit or
refuse, which is the check that the capture path is the one being measured), and writes registrations in the
table's schema — `measured_pairings` keyed by repository layer, each entry `{pair, relative, basis}` with all
eleven pair fields (`fit_dtype`, `fit_width`, `capture_dtype`, `capture_width`, `lens_side`, `nu_sha256`,
`lens_sha256`, `positions`, `reduction`, `endpoint`, `context_tokens`) derived from the admission, the capture
manifest and the cell, and `basis` naming the measurement in words with the prompt's identity and both
schedules. One registration per cell per layer; the file carries provenance (checkpoint hashes, capture and
positions digests, torch and kernel flags, the source commit). Refusals by name: hash mismatch, a cell outside
the capture, a width-1 residual that does not reproduce the stored one, non-finite values, an existing output.
Tests on the CPU with a tiny model and a synthetic capture: the width-1 reproduction check passes on an intact
capture and refuses on a perturbed one; two schedules that differ give a nonzero term and identical schedules
give zero; every refusal exercised; the written table validates through `_supplied_pairing_table` and matches
its cells through the runner's `_cell_pairing`. Laptop only; commit on the tests' own exit code; report by commit
with the checked set named. The Chief runs it on the card for the 300-decision population at both positions
after the day's masking passes, registers the result, and states the domain of validity beside it.

*2026-09-10 UTC, ~12:05Z (card clock):* **the shipped-token example files are on the card**, fetched at the Director's authorisation through his own login by `lab-device fetch-dictionary --all-layers --examples` (the credential never handled): 34 files for the 4B suite and 48 for the 12B, every-layer `resid_post_all` 16k `l0_small`, each verified against the hub's declared sha256 for its exact path and recorded in the folder's `DIGEST.json` beside `config.json` and `params.safetensors` — 82 receipts, 82 carrying the examples file; ~67 GB, 127 GB left on the card. Item 8 of the audit is closed; the raw-first convention discriminator runs when A1 runs.

*2026-09-10 UTC, ~12:30Z (card clock):* **T5 delivered and accepted** (`5ce2635`, `SAE-J-PAIRINGS-2026-09-10`): `measure_pairings.py` admits every file before loading a model, replays the capture path with the capture's own forward (the outer LM, attentions per the sample, the two kept logits) and refuses unless the width-1 residual reproduces the stored bytes, runs the fit path at each admitted chunk width with replicated inputs and refuses non-identical replicas, retains every chunk term and registers their maximum as the scalar, binds each registration to its cell receipt in `_provenance.measurements` (eleven pair fields cannot distinguish two prompts of one length; the runner now requires the receipt), and writes provenance and named refusals; 323 tests on Codex's checked set, 206 on the Chief's laptop across the four suites. Unverified on purpose until the Chief runs it: real checkpoint execution and bitwise reproduction on the card. Next, the Chief's: the canonical admissions beside the originals, the positions file for the 300-decision population, the pairing runs, the domain statement, then A1 and A2.

*2026-09-10 UTC, 13:45Z (card clock):* **the Chief's runs, first pass** (`SAE-J-DEVICE-RUNS-2026-09-10`, e1307a8). The 4B pairing registration is measured: 600 cells × 33 layers, every cross-path term below 3.4 × 10⁻⁵ with medians at float32 rounding, the width-1 path reproducing the capture bit for bit on every cell; the record states the mathematics of the term and the domain-of-validity statement beside it. Both bridge configs are declared with a basis per threshold and both admission dry-runs passed, the 4B one matching all 600 registrations through `_cell_pairing`. **The 12B A1 at repository layer 24 is complete:** the shipped file applies the final gain (raw arm 0.134 recorded first, gain arm 0.981), two products agree to 1.9 × 10⁻⁶, the layer-3 control shares nothing. **The 4B A2 at layer 18 was refused by the identity check and the refusal was rounding:** the float32 sum over 16,384 terms missed the identity by 1.2 × 10⁻³ (median; 5.0 × 10⁻³ max) while float64 held it at 10⁻¹² on the same inputs, and the declared 1e-3 line measured that rounding against a net score of 0.12. Ruled and done by the Chief on main at `468756c`: `decompose_position` asserts the identity in float64 from the same float32 `h` and `z`, reports the float32 gap and the terms' absolute sum, and keeps a second guard that the float32 score path agrees with the float64 identity, so an orientation error is still refused (a test breaks the score path's orientation and sees the refusal; the three bridge suites pass, 104). **For Codex, file-only, whenever convenient:** review `468756c` against T3's rule that no default admits and T4's identity assertion — is the float64 assertion with the float32-path guard the right instrument, and is the reported `identity_gap_float32` enough for a reader to see what the float32 path did? The fixed A2 runs at 4B layers 18 and 24 run from a fresh checkout of `468756c`; results follow in the record. Ahead of them the diagnostic already shows layer 18's shape: raw reconstruction share 0.066 median, lens-score error share 0.63 median — T3's "rankable by the raw share, refused by the score share" case, on real activations.

*2026-09-10 UTC, 14:10Z (card clock):* **the 4B A1/A2 results, and a second correction to the identity check** (`SAE-J-DEVICE-RUNS-2026-09-10`, ed1d5fc). The first fix (`468756c`) kept a guard that the float32 score path agrees with the float64 identity, measured against the net score; on the device it refused the same cell again — a 3.8 × 10⁻³ rounding gap in one dot product of length 2,560 against a net score of 0.11. Corrected at `aa2daa4`: that guard is measured against the dot product's term scale ‖u‖‖W[v]‖, of which float32 rounding is about 1.5 × 10⁻⁴ and a wrong orientation an order-1/√d fraction; the gap and its scale are reported per cell. Codex's file-only review request covers both commits. **Results, both 4B layers from a fresh checkout of `aa2daa4`:** A1 passes every check at layers 18 and 24 (layer 18's convention numbers reproduce the laptop's, 0.245 / 0.995; layer 24's raw arm is above the line, so the discriminator is uninformative there and says so; two products to 4 × 10⁻⁶; the layer-2 control shares nothing). **A2 ranks nothing in any of the 600 cells at either layer:** the dictionaries reconstruct the residual to 6–14% in its own metric, but the unexplained part carries 59–71% of the lens-score norm — the readout amplifies the dictionary's residual 5–11× relative to the whole activation, in every cell. That is T3's "rankable by the raw share, refused by the score share", on real activations, and the record states what it does and does not license: no feature-level account of what the model is about to read out can be built from these dictionaries at these out-of-domain positions; whether the ratio falls in the dictionaries' own domain, with a wider or denser dictionary, or is a property of the residual-metric training objective, are three measurements not made. **Running:** the 12B pairing measurement on the GPU (600 cells × 47 layers, widths 16/16/8 against 1), its positions file carrying the 12B domain statement (licensed without qualification only at layers 46–47, where W-5 resolves every cell). **Next, after it lands:** the 12B A2 at repository layer 47 (dictionary `layer_46`), inside the statement, and the same two shares; then the in-domain control the record names first — the same dictionaries and lenses on prose prompts of the fit's own length, to learn whether the amplification is the positions' or the dictionaries'. That control needs a small capture of the fit corpus at width 1, which the capture tool already does; the Chief runs it on the card after the 12B pairings.

*2026-09-10 UTC, 17:00Z (card clock):* **the 12B pairings, the 12B A2 at layer 47, and the in-domain control** (`SAE-J-DEVICE-RUNS-2026-09-10`). The 12B registration: 28,200 cells (600 × 47 layers, each fit chunk's width measured), every term below 5.1 × 10⁻⁵. The 12B at repository layer 47, inside its domain statement at the action position: A1 passes (discriminator uninformative, both arms above the line; two products 1.2 × 10⁻⁶; control shares nothing); A2 amplification about 3×, and 90 of 300 action-position cells fall under the 0.5 score-share line and are ranked, unlabelled. **The in-domain control answers §3.3's first open question:** the same dictionaries, lens, budget and decomposition on the lens's own fit prompts (201 prose prompts, 128 tokens, five positions) give ratios of 6.6 at layer 18 and 4.4 at layer 24 — the same order as out of domain — so the amplification of the dictionary's residual by the readout is a property of the dictionary–readout pair, not of the agent-transcript positions; consistent with the published observation that SAE reconstruction errors are pathological for the next-token distribution. Recorded with its algebra and its limits. Not measured and not guessed: wider or denser dictionaries; a readout-metric dictionary. **For Codex, in the same file-only review:** the control script `indomain_control.py` — is its residual convention the capture's (block output at L, coherent float32, eager, width 1), and is its labelling as a control rather than an A2 result sufficient? **Next on the card:** the same control on the 12B at layers 24 and 47; then the W-2 12B record when its CPU pass ends.

*2026-09-10 UTC, 22:05Z (card clock):* **the dictionary width and sparsity control, launched overnight on Daniel's fetch authorisation.** What the hub ships at `resid_post_all`: 16k and 262k widths at `l0_small` and `l0_big` at every layer, nothing at 65k and nothing at `l0_medium`; the `resid_post` site carries the 65k ladder (small, medium, big) at 4B blocks 9, 17, 22, 29 and 12B blocks 12, 24, 31, 41. Fetched through the card's login, 44 GB, all seven receipts verified: 16k big with examples at 4B blocks 17 and 23 and 12B blocks 23 and 46; 262k small (no examples ship) at the same four; 65k small, medium and big with examples at 4B block 17. The driver `width_control_run.sh` runs, per dictionary, its own admission dry-run, the runner on the 600 out-of-domain cells (both stages where examples exist, A2 alone for 262k), and the in-domain control on the fit prompts; eleven runner passes and seven controls, from `lab-a2` at aa2daa4; results and the reading follow in `SAE-J-DEVICE-RUNS-2026-09-10`. **For Codex, small:** `lab-device fetch-dictionary` exits 0 when the hub has none of the requested files ("has no such files" printed, rc=0), so a driver counting exit codes reads absence as success; it should exit nonzero. Also for the record: the 12B W-2 is read (workspace record a4286dc; the artifact carries the 12B panel).

*2026-09-11 UTC, 00:05Z (card clock):* **a review of the A2 finding, and the model's answer to it** (`SAE-J-DEVICE-RUNS-2026-09-10/REVIEW-2026-09-11.md`, verbatim; README §4, cdc28a9). The review's point: the vocabulary-wide lens-score share is not a measurement of the decision, and "11× amplification" is against the activation, not against a random error of the same size. Its experiment 3 ran first, on the model: substituting the dictionary reconstruction at 32 action positions, 4B layers 18 and 24, **changes no decision** (0 of 64 argmax flips; margins 16 → 14 and 12 logits; KL 10⁻⁷–10⁻⁵, the order random errors of the same norm produce), though at layer 24 it perturbs about nine times more than random with a long tail; the exact derivative along the error predicts the top-10 gap changes (correlation 0.83) and the averaged lens does not (0.24). Ruled: the A2 refusals stand as verdicts of the declared budget; the reading "the readout-relevant part of the activation lives in the dictionary's error" is withdrawn as a statement about the decision; a decision-focused fidelity measure is added beside the budget, never in place of it, once the saved-array diagnostics (experiments 1, 2, 4, running) fix its form; the fine-tuning remedy waits on those. **For Codex, file-only:** `substitution_test.py` and `a2_diagnostics.py` in the record — their hook convention (block output at L, position P, batch element 0), the angle-matched control, the RMS scaling of the lens prediction, the inlined NNLS; and the review's six experiments as an implementation queue for a decision-focused measure in `sae_bridge.decompose_position` (a declared token set, centred scores, the gap changes and the margin) — specify before implementing.

*2026-09-11 UTC, 01:30Z (card clock):* **the review's experiments, answered on the model and on the arrays** (`SAE-J-DEVICE-RUNS-2026-09-10` §4, through 593ba79; the second review's three corrections applied in `substitution_test_v2.py`). (3) Substitution, corrected: on decisions with a margin nothing moves at either layer; on the 32 closest decisions the reconstruction at layer 18 changes 9 of 32 next tokens — and, in the complete-call test, 9 of 32 whole calls, five with the path — against 15 of 128 random and 13 of 128 angle-matched errors of the same size; at layer 24 it changes none against 16 of 128 random. The derivative's error scales with strength (0.15/0.3/0.6 at ¼/½/1: second-order terms at the reconstruction's size); the lens map does not propagate the perturbation even with the readout's own normalisation responding (relative error above one at every strength; pre-norm cosine −0.04 at 18, 0.2–0.5 at 24). (2) A random error of the dictionary's norm has a lens-score share of 1.2–1.7 against the dictionary's 0.6–0.7, and the actual error ranks lowest of sixteen matched controls in every one of the 600 cells at both layers: the "amplification" was the gain-only vocabulary readout's sensitivity to any error, and the dictionary's error lies in its less sensitive half. (1) Centring changes the share by a hundredth; lens-score space exaggerates (the lens-score argmax survives in 203 of 300 at 18 and 105 of 300 at 24 where the model's survives everywhere with a margin). (4) NNLS on the same atoms gains half a point of raw share, greedy re-selection one point: the limit is what twenty atoms express, not the encoder. **Rulings.** The A2 budget's vocabulary-wide lens-score share is retained as a declared refusal line and its verdicts stand, but it is renamed in the record for what it measures; a decision-focused fidelity measure joins it: at a declared token set (the six tool first-tokens plus the model's top-10 at the cell), the logit-scale gap changes and the argmax margin under substitution **on the model**, with matched random and angle-matched controls, as `substitution_test_v2.py` computes them — the lens is not used to predict perturbation effects until a map that does so is fitted and shown to. The fine-tuning remedy stays behind the width control. **For Codex, file-only:** `substitution_test_v2.py`, `call_test.py`, `a2_diagnostics.py` (with its inlined NNLS) — conventions and controls; then a specification, not an implementation, for the decision-focused measure inside the bridge.
