# SAE-to-J-lens bridge, stage A: the exact score decomposition

**Engineer B, 2026-09-09**, against `design_specifications/pending/SAE-J-BRIDGE-ORDER-2026-09-08.md`
and both its amendments. Stage A only. Stage B is not built, on the order's own condition; the
causal-abstraction programme is not built, on the order's own arithmetic.

Everything here ran on CPU on one laptop. **No model was loaded and no model forward was run.** The
three matrices were read from disk and multiplied. The runs that took time were announced as
windows with a projected peak and its basis, and the box state at the time of each run is in its
artefact.

## The identity, and the one convention it forced

For a residual layer with lens map `J`, readout `W`, final-norm gain `g`, and decoder `D` (one unit
direction per row):

    L x  =  W ((J x) ⊙ g)                                 the lens score map, linear
    L h  =  L b  +  Σ_i z_i (L d_i)  +  L e               exact, for every h = b + D^T z + e

Feature `i`'s signed contribution to the score on token `v` is `z_i (L d_i)_v`; everything the
dictionary did not explain sits in `L e`. The code computes each term and **asserts the identity on
the emitted token before it reports anything**, so a wrong orientation cannot produce a ranked
table.

**The gain, not the full norm — stated because it is the only place a choice was made.** Gemma's
final RMSNorm is `x · rsqrt(mean(x²) + ε) · (1 + w)`. The `rsqrt` factor is a positive scalar *per
input vector*: it moves no ranking within a vector, and it makes the map nonlinear, which would stop
the decomposition being an identity. So `L` applies only the gain `g = 1 + w`, the scalar is
declared dropped, and **score values here are not logits**. The second amendment's proof that
full-norm and gain-only give identical top-k (max deviation `9.2e-14` at the real width) is what
licenses this for ranks and overlaps.

## Inputs, each identified rather than named

| input | what | identity |
|---|---|---|
| dictionary | Gemma Scope 2, `resid_post_all/layer_17_width_16k_l0_small`, JumpReLU, L0 20 | `params.safetensors` sha256 `507810c3…2671e`, **verified against the hub's LFS digest, never by length** — the amendment's trap is real, `l0_small` and `l0_big` are byte-identical in size |
| lens | hosted Jacobian lens, `models/jlens/gemma-3-4b-it_jacobian_lens.npz`, via `LensMaps.load` | `npz_sha256 a14ab264…3193`, identity `google/gemma-3-4b-it`, stored float16 |
| readout | `models/gemma-3-4b-it-bf16`, `embed_tokens.weight` (tied, no `lm_head`) and the final `norm.weight` | read by tensor name from the safetensors index; no model class imported |
| shipped readout | the dictionary's own `examples.safetensors`: `top_tokens`, `top_logits`, `bottom_*`, `feature_frequencies` | read by **HTTP range**, 2.2 MB of an 815 MB file, schema recorded beside it |

Three facts read off the dictionary rather than assumed: every threshold is strictly positive
(minimum 63.7), so the JumpReLU is unambiguous; every decoder row has unit norm to four places, so
`|L d_i|` is comparable across features; and the encoder is **not** the decoder's transpose (mean
cosine 0.63), which is why a fixture for A2 needed a tied decoder to make encode-then-decode exact.

**The readout is bfloat16 on disk, and that broke the first real run.** NumPy has no bfloat16, so
the convenience reader that hands NumPy arrays back cannot read the embedding at all. Every fixture
had written float32, so the loader path the real artefact takes was untested by construction — the
fixture rule from the WS-C record, broken by its own author within the day. The module now reads the
safetensors header and bytes itself and widens bfloat16 by bit shift, which is exact; the fixture
writes bfloat16 bytes by hand, and a test checks the widening against known bit patterns.

**Hook alignment is a test, not a comment.** The dictionary names its hook as the output of block
17. This repository's capture convention numbers that residual as layer 18 and the lens archive
keys it as `J17`. `layer_for_hook` is the one place that arithmetic lives, and a gated test loads
the real dictionary, lens and checkpoint and asserts the map the loader hands back at the hooked
layer is byte-equal to the archive's `J17`. It passes.

## What is never materialised

`L D^T` is vocabulary by width — 17.2 GB at this model — and is never formed. `U = D J^T` is formed
once (width by hidden, 0.17 GB) and `U W^T` is taken a chunk of features at a time, keeping only
each feature's top-k. **A test enforces this at the matmul**, not by reading the code: an `ndarray`
subclass standing in for the readout records the row count of everything that multiplies it,
asserts it saw at least one product so it cannot pass vacuously, and asserts none had the full
width.

## A1 on the real matrices, block 17 → layer 18

Source commit `6f605f8`, script unmodified at run time. Run under this seat's own window, model lock
free, console block-buffered through the redirect so it appeared at exit.

**The machine behind the timing, measured rather than assumed.** A 12-core Apple M4 Pro, NumPy
2.5.2 bound to Apple Accelerate. A 4096² float32 matmul ran at **2.36 TFLOP/s** with a CPU-to-wall
ratio of 2.7, so the AMX units are doing the work and thread count is not the right lens on it.

| stage | time | memory |
|---|---:|---:|
| load: lens (33 maps), dictionary, readout widened from bfloat16 | 3.1 s | 4.61 GiB |
| raw arm, all 16,384 features | 37 s | |
| gain arm, all 16,384 features | 38 s | |
| **A1 through the lens, 16,384 features × top-10** | **39 s** | 4.77 GiB |
| total, with controls and the two-ways check | 131.7 s | **peak 5.27 GiB** |

The order's cost table put one layer at "about 150 seconds on CPU BLAS"; one full pass here is
**39 seconds**, 3.8× faster, on a different BLAS. The window declared a projected peak of about
5 GiB with its basis (readout 2.7 GB float32, lens maps 0.9 GB, dictionary 0.3 GB, chunk-256 score
block with its int64 argpartition 0.8 GB); the measured peak was 5.27 GiB, so the declaration held.

### The second amendment's discriminator, raw arm first

Two arms against the dictionary's shipped `top_tokens`, the raw arm run and recorded **before** the
gain arm was looked at, so the first number was not chosen after seeing which convention agrees.

**Declared before the arms were run.** The gain `1 + w` read from the checkpoint has mean +10.99,
standard deviation 7.52, and runs from −0.06 to +54.50; the ratio of its 99th to 1st percentile in
magnitude is 8×. It is strongly non-uniform, so the expectation written first was: the two arms
should rank differently, and exactly one should match the shipped tokens.

| arm | overlap@10 with shipped | |
|---|---:|---|
| raw `top_k(W d_i)` | **0.245** | recorded first |
| gain `top_k(W (d_i ⊙ g))` | **0.995** | |

**The finding is about the shipped file, not the model**: the shipped `top_tokens` in
`examples.safetensors` were made through the final gain — raw arm 0.245, gain arm 0.995 — and so,
by the amendment's table, the convention is settled and the bridge proceeds. This is the
measurement Codex's caveat asked for, and it says what that one artefact did; it says nothing
about what any other suite's file did, which is measured the same way when it arrives. The
prediction held: one arm matched and one did not. The 0.5% of features where even the gain arm
disagrees is the near-tie rate one expects at the tenth position of a 262k-token vocabulary under
two float32 paths, and not a signal.

### A1 through the lens

All 16,384 features, top-10, in 39 seconds; `a1/layer18_top10.npz`. A sanity read the acceptance
list does not name, because it is the pathology a wrong gain or a wrong orientation shows at scale:
a single token dominating every feature's top-1. It does not. 5,746 distinct top-1 tokens across
16,384 features; the most common top-1 (`</strong>`) is 4.3% of features; the five most frequent
tokens in any top-10 — all HTML closing tags — cover 4.5% of slots.

### Two ways, to float32

Eight named features (61, 403, 1452, 1694, 7616, 9546, 13022, 16363), through the composed path and
by a direct product each: **top-10 sets identical, zero order differences, worst relative score gap
1.71e-06.** Float32.

### The negative control, biting and shown to fail

| against layer | mean overlap@10 | identical | distinguishes |
|---:|---:|---:|---|
| 2 | 0.047 | 0.000 | yes |
| 10 | 0.231 | 0.004 | yes |
| 30 | 0.152 | 0.000 | yes |
| **18 (itself)** | **1.000** | **1.000** | **no** |

The control bites at every layer tried, and the same layer twice is reported as indistinguishable —
**the check shown to fail under a deliberate mismatch**, which is what a control must do or it is
not one.

**And my prediction of the ordering was wrong, twice, which is recorded because I wrote in advance
that a different order would be the finding.** Before the numbers, from each map's distance to the
identity (layer 2 farthest, layer 30 nearest), I declared: least overlap with 2, most with 30, 10
between. Observed: 2 < **30** < **10**. After the numbers I tried the statistic I should have used —
each map's distance to layer 18's — and it fails the same way: by that measure layer 30's map is the
*closest* to layer 18's (3.1 against 6.5 for layer 10, in the sidecar's units), yet layer 10 shares
more of the readout.

| layer | ‖J − I‖/√d | ‖J − J₁₈‖/√d | cos(J, J₁₈) | observed overlap |
|---:|---:|---:|---:|---:|
| 2 | 21.0 | 20.8 | 0.139 | 0.047 |
| 10 | 7.1 | 6.5 | 0.417 | **0.231** |
| 30 | 1.6 | 3.1 | 0.432 | 0.152 |

Neither aggregate predicts a per-feature top-10 statistic, and I am not offering a third model
after two misses. This is the stream's own rule turned on its author: a global norm is a quantity
downstream of the mechanism, and the top-k set is the mechanism. The honest next step was to
measure the curve rather than guess it.

### The control curve, measured at every layer — exploratory, chosen after the controls were seen

This sweep was decided on after the three pre-registered controls had been read and my ordering
of them had failed. It is therefore **not a pre-registered control** and is not offered as one:
it is the exploratory measurement that replaced the model I would otherwise have fitted, and the
pre-registered prediction and its failure above stand as recorded, before it. The raw-arm-first
rule applies here as it did to the overlap: the declaration comes first, the sweep second.

`control_sweep.py`, the same 512-feature draw as the A1 controls (the six named features first,
then 512 under `default_rng(20260909)`), read through every one of the 33 lens maps and once with
no lens at all; `a1/control_sweep.json`, stamped with the script's own sha256 so the artefact names
the file that produced it. 34 readouts in 78 seconds. The three A1 points reproduce to the third
decimal.

| layer | overlap@10 | | layer | overlap@10 | | layer | overlap@10 |
|---:|---:|---|---:|---:|---|---:|---:|
| 1 | 0.039 | | 12 | 0.384 | | 23 | 0.572 |
| 2 | 0.047 | | 13 | 0.393 | | 24 | 0.537 |
| 3 | 0.046 | | 14 | 0.428 | | 25 | 0.517 |
| 4 | 0.057 | | 15 | 0.405 | | 26 | 0.485 |
| 5 | 0.055 | | 16 | 0.615 | | 27 | 0.443 |
| 6 | 0.073 | | 17 | 0.713 | | 28 | 0.364 |
| 7 | 0.127 | | **18** | **1.000** | | 29 | 0.360 |
| 8 | 0.153 | | 19 | 0.808 | | 30 | 0.152 |
| 9 | 0.220 | | 20 | 0.728 | | 31 | 0.170 |
| 10 | 0.231 | | 21 | 0.655 | | 32 | 0.159 |
| 11 | 0.343 | | 22 | 0.629 | | 33 | 0.125 |
| | | | | | | none | 0.106 |

Three things the curve says that neither aggregate did.

**It is single-peaked at the hooked layer and asymmetric.** Overlap rises monotonically from layer
1 to the peak at 18, apart from one dip at 15, and falls monotonically after it. The rise is slow
(seventeen layers to climb from 0.04 to 0.71) and the fall is fast (twelve layers to drop from 0.81
to 0.15). Adjacent layers share most of the readout — 0.71 below and 0.81 above — and nothing
further than three layers away shares half of it.

**Why layer 10 beat layer 30.** Layer 10 sits on the slow rise, where the maps are still moving
toward layer 18's in the directions the readout cares about; layer 30 sits on the fast fall, where
the last few maps approach the identity from a different direction. The `none` row is the tell:
reading the decoder direction straight through the gain, with no map at all, shares 0.106 of the
readout with layer 18 — and layer 30's map, the third-closest to the identity of the 33, shares
0.152, barely more than no map. Late maps are near the identity, and near the identity is far from
layer 18's readout. The Frobenius distances said layer 30 was "closest to 18" because they measure
a whole matrix; the top-10 set measures the few directions that reach the vocabulary, and the
identity is not among them.

**The negative control is therefore a graded instrument, not a binary one.** A control taken from
a neighbour would be a weak control (0.7–0.8 shared); a control taken from twelve or more layers
away in either direction is a strong one (under 0.25). The A1 choice of 2, 10 and 30 was, by luck
rather than design, one strong control on each side and one weak-to-moderate one in the middle,
which is a reasonable set — but the next stage should choose controls from this curve rather than
from a distance to the identity, and I have written that in the record so the mistake is not
repeated with a new justification.


### A sample of the readout, headed as ours

These are **our** top tokens read from layer 18 through the lens. They are not published labels and
no artefact presents them as one.

| feature | top tokens through layer 18 (ours) |
|---:|---|
| 1452 | ` incorrectly`, ` erroneously`, ` mistakenly`, ` accidentally`, ` incorrect`, ` confusing`, ` annoying`, `!).`, ` inaccur`, ` unnecessarily` |
| 7616 | ` \...`, ` $\$`, `$\$`, ` \&`, ` \$`, ` ...`, ` (\$`, `\$`, `\%`, `\"` |
| 9546 | `</h4>`, `﻿…`, `.﻿`, `<sup>`, `</strong>`, `</h2>`, `</th>`, `◆`, `.•`, `▲` |
| 13022 | ` ‘`, ` «`, `‘`, `’).`, `«`, `Df`, `’)`, `«.`, `)’`, ` ।` |
| 16363 | `</em>`, ` \...`, `»)`, `…</`, `»).`, `[…]`, `…)`, `\...`, `»),`, ` […]` |
| 61 | ` científicos`, `<start_of_image>`, ` estadounidense`, ` doenças`, ` росій`, ` biotechnology`, ` médicos`, ` biochemical`, `欧洲`, ` étn` |

Feature 1452 reads as one thing; 7616, 9546, 13022 and 16363 read as formatting families; 61 reads
as multilingual science-and-medicine vocabulary. That is as far as a token list licenses reading.

## Handing the dictionary to the decoder intervention

Codex's `sae_intervention.SAEIntervention` (on `cuda-migration` at e8dbcc3) takes the decoder as
`[residual, feature]`, the bias separately, and an encoder callable that returns a vector in the
residual's dtype and device. `intervention_parts(dictionary, dtype=…, device=…)` produces exactly
those three from a loaded dictionary: the decoder is the stored `w_dec` transposed, the bias is
`b_dec` never folded in, and the encoder computes in the **declared** dtype and casts its code to
the residual's. The declaration is returned as a dict for the record. A test constructs the real
wrapper from the fixture dictionary, checks the encoder agrees with the numpy reference encode,
that a bfloat16 residual gets a bfloat16 code, and that the edit lands on the chosen feature. The
wrapper preserves the base reconstruction error and reports re-encoding as numbers; nothing here
claims attainment.

## Labels: unavailable, and why, verbatim

The label field of every artefact is `unlabelled`, with the second amendment's reason recorded
word for word:

> no Neuronpedia source maps to `resid_post_all`; the residual labels index
> `resid_post/layer_17_width_16k_l0_medium`, a different training run at a sparsity this suite
> never published.

A description this bridge computes is headed as ours and names the layer it was read from. The
distinction is the Director's rule and survives even a perfect mapping.

## A2: built, tested, and unexecuted on real activations by construction

`decompose_position` computes every term at one position, asserts the identity, reports
`|L e| / |L h|`, and **refuses to rank features when the residual dominates** (default: share above
one half), because the decomposition is then describing the dictionary's failure and not the model.
All of that is tested on a fixture where the answer is known.

It has **not** run on real activations, and cannot from this checkout: **no Gemma 3 residual
vectors exist on disk.** The map record stored ranks, not residuals; the pilot captures on disk are
Qwen's; the ARM-A residuals are Qwen's. Producing one needs a forward pass, which is a model load,
which is the device's. What it needs, so the first device session can run it, stated in the
capture's own contract: `capture_residuals(view, token_ids, [18], positions="all")[18]`, which is
float32 `(T, hidden)` at repository layer 18 — the residual after block 17, which is the dictionary's
hook — plus the emitted token id at each position. `decompose_position` takes one row of that and
one id.

## What this may and may not be said to show

It is a measurement instrument. A large feature contribution to a lens score is a contribution to
that score, at that layer, under this averaging convention and this gain-only linearisation. It is
not a token probability, not a causal derivative, and not evidence of a workspace, a maintained
state, or flexible access. The order is careful about this and the artefacts inherit its care.

## The suite, with its skips named

Full suite on this branch after merging `cuda-migration` at e8dbcc3 (Codex's intervention wrapper
and the WS-D reference), box free, no model loaded: **2449 passed, 10 skipped, 0 failed**
in 152 s wall. The count is read from the progress marks because the repository's `-q` and mine
stacked to `-qq`, which suppresses the summary line; the ten skips sum to the same ten. All are
absent-data gates on this checkout, none in the bridge, and each names its reason:

| where | n | reason |
|---|---:|---|
| `tests/test_integrity.py:823` | 1 | saved B/C evaluations are not available |
| `tests/test_patch.py:703` | 1 | protected saved evaluations not present on this checkout |
| `tests/test_patch.py:2348` | 1 | protected saved evaluations not present on this checkout |
| `tests/test_patch.py:3605` | 1 | protected saved evaluations not present on this checkout |
| `tests/test_power.py:445` | 1 | run artifacts are untracked; the report carries their paths and fields instead |
| `tests/test_power.py:497` | 1 | run artifacts are untracked; the report carries their paths and fields instead |
| `tests/test_probes.py:1829` | 1 | the base capture has not been made |
| `tests/test_tasks.py:389` | 2 | configured protected replay directory is absent: /Users/daniel.tipton/worktrees/cuda-ws-c/data/chat_replay |
| `tests/test_tasks.py:417` | 1 | run D dataset is not present on this checkout: /Users/daniel.tipton/worktrees/cuda-ws-c/data/agent_v2d |

## Unexecuted

A2 on any real activation. Stage B entirely. Every layer other than 18 for the full readout — the
readout is 39 seconds per layer and the order says take the layers the map makes interesting, not
the set. The `!`-slot test recorded in the WS-C record, which is a different instrument and a
different stream.
