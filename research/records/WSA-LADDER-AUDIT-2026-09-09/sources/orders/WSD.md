# WS-D, for the D-CRO: un-port the lens to upstream, then extend it

**From the Chief, 2026-09-09.** Read the plan §2, §5 WS-D, §6, §11. The Director's instruction is to
**refine, extend, optimise and make upstream higher resolution**, using its code directly where we
can and resolving a confound wherever the implementation lets us. The stage-two analyses are closed;
WS-D begins when torch is installed and not before, on CPU float32, with your own rule in force —
nothing unrun is handed over as run.

## Use upstream directly, modified

| upstream | use |
|---|---|
| `jlens/fitting.py::jacobian_for_prompt` | **called**, through an ~80-line adapter that feeds it our corpus rows and writes our artefact (npz + sidecar + R57 identity + declared ν) |
| `jlens/fitting.py::valid_position_mask`, `SKIP_FIRST_N_POSITIONS` | generalised to a **cotangent selector** (WS-A builds the graph-once estimator with it; you use it) |
| `jlens/lens.py::JacobianLens.save/load/merge` | the in-memory lens object; our converter already maps `.pt` to npz; `merge` is the reduction for sharded fits |
| `jlens/hooks.py::ActivationRecorder` | via WS-A's `TorchCapture`; the sub-block hooks of §6.5 extend it |

**Deleted from `jacobian.py`:** the finite-difference machinery — `unit_directions`,
`finite_difference_steps`, `reference_responses`, `cached_responses`, `full_jacobian`, `self_check`,
`copy_cache`, `pre_norm_tail`, `PositionState`, `prepare_position`, `_directions`, ~450 lines.
**Kept:** `make_plan`/`freeze_plan`/`read_plan`, `Convergence`, `memory_gate`/`time_gate`/
`WorkloadMemoryGuard` (extended from *stop* to *choose*, plan §10.2), `benchmark`, `write_record`,
`run_jacobian_stage`. `regression.py` ports line for line (§3.7).

## What you build, in order

1. **The adapter and the un-port golden test.** Refit the hosted recipe — WikiText-103 train,
   128 tokens, bf16 on the remote, fp32 on CPU with fewer prompts — and compare per layer against
   `neuronpedia/jacobian-lens` with your own `compare_maps.py` (now on the branch). Agreement
   validates the un-port; **the residual is the finite-difference-versus-exact estimator
   difference, measured for the first time**, and it retroactively bounds every MLX-fitted lens.
2. **§6.1 declared ν.** Endpoint, position weighting, pair weighting, corpus, precision, estimator,
   in the sidecar; the hosted lens's sidecar gains upstream's defaults retroactively.
3. **§6.6 the second moment**, as you verified it: one `einsum('bpi,bpj->ij', g, g)` per pass,
   within-prompt and between-prompt terms **reported separately**, cost and +0.81 GiB in the §10.2
   table.
4. **§6.2 position bands** on the transcript corpus (Codex's task-1 recorder and corpus builder,
   already on their worktree, are the input; the 2,816-token coverage gate is theirs). This is what
   makes amendment 9 computable and the window secondary interpretable.
5. **§6.4 span-conditioned lenses** — the same selector, a different mask, from the labels the
   recorder emits.
6. **§6.5 sub-block and multi-target** — hooks on `self_attn` for 68 read points; `target_layer`
   looped for `J_{L→M}`.

## Confounds this stream resolves

The 21.8x position extrapolation on every published map number (bands); the prose-domain bias
toward notes (spans); the averaging cancellation the derivation warns of (second moment); the
23-to-24 commitment's attribution to attention or MLP (sub-block); the estimator difference nobody
has measured (the un-port test); and the ν that yesterday's hosted-versus-fitted comparison was made
without.

## Golden tests

Step 1's per-layer comparison; for each extension, the **canonical ν recovers the unconditioned lens
as the special case** — a band selector covering all positions reproduces the plain fit to float32
tolerance, a span selector over all spans likewise. That is the regression test that keeps
"higher resolution" from becoming "different instrument".

## Budget

~80 adapter, ~45 regression edits, ~400 extensions; ~450 deleted.

---

## Corrections from the survey (plan §13), which supersede anything above they contradict

Read plan §13 in full. The items below are the ones that change this order.
- **The un-port changes ν and must declare it.** Our MLX fit uses the same-position reduction;
  upstream sums over targets. The adapter supports **both** through the selector, the sidecar
  records which, and `profiles.py` keeps `jacobian` distinct from `hosted-jacobian`. A lens fitted
  under upstream's default is not the same estimator as ours; the un-port golden test compares
  like with like by running upstream's ν.
- **Finite differences versus exact autograd is Q3, the Director's**, decided by the pre-registered
  c-sweep: no plateau means instrument-limited and autograd; a plateau keeps the estimator. The
  saving is ~15.5x from the layer loop, not more.
- The residual dtype your regression fit consumes is **Q5**; `residual_source` in the artefact must
  name it either way.


## The Research Division's answers (plan §14) supersede the above where they conflict

Read plan §14 and `CUDA-MIGRATION-RESEARCH-BRIEF-ANSWERS-2026-09-09.md` in full.
- **Upstream fits at 128 tokens by default** and the published lenses were fitted so. Your golden
  test at 128 tokens is like-for-like. §6.2's transcript-length bands are a **different estimator by
  construction** and are recorded as a declared departure with their own memory model, never
  compared to the hosted lens as if they were one. The position selector exists only at readout
  upstream; the fit-time selector is ours.
- **`attn_implementation="eager"`** on the fitting model for determinism only; that batched rows
  regress under `sdpa` was a warm-up artefact, withdrawn (plan §14.2, `c0e4233`).
- Upstream has one commit and is unmaintained; there is nothing to track and no runner to diff.

## Upstream defaults, verified against the clone rather than reported

**Checked at `581d398` before any of this reached code, because three of them are load-bearing for
the adapter and one was not in anyone's message.**

| claim | verified |
|---|---|
| `fit(max_seq_len=128, skip_first=16)` | **yes** — both are defaults of `fit` *and* of `jacobian_for_prompt` |
| no fit-time position selector | **yes** — `jacobian_for_prompt` takes only `skip_first`; there is no `positions` parameter anywhere in the fit path |
| `JacobianLens.apply(positions=)` exists at readout | **yes** |
| upstream sets `attn_implementation` | **no** — it appears nowhere in `jlens/`, so the fitting model inherits transformers' default |

So 128 tokens is the reference estimator's **definition**, not a Neuronpedia choice; the fit-time
selector §6.2 needs is genuinely ours to add; and eager attention must be set by us because upstream
never mentions it.

**A fourth default, in the same class, that follows from reading the two signatures together.**
`HFLensModel.encode` defaults to `max_length=512`, and `JacobianLens.apply` defaults to
`max_seq_len=512`, while `jacobian_for_prompt` passes its own `max_seq_len=128` down into `encode`.
**So upstream fits at 128 and reads out at 512 by default.** Nothing in either signature warns of it;
a caller who fits with `fit(...)` and reads with `apply(...)`, both at their defaults, is applying a
lens four times outside the length it was fitted at and will see no error.

That is our own 21.8x extrapolation arriving from upstream's defaults rather than from our corpus,
and it is a trap for anyone told to "use upstream directly". The adapter passes lengths explicitly
at both ends and never relies on either default.

## Amendment, Chief, 2026-09-10 — row one of the golden test, and the float32 control that comes before rows two and three

Read from `out/golden-4b/golden-report.json` on the card at 10:57Z, the D-CRO's run under
`/workspace/wsd-checkout`, held row 39, one prompt of 128 tokens, two valid positions, native
bfloat16, ε at 1% of the residual norm per layer (101 at layer 0, 8,818 at layer 32). Exact
reproduces itself at 0.0. Finite-difference against exact, per layer:

| layer | relative difference | cosine |
|---|---:|---:|
| 1 | 1.014 | 0.015 |
| 17 | 0.973 | 0.295 |
| 33 | 0.569 | 0.825 |

monotone in between; the three controls separate (transposed 1.414, wrong corpus 1.466,
layer-shifted 238); halving ε at layer 17 moves 0.973 to 0.844, a ratio of 1.15 against the
laptop's 3.9. A relative difference of 1.0 with cosine 0 is not a wrong map of the right size — that
is √2, which is what the transposed control shows — it is a map with about a fifth of the reference's
norm and no direction in common. The finite-difference estimator saw almost nothing at the early
layers and progressively more as ε grew with depth.

**Reading, as a hypothesis to test and not a finding:** the output change per direction sits below
the bfloat16 resolution of the logits at early layers and rounds to zero. The laptop's halving
ratio of 3.9 is the truncation regime, which says the laptop's FD was effectively higher precision.

**Schedule amended.** Rows two and three do not run yet; in bfloat16 they would spend 52 minutes
re-measuring row one. First, about three minutes: the FD estimator on the same row and positions
with the perturbation and the readout in float32 (weights in float32 if no promoted path exists),
at layers 1, 17 and 33, same ε scale. If it agrees with exact at the fixture's order (3.6e-3), the
finding is *the finite-difference estimator is precision-bound — unusable in native bfloat16, sound
in float32* — rows two and three are unnecessary, and the 12B smoke row follows. If it still
disagrees, the difference is not precision and rows two and three run as ordered, because the
finding then needs the extension.

**The record states, as measurements:** the dtype the laptop's FD ran in, which decides what this
says about the MLX-fitted lenses; why row 39 has two valid positions of 128, since a per-row
Jacobian over two positions is a thin basis and §6.2 depends on the mask; and the fraction of FD
directions whose response was exactly zero per layer, the direct test of the rounding reading.

## Correction to the amendment above, Chief, 2026-09-10 — the rounding reading is refused by the artefact; the cause is unsettled; the float32-only control is withdrawn

The D-CRO's record at `b826558` §8, from the artefact and the repository with no card time:

- **The laptop's finite-difference estimator ran in float32** (`jacobian.py` promotes before it
  perturbs) on the 4-bit checkpoint. Row one ran native bfloat16. They are different estimators
  under the arithmetic-path definition, and **row one is not evidence about the MLX-fitted lenses**.
- **The two positions of 128 are the order's declaration** (index 8 and the row's last position),
  handed identically to both estimators; the mask never ran.
- **No response rounded to zero: 0 exactly-zero columns of 84,480.** The map is systematically too
  small instead — ‖FD‖/‖exact‖ 0.185 at layer 1, 0.320 at 13, 0.539 at 21, 0.756 at 33 — which is
  the signature of a saturating response to a displacement outside the linear neighbourhood, the
  Director's reading, not mine. At layer 0 the step of 101.6 moves one coordinate 5.7 times its own
  size (typical magnitude 17.8) while the vector's norm moves 11%.

The step rule explains why the fixture passed and the model did not without invoking precision:
1% of the whole sequence's Frobenius norm on one coordinate displaces that coordinate by
0.01·√(L·d) of its root-mean-square, which is 5.7 at 128 × 2,560 and well under one on a small
fixture. The mechanism in my amendment was wrong; the algebra (a map with a fifth of the norm and
no shared direction) was right. **The control I ordered — float32 finite differences against the
bfloat16 exact map — is withdrawn**: it would confound the estimator with the forward path. The
separating design is a matched-precision pair crossed with the step rule (global 1% of the
sequence norm against a per-coordinate- or per-position-relative step), at one shallow and one
deep layer, the exact side at float32 preceded by a memory smoke since it peaked at 43.9 GiB in
bfloat16. The Director is consulting on that design; nothing runs on the card until it is ruled.
Rows two and three are withdrawn: three rows of a step-limited estimate average nothing.

## Ruling, Chief, 2026-09-10 — the Director's decision: the matched whole-float32 comparison and step sweep, under the consulting mathematician's protocol

**The Director, 2026-09-10:** *proceed with the matched whole-float32 comparison and step sweep; the
consultation is complete; the cause is still unresolved.* The evidence supports two possibly
simultaneous problems — small steps amplify rounding error, and the original step may be too large
to measure a local derivative — and the decisive experiment holds the layer, position, direction
and forward schedule fixed and varies the step with both estimators in float32, preserving
individual responses before averaging. Another full bfloat16 map cannot separate them.

**The order is the protocol**, `research/records/FD-NUMERICS-2026-09-09/RESOLUTION-PROTOCOL.md`
at 558d424 (Codex, on the Director's instruction; merged into `cuda-migration` with this ruling),
§1–§6, executed by the D-CRO under their seat alone. In the order the protocol gives: freeze the
function (row 39, 128 tokens, positions {8, 127}, endpoints the decoder-block outputs before the
final norm, every (source, target) contribution preserved before reduction); verify the
intervention boundary under the actual forward schedule before any derivative is read, with the
graph-once-versus-sequential VJP cross-check, the source-equals-target identity control and the
sign control; the matched arithmetic triangle on a coherent float32 model — the same stored bf16
weight values cast to float32, autocast off, TF32 off, highest matmul precision, eager attention,
determinism pinned, all stamped into the manifest — with the native bf16 pair retained and the
anchored bf16-AD-against-float32-AD control recording its anchor; directional checks at repository
layers 1, 17 and 33 on a small pre-registered set of directions and cotangents over the ladder
h × 2^{-k}, k ∈ {0, 2, 4, 6, 8, 10}, refined between useful scales; §5's quantities per cell; §6's
table for interpretation. **No further full map until a useful interval is demonstrated**, and an
execution error is never the "float32 still disagrees" branch. The directional stage needs no
memory smoke; a float32 full map, if the interval justifies one, gets its own.

**Two conclusions of the golden record's §10 exceed its evidence**, per the Director's audit at
6983d8e (`research/records/FD-SATURATION-AUDIT-2026-09-09/`, merged here): nonzero saved columns
do not exclude rounding loss, since a column aggregates many responses and nonzero contributions
can cancel; and the −0.872 correlation does not establish saturation, because both plotted
quantities carry the exact map's magnitude in opposite directions and a purely linear
counterexample yields a perfect inverse correlation. The calculations reproduce. §10 is marked
reproduced-but-not-attributive and cites the audit; the scaling identity of §9 stands as geometry,
not as a statement about local curvature.

**Schedule on the card:** the calibration first, alone; then the 12B smoke row alone; then the 12B
exact fits with the plan-progress captures in the same pass, as ordered — the exact estimator is
unaffected by this calibration and the lens work does not wait on it. Record:
`research/records/WSD-FD-CALIBRATION-2026-09-10/`, citing 558d424, 6983d8e and c3abdbe. Rows two
and three of the bfloat16 golden test are withdrawn by the Director's ruling.

**For the eventual instrument**, from the protocol's last section and the Director's note: two
outputs, kept separate — the local sensitivity from the validated autograd path, and the actual
response to a finite intervention in native precision — with their agreement measured, never
assumed.

## Amendment, Chief, 2026-09-10 — the boundary check (`4a2cfe2`): the hook is exact, the bfloat16 forward is not batch-invariant, and the matched pair runs at width 1

The protocol's §2, run first as ordered, 23 seconds of card time. The replacement hook fed its own
unperturbed capture reproduces the ordinary forward **bit for bit** at repository layers 1, 17 and
33 at batch 1, and the source-equals-target control is exact; the replacement path is sound. At
widths 64 and 256 it is not exact, and the unhooked control says why: with no hook at all, the same
frozen row at width 64 differs from itself at width 1 in every one of the 34 blocks, from the first
(mean absolute 0.0091, maximum 8), compounding by about ×1.35 per block to a mean absolute
difference of 76.4 at the target, every row of a batch agreeing with every other. The batched
forward computes a different function — a kernel choosing its reduction by batch size, amplified
through a 34-block residual stack — and the golden run's two estimators did not share it: exact
autograd at `dim_batch = 64`, finite differences at `direction_batch = 256`, anchored on a residual
captured at width 1. `assert_estimator_is_the_only_difference` passed because batch width was not
among the ν fields it compares: a gate that could not fail on the thing that mattered.

The D-CRO does not claim this explains the compression, and neither does this order: a candidate
with the right depth profile is exactly what the Director's audit refused an hour earlier, and the
finite-difference arms share a width, so the offset cancels to first order between them.

**Ruled.** (1) ν gains the batch schedule — each estimator's forward width and the width its anchor
was captured at — and the gate refuses a cross-width comparison; laptop, first. (2) The matched
pair runs **at width 1 throughout**: anchor, autograd (one prompt, VJPs with the pre-registered
cotangents) and finite differences, §4's directions and ladder, both precisions. (3) Then the width
rows at 64 and 256 for the same directions and a few steps, anchor and forward at that width, the
unchanged-residual check repeated at each width, reported as the schedule term of the protocol's
§3 decomposition. When a full map later needs batching, both estimators share the width and the
anchor is captured at it. (4) The method entry follows the sweep.

**Two consequences for other streams, recorded now:** the plan-progress captures are made at width
1 with `forward_batch` in the manifest, since a batched capture carries a width-dependent
arithmetic-path term into every probe fit; and the ν of every lens fitted at `dim_batch = 64`
declares that width, because the map was fitted on the width-64 function.

**Correction to the amendment above, Chief, 2026-09-10, from Codex's file-only review.** The
calibration record's "the hook is not implicated, and nothing about the hook needs repairing" is
withdrawn as a verdict: its field in `batch_invariance.py` is `not all(...) and False`, false by
construction, and the wider widths were only ever tested against a capture made at width 1, so a
hook defect at width > 1 was never separable from the forward's batch-dependence. What stands: the
hook is exact at batch one; the forward is not batch-invariant; the hook at wider widths is
**untested** until the unchanged-residual check is run at width w with a capture made at width w,
which step 3 above requires. The field is removed or computed from that test.

**C2, from the same review, for the calibration record and the next run:** the record states which
pieces of the protocol are **unexecuted** — matched float32 pairs, graph-once against sequential
VJP, nonzero identity-derivative controls, the ladder with realised displacements, midpoints and
spacing, per-direction autograd predictions — as unexecuted and not as failed; the next run
preserves the individual response vectors per position and direction before any reduction, as §5
requires; the sign control computes the flipped even remainder it promises; and the native control's
JSON links its runtime manifest and the frozen-token digest to the boundary manifest so the two
runs form one provenance chain.

## The width-1 ladder, Chief, 2026-09-10 — `5b0feb7`/`15eddfa`: float32 has a useful interval at every layer, native bfloat16 has none; the width rows and the first map inside an interval are ordered

Row 39, positions {8, 127}, six directions × three cotangents, both precisions, width 1 throughout,
28 seconds of card time. Median relative error |d_h − a| / |a| against autograd on the same forward:

| precision | layer | k = 0 | k = 2 | k = 4 | k = 6 | k = 8 | k = 10 | k = 12 | k = 16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16 | 1 | 0.99 | 1.04 | 1.62 | 2.06 | 5.38 | 17.2 | 134 | 1.00 |
| bf16 | 17 | 1.14 | 0.33 | 0.21 | 0.86 | 2.97 | 27.5 | | |
| bf16 | 33 | 0.113 | 0.053 | 0.099 | 0.44 | 1.36 | 2.04 | | |
| float32 | 1 | 0.98 | 0.93 | 0.65 | 0.15 | 1.1e-2 | 4.3e-3 | 6.0e-3 | 0.16 |
| float32 | 17 | 1.29 | 0.35 | 3.2e-2 | 1.9e-3 | 5.8e-4 | 1.9e-3 | | |
| float32 | 33 | 8.1e-2 | 1.3e-2 | 8.0e-4 | 8.6e-5 | 1.2e-4 | 7.0e-4 | | |

**Findings, as the record states them.** Float32 has a proper minimum at every layer tested,
flanked within about a factor of three on both sides, at k = 6, 8, 10 for layers 33, 17, 1: the
shallower the source, the smaller the step it needs, by 2⁶ across three layers, which alone refutes
one `epsilon_scale` for every layer. Native bfloat16 has no useful interval anywhere; at layer 1 the
fraction of target components returned exactly equal between the two arms reaches 1.000 by k = 16,
the difference is exactly zero, and the row pins at 1.00 — the loss measured before aggregation, the
quantity the Director's audit said the zero-column count could not reach. The unchanged-input
fraction is 0.000 in every cell, so the displacement always lands and the loss is downstream. At
k = 0, the golden run's step, the error is about 100% at layer 1 in **both** precisions: the step
alone was outside any useful interval, independent of precision. Two rows of the protocol's §6 fire
together — truncation at the original step, and precision limiting native finite differences — and
**the golden residual is not evidence about the model**. It does not refute saturation, and the
record says so; it removes the need to invoke any mechanism for the disagreement. Eighteen scalar
checks at one position on one row: the step to the full map is an inference and is marked as one.

**The hook is sound at every width, measured.** Anchoring the capture at the width it replays at
gives twelve bitwise-identical checks at widths 1, 64 and 256, while the eighteen anchored-at-one
rows in the same run fail; the `and False` verdict is removed. Correction 2 applied: ε_main 0.17
(the bound at 240 episodes is 0.1604 and 0.16 would need 242), ε_ord 0.09, ε_sub 0.11, the
decision-unit numbers kept beside them and labelled; both digests pinned.

**Consequence for the lenses fitted on the laptop, as an inference and marked as one:** they were
fitted by finite differences in float32 at the same step rule, which this ladder places at k = 0 —
about 98% relative error at layer 1, 129% at layer 17, 8% at layer 33 in float32 — under a
different reduction (same-position rather than source-mean, target-sum). Until a directional check
runs under their reduction, the MLX-fitted early- and mid-layer lenses are treated as not
derivative estimates, and every published number that read through them is bounded by this table.

**Ordered.** Step 3, the width rows at 64 and 256 with the anchor and forward at that width and
§3.1's check first at each. Then **the first finite-difference maps fitted inside a demonstrated
interval**: layer 33 at k = 6 and layer 1 at k = 10, in float32, at the width the anchored check has
cleared with the exact map at the same width, ν carrying `epsilon_per_layer`, `forward_batch` and
`anchor_batch`, a memory smoke row before the first float32 full map (`dim_batch` 32 if the smoke
says so). Then the 12B smoke row alone, then the 12B exact fits with the captures. Timing is the
D-CRO's by their own context; idle card time is the waste, an out-of-context seat the worse one.
