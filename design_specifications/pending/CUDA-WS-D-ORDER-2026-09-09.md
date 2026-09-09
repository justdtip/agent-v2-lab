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

## The width rows, Chief, 2026-09-10 — `9380e74`: in bfloat16 the derivative itself depends on the batch width; the lenses are fitted in float32 from here

Relative change of the autograd directional derivative `a = (Jᵀw)ᵀv` from width 1 to width 64,
over eighteen direction-and-cotangent pairs, nothing else changed:

| precision | layer | min | median | max |
|---|---:|---:|---:|---:|
| bf16 | 1 | 0.153 | 0.756 | 2.00 |
| bf16 | 17 | 0.012 | 0.596 | 2.97 |
| bf16 | 33 | 0.010 | 0.119 | 6.93 |
| float32 | 1 | 5.5e-6 | 2.7e-5 | 5.3e-3 |
| float32 | 17 | 2.2e-6 | 2.1e-5 | 6.6e-5 |
| float32 | 33 | 1.4e-7 | 3.3e-6 | 1.9e-3 |

**There is no width-independent bfloat16 Jacobian at these layers.** The map is a property of the
schedule as much as of the model; in float32 the same change is arithmetic noise from a different
reduction order. The bf16 width-64 ladder rows are therefore not comparable with the width-1 rows
and are not tabled beside them. Float32 at width 64 still converges, one or two rungs deeper and
two to five times looser (2.0e-4 at layer 33, 6.1e-4 at 17, 4.5e-2 and still falling at 1). §3.1's
anchored-at-width check gated first at every width and precision; the refactor reproduces the
width-1 table bit for bit (648 of 648 cells). Width 256 was not run: the retained graph exhausted
the card at 94.71 GiB; recorded with the number, and **not ordered** — widths 1 and 64 answer the
question and the frozen quantity is not changed to fit the card. Counts corrected: three
interventions at width 1, six anchored checks.

**Ruled: the exact estimator fits in float32 from here** — the stored bf16 weights cast, autocast
off, TF32 off, highest matmul precision, eager attention, determinism pinned — for the 4B refits and
the 12B fits, ν carrying `fit_dtype`, `forward_batch` and `anchor_batch`. A map that moves by a
median 76% under a change of schedule is a derivative of the rounding structure, not of the model,
and the float32 estimator is the one the ladder validated against an independent estimator. The
**captures stay native at width 1**: they are the readings of the deployed computation, and a
float32 lens read against a native capture carries the path difference as a declared term (the
first hour's promoted floor, 1.24% at 64 tokens) on every reading. The registry field is WS-E's and
SWE-2 carries it. The two in-interval float32 maps ordered above are the first fits under this rule.

**Step 1 of this order is amended:** upstream's hosted lens was fitted by bfloat16 autograd at the
width of their recipe and is by this measurement a schedule-specific object; the un-port comparison
needs their width declared before it means anything, and reports it as such.

## Correction, Chief, 2026-09-10 — Codex's audit of the ladder (`WSA-LADDER-AUDIT-2026-09-09`, `2678218`): five statements of the ladder section above are withdrawn or narrowed

The audit recomputes every cell from the saved record and is accepted. The ladder's headline holds
in its narrowed form: **float32 convergence is supported for the tested projections, and no broadly
accurate native-bfloat16 interval has been demonstrated over the tested grid.** What does not hold,
and was written into this order from the producer's summary rather than from the artefact:

1. *"The unchanged-input fraction is 0.000 in every cell, so the displacement always lands and the
   loss is downstream"* — false. At layer 1, k = 16, four of six coordinate directions are input
   no-ops with realized displacement exactly zero; the two dense directions keep nonzero responses.
   The 1.000 was a median of four no-ops over two live directions. Input representability and
   downstream rounding are separate mechanisms; the second is unlocalized.
2. *"The shallower the source, the smaller the step it needs, by 2⁶ across three layers"* — the
   normalized scales at k = 6 and k = 10 differ by 16; the absolute steps by 1,390, because the source
   norms differ. Differing optima do not refute a common adequate scale: at k = 10 all three float32
   medians are at most 0.0043.
3. *"Flanked within about a factor of three on both sides"* — not everywhere; layer 33 at k = 4 is
   9.24 times its minimum.
4. *"The eighteen anchored-at-one rows fail"* — six at width one pass; the twelve cross-width rows
   fail, as intended.
5. *The laptop lenses "bounded by this table" at 98%, 129%, 8%* — withdrawn to **unknown**: a
   different reduction, one source position probed, and a regression lens is not evaluated by a
   finite-difference check. They are treated as unvalidated pending a matched directional check
   under their own reduction, which is a policy and not a measured failure of every archive.

**Ordered with the maps:** a full-target no-op comparison before the first derivative at each new
precision, source and schedule, bound to the records that follow; the width-one failure path in
`boundary.py` gated like every other width, with the mutation as a regression; the pre-reduction
archive (native plus, minus and zero outputs per target position; requested and realized input
vectors) from which equality, signed odd/even, midpoint, representable spacing and AD-predicted
responses are derived; an accuracy and small-signal policy declared before the maps, with an
absolute-error floor beside relative error so a small reference signal reads *unresolved*; k = 6
and k = 10 carried as candidate settings; the two maps labelled as the test of the projection-to-
column inference, with J − I and FD − I beside the errors; manifests, frozen invocation, progress
events, frozen-token digest and measured autocast state bound to every record.

## Correction, Chief, 2026-09-10 — Codex's width audit (`WSA-WIDTH-AUDIT-2026-09-09`, `d037454`): the widened ladder's step was eight times its rung; the autograd drift stands, its cause is not yet isolated; the path term is context, not an error bar

**W1.** The widened ladder took its norm over all 64 replicated copies, so every width-64 step was
√64 = 8 times the width-1 step at the same rung (recorded ratios 8.00 in float32, 7.8–8.0 native).
"Float32 at width 64 still converges, one or two rungs deeper and two to five times looser" is
withdrawn: width-64 rung k is width-1 rung k − 3 in physical step, the even-rung grids supply no
equal-h pairs, and at layer 1 the width-64 minimum's small-step side was never measured. The
production fitter uses one replica's norm and is not affected; the ladder is corrected to the same,
with a fixture that repeating a batch does not change the per-row step. The width-64 FD rows stand
as within-schedule measurements at their stated h. **Before the two float32 maps are cited, the
record states their realized h against the width-1 ladder's; a map that inherited the batched norm
sits three rungs off its label and is re-run at the corrected step.**

**W2.** The autograd drift reproduces (0.7564 / 0.5962 / 0.1192 native; ~1e-5 float32) and is
independent of W1. Its attribution is not isolated: the measured `a_64(x_64) − a_1(x_1)` mixes
changed arithmetic with a changed anchor, and a smooth function can move its derivative by the
observed proportion between two anchors a fraction of a percent apart. The protocol's same-anchor
control is ordered: width fixed with the two saved anchors inserted in turn, then anchor fixed
across widths 1 and 64, per-projection vectors kept, both brackets reported. Until it exists the
wording is *native-path autograd readings at their respective anchors are strongly
schedule-sensitive for the tested projections, and the matched float32 readings are far more
stable*; "a derivative of the rounding structure" is withdrawn. The float32 fitting policy stands
on the stability evidence and is not reopened.

**W3.** The 1.24% at 64 tokens is historical context and not an error bar for a float32 lens read
against a native capture: relative readout error has its own denominator, and normalisation,
ranking and thresholded dictionary activations add questions a raw-residual percentage cannot
answer. That application difference is **unmeasured** until a paired comparison at the reading's
own positions and context exists; the bridge's constant is amended to say so (SWE-2).

**Provenance:** `responses.npz` written durably per completed unit with a hash and index, bound to
its scalar cells; width-specific manifests, observed precision settings, frozen invocation, token
digest and gate outcomes archived with each run.

## The first maps inside an interval, Chief, 2026-09-10 — `17bb82d`: the golden test passes on its own terms in float32 at width 1

| | first golden run | this one |
|---|---|---|
| precision | native bf16 | coherent float32, TF32 off, `highest` matmul |
| widths | exact 64, difference 256, anchor 1 | 1 on both sides |
| step | `epsilon_scale` 0.01 at every layer | k = 10 at layer 1, k = 6 at layer 33 |
| worst relative difference | 1.0245 | 0.00485 (layer 1), 0.00286 (layer 33) |
| cosine | 0.015 (layer 1), 0.825 (layer 33) | 0.9999907, 0.9999962 |

Exact reproduces itself at 0.0; the transposed control separates by 292× and 348×; both findings
sit above the float32 storage floor (5.96e-8), so this is agreement measured. The scalar ladder
predicted the map with nothing tuned: layer 1's median of 4.3e-3 over eighteen projections against
a worst of 4.85e-3 over 2,560 columns. Twelve minutes of card time; the memory smoke read 14.56 GiB
at width 1 against 43.9 GiB for the bf16 exact side at width 64.

**Two deviations from the ruling above, both accepted.** The maps were fitted at **width 1**, not
64: the k values are width-1 minima, the width rows moved the intervals, and the width audit shows
the widened ladder's step was eight times its rung, so the width-64 minima were mislabelled. Width 1
is the canonical function and the maps stay there. The **layer-shifted control** is recorded
`available: false` with its reason — each map has one layer at its own step — and the ν question
is answered: a finite-difference map is per layer at its own step, ν carries `epsilon_per_layer`,
and the control is constructible later if needed; the transposed control gates now. The record
states the realized h per map beside the ladder's, and that the batched-norm defect does not apply
at width 1. The D-CRO's branch had never reached the card's bare repository (its `origin` is local);
it now has, and the maps ran from a worktree outside the shared checkout.

**Ordered:** the 12B smoke row alone, in float32 at width 1, with the same-anchor control of W2 in
the same seat; then, on the laptop, the capture code to the contract (width 1, key `(task_id, step)`,
`prompt_sha256`, token index, `rendered_rows`, manifest fields, native precision, outside the
checkout) with tests, and P1–P6 with the remaining C2 items; the 12B exact fits with the captures run
only when the capture code meets the contract. `cuda-ws-d` is reviewed and merged in the morning,
where SWE-2's registry declaration meets the D-CRO's ν fields.

## The same-anchor control, Chief, 2026-09-10 — `5c1c004`/`6c94626`: the width shift is the anchor at depth and the arithmetic at the first layer; what survives everywhere is conditioning

The protocol's same-anchor decomposition, fourteen seconds of card time, identity residual exactly
0.0 in all 54 cells:

| layer | anchor moved by | observed shift | anchor term | arithmetic term |
|---:|---:|---:|---:|---:|
| 1 | 0.19% | 0.756 | 0.502 | 1.108 |
| 17 | 1.5% | 0.596 | 0.610 | 0.032 |
| 33 | 8.8% | 0.119 | 0.119 | 0.005 |

**Ruled wording.** At layers 17 and 33 the width shift is the anchor: hold the point fixed and change
only the width, and the bfloat16 derivative barely moves. "No width-independent bf16 Jacobian" is
**withdrawn for those layers.** At layer 1 the arithmetic term is real and the larger, the two partly
opposing, where thirty-two blocks of backward accumulation run below the source. What survives at
every depth is **conditioning**: a 0.19% move of the anchor at layer 1 moves the derivative read at
it by 50%, which is not about batching at all. The float32 fitting policy stands on the stability
evidence — the float32 derivative moved 1e-5 under the same change — and not on the attribution.
Ordered: the same anchor displacement inserted into the float32 path, which separates an
ill-conditioned function from ill-conditioned arithmetic, before the 12B lens is read at early
layers.

R1, R3, R4, R5 and the width fix are applied at `6c94626`: the k = 16 median was four input no-ops
outvoting two live directions, and the maps are fitted inside the representable region (unchanged-
input fraction 0.000 through k = 10); a width-one failure could never have entered `boundary.py`'s
failure list, and every matched schedule now gates by intervention; the headline reads *no broadly
accurate native-bf16 interval has been demonstrated over this tested grid*; an absolute-error floor
is declared and the small-signal cell is *unresolved*; the minima differ by 16 in normalised units
and "a single `epsilon_scale` is refuted" is withdrawn in favour of *the estimator's k = 0 sits far
outside every layer's interval*; the laptop inference is unknown; the ladder norms one replica with
a fixture. Queued for the morning: R2's unreduced archive, the float32 no-op boundary at each new
precision, `responses.npz` durable per unit, the capture code to the contract, P1–P6 with C2.

## The 12B smoke row and the 12B same-anchor control, Chief, 2026-09-10 — `a7467c0`: float32 at width 1 is comfortable at 12B and width-stable at every depth

Per-process allocated, width 1, idle card, 52 seconds across two runs:

| | bf16 | coherent float32 |
|---|---:|---:|
| weights loaded | 22.01 GiB | 43.97 GiB |
| free after load | 72.11 | 38.12 |
| exact fit, one layer, `dim_batch = 1` | 22.11 peak, 3.9 s | 44.10 peak, 6.0 s |
| the VJP control at width 8 | 31.96 peak | 59.01 peak |

**A float32 12B exact fit at width 1 is comfortable**: 44 GiB against 95, six seconds a layer for one
row; the promotion is the whole of the cost, and there is no memory argument against the fitting
policy at the larger model.

The same-anchor control on the 12B, width 1 against 8, both precisions, identity 0.0 in all 108
cells: bf16 anchor-dominated at every depth (observed 0.370 at layer 1 falling to 0.022 at layer 47;
anchor 0.330, 0.029, 0.033; arithmetic 0.096, 0.028, 0.014); **coherent float32 width-stable at
every depth by four to five orders of magnitude** (4.3e-5, 7.3e-6, 2.3e-6). The bf16 pattern is not
compared with the 4B's — that control changed the width by 64 and this one by 8, and the arithmetic
term scales with the schedule perturbation — and only the within-model trend is claimed.

**A control that could not fail, caught and closed:** the first 12B run promoted to float32 before
the control, so the two anchors differed by 4.6e-7 and both terms vanished by construction; the
native run was added, the precision is an argument, and the trap is named in the script's help. It
belongs in the method entry after the sweep. §9 carries the realized-h line: both maps at the
width-1 ladder's h to ratio 1.0000000000, for two independent reasons.

**Queued for the morning, in order:** the capture code to the contract with tests; P1–P6 with the
remaining C2 items; R2's unreduced archive, the float32 no-op boundary before the first derivative
at each new precision, `responses.npz` durable per unit, and the float32 leg of the
anchor-displacement control (the bf16 displacement inserted into the float32 path); the 12B exact
fits with the captures only once the capture code meets the contract. `cuda-ws-d` is reviewed and
merged first thing.

## The float32 displacement control, Chief, 2026-09-10 — `e771c2b`: the conditioning is the function's, and an early-layer lens does not transport to a nearby anchor

The same anchor displacement the width change produced natively, applied in coherent float32 at
width 1 with no schedule change, nine seconds of card time, 108 cells:

| precision | layer | ‖δ‖/‖x‖ | derivative change | from a random δ of equal norm |
|---|---:|---:|---:|---:|
| bf16 | 1 | 0.19% | 0.471 | 1.006 |
| float32 | 1 | 0.19% | 1.035 | 0.749 |
| bf16 | 17 | 1.5% | 0.585 | 1.560 |
| float32 | 17 | 1.5% | 0.697 | 1.277 |
| bf16 | 33 | 8.8% | 0.120 | 0.313 |
| float32 | 33 | 8.8% | 0.125 | 0.299 |

As amplification — derivative change per unit of anchor movement — 245× native and 539× float32
at layer 1, 39× and 46× at 17, 1.4× in both at 33. **Ruled: the sensitivity is the function's.** The
model's own map is steep at early layers and well conditioned at the last block; a random
displacement of the same norm does the same or more, so it is not a special direction. This
explains the ladder's per-layer minima without precision: a map whose derivative moves by its own
size under a 0.19% input move needs a very small step for its secant to approximate its tangent.
The float32 fitting policy is unchanged; what changes is how a lens may be read.

**Rule, carried to the bridge and the readings:** a Jacobian lens is a statement about the
neighbourhood it was fitted in, and at early layers that neighbourhood is 0.2% wide. **An
early-layer float32 lens is not read against a native capture — or any capture from a different
width, precision or path — until that pairing is measured at that layer.** The path term is
therefore per layer: benign at the last block, the whole answer at the first. SWE-2's bridge
records it per layer, with `measured: false` until the paired comparison exists, and refuses the
early-layer cross-path reading by name rather than annotating it. Readings on the same path as
the fit — a float32 generation read through a float32 lens — carry no such term. One row, one
position, eighteen projections, three layers; the record says so.
