# The golden test on the card: the exactness gate passes, and the finite-difference operand does not survive bf16

**D-CRO, 2026-09-09, on the rented RTX PRO 6000.** Gemma 3 4B, `google/gemma-3-4b-it`, bf16, eager
attention, `cuda:0`, determinism pinned, upstream at `581d398613e5602a5af361e1c34d3a92ea82ba8e`
checked inline and matching. Every figure below is `basis: measured-here` unless it says otherwise.
Artefacts under `/workspace/wsd/out/`; the run wrote each result as it completed.

**The short form.** The gate that has no tolerance passed exactly. The measurement the workstream
exists to produce did not, and the reason is the instrument rather than the model: the
finite-difference estimator recovers nothing of the exact map at the shallow layers, at any step over
four decades. The instrument's own controls are what said so.

**Read §3 and §8 together, and prefer §8's reading of the cause.** §3 was written first and attributes
the failure to bf16 cancellation. §8 answers the Chief's later question by counting the artefact's
exactly-zero columns — **none, of 84,480** — which refutes the rounding-to-zero mechanism, and shows
instead a map systematically *compressed*, at 0.13 to 0.76 of the reference's norm. That is the
signature of a saturating response to an oversized step, and it supports the Director's reading: the
step is 1% of the whole sequence array's Frobenius norm applied to one coordinate of one position, so
a coordinate moves several times its own size while the vector's norm moves 11%. **The cause is not
settled between step size and precision**, the two are not separable by anything measured here, and
the control that separates them is the Chief's matched-precision pair crossed with a step rule.
Nothing in §3's numbers changes; its sentence about the mechanism does.

## 1. What passed

| gate | expected | measured |
|---|---|---|
| upstream provenance, commit and vendoring | `581d398…`, `vendored: false` | matched, from the installed distribution |
| exact estimator reproduces itself | **exactly 0.0** | **0.0**, 33 layers, twice, 15.8 s |
| controls formally separate | each further than the candidate | all three, but see §4 |

The exactness gate is the one the order fixed at zero rather than at a tolerance, on the argument
that a nonzero value there is nondeterminism and a tolerance would hide it. On CUDA, with
`device.pin`, it is zero.

## 2. What the golden comparison produced

Exact autograd against finite difference, same rows, same two positions, same model object, same
declared precision, `capture_dtype: native` on both. Worst relative difference **1.0245**, and the
per-layer shape is the finding:

| repo layer | relative difference | cosine |
|---:|---:|---:|
| 1 | 1.0142 | **+0.015** |
| 9 | 0.9967 | +0.112 |
| 17 | 0.9725 | +0.295 |
| 25 | 0.8500 | +0.528 |
| 33 | 0.5690 | +0.825 |

A cosine of 0.015 is not a noisy estimate of a map. It is noise. The agreement climbs monotonically
with proximity to the target, which is what a signal attenuated through blocks looks like, not what
an estimator difference looks like.

**The step-halving check said so before the per-layer table did.** The order requires one layer at
the halved step so the fixture's divide-by-four is shown on the card. The fixture gives 3.9. The card
gave **1.15** — 0.9725 against 0.8436 at layer 17. A central difference whose error does not fall
with the step is not truncation-limited, and that single number is what turned a disappointing
residual into a diagnosis.

## 3. The diagnosis, run because of that ratio

Four decades of step size at the worst layer and the mildest, in the model's own bf16 and in
promoted float32. Six point eight minutes.

| upstream layer | path | epsilon scale | epsilon | relative | cosine |
|---:|---|---:|---:|---:|---:|
| 0 | native | 1e-2 | 101.6 | 1.014 | +0.015 |
| 0 | native | 1e-3 | 10.16 | 1.214 | **+0.157** |
| 0 | native | 1e-4 | 1.016 | 5.288 | +0.005 |
| 0 | native | 1e-5 | 0.102 | 45.73 | **−0.005** |
| 32 | native | 1e-2 | 8818 | 0.569 | +0.825 |
| 32 | native | 1e-3 | 881.8 | **0.286** | **+0.959** |
| 32 | native | 1e-4 | 88.18 | 0.413 | +0.911 |
| 32 | native | 1e-5 | 8.818 | 1.391 | +0.427 |
| 0 and 32 | promoted-float32 | all four | — | raises | see §5 |

**At layer 32 the curve is textbook**: truncation error falling as the step falls, roundoff rising
below it, a minimum at 1e-3. That minimum is still a **29% relative difference**, which is not
agreement between two estimators of one quantity.

**At layer 0 there is no minimum worth having.** The best cosine over four decades is 0.157, and by
1e-5 the relative difference is 45.7 with a *negative* cosine: the estimate is roundoff amplified by
1/2ε. The roundoff wall arrives before the truncation error becomes small, so the window in which a
central difference works does not exist there at this precision.

The mechanism is consistent across both tables. A perturbation at layer L reaches the target through
33−L blocks; the change it makes to the target shrinks with that distance, and bf16 carries about
eight bits of mantissa, so the difference of two nearly-equal forwards loses the signal to
cancellation first at the layers furthest from the target. The depth gradient in §2 and the missing
minimum at layer 0 are the same fact seen twice.

## 4. Two things the controls say that the pass/fail does not

The report records all three controls as separating, and they do, but the margins are the point:

| control | worst relative | margin over the candidate |
|---|---:|---:|
| transposed | 1.414 | **1.38×** |
| wrong corpus | 1.466 | **1.43×** |
| layer-shifted | 238.0 | 232× |

**The candidate is 38% closer to the reference than a transposed lens.** The harness's own rule is
that a control which barely separates is not a control; here it is the *candidate* that is barely
distinguishable from a deliberately wrong lens, which is the same sentence from the other end.

**And the layer-shifted control separates for the wrong reason.** The reference maps' Frobenius norms
are 18,417 at layer 1 and 77.4 at layer 33 — a factor of 238 across depth — so shifting the layers
compares maps of wildly different magnitude and the control passes on scale rather than on geometry.
It is a weak control in this regime and should be normalised before it is trusted. That was not
visible on the fixture, whose four layers have comparable norms.

## 5. A correction to something I wrote yesterday

I recorded the arithmetic-path hazard as *"loud on a fixture, invisible on the model that matters"*,
reasoning that a real HF block would silently promote a float32 tensor while the toy fixture raised.
**On this model it raises too**, at every step and both layers:
`RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16`.

So the promoted path is not merely refused by the golden gate, it is **unavailable** on a bf16 Gemma,
and my justification for the gate was wrong even though the gate itself is right. The gate stands on
the plain ground that two fits which ran different arithmetic are not two measurements of one
quantity. WS-A's 69.4% figure must therefore come from promoting the whole block rather than from
injecting a promoted tensor into a bf16 one, and those are different experiments; whoever owns that
row should say which it was.

## 6. What this means for the golden test, and what I recommend

The golden test as specified — finite difference against exact autograd on the same rows — **cannot
be run at bf16 on this model**. It is not a threshold question: at the shallow layers there is no
step at which the estimator is measuring the Jacobian at all.

Three routes, and the choice is the Chief's:

1. **Both estimators at float32.** Load the model in float32 and repeat. The exact side peaked at
   43.9 GiB in bf16; float32 roughly doubles the activation term, which is plausible inside 96 GiB
   but must be smoked before it is scheduled. This is the experiment that decides whether the
   estimator is unusable *in principle* or unusable *at this precision*, and it is the one I would
   run next.
2. **Restrict the comparison to the layers where it is defined.** Report the estimator difference
   only where a step window exists, with the window measured per layer, and say plainly that the
   shallow layers are not measured. That is honest but it is a much smaller claim than the
   workstream was ordered to make.
3. **Retire the finite-difference operand.** Its own docstring says it exists to be run once and
   retired; this may be that once, with the answer being that the two estimators cannot be compared
   at the precision the programme runs at.

**Rows two and three are not queued.** The Chief authorised extending the subset from one row to
three. Extending a measurement whose instrument is noise at most layers would spend fifty-four
minutes of a rented card to average three noisy estimates, so the extension waits on the precision
question above. That is the one instruction in today's orders I have not carried out, and this
paragraph is why.

## 7. Provenance

Run at 10:26–10:58Z, 31.2 minutes, under the `d-cro` window with `AGENT_V2_BOX_STATE_DIR=/workspace/box-d-cro`,
sharing the card with no other GPU job at the time. Diagnosis at 11:00–11:07Z, 6.8 minutes. One row
of the held split, drawn with seed 20260910, index 39; the wrong-corpus control is index 84. Two
positions per row, index 8 and the last. `max_seq_len` 128, `direction_batch` 256, epsilon scale
0.01 for the main run. The subset departure from three rows to one, and the 48.23 s per row per
layer that forced it, are in `manifest.json` and were written before the run.

Timings are wall clock on a shared card and carry `basis: shared-card` where the runbook requires it;
the per-process allocated peaks — 43.9 GiB exact, 10.25 GiB finite difference — are per-process and
stay valid.

---

## 8. The Chief's three standing questions, answered from the record and the artefact

Asked after row one was read. **No card time**: (a) and (b) are repository and manifest reads, (c) is
a CPU `numpy` read of an artefact already on disk. The GPU was at 1 MiB before and after.

### (a) The laptop's finite-difference estimator ran in **float32**, and that changes what row one says about the MLX lenses

`lens_fitting/jacobian.py` promotes before it perturbs: `full = full.astype(mx.float32)`,
`h.astype(mx.float32)` in `PositionState`, directions and responses `np.float32`, and its own
recorded parameter string says so in as many words —
`epsilon="float32: 0.01 * norm(full sequence primal) / norm(one tangent); zero primal: 0.01"`.

So every finite-difference lens this programme has ever fitted was fitted **on the promoted path**,
on a 4-bit checkpoint, in float32. Row one ran **native bf16**. Under the arithmetic-path definition
this record itself introduced, those are not the same estimator, and row one is therefore not
evidence about the MLX-fitted lenses at all: it is the first measurement of a *different* one. The
comparison that would speak to those lenses is a float32 finite difference against a float32 exact,
which is the matched-precision pair the Chief has since specified.

### (b) The two valid positions of 128 are the **declared subset**, not the mask

They are the order's own declaration — *"two positions per row, index 8 and the row's last
position"* — implemented by a `two_positions` selector passed identically to both estimators, so
upstream's `valid_position_mask` never ran and `skip_first` never applied. `n_valid: 2` in the
manifest is the selector's output and nothing about `max_seq_len=128` or a long episode.

The thin basis is therefore a property of the declared design rather than an accident, and it is a
real constraint on §6.2: a per-row Jacobian averaged over two source positions estimates the
position-averaged object from two samples of it. Widening it is a change to the declaration and to
the cost — the estimator's forwards scale linearly in the position count, so ten positions is five
times the run.

### (c) Nothing rounded to zero: **0 exactly-zero columns of 84,480**

A direction whose response was exactly zero at both selected positions leaves its column exactly
zero, because the accumulator is only ever incremented by that direction's own delta. Across all 33
layers and all 2,560 directions: **none**. The reading that the response fell below bf16 resolution
and rounded away is refused by the artefact.

What the artefact does show is that the map is systematically **too small**, monotonically less so
with depth:

| repo layer | median column norm, FD | median column norm, exact | ‖FD‖ / ‖exact‖ |
|---:|---:|---:|---:|
| 1 | 56.1 | 272.9 | **0.185** |
| 5 | 20.8 | 139.4 | 0.131 |
| 13 | 3.34 | 10.69 | 0.320 |
| 21 | 1.94 | 2.30 | 0.539 |
| 33 | 1.07 | 1.09 | **0.756** |

Ratio across all layers: 0.128 to 0.756. A map at a fifth of the reference's norm with a cosine of
0.015 is not a response lost to rounding — a lost response gives a zero column, and there are none.
It is the signature of a **compressed** response, which is what a saturating nonlinearity returns to
a displacement far outside its linear neighbourhood.

**That supports the Director's reading over mine and over the Chief's.** The step is 1% of the
Frobenius norm of the *whole sequence's* residual array, applied to a *single coordinate* of a
*single position*. At layer 0 that is ε = 101.6 against a per-position residual norm of about 898
and a typical coordinate magnitude of about 17.8 — so the coordinate moves by **5.7 times its own
size** while the vector's norm moves 11%. The global figure sounds small and the local displacement
is enormous, and the compression is exactly what one would expect of it.

It also explains the depth gradient without appealing to precision: ε grows with the residual norm,
but so does the local structure, and the number of nonlinear blocks the displacement must traverse
falls from 33 to 1. Fewer blocks, less compounding.

**What this implies for the control, stated as a design note and not as a decision.** A step defined
relative to the coordinate being perturbed — or to the position's own residual norm — would test the
linear regime that the estimator assumes. Separating that from precision needs the matched-precision
pair the Chief has specified, and the two can be crossed in one sweep: step rule × precision, at one
shallow layer and one deep. That is the Chief's and the consulting mathematician's to settle; the
measurement above is offered as an input to it, not as a conclusion.
