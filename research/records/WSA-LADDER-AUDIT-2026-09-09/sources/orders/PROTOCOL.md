# Resolution protocol: calibrate derivatives before fitting another full map

**Status: proposed, not executed by Codex.** The Chief's relay names the matched whole-float32 pair as the front-runner and withdraws the extra bf16 rows. The GPU owner schedules and executes the control. No three-minute runtime or memory projection is asserted from fixture measurements.

## 1. Freeze the differentiated quantity

Retain row 39's exact token IDs, length 128, selected source/target positions `{8,127}`, checkpoint digest, attention masks and position IDs. Keep the source and target as decoder-block outputs before final normalization, with the current target sum and source mean. Record all three conventions separately: source layer, target layer, and reduction. Use repository conversion helpers for indices.

The declared map is

\[
J_l^\nu=\frac1{|V|}\sum_{p\in V}\sum_{q\in V}
\frac{\partial H_T[q]}{\partial H_l[p]}.
\]

Preserve each `(source position, target position)` contribution before its reduction. The mean must not hide a dead response or cancellation. The pair with target before source is a useful causal-zero check.

Use the same numerical weight values for arithmetic comparisons: a bf16 value cast to float32 is represented exactly. Do not change revision, adapters, tokenizer, attention implementation or selector. Reuse the package's loader, upstream recorder and existing replacement hook; no parallel architecture loop is needed.

## 2. Verify the intervention before differentiating

Start with one input row. Compare the ordinary forward to a hook that replaces the source residual with its identical, unperturbed capture. Match dtype, device, shape and batch schedule. Measure every target before reduction. A changed result is a boundary/schedule failure to diagnose before any FD interpretation.

Initially use the same forward batch size for AD and FD. The graph-once/VJP machinery avoids duplicating the forward just to supply cotangents. First cross-check its directional result against a sequential VJP on the same forward schedule where checkpoint-level equivalence is not yet measured. If larger batches are later needed, repeat the unchanged-residual check at each actual batch width. The old 64/1/256 schedule is not a proven equivalence.

Include a source-equals-target control with the same reduction: its mathematical AD map is identity. Native FD must instead match the realized displacement divided by the nominal `2h`; input rounding can make that differ from the requested direction even for identity. This is a diagnostic fixture/control extension, not another production source layer. Flip exactly one of the tangent or cotangent for the sign negative control. These controls supplement, rather than replace, the existing transpose and selector controls.

## 3. Run a matched arithmetic triangle

First run **whole-float32 AD against whole-float32 FD**, including coherent block parameters, buffers and replacement tensors. Both use the same new float32 primals. Disable autocast and stamp the installed runtime's actual highest/IEEE matmul settings, attention kernel, determinism and observed dtypes; do not infer internal precision from tensor labels alone.

Separately, retain **native-bf16 AD against native-bf16 FD**. Those captures remain native; this protocol does not promote the model's ordinary records.

For attribution, compare **bf16 AD against float32 AD at the same stored source residual**, cast losslessly into float32, with fixed weight values. This anchored-tail control differs from recomputing a new float32 prefix; record which is used. A hook on a coherent float32 model can insert the frozen source residual without inventing a manual tail.

At identical anchors, directions and steps, the vector identity

\[
FD_b-AD_b=(FD_b-FD_f)+(FD_f-AD_f)+(AD_f-AD_b)
\]

separates changed arithmetic, FD error on the float32 function, and changed AD arithmetic. Norms of these vectors are not additive; retain their alignments. If anchors differ, add and report that difference rather than using this identity as causal attribution.

## 4. Use directional checks before full matrices

At repository layers 1, 17 and 33, preregister a small fixed set of normalized coordinate and dense directions and several fixed output cotangents. Begin at the early and late layer if resources require staging. Reuse the saved forward graph for VJPs through the existing recorder. The scalar check is

\[
a=(J^\top w)^\top v,\qquad
 d_h=w^\top\{F(x+hv)-F(x-hv)\}/(2h).
\]

Use multiple projections: one can miss an error orthogonal to it. FD output vectors can be retained without materializing a full Jacobian. Report both absolute error and error relative to the measured signal; a near-zero derivative whose error dominates is unresolved, not a relative-error pass. Projection tests localize failure; they do not certify a full matrix.

Sweep geometrically spaced steps, including the original step and successively smaller values. A practical first diagnostic ladder is the original `h` multiplied by `2^{-k}` for `k=0,2,4,6,8,10`, refined between neighboring useful scales. This is a declared trial ladder, not a universal step rule. Record each step relative to the affected token norm, local coordinate scale and actual representable spacing. Do not replace the old full-sequence heuristic with another unmeasured constant.

Before further full maps, require a demonstrated useful interval: several neighboring steps showing stable derivatives and reduced error against matched AD, with large and small steps exposing their respective limits where observable. Preserve the historical 3.6e-3 fixture result as a benchmark, not a machine-precision guarantee. Report achieved accuracy and the scope of the checks; choose a scientific error budget explicitly before issuing a global acceptance claim.

## 5. Record the quantities that identify the mechanism

For every layer, position, direction, precision and step, save:

- Requested displacement, realized `x_plus − x` and `x − x_minus`, their norms, and midpoint shift `(x_plus+x_minus)/2 − x`.
- Fraction of input components unchanged **within the intended perturbation support**, and fraction of **output pairs** exactly equal, before target summation. Also report component-wise output zero fractions; a zero aggregate can be cancellation.
- Signed target differences, actual output spacing at the observed magnitudes, and AD-predicted changes. A large representable input change does not guarantee a resolvable downstream response.
- Odd response `F_plus − F_minus` and even remainder `F_plus + F_minus − 2 F_zero`, with dtype and accumulation provenance.
- Absolute/relative discrepancy, reference and FD norms, cosine, and position-level contributions. Compute all diagnostics after preserving the native readings; casting afterward does not undo earlier rounding.

For the realized direction `v_actual=(x_plus−x_minus)/(2h)`, compare with `J v_actual` as well as `J v`. This helps isolate intervention rounding. It does not remove midpoint shift or downstream rounding. A scalar denominator correction is valid only for a collinear displacement. An unchanged pair is unresolved; never normalize a zero displacement.

When full maps become justified, also report `J−I` and `FD−I` in this raw residual coordinate space. The shared skip identity can improve late-layer cosine without validating the learned correction. This diagnostic does not apply unchanged after normalization or unembedding.

## 6. Interpret results without changing the question

| observation | defensible conclusion | next action |
|---|---|---|
| unchanged-residual hook changes output | boundary or schedule mismatch | repair/localize that seam before derivative interpretation |
| coherent float32 FD converges to coherent float32 AD | tested float32 local derivatives are supported | compare native diagnostics, then validate full maps at the chosen interval |
| bf16 loses its useful interval while float32 retains one | precision limits native FD at these scales | keep AD as the sensitivity instrument; report native FD unresolved |
| float32 error improves greatly as step shrinks | original step had substantial truncation/nonlinear error | calibrate a smaller interval; do not infer a universal step |
| float32 still fails at one step | inconclusive about cause | inspect its sweep, anchors, schedules and realized directions |
| both precisions fail matched sweeps | implementation/definition mismatch or unresolved conditioning remains | localize on positions/directions; do not average additional broken maps |
| local AD agrees but finite native interventions do not | derivative does not predict that finite-amplitude operation | measure the operational response separately |

The intended instrument has two outputs: **local sensitivity**, estimated by the validated AD path, and **finite native intervention response**, measured in the actual run precision. The former informs what a small change would do under the declared derivative convention; the latter tests whether a feature-sized edit behaves that way. Their discrepancy is useful evidence rather than an excuse to silently change either definition.
