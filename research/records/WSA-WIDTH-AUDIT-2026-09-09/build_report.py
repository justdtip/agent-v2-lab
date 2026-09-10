"""Generate the review from verified source-derived data."""
from pathlib import Path
from analyze import analyze
ROOT=Path(__file__).resolve().parent
x=analyze()
rows=[]
for s in x['summaries']:
    rows.append(f"| {s['precision']} | {s['layer']} | {s['AD_relative_change_median']:.7g} | {s['AD_relative_change_max']:.7g} | {s['h64_over_h1']:.9g} |")
table='\n'.join(rows)
report=f"""# Width comparison audit: AD drift reproduces; the FD ladder also changed its step

**Codex, 2026-09-09. Material new finding.** The width-64 ladder increases the requested
per-row perturbation by about eight, because its norm includes all 64 copies. Therefore its
finite-difference accuracy cannot be compared rung-for-rung with width one as a pure schedule
effect. The autograd width drift is independent of that step bug and reproduces. Its attribution
still combines a changed numerical forward with changed residual anchors.

Sources: producer `9380e74`, orders at `a3614be`, protocol `558d424`. The previous ladder audit
`2678218` is accepted and merged at `798aec7`; this record addresses the new width experiment.
All source bytes and full commits are in [SOURCE-MANIFEST.json](SOURCE-MANIFEST.json). Every
experiment number below is recomputed from saved producer device cells. No checkpoint, live run,
source experiment, order, lock, or model process was changed. Native captures and the newly
approved float32 fitting policy are retained; this review does not select a new precision.

## W1 — batch replication multiplies the perturbation; the fitter uses a different norm

In [ladder.py](sources/calibration/ladder.py), `source` now has shape `[width, sequence, hidden]`,
and `norm = vector_norm(source.float())` uses all its entries. The perturbation applied to each
row is `h = 0.01 * norm * 2**(-k)`. For B identical copies,

`norm(repeat(x, B)) = sqrt(B) * norm(x)`.

Thus B=64 multiplies each row's step by eight. The recorded h ratios confirm it:

| precision | repo layer | median relative AD change, 1→64 | maximum relative AD change | h64 / h1 at the same k |
|---|---:|---:|---:|---:|
{table}

The small departures from eight in bf16 include the already measured change in the residuals.
In float32 the ratio is eight to approximately the precision shown. This is a source-level
implementation effect with a direct numerical check, not a proposed model mechanism.

For the same physical step in the approximately invariant float32 case, width-64 rung k corresponds
to width-one rung **k−3**, not k. The apparently deeper optimum is therefore confounded by a
three-unit horizontal shift of the step axis. The existing even-rung grids do not supply equal-h
pairs; do not interpolate a new measured match between them. At layer 1 the width-64 best reported
rung is also the last one tested, so the lower-step side of its minimum is unmeasured.

The production [finite-difference fitter](sources/fitter/finite_difference.py) explicitly keeps
`recorder.activations[layer][:1]` before computing the norm. It therefore uses the norm of a single
copy. The widened ladder and this fitter **do not implement the same physical step at a shared k**.
That does not prove the next map's chosen step is poor; it means the width-ladder label cannot
certify it without reporting and comparing the actual h.

**Action before attributing FD differences to width:** calculate the norm on one captured replica,
or freeze explicit physical steps for all widths. Keep actual h, its per-token scale, and its
realized displacement in each cell. A pure schedule comparison must hold requested steps fixed
as well as directions and numerical weight values. Add a fixture that repeating the same residual
batch does not change its per-row step. The analytic fixture here uses residual `(3,4)`: its norm
is 5, and 64 copies have norm 40. For the unchanged smooth function `F(x)=x³`, central-difference
truncation error scales as h²; this accidental eightfold step increase alone makes that error
**64 times larger**. This is an exact mathematical counterexample, not a fit to the device curve.

The reported width-64 degradation cannot currently be called a measured larger numerical floor,
or solely the variation of the batch offset with perturbation. It mixes changed schedule, changed
anchor and changed h. Keep the FD results as valid measurements of their respective declared
experiments, and compare accuracy at matched physical steps before assigning that mechanism.

## W2 — the AD drift stands, but different anchors do not isolate derivative arithmetic

All **648** width-one pairs of `a_requested` and `d_h` reproduce the pre-refactor values exactly.
The width comparison has 18 distinct AD projection pairs per precision/layer; each is repeated
across six k values. The table uses each pair once. Native medians 0.756390, 0.596226 and 0.119226
reproduce; float32 medians are much smaller. The derivative is computed before h is used, so W1
**does not explain away the AD result**.

However, the new code captures a fresh source residual after forwarding at each width. Write a
projected derivative as `a_B(x_B)`, where B is the schedule and x_B is one prompt's captured
residual tensor, replicated to B rows where needed to define the same intervention space. The measured difference is

`a_64(x_64) − a_1(x_1)`.

It contains both changed arithmetic at a fixed anchor and movement to another anchor. One exact
add-and-subtract decomposition is

`[a_64(x_64) − a_64(x_1)] + [a_64(x_1) − a_1(x_1)]`.

Neither bracket is measured separately. The protocol's same-anchor control exists precisely for
this distinction. A smooth function can change its derivative by the observed proportion without
any change in arithmetic: for `F(x)=(x−1000)²/2`, anchors 1001 and 1001.756 give a 75.6% derivative
change. The exact fixture verifies this; it does not claim the model is quadratic or simulate bf16.

The supported claim is **native-path AD readings at their respective anchors are strongly
schedule-sensitive for these tested projections**, while the matched float32 readings are much
more stable. Calling the native map a derivative of the rounding structure rather than of the
model assigns a cause not isolated here. AD applies derivative rules through the computation;
it is not an independent experiment separating anchor movement from derivative arithmetic.
[Baydin et al., Automatic Differentiation in Machine Learning](https://jmlr.org/papers/v18/17-468.html).
PyTorch documents that mathematically equivalent batched and sliced operations need not agree
bitwise, but that fact supplies neither a magnitude bound nor this causal attribution.
[PyTorch numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html).

The bf16 width-one and width-64 FD results are comparable as **within-schedule estimator errors**
when their reference, anchor, width and step are stated. They are not interchangeable estimates of
one frozen derivative. The source's wording that they cannot be compared "at all" discards a useful
comparison: whether FD approximates its own declared reference under each schedule. It is the
cross-schedule causal interpretation that needs additional controls.

No new execution is ordered by this review. If attribution is needed, keep width fixed while
inserting the two saved anchors, then keep the anchor fixed across widths, preserving per-projection
vectors. Until those values exist, report the combined difference rather than assigning it to
one bracket. The approved float32 fit remains a defensible operational choice based on its better
measured stability; this narrower interpretation does not reverse it.

## W3 — a short residual-path difference is not an error bar for every lens reading

The new order retains native width-one captures and float32 fits. That creates a declared
cross-path application. The earlier **1.24% at 64 tokens** can accompany it as historical context,
not as a calibrated error term on each reading or at longer contexts.

For a fixed linear lens L, an absolute perturbation bound is

`norm(L * delta_h) <= operator_norm(L) * norm(delta_h)`.

Relative readout error also depends on the denominator `norm(L*h)`, which may be small. A simple
counterexample: h=(100,1), delta_h=(0,1), and L selecting the second coordinate. Residual-relative
change is about 1%, while relative readout change is 100%. The script checks these values.
Nonlinear normalization, token ranking and thresholded dictionary activation add further questions.
A short-run raw-residual percentage cannot answer them. A reading through a float32-fitted lens
on a native residual needs its own paired comparison at the relevant positions and contexts;
until then, mark that application difference unmeasured. This is an interpretation of the declared
term, not a request to promote the native captures or change the approved model precision.

## Progress and remaining evidence

The widened source now performs a full-target no-op comparison and exits on failure before the
VJPs, at each precision and source; that is the intended gate ordering. It also computes output
equality before target summation and stores an individual odd vector per target position, improving
on the original ladder. The scalar files contain per-position odd norms. **These improvements are
visible in source, but their complete as-run evidence is not all committed here.**

`responses.npz` is written by the producer but is absent from this pinned Git record, so its bytes,
completeness and linkage have not been verified in this audit. Moreover, odd vectors alone cannot
reconstruct raw plus/minus/zero values, local spacing or signed input displacements. The source
holds the response dictionary in memory and writes it at the end; intermediate failure can leave
completed scalar rows without their vectors. Preserve each completed unit durably with a hash and
index, and bind it to the corresponding scalar cells. The committed manifest/progress files still
describe earlier stages; archive the width-specific manifests, observed precision settings, frozen
invocations, token digest and gate outcomes with the new runs. Do not infer an execution failure
from the missing committed artefacts.

Width 256 is explicitly unexecuted after a reported allocation failure and is no longer ordered.
It supplies no mathematical comparison. This review neither retries it nor converts its reported
memory failure into a finding about derivatives. No new full map or completed P1–P6 pre-registration
amendment is in the reviewed producer commit; those remain for the next consulting check.

## Verification and reproduction

Run `python3 analyze.py` and then `python3 analyze.py --check`. The standard-library calculation
verifies all width identities, pairs cells by explicit keys, confirms the 648-cell width-one
reproduction, counts each repeated AD projection only once, reproduces the h ratios and AD table,
and verifies three analytic counterexamples. It rejects a byte corruption of each frozen source.
`python3 build_report.py` regenerates this report. No producer script is imported or run; snapshots
are evidence only. See [VERIFICATION.json](VERIFICATION.json) for the final checks and hashes.
"""
(ROOT/'README.md').write_text(report)
print('README.md written')
