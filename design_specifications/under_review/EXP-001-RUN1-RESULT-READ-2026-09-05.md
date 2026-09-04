To the Chief AI Research Scientist, from the Head of Interpretability. 2026-09-05.
Subject: run 1, the 3B base comparator, read against the pre-registration; and one further
instrument finding that bears on R35 rather than on this table.

Artifact: `outputs/probes/jspace-qwen25-coder-3b-base-2026-09-05/`
(`sweep.json`, `sweep.md`, `run.log`, `events.jsonl`, `provenance.json`).
Status ok, 46:03, 42/42 points, citing `c349739`, a records-only child of `f1230ac`.

## 1. What this run can and cannot decide

EXP-001 asks whether the **4B** holds hidden task state. Run 1 is the 3B comparator, so it
decides none of §2's four rows. It does two things: it establishes whether the instrument works,
and it supersedes the recorded 3B table under the adopted estimator (§5, B1/R35). Both hold.

## 2. Conformance first, as pre-registered

The R34 block is complete and it is what makes the rest readable.

- **Layer index convention stated**, and it is the writer convention: layer L is the residual
  after block L-1, and a probe layer's kind is the kind of the block that wrote it. This was the
  ambiguity that would have produced two different per-kind tables from one run.
- **Roles**: layers 12, 18, 24, 30 primary; 6 and 36 reported, not decisive. The §2 band ruling
  is in force in the artifact, not just in the spec.
- **Both reduction axes named**: summed within a sample over the readout's window, averaged
  across samples, with the note that a skipped sample lowers the divisor rather than entering as
  a zero. C2 as ruled.
- **Windows are real, not structural**: median future window 67.5, minimum 32, over contexts of
  median 136 tokens against a requested minimum of 128. The A1 defect is closed in production.
- **Derivative**: `finite_difference`, source `cli`, which is §5's deliberate override.
- The block also records, unprompted, that `self` is the source paper's self-only limiting case
  and that every J-lens number in this repository before EXP-001 was drawn under it. That is
  R34 doing more than it was asked.

**One deviation, recorded honestly by the artifact itself:** `capture_dtype` requested `native`,
**effective `float32`**, `view_supports_dtype: false`, because `ArchitectureView` casts every
block output to float32. §3.2 pre-registers native capture. It is not achieved, and it cannot be
through this view. See §4.

## 3. The reading

**The instrument works.** The positive control passes overwhelmingly. It is not labelled as such
in the artifact and must be read as the complement of the primary comparison: at layer 24 the
true (hidden) suffix is preferred 5/42 in matched context against 19/42 in mismatched, so the
already-read suffix, which *is* in context, is preferred 37/42 (88%) matched against 23/42 (55%)
mismatched, at p = 4.4e-7. The recorded table's control was 36/42 (86%) at p = 2e-6. The lens
detects an in-context filename, strongly, through the new estimator.

**World A holds on the 3B base.** The decisive measurement is the model's own next-token
distribution (§2, as ruled): 16/42 (38%) matched against 15/42 (36%) mismatched, p = 0.164. At
chance, and the two columns agree, which is the pre-registered World A signature. No primary
layer beats its null in the direction of the true suffix under Holm. The significant primary rows
are all in the *opposite* direction, which is the control showing through the same comparison.

**It reproduces the superseded table closely.** Layer 24 `self` is 5/42 tonight against 6/42
recorded, mismatched 19/42 against 17/42. A different policy, a different derivative method, a
different estimator, twice the corpus and three source positions, and the cell moves by one case.

**What that licenses.** Nothing about the 4B. It licenses the 3B arm of the comparison, retires
the recorded table to historical record, and clears the instrument for run 2. The decision memo's
World A footnote (§6 answer 3) can now name the estimator: the conclusion does not move.

## 4. The further instrument finding: the float32 gap is model-dependent and unrecorded

§3.2 requires the preflight's float32 deviation block **copied into the artifact**. The slot
exists in the R35 comparability block as `fp32_manual_vs_native` and it is **null**. The numbers
are in the preflight artifacts and they are not alike:

| | 3B (native float16) | 4B (native bfloat16) |
| --- | --- | --- |
| Frobenius relative error | 0.0042 | 0.0285 |
| Max relative error | 0.0033 | 0.0547 |
| native epsilon | 0.00098 | 0.0078 |

The float32 capture path sits about seven times further from the 4B's native forward than from
the 3B's, because bfloat16's epsilon is eight times float16's. Both are within their own
tolerances and neither gates, which is R18a working as ruled. But three consequences follow.

1. **R35's comparability block would call the two tables comparable on this axis when they are
   not.** Two artifacts each carrying `fp32_manual_vs_native: null` compare as equal. Populate
   the field from the preflight record and the asymmetry becomes visible instead of implied.
2. **The decisive measurement rides on the float32 path.** The model's own next-token
   distribution, which §2 makes decisive, is computed through a forward the deployment never
   runs. On the 3B that is a 0.42% Frobenius perturbation; on the 4B it will be 2.85%.
3. **The guard is the positive control, and it held here.** A path that still detects an
   in-context filename at p = 4.4e-7 is not a path that has destroyed the signal we are looking
   for. I would let the control carry it on the 4B too, rather than re-engineer the view.

**Requested, none of it blocking run 2:** populate `fp32_manual_vs_native` from the preflight
record in every artifact; amend §3.2 to say the capture is float32 through `ArchitectureView`
with the native request recorded and unmet, rather than claiming native; and state in the reading
of the 4B that the decisive measurement was taken through a path 2.85% from its native forward,
with the control as the bound on what that could hide.

## 5. On the two findings already ruled

The final-layer structural zero (#68) I found independently of the Deputy and agree with the
ruling as dispatched: raise in `jlens_readouts` at L = num_layers for `future` and `all`, compute
only `self` and the logit lens there, record the exclusion, and drop that layer from those two
Holm families. In this artifact it cost nothing, because layer 36 is *reported* rather than
primary, but its p-value of 4.5e-13 sits in the table looking like a result.

The hybrid-period accessor is fixed at your gate. Run 1 is unaffected: a dense backbone has one
block kind, so `hybrid_period: unavailable` is the correct reading there and the six-layer family
is right for this model.

## 6. Verdict

**Run 1 is valid and I accept it as the 3B arm.** The instrument is sound, World A holds on the
3B base under the adopted estimator, and the recorded table is superseded rather than contradicted.
Run 2 should proceed on the corrected code, after the 3B is rerun under it so both tables come
from one instrument, as ruled.

— Head of Interpretability

---

## Addendum: the 3B base rerun on the corrected instrument (2026-09-05, 06:06)

Artifact: `outputs/probes/jspace-qwen25-coder-3b-base-2026-09-05-rerun/`. Status ok, 46:03,
42/42, citing `927807b`, a records-only child of the fix `2bb2761`. Run 1's artifact is untouched
and remains the record of the first instrument.

**This is an instrument validation, not a second result.** The reading of §3 above stands
unchanged and is not re-argued here.

**Every scored cell is identical to run 1 except two**, across six layers and three readouts.
Diffed cell by cell from `sweep.json` rather than from the rendered table, which hides the first
of them at three decimals. Two rows are absent from the rerun, `jlens_L36_all` and
`jlens_L36_future`, and exactly two values moved:

| | run 1 | rerun | ratio | Holm rank |
| --- | --- | --- | --- | --- |
| `jlens_L24_all`, Holm-adjusted | 0.00041263 | 0.00034386 | 1.200000 | 1 |
| `jlens_L30_all`, Holm-adjusted | 0.09760236 | 0.07808189 | 1.250000 | 2 |

Both movements are in the `all` family and both are exactly the factor their own rank predicts.
Holm multiplies the k-th smallest p by m-k+1, so a family shrinking from six members to five
scales rank 1 by 6/5 and rank 2 by 5/4. The observed ratios are 1.2 and 1.25 to ten decimal
places. The `self` family kept `jlens_L36_self`, which is a real measurement, and did not move at
all. The correction found by the Deputy and verified by the Chief and by me: I first reported one
movement, having read the rendered table where `jlens_L24_all` is 0.000 in both runs.

So issue #68 was **not cosmetic**, and it inflated the correction on **two** primary layers
rather than one. In one of them it did not matter, since `jlens_L24_all` was already far past any
threshold. In the other it is the difference between two readings a person would describe
differently: 0.098 and 0.078 sit on opposite sides of a conventional line, in the direction of
understating an effect. A structural zero was buying every other member of its family a laxer
correction than it had earned.

**The sweep is deterministic across runs.** Twenty-three rows reproducing exactly, on a
quantised model through a finite-difference derivative over sixteen contexts and three sources,
is worth stating because it bears on how the 4B table may be read: no difference between the two
models' tables will be run-to-run noise, and no agreement between them will be luck.

**The three repairs are visible in the artifact, not merely claimed.**

1. The dense model's absent period is now a sentence: `None (from view.blocks: no
   linear-attention block, so the backbone is dense and has no hybrid period)`, where run 1
   carried a bare dash. The field reports why rather than that.
2. The final-layer exclusion is recorded with the layer, what was computed there (`self` and the
   logit lens), what was excluded (`future`, `all`), the reason naming the position-wise tail,
   and the consequence for the Holm families in its own words. A reader coming to it cold does
   not need to know issue #68 to understand why two cells are absent. Structurally confirmed:
   the only layer-36 rows in the artifact are `jlens_L36_self` and `logit_lens_L36`.
3. `fp32_manual_vs_native` carries the real block, Frobenius relative 0.00424 and max absolute
   0.470, where run 1 and every probe artifact since `6f84217` carried `null`. The §4 finding
   above is closed for artifacts written from here on; it remains true of every artifact written
   before this commit.

**One stale string, non-blocking, same class as #71.** The `family derived` reason still reads
"no hybrid period in the model configuration (`full_attention_interval` absent)" while the
`hybrid period` line above it correctly cites the structural route. Both are true of a dense
model, but the reason attributes the absence to the configuration when the blocks are what
answered, and after this week it will read as though the walk that failed is still in charge.

**Verdict.** The corrected instrument reproduces the first instrument exactly on every cell it
still computes, removes two cells that were never measurements, and corrects one Holm value as a
direct consequence. The 3B arm is settled and is the comparator for the 4B under R35.

— Head of Interpretability
