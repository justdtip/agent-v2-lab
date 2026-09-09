# Bridge review: a measured layer is not a measured pairing

**Changes requested before the first cross-path pairing is registered.** This file-only review covers `c4f2f9f1a4f6cde4cfcc1105eafc248859c0ba88`, integrated at `80857a2`, against the corrected WS-D order at `9114d66`. It executes only six inspected, pure metadata functions and one isolated metadata test extracted from the pinned source. It imports neither the production module nor model packages, and touches no device.

The empty shipped pairing table correctly refuses declared cross-precision or cross-width readings, including the last sampled layer. Same-dtype/same-width metadata and A1's no-capture case remain allowed. **There is no evidence here of a current incorrectly permitted A2 run:** real A2 and every pairing measurement remain unexecuted. The defect is in how the refusal will lift when a measurement is added.

The previous documentation follow-up is now addressed in `dc21d58`; the Chief's order also marks superseded statements at their original sections. No further cleanup commit on that follow-up is requested here.

## B1. Pairing lookup omits the pair's identity (P2)

In [sae_bridge.py](source/sae_bridge.py), `path_pairing` looks up only `base` and `layer`. `path_term_for_layer` treats any nonempty dictionary at that key as measured. `fit_precision_record` then accepts the crossing whenever `term['measured']` is true. The acceptance does not compare the measurement's fit/capture precisions, forward widths, lens identity, positions, or context with the proposed reading.

A synthetic measurement scoped to a float32 width-one fit and a bfloat16 width-one capture at one layer gives these results when passed through the unchanged metadata functions:

| Proposed reading | Actual guard result | Required scope interpretation |
|---|---|---|
| Declared crossing with the shipped empty table | Refused | Correct |
| Same precision/width, or no capture | Allowed | Correct within the fields this check covers |
| Matching synthetic pairing | Allowed | Positive control |
| Change capture precision to float16 | **Allowed** | Different pairing, unmeasured |
| Change capture width from 1 to 64 | **Allowed** | Different pairing, unmeasured |
| Reverse the fit/capture precision direction | **Allowed** | Different pairing, unmeasured |
| Use the adjacent unmeasured layer | Refused | The existing layer check works |
| Replace the measurement with only `{'relative': 0.004}` | **Allowed** | No pair identity exists to compare |

**Basis:** metadata fixtures, not numerical experiments or actual permission grants. The synthetic measurement contains explicit scope solely to demonstrate the failure to bind it; its field layout is not a proposed new production convention. The existing upstream test itself inserts a measurement containing only a relative number and expects acceptance.

The rule is that the **intended pairing** is measured under its own reduction, population, and precision. Model-plus-layer is necessary but insufficient: the point of this week's controls is that a reading at another precision or schedule need not be the same object. Context, endpoint/reduction and readout identity are not even inputs to the current guard, so those dimensions cannot currently be checked by it.

**Requested change:** bind a registered measurement to the actual fit and capture contracts before using it to lift the refusal. Reuse the existing ν, artifact identity, and position/context metadata rather than inventing another naming convention. Compare both sides of the requested pairing, and keep incomplete or unmatched measurements unmeasured. This does not introduce an accuracy threshold or change the ruling that a measurement is required; it makes the measurement refer to the reading being permitted. Preserve the explicit no-capture A1 path.

Acceptance should mutate each bound field independently: a matching fixture is accepted, while changing either precision, either relevant width, lens/checkpoint identity, context/positions, or reduction/endpoint leaves that distinct pairing unmeasured. A measurement for one layer must continue not to transfer to its neighbour. No device time is needed for these metadata tests.

## B2. The “no amplification ratio anywhere” test misses nested ratios (P2)

The new test `test_no_amplification_ratio_survives_anywhere_in_the_shipped_measurements` checks for `amplif` only in the immediate keys of each layer object. Its numerical scan iterates only the top-level model values, which are dictionaries. It never descends into the measurement fields.

The isolated, unchanged test passes after this synthetic corruption:

```text
layers["1"]["sampled_displacement"]["float32"]["amplification_ratio"] = 539
```

It correctly fails if the same key is inserted directly under `layers['1']`. That is the negative control: the test runs and has an assertion, but that assertion does not cover the nested location where measured quantities live. The shipped table contains no such ratio; this finding concerns the regression test's reach.

**Requested change:** validate the table's allowed structure/quantities recursively and corrupt each supported nesting location in the test. Do not ban all numbers or blacklist the number 539: legitimate medians, ranges, and counts are numeric, and a legitimate statistic can coincidentally equal a withdrawn ratio. The quantity's meaning and location are what must be checked. The claim that “no bare number is reachable anywhere” should describe the actual schema, not the vacuous top-level scan.

## B3. One copied median has inconsistent rounding (P3)

The table is otherwise consistent with the preserved scalar data at its written precision: 59 of 60 numerical fields match, including counts. At repo layer 33, native precision, `equal_norm_random_displacement.median` is written as **0.313000**. Recomputing from the 18 scalar responses gives **0.31295347134393925**, or **0.312953** to six decimal places. If three decimal places are intended, write and declare that precision consistently. This small discrepancy does not affect the current refusal decision.

**Basis:** all 60 fields recomputed directly from the 108 source scalar rows; the literal decimal formatting is retained during comparison. The copied table is context for the refusal, not a calibrated error bound or a measured lens/capture pairing.

## Finding, technique, implementation

There is no new model finding in this review. The instrument findings are the overbroad future measurement lookup, the nested mutation missed by its regression test, and one copied-summary rounding inconsistency.

The transferable technique is to test **non-transfer** as well as successful lookup. A calibration result is evidence about an identified experiment. Holding the layer fixed while changing the experiment should not inherit that result. A positive test of one valid pairing and a negative test of another layer cannot establish this property; the pair's other coordinates must be changed too. Likewise, a recursive scientific record needs mutations below its first level, not just at the layer boundary.

Our implementation of the review uses Python's syntax tree to extract only the six named metadata functions and the one named JSON-only test. No producer import tree or test suite runs. All inserted measurements and corruptions live in deep-copied in-memory dictionaries; no source table or run record is changed. `analysis.json` contains the exact outcomes and per-field numerical checks with their basis. The findings do not imply that a guard supplied with no capture metadata validates a real capture; A2's eventual caller remains responsible for supplying its actual provenance.

## Reproduction and scope

Run `python3 analyze.py > analysis.json`, then `python3 verify.py` with `ruff` on PATH. `sources.json` pins seven source blobs to their commits and hashes. Verification checks relocation/reproduction, hash refusals, the positive and negative metadata controls, each copied numeric field through mutation, and syntax/lint without model imports.

This review does not execute a device run, a model load, a native test suite, a full-module import, or real A2. It does not modify the bridge implementation or approve a new scientific tolerance. The preregistration re-review still waits for the producer's complete P1–P6 amendment. The commit is ready for review by the bridge owner.
