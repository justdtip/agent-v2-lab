# Bridge fix verification: declaration identity and artifact identity are different

**The ruled fixes pass the metadata checks; one refinement is needed before a real pairing is registered.** Review of `38c6e488bd0478920d8802608cd4876b392b30bb`, merged at `c633510`, against the ruling at `035d3e6`. No model or device operation was performed.

The implementation now compares every one of the ten named pairing fields, refuses incomplete readings, validates nested sensitivity quantities, and fixes the copied median. However, the ruling's use of the lens's ν digest does **not** identify its actual matrix archive. The new code implements that instruction literally; the instruction needs the existing artifact hash beside the declaration hash if the measurement is to apply to the same lens.

## Closed findings from the preceding review

| Check | File-only verification |
|---|---|
| B1, registered pairing versus proposed reading | Matching fixture accepted; each of the ten declared fields independently changed and refused |
| B1, absent identity fields | Each of the ten fields independently removed from the proposed identity and refused |
| B1, entry-point propagation | Different capture precision, width, context, adjacent layer, and missing reading provenance refused through `fit_precision_record` |
| B1, valid unaffected cases | No-capture A1 and the declared same-precision/same-width case still accepted |
| B2, nested quantities | Six corruptions across model, layer, arm and scalar-summary levels refused by the new validator |
| B3, copied values | All 60 fields match the 108 original scalar rows at the newly declared rounding precision |

These are executed metadata fixtures, not a native suite or a model measurement. The pairing table shipped in the reviewed commit is still empty. Nothing in this review establishes that a real cross-path reading has occurred or has been calibrated.

## C1. ν identifies the fitting declaration, not the resulting matrix archive (P2; ruling refinement)

The distinction is visible in the existing source:

- [`nu_digest`](source/sae_bridge.py) hashes JSON supplied as `nu`. `reading_identity` includes that digest in the pairing key, but it receives no lens archive hash.
- [`declare_nu`](source/upstream.py) describes estimator, endpoints, positions, weighting, corpus, precision and provenance. It does not digest the resulting matrix entries.
- [`read_declared_nu`](source/upstream.py) correctly checks that a sidecar belongs to its adjacent archive through `npz_sha256`, then returns the ν block alone. Two separately authenticated archives can still carry equal ν blocks. The sidecar-to-archive check does not make the declaration unique to one archive.
- [`LensMaps`](source/instruments.py) already exposes the loaded archive's verified `sha256`. [`hook_alignment`](source/sae_bridge.py) receives this lens object but checks its base, layer and width before passing only ν and reading metadata onward. The archive hash is not used for pairing lookup.

This does not require a hash collision. Equal declarations produce equal declaration hashes by design, even when the result arrays differ.

### Concrete counterexample through the entry point

The review supplies two two-dimensional synthetic lens stand-ins with identical base, shape and ν, but different matrices and content hashes:

```text
J_A = [[1,     0],       J_B = [[0.001, 0],
       [0, 0.001]]              [0,     1]]
```

A single synthetic registered pairing matches their common declaration. **Both objects pass the unchanged `hook_alignment` entry point.** Changing ν while leaving the second lens in place makes the guard refuse, proving that the ν comparison is active and the matrix identity is the missing distinction.

To show why that distinction matters mathematically, use the same residual and change for both:

```text
h = (100, 1),  Δh = (0, 1)
relative readout change = ||J Δh|| / ||J h||
```

For `J_A`, the ratio is about **0.00001**. For `J_B`, it is about **0.99504**. A calibration of a readout through one linear map need not apply to another map with the same input and output dimensions. These values are an analytic example, not observations about Gemma, a fitted lens, or a chosen tolerance.

The fixture hashes identify the JSON-encoded example matrices. They are stand-ins for real archive hashes, not claims to have loaded or generated NPZ lenses. The entry point uses the actual pinned function body; only canonical model-name resolution is replaced by an identity-preserving fixture function so no registry/model package is imported.

### Requested refinement

Bind the registered pairing to the **existing verified lens artifact hash**, while retaining the ν digest and the other ruled fields. Obtain the hash from the loaded `LensMaps` object (and keep the established sidecar binding); do not ask the caller to assert that two arbitrary matrices are the same lens. A different archive is a different identity unless an explicit, separately verified equivalence rule says otherwise.

Acceptance: register a pairing for lens A; accept A; then present lens B with equal ν but a different archive hash and refuse B. The existing test that a changed ν is refused stays as a distinct check. This adds no numerical threshold and requires no device time. It distinguishes the specification of a fit from its result, rather than weakening any newly added comparison.

The current table remains empty and real A2 remains unexecuted. This refinement is for the first real registration; it does not demand a change to an active experiment. The literal ten-field ruling is implemented, so this is a refinement of the identity contract, not a claim that those ten comparisons still fail.

## Finding, technique and implementation

There is no new model finding. The instrument finding is that a calibration keyed by fitting metadata can transfer to a distinct matrix artifact. The technique is to hold the declaration fixed while changing the result, then reverse the control by changing the declaration. A test that changes only metadata cannot detect this class of identity gap.

`analyze.py` extracts fourteen inspected functions by syntax tree and executes only metadata call paths. A guard raises if a production/model package import is attempted; none is. The only model-name-resolution replacement returns the fixture's already-canonical base id. No full bridge module, test module, checkpoint, native pytest suite, or model forward is imported or run. The numerical example uses standard-library arithmetic on two small lists.

Seven source files are pinned by commit and hash in `sources.json`; `analysis.json` carries outcomes, source identities, numerical bases, and unexecuted work. Run `python3 analyze.py > analysis.json` and `python3 verify.py` with `ruff` on PATH. Verification covers reproduction, relocation, changed-source refusal, the declared-field controls, nested corruptions, all copied numerical fields, and import restrictions. The complete preregistration amendment is still pending, so no seal review is implied.

This record is ready for review by the Chief and bridge owner. No producer source or record was modified.
