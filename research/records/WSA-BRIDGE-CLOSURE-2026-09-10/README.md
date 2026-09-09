# Bridge identity review closed on metadata fixtures

**Verdict: C1 is closed at `a2c877e65f3ce1079af5bb5cc01b060e87184024`, integrated at `06535be`.** The guard now binds a calibration to the lens archive's verified content hash as well as its fitting declaration and the other pairing fields. The corrected ruling is `93f227f`. No additional finding or ruling is requested by this review.

The exact failure case from `c55752f` now produces the required result through the pinned `hook_alignment` entry point:

| Case | Observed result |
|---|---|
| No measurement registered | Refused |
| Calibrated lens A with its original declaration | Accepted |
| Different lens B, identical declaration | Refused on `lens_sha256` |
| Original lens A, changed declaration | Refused on `nu_sha256` |
| Loaded lens stand-in with no archive hash | Refused |
| Old registration lacking an archive-hash field | Refused |
| No-capture A1 | Accepted |
| Declared same-precision/same-width path | Accepted |

All eleven pairing fields were also changed and omitted independently: every variant was refused. The archive hash comes from the lens object in the actual entry point, not from the reading dictionary. This closes the transfer to a different lens archive without weakening the declaration comparison or the existing no-capture path.

**Basis and limits:** these are executed metadata fixtures using two small synthetic lens stand-ins, not model measurements. The matrix hashes identify the example matrices' JSON bytes; no NPZ archive or checkpoint was loaded. The shipped table still has zero measured pairings. This review does not establish real lens/capture compatibility, approve A2, or certify a full native test suite. The earlier B1–B3 fixes were verified in `c55752f`; this pass is limited to the newly added artifact-identity distinction and its surrounding controls.

## Transferable result and technique

There is no new model finding. The instrument issue is resolved: a calibration cannot transfer to another artifact merely because its fitting declaration is equal. The verification uses complementary controls—change only the artifact, then change only the declaration—to establish that both identities matter. Missing-identity checks establish the refusal direction. This technique applies to any calibrated instrument whose recipe and resulting artifact have separate identities.

## Reproduction

`python3 check.py > check.json` reproduces the metadata results. `python3 verify.py`, with `ruff` on PATH, checks reproduction, relocation, source corruption refusals, source identity against Git, and lint; its result is `verification.json`.

`sources.json` pins five inputs. The previously verified metadata extractor is copied unchanged from `c55752f` and reused; its old audit entry point is not executed. It extracts the fourteen inspected function bodies from the new bridge source, with production/model imports guarded against. Only canonical model-name resolution is replaced by a fixture that returns the already-canonical id. Producer modules and test suites are never imported.

No model forward, checkpoint load, native pytest, remote operation, pairing registration, or producer-file write occurred. The preregistration re-review still awaits the D-CRO's complete P1–P6 amendment; other device work remains in its existing queue. This closure commit is ready for review.
