# Amendment 1 revision 5 re-check

**PASS — target `1d936087ca944c4934c9bfec7185005f302209fa` on `cuda-ws-d`.**
The four findings against revision 4 at `615116f` are closed in this checked set. This clears
the requested file-only re-check before the D-CRO's addendum seal; it neither creates that seal
nor licenses a reading before the remaining gate sequence.

| Checked set | Verdict and fresh evidence |
|---|---|
| R5 fitter and producer tests: `test_state_transport.py`, `test_state_decodability.py`, `test_state_read_gate.py` | **PASS**, 43 tests, pytest exit 0, 1.19 seconds. Exact target blobs in isolated temporary packages; model-library imports blocked. |
| Regression discrimination: R5's 15 transport tests against `b0cd61c` | **PASS as a negative control**: 7 failed, 8 passed; pytest exit 1, 0.24 seconds. The failures detect the old implementation rather than merely mirroring the new one. |
| `check_amendment.py`, `check_prereg.py`, and each `--self-test` | **PASS**, all four processes exit 0. Independent insertion of withdrawn and current capability figures into §7 is rejected. |
| Original seal and E1 committed-byte identity | **PASS**: 13 sealed files match their seal hashes and baseline `acc130a`; M remains 12. E1 reader, scalar fitter, readings and summary match `5b1c9a1`. |
| Amendment §§2,5,7,10 and `measure_gate.py` / `gate-table.json` | **PASS for the scoped re-check**: source inspection, table arithmetic, and seal/fold identity checks described below. No capture arrays opened. |

F1: full SVD replaces the start-vector power iteration, with a fixed sign and finite guards.
The zero-start and certified-non-leading witnesses now produce finite, leading directions.
The direct successor-map family and deflation algebra remain the declared form.

F2: the restart witness now discriminates. The suite also covers a nested 32-component ladder,
full-rank agreement with least squares, narrower targets, and non-finite-input refusal.

F3: §7 refers to §5's current capability table and contains no copied capability number.
The checker rejects both current and withdrawn three-decimal literals planted in §7.

F4: the bound is explicitly restricted to the same rows, depth, rank, transition cost and
averaging weights. Other cohorts and depths receive a capability reference, not a ceiling or
a null. The mandatory report is retained.

The saved SVD gate table binds to the original seal and assignment. All twelve fractions are
integer counts over 4,122 and round to §5. At headline rank, 4B has 3,170 / 4,122 and 12B has
3,097 / 4,122; both exceed one half and both report no short folds. The measurement CLI calls
`require_seal` before capture access; the seal fixes `folds.json` bytes and the assignment
digest is checked before fitting. No amended corrective reader was executed in this review.

The original seal SHA256 remains
`998b3bcafa9d6aaffa21ebd43df3935b1634ba7b12fe187073837acd942ce521`.
E1 readings remain
`ba841d7c0601b6326c621c68701852140e92e931ced570cb2ca2dd03d36f303f`.

Evidence limits: the D-CRO's fresh card remeasurement and E1 rerun are producer attestations;
this review verifies their committed outputs/identities, not a separate execution. The exact
maximum old-versus-new difference of 0.00049 cannot be recomputed from the rounded old table;
agreement at every published three-decimal cell can. The prior two disclosure clarifications
about “nothing has run” and inferring corrective behavior from ordinary rows remain wording
caveats, not reopened F1–F4 findings or a reversal of the Chief's reuse ruling.

`checks.json` records target source hashes, commands, process exit codes, negative-control
failures and byte checks. The six adjacent logs contain the direct process outputs. Testing
used the existing laptop Python, NumPy and pytest; no dependency install, checkpoint load,
model-library import, real capture-array read, card call, production edit or merge occurred.
