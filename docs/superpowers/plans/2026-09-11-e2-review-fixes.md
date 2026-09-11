# E2 review fixes implementation plan

**Goal:** Address F3 and C1–C3 in review 3cb6247 using synthetic CPU regression evidence.
**Target:** Existing cuda-ws-d checkout at dce2401; the review branch contains only reports.
**Specification:** PREREGISTRATION §7, Amendment 1, and the two reports committed at 3cb6247.
**Constraints:** Preserve sealed transport.py, seal/addendum bytes, model admission, fixed folds,
float32 retrieval and tie handling. No real captures, model loading, releases or cost claims.

- [x] Reproduce both immutable review scripts in a source/metadata-only scratch snapshot.
- [x] Add synthetic regressions; observe the multiplicity, population, interval, count, missing
  coverage, false certificate, rank-floor and floating-point parity failures before implementation.
- [x] Reader: sample the union of eligible fit/evaluation episodes once within each sealed stratum;
  retain multiplicities in training and episode reductions; preserve fold exclusion; any failed or
  incomplete draw invalidates the interval without redraw. Compare Hoeffding to the complete
  10,000-refit interval; leave fixed-fit bootstrap descriptive. Refuse unsealed count overrides.
- [x] Certificate: keep Krylov diagnostics explicitly uncertified; use reference full SVD for every
  returned direction. Mirror the sealed fitter's rank stop, finiteness check and deflation order.
  Retract universal numerical identity and speed claims for the former approximate path.
- [x] Verify affected tests and producer witnesses in isolation; inspect the final diff and unchanged
  seal/rule hashes. Deliver local edits with the mandatory fresh addendum/review still required.

The invalid-draw policy is explicit unavailability, not conditional inference from successful draws.
An optimized bypass remains deferred until a global leadingness argument and an equivalence contract
are reviewed. Work is executed in this session; no additional task windows are needed.

## Completion evidence — 2026-09-11

All seven changed files are local, uncommitted edits on `cuda-ws-d`, whose HEAD remains `dce2401`.
`codex/cuda-torch-seam` at `3cb6247` is the review-only branch and remains unchanged.

| Requirement | Correction | Fresh evidence |
|---|---|---|
| F3-A episode multiplicities | Weighted reduction after each episode's transition mean | Original -1/6 counterexample now returns 0; repeated fit rows and fixed fold exclusion verified |
| F3-A fitting population | Union of eligible fitting and evaluated episodes, one draw weight per episode | Ordinary-only singleton enters fit; disjoint fitting/evaluation populations refit successfully |
| F3-B governing pair | Hoeffding versus valid refitted bootstrap; fixed-fit bootstrap stays descriptive | 600-episode .2 contrast now straddles .15; wider-refit direction also tested |
| F3-C required coverage/count | Incomplete draw invalidates entire interval, no redraw; enforce 10,000 at admission and inference | Missing, partial, malformed, nonfinite, short and unsealed-count cases refuse inference |
| Missing capability | Print unavailable fraction and reach named refusal | Main-path negative test passes |
| C1 false certification | Diagnostics always uncertified; public direction always full-SVD reference | Zero-residual, near-degenerate, fitter and padded-width counterexamples require fallback |
| C2/C3 fitter parity | Reference singular-value floor, finiteness refusal and deflation order | Tiny matrix reaches rank zero; coefficients/predictions exactly match in the same NumPy environment |
| Cost projection review finding | Refuse cost estimate when requested refits did not complete | Reproduced the undercount first; both refusal and successful no-reading output paths pass |

Verification:

- Both immutable review scripts reproduced their original FINDINGS before changes, using only
  allowlisted source/metadata snapshots and synthetic NumPy inputs.
- New regression suite: 30 failures and 3 passes before initial fixes; all 33 passed afterward.
  Independent review identified the timing-mode consequence; its new negative test failed before
  correction. Final new suite: **35 passed**.
- Fresh isolated affected suite: **84 passed in 7.71 seconds**, exit 0. Includes all of
  `test_e2_review_regressions.py`, `test_state_transport_certified.py`, `test_state_transport.py`,
  `test_state_read_gate.py` and `test_state_tolerances.py`, including the 1,024/3,840-wide synthetic
  cases. Runtime imports were actively blocked; repository-wide conftest/plugins were excluded.
- Updated producer `check_findings.py`: **8 witnesses passed**, exit 0, on the allowlisted scratch
  snapshot with capture loaders forbidden.
- Independent read-only review: no remaining actionable scoped findings after the timing fix;
  its fresh 35-test run passed. It additionally checked 160 float32/float64 rectangular fitter
  combinations across 20 seeds and ranks 0/1/3/6 against all reference arrays and singular values.
- Ruff passes on both affected test modules and `transport_certified.py`. The two research scripts
  retain only pre-existing findings: reader 16 (baseline 19), witness 5 (baseline 5); no new findings.
  `git diff --check` passes.
- Byte comparisons against HEAD confirm unchanged `transport.py`, `seal.json`, `addendum-1.json`,
  `PREREGISTRATION.md` and `folds.json`. The sealed transport SHA-256 remains
  `6e8b71351cea460991c1a7012be4023c34c81807cc4138fe6ac127201e4cc276`.

No real captures, model/checkpoint execution, actual-data 10,000-resample inference, deployment,
commit, push or release was performed. No speedup is claimed: all draft directions use the reference
fallback, and the old timing projection is withdrawn. Real use of the changed reader requires its
fresh reviewed addendum; section 7 and the separate Research Director cost decision remain held.
