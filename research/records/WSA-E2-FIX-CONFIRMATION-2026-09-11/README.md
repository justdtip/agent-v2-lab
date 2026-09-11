# Confirming review of the uncommitted E2 fixes — PASS

**PASS for the scoped F2/F3 and C1–C3 corrections from review `3cb6247`. No remaining actionable finding was identified in the reviewed bytes. This is a source/fixture verdict, not an addendum, inference release or cost authorisation.**

The reviewed implementation is the **uncommitted** state of `/Users/daniel.tipton/worktrees/cuda-ws-d` based on `dce24012e36f437400c25f0318a52ae3a35da5ad`. That base commit alone does **not** contain these fixes and is not the approved identity. `reviewed-files.json` records the SHA-256 of each of the 40 allowlisted source, test and metadata files read. Their hashes were checked before and after verification; no reviewed file changed during the pass. A subsequent implementation commit must preserve those bytes for this verdict to apply.

Principal identities:

| File | SHA-256 |
|---|---|
| `read_e2.py` | `43459950801d6f750d5a29b1f8b5cb04ea80f54613db106c40b98da471d8697a` |
| `transport_certified.py` | `5e90554a787d02626f2bacaf4e9192ce5f6c1238a0e433069375692a1b1b1d0a` |
| `AMENDMENT-2-CERTIFIED-DIRECTION.md` | `76509aab11cc0d924df766a657efb3fe4e0901d0646377456d4838901087efcd` |
| unchanged `transport.py` | `6e8b71351cea460991c1a7012be4023c34c81807cc4138fe6ac127201e4cc276` |

## Requirement-to-evidence result

| Finding | Confirmed behavior | Result |
|---|---|---|
| F2 — model pair and gate coverage | The budget supplies the pair; missing, unexpected and empty model populations refuse. Either failed gate and incomplete coverage prevent scoring. An unavailable capability now reaches the named refusal rather than failing during number formatting. | PASS |
| F3-A — population and weights | Fitting and evaluation episodes are sampled jointly within `(split, family, variant)`. The same multiplicity repeats eligible training transitions and weights the episode's mean score. Fixed episode-fold exclusions remain in force. | PASS |
| F3-B — governing interval | The required comparison is Hoeffding versus the refitted bootstrap. The fixed-fit bootstrap is labelled separately and no longer governs. The interval's position relative to epsilon determines the reading. | PASS |
| F3-C — incomplete or absent inference | An empty, failed or incompletely scored draw invalidates the entire interval without discarding or replacing that draw. Missing, malformed, nonfinite, short or incomplete refit intervals cannot produce a registered inference. Unsealed count overrides refuse. | PASS |
| Registration and timing consequences | `ordinary_test` stays descriptive at headline coordinates, and the other non-headline cells are exploratory. Timing mode refuses incomplete work before projecting a cost and writes no reading on its successful fixture. | PASS |
| C1 — false leadingness certification | Krylov candidates always remain uncertified. Every public direction uses the unchanged reference full-SVD function, including the old zero-residual/non-leading and near-degenerate witnesses. | PASS |
| C2 — stopping and arithmetic | The singular-value floor, score-norm floor, finiteness refusal and deflation operation order match the reference. The tiny-cross-product witness now retains rank zero. | PASS |
| C3 — equivalence and cost claims | Checked same-environment fitted arrays, singular values, coefficients and predictions match exactly. The draft withdraws the accelerated bypass, the old speed projection and universal cross-environment identity claims. | PASS |

## Fresh execution

The five affected test modules ran in an isolated source/metadata snapshot, with model-runtime imports blocked and repository-wide conftest/plugins excluded:

- `test_e2_review_regressions.py`
- `test_state_transport_certified.py`
- `test_state_transport.py`
- `test_state_read_gate.py`
- `test_state_tolerances.py`

**84 tests passed in 5.98 seconds, exit 0.** This includes the larger 1,024/3,840-wide synthetic cases, which the previous review left unexecuted. The updated producer's **eight reader witnesses passed**, exit 0. The changed reader itself still refuses the real record's superseded-only addendum state before accessing even a nonexistent captures directory. The parent seal's 13 baseline file hashes were independently rederived from Git.

I also checked resampling against a separate numerical reduction: 32 draws, several strata, two fixed folds, unequal transition counts per episode, duplicated draws, and episodes eligible only for fitting. The oracle uses the unchanged reference fit, independently performs retrieval and takes each episode's transition mean before applying its draw multiplicity. Its endpoints `[-0.16666666666666663, 0.25000000000000006]` match the reader's `[-0.16666666666666666, 0.25000000000000006]` to floating-point rounding. The endpoint convention is the sealed helper's one-sided alpha and one-minus-alpha, **not a newly claimed two-sided 95% interval**.

The separate certificate check executed eight direction controls and eighteen full-fitter fixtures. Each public direction called the reference exactly once. The fitter checks included float32/float64 inputs, rectangular targets, constant coordinates, rank deficiency and the old rank-floor counterexample. All retained ranks, singular values, stored arrays, coefficients and predictions matched the reference in the same NumPy environment. Injected diagnostic numerical failures and actual finite-input overflow in the diagnostic fell back; non-finite reference inputs refused.

The independent calculation and certificate checks were rerun from their integrated review scripts. Results and outputs are in `evidence.json` and `certified-evidence.json`. Python 3.13.7; NumPy 2.5.2. These are fresh checks, not adoption of the implementer's reported test result.

## Reproduction

From the repository root, with a source checkout containing the reviewed bytes:

```sh
.venv/bin/python research/records/WSA-E2-FIX-CONFIRMATION-2026-09-11/verify.py --source /Users/daniel.tipton/worktrees/cuda-ws-d
.venv/bin/python research/records/WSA-E2-FIX-CONFIRMATION-2026-09-11/check_certified.py --snapshot /Users/daniel.tipton/worktrees/cuda-ws-d --out /private/tmp/e2-certified-confirmation.json
```

Both scripts verify their input identities against `reviewed-files.json` and use synthetic inputs. `verify.py` copies its allowlist into temporary storage before executing checks; the certificate script reads its named modules into memory. A changed source identity must receive a new review rather than editing the manifest to make the check pass.

## What this PASS does not release

No residual captures, model checkpoints or real-data estimands were opened; no device job, actual 10,000-refit inference, deployment or push was run. The implementation files remained uncommitted and untouched by this review. The review branch receives only this report, the source-hash manifest and reproductions/evidence.

The sealed pre-registration, folds, `transport.py` and existing seals/addenda remain unchanged. The reader still uses `transport.py`; the draft diagnostic wrapper is not an accelerated production fitting path. There is no replacement speedup to claim. A fresh reviewed addendum covering the committed reader bytes and the Research Director's separate cost ruling are still required before real use.
