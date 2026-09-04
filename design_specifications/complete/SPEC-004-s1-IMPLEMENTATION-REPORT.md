# SPEC-004 §1 remediation implementation report

Date: 2026-09-03
Scope: GitHub issue #3, ratified corrections C1–C6 under rulings R7–R8
Base named in the assignment: `60dfd51f743505b8bb5af85d1dd7f4179ea1667a`
Integrated checkout HEAD before this implementation: `6325d3aedcaee075b9c7e6207b877a6234119ae7`

## Status

`DONE_WITH_CONCERNS`. C1–C5 are implemented and C6 completed using the permitted CPU-only
offline command. The two ratified within-position readings reproduce exactly after rounding,
but the ratified Holm-support list materially disagrees with the regenerated reportable
`all_rows` cohort. The generated artifacts were retained without altering the statistics.

## Ratified sources and scope

Before product edits, the implementer read in full:

- `design_specifications/under_review/SPEC-004-s1-REVIEW-round1-2026-09-03.md`;
- `superpowers:receiving-code-review`;
- `superpowers:test-driven-development` and `writing-good-tests.md`;
- `codex-engineering-guardrails:code-work`;
- `superpowers:verification-before-completion`.

The pending specifications and ratified review remained read-only. D3a, D3b, and D4 were not
implemented because they are explicitly unratified. No model, tokenizer, model checkpoint,
training, evaluation, rollout, selection, branching, preference, preflight, GPU inference, or
MLX model execution occurred.

## Files and line counts

| File | Result | Current lines | Working diff at evidence capture |
| --- | --- | ---: | ---: |
| `src/local_llm_lab/probes/state_probe.py` | C1–C4 implementation plus gap-16 helper rename | 2,922 | +147 / −13 |
| `tests/test_probes.py` | historical probe tests plus existing-test C1/C3/C4 assertions | 2,883 | +60 / −8 |
| `tests/test_reanalysis.py` | R10-focused remediation and gap-16 tests | 62 | +62 / −0 |
| `outputs/probes/state-corrected-hardened-20260903T172549/state-base-mix.reanalysis.json` | regenerated C6 artifact, intentionally uncommitted | 520,409 bytes | generated |
| `outputs/probes/state-corrected-hardened-20260903T172549/state-base-mix.reanalysis.md` | regenerated C6 artifact, intentionally uncommitted | 80,562 bytes | generated |
| `design_specifications/under_review/SPEC-004-s1-IMPLEMENTATION-REPORT.md` | this C5 evidence report | 227 | new |
| `.superpowers/sdd/2026-09-03-spec-004-s1-remediation/progress.md` | C5 evidence entry only | 22 | append only |

## C1–C6 mapping

| Correction | Implementation and evidence |
| --- | --- |
| C1 / R8 | Each within-position `overall`, difficulty, and family cell now records `estimate`, bootstrap `median/lower/upper`, `n_test`, `n_cells`, and `eligible`. Counts are the conservative minima across contributing split fits. Eligibility requires at least 24 test rows and 5 eligible `(family, step)` cells. Markdown prints `n/a (n=…)` for ineligible or undefined cells and never prints their numeric estimate. It prints the pooled `overall` row. The point is the mean of the five full-fit split scores; intervals retain the existing task-ID cluster bootstrap restricted to the eligible within-position rows. |
| C2 | Markdown defines `Holm support` as requiring the Holm-adjusted 0.05 threshold plus both paired intervals excluding zero, and explicitly names `margin vs position` and `margin vs surface`. |
| C3 | JSON metadata records measured `elapsed_seconds` and `model_spec` in the unresolved data-stage `asdict(ModelSpec)` form. Registry YAML is selected from the captured model reference; no resolved spec or model runtime is loaded or invented. |
| C4 / R7 | Both JSON analyses carry the exact labels `all_rows (reportable for base)` and `sft_disjoint (paired adapter comparisons, within difficulty only)`. Each analysis records and Markdown prints a rows/tasks table by difficulty. Exclusion remains only the 737 `train-` rows; `p2mix` is not excluded. Base conclusions are sourced from `all_rows`. |
| C5 | This report and one append-only SDD evidence entry record the implementation and verification. |
| C6 | The corrected code regenerated exactly the authorized JSON/Markdown pair from the saved NPZ using the CPU-only command below. |

Priority-amendment conformance:

- R10: only the two newly introduced C1/C2 remediation test definitions moved from
  `tests/test_probes.py` into `tests/test_reanalysis.py`; historical SPEC-004 tests and their
  fixture remain in `tests/test_probes.py`. The new file also owns the gap-16 regression.
- Wiring-map gap 16: `_preflight_section` was renamed exactly to `_gate_section`, and its sole
  caller was updated. The rendered heading remains `deconfounding gate`.
- R11: the authorized JSON and Markdown outputs are 520,409 and 80,562 bytes, below 1 MiB and
  contain no serialized arrays. The existing atomic temp/fsync/replace writer remains valid;
  no writer changes were required.
- R9 is unrelated cache work and was not touched.

## TDD evidence

Initial focused RED:

```text
.venv/bin/python -m pytest tests/test_probes.py -k 'offline_reanalysis_reports_task_bootstrap or reanalysis_markdown_applies_r8 or reanalyse_cli_is_deterministic' -o addopts='' -q
```

Exit 1: 3 failed, 117 deselected in 0.82 s. Failures were the absent cohort label, absent R8
Markdown/cohort table, and absent `elapsed_seconds` metadata.

Threshold mutation RED:

```text
.venv/bin/python -m pytest tests/test_probes.py -k 'r8_suppression_requires_24' -o addopts='' -q
```

With the implementation deliberately mutated to 23 rows / 4 cells, exit 1: 1 failed,
120 deselected in 0.13 s; a 23-row result incorrectly printed a numeric estimate. The ratified
24/5 constants were immediately restored.

Focused GREEN:

```text
.venv/bin/python -m pytest tests/test_probes.py -k 'offline_reanalysis_reports_task_bootstrap or reanalysis_markdown_applies_r8 or r8_suppression_requires_24 or reanalyse_cli_is_deterministic' -o addopts='' -q
```

Exit 0: 4 passed, 117 deselected in 0.86 s.

Gap-16 RED and relocation GREEN:

```text
.venv/bin/python -m pytest tests/test_reanalysis.py -k gate_section -o addopts='' -q
.venv/bin/python -m pytest tests/test_reanalysis.py -o addopts='' -q
```

Before the production rename, the focused test exited 1 because `state_probe._gate_section`
did not exist. After the exact rename, all 3 focused remediation tests passed in 0.30 s.
Combined collection found 119 tests in `tests/test_probes.py` and 3 in
`tests/test_reanalysis.py`, 122 total, with no duplicate test names; the pre-relocation probe
baseline was 121, so the two moved tests were neither duplicated nor lost and gap 16 added one.
After the Task 3 boundary release, collection again found 122 tests and the focused remediation
file again passed 3/3 in 0.28 s. The remaining probe file then passed 104 and failed 15 solely
at the released Task 3 jlens/capture API boundary; the earlier 119-pass and combined 122-pass
runs remain the last green integrated evidence before those concurrent API changes.

Broader verification before artifact generation and commit:

```text
.venv/bin/ruff check src/local_llm_lab/probes/state_probe.py tests/test_probes.py
git diff --check -- src/local_llm_lab/probes/state_probe.py tests/test_probes.py
.venv/bin/python -m pytest tests/test_probes.py -o addopts='' -q
uv run pytest -q
```

Results before the priority amendment: targeted Ruff and diff checks passed;
`tests/test_probes.py` passed 121 tests in 2.60 s; the full `uv` suite produced 244 passing
progress markers and exited 0. After R10/gap 16, the remaining `tests/test_probes.py` passed
119 tests in 2.34 s and the combined run passed 122 tests in 2.39 s. A concurrent unfiltered
full-suite rerun encountered eight failures solely in foreign untracked Task 3 tests added
during execution. The required bounded gate excluded exactly
`tests/test_adapter_delta.py`, `tests/test_capture.py`, `tests/test_jlens.py`,
`tests/test_repository_rules.py`, and the present incomplete `tests/test_runner.py`. Its first
attempt was interrupted at collection by a transient syntax error in the active Task 3
`pipeline/jlens.py` edit. The final bounded command was:

```text
uv run pytest -q -o addopts='' --ignore=tests/test_adapter_delta.py --ignore=tests/test_capture.py --ignore=tests/test_jlens.py --ignore=tests/test_repository_rules.py --ignore=tests/test_runner.py
```

It collected all other tests, passed 229, and failed 17 existing tests because the released
Task 3 implementation had not preserved the previous jlens/capture APIs (`_encode`,
`jlens_map(..., stats=...)`, response activation return shape, and injection/masking behavior).
Those files and corresponding out-of-scope implementation are not part of this change, so no
extra exclusions or fixes were made.

## CPU-only artifact command and provenance

```text
.venv/bin/agent-v2-probe-state reanalyse --input outputs/probes/state-corrected-hardened-20260903T172549/state-base-mix.npz --output outputs/probes/state-corrected-hardened-20260903T172549
```

Measured by `/usr/bin/time -p`: real 206.73 s, user 208.51 s, sys 1.78 s. JSON metadata
records `elapsed_seconds = 206.54776779201347`, the five split seeds
`20260903`–`20260907`, 1,000 task-ID bootstrap resamples, and unresolved registry model
`qwen25-coder-3b` / `mlx-community/Qwen2.5-Coder-3B-Instruct-4bit`.

| Artifact | SHA-256 | Size |
| --- | --- | ---: |
| JSON | `5d01769656c77963269abfd2f2a5445f5bd4327d74974e418e83e732d4dd0f79` | 520,409 bytes |
| Markdown | `333f4cceaf39959aa9519de0ffb94618450a345ff09f4c07824f83800aadd3c3` | 80,562 bytes |

## Preservation evidence

Before C6, 1,292 files across `outputs/`, `data/`, and `reports/` were inventoried and hashed,
excluding only the two authorized reanalysis outputs. The capture directory separately had 364
non-target files inventoried and hashed. After C6, both inventories were identical and all
1,292 and 364 SHA-256 checks passed with zero mismatches.

The original NPZ SHA-256 before and after was identical:

```text
17d46ab1b3fdd6642b887c1d4403b3dad79bfad142063184c2ece40b0231c838
```

Thus `data/`, `reports/`, the original NPZ, all 360 capture checkpoint shards, and every other
output remained byte-identical. Only the authorized JSON/Markdown pair changed.

## Ratified readings verdict

The reportable `all_rows` within-position values match the ratified table exactly after its
display rounding:

- `pending_count` L18: 0.5018 [0.1996, 0.7039] → **0.50 [0.20, 0.70]**;
- `first_bucket_count` L6: 0.9252 [0.4625, 0.9829] → **0.93 [0.46, 0.98]**.

Other exact matches are `pending_count` Holm support at L12/L18/L24, `next_tool` at L30/L35,
and `is_new_max` unsupported/underpowered. The reportable `all_rows` support flags materially
disagree with three ratified rows:

| Target | Ratified | Regenerated reportable `all_rows` |
| --- | --- | --- |
| `phase` | L35 | L30, L35 |
| `hidden_error` | L12–L30 | L6, L12, L18, L24, L30, L35 |
| `first_bucket_count` | L12, L24, L30 | L6, L12, L18, L30 |

The expected `phase`, `hidden_error`, and `first_bucket_count` support lists exactly match the
non-reportable `sft_disjoint` cohort, while the expected `pending_count` list and both quoted
within-position values come from `all_rows`. This indicates that the ratified readings table
combines cohorts despite R7. No statistical flag or artifact value was changed to force the
ratified wording.

## Deviations and remaining unverified items

- Concern: the ratified Holm-support table does not describe one R7-compliant reportable cohort;
  the discrepancy above requires a review ruling, not an implementation substitution.
- `n_test` and `n_cells` use conservative per-split minima because the report pools five split
  fits while R8 specifies singular counts. This makes suppression fail closed if any
  contributing split is below threshold.
- The optional whole-file formatter remains affected by pre-existing formatting outside the
  remediation hunks; targeted Ruff lint and `git diff --check` are clean.
- D3a, D3b, and every D4 wishlist/coverage item were explicitly not implemented. Fix round 1
  removed the remediation's four-line exact-two-output-files assertion after review identified
  it as the unratified D4 coverage gap; no replacement D4 assertion was added.
- Adapter comparisons and all gated SPEC-004 work remain deferred.

## Fix round 1 evidence

The reviewer-approved production code and artifacts were left byte-untouched. The only test
change removed the unratified exact-two-output-files assertion from
`test_reanalyse_cli_is_deterministic_and_never_calls_model_loading`. A repository search after
the edit confirmed that neither remediation test file newly asserts the exact output-file set.

Fresh focused verification after the removal:

```text
.venv/bin/python -m pytest tests/test_reanalysis.py -o addopts='' -q
.venv/bin/python -m pytest tests/test_probes.py -k 'reanalysis_recodes_running_max or reanalysis_surface_counts_only or reanalysis_rejects_a_data_seed or reanalysis_fails_closed_when_capture_has_no_data_seed or reanalysis_rejects_saved_labels or offline_reanalysis_reports_task_bootstrap or reanalysis_marks_zero_variance or reanalyse_cli_is_deterministic' -o addopts='' -q
```

The dedicated remediation file passed 3 tests in 0.40 s. The eight selected historical and
remediation probe tests passed with 111 deselected in 0.84 s. These focused selections avoid
the separately owned Task 3 API boundary documented above.
