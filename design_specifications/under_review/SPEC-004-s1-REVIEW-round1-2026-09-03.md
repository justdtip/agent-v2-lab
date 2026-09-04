# SPEC-004 §1 review, round 1, ratified (2026-09-03, commits f90578b and 75f55f2)

Reviewer: Claude (Chief). Based on the Deputy's draft `SPEC-004-s1-REVIEW-draft-2026-09-03.md`,
whose artifact-level findings A1 to A7 and sections B and C are adopted with the rulings below.
Section D (code-level conformance) is appended by the Deputy when its delegated diff review
returns; it changes this verdict only if it finds a model load, a protected-directory write, or
a missing §1 item.

## Verdict: sent back for corrections C1 to C6. The analysis itself is sound and its outputs are usable now under rulings R7 and R8.

## Rulings (also in wiring map §7)

- **R7, cohorts.** The Deputy's cross-tabulation shows the only rows byte-identical to the
  adapters' SFT data are the 737 `train-` rows; `p2mix` tasks are seeded by their own split name
  and are different tasks at the same difficulty. The exclusion as implemented is therefore
  correct and the decision memo's "1,050 rows" was my error (memo corrected). For a base-model
  run the `all_rows` cohort is the reportable one; `sft_disjoint` exists only for future paired
  comparisons with adapter runs and must be compared **within difficulty**, never pooled,
  because excluding `train-` rows shifts the difficulty mix (313/451/866 against 1050/451/866).
- **R8, within-position reporting.** Within-position R² bootstrapped over task ids is not a
  usable statistic at the current cell sizes (intervals below −1, 30 `nan` cells). Every
  within-position cell reports `n_test`, `n_cells`; cells with fewer than 24 test rows or 5
  eligible cells are printed as `n/a (n=…)`, never as a number; the pooled `overall` row is
  printed in the markdown; and the interval shown is the bootstrap over task ids **restricted to
  eligible cells**, with the point estimate from the full fit.

## Correction (2026-09-04 morning, closes issue #5)

The original readings table below carried the pre-remediation readme's `sft_disjoint` layer
sets under an `all_rows` label — a transcription error by the Chief, confirmed by two
independent derivations (the Deputy's pre-rerun JSON extraction and the C6 rerun, which agree
wherever cross-checkable). The corrected, reportable `all_rows` readings from the C6 rerun
(commit `b39d6ea`) are:

| Target | Supported layers (all_rows, Holm-adjusted, both controls) |
| --- | --- |
| `pending_count` | L12, L18, L24 (unchanged) |
| `phase` | L30, L35 |
| `hidden_error` | L6, L12, L18, L24, L30, L35 |
| `next_tool` | L30, L35 (unchanged) |
| `first_bucket_count` | L6, L12, L18, L30 |
| `running_max` / `is_new_max` | not supported (unchanged) |

The qualitative conclusion is unchanged and slightly strengthened. The original table is kept
below for the record and must not be quoted.

## Readings under R7 and R8 — ORIGINAL (superseded, do not quote)

| Target | Reportable result (`all_rows`) |
| --- | --- |
| `pending_count` | position margin supported after Holm at L12, L18, L24; within-position L18 0.50 [0.20, 0.70] |
| `first_bucket_count` | supported at L12, L24, L30 against both position and surface baselines; within-position L6 0.93 [0.46, 0.98] |
| `phase` | supported at L35 |
| `hidden_error` (re-coded) | supported at L12 to L30 |
| `next_tool` | supported at L30, L35 (positive control) |
| `running_max` / `is_new_max` | not supported; underpowered |

The decision memo §2.2 within-position point estimates are superseded by the intervals above;
the qualitative conclusion (state readable from the note beyond position; strongest for the
bucket count; weak for the running maximum) stands.

## Corrections owed (Codex)

| # | Correction | Source finding |
| --- | --- | --- |
| C1 | Implement R8: per-cell `n_test`/`n_cells`, suppression threshold, `overall` row in markdown, eligible-cell bootstrap | A4, A5 |
| C2 | Markdown defines "Holm support" and names which margin each flag refers to | A7 |
| C3 | Record `elapsed_seconds` and the unresolved registry `ModelSpec` (data-stage form, per Codex's provenance ruling) in `metadata` | A6 |
| C4 | Label the cohorts per R7 in both outputs: `all_rows (reportable for base)`, `sft_disjoint (paired adapter comparisons, within difficulty only)`; add a by-difficulty cohort table | A2 |
| C5 | Write `under_review/SPEC-004-s1-IMPLEMENTATION-REPORT.md` and an SDD ledger entry | B |
| C6 | Re-run `reanalyse` after C1 to C4 (CPU only, permitted output files only) and confirm the readings table above | A3 |

## Adopted without change

A1 (artifact hygiene and provenance), A3 (headline readings), C (banned constants clean).

---

# Section D, appended by the Deputy (2026-09-03 22:20)

**Verdict impact: none.** The delegated diff review found no model load on the `reanalyse` path,
no write outside the two permitted output files, and no missing §1 item. All six items of §1 are
implemented. The verdict and corrections C1 to C6 above stand as ratified.

D3a, D3b and the coverage gaps in D4 are new defects not visible from the artifacts and are put
forward as candidate corrections for the Chief to fold into the list above; the Deputy has not
numbered them as corrections.

## D. Code-level conformance

Delegated to a read-only Sonnet summariser; the two defects in D3 and the coverage claim in D4
were re-verified in source by me.

### D1. All six §1 items are implemented

| Item | Where | Note |
| --- | --- | --- |
| Bootstrap over task ids, 1,000 resamples, five split seeds, median and 2.5/97.5 | `state_probe.py:110-111` (defaults), `1715-1724` (`_task_bootstrap_plan`), `1728-1734` (`_interval`) | resampling buckets rows by `task_id` and samples unique ids with replacement, so the unit is correct |
| Surface baseline, same estimator | `state_probe.py:112-119`, `1586-1591`, `1879-1888` vs `1896-1905` | the surface fit and the activation fit call the same `_prediction` helper with the same `alpha`, `l2` and `steps`; the surface array never touches `cohort.features`, so no activation leakage |
| SFT-row exclusion by task-id prefix | `state_probe.py:2065-2067, 2080-2090` | mechanism correct; see A2 for what it does and does not exclude |
| Target re-coding from regenerated truth | `state_probe.py:88-97` (`REANALYSIS_TARGETS`), `1441-1488` (`reanalysis_row_labels`), `1592-1593` | all four re-codings present and sourced from a generator replay, not from stored npz labels; a stored label that disagrees with regenerated truth raises (`1557-1574`) |
| Within-position by difficulty and family | `state_probe.py:1979-2003` | present; see A4 and A5 for what the output does with it |
| Holm across all cells, cell count reported | `state_probe.py:1813-1822` (`_holm_adjust`), `2019`, `2030` | genuine step-down Holm with the monotonic running max, pooled once per cohort across every target and layer, not per target and not Bonferroni |

Two details are right in a way worth recording. The per-cell p-value is
`max(p_position, p_surface)` (`state_probe.py:2016`), the conservative intersection-union
choice for a "must beat both" rule. And the bootstrap p-value floor is
`1 / (n_finite + 1)`, which at 5,000 samples is 0.0002; times 48 cells that is the 0.0096
adjusted minimum seen throughout the output, so the reported numbers are internally consistent.

### D2. The `reanalyse` path loads no model

`main()` dispatches to `_main_reanalyse` and returns before reaching `load_policy`, `mlx.core`
or the `--model` argparse block (`state_probe.py:2595-2597` against `2599, 2708, 2715`). The
path's imports are `build_rows`, `parse_turn`, `Task`, and deferred `Simulator` and
`make_tasks`; none of those chains import mlx. `build_prompt` is imported at module level but is
called only from the capture path (`state_probe.py:455`). `tests/test_probes.py:1829`
monkeypatches `load_policy` and `build_prompt` to raise and the CLI still passes.

### D3. Two defects

- **D3a (JSON schema differs between cells, minor).** `raw_p`, `holm_adjusted_p`,
  `interval_supported` and `holm_supported` are set only inside the
  `if finite_position and finite_surface:` branch (`state_probe.py:2007-2017`). A layer whose
  margin bootstrap is entirely NaN is still written into `entry["layers"]` at line 2017 but
  silently lacks those four keys, with nothing marking why. The markdown hides this by
  defaulting to "no". A JSON consumer testing `"holm_supported" in entry` and one reading it as
  `False` get different answers.
- **D3b (recorded seed can name the wrong split seed, minor but exactly the wrong kind of bug).**
  `within_distributions[layer]` is appended only when `_within_prediction` returns non-None
  (`state_probe.py:1956-1957`). The three within-position bootstraps then seed themselves with
  `split_seeds[offset % len(split_seeds)]` where `offset` enumerates the successes list, not the
  split-seed loop (`1969, 1976, 1985, 1994`). If seed 1 is skipped and seed 2 is not, seed 2's
  result is seeded as though it came from seed 1. The statistics stay valid and deterministic;
  the provenance does not, in a feature whose purpose is provenance.

### D4. Test coverage: eight tests, and the Holm path is untested

Eight tests added, all on the generator and constructed numpy arrays, none loading a checkpoint
(`tests/test_probes.py:1699, 1716, 1732, 1744, 1757, 1770, 1819, 1829`). They cover the target
re-coding against literal expected values, the surface counts excluding the tool call, three
fail-closed provenance guards, the metadata and interval shape, the zero-variance R² guard, and
CLI determinism with model loading monkeypatched to raise.

Not covered:

- The Holm arithmetic. `grep -c "holm\|raw_p\|interval_supported" tests/test_probes.py` returns
  0. Item 6 is entirely unverified.
- Worse, it cannot be verified at the sample sizes the tests use. Every test overrides the
  defaults to 12 resamples and 2 seeds, so the p-value floor is 1/25 = 0.04; multiplied by even a
  small cell count that never reaches alpha, so no cell can ever be supported in a test. A Holm
  test needs enough resamples for the floor to clear `alpha / cells`, or a direct unit test of
  `_holm_adjust` against hand-computed values.
- No test asserts the shipped defaults are 1,000 resamples and five seeds.
- The "must beat both" AND-gate is never exercised with a case that beats one baseline and not
  the other.
- `pending_count_ordinal` is asserted only to exist as a key; its values are never checked, and
  it is built as `str(int(...))` at `state_probe.py:1472` with no clamp to the 0-6 range the spec
  states.
- No test asserts that exactly two files are written to the output directory.
