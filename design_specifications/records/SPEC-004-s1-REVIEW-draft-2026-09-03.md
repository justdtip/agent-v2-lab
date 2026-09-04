# SPEC-004 §1 review, DRAFT (2026-09-03, commits f90578b and 75f55f2)

**DRAFT — not ratified. Deputy Chief of AI Research, first pass. Nothing here is approved.**

Scope: SPEC-004 §1, offline re-analysis of the saved hardened P2 npz. No model was run. The
saved artifacts were read; `tests/test_probes.py` was run on fakes.

Tests: `uv run pytest -q tests/test_probes.py` → 119 passed in 2.43 s. The full suite does not
collect, but for an unrelated reason: the dirty, uncommitted `tests/test_pipeline.py` imports
`iter_task_records` from `transcript.py`, which does not exist yet (issue #2 F2, in progress).
`state_probe.py` and `tests/test_probes.py` are clean at HEAD, so the probe run above is the
committed state and no checkout was needed.

## A. Artifact-level findings

### A1. The re-analysis ran and wrote only what it was permitted to write

`outputs/probes/state-corrected-hardened-20260903T172549/state-base-mix.reanalysis.{json,md}`
exist, written 21:50. `find outputs reports data -newermt "2026-09-03 21:00"` returns those two
files and nothing else, so briefing §1.2 is satisfied.

Provenance in the JSON `metadata` block is good: `bootstrap_resamples` 1000, `bootstrap_unit`
`task_id`, five `split_seeds`, `interval_percentiles` [2.5, 50, 97.5], the three controls, the
six surface features, the fit hyper-parameters, `data_seed`, the model reference with its HF
revision hash, and the full command line.

### A2. (blocking) The `sft_disjoint` cohort is not SFT-disjoint, and the exclusion is perfectly
confounded with difficulty

Cross-tabulating the saved npz (`task_ids` × `difficulty`) gives:

| split prefix | difficulty 0 | difficulty 1 | difficulty 2 | rows |
| --- | --- | --- | --- | --- |
| train | 737 | 0 | 0 | 737 |
| p2mix | 313 | 451 | 0 | 764 |
| test | 0 | 0 | 866 | 866 |

The exclusion drops rows whose `task_id` starts with `train-`
(`state_probe.py:2065-2067, 2080-2089`). That is exactly what SPEC-004 §1.3 asks for, and it
drops 737 rows. But the memo (§2.2 item 1) and the briefing (§3, "the mixed P2 dataset's
difficulty-0 rows are byte-identical to SFT rows") put the confounded population at 1,050
difficulty-0 rows. 313 of those are `p2mix`-prefixed and remain in the cohort the output
labels `sft_disjoint`, 19% of its 1,630 rows.

Second and worse: `train` is 100% difficulty-0 and is the only difficulty-0-heavy split, so the
exclusion also re-weights the difficulty mix from 1050/451/866 to 313/451/866. Any difference
between the two cohorts confounds memorisation removal with a difficulty shift. The
`pending_count` collapse in A3 is exactly such a difference and cannot be attributed to either
cause from this output.

This is a spec-versus-memo conflict, not an implementer error: §1.3 names the mechanism, and
Codex implemented the named mechanism. For the Chief to rule on.

### A3. (headline) `pending_count` does not survive; `first_bucket_count` does

Answering the Director's question directly. Support requires both margin intervals to exclude
zero and a Holm-adjusted p ≤ 0.05 across 48 cells (`multiple_comparisons` block).

`pending_count`, SFT-disjoint cohort: no layer is supported at any depth. Raw position margins
at L12 and L18 are 0.056 [0.010, 0.120] and 0.063 [0.008, 0.138], intervals excluding zero, but
Holm-adjusted p are 0.33 and 0.32. In the `all_rows` cohort the same target is supported at
L12, L18 and L24. The memo's "+0.087 R² at L12" survives as a raw margin and dies under
adjustment plus exclusion.

`first_bucket_count`, SFT-disjoint cohort: supported at L12, L24 and L30. Position margins
0.153 [0.126, 0.250], 0.145 [0.117, 0.218], 0.134 [0.104, 0.205]; surface margins 0.142, 0.128,
0.114, all excluding zero. The memo's "+0.179 R² at L6" holds in direction and survives both
controls at three depths.

The `readme` string in both outputs states the survivors as phase (L35), hidden_error (L12, L18,
L24, L30), next_tool (L30, L35), first_bucket_count (L12, L24, L30). That list is consistent
with the tables I checked.

### A4. The within-position quantity the memo actually quoted is absent from the markdown and
unstable in the JSON

The memo §2.2 reports within-position figures (`pending_count` R² 0.56-0.62 at L6-18;
`first_bucket_count` 0.91-0.97). The JSON keeps that pooled figure as
`within_position.overall`, but the markdown's within-position tables print only the
`by_difficulty` and `by_family` rows: `grep -c "| overall |"` on the markdown returns 0. A
reader of the human-readable output cannot find the number the programme has been quoting.

Bootstrapped, the pooled figure is much weaker than the memo's point estimates:

| target | cohort | L6 | L12 | L18 |
| --- | --- | --- | --- | --- |
| pending_count | all_rows | 0.443 [0.087, 0.668] | 0.431 [0.066, 0.659] | 0.501 [0.200, 0.704] |
| pending_count | sft_disjoint | 0.402 [-0.179, 0.658] | 0.360 [-0.184, 0.617] | 0.435 [-0.205, 0.709] |
| first_bucket_count | all_rows | 0.933 [0.462, 0.983] | 0.912 [-0.054, 0.982] | 0.871 [-0.609, 0.970] |
| first_bucket_count | sft_disjoint | 0.876 [-1.424, 0.994] | 0.935 [-1.223, 0.985] | 0.833 [-3.454, 0.985] |

Every SFT-disjoint within-position interval includes zero. Several lower bounds are far below
-1 (`first_bucket_count` L30 sft_disjoint: -9.153), which is not a small-sample nuisance but a
sign that within-position R² under task resampling is not a usable statistic at these cell
sizes.

### A5. Degenerate within-position cells are printed as if they were measurements

The markdown contains 30 cells reading `nan [nan, nan]` and 7 whose median is below -100; the
worst is `cross_reference` at L6, `-161877.104 [-453905.531, -11976.593]`
(reanalysis markdown line 32). Neither the JSON `within_position` block nor the markdown table
carries a row count per cell, so a reader cannot separate an underpowered cell from a real
negative result. The memo already made this mistake's cost explicit for `running_max`
("underpowered and ill-posed, 8 cells, 57 rows"). A per-cell `n`, and suppression or explicit
flagging below a minimum, would fix it. There is a zero-variance guard
(`test_reanalysis_marks_zero_variance_bootstrap_r2_undefined`), but it does not catch tiny
non-degenerate cells.

### A6. Two acceptance clauses cannot be checked from the artifact

SPEC-004 §7 requires §1 to run "in under ten minutes on CPU". No wall-clock or elapsed field is
recorded anywhere in the JSON, so the criterion is unverifiable without a rerun. §7 also
requires every result file to record "the resolved `ModelSpec`"; the metadata records
`model: {reference, artifact}` and a `capture_context`, which is provenance but not the
SPEC-001 §1 resolved spec object.

### A7. The markdown does not define its own support column

"Holm support" is defined only in the JSON (`support_requires`: both paired margin intervals
exclude zero and Holm-adjusted p ≤ 0.05). The markdown prints the column with no definition and
with two margin columns beside it, so a reader cannot tell which margin the flag refers to.

## B. Hand-off (briefing §7)

Not followed. There is no `SPEC-004-IMPLEMENTATION-REPORT.md` in `under_review/`, no SDD ledger
entry (`.superpowers/sdd/` holds only the spec-001 and spec-002 directories), and SPEC-004 is
still in `pending/`. The Codex coordination lane for this work lists the two output files among
its owned paths but no report path, so the report was never planned. Same finding as SPEC-002
F4.

## C. Banned constants (briefing §1.7)

Grep of the two commits' added lines for `36`, `2048`, `35`, `<|im_end|>`, `model.model.layers`
and projection-name lists returns nothing. The layer list `[6, 12, 18, 24, 30, 35]` in the
outputs comes from the saved capture metadata, not from new source.

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
