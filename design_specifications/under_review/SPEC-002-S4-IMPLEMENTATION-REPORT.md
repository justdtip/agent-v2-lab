# SPEC-002 §4 runner/evaluator remainder — implementation report

## Outcome and scope

Lane E implements the fake-only runner/evaluator remainder of SPEC-002 §4:

- `detect_loop` rule 2 now recognizes four consecutive same-tool error observations while
  preserving the independent three-identical-call and eight-same-shape rules.
- `evaluate_tasks` is characterized at its fake boundary: clean evaluation passes `faults=None`
  and stress evaluation passes `STRESS_FAULTS` to `runner.run_task`.
- `summarize` reports total `think_tokens` and a zero-safe, two-decimal
  `think_tokens_per_task` mean while preserving the structured integrity summary exactly.
- The pre-existing `generate_turn_with_count` C901 failure was resolved by extracting private
  thinking-state bookkeeping. Existing fake generation, thinking-budget, and cache-equivalence
  tests preserve its behavior.

The owned source and test files changed are:

- `src/local_llm_lab/pipeline/runner.py`
- `src/local_llm_lab/pipeline/evaluate.py`
- `tests/test_runner.py`
- `tests/test_evaluate.py`
- `tests/test_pipeline.py` (one complete test and its private helper/import removed)

## File line counts

| File | Added | Removed |
| --- | ---: | ---: |
| `src/local_llm_lab/pipeline/runner.py` | 52 | 37 |
| `src/local_llm_lab/pipeline/evaluate.py` | 4 | 0 |
| `tests/test_runner.py` | 46 | 1 |
| `tests/test_evaluate.py` | 87 | 0 |
| `tests/test_pipeline.py` | 0 | 41 |
| `docs/superpowers/plans/2026-09-04-spec-002-s4-remainder.md` | 221 (new) | 0 |
| `design_specifications/under_review/SPEC-002-S4-IMPLEMENTATION-REPORT.md` | 241 (new) | 0 |

The owned evidence paths are this report,
the ephemeral `.superpowers/sdd/2026-09-04-spec-002-s4-remainder/task-1-report.md`, and the
durable task plan. No public signature changed, so no callers required an update. Interface-map
rows touched: §2.7 `runner.py`, §2.10 `evaluate.py`, and wiring gap §4.3 for trajectory thinking
fields.

## Decisions taken

- The stress seam already implemented the specified fault selection. Its new test passed on
  first execution, so this is characterization evidence; no production change was manufactured.
- The aggregate test asserts the full literal integrity structure before looking up the new
  thinking-token keys. The required RED therefore proves that integrity already matched and the
  failure was specifically the absent thinking metric.
- The exact C901 command initially found `generate_turn_with_count` at complexity 14. Because
  that command is a required Lane E gate over an owned runner file, the state bookkeeping was
  extracted without changing behavior rather than suppressing the finding.
- The full-suite failures in foreign lanes were not edited around. Lane E retained its claim
  until those owners could land their work, as required by R13.
- After a formal scope amendment, the complete legacy loop-pattern test moved subtractively
  from `tests/test_pipeline.py` to `tests/test_runner.py`. Its six/five oracle became the
  specified four/three oracle, and the earlier standalone rule-2 test was consolidated into
  this authoritative all-rules test rather than leaving duplicate coverage.

## RED → GREEN evidence

### Four-error loop threshold

RED command:

```text
uv run pytest -q tests/test_runner.py::test_detect_loop_flags_four_same_tool_errors_but_not_three
```

RED result: exit 1. The three-step prefix returned `False`, then the four-step assertion failed
because `detect_loop(errors)` returned `False` under the old threshold of six.

GREEN result after changing `LOOP_SAME_TOOL_ERRORS` to 4 and updating the rule-2 contract text:
exit 0, one passing test (`.`). The focused behavior was subsequently consolidated into the
subtractive migration described below.

Migration verification:

```text
uv run pytest -q tests/test_runner.py::test_detect_loop_flags_repetition_patterns
uv run pytest -q tests/test_pipeline.py
```

Results: exit 0 with one passing migrated node, then exit 0 with 50 passing source-monolith
tests. The migrated test independently covers four distinct-argument same-tool errors as a loop,
three as non-looping, all other repetition rules, parse-error omission, and the empty input.

### Stress-boundary characterization

Command:

```text
uv run pytest -q tests/test_evaluate.py::test_evaluate_tasks_passes_stress_faults_to_runner
```

Result: exit 0, two passing parametrized cases (`..`). The test replaces sampler creation,
task execution, and integrity checking with in-memory fakes and independently expects `None`
for clean mode and `evaluate.STRESS_FAULTS` for stress mode. Production was already correct.

### Aggregate thinking metrics with intact integrity

RED command:

```text
uv run pytest -q tests/test_evaluate.py::test_summarize_aggregates_thinking_tokens_and_preserves_integrity
```

RED result: exit 1. The exact integrity literal passed first, then lookup of
`summary["think_tokens"]` failed with `KeyError: 'think_tokens'`.

GREEN result after adding the accumulator and summary fields: exit 0, one passing test (`.`).
The two literal trajectories carry 3 and 8 thinking tokens; the summary reports 11 total and
5.5 per task. The exact expected integrity result remains:

```text
clean_trajectories=1, clean_rate=0.5, affected_trajectories=1, violations=2,
by_kind.value_drop={violations=2, affected_trajectories=1},
by_family.read={trajectories=2, clean_trajectories=1, clean_rate=0.5,
violations=2, affected_trajectories=1}, failed_trajectories=1,
failed_with_violation=1, failure_explained_rate=1.0
```

## Focused fake-only verification

Existing generation/refactor characterization:

```text
uv run pytest -q tests/test_runner.py::test_generate_turn_stops_at_closing_fence \
  tests/test_runner.py::test_generate_turn_force_closes_thinking_at_budget_and_continues \
  tests/test_runner.py::test_fake_greedy_generation_is_equivalent_for_cache_strategies
```

Result: exit 0, six passing cases (`......`).

Combined owned test files after the subtractive migration:

```text
uv run pytest -q tests/test_runner.py tests/test_evaluate.py tests/test_pipeline.py
```

Result: exit 0, 78 passing tests.

Scoped Ruff:

```text
uv run ruff check src/local_llm_lab/pipeline/runner.py \
  src/local_llm_lab/pipeline/evaluate.py tests/test_runner.py tests/test_evaluate.py \
  tests/test_pipeline.py
```

Result: exit 0, `All checks passed!`.

Scoped complexity gate:

```text
uv run ruff check --select C901 src/local_llm_lab/pipeline/runner.py \
  src/local_llm_lab/pipeline/evaluate.py tests/test_runner.py tests/test_evaluate.py \
  tests/test_pipeline.py
```

Initial result: exit 1, `generate_turn_with_count` complexity `14 > 10`.
Final focused result after the private extraction: exit 0, `All checks passed!`.

The owned-path `git diff --check -- <seven durable owned paths plus ephemeral SDD path>` exits 0.
The exact global `git diff --check` exits 2 only for the formally exempt foreign protected-path
discrepancy `design_specifications/pending/02-INTERFACE-AND-WIRING-MAP.md:560: new blank line at
EOF`. Lane E did not edit that file.

## Diff, protected-path, and no-model evidence

Lane E ran no checkpoint load, model generation, training, selection, evaluation, rollout,
branch, preference, preflight, probe, stress, or research command. Every executable behavior
check above is pytest with in-memory fakes; static checks are Ruff and Git only. No command
wrote under `data/`, `outputs/`, `reports/`, `design_specifications/pending/`, or
`design_specifications/research/`.

Before commit, the pre-existing index and exact staged names are inspected. The durable staged
set is exactly the five owned code/test files plus this report and the task plan; the ignored
SDD workspace is not forced into Git. The final commit's `git show --name-only` is the
authoritative protected-path check: its intersection with `data/`, `outputs/`, `reports/`,
`design_specifications/pending/`, and `design_specifications/research/` is empty. A scan of added
Lane E lines for `36`, `2048`, standalone `35`, `<|im_end|>`, and `model.model.layers` returned
no hits (grep exit 1).

## R13 full fake-only suite

Initial non-handoff run:

- Tested HEAD: `ed8e4cbae1c6995916f07187d041dcf958b6ee3c`
- Command: `uv run pytest -q`
- Exit: 1
- Counts: 453 passed, 7 failed
- Failing nodes:
  - `tests/test_capture.py::test_replacement_overwrites_selected_rows_without_addition`
  - `tests/test_capture.py::test_replacement_maps_ordered_rows_across_cached_calls`
  - `tests/test_capture.py::test_replacement_validates_its_shape[vector0-at_positions0-hidden width]`
  - `tests/test_capture.py::test_replacement_validates_its_shape[vector1-at_positions1-row count]`
  - `tests/test_capture.py::test_replacement_rejects_nondefault_alpha`
  - `tests/test_integrity.py::test_retroactive_saved_evaluations_match_memo_contract`
  - `tests/test_pipeline.py::test_detect_loop_flags_repetition_patterns`

This was not a completion gate. Five failures belonged to the in-flight capture replacement
lane, one was the known R12 replay dependency, and one was the legacy monolithic assertion for
the superseded six-error rule. Lane E did not edit any of those files.

Earlier green integration run:

- Tested HEAD: `bac1636a18a716dead70f3ef389a28d65064d1cd`
- Command: `uv run pytest -q`
- Exit: 0
- Counts: 460 passed, 0 failed
- Failing nodes: none (`[]`)

Interim non-handoff run after new Lane D tests appeared:

- Tested HEAD: `9f1c0a5164c4301b43682e0f43d07c60afeb7fa0`
- Command: `uv run pytest -q`
- Exit: 1
- Counts: 457 passed, 3 failed
- Failing nodes:
  - `tests/test_cli.py::test_resolve_training_spec_uses_registry_hf_id_and_configured_key_policy`
  - `tests/test_cli.py::test_lora_config_uses_resolved_architecture_and_fresh_keys`
  - `tests/test_cli.py::test_stage_train_clears_only_its_checkpoint_directory`

Those failures were the in-flight Lane D test-first state. Lane E did not edit
`tests/test_cli.py` or `pipeline/cli.py`.

Final handoff run after Lane D landed:

- Tested HEAD: `d3a4721b5084ee244686db9e48cb000c97575654`
- Command: `uv run pytest -q`
- Exit: 0
- Counts: 472 passed, 0 failed
- Failing nodes: none (`[]`)

## Deviations and unverified work

There is no functional deviation from SPEC-002 §4. The C901 extraction is an additional
behavior-preserving implementation step required by the task's exact static gate. Real-model
behavior is intentionally unverified because model execution is prohibited; no requested
acceptance criterion requires it.
