# R12 versioned replay implementation report

## Scope

Implemented versioned, thought-template-only replay for generator versions 1 through 4.  The
current generator remains the only normal generation path; replay retains current files, actions,
prompts, variants, RNG-derived structure, and difficulty.  Integrity and saved-NPZ reanalysis now
require a recorded generator version or an explicit matching legacy binding.

## Test-first evidence

- RED: `uv run pytest -q tests/test_tasks.py -k 'replay or historical_generator'` initially
  failed collection because `replay_task_from_id` did not exist.
- RED: focused integrity resolver tests initially failed because the artifact-version resolver and
  binding path did not exist.
- RED: state-probe metadata tests initially failed because captured datasets did not record a
  generator version and reanalysis had no resolver.
- GREEN: `uv run pytest -q tests/test_tasks.py -k 'replay or historical_generator'`
  completed with `19 passed`.
- GREEN: `uv run pytest -q tests/test_tasks.py tests/test_integrity.py tests/test_state_probe.py`
  completed with `93 passed`.

The replay tests cover v1/v2 clean parity, v3 Run-D notes, v4 state-preserving corrections, every
changed long family, and v1-to-v2 plus v3-to-v4 wrong/stale/failed-edit injected and recovery
notes.  The state-probe tests cover a saved NPZ round-trip without model loading, explicit CLI
forwarding, checkpoint-context recording, and independently parsed v4 `row_labels` progress
oracles with non-vacuous pending/load/value/batch coverage.

Issue #13 also replaces stale literal model-registry expectations in the deterministic offline
reanalysis test with JSON-normalized `asdict(load_model_spec(...))` values while preserving the
independent exact `ModelSpec` schema assertion.  The fake-only path is parameterized for both
`qwen25-coder-3b` and `qwen35-4b`.

## Quality and boundaries

- Normal scoped Ruff: `All checks passed!`
- Strict C901: three existing `state_probe.py` exceptions remain: `build_probe_dataset` (16),
  `_analyse_cohort` (20), and `main` (18), each above the configured 10 threshold.  No R12
  function was reported.
- `git diff --check` reports only the pre-existing foreign pending-spec EOF blank line in
  `design_specifications/pending/02-INTERFACE-AND-WIRING-MAP.md`.
- Protected-path listing contains only foreign pending-spec changes; no R12 `outputs/` or `data/`
  changes were made.  No model, tokenizer, MLX, probe, training, or evaluation workload was run.
- The shared task board's parent-owned claim remains the authority; no second claim was created.

## Review fix round 1

- Replayed all five recovery variants for v1-v3, including transient duplicate calls,
  unknown-tool recovery, v1 list-first wording, and historic failed-edit ordinary/recovery notes.
- Reanalysis now rejects inconsistent per-row family metadata; checkpoint capture rejects a stale
  generator-version context rather than preserving it.
- Added artifact-consumer binding tests and independent `highest so far` row-label parsing.
- Focused owned suite: `101 passed`; normal scoped Ruff: `All checks passed!`.
- The earlier patch-probe gate block is superseded by the current green R13 evidence below.

## R13 full fake-only suite

- HEAD: `f0c8d8bba30334165aa27da4d6445e9d96bafcdd`
- Command: `uv run pytest -q`
- Exit code: `0`
- Result: `506 passed, 0 failed, 0 xfailed`
- Failing node names: none

## Review fix round 2

- Added replay-forwarding, top-level compatibility, explicit CLI, invalid-binding, and saved
  recovery NPZ/difficulty offline acceptance coverage.
- Historical recovery reanalysis now exercises real offline rows and `reanalyse_dataset` under
  hard-failing model/prompt seams, asserts a version-distinguishing lexical surface cell, and
  pins returned generator-version metadata plus non-default saved difficulty.
