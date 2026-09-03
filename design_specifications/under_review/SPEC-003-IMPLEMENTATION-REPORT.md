# SPEC-003 implementation report: Run D data recipe

## Outcome

This implementation delivers SPEC-003 sections 1–3 only. It bumps the generator from v2 to v4,
introduces completion-safe state notes and `render_expert_note`, aggregates named logical splits
into role files through `SplitSpec`, normalizes both YAML schemas at the CLI seam, and adds D3,
D4, and B4 configs. Training/evaluation matrix work was not run.

## Wiring and callers

- Map 2.4: preserved `Task`, `make_tasks`, `task_from_id`, RNG derivations, task ids, actions,
  files, variants, and difficulty behaviour; added the pure bounds-checked note renderer.
- Map 2.5: added frozen validated `SplitSpec`; `write_dataset` creates per-logical-split
  manifests and declaration-ordered `train/valid/test.jsonl` role outputs.
- Map 2.11 and gap 10: `dataset_splits` converts legacy `tasks:` or explicit `splits:` exactly
  once before `stage_data` calls `write_dataset`. The existing data-stage provenance call remains
  unchanged and receives the full returned manifest and generator version.

## Decisions

- Chat replay attaches only to the first logical chunk of each role before that chunk’s unchanged
  `random.Random(f"{seed}:{split}")` shuffle. Extras attach once to train likewise.
- Recovery repeats are limited to all `role == "train"` chunks and remain `{transient: 1,
  wrong_path: 2, unknown_tool: 2, stale_path: 6, failed_edit: 6}`.
- Manifest preserves historical per-split keys for a one-chunk legacy role and adds explicit
  `count`, `difficulty`, `perturb`, `role`, and `outputs` hashes/counts.

## Generator hashes

| version / oracle | train | valid | test |
| --- | --- | --- | --- |
| v2 generator-only | `e7fa63ef9a2fe70b3563a23fa5421b4a6b11d11971802d2c4b6bce5a0a3c5d58` | `816543d1dee8299b2dbf514e93a2de480b69f84d6e8c53bda955c20c79f6213f` | `10042da5d9a4789f9a28fc6c0c9a8ea8efc59d090688f1e167c6de8d1de66877` |
| v3 generator-only | `6a2875aff061e7dcdde0b8a394fc3115ff5c6f2b4dbcb8f399db5af373c6970e` | `6df3a890d09e989c83baaf6078a27d56ec9da035807c8820e83b44c09a8259e3` | `925f5b282885e4c52f2a20280372804a17dfc9c2f1d8039478de98211eee1303` |
| v2 configured replay | `ad660e83cd89958dcee9fba2ab1e53115d1fb079813ea694cd4b0e89530b3a79` | `d6dbc53573744751d74565a0de6ca5c6d381cba6b488ff6410194bf9b0d4e6d8` | `fc69b03fef8f423ee85a174214ad955fe3f4d324554217510b92f53435841ce3` |
| v3 configured replay | `92f40d1868438553f306be192b09cace7b3d5efc8cff4a55c84fc3a96827ddf2` | `6e2617d5159e42fc7d26077e332a83e9f27fc0e755d50140d6336c75116c44b7` | `f79d6fc673674719cc17664278a15ee8ba9fcb6de8f9418d9226b99b5d24de59` |
| v4 generator-only | `99176c0b378d85502e738f23b8d174dd8321cb48a51e10c3a9bcd7fd9db3f035` | `41ec07d6f122a123ee7780885d59ba2bc77ca5ce836e896ed8ddd2861f680deb` | `8b8aeaf28460798e0863ab0e5cb217da25b8661b7b3a802d21182fcd36a294a7` |
| v4 configured replay | `67d49cddc00c95fafed9a0166b431021d405348354d029cb7941006d31117eec` | `27ec5981da0864f094de8e764a28140ec063443a1a20c89e4f7a40f489d6c6a2` | `67e4ead6a0a899de82395cc8c3101fdd81d5b9b853328b8d3d53b2bde908f0a6` |

The v4 values were reproduced through independent pytest temporary directories `pytest-535`
(intentional pre-pin RED) and `pytest-541` (post-pin GREEN) before acceptance.

## Verification

- RED generator/integrity: 9 intended failures.
- RED data: missing `SplitSpec` import error.
- RED CLI: raw-count conversion failure and missing Run D configs.
- Current focused review suite: `29 passed, 48 deselected`.
- Current owned suite: `76 passed, 1 failed` (historical compatibility blocker below).
- Scoped Ruff: clean; diff check: clean.
- Strict C901 (`max-complexity=9`): only established `cli.main=17` and `tasks._wrong_old=10`
  remain.
- Banned-constant scan: no new model-layout constants were introduced in owned production paths.
- Full fake-only suite: `410 passed, 10 failed` in the concurrent shared checkout. The introduced
  bounded conflicts are the v2/v4 historical reconstruction node plus legacy note-form assertions in
  `tests/test_pipeline.py`/`tests/test_probes.py`; remaining failures are concurrent SPEC-001
  cache, runner, registry, and fixture work. No model loading occurred.

## Compatibility dependency / observed not fixed

The protected B/C saved evaluations have no generator-version metadata. Their read-only
integrity renderer reconstructs tasks with `task_from_id`, which cannot distinguish historical
v2 `test-*` records from v4 Run D records sharing ids/actions. Preserving retroactive memo counts
requires an integrity/provenance owner to add and consume generator-version evidence; this task
did not alter those protected seams or artifacts.

## Changed paths and commit

Baseline: `18c4bd9`. Source implementation: `68e4dac feat: implement SPEC-003 Run D data
recipe`. No pending spec, protected data/output/report artifact, model, or GPU operation was
touched.

## Review-fix evidence

- RED: `uv run pytest -o addopts='' -q tests/test_tasks.py::test_run_d_conditional_and_recovery_notes_preserve_family_state`
  produced `4 failed` before the conditional/recovery state patch. `uv run pytest -o addopts='' -q
  tests/test_data.py::test_split_specs_preserve_chunk_boundaries_effects_hashes_and_recovery_repeats`
  produced `1 failed` (`KeyError: difficulty`) before split metadata plumbing. The two hash-oracle
  nodes produced `2 failed` before v4 was pinned.
- GREEN: `uv run pytest -o addopts='' -q tests/test_tasks.py tests/test_data.py tests/test_cli.py
  tests/test_integrity.py` produced `76 passed, 1 failed`; the one failure is the protected
  historical reconstruction node described above. The review-specific focused command produced
  `29 passed, 48 deselected`; the independent v4 hash rerun produced `2 passed`.
- `uv run ruff check` on owned production and test paths and owned-path `git diff --check` were
  clean. Strict `uv run ruff check --select C901 --config lint.mccabe.max-complexity=9` reports
  only pre-existing `cli.main=17` and `tasks._wrong_old=10`; it reports no new owned function.
- Full fake-only suite: `410 passed, 10 failed`. The introduced bounded legacy-note/version
  conflicts are the protected reconstruction node and old exact-form assertions in
  `tests/test_pipeline.py`/`tests/test_probes.py`; concurrent SPEC-001 registry/probe failures
  account for the remainder.
- Evidence/documentation commit already recorded: `c0e2893 docs: record SPEC-003 implementation
  evidence`. Before this review update, `git diff --numstat 18c4bd9 -- <owned paths>` recorded
  config `55/0` each (three files), plan `192/0`, task/data/CLI `99/64`, `173/39`, `34/2`, tests
  `104/2`, `135/2`, `94/1`, `71/14`, and report `66/0`.
- Review-fix implementation commit: `83ca7e1 fix: strengthen SPEC-003 Run D contracts`.

## Review fix round 2

No production change was needed: new independent acceptance oracles passed against the corrected
implementation. They require a supervised stale `list_files` recovery immediately after every
realized stale guessed read and assert the family-specific Hop/key, values/split, or loads state.
The completion oracle now requires completion-pattern notes to use `finish` and includes the
current pre-action in its derived queue. The data oracle uses all six literal Run D split values,
regenerates each expected task population independently, and compares task count, difficulty,
horizon/family/variant summaries, task IDs, hashes, and exact recovery counts by variant.

Round-2 verification: focused review `29 passed, 48 deselected`; owned modules `76 passed, 1
failed` (the protected historical provenance node); full fake-only `410 passed, 10 failed`.
Normal Ruff and owned diff checks are clean. Strict C901 max 9 reports only baseline `cli.main=17`
and `_wrong_old=10`.

## Issue #11 — subtractive v4 test migration

Following R10, removed eight obsolete phrase/parser assertions from `tests/test_pipeline.py` and
`tests/test_probes.py` and their now-unused helpers/imports. Their lasting generator behavior now
lives in `tests/test_tasks.py`: all four levels and realized variants prove action/file-derived
cross-reference hop/key and new-match choices, aggregate append-only values/numeric split/total,
batch worker-order progress/queue/verify structure, and conditional full loads/true maximum.
No production code or protected artifact changed.

The required legacy RED command returned `8 failed, 3 passed`: all eight named old tests failed
solely on retired v2/v3 note shapes. Green evidence: focused generator/integrity `24 passed, 39
deselected`; migrated tasks plus `test_pipeline.py` `72 passed`; affected pipeline/probe modules
have only the protected reanalysis contract failure. Full fake-only evidence is `412 passed, 2
failed`: the unchanged retrospective memo test and unchanged reanalysis metadata-shape test.

### Chief ruling request: historical replay version selection

Read-only inspection finds B/C JSON summaries contain `keep_last: 2` and run metadata (label,
model, adapter, split, stress, temperature, cache, elapsed time); trajectories contain `task_id`,
family, variant, prompt, steps, and verdict. They lack both `generator_version` and `data_seed`.
Current public signatures are `task_from_id(task_id: str, seed: int, difficulty: int | None =
None) -> Task` and `_render_evaluations(paths: list[Path], seed: int) -> str`.
`_analyse_evaluation(path, fallback_seed)` calls `task_from_id(task_id, seed,
difficulty=difficulty)` at `pipeline/integrity.py:469`, using the fallback when saved data seed is
absent. Identical historical `test-*` ids/actions therefore cannot select v1/v2 rather than v4
notes today. Please choose one policy: (1) versioned replay, adding/consuming saved generator
version metadata; or (2) named-version refusal for unversioned historical records. This task does
not implement either option.
