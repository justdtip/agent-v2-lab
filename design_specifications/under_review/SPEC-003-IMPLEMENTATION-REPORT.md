# SPEC-003 implementation report: Run D data recipe

## Outcome

This implementation delivers SPEC-003 sections 1–3 only. It bumps the generator from v2 to v3,
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

The v3 values were reproduced in two independent pytest temporary directories before pinning.

## Verification

- RED generator/integrity: 9 intended failures.
- RED data: missing `SplitSpec` import error.
- RED CLI: raw-count conversion failure and missing Run D configs.
- Focused final: 70 passed, 1 deselected (historical compatibility blocker below).
- Scoped Ruff: clean; diff check: clean.
- Strict C901: only established `cli.main=17` and `tasks._wrong_old=10` remain.
- Banned-constant scan: no new model-layout constants were introduced in owned production paths.
- Full suite: 366 passed, 27 failed in the concurrent shared checkout. The introduced bounded
  conflicts are the v2/v3 historical reconstruction node plus legacy note-form assertions in
  `tests/test_pipeline.py`/`tests/test_probes.py`; remaining failures are concurrent SPEC-001
  cache, runner, registry, and fixture work. No model loading occurred.

## Compatibility dependency / observed not fixed

The protected B/C saved evaluations have no generator-version metadata. Their read-only
integrity renderer reconstructs tasks with `task_from_id`, which cannot distinguish historical
v2 `test-*` records from v3 Run D records sharing ids/actions. Preserving retroactive memo counts
requires an integrity/provenance owner to add and consume generator-version evidence; this task
did not alter those protected seams or artifacts.

## Changed paths and commit

The explicit-path commit is recorded in the task execution report after commit. No pending spec,
protected data/output/report artifact, model, or GPU operation was touched.
