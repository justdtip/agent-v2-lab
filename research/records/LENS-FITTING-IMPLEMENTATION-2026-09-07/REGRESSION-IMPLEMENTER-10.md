# Task 2 — regression and runtime handoff

Implemented on `codex/lens-fitting`, base `dd83be80ca31faaa435feb12f11f631a020c4571`, inside the persistent worktree. Owned files only: `regression.py`, `runtime.py`, `artifacts.py`, `scripts/lens_fit.py`, `tests/test_lens_regression.py`, `tests/test_lens_runtime.py`. Parent claim revision 1 covers these paths; active board was read, no conflicting action. No corpus writes, adapters, registry/architecture/runner/live-lens changes, checkpoint loads, network/downloads, merges, pushes, or full-suite launch.

## Implemented contracts

- `regression.ALPHA_GRID` is exactly `(0.001, 0.01, 0.1, 1.0, 10.0)` with no grid override argument. `SufficientStats.zeros/add`, `accumulate`, `squared_error`, `solve_layer`, `fit_regression` provide the small numerical API.
- `accumulate(view, rows)` calls `view.residuals(ids, tuple(range(1, depth+1)))` once for each retained sequence. It casts residuals to float32 before products. Separate fit and held sums contain `XTX`, `XTY`, scalar `trace(YTY)`, and integer position counts. All updated layer sums are evaluated every sequence, then residual references dropped. There is no activation artifact.
- `solve_layer` solves row-oriented `(XTX + alpha*dbar*I) W = XTY` for every alpha with MLX's CPU-only linear solve. Accumulators remain float32 MLX arrays. Reports all candidate errors and selected alpha, dbar (total mean diagonal), n, absolute penalty, equivalent per-position lambda, split counts, held target energy, relative SSE and explicitly uncentered R². Fixed-grid errors fail the operation rather than skipping a candidate. Raw float32 SSE is retained, with cancellation caveat, rather than silently clamped; very small negative SSE/R² slightly above one can therefore reflect rounding.
- `fit_regression` returns `RegressionResult(maps, per_layer, counts, elapsed_s, peak_memory_gib)` with maps keyed by repository residual index. Stored maps are `J=W.T`. Each layer's sufficient sums are popped/released after solve; no fit/held matrices remain when artifact round-trip begins. Peak is the MLX process peak (including loading), not an OS-wide RSS metric.
- `artifacts.validate_output` checks suffix, either existing artifact/symlink, and parent structure. `write_lens` requires every nonfinal map, finite square float32 arrays, bounded archive size below decimal 1 GB, exclusive NPZ/JSON creation, and strict unchanged `LensMaps.load` plus exact all-layer array comparison. Existing artifacts are never overwritten. A failure after exclusive creation leaves partial files for explicit diagnosis; a fresh output path is needed.
- `runtime.prepare_fit(corpus_path, spec, output, kind='regression', revision='main')` uses verified `corpus.read_corpus`, validates model/split/output naming, pins exact manifest-file SHA, resolves an existing local snapshot offline, and checks vocabulary limits when config supplies them. Output basename must include model name, domain, kind. No adapter option.
- `runtime.primary_worktree()` reads Git's first worktree record using sanctioned `spawn.run`; no unsafe subprocess call after Metal.
- `runtime.resolve_snapshot` uses explicit `HF_HUB_CACHE`/`HF_HOME` when present, otherwise primary checkout `.cache/huggingface/hub`, always `local_files_only=True`. It does not search for a newest snapshot after loading. `snapshot_identity` binds original HF ID, immutable cached commit (or explicit local content identity without invented Hub revision), exact config, every MLX-selected `model*.safetensors`, weight indices, tokenizer/config/custom-code assets and their individual hashes/sizes. Missing index-referenced shards fail before load. Full identity is rechecked before and after loading.
- `runtime.load_runtime(prepared, capture=False)` holds the PRIMARY process lock before importing `load_policy`. It binds ONLY `runlock.PROJECT_ROOT` to primary for the process so `load_weights`' subsequent default acquisition names the same lock; `project.PROJECT_ROOT` and ordinary worktree paths remain unchanged. The local snapshot directory is passed via `replace(spec, hf_id=...)` through `load_policy`. Returned `LoadedRuntime.spec` and `resolved.spec` restore the original HF ID, with the verified resolved revision. Actual cache resolution metadata is preserved. Future capture callers use `capture=True`, which resolves `cache_strategy='none'` before loading. No lock clearing or process termination.
- CLI is regression-only now; Task 3 may extend it. Sidecar includes exact hashes, dimensions/layers, penalties, counts, times, process peak and convention. It explicitly calls held sequences alpha selection that may share trajectories with fit, and disjoint pilot generalisation unmeasured.

## Evidence

All commands use `PYTHONPATH=src`, `/Users/daniel.tipton/Desktop/An app/.venv/bin/python`, the persistent worktree, and escalated permissions where writes/native testing require them.

1. Red runtime run: `python -m pytest -q tests/test_lens_runtime.py` — 7 expected missing-module/script failures before implementation.
2. Pure primary inventory before native red launch: `read_lock(primary/outputs/.model-run.lock)` printed `None`; `running_model_processes()` printed `[]`.
3. Red native run: `python -m pytest -q tests/test_lens_regression.py` — 8 missing-module failures before implementation, no checkpoint.
4. Initial runtime green attempt: 6 pass, 1 test fixture failure because repository conftest overrides `default_lock_path`; corrected the test to exercise real temporary explicit and default holds with the production root formula. Then 7 passed.
5. Fresh inventory again `None` / `[]`; native run 8 passed in 1.27 s.
6. Added primary-cache test first failed because runtime still used ordinary worktree cache. Changed resolver to explicit primary cache default. Then 12 pure runtime tests passed.
7. Added CLI help/invalid-input/adapter refusal, snapshot changes before/during loading and first-Git-record tests; 17 pure runtime tests passed. CLI tests assert no `mlx` module import in subprocesses.
8. Added all-candidate direct-error/held-selection check initially failed at a too-tight absolute tolerance: native float32 SSE 161.640533 vs independent float64 161.640559 (relative discrepancy ~1.6e-7). Changed test to `rel=2e-6, abs=2e-6`, consistent with float32 sums, without a production change.
9. Final fresh inventory immediately before native launch: primary lock `None`, mapped processes `[]`.
10. **Final focused run:** `python -m pytest -q tests/test_lens_regression.py tests/test_lens_runtime.py` — **26 passed in 1.35 s**, exit 0.
11. **Final lint:** Ruff check of the six owned code/test paths — all checks passed, exit 0. Ruff format finished. Earlier non-escalated Ruff cache write was sandbox-denied and retried escalated; early formatting/import/E501 findings resolved. `git diff --check` exit 0; index empty before staging.

Tests demonstrate replication-invariant ridge scaling; direct vs sufficient SSE; held targets changing optimal alpha; one all-layer forward and separated sums; tiny installed Qwen3.5 residual recovery of a nonsymmetric known linear map (relative weight error bound <0.08, transpose substantially worse); all-layer float32 archive/readout; finite/shape/size/no-overwrite refusal; exact identity/missing shards/mutation refusal; pre-load model/corpus/output errors; primary lock refusal preserving foreign bytes; idempotent explicit/default primary lock holds with unchanged ordinary project root; capture none; CLI hashes/sidecar/caveats and no checkpoint.

## Integration and limits

- `tests/test_lens_regression.py` explicitly imports MLX at top level and is native at collection AND execution. It requires a free MLX slot. Parent will merge ROOTS-FIX and add this file to the exact `_TESTS_THAT_LOAD_MLX` pin. No guard algorithm/pin was edited here; full guard/suite intentionally not run.
- Actual checkpoint-scale sub-hour timing, measured peak/buffer pressure, fit quality and scientific pilot generalisation remain unmeasured. No real checkpoint fit, acceptance reading, or profile was produced. Parent's analytic dimensions imply raw float32 archive payload 812,646,400 bytes, under 1 GB, but this is not measured peak-memory evidence.
- Parent owns run-window declaration, heartbeat launch/end records, independent review, whole-suite integration and subsequent Task 3 CLI extension.

## Commit

`f86198b12e004fdf9d4adceaa07c9f798191b372` — `Fit all-layer regression lenses with pinned runtime provenance`.

Exactly the six owned code/tests committed. The parent-owned untracked `SOURCE-NOTES-05.md` and `PILOT-REPLAY-INVENTORY-06.json` were preserved and not staged. No task-2 tracked changes remain.

## ROOTS-FIX integration and offline snapshot correction

Follow-up base: `3f430bdd27d258cfc5bbfe573af82a3e6d4f086c`, incorporating parent's merge of ROOTS-FIX `5c1c9e1`. Parent explicitly extended scope to the native pin in `tests/test_repository_rules.py`; claim revision 2 was read from the primary board. The only guard change is sorted insertion of `test_lens_regression.py` into `_TESTS_THAT_LOAD_MLX`. No walker algorithm, grep expectation, other guard hunks, primary dirty issue-89 changes, or parent records were changed.

Exact pin red/green:

- `python -m pytest -q tests/test_repository_rules.py::test_the_tests_that_load_mlx_are_the_pinned_set tests/test_repository_rules.py::test_the_grep_that_this_rule_replaces_is_wrong_in_both_directions` initially produced one expected failure (missing `test_lens_regression.py` at sorted index 7) and one pass.
- After the one-line pin, both passed. The grep expectation needs no change.
- Primary inventory then printed `None` / `[]`; `python -m pytest -q tests/test_lens_regression.py tests/test_lens_runtime.py tests/test_repository_rules.py` passed all **58** tests before the subsequent runtime-only correction. Ruff all seven source/test paths and diff check passed.

Parent's actual checkpoint-free `prepare_fit` smoke found an additional runtime error: unrestricted `snapshot_download(local_files_only=True)` raised `IncompleteSnapshotError` on the real complete model snapshot because Hub `.gitattributes` and `README.md` were absent. This was not a missing weight and required no download.

Added a synthetic on-disk Hugging Face cache/tree regression using the installed library (no checkpoint/HTTP): unrestricted offline resolution reproduces the missing-doc failure, but lens resolution must accept all present runtime assets. Removing a model shard must still fail. The initial new test failed at the same missing README/.gitattributes check. A first edit missed the formatted call's argument line and both expected pattern tests still failed; adding the actual argument resolved it. `resolve_snapshot` now supplies `SNAPSHOT_ALLOW_PATTERNS`, matching installed `mlx_lm.utils._download` asset classes. The same set drives recursive provenance hashing using repository-relative names, including nested templates/code/config assets and `.jsonl`; missing index-referenced shards remain rejected. No docs downloaded and no missing-weight check removed.

Pure runtime verification after this fix: **18 passed**. Actual validation-only smoke succeeded:

- Corpus: `data/lens-fitting/qwen35-4b-agentic/manifest.json`, **518 sequences**.
- Requested output: `models/lenses/qwen35-4b-base-agentic-regression-2026-09-07.npz`.
- Revision: `32f3e8ecf65426fc3306969496342d504bfa13f3`.
- Snapshot SHA256: `d8e5ca95bde5faf0c52628b26ae58be3d0439c5d2a4107cb59081e051f5ebd9b`.
- Exact corpus-manifest file SHA256: `e85795b69a20c877787893ec30662f3995d58322dfc674519f786eb75de5d3e0`.
- Hashed assets: `chat_template.jinja`, `config.json`, `model.safetensors`, `model.safetensors.index.json`, `preprocessor_config.json`, `processor_config.json`, `tokenizer.json`, `tokenizer_config.json`, `video_preprocessor_config.json`, `vocab.json`.
- Assertions confirmed no `mlx` module imported and neither requested NPZ nor sidecar created. No weights loaded and no network call.

A fresh inventory for the final native rerun printed lock `None` but reported mapped MLX PID **21027**, another scratch `wt-84` pytest process. **No native rerun was launched**, nothing killed or cleared. Parent directed completion with pure checks and deferred consolidated native/full verification to a free slot. The nine numerical tests' passing evidence remains explicitly bound to source `f86198b` (and the earlier 58-test integration run before this runtime-only change).

Final current-tree verification: `python -m pytest -q tests/test_lens_runtime.py tests/test_repository_rules.py` — **50 passed**, exit 0. Ruff check across the seven changed/owned source/test files — all checks passed, exit 0. `git diff --check` exit 0. The final diff touches only `runtime.py`, `tests/test_lens_runtime.py`, and the one native pin line; index was empty before staging.

Follow-up commit: `1acabd67a20068ae25c76be96c1a768e1b17a1db` — `Resolve offline lens snapshots using runtime assets and pin native tests`. Exactly the three reviewed follow-up paths committed; parent untracked records preserved.
