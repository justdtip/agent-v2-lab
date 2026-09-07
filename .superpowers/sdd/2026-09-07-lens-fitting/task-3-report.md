# Task 3 implementation report — 7 September 2026

Implemented source/tests for requirements §§2, 3.3 and 3.6 under task-3-brief.md. No checkpoint load, fit on research data, downloads, full suite, process killing, lock clearing, merges or pushes occurred. Parent revision2 primary claim was read and retained; no competing claim was published.

## Commits and ownership

- `d93c636` — frozen full-basis Jacobian fitting, validation, fit CLI extension, bounded runtime allocator/progress helpers, focused Jacobian/runtime tests, and the one empirical `held_but_clean` filename/count update in repository rules. No guard algorithm or native import filename pin changed. No top-level native imports were added.
- `3dc8b10` — separately authorized parent preflight review fixes in scripts/lens_regression_preflight.py and tests/test_lens_regression_preflight.py only.
- The final report commit also contains the exact zero-primal finite-difference step helper/test follow-up. See Git history for its immutable hash.

## APIs and measurement contracts

`lens_fitting.jacobian` exposes:

- `sample_positions(rows, split, count, seed)`: deterministic equal-span cycles, then uniform prompt and uniform token within the chosen span; sampling with replacement is explicit. Each draw is one prompt and one same source/target position. Selected records retain row/source/index, step/task/token-start identity when present, span, position, draw number and split. Fit draws only use fit rows. Frozen corpus validation remains owned by reviewed runtime/corpus modules.
- `make_plan`, `freeze_plan`, `read_plan`, `prepare_plan`: all 400 planned fit draws, at least150 held draws, seeds, three self-check layers, full dimension, fixed stopping rule, explicit self/response/stability bounds, exact corpus/snapshot identities and memory envelope are persisted before model execution. Exclusive writes reject overwrites. Read verifies digest, rows, architecture, snapshot and deterministic protocol reconstruction. No default scientific tolerances exist.
- `copy_cache`: supports exact native KVCache and ArraysCache classes only; preserves keys/values/offset, convolution/recurrence arrays, lengths and left_padding. Copies or broadcasts batch axes, preserving temporal values. Unknown cache states/subclasses fail closed. Empty KVCache never accesses its broken empty `.state` property. Position zero builds fresh native caches; the existing runner clone helper is untouched.
- `prepare_position`: executes the complete prefix through every block; creates a separate clone to derive the current residual. Snapshot mask construction therefore sees uniform prefix offsets. The original full-sequence primal is retained for the reference and its float32 norm sets epsilon.
- `pre_norm_tail`: explicitly runs view blocks, never ArchitectureView.tail or final_norm. Neither ArchitectureView nor live_lens semantics changed.
- `finite_difference_steps`, `reference_responses`, `cached_responses`, `full_jacobian`: float32 central finite differences with epsilon0.01 times the entire full-sequence primal norm divided by one tangent norm; absolute epsilon0.01 fallback for zero primal. The same helper is used by both paths. Batch shape and cached row norm cannot rescale epsilon. Restore uses a fresh cache per direction/sign; broadcast uses a fresh cache per batch/sign. Full basis is enumerated in bounded batches and stored output-by-input; no subspace substitute.
- `Convergence`: float32 equal-span running mean, curve every10 draws; no stability decisions below100; two consecutive changes below0.002 and minimum150 for acceptance. All400 without convergence is a failure, never accepted. Curve includes eligibility/streak and termination reason. Fit and validation use the same equal-span/source/target averaging.
- `self_check`, `require_self_check`: 16 random unit directions at three frozen layers, separately for restore and broadcast, compared with full-sequence uncached reference. Reference epsilon/half-epsilon stability uses predeclared bounds. Both paths must pass. Larger batch candidates get fresh independently checked proof before timing.
- `benchmark_plan`, `benchmark`, `require_benchmark`: batch8 first for both paths/all layers; project every larger batch up to64 before measuring, conservatively retaining the initial declared envelope and preceding measured peak scaling. Includes fit/held longest full-sequence and longest-prefix extrema and uses the worse timing. Projects preparation plus the complete basis for400 positions per layer. A projected cost above1500s/layer stops before any all-layer fit. Missing/failed proof, overstated device working set, measured envelope falsification and memory above0.6 working set fail closed. Byte measurements explicitly do not establish live Metal buffer-count safety.
- `fit_jacobian`: complete nonfinal-layer maps, source identities, seeds via plan, per-layer convergence curves/reasons/span counts/wall time/MLX peak. Failed convergence raises FitStopped with report rather than returning an accepted artifact.
- `run_jacobian_stage`: writes self-check proof before timing, benchmark before fit, fit evidence before held validation, and a stop report on gate errors. Every invocation uses fresh proof/timing; a previous process's benchmark is not substituted in the CLI path.

`lens_fitting.validation` exposes `validate_layer` and `validate_maps`. They measure32 shared random unit directions at held positions, compare epsilon with half-epsilon under frozen stability bounds, average measured responses by the same equal-span rule as fitting, and compare that mean to directions@J.T through existing response_agreement. A local derivative is never compared directly with the averaged map. Instability is inconclusive. Existing layer_verdict alone requests refits; concept/output hooks remain inconclusive/None. Detailed per-position stability and per-layer verdict records are retained.

## Operational CLI

Existing regression arguments/output/grid remain intact. New kind is `jacobian`; no adapter option.

1. `--kind jacobian --jacobian-stage plan --jacobian-plan PATH --plan-config CONFIG` (plus existing model/corpus/out/revision arguments) performs pure plan freeze only. CONFIG must contain exactly `seed`, `held_count`, `working_set_bytes`, `initial_peak_bytes`, `self_bounds`, `response_bounds`, `stability_bounds`; each bound has atol/rtol. Snapshot config supplies dimension/depth. Caller must justify values before execution; no values were scientifically chosen in this task.
2. `--jacobian-stage check|benchmark|fit --jacobian-plan PATH --record-dir NEW_DIRECTORY` uses reviewed prepare_fit/load_runtime, requires the frozen plan, and exclusively creates append-only run records. Check runs R52. Benchmark runs fresh R52 and measurements. Fit runs fresh R52/benchmark before all-layer fitting and held validation, then existing write_lens. Stop-required stages exit2 and save reasons.
3. Both kinds now call `configure_allocator_cache` after model.eval: set unused allocator cache limit0 and record previous/current bytes. Parent explicitly authorized matching the preflight allocation regime. This is not a buffer-count guarantee. `resource_snapshot` and CLI progress retain original events/counts and add elapsed, tokens done, process peak GiB and actual device-working-set share. Model loading remains included in the fit process peak.

Records live only under caller-selected run directories; no runnable launcher was placed under research/records.

## Evidence and failures

Commands use `PYTHONPATH=src '/Users/daniel.tipton/Desktop/An app/.venv/bin/python' -m pytest -q ...` from the persistent worktree. Every native launch was immediately preceded by pure runlock.read_lock(primary outputs/.model-run.lock) returning None and runlock.running_model_processes() returning []; parent was notified. No lock was acquired/cleared by these tiny tests.

- Initial pure Jacobian tests:5 expected missing-module failures, then5pass after implementation.
- Initial native tests: first sandboxed command aborted at native import (exit134); no numeric result claimed. Fresh empty inventory followed by escalated Metal-enabled retry produced3 expected missing-API failures (exit1), then the native implementation passed with8 total tests.
- Proof/benchmark/equal-span slice:3 expected missing-API/argument failures, then8 pure tests green. A test exposed a too-weak projection expression; fixed to retain the declared initial scaling before larger batches.
- Pure plan preparation:1 missing-API failure, then9 pure tests green.
- Shared allocator/CLI policy:2 missing-helper failures, then green fake-only CLI assertions of cache0 and preserved sidecar semantics.
- Shared R46 progress:2 missing-helper failures, then green fake-only elapsed/tokens/peak/share assertions.
- Projection falsification:1 genuine failure because a second candidate remained reachable; fixed to stop immediately after the observed peak falsifies its projected envelope.
- Zero-primal fallback:1 missing-helper failure, then pure red/green coverage of absolute0.01 fallback even for nonunit tangents; both paths use the helper.
- Tiny native integration additionally proves analytic full-basis orientation and epsilon, KV/SSM metadata preservation/immutability, cached/reference agreement at layers1/2/3 and positions0/3 for both paths, and an all-layer150-position fit with held32-direction mean validation. The analytic fit uses installed tiny Qwen3.5 embedding/cache interfaces with a known linear stand-in tail; it is not a claim that random initialized native nonlinear blocks have a constant analytic map. The separate native equivalence test uses actual tiny Qwen3.5 blocks.
- Final focused command: `... pytest -q tests/test_lens_jacobian.py tests/test_lens_runtime.py tests/test_lens_regression_preflight.py` —46pass, exit0 (native access escalated).
- Focused pure guard command before the final zero-primal helper: `... pytest -q tests/test_lens_jacobian.py tests/test_lens_runtime.py tests/test_repository_rules.py -k 'not native'` —62pass, exit0. An earlier sandboxed guard run passed but pytest emitted cache-write warnings; subsequent escalated focused verification was clean.
- Final `ruff check --no-cache` over all changed owned source/test files — clean, exit0. `git diff --check` — clean. Initial ruff attempt without --no-cache could not create a sandboxed cache; no product failure inferred. Formatting/lint issues were fixed before final verification.
- No full suite ran; consolidated verification and checkpoint runtime remain the parent's responsibility.

## Separately authorized regression preflight fixes

Independent parent review found that the first measured peak could falsify initial_bound_bytes while staying below cap and the second calibration would still run. Added a test with bound100/cap150/peak140 asserting only first callback runs. It failed before the fix. The run_ladder check now stops after a falsified initial bound, after preserving the existing above-cap breach classification.

The review also found solve peak included the preceding forward. A pure fake-backed main-path test asserts reset occurs immediately before solve and verifies successful measured_solve includes peak50 and projected74. It failed before implementation. Source now resets the peak before the grid and emits measured_solve (measured/projected bytes and elapsed) before existing cap classification. Preflight tests:8pass/2fail red;10pass green. Commit3dc8b10 contains only those source/test changes. No native preflight or checkpoint was launched.

## Honest limits / review attention

No real-model R52 acceptance numbers, memory/timing calibration, convergence claim, saved scientific fit or validated research lens exists from this task. Numeric tolerances, actual memory registration and any over25-minute or above0.6-window Director ruling remain unprovided. Source intentionally has no bypass authorizing those runs. The supported cache surface is deliberately restricted to the two inspected exact native classes. Memory telemetry is MLX process active peak, not OS pressure or observable live buffer count; kernel-pressure policy and supervision stay with parent. Peak envelopes and time projections are conservative estimates requiring real preflight, not proof of safety/performance. Native tests are implementation evidence only. Additional integration review should focus on scientific bounds, real-model cached/uncached equivalence, native cache evolution and actual resource calibration before execution.
