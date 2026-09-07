# Regression resource preflight source — 7 September 2026

Parent-authored bounded addition to Task 6 while Task 3 implements separate files. scripts/lens_regression_preflight.py and tests/test_lens_regression_preflight.py only. No checkpoint or MLX execution in this source work.

The operational script validates an existing registration, frozen corpus/snapshot and unused planned lens path, creates an exclusive streaming JSONL report, and loads via the reviewed primary-lock load_runtime contract. It measures two copies of the longest fit row into the fit/held statistics storage slots solely for resource calibration. The labels here exercise allocation, not scientific split reassignment; no held data, maps or fit-quality values are written. It probes lengths 256, 512, 1024 and the corpus maximum (deduplicated/limited to that maximum). The first two sizes require a caller-supplied inspected bound. Later sizes are projected before execution from the last two measured peaks, using fixed model/sums storage plus a variable term growing quadratically or faster if measurements imply it. A projection above 0.6 of device working set refuses the next callback. Measured breaches remain labelled breaches. No window override is implemented.

A final complete fixed-alpha solve grid at one layer supplies a timing factor, with an explicit pre-solve allocation estimate (active bytes plus sixteen dense float32 matrices). It discards the map/error outputs. Projection includes every frozen sequence and every nonfinal-layer fixed-grid solve, but excludes loading/serialization and is not sub-hour acceptance. Cold forward timing is retained rather than hidden by a warm-up; caching allocator bytes are disabled and recorded. Actual fitting must use the same allocator policy to interpret this calibration. Byte metrics cannot bound live Metal buffer count, and the accumulation component is explicitly unobservable. Source and registration hashes are in the start event; no automatic fit follows.

## Verification before review

- RED: eight-function test file originally contained six tests; first six failed with FileNotFoundError because the script was absent.
- First GREEN attempt: four passed, two failed. An initial projection implementation unnecessarily reduced even a valid fixed floor; corrected it to reduce only a floor at or above the measured minimum. That keeps the variable term positive without distorting a valid independently known floor.
- Added registration-before-preparation and existing-report-before-loading refusal tests, both pure with loader spies.
- Focused final command: PYTHONPATH=src primary virtualenv python -m pytest -o addopts= -q tests/test_lens_regression_preflight.py: eight passed in 0.07s. Ruff format/check clean.
- Combined with repository guard during concurrent Task 3 source work: 39 passed, one guard-table failure. New function-local imports in test_lens_jacobian.py correctly extend the guard's empirical grep-versus-closure list; its owner has the narrow table amendment. No failure in the eight preflight tests. Consolidated guard evidence follows Task 3 completion.

Independent source review and actual resource measurements are still pending. No claim of safe full-corpus memory, under-hour completion or regression quality is made here.
