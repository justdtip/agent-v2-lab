# SPEC-001 §10 preflight implementation report

## Approved scope

This change adds the fake-testable `agent-pipeline preflight --model <name>` stage, its
deterministic JSON evidence artifact, and one central pre-load guard for `train`, `select`,
`eval`, and `all`. It deliberately does not edit or wire probe commands; that remains a
follow-up for the owning probe lane after its guard interface is released.

## Artifact and guard contract

The artifact is written directly to `outputs/preflight/<registered-model-name>.json` with
`json.dumps(..., indent=2, sort_keys=True) + "\\n"` (R11). Stable top-level identity/status
fields are `schema_version`, `model_name`, `hf_id`, `snapshot_revision`, and `passed`.

Inspection records architecture counts and dimensions, native cache entry classes and resolved
strategy, exact 64-token residual equivalence, middle-layer JVP method/finite status,
four thinking-mode prompts and token counts, resolved LoRA paths/count, and the parameter plus
activation memory estimate. The activation formula is exactly
`batch_size * max_seq_length * hidden_size * num_layers * 4`; all components and the budget
result are serialized.

`require_preflight` reads the direct artifact, then checks its mapping shape, registered name,
`hf_id`, current cache `refs/main` revision, passed status, and memory budget before any caller
can load a model. `skip=True` returns without reading either cache or artifact. The cache
revision is discovered after `configure_local_cache()` from
`${HF_HOME}/hub/models--<org>--<repo>/refs/main`, without eager model imports.

## Rulings carried forward

- R10: module tests are in `tests/test_preflight.py`; CLI wiring tests are in `tests/test_cli.py`.
- R11: the small JSON metadata artifact is directly written, not routed through NPZ/array output
  machinery.
- R13: focused checks, Ruff, scoped diff checking, collect-only count, and one exact full suite
  are recorded below.
- Reports exclude elapsed/operational measurements. Those are reserved for the controller's later
  serialized real-model execution record.

## Fake-only TDD evidence

RED: `uv run pytest -q tests/test_preflight.py` exited nonzero during collection with
`ModuleNotFoundError: local_llm_lab.pipeline.preflight` before the module existed.

GREEN: `uv run pytest -q tests/test_preflight.py` passed 11 tests after the core and guard were
implemented. The report tests inject loader, view, resolver, JVP, revision reader, and array
backend fakes; no checkpoint is loaded or downloaded. The fallback test feeds a non-finite
forward JVP and verifies finite-difference selection.

RED: `uv run pytest -q tests/test_cli.py -k 'preflight'` failed six tests because the parser,
flags, and guard wiring were absent.

GREEN: `uv run pytest -q tests/test_preflight.py tests/test_cli.py` passed 36 tests. The CLI
tests replace the executor, guard, and stages; no checkpoint is loaded.

Focused lint: `uv run ruff check src/local_llm_lab/pipeline/preflight.py
src/local_llm_lab/pipeline/cli.py tests/test_preflight.py tests/test_cli.py` exited 0 with
`All checks passed!`.

## Baseline and files

The approved baseline was `65e93fbb91e0e077413d13b4c9be905f7c08eee8` with 498 passing tests.
Changed task files are:

- `src/local_llm_lab/pipeline/preflight.py`
- `src/local_llm_lab/pipeline/cli.py`
- `tests/test_preflight.py`
- `tests/test_cli.py`
- `design_specifications/under_review/SPEC-001-S10-PREFLIGHT-IMPLEMENTATION-REPORT.md`

## R13 final verification

Immediately before the full run, `git rev-parse HEAD` returned
`d4a5f8b66bfc1c5e2a31dcf0986531c552c9500d`; `uv run pytest --collect-only -q` collected 523
nodes. The exact required `uv run pytest -q` then exited 0 with 523 passing tests. No real
preflight/model command was executed.

## Independent review and controller verification

The principal review found one Important issue: the first implementation did not validate the
top-level `schema_version`. Fix commit `84992f2` now rejects missing or unsupported schema
versions, and the scoped re-review approved the fix. The final whole-change review approved both
spec compliance and engineering quality with no Critical or Important findings. It retained three
non-blocking follow-ups: normalize invalid-UTF-8 artifact errors, add raised/non-finite JVP fallback
tests, and extend true skip-flag coverage to select/eval/all.

Immediately before serialized model execution, the controller recorded HEAD
`001a35da2e85ad6d90a716381865f60b5b7d2173`, 526 collected nodes, 37 focused preflight/CLI tests,
scoped Ruff success, and exit 0 from exact `uv run pytest -q` with 526 passing tests.

## Controller-only `qwen35-4b` execution evidence

The controller re-listed the active board, acquired exclusive action `model-execution` at claim
revision 3, and immediately ran exactly:

```text
uv run agent-pipeline preflight --model qwen35-4b
```

The command downloaded 10 files into the project-local Hugging Face cache and exited 0. The
download progress reported 4 minutes 9 seconds; end-to-end command/release handling was under
5 minutes. The exclusive action was removed promptly at claim revision 4. No training,
evaluation, probe, push, second model command, or skip override was run.

The sole artifact is `outputs/preflight/qwen35-4b.json` (2.8 KiB), SHA-256
`499efee17d6ba76f389ff975dbc227d7e569d5e4836963be4260fcafba9d5679`. It matches cached
revision `32f3e8ecf65426fc3306969496342d504bfa13f3` and records:

- architecture: 32 layers (24 `linear_attention`, 8 `attention`), hidden size 2560,
  vocabulary 248320, tied embeddings;
- residual equivalence: 64 tokens, maximum absolute error `2.5591506958007812`, **failed**;
- JVP: layer 16, forward mode did not yield usable finite evidence, finite-difference fallback
  finite;
- cache: 24 `ArraysCache` and 8 `KVCache` entries; strategy `none` because equivalence remains
  unverified;
- LoRA: 12 resolved target suffixes and 32,464,896 trainable parameters;
- memory estimate: 2,367,118,848 parameter bytes plus 1,761,607,680 activation bytes,
  4,128,726,528 bytes total (3.8451762199401855 GiB), within the 22 GiB budget;
- prompt token counts: unsupported 24, off 26, inference 24, trained 24; and
- overall `passed: false` because residual equivalence failed.

The cache footprint increased from 1.6 GiB to 4.5 GiB. Actual MLX peak memory was not emitted by
this deterministic-report command and is therefore unavailable after process exit; the report
does not infer it from the estimate. External API cost was zero: execution and inference were
local, with only an unauthenticated Hugging Face model download.

Finally, a no-model guard invocation rejected the artifact with `preflight did not pass for
qwen35-4b`, confirming train/select/eval remain fail-closed. The failed result must not be bypassed
without a new explicit decision; the residual mismatch requires a separately scoped diagnosis.
