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

## Controller-only execution evidence (reserved)

Do not fill this section during fake-only implementation. After independent approval, the
controller may record the one authorized `qwen35-4b` preflight execution here: wall time, memory
estimate and observed MLX peak if available, JVP method, cache kinds/strategy, local cache
footprint, and zero external API cost.
