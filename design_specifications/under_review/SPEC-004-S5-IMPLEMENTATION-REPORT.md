# SPEC-004 §5 P6 causal patching — implementation report

## Scope

Implemented the fake-tested P6 vertical only: exact residual replacement in `InjectionHook`,
the gated `agent-v2-probe-patch` entry point, deterministic saved-evaluation selection,
transcript replay, token grouping, greedy generation, task-level Wilson aggregation, and
aggregate-only JSON/Markdown rendering.

## Decisions and evidence

- `replace=True` keeps the established additive path unchanged. Replacement preserves ordered,
  de-duplicated absolute positions, maps rank-two vectors row-for-row across cached forwards,
  validates width/count, rejects non-default alpha, preserves output dtype, counters,
  offsetless-cache bookkeeping, and restoration.
- P6 selects only B-success/C-first-`value_drop` `aggregate_report` and `ledger_reconcile`
  records and reconstructs tasks using C's recorded seed and difficulty. Replay changes only
  the immediately preceding C note to `render_expert_note`.
- All grouping/generation/CLI tests use small fakes. The CLI validates inputs before registry or
  model loading and was never executed outside those fakes.

RED/GREEN evidence:

```text
uv run pytest -q tests/test_capture.py -k replace
RED: 5 failures, each raised the original
NotImplementedError: replace=True is owned by SPEC-004

uv run pytest -q tests/test_capture.py -k replace
GREEN: 5 passed

uv run pytest -q tests/test_patch.py
RED: 7 failures importing the absent local_llm_lab.probes.patch module
GREEN: 9 passed after implementation
```

Focused/static/import evidence:

```text
uv run pytest -q tests/test_capture.py tests/test_patch.py
24 passed
uv run ruff check src/local_llm_lab/probes/capture.py src/local_llm_lab/probes/patch.py tests/test_capture.py tests/test_patch.py
All checks passed!
uv run ruff check --select C901 src/local_llm_lab/probes/capture.py src/local_llm_lab/probes/patch.py
All checks passed!
python3 -m py_compile src/local_llm_lab/probes/capture.py src/local_llm_lab/probes/patch.py tests/test_capture.py tests/test_patch.py
exit 0
git diff --check -- [owned paths]
exit 0
```

## R13 full fake-only suite

- Tested code commit: `d75415dd566fbb7ea41312a346bc744f36bbdbf0`
- Required command: `uv run pytest -q`
- Exit code: 0
- Summary-enabled equivalent: `uv run pytest -q -o addopts='' --tb=no` (exit 0)
- Counts: 505 passed, 0 failed, 0 xfailed
- Failing nodes: none

## No-model and protected-path evidence

No model, checkpoint, real tokenizer, adapter, probe CLI, preflight, cache-equivalence, or
J-space workload was invoked. Every executed probe/capture/CLI path used monkeypatched fakes.
No files under `data/`, `outputs/`, `reports/`, `research/`, or `design_specifications/pending/`
were written by this lane; all test files use `tmp_path`.

## Changed tracked files

- `src/local_llm_lab/probes/capture.py`
- `src/local_llm_lab/probes/patch.py`
- `tests/test_capture.py`
- `tests/test_patch.py`
- `pyproject.toml`
- `design_specifications/under_review/SPEC-004-S5-IMPLEMENTATION-REPORT.md`

## Commit

- `dd036a48ce22dd47ed4623d02bfc676f5887d81e` — `feat: add fake-tested P6 causal patching`
- `a0b88a976639caf243b9f38bded53929d67f6eb7` — `fix: correct P6 causal patch controls`
- `d75415dd566fbb7ea41312a346bc744f36bbdbf0` — `fix: enforce P6 control cardinality`
- This report is committed separately after its final evidence update.

## Fix Round 1

The correction derives named source and target groups from each rendered replay, captures every
requested residual layer exactly once for each failing/counterfactual prompt, and replaces each
failing group with its counterpart. `unrelated_task` rotates the selected cases and deterministically
truncates/cycles rows; `random_positions` uses seeded, non-treatment positions and fails closed when
the required cardinality cannot be drawn. Each path is generated and scored independently. Layer
syntax (including non-finite values) is rejected before policy/model loading; resolved bounds remain
validated after model resolution. The flip oracle now checks the exact decision step.

RED evidence:

```text
uv run pytest -q tests/test_patch.py -k 'malformed_layers or named_groups or flip_scoring'
.FF
- malformed --layers reached load_policy before raising
- vertical capture received positions=(5,) instead of positions="all"
```

GREEN and verification evidence on `a0b88a9`:

```text
uv run pytest -q tests/test_patch.py -k 'malformed_layers or named_groups or flip_scoring'
3 passed
uv run pytest -q tests/test_patch.py
13 passed
uv run pytest -q tests/test_capture.py tests/test_patch.py
28 passed
uv run ruff check src/local_llm_lab/probes/capture.py src/local_llm_lab/probes/patch.py tests/test_capture.py tests/test_patch.py
All checks passed!
uv run ruff check --select C901 src/local_llm_lab/probes/capture.py src/local_llm_lab/probes/patch.py
All checks passed!
python3 -m py_compile src/local_llm_lab/probes/capture.py src/local_llm_lab/probes/patch.py tests/test_capture.py tests/test_patch.py
exit 0
git diff --check a0b88a9^ a0b88a9 -- src/local_llm_lab/probes/patch.py tests/test_patch.py
exit 0
uv run pytest -q
exit 0
uv run pytest -q -o addopts='' --tb=no
498 passed in 7.40s
```

## Fix Round 2

Random source and target samples now come from one RNG seeded exactly
`f"{seed}:{task_id}:{layer}:{group}"`; each contains the treatment target cardinality and excludes
its respective treatment group. The implementation fails closed when either candidate pool is too
small. Treatment now requires equal source/target cardinality rather than truncating or cycling;
only the unrelated-task rotation retains deterministic row cycling/truncation.

RED/GREEN evidence on the new mutation-sensitive fake vertical:

```text
uv run pytest -q tests/test_patch.py -k 'exact_trace or unequal_treatment'
RED: 2 failed
- random injections used the old suffixed seeds and did not match the exact ordered trace
- unequal treatment groups reached InjectionHook instead of failing closed

uv run pytest -q tests/test_patch.py -k 'exact_trace or unequal_treatment'
GREEN: 2 passed
uv run pytest -q tests/test_capture.py tests/test_patch.py
29 passed
uv run ruff check src/local_llm_lab/probes/capture.py src/local_llm_lab/probes/patch.py tests/test_capture.py tests/test_patch.py
All checks passed!
uv run ruff check --select C901 src/local_llm_lab/probes/capture.py src/local_llm_lab/probes/patch.py
All checks passed!
python3 -m py_compile src/local_llm_lab/probes/capture.py src/local_llm_lab/probes/patch.py tests/test_capture.py tests/test_patch.py
exit 0
git diff --check d75415d^ d75415d -- src/local_llm_lab/probes/patch.py tests/test_patch.py
exit 0
uv run pytest -q
exit 0
uv run pytest -q -o addopts='' --tb=no
505 passed in 7.26s
```
