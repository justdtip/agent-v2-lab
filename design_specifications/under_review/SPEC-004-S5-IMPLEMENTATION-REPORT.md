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

- HEAD: `bac1636a18a716dead70f3ef389a28d65064d1cd`
- Command: `uv run pytest -q -o addopts='' --tb=no`
- Exit code: 0
- Counts: 460 passed, 0 failed, 0 xfailed
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
- This report is committed separately after its final evidence update.
