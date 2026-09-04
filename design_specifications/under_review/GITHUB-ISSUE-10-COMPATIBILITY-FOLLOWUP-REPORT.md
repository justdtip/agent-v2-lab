# GitHub Issue #10 compatibility follow-up — Task 1

## Scope

Threaded the selected immutable `ModelSpec` through branch-mining prompt rendering while preserving
the public `run_branch_mining(model_name, adapter, ...)` caller contract. The pre-migration loader
still receives `spec.hf_id`; `pipeline.evaluate` was not changed.

## RED/GREEN evidence

```text
uv run pytest -q tests/test_branch.py -k 'active_spec or active_model'
RED: 2 failed
- _continue() got an unexpected keyword argument 'spec'
- mine_pairs() got an unexpected keyword argument 'spec'

GREEN: 2 passed
uv run pytest -q tests/test_branch.py
6 passed
uv run ruff check src/local_llm_lab/pipeline/branch.py tests/test_branch.py
All checks passed!
uv run ruff check --select C901 src/local_llm_lab/pipeline/branch.py
All checks passed!
python3 -m py_compile src/local_llm_lab/pipeline/branch.py tests/test_branch.py
exit 0
git diff --check -- src/local_llm_lab/pipeline/branch.py tests/test_branch.py
exit 0
```

## Changed behavior and self-review

- `_continue` and `mine_pairs` require keyword-only `spec` and pass that exact declaration to
  every direct `build_prompt` call.
- Seed `run_task` and branch continuation receive the same declaration; mutation-sensitive fakes
  record all three seams.
- `run_branch_mining` resolves `load_model_spec(model_name)` exactly once, loads `spec.hf_id`, and
  passes the declaration to `mine_pairs` without changing its public parameters.
- The focused diff is limited to `branch.py`, `test_branch.py`, and this report. Existing branch
  tests explicitly supply the legacy declaration for the newly required API.

## Safety and concerns

No real model, tokenizer, checkpoint, adapter, evaluation, CLI, probe, preflight, cache,
training, or J-space workload ran. All model seams were faked. No protected data/output path was
written. There are no remaining Task 1 concerns.
