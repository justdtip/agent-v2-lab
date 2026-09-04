# SPEC-001 Sections 6 and 9 — Implementation Report

## Status

Implementation is complete through the verified fix round. The authoritative integrated fake-only
gate passed at committed HEAD `37709c33edcf8822d369b6ee3e8cd91caee8343f`; this report is
committed separately below.

## Requirement-to-evidence map

| Requirement | Implementation and evidence |
| --- | --- |
| Deterministic run provenance | `b09f97faeeb059eab3f3112abf6239e67a86fe29` adds `source_tree_hashes` and the stable provenance schema. It records command, generator version, Git metadata, package versions, resolved-or-declared model metadata, sorted source/config/lock hashes, and caller extras with deterministic JSON serialization and a trailing newline. `tests/test_provenance.py` covers the schema and deterministic ordering. |
| Architecture-aware training LoRA resolution | `d3a4721b5084ee244686db9e48cb000c97575654` removes static CLI target keys. Training derives an effective frozen model specification from configured rank, scale, dropout, and key policy; it loads the registry `hf_id`, lets `ModelSpec.resolve` discover targets, clears only the temporary model cache, and emits post-success training provenance. `lora_config` consumes `ResolvedSpec`, preserving resume and training settings while producing a fresh resolved-key list. |
| Evaluation provenance | `17981f00f0fad68d328fb31c68b34e35741be87e` collects policy summaries in execution order and writes exactly one post-loop record using `resolved=None`, one raw loaded `ModelSpec`, and `{"stage": "eval", "evaluations": summaries}`. |
| Rollout provenance | The same commit captures the rollout return summary and writes exactly one post-run record using `resolved=None`, one raw loaded `ModelSpec`, and `{"stage": "rollout", "summary": summary}`. |
| `all` evaluation semantics | The same commit makes the best adapter the default policy, sets `--limit` to 180, and adds `--base` and `--stress`. `all --base --stress` invokes one evaluation call with the explicit best-adapter path plus `base=True` and `stress=True`; `--limit 17` overrides the default. |
| Review round 1 remediation | `f718aca1bb52468008d12e60beb1ef742a624738` places base-model loading inside `_resolve_training_spec`'s protected region, deletes temporary model/tokenizer references before cache clear, and retains cleanup on loading or resolution errors. Its tests also make eval provenance ordering/no-partial-write behavior and Git command/cwd/raw-binary-digest mechanics mutation-resistant. |

## Verification

- RED: before Task 3 production edits, the focused fake tests failed as intended: eval and rollout made no provenance calls; default `all` made two legacy calls with no 180 limit; and `all --base --stress` was rejected by argparse.
- GREEN: `.venv/bin/python -m pytest -q tests/test_cli.py::test_stage_eval_writes_one_ordered_provenance_record tests/test_cli.py::test_stage_rollout_writes_one_post_run_provenance_record tests/test_cli.py::test_main_all_wires_one_best_adapter_evaluation` passed (`5 passed`).
- Review round 1 RED/GREEN: lifecycle fakes changed from `2 failed, 1 passed` to `3 passed`; strengthened eval ordering/failure tests passed (`2 passed`); and exact Git provenance mechanics passed (`1 passed`).
- Focused: `.venv/bin/python -m pytest -q tests/test_provenance.py tests/test_cli.py tests/test_selection.py` passed (`26 passed`).
- Static: `.venv/bin/ruff check src/local_llm_lab/pipeline/cli.py tests/test_cli.py` and scoped `git diff --check` passed. Strict C901 at max 9 reports only inherited `cli.main=17`; the Task 3 base has the same result.
- Historical R13: `uv run pytest -o addopts='' -q` at committed HEAD `26ce0afff479b4bf50824c0698e6a2119304bb95` passed (`478 passed in 8.59s`).
- Authoritative R13: the same command at exact committed HEAD `37709c33edcf8822d369b6ee3e8cd91caee8343f` passed (`498 passed in 7.44s`).

## Compatibility and safety notes

- `stage_eval` and `stage_rollout` retain their existing signatures and output/transcript paths. Provenance happens only after successful evaluation-loop completion or rollout return.
- The training flow retains configured resume, iteration, checkpoint, and other YAML settings; only static LoRA target selection changed to resolved architecture metadata.
- Every new test replaces model loading, resolution, transcript setup, subprocesses, evaluation/rollout, or provenance writing as appropriate. No model, checkpoint, tokenizer, training, evaluation, data-generation, or rollout workload was executed.

## Residual risks

No known implementation concerns. The sole strict-complexity result is pre-existing `cli.main=17` and is unchanged from the Task 3 baseline.

## Commits

- `b09f97faeeb059eab3f3112abf6239e67a86fe29` — deterministic provenance core.
- `d3a4721b5084ee244686db9e48cb000c97575654` — resolved training LoRA targets.
- `17981f00f0fad68d328fb31c68b34e35741be87e` — evaluation/rollout provenance and `all` wiring.
- `f718aca1bb52468008d12e60beb1ef742a624738` — lifecycle cleanup and provenance-oracle fixes.
