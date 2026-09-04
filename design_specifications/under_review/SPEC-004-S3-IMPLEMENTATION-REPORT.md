# SPEC-004 §3 block-ablation implementation report

Date: 2026-09-04
Lane: B — fake-only implementation
Dispatch base: `bdd972a126affc9905d5998629b31414af465714`

## Status

Implementation, focused tests, scoped static checks, and the exact R13 full fake-only suite are
complete. The first R13 attempt was red only in paths owned by other active lanes; its evidence
is retained below alongside the final integrated green gate.

No model, tokenizer, checkpoint, probe, MLX computation, or real evaluation was loaded or run.
No `agent-v2-*` command was invoked. No file below `outputs/`, `data/`, or `reports/` was written.

## Requirements and implementation

| Requirement | Implementation |
| --- | --- |
| Balanced consecutive blocks | `_layer_blocks` uses `divmod(num_layers, blocks)`, assigns the remainder to the earliest blocks, and rejects values outside `1 <= blocks <= num_layers`. The literal 7/3 oracle is `[(0, 1, 2), (3, 4), (5, 6)]`. |
| Conditions | `run_block_ablation` evaluates the full adapter, the empty adapter, and one leave-one-block-out condition per partition. Each condition records exact kept and removed layers and the masked-module count yielded by `capture.lora_block_mask`. |
| Screen aggregation | Each condition evaluates every `select.screen` cell. Overall and per-family successes/tasks are summed as counts, never averaged as per-cell rates. `evaluate.wilson` is applied to those exact aggregate counts. |
| Most costly removal | Cost is `full_adapter.success_rate - removal.success_rate`; ties use the lowest block index deterministically. |
| Exception safety | The mask is entered once per condition and exits on evaluation failure. `_reuse_loaded_policy` restores `evaluate.load_policy` in `finally`, so all cells reuse the one adapter-attached policy without leaving a process-global replacement behind. |
| CLI | Existing plural `--adapters` static mode remains. Ablation adds singular `--adapter`, `--blocks` (default 6), and `--screen`; missing and cross-mode flags fail before the GPU guard. The real gated path runs the existing guard, resolves a registry name to `spec.hf_id`, loads one attached policy, builds `ArchitectureView`, and records `spec.resolve(...).as_dict()`. |
| Evaluation files | The evaluator receives one unique JSON path below `<output>/evaluations/<condition>/` per screen cell. The task did not execute this gated path. |
| Final files | The CLI directly writes metadata-sized `ablation.json` and `ablation.md` under the caller's output directory, permitted by R11. JSON includes command, resolved model, adapter, screen path/cells, block count, control, anchors, every condition, and the most-costly removal. Markdown includes anchors and overall/per-family Wilson rows. |

## Files and current line counts

| File | Result | Lines | Diff at evidence capture |
| --- | --- | ---: | ---: |
| `src/local_llm_lab/probes/adapter_delta.py` | implementation and a green refactor of the pre-existing readout renderer | 1,085 | +406 / −30 |
| `tests/test_adapter_delta.py` | fake-only literal-oracle, orchestration, restoration, validation, and CLI tests | 480 | +331 / −1 |
| `docs/superpowers/plans/2026-09-04-spec-004-s3-block-ablation.md` | approved plan with completion marks/evidence | 99 | new |
| `design_specifications/under_review/SPEC-004-S3-IMPLEMENTATION-REPORT.md` | this report | new | new |
| `.superpowers/sdd/2026-09-04-spec-004-s3-block-ablation/` | ignored task report and append-only progress evidence | ignored | not committed |

No third-party dependency was added. No external public signature changed. The new public seam is
`run_block_ablation`; `_layer_blocks` and `_render_ablation_markdown` are module-private as
specified. Existing static analysis still enters through `main`, `analyse_adapter`,
`compare_adapters`, and `render_markdown`.

## Interface-map coverage

- §2.13: consumes `capture.lora_block_mask(view, keep_layers)` without modifying `capture.py`.
- §2.15: implements the `agent-v2-probe-delta --ablate --blocks N --screen <config>` extension
  while retaining singular `--adapter` for ablation and plural `--adapters` for static analysis.
- §4 gap 13: loads `select.screen` through `pipeline.cli.load_config` and calls
  `pipeline.evaluate.run_evaluation` once per condition/cell.
- §6: the no-model import check, block-mask coverage through a fake context, full fake-only
  pytest gate, banned-constant scan, and caller review are recorded here.
- R3/R10/R11/R13: pending documents remained read-only; tests stayed in
  `tests/test_adapter_delta.py`; final metadata writers are below 1 MiB and array-free; hand-off
  remains gated on a green exact full suite.

## Decision: leave one block out

The binding task ruling resolves SPEC-004 §3's conflicting wording in favor of full and empty
anchors plus leave-one-block-out conditions. This directly measures “the block whose removal
costs most” and matches the original research design's eight-condition interpretation for six
blocks. Every record retains exact `kept_layers` and `removed_layers`.

Cost if wrong: invert only each per-block keep set to a block-only condition. The partition,
anchors, count aggregation, Wilson intervals, metadata, and output layout can remain unchanged.

## TDD evidence

The first invocation of the required RED command inside the restricted sandbox exited 2 before
collection because uv could not open its existing cache. Re-running the exact command with cache
access produced the valid behavior RED:

```text
uv run pytest tests/test_adapter_delta.py -q
```

Exit 1 at HEAD `a9fe3701d69cec83d5ba7652c0c4d731e15c93a6`: 6 failed, 6 passed.
Failures were the absent `_layer_blocks`, `run_block_ablation`, `_reuse_loaded_policy`, CLI
`--blocks`/`--screen` flags, cross-mode validation, and ablation path. No model or MLX runtime was
touched.

After the minimal core, the core selection passed 4 tests. After the CLI/output slice, the full
focused file passed 12 tests. A registry-boundary correction then received its own RED/GREEN:

```text
uv run pytest tests/test_adapter_delta.py::test_ablation_cli_uses_one_loaded_policy_and_writes_resolved_metadata -q
```

RED exit 1: the new path incorrectly passed registry name `fake-model` to the model loader rather
than resolved `fake/hf`. GREEN exit 0 after both initial load and evaluator calls used
`spec.hf_id`.

Final focused command:

```text
uv run pytest tests/test_adapter_delta.py -q
```

Exit 0: 12 passed in 0.24 s. Tests use fake views, mask contexts, model/spec objects, policy
loaders, and evaluation summaries; they create only pytest temporary files.

## Scoped static and integration checks

```text
uv run ruff check src/local_llm_lab/probes/adapter_delta.py tests/test_adapter_delta.py --select C901
uv run ruff check src/local_llm_lab/probes/adapter_delta.py tests/test_adapter_delta.py
git diff --check -- src/local_llm_lab/probes/adapter_delta.py tests/test_adapter_delta.py
.venv/bin/python -c "import local_llm_lab.pipeline.cli, local_llm_lab.probes.state_probe, local_llm_lab.probes.assistant_axis, local_llm_lab.probes.adapter_delta, local_llm_lab.pipeline.jlens, local_llm_lab.pipeline.integrity"
```

Final result: every command exited 0; both Ruff runs printed `All checks passed!`; diff check and
the no-model import check emitted no output. The first C901 run identified the pre-existing
`render_markdown` at complexity 11; after focused tests were green its direction-readout block
was extracted without behavior change, and C901 passed.

The scoped added-line scan for `36`, `2048`, `35`, `<|im_end|>`, and `model.model.layers` emitted
no hits. No fixed projection-name list was added. The implementation derives layer count from
`ArchitectureView.num_layers` and uses the registry's resolved HF id.

## Protected paths and foreign work

Before implementation, 1,295 files under `outputs/`, `data/`, and `reports/` were inventoried
and hashed; their checksum manifest digest was
`984eeddabb66478078162bc8963bbcb697f08087864c4a29846506adccb10746`. After focused work,
the inventory was identical and all 1,295 hashes passed with zero mismatches. Only claimed paths
were edited. Foreign dirty/untracked changes were preserved, and the Git index remained empty.

## R13 full fake-only suite

First attempt HEAD: `bac1636a18a716dead70f3ef389a28d65064d1cd`
Command: `uv run pytest -q`
Exit code: 1
Counts: 4 failed, 450 passed, 0 xfailed
Failing nodes:

- `tests/test_integrity.py::test_retroactive_saved_evaluations_match_memo_contract`
- `tests/test_patch.py::test_position_groups_are_exact_and_fail_closed`
- `tests/test_patch.py::test_greedy_generate_uses_cached_masked_forwards`
- `tests/test_pipeline.py::test_detect_loop_flags_repetition_patterns`

All four were in separately claimed active lanes and did not touch this lane's owned paths.

Final integrated gate HEAD: `9f1c0a5164c4301b43682e0f43d07c60afeb7fa0`
Command: `uv run pytest -q`
Exit code: 0
Counts: 460 passed, 0 failed, 0 xfailed
Failing nodes: none

The exact run completed in 7.11 s. A separate read-only collection count returned `collected=460`;
the gate output contained only passing progress markers.

## Deferred gated verification and observed issues

- No real block ablation was run. A later authorized gated stage must run B and D3 and confirm
  policy reuse, runtime, evaluator artifacts, and behavioral localization on actual adapters.
- The planned 3B cost is about eight screen evaluations × twelve minutes, approximately 1.6
  hours per adapter; 4B costs remain empirical.
- A real run should verify that `ModelSpec.resolve` over the adapter-attached model reports the
  intended LoRA targets for that installed runtime. This task verifies the resolution boundary
  on complete fakes only.
- The four foreign R13 failures above were observed and not fixed because their files are outside
  this lane's claim.

## Review fix round 1 — reject static-only flags in ablation mode

The reviewer correctly identified that explicit `--no-base`, `--top`, and `--readout-layers`
arguments were accepted with `--ablate`. The parser now keeps `None` sentinels for these options,
rejects any explicit use before `_run_ablation_cli` (and therefore before the GPU guard), and
applies the unchanged static defaults (`False`, `16`, and `""`) only after ablation validation.

Focused RED command:

```text
uv run pytest tests/test_adapter_delta.py::test_cli_rejects_missing_and_cross_mode_ablation_arguments -q
```

Exit 1: 5 passed and the 3 new cases failed because `--no-base`, `--top 3`, and
`--readout-layers 1` each reached the fake GPU guard. After the minimal sentinel/default change,
the same command exited 0 with 8 passed. The full focused file then exited 0 with 15 passed.

Both scoped Ruff commands (including `--select C901`) printed `All checks passed!`, and the
scoped `git diff --check` exited 0 without output. Fresh exact R13 evidence at HEAD
`7ed1973775c4dccfd812d44b74022e02cad616b1`:

```text
uv run pytest -q
```

Exit 0: 481 passed, 0 failed, 0 xfailed; failing nodes: none. A separate read-only collection
count confirmed 481 tests. No model, tokenizer, checkpoint, probe, GPU/MLX computation, real
evaluation, or `agent-v2-*` command ran. The review's aggregation-oracle observation remains a
deferred Minor and was not expanded into this fix round.

A peer commit landed before the exact-path fix commit. The focused file and exact R13 command
were therefore rerun at actual fix commit `ffc6665656551dbf22acef64047196009c772b35` and again
exited 0 with 15 focused passes and 481 full-suite passes; no failing or xfailed nodes.
