# SPEC-004 §3 Block Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the adapter-delta `--ablate` stub with a fake-tested layer-block ablation workflow that reads a SPEC-002 screen config, evaluates full/empty/removal conditions, and reports family success with Wilson intervals.

**Architecture:** Keep the existing static adapter-delta analysis unchanged. Add pure block partitioning and summary aggregation helpers plus one orchestration seam in `adapter_delta.py`; the CLI loads one adapter-attached policy only in the real gated path, temporarily supplies that already-loaded policy to `evaluate.run_evaluation`, and applies `capture.lora_block_mask` for each condition. Tests use fake views, masks, loaders, and evaluation summaries, so this task never loads a model or runs evaluation.

**Tech Stack:** Python 3.13, argparse, pathlib, JSON/Markdown, pytest fakes, existing `ArchitectureView`, `capture.lora_block_mask`, `pipeline.cli.load_config`, `pipeline.evaluate.run_evaluation`, and `pipeline.evaluate.wilson`.

**Spec:** `design_specifications/pending/SPEC-004-probe-program-revision.md` §3 and §7, with `design_specifications/pending/02-INTERFACE-AND-WIRING-MAP.md` §§2.13, 2.15, 4.13, 6, and rulings R3/R10/R11/R13.

## Global Constraints

- Work only in `src/local_llm_lab/probes/adapter_delta.py`, `tests/test_adapter_delta.py`, this plan, `design_specifications/under_review/SPEC-004-S3-IMPLEMENTATION-REPORT.md`, and `.superpowers/sdd/2026-09-04-spec-004-s3-block-ablation/`.
- Use `capture.lora_block_mask(view, keep_layers) -> Iterator[int]` read-only; do not edit or claim `capture.py`.
- Replace the `--ablate` stub and add `--adapter <dir>`, `--blocks N`, and `--screen <config>` without breaking existing `--adapters` static-analysis mode.
- Evaluate the SPEC-002 `select.screen` cells via `evaluate.run_evaluation`; report full-adapter and empty-adapter anchors plus one leave-one-block-out condition per consecutive layer block.
- Ruling: interpret “the block whose removal costs most” as leave-one-block-out ablation. This matches the original research design and makes the required removal-cost statistic observable; each record still carries exact `kept_layers` and `removed_layers`. Cost if wrong: the per-block keep sets must be inverted to block-only conditions.
- Partition all `ArchitectureView.num_layers` into exactly `N` non-empty, consecutive, near-equal blocks; every layer appears in exactly one removed block, including when the layer count is not divisible by `N`.
- Every condition reports exact overall and per-family success counts, rates, and Wilson 95% intervals. The highest `full_adapter.success_rate - condition.success_rate` is `most_costly_removal`, with the lowest block index as the deterministic tie-break.
- Final JSON records the command line, resolved `ModelSpec`, screen config path and cells, adapter path, block count, controls, conditions, and most-costly removal. Final Markdown renders anchors and per-family condition rows with intervals.
- Tests are fake-only and belong in `tests/test_adapter_delta.py` under R10. Do not move historical tests from `tests/test_probes.py`.
- Do not load a checkpoint, tokenizer, model, or probe; do not execute MLX/model evaluation; do not run any `agent-v2-*` CLI. Do not write `outputs/`, `data/`, or `reports/`.
- Pending design documents are read-only under R3. Record decisions and discrepancies only in the implementation report and SDD ledger.
- R13 is a hard hand-off gate: before review, the implementation report section `R13 full fake-only suite` records the exact HEAD, the exact `uv run pytest -q` command, exit code, pass/fail/xfail counts, and every failing node. Hand off only on exit 0.
- Before review also record focused tests, scoped Ruff including C901, diff checks, banned-constant grep, no-model confirmation, and protected-path confirmation.
- Preserve all foreign dirty and staged work; stage only the five owned paths explicitly and commit with exact pathspecs. Do not switch branch/worktree, push, merge, reset, restore, stash, or clean.

---

### Task 1: Implement and document fake-only block ablation

**Files:**
- Modify: `src/local_llm_lab/probes/adapter_delta.py`
- Test: `tests/test_adapter_delta.py`
- Create: `design_specifications/under_review/SPEC-004-S3-IMPLEMENTATION-REPORT.md`
- Update: `.superpowers/sdd/2026-09-04-spec-004-s3-block-ablation/progress.md`

**Interfaces:**
- Consumes: `capture.lora_block_mask(view: ArchitectureView, keep_layers: Sequence[int]) -> Iterator[int]`, `pipeline.cli.load_config(path: Path) -> dict[str, Any]`, `pipeline.evaluate.run_evaluation(**kwargs) -> dict[str, Any]`, `pipeline.evaluate.wilson(successes: int, n: int) -> tuple[float, float]`, `ModelSpec.resolve(model, tokenizer).as_dict()`.
- Produces: `_layer_blocks(num_layers: int, blocks: int) -> list[tuple[int, ...]]`, `run_block_ablation(view, *, blocks, screen, evaluate_condition) -> dict[str, Any]`, `_render_ablation_markdown(payload: dict[str, Any]) -> str`, and CLI support for `--ablate --adapter --blocks --screen`.

- [x] **Step 1: Add failing partition and orchestration tests**

  Add literal-oracle tests that require 7 layers split into three consecutive blocks `[(0, 1, 2), (3, 4), (5, 6)]`, reject zero/more-blocks-than-layers, exercise full/empty/three leave-one-block-out conditions through a fake mask and fake screen evaluator, aggregate two screen cells by family, verify Wilson intervals and deterministic most-costly-removal selection, and verify mask restoration if evaluation raises. Name the concrete production mutation each test catches in a short comment or docstring.

- [x] **Step 2: Run RED and retain the expected failure**

  Run:

  ```text
  uv run pytest tests/test_adapter_delta.py -q
  ```

  Expected: fail because `_layer_blocks` / `run_block_ablation` and the CLI ablation behavior do not yet exist, not because a real model or MLX runtime was touched.

- [x] **Step 3: Implement the minimal block and summary core**

  Implement balanced consecutive partitions using `divmod(num_layers, blocks)`, validate `1 <= blocks <= num_layers`, aggregate exact `(successes, tasks)` overall and per family from `run_evaluation` summaries, call `wilson` on those exact counts, and build the full/empty/leave-one-block-out records. Use the full-adapter rate as the removal-cost control and make ties deterministic by block index.

- [x] **Step 4: Implement the gated CLI wiring**

  Preserve `--adapters` behavior for the existing static-analysis mode. Add singular `--adapter`, integer `--blocks` (default 6 for ablation), and path `--screen`; reject missing or cross-mode arguments with `parser.error`. In ablation mode only, run the existing GPU guard, load the one adapter-attached policy, create `ArchitectureView`, resolve the registry spec, read `select.screen`, and call `evaluate.run_evaluation` for each screen cell while a scoped, exception-safe policy-reuse context supplies the already-loaded model/tokenizer and `lora_block_mask` applies the condition. Restore every temporarily replaced callable in `finally`.

- [x] **Step 5: Write final metadata outputs and Markdown**

  Under the caller-provided output directory, write metadata-sized `ablation.json` and `ablation.md` directly (R11). Include command, resolved model, adapter, screen path/cells, block count, full/empty controls, every condition's kept/removed layers and overall/per-family Wilson records, and the most-costly removal. Place per-cell evaluator JSON below an `evaluations/` child in the requested output directory; no task-owned run invokes this path.

- [x] **Step 6: Run GREEN and focused static checks**

  Run:

  ```text
  uv run pytest tests/test_adapter_delta.py -q
  uv run ruff check src/local_llm_lab/probes/adapter_delta.py tests/test_adapter_delta.py --select C901
  uv run ruff check src/local_llm_lab/probes/adapter_delta.py tests/test_adapter_delta.py
  git diff --check -- src/local_llm_lab/probes/adapter_delta.py tests/test_adapter_delta.py
  ```

  Expected: all focused tests pass, C901 and normal Ruff exit 0, and diff check emits no output.

- [x] **Step 7: Write the implementation report and SDD evidence**

  Create `design_specifications/under_review/SPEC-004-S3-IMPLEMENTATION-REPORT.md` with requirement-to-code mapping, changed-file line counts, interface-map rows touched, the leave-one-block-out ruling and cost if wrong, TDD RED/GREEN output, focused/static check outputs, banned-constant grep results, fake-only/no-model statement, protected-path/status evidence, and anything requiring a future gated real run. Do not edit any pending document.

- [x] **Step 8: Run the R13 full fake-only suite before review**

  Capture `git rev-parse HEAD`, then run exactly:

  ```text
  uv run pytest -q
  ```

  Record under the exact report heading `R13 full fake-only suite`: HEAD, command, exit code, pass/fail/xfail counts, and failing node IDs (`none` on green). Hand off only if exit code is 0. Append the same compact evidence to this plan's SDD ledger.

- [x] **Step 9: Inspect, stage explicitly, and commit**

  Inspect owned diffs and the pre-existing index; stage only `src/local_llm_lab/probes/adapter_delta.py`, `tests/test_adapter_delta.py`, this plan, and `design_specifications/under_review/SPEC-004-S3-IMPLEMENTATION-REPORT.md`. Verify staged names and patch, then commit those exact pathspecs. Do not stage the ignored SDD workspace or any foreign path.
