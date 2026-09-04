# SPEC-001 Section 7 In-Process Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the subprocess MLX-LM training entry with a fail-closed, in-process `train_model(args, model, train_set, valid_set, training_callback=None)` call over repository-owned rendered datasets, persist structured training metrics, and preserve effective configuration, provenance, preflight protection, and memory cleanup.

**Architecture:** `pipeline/cli.py` will lazily validate the installed `mlx-lm` version and exact trainer parameter names, load the base model once, resolve its architecture, construct one effective `SimpleNamespace`, serialize that namespace to `lora.yaml`, and call the validated trainer with rendered train/valid datasets plus a repository-owned metrics callback. A `finally` boundary drops all model-associated references and clears the MLX cache on success or failure; trainer output is teed to the console and `train.log`, while checkpoint selection reads validation loss only from structured `metrics.jsonl` records.

**Tech Stack:** Python 3.13, `mlx-lm==0.31.3`, PyYAML, pytest, Ruff.

**Spec:** `design_specifications/pending/SPEC-001-model-agnostic-backbone.md` §7 and §9, as corrected by ratified R14 on GitHub issue #1; `design_specifications/pending/02-INTERFACE-AND-WIRING-MAP.md` §2.6, §2.11, and R14; `design_specifications/pending/01-IMPLEMENTER-BRIEFING.md` standing rules; GitHub issue #14.

## Global Constraints

- Do not load model weights or run training, inference, preflight, selection, evaluation, rollout, branch mining, preference training, or gated research commands; tests use fakes only.
- Do not modify `src/local_llm_lab/arch.py`, `src/local_llm_lab/pipeline/preflight.py`, their tests, pending specifications, research records, model registry files, lockfiles, `outputs/`, `data/`, or `reports/`.
- Keep the current CLI preflight guard and `--skip-preflight-check` behavior unchanged until issue #15 releases a stage-scoped API.
- The only accepted trainer call contract is the R14-corrected `mlx_lm.lora.train_model(args, model, train_set, valid_set, training_callback=None)`. Abort before checkpoint cleanup or model loading when the installed distribution is not `mlx-lm==0.31.3`, when the import fails, or when the ordered parameter names or callback default differ.
- The tokenizer belongs only to `load_rendered_splits`; it must not be passed to `train_model` or added to the dependency signature.
- `RenderedRowsDataset.__getitem__` remains `(tokens: list[int], offset: int)`, with prompt and completion tokenized separately, the offset equal to the prompt-token count, completion-only truncation, and stable length ordering.
- The exact effective namespace passed to the trainer is the mapping written to metadata-sized `outputs/<run>/lora.yaml`; it must retain resolved `num_layers`, resolved `lora_parameters.keys`, resume state, `mask_prompt: true`, and imported MLX-LM defaults for unspecified keys.
- A training callback must write newline-delimited JSON records to `outputs/<run>/metrics.jsonl`; every record has `step`, `train_loss`, `val_loss`, `tokens`, and `elapsed`, using `null` for the loss that does not apply. For validation callbacks, convert MLX-LM's zero-based completed-iteration value to the checkpoint/log step with `step = iteration + 1`; for training callbacks use `step = iteration` and `tokens = trained_tokens`.
- `stage_select` must read validation losses from structured `metrics.jsonl`, not parse `train.log`. Malformed metric JSON or invalid typed fields fail with a path-and-line diagnostic; train-only records are ignored for loss lookup.
- `train.log` must still receive the trainer's stdout/stderr while progress remains visible on the parent console, but no selection behavior may depend on its prose format.
- Training provenance is written once and only after successful training, using the serialized effective configuration; failures write no provenance.
- On every exit after model loading begins, release model, tokenizer, and dataset references before clearing the MLX cache. Preserve the original failure if cleanup succeeds.
- R10: training/callback tests stay in `tests/test_cli.py`, rendered dataset tests in `tests/test_tuner_data.py`, and structured selection-metric tests in `tests/test_selection.py`; do not duplicate tests elsewhere.
- R11: `lora.yaml`, `train.log`, the SDD ledger, and the implementation report are metadata-sized direct writes; no array or over-1-MiB artifact is introduced.
- R13: no implementation report, review request, or claim release until the exact full fake-only suite is green.
- No new dependency, no push, no pending-spec move, and no bypass of the preflight requirement.

---

### Task 1: In-process training entry and rendered-dataset contract

**Files:**

- Modify: `src/local_llm_lab/pipeline/cli.py`
- Verify unchanged implementation contract: `src/local_llm_lab/tuner_data.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_tuner_data.py`
- Modify: `tests/test_selection.py`

**Interfaces:**

- Consumes: `load_model_spec(name) -> ModelSpec`; `ModelSpec.resolve(model, tokenizer) -> ResolvedSpec`; `load_rendered_splits(data_dir, tokenizer, *, max_seq_length) -> (train, valid, test)`.
- Produces: lazy runtime validation for the exact ordered trainer parameters; an effective `SimpleNamespace`; `stage_train(config, iters, resume_from=None)` calling `train_model(args, model, train_set, valid_set, metrics_callback)` in-process; structured metrics consumed by selection; post-run provenance and deterministic lifecycle cleanup.
- Preserves: `lora_config(config, resolved, iters=None, resume_from=None) -> dict[str, Any]`, `stage_train`'s public signature, checkpoint-directory cleanup, console logging, `train.log`, and current `main()` preflight ordering.

- [ ] **Step 1: Add focused failing trainer-boundary tests**

  Add mutation-sensitive tests in `tests/test_cli.py` that use fake callables and sentinels, never `mlx` or model weights:

  ```python
  def exact_train_model(args, model, train_set, valid_set, training_callback=None):
      calls.append((vars(args).copy(), id(model), train_set, valid_set, training_callback))
      training_callback.on_val_loss_report({"iteration": 0, "val_loss": 1.25})
      training_callback.on_train_loss_report(
          {"iteration": 1, "train_loss": 1.0, "trained_tokens": 64}
      )
      print("Iter 1: Val loss 1.25")
  ```

  Cover each independent behavior:

  - runtime validation accepts only distribution version `0.31.3` and the ordered names/default of `args`, `model`, `train_set`, `valid_set`, `training_callback=None`;
  - a wrong version and a tokenizer-bearing `(args, model, tokenizer, train_set, valid_set)` signature both abort before `_load_training_base` is called;
  - `stage_train` calls `load_rendered_splits(config["data"], tokenizer, max_seq_length=args.max_seq_length)` and passes the returned train/valid objects, the same model object, the effective args, and a callback in the exact five positional slots; the tokenizer must not reach the trainer;
  - `vars(args)` exactly equals parsed `lora.yaml`, including resolved layer count/keys, resume path, overrides, and unspecified keys copied from fake `CONFIG_DEFAULTS`;
  - fake trainer output appears in `train.log`; callback events produce exact JSONL fields and step conversion in `metrics.jsonl`; successful provenance receives that same YAML mapping; and the stale `checkpoints/` directory alone is removed;
  - trainer failure propagates, writes no provenance, and releases temporary model/tokenizer objects before the single cache-clear call.

- [ ] **Step 2: Add selection-metric tests and verify RED**

  Replace the log-backed selection fixture in `tests/test_selection.py` with newline-delimited records containing all five R14 keys. Add a focused test proving malformed metrics fail with their path and line number while train-loss-only records are ignored.

  Run:

  ```bash
  uv run pytest -o addopts='' -q tests/test_cli.py -k 'training_entry or stage_train'
  uv run pytest -o addopts='' -q tests/test_selection.py -k 'validation_losses or stage_select'
  ```

  Expected: failures show the runtime validator/effective namespace do not exist, `stage_train` still tries the subprocess path rather than the fake five-argument trainer, and selection still parses prose `train.log` instead of structured metrics.

- [ ] **Step 3: Implement the minimal in-process runtime and lifecycle**

  In `pipeline/cli.py`:

  ```python
  _PINNED_MLX_LM_VERSION = "0.31.3"
  _TRAIN_MODEL_PARAMETERS = (
      "args",
      "model",
      "train_set",
      "valid_set",
      "training_callback",
  )

  def _load_training_entry() -> tuple[Any, dict[str, Any]]:
      installed = distribution_version("mlx-lm")
      if installed != _PINNED_MLX_LM_VERSION:
          raise SystemExit(
              f"mlx-lm {_PINNED_MLX_LM_VERSION} required; found {installed}"
          )
      try:
          module = importlib.import_module("mlx_lm.lora")
      except ImportError as error:
          raise SystemExit(f"cannot import pinned mlx_lm.lora: {error}") from error
      trainer = module.train_model
      parameters = tuple(inspect.signature(trainer).parameters.values())
      actual = tuple(parameter.name for parameter in parameters)
      if actual != _TRAIN_MODEL_PARAMETERS or parameters[-1].default is not None:
          raise SystemExit(
              f"mlx-lm train_model signature mismatch: expected "
              f"{_TRAIN_MODEL_PARAMETERS} with training_callback=None; found {actual}"
          )
      defaults = module.CONFIG_DEFAULTS
      if not isinstance(defaults, dict):
          raise SystemExit("mlx_lm.lora.CONFIG_DEFAULTS must be a mapping")
      return trainer, dict(defaults)

  def _effective_lora_args(lora: dict[str, Any], defaults: dict[str, Any]) -> SimpleNamespace:
      return SimpleNamespace(**{**defaults, **lora})
  ```

  Extract a pure effective `ModelSpec` builder so `_resolve_training_spec` keeps its existing load/resolve/release semantics while `stage_train` can retain the single loaded model/tokenizer through training. Add a small duck-typed callback that appends stable JSON records with the five R14 fields and elapsed wall-clock time. In `stage_train`, validate the runtime first; load and resolve once; build and serialize the effective args; load all three repository-owned rendered splits using the same tokenizer and max sequence length; remove only stale checkpoints; create/truncate `metrics.jsonl`; tee the exact in-process trainer output to console plus `train.log`; call the trainer without a tokenizer and with the callback; then write provenance only after trainer success. In `finally`, drop callback/train/valid/test/model/tokenizer references and clear the model cache once for every path after loading begins.

- [ ] **Step 4: Run focused tests and verify GREEN**

  Run:

  ```bash
  uv run pytest -o addopts='' -q tests/test_cli.py -k 'training_entry or stage_train or resolve_training_spec or lora_config'
  ```

  Expected: all selected tests pass with pristine output.

- [ ] **Step 5: Strengthen prompt-mask and token-identity characterization**

  In `tests/test_tuner_data.py`, add or refine a fake-tokenizer test that would fail if the dataset jointly tokenized `prompt + completion`, changed the prompt offset, included special tokens, or truncated prompt tokens:

  ```python
  assert dataset[0] == ([prompt_token, completion_token], 1)
  assert tokenizer.calls == [
      ("prompt", False),
      ("completion", False),
  ]
  ```

  Keep the existing stable length ordering, completion truncation, and split-file coverage. This characterizes an already-implemented dependency contract and may pass immediately; change `tuner_data.py` only if the characterization exposes a real contract defect.

- [ ] **Step 6: Run the rendered-dataset tests and verify RED/GREEN evidence**

  Run before any required production correction, then again after it:

  ```bash
  uv run pytest -o addopts='' -q tests/test_tuner_data.py
  ```

  Expected: all tests pass with pristine output if the existing contract is intact. If the characterization exposes a production defect, record its failing output before the minimal correction and then re-run to green.

- [ ] **Step 7: Run static and focused gates**

  Run:

  ```bash
  uv run ruff check src/local_llm_lab/pipeline/cli.py src/local_llm_lab/tuner_data.py tests/test_cli.py tests/test_tuner_data.py
  uv run pytest -o addopts='' -q tests/test_cli.py tests/test_tuner_data.py tests/test_data.py tests/test_selection.py
  python -c "import local_llm_lab.pipeline.cli, local_llm_lab.tuner_data"
  git diff --check
  ```

  Expected: every command exits 0; no import loads a model.

- [ ] **Step 8: Run the R13 hand-off gate**

  Run exactly once on the complete implementation state:

  ```bash
  uv run pytest -o addopts='' -q
  ```

  Expected: the full fake-only suite passes before review or reporting.

- [ ] **Step 9: Commit the task**

  Stage only the exact task files, inspect the staged patch and names, then commit:

  ```bash
  git add src/local_llm_lab/pipeline/cli.py tests/test_cli.py tests/test_tuner_data.py tests/test_selection.py
  git diff --cached --check
  git commit -m "Replace subprocess training with in-process entry"
  ```

  Include `src/local_llm_lab/tuner_data.py` in the explicit path list only if Step 5 found and fixed a real production defect. Never stage foreign files or broad paths.
