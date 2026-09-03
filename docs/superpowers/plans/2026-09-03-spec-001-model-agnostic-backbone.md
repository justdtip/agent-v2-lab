# SPEC-001 Model-Agnostic Backbone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:test-driven-development` for every behavior change and `superpowers:systematic-debugging` for any unexpected failure. The coordinator dispatches one implementation worker at a time and an independent reviewer after each task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the model-agnostic registry, architecture abstraction, rendering, cache, training, probe, provenance, and gated preflight contracts in SPEC-001 without loading a real model or changing legacy 3B behavior.

**Architecture:** `ModelSpec` owns declared model policy and `ResolvedSpec` owns discovered runtime facts. `ArchitectureView` is the only code allowed to inspect and execute decoder internals. All downstream rendering, training, evaluation, and probe paths consume these two abstractions, with fake dense and hybrid models proving the contracts.

**Tech Stack:** Python 3.13, MLX/MLX-LM 0.32.2/0.31.3 interfaces behind test fakes, PyYAML, pytest, NumPy.

**Spec:** `design_specifications/pending/SPEC-001-model-agnostic-backbone.md`

## Global Constraints

- Do not load a checkpoint, call `mlx_lm.load`, or execute any model-loading pipeline, probe, preflight, cache-equivalence, or J-space command.
- Tests use NumPy/MLX fakes only. A monkeypatched loader must fail if a test accidentally reaches a real model load.
- Do not modify or delete existing files under `outputs/`, `data/`, or `reports/`.
- Do not edit `research/*.md`; only update the two Python research utilities explicitly owned by SPEC-001.
- Activations, residuals, and tangents are float32. Model weights remain quantized.
- Decoder blocks always receive masks built through the model-family mask helpers represented by `ArchitectureView.masks`; never pass `None` as a shortcut.
- Hard-coded model constants (`36`, `2048`, `35`, `<|im_end|>`, `model.model.layers`, projection-name lists) may appear only in `configs/models/`, `src/local_llm_lab/arch.py`, tests, or legacy modules.
- Preserve the existing 3B prompt bytes, task RNG derivations, observation windowing, and token behavior when `qwen25-coder-3b` uses `thinking: unsupported`.
- Existing A/B/C data and adapters are read-only. Compatibility tests may read saved files but must skip when absent.
- All random choices accept and record an explicit seed. Large writes remain atomic.
- Public signatures follow `design_specifications/pending/02-INTERFACE-AND-WIRING-MAP.md`; keyword-only markers are binding.
- Python modules use `from __future__ import annotations`, type hints, frozen record dataclasses, contract docstrings, and explicit `__all__` where broadly imported.
- Work stays on the shared branch `codex/agent-v2-specs`. No worktree, branch switch, merge, push, or real model run is part of this plan.

## Compatibility Rulings

- A raw HF id equal to the current 3B id resolves to the registered `qwen25-coder-3b` defaults. Unknown HF ids get conservative non-thinking defaults and must not silently inherit Qwen3.5 behavior.
- Data generation records the unresolved `ModelSpec` when no model is loaded; `ResolvedSpec` is optional in provenance at this stage. Data generation never loads a checkpoint merely to fill derived metadata.
- `cache_equivalence_verified` is represented as optional cache metadata in `ModelSpec`; `auto` keeps trim reuse for the dense 3B model and falls back to `none` for an unverified snapshot strategy.
- Existing configs that use `tasks:` and omit explicit split difficulties remain the historical compatibility path. New `splits:` behavior is owned by SPEC-003.
- This plan uses serialized subagent development in the primary checkout because Coordinator requires a shared worktree. The baseline commit `bda5ff5` and per-task commits provide the review boundary.

---

### Task 1: Model registry and immutable specifications

**Files:**

- Create: `configs/models/qwen25-coder-3b.yaml`
- Create: `configs/models/qwen35-4b.yaml`
- Create: `configs/models/qwen35-9b.yaml`
- Create: `src/local_llm_lab/models.py`
- Modify: `configs/agent_v2.yaml`
- Modify: `configs/agent_v2b.yaml`
- Modify: `configs/agent_v2c.yaml`
- Test: `tests/test_pipeline.py`

**Interfaces:**

- Produces: `ChatSpec`, `LoraSpec`, `ModelSpec`, `ResolvedSpec`, `load_model_spec`, and `registered_models` exactly as declared in wiring-map §2.1.
- Preserves: current raw 3B HF id compatibility, seven attention+MLP LoRA targets, trim cache, unsupported thinking, rank 16, scale 32, dropout 0, and the existing training dimensions.
- Consumed by: every later task in this plan and SPEC-002 through SPEC-004.

- [ ] **Step 1: Write failing registry tests**

```python
def test_registered_models_include_all_three_backbones() -> None:
    assert registered_models() == ["qwen25-coder-3b", "qwen35-4b", "qwen35-9b"]


def test_raw_legacy_hf_id_uses_qwen25_compatibility_defaults() -> None:
    spec = load_model_spec("mlx-community/Qwen2.5-Coder-3B-Instruct-4bit")
    assert spec.name == "qwen25-coder-3b"
    assert spec.chat.thinking == "unsupported"
    assert spec.cache_strategy == "trim"
    assert spec.lora.keys == "attention+mlp"
```

- [ ] **Step 2: Run the registry tests and verify they fail because the module/configs do not exist**

Run: `uv run pytest -q tests/test_pipeline.py -k 'registered_models or raw_legacy_hf_id'`

- [ ] **Step 3: Implement the exact frozen dataclasses and deterministic YAML loader**

```python
@dataclass(frozen=True)
class ChatSpec:
    thinking: Literal["unsupported", "off", "inference", "trained"]
    template_kwargs: dict[str, Any]
    end_of_turn: str
    extra_stop_tokens: tuple[str, ...]
    max_think_tokens: int = 512


def load_model_spec(name_or_hf_id: str) -> ModelSpec: ...


def registered_models() -> list[str]: ...
```

- [ ] **Step 4: Define the resolution contract without duplicating architecture inspection**

Define `ResolvedSpec` and make `ModelSpec.resolve` lazily delegate to `ArchitectureView.from_model`; Task 2 supplies that module and activates the runtime path. Task 1 tests configuration validation only. Do not duplicate structural traversal in `models.py`. Reject invalid layer fractions, unsupported thinking values, and invalid cache policies with actionable `ValueError` messages.

- [ ] **Step 5: Replace raw model ids in the three v2 pipeline YAMLs with `model: qwen25-coder-3b`**

Keep every other historical value byte-for-byte. Do not add SPEC-002 screen or SPEC-003 split fields in this task.

- [ ] **Step 6: Run focused and neighboring tests**

Run: `uv run pytest -q tests/test_pipeline.py -k 'model_spec or registered_models or config'`

- [ ] **Step 7: Commit only Task 1 files**

```bash
git add -- configs/models configs/agent_v2.yaml configs/agent_v2b.yaml configs/agent_v2c.yaml src/local_llm_lab/models.py tests/test_pipeline.py
git commit -m "feat: add model registry"
```

---

### Task 2: ArchitectureView for dense and hybrid decoders

**Files:**

- Create: `src/local_llm_lab/arch.py`
- Modify: `src/local_llm_lab/models.py`
- Modify: `tests/test_probes.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**

- Consumes: Task 1 `ModelSpec`/`ResolvedSpec` and activates `ModelSpec.resolve`.
- Produces: `ArchitectureView` exactly as wiring-map §2.2, including `residuals(ids, layers)` and `tail(layer)`.
- Owns: the only sanctioned hard-coded structural compatibility table and model constants.

- [ ] **Step 1: Add a hybrid fake whose text module lives at `language_model.model`**

The fake must alternate three linear-attention blocks then one full-attention block, reject the wrong mask kind, expose list-state `ArraysCache` for linear blocks and offset-state `KVCache` for attention blocks, and support tied and untied unembedding variants.

- [ ] **Step 2: Write failing view contract tests**

```python
@pytest.mark.parametrize("fake_factory", [make_dense_fake, make_hybrid_fake])
def test_architecture_view_matches_model_forward(fake_factory) -> None:
    model, ids = fake_factory()
    view = ArchitectureView.from_model(model)
    residuals = view.residuals(ids, tuple(range(view.num_layers + 1)))
    expected, recorded = model.recording_forward(ids)
    for layer, hidden in recorded.items():
        np.testing.assert_allclose(residuals[layer], hidden, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(view.unembed(view.final_norm(residuals[view.num_layers])), expected)
```

- [ ] **Step 3: Run the focused tests and verify missing `ArchitectureView` is the failure**

Run: `uv run pytest -q tests/test_probes.py -k 'architecture_view or hybrid'`

- [ ] **Step 4: Implement structural module discovery and residual execution**

```python
class ArchitectureView:
    @classmethod
    def from_model(cls, model: nn.Module) -> ArchitectureView: ...

    def masks(self, h: mx.array, cache: list[Any] | None) -> dict[str, Any]: ...
    def run_block(self, index: int, h: mx.array, masks: dict[str, Any], cache_i: Any | None) -> mx.array: ...
    def residuals(self, ids: mx.array, layers: Sequence[int]) -> dict[int, mx.array]: ...
    def tail(self, layer: int) -> Callable[[mx.array], mx.array]: ...
```

All activation-returning methods cast to float32. `masks` must dispatch to `create_attention_mask` and `create_ssm_mask` when available and must preserve equivalent fake helpers in tests.

- [ ] **Step 5: Implement tied/untied unembedding, cache inspection, layer kinds, probe indices, and LoRA discovery**

`lora_targets("attention+mlp")` resolves only the seven dense suffixes that actually exist; `all-linear` also admits `in_proj_qkvz`, `in_proj_ba`, and `out_proj`; `auto` selects by layer kinds; explicit tuples are validated against the module tree.

Complete `ModelSpec.resolve` through the view and test that it populates layer types, sizes, tied embeddings, explicit LoRA keys, trainable parameter count, resolved probe indices, cache strategy, snapshot revision, and JVP method.

- [ ] **Step 6: Run dense and hybrid contract tests**

Run: `uv run pytest -q tests/test_probes.py tests/test_pipeline.py -k 'architecture_view or hybrid or lora_target or residual'`

- [ ] **Step 7: Commit only Task 2 files**

```bash
git add -- src/local_llm_lab/arch.py src/local_llm_lab/models.py tests/test_probes.py tests/test_pipeline.py
git commit -m "feat: add architecture view"
```

---

### Task 3: View-based capture and J-lens

**Files:**

- Modify: `src/local_llm_lab/probes/capture.py`
- Modify: `src/local_llm_lab/pipeline/jlens.py`
- Modify: `src/local_llm_lab/probes/adapter_delta.py`
- Modify: `research/jspace_sweep.py`
- Test: `tests/test_probes.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**

- Consumes: Task 2 `ArchitectureView`.
- Produces: wiring-map §2.13 capture functions and §2.14 J-lens functions, except `replace=True` remains owned by SPEC-004.
- Preserves: residual index convention, response boundary-merge repair, float32 numerics, injection restoration, and block-mask restoration.

- [ ] **Step 1: Write failing dense/hybrid capture equivalence tests and method-aware JVP tests**

```python
@pytest.mark.parametrize("method", ["forward", "finite_difference"])
def test_jlens_averaging_and_linearity_for_view(method: str) -> None:
    mapped, stats = jlens_map(view, layer, probe, corpus_ids, method=method)
    assert stats["method"] == method
    np.testing.assert_allclose(map_positive + map_negative, 0.0, atol=1e-4)
```

Also assert one head pass serves all requested residual layers, the hybrid fake sees its correct mask at every tail block, and injection position offsets work for both cache representations.

- [ ] **Step 2: Run focused tests and verify the old model-based signatures fail**

Run: `uv run pytest -q tests/test_probes.py tests/test_pipeline.py -k 'capture or jlens or injection or block_mask'`

- [ ] **Step 3: Refactor capture functions onto `ArchitectureView`**

```python
def capture_residuals(
    view: ArchitectureView,
    token_ids: Any,
    layers: Sequence[int],
    *,
    positions: Literal["last", "all"] | Sequence[int] = "last",
) -> dict[int, Any]: ...
```

Make `lora_block_mask` find LoRA modules by type and layer ownership rather than the historical seven names. Keep context-manager restoration exception-safe.

Until Task 6 updates every probe caller, a private coercion helper may wrap a legacy model argument with `ArchitectureView.from_model`. Task 6 removes that migration path so the final public contracts are view-only.

- [ ] **Step 4: Refactor J-lens onto view residuals and view tails**

Promote `_encode` to `encode` and `_distribution` to `distribution`; update adapter-delta and J-space callers. Implement central finite differences with `eps = 1e-2 * ||h|| / ||v||` in float32 and record the selected method.

- [ ] **Step 5: Remove private dense-layout reads from non-legacy probe code**

Run: `grep -RInE 'model\.model\.layers|_causal_mask|_distribution|_encode' src/local_llm_lab/probes src/local_llm_lab/pipeline/jlens.py research/jspace_sweep.py`

Expected: no direct layout reads or private helper imports remain in owned files.

- [ ] **Step 6: Run focused and import tests**

Run: `uv run pytest -q tests/test_probes.py tests/test_pipeline.py -k 'capture or jlens or injection or block_mask or adapter_delta'`

- [ ] **Step 7: Commit only Task 3 files**

```bash
git add -- src/local_llm_lab/probes/capture.py src/local_llm_lab/pipeline/jlens.py src/local_llm_lab/probes/adapter_delta.py research/jspace_sweep.py tests/test_probes.py tests/test_pipeline.py
git commit -m "refactor: run probes through architecture view"
```

---

### Task 4: Canonical rendering and rendered training rows

**Files:**

- Modify: `src/local_llm_lab/pipeline/protocol.py`
- Modify: `src/local_llm_lab/pipeline/data.py`
- Create: `src/local_llm_lab/tuner_data.py`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_probes.py`

**Interfaces:**

- Consumes: Task 1 `ModelSpec`.
- Produces: wiring-map §2.3 and §2.5 rendering seam plus §2.6 dataset protocol.
- Preserves: no native `tools=` chat-template rendering, two-observation windowing, fenced-JSON action format, and legacy 3B prompt bytes.

- [ ] **Step 1: Write failing tests for all four thinking modes and suffix assertions**

```python
@pytest.mark.parametrize("mode", ["unsupported", "off", "inference", "trained"])
def test_build_prompt_ends_with_declared_generation_suffix(mode: str) -> None:
    spec = fake_spec(mode)
    prompt = build_prompt(tokenizer, messages, spec=spec)
    assert prompt.endswith(generation_suffix(spec))


def test_strip_thinking_round_trips_note_and_call() -> None:
    thinking, remainder = strip_thinking("<think>plan</think>\n\nNote\n```json\n{}\n```")
    assert thinking == "plan"
    assert remainder.startswith("Note\n```json")
```

- [ ] **Step 2: Write failing dataset tests for separate tokenization and boundary merges**

The fake tokenizer must intentionally merge the final prompt token with the first completion token when tokenized jointly. Assert `RenderedRowsDataset` returns the concatenated separately tokenized sequence and `offset == len(prompt_tokens)`.

- [ ] **Step 3: Run focused tests and verify the new APIs are absent**

Run: `uv run pytest -q tests/test_pipeline.py tests/test_probes.py -k 'generation_suffix or strip_thinking or rendered_rows or boundary_merge'`

- [ ] **Step 4: Implement the exact protocol signatures**

```python
def build_prompt(tokenizer, messages: list[dict[str, Any]], *, spec: ModelSpec, keep_last: int = DEFAULT_KEEP_LAST, generation: bool = True) -> str: ...
def generation_suffix(spec: ModelSpec) -> str: ...
def strip_thinking(text: str) -> tuple[str | None, str]: ...
def render_completion(thought: str, action: Action, *, spec: ModelSpec) -> str: ...
```

`Turn.thinking` defaults to `None`; `turn_is_complete` ignores code fences while the first think block remains open.

Keep a temporary `spec=None` compatibility default that resolves only to the registered 3B spec so the repository remains green before Task 6 updates every caller. Task 6 removes the default and enforces the exact keyword-only `spec` contract.

- [ ] **Step 5: Implement `render_rows`, `RenderedRowsDataset`, and split loading**

Rows keep `messages` and `metadata`, then gain `prompt` and `completion`. `load_rendered_splits` reads `train.jsonl`, `valid.jsonl`, and `test.jsonl`, enforces `max_seq_length`, and returns dataset-protocol objects without invoking MLX-LM's `ChatDataset`.

- [ ] **Step 6: Add a legacy 3B renderer compatibility test**

Render representative current rows with the old fake-template expectations and assert exact prompt bytes and token ids. This test must not rewrite `data/agent_v2c`.

- [ ] **Step 7: Run rendering/data tests**

Run: `uv run pytest -q tests/test_pipeline.py tests/test_probes.py -k 'prompt or thinking or render or dataset or window'`

- [ ] **Step 8: Commit only Task 4 files**

```bash
git add -- src/local_llm_lab/pipeline/protocol.py src/local_llm_lab/pipeline/data.py src/local_llm_lab/tuner_data.py tests/test_pipeline.py tests/test_probes.py
git commit -m "feat: canonicalize model rendering"
```

---

### Task 5: Thinking-aware runner and cross-turn cache strategies

**Files:**

- Modify: `src/local_llm_lab/pipeline/runner.py`
- Modify: `src/local_llm_lab/pipeline/transcript.py`
- Modify: `research/cache_equivalence.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**

- Consumes: Task 1 specs, Task 2 view, and Task 4 protocol.
- Produces: wiring-map §2.7 `TurnCacheBase`, `TrimCache`, `SnapshotCache`, `make_turn_cache`, three-value `generate_turn_with_count`, and spec/view-aware `run_task`.
- Preserves: dense trim-cache behavior and old `Trajectory(**record)` compatibility through defaults.

- [ ] **Step 1: Write failing cache strategy tests**

```python
@pytest.mark.parametrize("strategy", ["trim", "snapshot", "none"])
def test_fake_generation_matches_without_cache(strategy: str) -> None:
    cached = run_fake_task(cache_strategy=strategy)
    uncached = run_fake_task(cache_strategy="none")
    assert cached.raw_turns == uncached.raw_turns
    assert cached.actions == uncached.actions
```

Assert snapshot restoration deep-copies list and array cache state, keeps exactly the immutable-prefix tokens, and reports encoded/reused counts honestly.

- [ ] **Step 2: Write failing thinking-budget and trajectory compatibility tests**

Assert the runner force-closes an open think block at `max_think_tokens`, continues until a complete note+call, counts think tokens separately, and can load an old trajectory record with all new fields defaulted.

- [ ] **Step 3: Run focused tests and verify old interfaces fail**

Run: `uv run pytest -q tests/test_pipeline.py -k 'turn_cache or snapshot or thinking or trajectory'`

- [ ] **Step 4: Rename the current cache to `TrimCache` and add the cache protocol/factory**

```python
class TurnCacheBase(Protocol):
    cache: Any
    reused_tokens: int
    encoded_tokens: int
    def prepare(self, token_ids: list[int]) -> list[int]: ...
    def commit(self, token_ids: list[int], generated: list[int]) -> None: ...
```

`SnapshotCache` primes and snapshots the immutable prefix on first prepare, then restores that state and returns only the suffix on later turns. `auto` uses trim when all caches are trimmable and snapshot only when registry verification metadata permits it; otherwise it warns and returns no cache.

- [ ] **Step 5: Implement thinking-aware generation and task execution**

`generate_turn_with_count(..., *, spec)` returns `(raw_text, total_tokens, think_tokens)`. `run_task` computes `prefix_tokens` from a non-generation prompt over the system+task messages, calls `strip_thinking` before `parse_turn`, and records `steps[*].thinking`.

Use temporary 3B-only defaults for newly required `spec`, `view`, and `resolved` arguments until Task 6 migrates all callers. The compatibility path must be isolated in one helper and removed by Task 6.

- [ ] **Step 6: Update transcript output and cache-equivalence CLI plumbing**

Thinking is collapsed to first/last line. `research/cache_equivalence.py` accepts `--model` and `--strategy` but remains model-gated and must not be executed in this plan.

- [ ] **Step 7: Run focused tests**

Run: `uv run pytest -q tests/test_pipeline.py -k 'turn_cache or snapshot or thinking or trajectory or transcript'`

- [ ] **Step 8: Commit only Task 5 files**

```bash
git add -- src/local_llm_lab/pipeline/runner.py src/local_llm_lab/pipeline/transcript.py research/cache_equivalence.py tests/test_pipeline.py
git commit -m "feat: add thinking and cache strategies"
```

---

### Task 6: Model-aware policy loading, consumer plumbing, and probe layers

**Files:**

- Modify: `src/local_llm_lab/pipeline/evaluate.py`
- Modify: `src/local_llm_lab/pipeline/rollout.py`
- Modify: `src/local_llm_lab/pipeline/branch.py`
- Modify: `src/local_llm_lab/pipeline/prefer.py`
- Modify: `src/local_llm_lab/pipeline/cli.py`
- Modify: `src/local_llm_lab/probes/policies.py`
- Modify: `src/local_llm_lab/probes/state_probe.py`
- Modify: `src/local_llm_lab/probes/assistant_axis.py`
- Modify: `src/local_llm_lab/probes/adapter_delta.py`
- Modify: `src/local_llm_lab/pipeline/jlens.py`
- Modify: `research/jspace_sweep.py`
- Test: `tests/test_pipeline.py`
- Test: `tests/test_probes.py`
- Test: `tests/test_probes_axis.py`

**Interfaces:**

- Consumes: Tasks 1–5.
- Produces: wiring-map §2.10 `load_policy(spec, adapter, *, lazy=False)` and end-to-end spec/view/resolved plumbing.
- Leaves for later specs: evaluation integrity/statistics, D split handling, observation stubbing, P2 analysis, ablation orchestration, and patching.

- [ ] **Step 1: Write failing monkeypatched loader and import tests**

```python
def test_load_policy_returns_model_tokenizer_view_and_resolved(monkeypatch) -> None:
    monkeypatch.setattr(evaluate, "load", fake_mlx_load)
    model, tokenizer, view, resolved = evaluate.load_policy(spec, None, lazy=True)
    assert view.num_layers == resolved.num_layers
    assert resolved.spec is spec
```

Add an import test for every consumer named in wiring-map §4.1–§4.2. The fake loader records arguments and never touches a checkpoint.

- [ ] **Step 2: Run focused tests and verify signature/unpacking failures**

Run: `uv run pytest -q tests/test_pipeline.py tests/test_probes.py tests/test_probes_axis.py -k 'load_policy or model_plumbing or import'`

- [ ] **Step 3: Update every `build_prompt`, `run_task`, and `load_policy` caller**

Pass `spec=` explicitly and unpack `(model, tokenizer, view, resolved)`. Preserve existing CLI defaults by resolving the model value from the selected pipeline YAML or registry alias.

Remove the temporary model-to-view coercion and optional-spec/default-runner migration helpers introduced by Tasks 3–5; the final public signatures must exactly match the wiring map.

- [ ] **Step 4: Resolve policy names and layers through model specs**

All probe CLIs accept `--model`, `--policy`, and fractional-or-integer `--layers`; outputs record both the original fractions and resolved indices. Adapter comparisons reject different base HF ids or snapshot revisions.

- [ ] **Step 5: Remove remaining hard-coded layout reads and policy tables in owned paths**

Run: `grep -RInE 'model\.model\.layers|\b36\b|\b2048\b|\b35\b|<\|im_end\|>' src/local_llm_lab/pipeline src/local_llm_lab/probes`

Classify every hit. Only `arch.py`, registry configs, tests, or unrelated legacy paths may contain a sanctioned hit.

- [ ] **Step 6: Run all fake-only pipeline/probe suites**

Run: `uv run pytest -q tests/test_pipeline.py tests/test_probes.py tests/test_probes_axis.py`

- [ ] **Step 7: Commit only Task 6 files**

```bash
git add -- src/local_llm_lab/pipeline/evaluate.py src/local_llm_lab/pipeline/rollout.py src/local_llm_lab/pipeline/branch.py src/local_llm_lab/pipeline/prefer.py src/local_llm_lab/pipeline/cli.py src/local_llm_lab/probes/policies.py src/local_llm_lab/probes/state_probe.py src/local_llm_lab/probes/assistant_axis.py src/local_llm_lab/probes/adapter_delta.py src/local_llm_lab/pipeline/jlens.py research/jspace_sweep.py tests/test_pipeline.py tests/test_probes.py tests/test_probes_axis.py
git commit -m "refactor: plumb model specs through pipeline"
```

---

### Task 7: In-process rendered-row training

**Files:**

- Modify: `src/local_llm_lab/pipeline/cli.py`
- Modify: `src/local_llm_lab/tuner_data.py`
- Modify: `src/local_llm_lab/models.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**

- Consumes: Tasks 1, 2, 4, and 6.
- Produces: an in-process `stage_train` that supplies explicit architecture-derived LoRA keys and `RenderedRowsDataset` instances to pinned `mlx_lm.lora.train_model`.
- Gated: tests may call only a monkeypatched fake `train_model`; `stage_train` must never run against a real model in this plan.

- [ ] **Step 1: Write a failing fake training-entry test**

The test supplies fake model/tokenizer/datasets and a recording `train_model`. Assert the namespace contains `num_layers == resolved.num_layers`, explicit `lora_parameters.keys == list(resolved.lora_keys)`, rank/scale/dropout from the spec, configured gradient checkpointing, and no subprocess call.

- [ ] **Step 2: Write failing version and memory refusal tests**

Assert a mismatched `train_model` signature raises `SystemExit` with the pinned-version message, and an estimate above `memory_budget_gib` aborts before the fake trainer is invoked.

- [ ] **Step 3: Run focused tests and verify the subprocess implementation fails them**

Run: `uv run pytest -q tests/test_pipeline.py -k 'stage_train or train_model or memory_budget or rendered_split'`

- [ ] **Step 4: Implement in-process training and effective-config serialization**

Keep the existing `lora.yaml` record, but populate keys, layer count, dataset paths, and effective values from the resolved spec. Do not implement SPEC-002 selection or SPEC-003 split concatenation here.

- [ ] **Step 5: Run focused tests**

Run: `uv run pytest -q tests/test_pipeline.py -k 'stage_train or train_model or memory_budget or rendered_split'`

- [ ] **Step 6: Commit only Task 7 files**

```bash
git add -- src/local_llm_lab/pipeline/cli.py src/local_llm_lab/tuner_data.py src/local_llm_lab/models.py tests/test_pipeline.py
git commit -m "feat: train from rendered rows in process"
```

---

### Task 8: Deterministic provenance and gated preflight

**Files:**

- Create: `src/local_llm_lab/provenance.py`
- Modify: `src/local_llm_lab/pipeline/cli.py`
- Modify: `src/local_llm_lab/pipeline/data.py`
- Modify: `src/local_llm_lab/pipeline/evaluate.py`
- Modify: `src/local_llm_lab/pipeline/rollout.py`
- Modify: `src/local_llm_lab/probes/state_probe.py`
- Modify: `src/local_llm_lab/probes/assistant_axis.py`
- Modify: `src/local_llm_lab/probes/adapter_delta.py`
- Modify: `src/local_llm_lab/pipeline/jlens.py`
- Test: `tests/test_pipeline.py`
- Test: `tests/test_probes.py`

**Interfaces:**

- Consumes: all prior tasks.
- Produces: wiring-map §2.12 provenance API, a fake-testable preflight core, `agent-pipeline preflight`, and preflight gates on every model-loading stage.
- Gated: the real preflight command is implemented but never executed.

- [ ] **Step 1: Write failing deterministic provenance tests**

```python
def test_provenance_is_byte_identical_on_unchanged_tree(tmp_path: Path) -> None:
    first = write_provenance(tmp_path / "run", resolved=None, spec=spec, extra={"seed": 7})
    before = first.read_bytes()
    second = write_provenance(tmp_path / "run", resolved=None, spec=spec, extra={"seed": 7})
    assert second.read_bytes() == before
```

Assert the JSON has sorted source/config hashes, `uv.lock`, installed-version fields, model/spec data, command, seed, controls, and caller extras. Test atomic replacement using a same-directory temp path.

- [ ] **Step 2: Write failing fake preflight and stale-artifact gate tests**

The fake core must report layer kinds, hidden/vocab/tied state, cache types, residual-equivalence error, JVP method, thinking render token counts, LoRA keys/counts, and memory estimate. The gate rejects mismatched HF id or snapshot revision and permits an explicit `--skip-preflight-check` override.

- [ ] **Step 3: Run focused tests and verify APIs are absent**

Run: `uv run pytest -q tests/test_pipeline.py tests/test_probes.py -k 'provenance or preflight'`

- [ ] **Step 4: Implement deterministic provenance and wire every stage**

```python
def write_provenance(
    run_dir: Path,
    *,
    resolved: ResolvedSpec | None,
    spec: ModelSpec,
    extra: dict[str, Any],
) -> Path: ...


def source_tree_hashes(root: Path) -> dict[str, str]: ...
```

Data-stage provenance uses `resolved=None`; model stages use the resolved view. Preserve unrelated files and use `os.replace` after flush+fsync.

- [ ] **Step 5: Implement the preflight subcommand and model-stage gates**

Preflight may load lazily only when a human later executes it. Unit tests inject a fake loader. Training, evaluation, and probe CLIs check `outputs/preflight/<name>.json` before their loader call and expose `--skip-preflight-check` without invoking it in tests.

- [ ] **Step 6: Run SPEC-001 integration checks**

Run: `uv run pytest -q tests/test_pipeline.py tests/test_probes.py tests/test_probes_axis.py`

Run: `uv run python -c "import local_llm_lab.pipeline.cli, local_llm_lab.probes.state_probe, local_llm_lab.probes.assistant_axis, local_llm_lab.probes.adapter_delta, local_llm_lab.pipeline.jlens"`

Run: `grep -RInE 'model\.model\.layers|\b36\b|\b2048\b|\b35\b|<\|im_end\|>' src/local_llm_lab/pipeline src/local_llm_lab/probes`

- [ ] **Step 7: Run the complete fake-only suite**

Run: `uv run pytest -q`

- [ ] **Step 8: Commit only Task 8 files**

```bash
git add -- src/local_llm_lab/provenance.py src/local_llm_lab/pipeline/cli.py src/local_llm_lab/pipeline/data.py src/local_llm_lab/pipeline/evaluate.py src/local_llm_lab/pipeline/rollout.py src/local_llm_lab/probes/state_probe.py src/local_llm_lab/probes/assistant_axis.py src/local_llm_lab/probes/adapter_delta.py src/local_llm_lab/pipeline/jlens.py tests/test_pipeline.py tests/test_probes.py
git commit -m "feat: add provenance and gated preflight"
```

---

## SPEC-001 Completion Gate

- [ ] All eight task reviews report spec compliance and code quality approved, or every residual finding has a recorded ruling.
- [ ] `uv run pytest -q` passes from a fresh invocation without loading a model.
- [ ] The import integration command exits zero.
- [ ] Banned-constant findings are classified and only sanctioned hits remain.
- [ ] `git diff bda5ff5..HEAD --check` reports no new whitespace errors.
- [ ] No tracked or untracked file under `data/`, `outputs/`, or `reports/` was modified by implementation.
- [ ] Real `preflight` remains unexecuted and its required future verification is listed in the implementation report.
- [ ] Move SPEC-001 and its implementation report to `design_specifications/under_review/`; do not move it to `complete/`.
