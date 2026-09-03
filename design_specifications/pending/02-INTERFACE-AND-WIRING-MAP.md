# Interface and wiring map for SPEC-001 to SPEC-004

Audience: every implementing agent. This document is the single source of truth for shared
signatures, file ownership, implementation order, and the integration checks that prove the
pieces connect. When two specs touch the same file, the owner listed here makes the structural
change and the other spec adds to it; both must run the integration checks in §6.

## 1. Implementation order and dependency graph

```
SPEC-004 §1  (offline P2 re-analysis)          no dependencies; may start immediately
SPEC-002 §3, §5 (verdict hardening, bug fixes)  no dependencies; may start immediately
SPEC-001 §1-2 (registry, ArchitectureView)      no dependencies
   └─ SPEC-001 §3-4 (rendering, thinking)        needs §1
        ├─ SPEC-001 §5 (cache strategies)         needs §2, §3
        ├─ SPEC-001 §6-7 (LoRA keys, training)    needs §1-3
        ├─ SPEC-001 §8 (probes on the view)       needs §2
        └─ SPEC-001 §9-10 (provenance, preflight) needs §1-8
SPEC-002 §1 (difficulty parameter, screen)      needs SPEC-001 §1 for config plumbing only
   └─ SPEC-002 §2 (integrity module)             needs the difficulty parameter and Task.difficulty
        └─ SPEC-002 §4 (runner/evaluator fields)  needs SPEC-001 §4 (thinking fields) and §2
SPEC-003 §1-3 (templates, variants, splits)     needs SPEC-002 §1 (difficulty) and §2 (invariants use integrity extractors)
   └─ SPEC-003 §4-5 (matrix)                     gated; needs everything above
SPEC-004 §2-5 (P2 redesign, ablation, P1, P6)   needs SPEC-001 §2, §8 and SPEC-002 §2
```

Parallelisable now: SPEC-004 §1, SPEC-002 §3+§5, SPEC-001 §1-2. Everything else waits on the
arrows.

## 2. Shared types and signatures (exact)

All signatures below are normative. Keyword-only markers (`*`) are part of the contract.

### 2.1 `src/local_llm_lab/models.py` (owner: SPEC-001 §1)

```python
@dataclass(frozen=True)
class ChatSpec:
    thinking: Literal["unsupported", "off", "inference", "trained"]
    template_kwargs: dict[str, Any]
    end_of_turn: str                      # "<|im_end|>"
    extra_stop_tokens: tuple[str, ...]
    max_think_tokens: int = 512

@dataclass(frozen=True)
class LoraSpec:
    keys: Literal["auto", "attention+mlp", "all-linear"] | tuple[str, ...]
    rank: int; scale: float; dropout: float

@dataclass(frozen=True)
class ModelSpec:
    name: str; hf_id: str; family: str
    chat: ChatSpec; lora: LoraSpec
    train: dict[str, Any]                 # max_seq_length, batch_size, grad_accumulation_steps, learning_rate, grad_checkpoint
    cache_strategy: Literal["auto", "trim", "snapshot", "none"]
    probe_layer_fractions: tuple[float, ...]
    memory_budget_gib: float
    policies: dict[str, str]              # {"A": "outputs/agent-v2/best-adapter", ...}, per model
    def resolve(self, model, tokenizer) -> "ResolvedSpec": ...

@dataclass(frozen=True)
class ResolvedSpec:
    spec: ModelSpec
    num_layers: int; hidden_size: int; vocab_size: int; tie_word_embeddings: bool
    layer_types: tuple[str, ...]          # "attention" | "linear_attention" per block
    lora_keys: tuple[str, ...]; trainable_parameters: int
    probe_layers: tuple[int, ...]
    cache_strategy: Literal["trim", "snapshot", "none"]
    snapshot_revision: str | None         # HF cache snapshot hash
    jvp_method: Literal["forward", "finite_difference", "untested"]
    def as_dict(self) -> dict[str, Any]: ...

def load_model_spec(name_or_hf_id: str) -> ModelSpec: ...   # configs/models/<name>.yaml or defaults
def registered_models() -> list[str]: ...
```

Consumers: `pipeline/cli.py` (every stage), `pipeline/evaluate.py::load_policy`,
`pipeline/protocol.py::build_prompt`, `pipeline/runner.py::make_turn_cache`,
`pipeline/data.py::write_dataset`, `probes/policies.py`, all probe CLIs, `provenance.py`,
`research/cache_equivalence.py`, `research/jspace_sweep.py`.

### 2.2 `src/local_llm_lab/arch.py` (owner: SPEC-001 §2)

```python
class ArchitectureView:
    @classmethod
    def from_model(cls, model: nn.Module) -> "ArchitectureView": ...
    text_module: nn.Module; blocks: list[nn.Module]
    num_layers: int; hidden_size: int
    def layer_kind(self, index: int) -> Literal["attention", "linear_attention"]: ...
    def embed(self, ids: mx.array) -> mx.array: ...                    # float32
    def masks(self, h: mx.array, cache: list | None) -> dict[str, Any]: ...
    def run_block(self, index: int, h: mx.array, masks: dict[str, Any], cache_i: Any | None) -> mx.array: ...
    def final_norm(self, h: mx.array) -> mx.array: ...
    def unembed(self, h: mx.array) -> mx.array: ...
    def make_cache(self) -> list[Any]: ...
    @property
    def cache_trimmable(self) -> bool: ...
    def lora_targets(self, policy: str | tuple[str, ...]) -> tuple[str, ...]: ...
    def residuals(self, ids: mx.array, layers: Sequence[int]) -> dict[int, mx.array]: ...  # one pass, deepest layer only
    def tail(self, layer: int) -> Callable[[mx.array], mx.array]: ...  # blocks[layer:] + final_norm with per-kind masks; used by J-lens
```

Consumers: `probes/capture.py` (all functions), `pipeline/jlens.py` (`residual_at`,
`jacobian_vector_product`, `jlens_map`, `readout`, `logit_lens`), `probes/adapter_delta.py`
(`base_weight`, readouts, `--ablate`), `pipeline/runner.py` (`SnapshotCache`),
`preflight`.

### 2.3 `src/local_llm_lab/pipeline/protocol.py` (owner: SPEC-001 §3-4)

```python
def build_prompt(tokenizer, messages: list[dict[str, Any]], *, spec: ModelSpec, keep_last: int = DEFAULT_KEEP_LAST, generation: bool = True) -> str: ...
def generation_suffix(spec: ModelSpec) -> str: ...          # asserted at the end of build_prompt output
def strip_thinking(text: str) -> tuple[str | None, str]: ... # (thinking or None, remainder)
def parse_turn(text: str) -> Turn: ...                        # unchanged; callers strip first
def turn_is_complete(text: str) -> bool: ...                  # ignores fences inside an open <think>
def render_completion(thought: str, action: Action, *, spec: ModelSpec) -> str: ...  # note + fenced call + end_of_turn
```

`Turn` gains `thinking: str | None = None`. Removed: the `tools=` parameter of `build_prompt`
(it was already discarded). Every caller in §3 must pass `spec=`.

### 2.4 `src/local_llm_lab/pipeline/tasks.py` (owners: SPEC-002 §1 for the signature; SPEC-003 §1-3 for templates and levels)

```python
def difficulty(split: str, index: int, override: int | None = None) -> int: ...
def make_tasks(split: str, count: int, seed: int = 20260902, *, perturb: bool | None = None, difficulty: int | None = None) -> list[Task]: ...
@dataclass(frozen=True)
class Task:  # existing fields plus
    difficulty: int
def task_from_id(task_id: str, seed: int, difficulty: int | None = None) -> Task: ...  # rebuild by id; SPEC-002 §2 uses it
```

`task_id` format is unchanged (`{split}-{family}-{index:04d}-{variant}`). Because a split's
difficulty can be overridden, every evaluation JSON, rollout file, probe npz, and manifest must
record `difficulty` per task (see §2.7, §2.9) so `task_from_id` can rebuild it.

### 2.5 `src/local_llm_lab/pipeline/data.py` (owner: SPEC-001 §3; SPEC-003 §3 adds splits)

```python
def build_rows(task: Task, *, keep_last: int = DEFAULT_KEEP_LAST) -> list[dict[str, Any]]: ...
# row = {"messages": [...windowed..., assistant], "metadata": {..., "difficulty": int}}   (unchanged shape)
def render_rows(rows, tokenizer, *, spec: ModelSpec) -> list[dict[str, Any]]: ...
# adds "prompt": build_prompt(messages[:-1]) and "completion": render_completion(...) to each row
def write_dataset(output: Path, splits: dict[str, SplitSpec], *, seed, keep_last, chat_dir, chat_repeats, recovery_repeats, extra_dirs, tokenizer, spec) -> dict[str, Any]: ...
@dataclass(frozen=True)
class SplitSpec: count: int; difficulty: int | None = None; perturb: bool | None = None; role: Literal["train", "valid", "test"] = "train"
```

`write_dataset` keeps writing `train/valid/test.jsonl` for mlx-lm compatibility; with SPEC-003
splits, `train.jsonl` is the concatenation of every split whose `role == "train"`. Rows keep
`messages` (probes and stripping need it) **and** carry `prompt`/`completion` (training needs
it). The manifest records the resolved `ModelSpec`, the per-split difficulty, and the hashes.

### 2.6 `src/local_llm_lab/tuner_data.py` (owner: SPEC-001 §7)

```python
class RenderedRowsDataset:               # mlx-lm dataset protocol: __len__, __getitem__ -> (tokens: list[int], offset: int)
    def __init__(self, rows: list[dict[str, Any]], tokenizer, *, max_seq_length: int): ...
def load_rendered_splits(data_dir: Path, tokenizer, *, max_seq_length) -> tuple[RenderedRowsDataset, RenderedRowsDataset, RenderedRowsDataset]: ...
```

Consumed by `cli.py::stage_train` through `mlx_lm.lora.train_model`.

### 2.7 `src/local_llm_lab/pipeline/runner.py` (owner: SPEC-001 §4-5; SPEC-002 §4 adds fields)

```python
class TurnCacheBase(Protocol):
    cache: Any; reused_tokens: int; encoded_tokens: int
    def prepare(self, token_ids: list[int]) -> list[int]: ...
    def commit(self, token_ids: list[int], generated: list[int]) -> None: ...
class TrimCache(TurnCacheBase): ...       # today's TurnCache
class SnapshotCache(TurnCacheBase):
    def __init__(self, model, view: ArchitectureView, prefix_tokens: int): ...
def make_turn_cache(model, view, resolved: ResolvedSpec, *, prefix_tokens: int) -> TurnCacheBase | None: ...
def generate_turn_with_count(model, tokenizer, prompt, sampler, max_tokens, turn_cache=None, *, spec: ModelSpec) -> tuple[str, int, int]: ...  # (text, tokens, think_tokens)
def run_task(model, tokenizer, task, *, sampler, spec: ModelSpec, view: ArchitectureView | None, resolved: ResolvedSpec | None, label, max_steps, max_tokens, keep_last, faults, transcript, use_cache) -> Trajectory: ...

@dataclass
class Trajectory:  # existing fields plus, all with defaults
    difficulty: int = -1
    think_tokens: int = 0
    integrity: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)   # ResolvedSpec.as_dict() subset
# steps[i] gains "thinking": str | None
```

Callers of `run_task`: `evaluate.evaluate_tasks`, `rollout.collect_rollouts`, `branch.mine_pairs`
and `branch._continue`, `research/cache_equivalence.py`. Callers of `Trajectory(**record)`:
`assistant_axis.trajectory_projections` (defaults make old evals load).

### 2.8 `src/local_llm_lab/pipeline/env.py` (owner: SPEC-002 §3)

```python
@dataclass(frozen=True)
class Verdict:  # existing fields plus
    unexpected_files: tuple[str, ...] = ()
    raw_answer: str | None = None
class Simulator:
    initial_files: dict[str, str]         # snapshot taken in for_task
    executed_tools: set[str]              # tools that executed without error; required_tools checked against this
```

New reason strings: `"unexpected file change: <name>"`. `evaluate.failure_reason` and
`report.render` must recognise it.

### 2.9 `src/local_llm_lab/pipeline/integrity.py` (owner: SPEC-002 §2)

```python
@dataclass(frozen=True)
class Fact: kind: str; value: str; observed_at: int   # kind in {"amount","metric","load","next_key","worker","path","total"}
def required_carry(task: Task, *, keep_last: int) -> dict[int, frozenset[Fact]]: ...   # step -> facts needed at that step and hidden by then
@dataclass(frozen=True)
class Violation: step: int; kind: str; detail: str
@dataclass(frozen=True)
class IntegrityReport:
    violations: tuple[Violation, ...]
    first_violation: Violation | None
    counts: dict[str, int]
    clean: bool
    def as_dict(self) -> dict[str, Any]: ...
def check_trajectory(task: Task, steps: list[dict[str, Any]], *, keep_last: int) -> IntegrityReport: ...
def completion_patterns() -> tuple[re.Pattern, ...]: ...   # shared with SPEC-003 generator invariant tests
def main() -> None: ...                                    # agent-v2-integrity CLI (pyproject entry)
```

Consumers: `evaluate.evaluate_tasks` (after each trajectory), `evaluate.failure_reason`
(integrity kinds precede loop/exhausted), `rollout.collect_rollouts` (keep filter),
`tests/test_pipeline.py` generator invariants (SPEC-003 §1.4-1.7), SPEC-004 §5 scorer.

### 2.10 `src/local_llm_lab/pipeline/evaluate.py` (owner: SPEC-002 §1, §4)

```python
def load_policy(spec: ModelSpec, adapter: Path | None, *, lazy: bool = False) -> tuple[Any, Any, ArchitectureView, ResolvedSpec]: ...
def run_evaluation(*, spec: ModelSpec, adapter, split, limit, difficulty, stress, ...) -> dict[str, Any]: ...
def summarize(trajectories) -> dict[str, Any]: ...   # adds wilson_95, integrity, think_tokens_per_task, by_difficulty
def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]: ...
def mcnemar(a: dict[str, bool], b: dict[str, bool]) -> dict[str, Any]: ...   # exact binomial on discordant pairs
```

Consumers of `load_policy`: every probe CLI (`state_probe.py:1810`, `assistant_axis.py`,
`jlens.py`, `adapter_delta.py`), `cli.py`, `rollout.py`, `branch.py`,
`research/cache_equivalence.py`, `research/jspace_sweep.py`.

### 2.11 `src/local_llm_lab/pipeline/cli.py` (owner: SPEC-001 §6-7, §10; SPEC-002 §1 for `select`)

Config schema additions (all YAMLs under `configs/`):

```yaml
model: qwen25-coder-3b            # registry name or HF id
splits:                           # SPEC-003; when absent, fall back to tasks: {train, valid, test}
  train:  {count: 240, difficulty: 0, perturb: true,  role: train}
  train1: {count: 120, difficulty: 1, perturb: true,  role: train}
  valid:  {count: 24,  difficulty: 1, perturb: false, role: valid}
  valid2: {count: 24,  difficulty: 2, perturb: false, role: valid}
  test:   {count: 180, difficulty: 2, perturb: false, role: test}
  test3:  {count: 60,  difficulty: 3, perturb: false, role: test}
select:
  screen: [{split: valid, per_family: {default: 1, long: 3}}, {split: valid2, per_family: {default: 1, long: 3}}]
train:
  lora_keys: auto                 # overrides spec.lora.keys when present
```

New stages: `preflight` (SPEC-001 §10). `stage_train` calls `mlx_lm.lora.train_model`
in-process (SPEC-001 §7) and writes `lora.yaml` with the resolved `lora_parameters.keys` and
`num_layers = resolved.num_layers`. `stage_select` implements the SPEC-002 §1 score and writes
component scores. Every stage calls `provenance.write_provenance`.

### 2.12 `src/local_llm_lab/provenance.py` (owner: SPEC-001 §9)

```python
def write_provenance(run_dir: Path, *, resolved: ResolvedSpec | None, spec: ModelSpec, extra: dict[str, Any]) -> Path: ...
def source_tree_hashes(root: Path) -> dict[str, str]: ...
```

### 2.13 `src/local_llm_lab/probes/capture.py` (owner: SPEC-001 §2, §8; SPEC-004 §5 adds `replace`)

```python
def capture_residuals(view: ArchitectureView, token_ids, layers, *, positions="last" | "all" | Sequence[int]) -> dict[int, mx.array]: ...
def response_mean_activations(view, tokenizer, prompt: str, response: str, layers) -> tuple[dict[int, mx.array], dict[str, Any]]: ...
class InjectionHook:
    def __init__(self, view, layer, vector, *, alpha: float, from_position=None, at_positions=None, replace: bool = False): ...
def lora_block_mask(view, keep_layers: Sequence[int]) -> ContextManager[None]: ...
def strip_state_fields(text: str) -> str: ...     # unchanged
def stub_observations(messages, *, keep_last: int = 0) -> list[dict[str, Any]]: ...   # SPEC-004 §2; wraps protocol.window_messages
```

### 2.14 `src/local_llm_lab/pipeline/jlens.py` (owner: SPEC-001 §2)

```python
def residual_at(view, token_ids, layer) -> mx.array: ...
def jacobian_vector_product(view, layer, primal, tangent, *, method: Literal["forward", "finite_difference"] = "forward") -> mx.array: ...
def jlens_map(view, layer, probe, corpus_ids, *, position=-1, method=...) -> tuple[mx.array, dict[str, Any]]: ...   # stats include "method"
```

`_causal_mask` is removed; masks come from `view.masks`. `jspace_sweep.py` and
`adapter_delta.py` currently import private helpers (`_distribution`, `_encode`); promote them to
public names `distribution` and `encode` and update both callers.

### 2.15 Probe CLIs (owner: SPEC-001 §8; SPEC-004 §2-5 extend)

Common flags on `agent-v2-probe-state`, `agent-v2-probe-axis`, `agent-v2-probe-delta`,
`agent-v2-jlens`, new `agent-v2-probe-patch`: `--model <registry name>`, `--policy <name|dir>`
(resolved through `spec.policies`), `--layers <fractions or indices>`, `--output`,
`--allow-busy-gpu`. `state_probe` adds `--stub-observations`, `--splits-plan <name>`
(`mix` today; `p2` for SPEC-004 §2), `reanalyse` and `compare` subcommands. `adapter_delta`
adds `--ablate --blocks N --screen <config>`.

### 2.16 `pyproject.toml` entry points to add

`agent-v2-integrity = "local_llm_lab.pipeline.integrity:main"`,
`agent-v2-probe-patch = "local_llm_lab.probes.patch:main"`. `agent-pipeline preflight` is a
subcommand, not a new entry.

## 3. File ownership matrix

| File | SPEC-001 | SPEC-002 | SPEC-003 | SPEC-004 |
| --- | --- | --- | --- | --- |
| `configs/models/*.yaml` (new) | owner | | adds `policies` for D adapters | |
| `configs/agent_v2*.yaml` | adds `model:` | adds `select.screen` | adds `splits:`; new `agent_v2d*.yaml` | |
| `src/local_llm_lab/models.py` (new) | owner | | | |
| `src/local_llm_lab/arch.py` (new) | owner | | | |
| `src/local_llm_lab/provenance.py` (new) | owner | | | |
| `src/local_llm_lab/tuner_data.py` (new) | owner | | | |
| `pipeline/protocol.py` | owner (§3-4) | | | |
| `pipeline/data.py` | owner (§3) | | adds `SplitSpec` handling | |
| `pipeline/tasks.py` | | owner of signature (§1) and §5 fixes | owner of templates and levels | |
| `pipeline/env.py` | | owner (§3) | | |
| `pipeline/integrity.py` (new) | | owner | uses `completion_patterns` in tests | uses `check_trajectory` in patch scorer |
| `pipeline/runner.py` | owner (§4-5) | adds fields (§4) | | |
| `pipeline/evaluate.py` | `load_policy` signature | owner (§1, §4) | | |
| `pipeline/cli.py` | owner (§6-7, §10) | `select` (§1) | `data` splits | |
| `pipeline/transcript.py` | prints thinking | truncation fix (§5) | | |
| `pipeline/report.py` | | owner (intervals, McNemar, new reasons) | | |
| `pipeline/rollout.py`, `branch.py`, `prefer.py` | `spec`/`load_policy` plumbing | integrity filter; `chosen = raw` | | |
| `pipeline/jlens.py` | owner | | | uses in P6 |
| `probes/capture.py` | owner | | | adds `replace`, `stub_observations` |
| `probes/state_probe.py` | view, layers, `--model` | | | owner of `reanalyse`, `compare`, p2 plan, stubbing |
| `probes/assistant_axis.py` | view, layers, `--model` | | | matched prompts, exemplar balance, judge off, CLOSED.md |
| `probes/adapter_delta.py` | parsing, view, JVP dedup | | | `--ablate` |
| `probes/patch.py` (new) | | | | owner |
| `probes/policies.py` | owner (per-model table) | | | |
| `research/cache_equivalence.py` | `--model`, `--strategy` | | | |
| `research/jspace_sweep.py` | public jlens helpers, `spec` | | | |
| `tests/test_pipeline.py` | rendering, cache, thinking | verdict, select, stress, integrity | generator invariants, levels, disjointness | |
| `tests/test_probes.py` | fake hybrid model, view contracts | | | reanalysis, stubbing, ablation, patch |
| `tests/test_probes_axis.py` | | | | matched design tests |
| `pyproject.toml` | | `agent-v2-integrity` | | `agent-v2-probe-patch` |

## 4. Known wiring gaps to close explicitly

These are places where a spec changes one end of a connection and the other end is easy to
forget. Each has an integration check in §6.

1. `build_prompt` signature change → callers: `runner.run_task`, `branch.mine_pairs`,
   `branch._continue`, `state_probe.build_probe_dataset` (line 427-432), `assistant_axis.trajectory_projections`
   (line 944-956), `jlens._replay_to_step`, `jspace_sweep.py`, `data.render_rows`.
2. `load_policy` signature change → callers listed in §2.10.
3. New `Trajectory` fields → `evaluate.summarize`, `transcript.step/finish`, `report.render`,
   `assistant_axis.Trajectory(**record)`, `rollout.trajectory_rows`, `branch`.
4. `Task.difficulty` and `task_from_id` → `state_probe.task_difficulties` (must read
   `task.difficulty`, not `tasks.difficulty(split, index)`), `integrity`, `evaluate` (record per
   trajectory), `rollout` and `branch` outputs, probe npz `difficulty` array.
5. `Verdict.unexpected_files` and the new reason string → `evaluate.failure_reason` ordering,
   `report.render` columns, `summarize.failure_reasons`, transcript PASS/FAIL line.
6. `spec.chat.thinking` → `runner.generate_turn_with_count` (budget, forced close),
   `protocol.turn_is_complete`, `protocol.strip_thinking` before `parse_turn`,
   `transcript.step` (collapsed print), `assistant_axis.response_mean_activations` (the response
   span must exclude the thinking block when measuring note position; record both spans),
   `state_probe` capture position (last prompt token is after the generation suffix in every
   mode; assert `generation_suffix(spec)`).
7. `lora_keys` resolution → `cli.stage_train` YAML, `provenance`, `adapter_delta` parsing (must
   accept `in_proj_qkvz`, `in_proj_ba`, `out_proj`), `capture.lora_block_mask` (must find LoRA
   modules by type, not by the seven names).
8. `ResolvedSpec.probe_layers` → `state_probe`, `assistant_axis`, `adapter_delta`
   `--readout-layers`, `jlens --layers`; all outputs record fractions and indices.
9. `SnapshotCache.prefix_tokens` → computed in `run_task` as the token length of
   `build_prompt(tokenizer, messages[:2], spec=spec, generation=False)`; must be recomputed if
   the system prompt or task prompt changes (they do not within a task).
10. `write_dataset` role-based concatenation → `cli.stage_data` manifest, `tuner_data.load_rendered_splits`,
    `state_probe` `--splits-plan p2` (reads `SplitSpec` from config, not hard-coded names).
11. `integrity.check_trajectory` needs `keep_last` and the `Task` → `evaluate_tasks` has both;
    `rollout` has both; the CLI rebuilds tasks via `task_from_id` using the eval JSON's recorded
    `seed` and per-trajectory `difficulty`; eval JSON must therefore record `data_seed`.
12. `mcnemar` in `report` needs paired task ids → both evaluations must record `task_id`
    (they do) and identical `split`/`difficulty`.
13. `adapter_delta --ablate` reuses `evaluate.run_evaluation` with a screen config → the screen
    definition lives in the pipeline YAML, so `--screen configs/agent_v2d.yaml` reads
    `select.screen` from it.
14. `probes/patch.py` needs C's saved trajectories (`outputs/agent-v2c/evals/best-adapter-test.json`)
    and B-style notes rendered from ground truth → import the note renderers from `tasks.py`
    as public functions (`render_expert_note(task, step_index)`), do not copy template text.
15. `preflight` writes `outputs/preflight/<name>.json` → `cli.stage_train/eval` and probe CLIs
    refuse to proceed without it unless `--skip-preflight-check`; the check compares the
    recorded `hf_id` and snapshot revision.

## 5. Config and artifact schema changes (summary)

| Artifact | Added keys |
| --- | --- |
| `data/<run>/manifest.json` | `model` (resolved spec), `splits[*].difficulty`, `splits[*].role`, `rendering: {thinking, template_kwargs, generation_suffix}` |
| `outputs/<run>/lora.yaml` | `lora_parameters.keys` (resolved list), `num_layers` from the view |
| `outputs/<run>/provenance.json` | new file, SPEC-001 §9 |
| `outputs/<run>/evals/*.json` | per trajectory: `difficulty`, `think_tokens`, `integrity`, `steps[*].thinking`; summary: `wilson_95`, `integrity`, `by_difficulty`, `think_tokens_per_task`, `model`, `data_seed` |
| `outputs/<run>/selection.json` | `components` per checkpoint, `wilson_95`, `val_loss`, `disagreement` |
| `outputs/probes/**/*.json` | `model` (resolved), `layers: {fractions, indices}`, `command`, `controls` |
| `outputs/preflight/<name>.json` | new file, SPEC-001 §10 |

## 6. Integration checks (run all before hand-off; add as tests where marked T)

1. T: `python -c "import local_llm_lab.pipeline.cli, local_llm_lab.probes.state_probe, local_llm_lab.probes.assistant_axis, local_llm_lab.probes.adapter_delta, local_llm_lab.pipeline.jlens, local_llm_lab.pipeline.integrity"` imports without a model.
2. T: grep for the banned constants (`01-IMPLEMENTER-BRIEFING.md` §1.7) returns hits only in
   `configs/models/`, `arch.py`, tests, and legacy modules.
3. T: `build_prompt` on the fake tokenizer ends with `generation_suffix(spec)` for each thinking
   mode; `strip_thinking` round-trips a synthetic `<think>…</think>note\n```json…``` turn.
4. T: a `Trajectory` round-trips through `as_dict()` and `Trajectory(**d)`; an evaluation record
   from `outputs/agent-v2c/evals/best-adapter-test.json` loads with the new defaults (skip if
   missing).
5. T: `task_from_id` reproduces `make_tasks` output for every split in a config, including
   overridden difficulties.
6. T: `agent-pipeline data` regenerates `data/agent_v2c` byte-for-byte into a temp dir under
   `qwen25-coder-3b` (`messages` rows), and the added `prompt`/`completion` tokenise identically
   to the old `messages` rendering through the fake tokenizer.
7. T: the fake hybrid model passes the view contracts, capture equivalence, snapshot-cache
   equivalence (greedy fake generation identical with and without the cache), and block-mask
   coverage.
8. T: `check_trajectory` on synthetic trajectories fires each violation kind exactly once;
   running the CLI over the saved B and C evaluations reproduces the memo's counts.
9. T: `summarize` and `report.render` handle records with and without the new fields.
10. Manual: `uv run pytest -q` green; the implementation report lists every caller touched for
    each row of §4.
