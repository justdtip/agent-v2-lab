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
    vocab_size: int; tie_word_embeddings: bool                        # accepted amendment, R3
    def lora_parameter_count(self, keys: Sequence[str], rank: int) -> int: ...   # accepted amendment, R3
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
   accept the installed split names `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj` and the combined `in_proj_qkvz`, `in_proj_ba`; classify by final path segment, never by a fixed list), `capture.lora_block_mask` (must find LoRA
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
16. Name collisions: `branch.render_completion(thought, action)` is replaced by
    `protocol.render_completion(thought, action, *, spec)` when SPEC-001 §3 lands (update
    `branch.py:33` and its callers); `state_probe._preflight_section` is a report section
    unrelated to the preflight stage and is renamed `_gate_section`.

## 5. Config and artifact schema changes (summary)

| Artifact | Added keys |
| --- | --- |
| `data/<run>/manifest.json` | `model` (resolved spec), `splits[*].difficulty`, `splits[*].role`, `rendering: {thinking, template_kwargs, generation_suffix}` |
| `outputs/<run>/lora.yaml` | `lora_parameters.keys` (resolved list), `num_layers` from the view |
| `outputs/<run>/provenance.json` | new file, SPEC-001 §9 |
| `outputs/<run>/evals/*.json` | `verdict.answer`/`verdict.expected_answer` are normalised, `verdict.raw_answer` original, `verdict.unexpected_files`; `transcripts.jsonl` starts with a `run_id` header record (R6); per trajectory: `difficulty`, `think_tokens`, `integrity`, `steps[*].thinking`; summary: `wilson_95`, `integrity`, `by_difficulty`, `think_tokens_per_task`, `model`, `data_seed` |
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

## 7. Rulings log (Research Scientist; newest last; these override earlier text)

- **R1 (2026-09-03 20:30) `cache_strategy: auto` semantics.** `auto` resolves to `trim` when
  `view.cache_trimmable`; otherwise to `snapshot` **only if** the registry file carries
  `cache.equivalence_verified: {date, sha256 of research/cache_equivalence.py output}`; otherwise
  to `none`, and `ResolvedSpec.cache_strategy_reason` records which branch fired. The runner
  prints one warning line when `none` is chosen on a hybrid. This reconciles the plan ruling
  (`none` for unverified snapshot) with SPEC-001 §5 (snapshot must be proven per model). An
  implementation that resolves `auto` straight to `snapshot` is not acceptable.
- **R2 (2026-09-03 20:30) allowed literal.** `"<|im_end|>"` may appear in
  `models.py::_default_spec` as the fallback `ChatSpec.end_of_turn` for unregistered HF ids, in
  addition to `configs/models/` and `arch.py`. Nowhere else.
- **R3 (2026-09-03 20:30) documents in `pending/` are not edited by implementers.** Record
  discrepancies and amendments in the task report and, if a shared decision is needed, append a
  dated bullet under "Implementer amendments" at the end of this file; the Research Scientist
  folds accepted amendments into the specs. The DeltaNet projection-name correction and the
  additive `ArchitectureView` fields (`vocab_size`, `tie_word_embeddings`,
  `lora_parameter_count(keys, rank)`) are accepted and now part of §2.2.
- **R4 (2026-09-03 20:30) registry schema addition.** `configs/models/<name>.yaml` gains
  `cache: {strategy: auto, equivalence_verified: null}`; `ModelSpec` gains
  `cache_equivalence_verified: dict[str, str] | None`.
- **R5 (2026-09-03 21:20) generator versioning replaces the C byte-for-byte rule.** SPEC-002
  §3/§5 legitimately change generated rows (failing-step notes, two prompts). Add
  `GENERATOR_VERSION: int` to `tasks.py` (1 = the generator at commit 97d197c that reproduces
  `data/agent_v2c`; 2 = HEAD after SPEC-002 §3/§5; bump on every row-changing change). Record it
  in every manifest and in `provenance.json`. Replace the acceptance "regenerates `data/agent_v2c`
  byte-for-byte" (SPEC-001 §11, briefing §1.8) with: a test pins the train/valid/test hashes of the
  current generator version for the reference config, and the SPEC-001 rendering migration test
  compares old `messages` rendering against new `prompt`/`completion` rendering on the *same*
  generator version. The last commit that reproduces run C's data is 97d197c.
- **R6 (2026-09-03 21:20) `transcripts.jsonl` header record.** The first record is
  `{"run_id": ...}`; every reader skips records without `task_id` via
  `transcript.iter_task_records(path)`.
- **R7 (2026-09-03 22:10) P2 cohorts.** Only `train-` rows are SFT-identical (737, not 1,050);
  `sft_disjoint` is for paired adapter comparisons and is compared within difficulty only;
  `all_rows` is the reportable cohort for a base run. See
  `under_review/SPEC-004-s1-REVIEW-round1-2026-09-03.md`.
- **R8 (2026-09-03 22:10) within-position reporting.** Every within-position cell reports
  `n_test` and `n_cells`; below 24 test rows or 5 eligible cells print `n/a (n=…)`; the pooled
  `overall` row appears in markdown; bootstrap intervals are computed over task ids restricted
  to eligible cells.
- **R9 (2026-09-03 22:40) no accidental snapshot.** Until Task 5 applies R1, hybrid registry
  files set `cache.strategy: none` explicitly; the temporary `auto → snapshot` branch is
  commented `TEMPORARY until R1` and its test is `xfail(strict=True)`.
- **R10 (2026-09-03 22:40) per-module test files.** New tests go in `tests/test_<module>.py`;
  existing tests move only when their owning lane next touches them. Purpose: stop file-level
  claims on the two monolithic test files from serialising independent lanes.
- **R11 (2026-09-03 22:40) atomic-write threshold.** Metadata-sized writers may write directly;
  anything over 1 MiB or containing arrays uses temp file, fsync, `os.replace`.
- **R12 (2026-09-04 09:00) versioned replay.** When `GENERATOR_VERSION` at HEAD exceeds the
  version in a saved artifact's manifest, retroactive tools (`agent-v2-integrity` over saved
  evaluations, `agent-v2-probe-state reanalyse`) replay the note-template functions of the
  artifact's recorded version, kept in-repo behind a version switch used for replay only;
  generation for training always uses HEAD templates; standing contract tests pin their
  artifact's version explicitly. (Deputy's §8 proposal of 06:15, promoted.)
- **R13 (2026-09-04 09:00) green gate at hand-off.** A lane may not report complete, request
  review, or release its claimed paths with a red full suite; a landing that turns the suite
  red is an incident requiring an issue within the cycle. Three red landings occurred on
  2026-09-03 night; each was caught, but the gate moves to hand-off.
- **R2 amended (2026-09-04 09:00):** the end-of-turn literal is additionally permitted in
  tests that assert registry or template round-trips; never in fixture logic that would mask a
  layout bug. (Deputy's proposal, promoted.)
- **Ratified 2026-09-04 09:00:** the four PROPOSED bullets of 00:20–05:50 in §8 are promoted:
  §2.13/§2.14 carry the Task 3 tuple returns, `lora_block_mask -> Iterator[int]`, and the
  declared `positions=` kwarg; §2.3 records the phased `build_prompt` contract with its two
  conditions (spec threading now; Task 6 removes the default and `tools=`); §2.5/§5 wording
  becomes "registry `ModelSpec` (data stage) / resolved spec (model stages)";
  §2.4 gains `family_balanced_tasks(split, *, difficulty, per_family, seed) -> list[Task]`;
  the §2.11 screen example gains its `difficulty:` key; `stage_select`'s missing
  `write_provenance` call remains owed.
- **R14 (2026-09-04 09:40) training entry API.** SPEC-001 §7's `train_model(args, model,
  tokenizer, train_set, valid_set)` was the Chief's error. Adopt the installed mlx-lm 0.31.3
  signature `train_model(args, model, train_set, valid_set, training_callback=None)`
  (`mlx_lm/lora.py:216`); the tokenizer is consumed only by `RenderedRowsDataset`. Keep the
  version pin check at import (`mlx_lm.__version__ == "0.31.3"`, else abort with a message).
  Pass a `training_callback` that writes `outputs/<run>/metrics.jsonl` (one record per report
  and per validation: step, train_loss, val_loss, tokens, elapsed) so `stage_select`'s val-loss
  component reads structured metrics instead of parsing `train.log`. No dependency change.
- **R15 (2026-09-04 13:30) training-run readiness gate.** Seven conditions in
  `records/COMPLETENESS-ASSESSMENT-round1-2026-09-04.md`; first arm B4. Lifting the ban is
  the Director's act per run.
- **R16 (2026-09-04 13:30) report accuracy.** Every claimed deliverable cites file:line; the
  reviewer spot-checks three; an uncited claim counts as not done.
- **R17 (2026-09-04 13:30) scanner and layer fractions.** Substring pass added to the
  banned-constant check; probe `--layers` accepts fractions and defaults to
  `spec.probes.layer_fractions`.
- **R18 (2026-09-04 13:30) precision.** Residual-equivalence tolerance is Frobenius relative
  ≤ 1e-2 with elementwise maxima reported; `ModelSpec.probes.capture_dtype: native|float32`
  (default native; float32 block path only for J-lens tail/JVP with deviation recorded);
  SPEC-004 §1 correction C7 (bfloat16-rounding refit of the saved P2 npz); memo caveat.
  Briefing rule 1.5 amended.
- **R18a (2026-09-04 14:05) preflight gate refined.** The residual-equivalence gate compares
  the view's manual loop run in the model's **native dtype** against the native forward
  (expected exact; tolerance Frobenius relative ≤ 1e-4). The float32-path deviation is measured
  and reported as the `fp32_manual_vs_native` block (Frobenius relative plus elementwise maxima; name amended 2026-09-04 19:30), never gated;
  it informs `capture_dtype` and is recorded in every probe artifact. Supersedes R18(a).
- **R19 (2026-09-04 14:05) independent reviewer.** Under the Deputy-run workflow every
  implementation is reviewed by a separate reviewing agent (not the implementer, not the
  dispatcher's own pass) before the Deputy's readiness verdict and the review-request issue;
  the reviewer's findings are attached to the issue verbatim.
- **R20 (2026-09-04 15:10) `mine_pairs` view/resolved debt.** Optional `view`/`resolved` in
  `branch.mine_pairs` is accepted debt, marked `# DEBT(R20)`, and expires with the
  condition-4 completion slice (`resolve_policy` on `ModelSpec.policies`, `branch.build_prompt`
  threading, `_load_training_base` via `load_policy(adapter=None, lazy=False)`), which makes
  them required and updates the two optional-path tests. Wave 1 ratified in
  `complete/WAVE1-LOAD-POLICY-REVIEW-round1-2026-09-04.md`.
- **R21 (2026-09-04 15:40) dataset write guard and the render stage.** (a) `write_dataset`
  and every stage that writes under `data/` refuse to write into a directory that already
  contains `manifest.json` unless `--force-overwrite` is passed explicitly (flag name amended 2026-09-04 20:40); targets inside a protected directory refuse too (K1, review round 1); `data/agent_v2`,
  `data/agent_v2b`, `data/agent_v2c`, and `data/chat_replay` are additionally listed in
  `PROTECTED_DATASETS` and refuse even with `--force-overwrite`. (b) New stage
  `agent-pipeline render --source <dir> --output <dir> --model <name>`: reads existing
  `messages` rows from `<source>/{train,valid,test}.jsonl`, applies `render_rows` for the
  named model's spec (thinking mode from the registry), writes the three files plus a
  manifest recording the source directory, its per-split SHA-256, the source manifest's
  `GENERATOR_VERSION` if present, and the rendering metadata, and `provenance.json`. It never
  calls `make_tasks`. (c) Arm B4 is defined as `render --source data/agent_v2b --output
  data/agent_v2b-qwen35-4b --model qwen35-4b`; `configs/agent_v2b_qwen35_4b.yaml` sets
  `data: data/agent_v2b-qwen35-4b` and `source_rows: data/agent_v2b`, and `stage_data` for a
  config with `source_rows` delegates to `render`. Regenerating with `stage_data` would produce
  generator-v4 rows, which are not run B's data.
- **R22 (2026-09-04 18:05) P6 inputs.** (a) `agent-v2-probe-patch` accepts `--data-seed`;
  evaluation JSON `data_seed` is used when present, the flag when absent, and the tool errors
  only when neither exists; the artifact records the seed and its source. (b) The
  counterfactual note is, by preference, the **passing run's saved note** at the same task id
  and step, used when the passing trajectory's actions up to the decision step equal the
  failing run's; otherwise `render_expert_note` at HEAD, and each case records
  `counterfactual_source: passing_transcript | generator_v<N>`. Rationale: run B's generator
  revision is unrecoverable and the in-repo v1 templates are run C's rewrite, so a regenerated
  note is not B's; the saved note is the one that empirically led to a pass.
- **R23 (2026-09-04 21:30, Director-ratified on issue #25; folded by the Chief).**
  Pre-versioning artifacts (runs A, B, C, their datasets and saved evaluations) bind to
  generator version 1 for every retroactive replay, on the recorded rational basis (R5 defines
  v1 as the C-reproducing generator; no version field could exist before versioning). Tools
  replaying them record both the version used and this basis. Post-versioning artifacts get no
  such latitude: an absent `GENERATOR_VERSION` there is a defect. A needed-but-missing binding
  fails closed with a named R12 error, never defaulting to HEAD.
- **R24 (2026-09-04 21:30) P6 scoring version.** Flip scoring may judge value reappearance at
  HEAD only when, for that case, the dropped value and the decision step are identical under
  the bound replay version and HEAD; the artifact records `scoring_version_stable` per case
  and the headline includes only stable cases, with unstable cases listed separately. The
  eligibility recomputation and the verdict filter (failing run must fail; SPEC-004 §5's
  universe) are ratified as the R22 extension.
- **R25 (2026-09-04 23:20) P6 treatment alignment (issue #28).** Grounded in `patch.py`:
  `InjectionHook(replace=True)` maps source row *i* to `at_positions[i]` and requires equal
  counts, and the failing prompt has no position for the dropped value. Therefore:
  (a) **Unequal groups align on the tail** (`previous_notes`, and any group whose cardinality
  differs): the last |target| source positions patch the target; the leading source residue
  is recorded per case as `unpatched_source_tokens`. The decision-adjacent context is the
  note's end, and tail alignment keeps absolute-position offsets to a few tokens.
  (b) **`note_value_tokens` splits into two cells.** `shared_value_tokens`: values present in
  both notes, aligned by string identity through `_note_values`, equal cardinality by
  construction. `dropped_value_slot`: for each dropped value, the source residual rows of its
  tokens are mean-pooled to one row and replace the target residual at the **single
  separator position immediately after the preceding shared value** (the slot where the value
  should have appeared); the artifact records the slot position, the token it overwrote, the
  source tokens pooled, and the dropped value. This is the write-side test the memo asks for.
  (c) Controls keep their existing resampling to the post-alignment treatment cardinality;
  `random_positions` for `dropped_value_slot` draws one non-treatment position.
  (d) `POSITION_GROUPS` becomes seven cells; the heat map, R24 stability record, and the
  five-case caveat are unchanged. Every cell records the alignment rule and cardinalities.
- **R26 (2026-09-05 03:20) run health as a training gate.** Adopts the Deputy's proposal
  (`under_review/RUN-LOGGING-DESIGN-AND-GATE-PROPOSAL-2026-09-05.md` §4) as an addition to
  R15's evidence format, amended: (a) every training and probe run writes `<output>/run.log`
  and `<output>/events.jsonl`; every training run also writes `<output>/health.json` on every
  exit path; (b) `non_finite_loss` aborts the run and its adapters are ineligible for
  selection; (c) a new fatal flag `incomplete_run` (iterations done < planned, or no checkpoint
  at the final iteration) makes the run ineligible unless the Director lifts it explicitly
  with the shortfall stated; (d) **Director's amendment 2026-09-05 03:40: training does not start until logging is in
  place.** R15 gains condition 8: lanes A (`runlog.py`) and B (training integration,
  `health.json`, abort path) are committed and their fake-only tests green before any
  training lift; the run's health record starts at iteration one. An arm's **evaluation
  lift** additionally requires `health.json` with verdict `healthy` or `warnings`, every warning listed in the lift request with the Deputy's
  reading, and `run.log`/`events.jsonl`/`health.json` cited by path and SHA-256; (e) the
  `start` event records the resolved `ModelSpec` name and hf_id, the config path, the data
  manifest hash, and the git commit, so `run.log` alone identifies the run; (f) `health.json`
  and the thresholds are copied into `provenance.json`; (g) probe runs emit at least one
  progress line per outer unit of work (case, layer, task, or checkpoint), so no run is silent
  for more than the length of one unit; (h) thresholds are engineering defaults recorded per
  run and changed only through the arm config's `train.health:` block; (i) `metrics.jsonl`
  and `train.log` stay byte-compatible (selection reads them). Contract in the proposal §5 is
  normative for the four lanes; lane A lands first.
- **R27 (2026-09-05 05:40) P6 strict flip scoring.** A flip requires: parseable turn; the
  note's value set contains every value present in the failing note and the dropped value
  (digit-bounded) and no number outside the canonical set; per-generation outcome
  `flip | corrupted | unchanged | parse_error` and the generated note text recorded in the
  artifact; per-case `dropped_value_visible_in_retained_observations` computed with the
  integrity module's fact extraction (True excludes the case); a content control (B's rows with
  the dropped value's positions swapped for an unrelated value's) added. Rationale: the
  stale-field coverage path in `integrity.py:384-395` scores any wrong number as "no value
  drop". Review: `under_review/P6-RESULT-REVIEW-round1-2026-09-05.md`.
- **R28 (2026-09-05 06:20) P2 split disjointness.** Tasks are seeded by
  `f"v2:{seed}:{split}:{index}"` (`tasks.py:130` after the B1a insertions) and embed the split name in every workspace
  path and key token, so new split names are disjoint from every training split by
  construction. The test asserts both the mechanism and the content: (a) no P2 split name
  (`p2-d0/1/2`) appears in any `configs/*.yaml` `splits:`/`tasks:` block; (b) a content
  fingerprint, SHA-256 over canonical JSON of (family, difficulty, variant, prompt with the
  split root and `KEY-/REF-<SPLIT>` tokens normalised, sorted files with the split root
  normalised, expected answer), collides with no fingerprint of any training split of any
  config. Absent-data configs are **regenerated through `make_tasks` from their split table**
  (deterministic, no files needed); the test never skips.
- **R29 (2026-09-05 06:20) `compare` pairing.** `reanalyse` emits a per-row prediction sidecar
  `<stem>.predictions.npz` (test-half rows: row id, task id, difficulty, label, and the probe,
  position-baseline and surface-baseline predictions per target × layer × split seed).
  `compare` computes the paired bootstrap of margin differences over **shared task ids** from
  two sidecars with one resample seed list, refusing unless both share task ids, split seeds,
  layers (as fractions), generator version and cohort; adapter-versus-base comparisons are
  reported within difficulty (R7). Recomputing fits from two npz files is the fallback only
  when a sidecar is absent, and is recorded as such.
- **Assignment (2026-09-05 06:20):** `ModelSpec.probes.capture_dtype` (R18b) lands in the P2
  redesign capture sub-slice (B1b), before any capture on Qwen3.5.
- **R23 addendum (2026-09-05 08:20):** the hardened P2 capture `state-base-mix.npz` binds its
  labels to generator v1 and its reanalysis/refit to generator v2 on the recorded basis that
  `--no-round` at v2 reproduces the baseline byte for byte while v4 flips 7 cells.
- **R26 amendment (2026-09-05 08:20):** on any exit before the planned iterations complete,
  error path included, `health.json` records `incomplete_run`; a crashed run can never read
  `healthy`. Evidence: the first live B4 attempt (0 iterations, status error, verdict healthy).
- **R29 addendum (2026-09-05 08:20):** `compare` adds the Holm-adjusted flag beside the
  interval flag on the difference table (follow-up line).
- **R30 (2026-09-05 08:20) P6 secondary condition and scorer refinements.** `_note_values`
  accepts words between `half` and the colon; the `aggregate_report` secondary condition's
  eligibility is judged under HEAD alone with the basis recorded (no v1-bound side); the
  expected set is field-scoped to the failing note's list field; `flip_unsatisfiable` cases are
  marked and excluded from the headline; a parseable note with no values scores `empty`;
  control rates are over applicable cases with n printed. Primary ledger rerun first (after the
  pooled content-control fix), secondary after these land.
- **R31 (2026-09-05 09:05) library-protocol tests.** Any test of a seam with mlx-lm (dataset
  protocol, trainer callback, cache classes, LoRA layer types, tokenizer boundary behaviour)
  drives the library's real class on that side, with fakes only for weights and compute.
  Evidence: three defects (view walker vs `LoRALinear`, `RenderedRowsDataset` vs
  `CacheDataset`, health finish path) reached live runs because fakes stubbed both sides.
- **R32 (2026-09-05 10:30) gated-delta training memory (issue #50).** (a) Training-time
  chunked checkpointing of `gated_delta_ops` approved as a SPEC-001 backbone slice (stage 1,
  bit-exact against the library's loop; installed only during `stage_train`), with the
  chunkwise-parallel form as stage 2 gated on measured step time (tolerance-tested) and a kernel
  VJP as stage 3. (b) `memory.budget_gib` = min(registry, device recommended working set)
  resolved at preflight. (c) `train.gated_delta_chunk` is an arm-config field (default 64),
  recorded. (d) Preflight's training footprint estimate (longest row, configured batch and
  chunk) joins R15 condition 1 with 10% headroom. Validation runs on the kernel path via
  `model.eval()` around the trainer's evaluate. Review:
  `under_review/GATED-DELTA-TRAINING-MEMORY-REVIEW-round1-2026-09-05.md`.
- **R32 addendum (2026-09-05 12:40):** the preflight training footprint is a calibrated
  upper-envelope model `peak = a × retained_state_bytes + b` fitted from the lane probe's
  measured peaks (points, coefficients and date recorded; re-fit when points are added);
  lane 2 does not gate until the fit exists. B4 attempt 4 gate: longest row steps under the
  working set with 10% headroom and 400 iterations project at ≤ 3 hours; otherwise stage 2
  first. mlx-lm already runs validation in eval mode; the wrapper is for restore-on-exception.
- **R33 (2026-09-05 12:40) no stashing on the shared tree.** Implementers never run
  `git stash` on the shared tree (two incidents); `git show HEAD:path` and worktrees instead. A
  stash is reported as an incident and the Deputy re-verifies every other lane's edits.
- **R34 (2026-09-05 23:50; proposed by the Head of Interpretability) J-lens conformance statement.** Every J-lens artifact and
  `pipeline/jlens.py` state which variant of the source paper's recipe was computed: target
  layer **and the index convention (layer L = residual after block L-1; kind = the block that
  wrote it)**, source positions, output positions read (`self` / `future` / `all`), corpus
  size and context length, median future window, JVP method (amended 2026-09-05 on the Head
  of Interpretability's addendum). A reading resting on a null cites the variant. The current implementation is the
  self-only limiting case at eight contexts; the 3B World A entry gains that footnote.
- **R35 (2026-09-05 23:50; proposed by the Head of Interpretability) cross-model comparability.** Two probe tables are comparable only
  where policy, derivative method, layer selection (as fractions and kinds), generator version,
  prompt rendering (template kwargs) and estimator variant are equal, or every difference is
  named in both artifacts. Base-versus-adapter is a named difference, not a comparison.
- **R36 (2026-09-05) review chains by domain.** Interpretability slices (SPEC-004 and EXP
  specs; anything touching `probes/`, `pipeline/jlens.py`, capture primitives, or a probe
  artifact): implementer → independent conformance reviewer (R19: signatures, banned
  constants, protected paths, no model loads, suite bare) → **Head of Interpretability's
  domain review and readiness verdict** → Chief's gate. Pipeline and training slices
  (SPEC-001 to 003, R32, logging): unchanged, implementer → R19 reviewer → Deputy's verdict →
  Chief's gate, with the Head consulted when a slice touches capture, J-lens, or a probe
  artifact. The Deputy dispatches and keeps the board for every lane. A correction the Head
  makes directly comes to the Chief with the R19 pass and no domain review. Readings of probe
  results are the Head's and ratified by the Chief.
- **R29 addendum 2 (2026-09-05) one Holm family definition.** `compare` uses the reanalysis's
  family: one family per (cohort, scope) spanning every (target, layer) cell, with the two
  controls collapsed into one cell by the larger p-value (a difference counts only if it holds
  against both baselines), within-difficulty scopes being separate families. `holm_supported`
  stays strictly additional to the interval flag. `COMPARE_HOLM_FAMILY` declares this, the
  artifact names it, and a test asserts the two tools partition a synthetic result
  identically. Raised by the Head of Interpretability on #57.
- **R37 (2026-09-05) preflight schema bumps land with their regeneration.** A commit that
  changes the preflight artifact schema is committed only as one operation with the
  regeneration of every `outputs/preflight/*.json` on the lane; the Director's approval of the
  commit is the lift for those runs, and no other stage runs between them.
- **R26 amended (2026-09-05):** on non-training stages `incomplete_run` is an end-event field
  meaning "exited before writing its report", carrying no verdict.

- **R37 amended (2026-09-05 night).** `outputs/` is untracked, so "land with the regeneration"
  means the regenerated artifacts are recorded in the commit message and the readiness
  checklists by path, snapshot revision and SHA-256 (as `f1230ac` does), never force-added.
- **R38 (2026-09-05 night; proposed by the Deputy, ruled by the Chief) a reader is tested
  against the real writer.** R31 extended from library seams to every seam: any test of code
  that reads an artifact, a record, a configuration or a library object must build its fixture
  through the real writer (the artifact's own `write_report`, the library's own class, the
  real `ArchitectureView` over a real tiny model), never by hand. A hand-made fixture certifies
  the reader's belief about the seam rather than the seam, and it passes exactly when the
  reader is wrong in the way its author expected it to be right. Three misses on one night
  had this shape: a period walk whose fake was not a `dict` subclass as `mlx.nn.Module` is; a
  precision-block reader whose fake carried the key at the level the reader hoped for rather
  than where the preflight writes it; and a stage nothing drove through its own entry point
  (#64). At a gate the reviewer checks the fixture's provenance, not only the assertion.
  Where a structural read and a configuration walk both exist, the structural read is
  primary and the configuration is the cross-check.
  Corollary (Head of Interpretability, 2026-09-05 morning): a verification is a statement
  about a specific tree and expires when the tree moves, whoever made it; a pre-run check is
  redone against the commit the run will cite.
- **R38 amended (2026-09-05, from the #70 audit summary): a soft default cannot tell a missing
  key from a moved one.** A reader of an artifact this repository writes indexes the keys it
  depends on and raises on absence; a default is permitted only where a named artifact on disk
  requires it, and the docstring names that artifact. A fixture built through the real writer
  is hygiene; the test is moving each depended-on key and confirming red. A brief's claim about
  the code is a hypothesis the implementer confirms or refutes, and a refutation is a good
  outcome. Record: `under_review/R38-AUDIT-SUMMARY-2026-09-05.md` (fifteen readers, four
  wrong, all four with a soft default).
- **R38 amended (2026-09-05, from the EXP-002 order): confirm-or-refute binds the officers.**
  A claim about the code in a ruling, a brief or a review is a hypothesis until re-measured,
  whoever made it. Three instances in one hour, all reasonable readings of source (two
  functions with one name in different modules; an inherited property absent from the
  subclass; a default that changes a return type), two the Deputy's, one the Head's, one the
  Chief's ("a cached run never reaches the sentinel": `KVCache.make_mask` returns `'causal'` by
  default). Each was caught by someone re-measuring rather than accepting.
- **R38 amended (2026-09-05, proposed by the Head of Interpretability): a quoted figure names
  its source.** Nothing recomputes a sentence. Three figures in one morning were right when
  written and outlived their ruling, all in prose: a "ten cases" size estimate transcribed into a
  pre-registration as a designed count; a summary describing a corrected rerun beside the
  attempt it did not describe; a readiness assertion against a tree that had moved. A number
  quoted in a checklist, a role document, a pre-registration or a reading names the artifact,
  ruling or computation it came from, so the next reader can recompute it instead of trusting
  it; a figure with no named source is an estimate and is labelled as one.
- **R38 note (2026-09-05, EXP-002 S3): a fixture can be real and still be the wrong real
  thing.** S1 and S2 were accepted against a four-block hybrid whose only attention block was
  last, so a mask applied only at the decision step and a mask carried on every forward gave
  identical logits (0.0) there and differed by 0.232 on an eight-block fixture with two
  attention blocks. The seam test's fixture must be able to exhibit the failure the test is for:
  where the mechanism is propagation, the fixture needs somewhere for it to propagate through.
- **R38 amended (2026-09-05 evening; proposed by the research division via the Head of
  Interpretability): a figure names the rows read against the rows available.** Naming the
  population is not enough: a script that asked for 400 rows of a split holding 381 read the
  whole population while reporting a 400-row sample, and a spread quoted from the mild end of a
  split read as the split's. Every count, sample or window states "n of N" with N the rows the
  source holds, so a sample that exceeds its population, or a window inside one, is visible on
  its face.

- **R34 amended (2026-09-05; proposed by the Head of Interpretability, ratified by the Chief):
  context-constant lens rows enter no Holm family (#78).** Holm families in the J-space sweep
  are built on the paired statistic, not on `matched_p`; a row with zero discordant pairs has an
  undefined paired test and enters no family by construction; the discordant-pair count is
  recorded per row and zero renders as unresolved; the conformance block states that
  `matched_p` is an unpaired test against a fair coin, reported and not decisive. No magnitude
  floor: the phenomenon is per-case agreement of the two contexts' win indicators (42/42 at
  `jlens_L5_future`, `jlens_L20_future`, `jlens_L12_self`, `logit_lens_L12` on the 4B artifact),
  not float32 resolution (0/84 equal cells) and not magnitude (`jlens_L12_self` constant at
  8.5e-3 while `jlens_L5_all` discriminates at 6.9e-7). The Chief's earlier resolution-floor
  ruling is withdrawn. Implementation changes the Holm code; a filter in front of the old
  families is not this ruling. Full text: §8, PROPOSED (Interp) 2026-09-05, ratified.

- **R39 (2026-09-05 evening; proposed by the Head of Interpretability with the research
  division, ruled by the Chief): a choice carries the reason that decided it.** A tie-break, a
  convention (sidedness, tie handling, a threshold) or a design constant recorded in a spec, a
  ruling or a reading is recorded with the reason it was chosen over its alternatives, so the
  next reader can check the choice rather than trust it. Kept separate from R38, which is
  mechanically checkable (a figure names its source and its n of N); this is a discipline about
  rules rather than numbers. Worked examples: EXP-003's partner for layer 16 (17 over 15 on a
  balance criterion, with the measurements that decided it); the P6 contrast's one-sided Fisher
  figure, which nearly escaped without its convention named.

## 8. Implementer amendments (append-only, dated)

- PROPOSED (Deputy, 2026-09-04 00:20): §2.13/§2.14 amendments to match the disclosed Task 3
  deviations (issue #6 N2), for promotion or deletion: `response_mean_activations` and
  `jlens_map` return a `(result, stats)` tuple for view callers and keep the legacy bare-dict/
  bare-array return for raw-model callers until `assistant_axis.py` and `test_pipeline.py`
  migrate (Task 8); `lora_block_mask` returns `Iterator[int]` yielding the masked-module count;
  `InjectionHook`'s extra `positions=` convenience kwarg is either declared in §2.13 or removed.
- PROPOSED (Deputy, 2026-09-04 00:20): reconcile R2 with §6 check 2. R2 permits the end-of-turn
  literal in `models.py::_default_spec`, `configs/models/` and `arch.py` "nowhere else"; §6
  check 2 additionally allows "tests, and legacy modules". Commit `b39d6ea` added a test
  asserting the registry round-trip (`tests/test_probes.py:1912`), a literal R2 hit that §6
  permits. Proposed text: R2 gains "and in tests that assert registry or template round-trips;
  never in fixture logic that would mask a layout bug."

- PROPOSED (Deputy, 2026-09-04 05:40): record Task 4's phased `build_prompt` contract in §2.3.
  Until SPEC-001 Task 6: `spec` defaults to `None` and resolves to the qwen25-coder-3b
  compatibility spec; `tools=` remains accepted and discarded. Two conditions attach: (a) the
  three probe CLIs and the runner thread their loaded spec into every `build_prompt` call now
  (issue #10 T1), re-enabling the generation-suffix assertion for non-3B models — the
  assertion being skipped in compatibility mode is the hazard; (b) Task 6 removes the default
  and the `tools=` parameter and this bullet. Bundled: the dataset manifest records
  `asdict(ModelSpec)`, not the resolved spec, because the data stage loads no model — either
  amend §2.5/§5's "resolved" wording to "registry `ModelSpec` (data stage) / resolved spec
  (model stages)" per the Codex provenance ruling on issue #3 C3, or require a follow-up.

- PROPOSED (Deputy, 2026-09-04 05:50): §2.4 addition — `family_balanced_tasks(split, *,
  difficulty, per_family, seed=20260902) -> list[Task]` (tasks.py:135), the screen-cell
  sampler `stage_select` uses; validates quota keys and non-negative ints. Also: the §2.11
  `select.screen` example omits the `difficulty:` key that `stage_select` (cli.py:298) and
  `family_balanced_tasks` require — add it to the example. Also: `stage_select` still lacks
  its `write_provenance` call (asked of the lane on issue #7); the map's "every stage" sentence
  stays normative.

- PROPOSED (Deputy, 2026-09-04 06:15): versioned replay ruling (issue #11). When
  `GENERATOR_VERSION` at HEAD exceeds the version recorded in a saved artifact's manifest,
  retroactive tools (`agent-v2-integrity` over saved evals, `agent-v2-probe-state reanalyse`)
  replay the note-template functions of the artifact's recorded version, kept in-repo behind a
  version switch used for replay only; generation for training always uses HEAD templates.
  Standing contract tests pin their artifact's version explicitly. Alternative if versioned
  replay is refused: retroactive tools refuse with a named error and their tests skip with the
  version mismatch stated.

- PROPOSED (Deputy, 2026-09-04 09:05): record the Director's model-execution rule as a numbered
  ruling (issue #12). The prohibition on model runs is lifted; execution is confined to a single
  designated lane under a single task, project-wide, whose UUID the coordinator records. Every
  other lane and every test remains fake-only: no checkpoint or tokenizer loads, no `mlx_lm.load`,
  no `load_policy` against real weights, and briefing §7 rule 4 ("never run a model to just
  check") still binds them. Briefing §1.1 needs the corresponding edit; its own wording
  anticipates the lift.

- PROPOSED (Deputy, 2026-09-04 11:20): definition attaching to the model-execution ruling
  (issue #12), from the Director. **Model execution means loading the model into memory and
  running inference**; the rule exists to prevent multiple resident models competing for the
  machine. A tokeniser load is not execution, so tokeniser downloads, chat-template rendering
  and dataset generation are ordinary fake-only work available to any lane. The single
  execution claim still gates `mlx_lm.load`, `load_policy` against real weights, safetensors
  weight loads, and any forward or backward pass: preflight, training, evaluation, rollouts,
  probe capture, and cache-equivalence attestation.

- PROPOSED (Deputy, 2026-09-04 16:20): fold the Director's ratification (issue #25) beside R12
  and R22: pre-versioning artifacts (runs A, B, C, their datasets and saved evaluations) bind to
  generator version 1 for retroactive replay on the recorded rational basis — R5 defines v1 as
  the C-reproducing generator, the v1/v2 and v3/v4 template structure brackets the residual
  uncertainty, and no version field could have existed before versioning did. Retroactive tools
  record the version used and the basis. Post-versioning artifacts get no such latitude: an
  absent field is a defect.

- RATIFIED by the Chief (2026-09-05 night; R38 applies) — proposed by the Deputy 2026-09-05 04:40: supersession, R32 stage 2 (K7). The analytic per-layer
  gated-delta **state-bytes model is superseded by the fitted empirical envelope in
  `pipeline/preflight.py`**, which is what actually gates a training arm; `chunkwise_state_bytes`
  had had no consumer since `f1230ac` replaced the state-shape-derived footprint estimate with a
  fit that reads no state shape at all, and it is removed with this slice. The reason is worth
  keeping on the record: an analytic memory model that nothing checks against measurements is
  what this work repaired — the old estimator's analytic retained-state sum predicted a 2.12x
  spread across chunk lengths where the measurement is 1.07x. Do not re-derive one; the note in
  `training/gated_delta_chunkwise.py`'s docstring says so at the site.

- **RATIFIED by the Chief (2026-09-05, promoted to §7) — PROPOSED (Interp) 2026-09-05: R34 amendment — unresolved lens rows, and the family they must
  not enter (#78).** For the Chief to promote into §7. Occasioned by rows in
  `outputs/probes/jspace-qwen35-4b-base-2026-09-05-rerun/sweep.json` whose matched and mismatched
  columns are the same data.

  **The phenomenon.** A readout can give the *same* per-case answer whether the candidate pair is
  scored against its own task's context or a foreign one. Exact diagnostic: per-case agreement
  between the two win-indicator vectors. Measured on that artifact, `jlens_L5_future`,
  `jlens_L20_future`, `jlens_L12_self` and `logit_lens_L12` each agree in **42 of 42** cases.
  Such a row carries no information about context, cannot inform a matched-versus-mismatched
  test, and yet its unpaired p-value against a fair coin consumes Holm correction from rows that
  can.

  **Two things it is not, both checked rather than assumed.** It is *not* a float32 resolution
  problem: at `jlens_L5_future`, **0 of 84** cells hold two values equal in float32 and the median
  gap is **7.0e6 ulps**. And it is *not* predicted by magnitude: `jlens_L12_self` is
  context-constant at a median probability of 8.5e-03 while `jlens_L5_all` discriminates at
  6.9e-07 and `logit_lens_L11` at 1.6e-06. **A magnitude floor would exclude rows that
  discriminate and keep rows that do not**, which is why this amendment states no floor.

  **Ruled.**
  1. **Holm families are built on the paired statistic, not on `matched_p`.** The paired test uses
     only discordant pairs, so a row with **zero discordant pairs has an undefined paired test and
     enters no family by construction.** No threshold, no calibration.
  2. **The discordant-pair count is recorded per row**, and a row with zero renders as
     **unresolved** rather than as a rate.
  3. **The conformance block states that `matched_p` is an unpaired test against a fair coin**,
     reported but not decisive and the basis of no family.

  **Implementation note for the dispatch:** this changes the sweep's Holm code rather than adding
  a filter in front of it. A floor plus a paired test bolted on afterwards is not this ruling.

  **Scope.** Verdicts resting on the model's own output distribution or on the P(true) /
  P(already-read) decomposition are untouched: neither is an ordering statistic over lens rows.
  On the 4B artifact the rows carrying the verdict sit at the bottom of the agreement ranking
  (`jlens_L27_future` 32 of 42, `jlens_L32_self` 34 of 42), which is what a row that responds to
  context looks like. EXP-001's conclusion stands and stands more firmly once context-constant
  rows stop consuming correction.


### R40 (2026-09-05, evening; Director approved D1 to D6 of pending/03)

- **R40a, reading instrument.** The hosted Jacobian lens (`neuronpedia/jacobian-lens`, `qwen3.5-4b`, n1000 file, SHA-256 in `outputs/probes/jlens-hosted-qwen35-4b-2026-09-05/provenance.json`) is the programme's readout for J-lens measurements. Readout = `softmax(unembed(final_norm(h @ J.T)))` through `ArchitectureView`; file index l is repository layer L = l + 1; L = 32 is the identity. The finite-difference JVP path is legacy, kept only for the self-only and future-only variants, and is not used for reading. Holm families are the layers under one instrument.
- **R40b, probe layers.** `probes.layer_fractions` is replaced by band-based layers 13, 16, 20, 24, 28 for Qwen3.5-4B; the kind-matched partner rule of EXP-001 section 3.5 is unchanged. EXP-001 stands as recorded on the old fractions. EXP-003 and EXP-004 adopt the band before any code. The 9B entry is a placeholder until its band is measured with the same corpus stage.
- **R40c, standing lift.** See `live/STANDING-LIFT-EXPERIMENTS-2026-09-05.md`: ready-and-validated experiments run without a per-run lift; one model-loading experiment at a time; every result reported to the Director; visualisations are a deliverable. **Status (2026-09-05 late evening, final): settled. The Proxy relayed the Director's clarification that research and test runs proceed freely and the concurrency rule is the actual constraint; the live note's banner carries it verbatim. Readiness (ratified spec, pre-registered reading, power, fixture green) remains the Chief's gate on the design; the model-run lock of issue 83 is the constraint's mechanism and lands before the first run under this rule.**
- **R40d, dispositions.** Per `pending/03` section 5: land wt-template-prefix (after the four account corrections), wt-r38-slice1 to 4, wt-p6-scorer, wt-hybrid-period; hold wt-power; keep wt-s3; fold wt-exp002 into WP4.

### R41 (2026-09-05, evening; on the Head's WP3 design)

- **R41a, WP3 ratified with the Head's four corrections.** `pending/WP3-TRANSPORT-WEIGHTS-DESIGN-2026-09-05.md` is the WP3 specification. C1: attention blocks are block indices 3, 7, ..., 31 and write layers 4, 8, ..., 32; in-band attention blocks 15, 19, 23, 27 write layers 16, 20, 24, 28 (64 query heads); every layer set is written as block-to-layer. C2: the absolute threshold is withdrawn (|alpha| <= 0.0884 beta by the library's q and k scaling); the retention ratio normalised to bin 1-4 and the mass share against a uniform null are the statistics, readings R1 to R4 as written. C3: the R31 fixture uses Hv/Hk = 2 with distinguishable per-head keys against the library's own ops; the pairing is h // 2 by mx.repeat. C4: bins to 2688. Production path: the backward scan; oracle: the explicit product. Scope limit carried into every reading: a recurrent null is a content-transport null with routing frozen. All four verified by the Chief in code and on random tensors on 2026-09-05.
- **R41b, R40b amended.** The probe layers 13, 16, 20, 24, 28 were kind-confounded (four of five written by attention blocks). The set becomes the kind pairs 12/13, 15/16, 19/20, 23/24, 27/28, ten layers, one DeltaNet-written and one attention-written at each depth, named as pairs in the registry under WP2. Any WP3 or WP5 result is reported per pair. EXP-001 stays as recorded.
- **R41c, order of the Head's work.** WP7 design, then WP9, then R36 verdicts on WP1 and WP5 as those slices arrive.

### R42 (2026-09-05, evening; on the Head's WP7 and WP9 designs)

- **R42a, the J-space is the paper's, not the span.** `pending/WP7-WP9-PROBE-DESIGNS-2026-09-05.md` is ratified with one restatement that runs through all five probes. The J-space at a layer is the union of nonnegative cones spanned by at most k = 25 J-lens vectors (paper section 2.3 and appendix A.8); the J-space component of a direction is its gradient-pursuit fit with that k, and the remainder is what the fit leaves. The "dims for 90 percent variance" statistic of the hosted-lens memo (0.23 to 0.55 across the band) describes the linear span of all J-lens vectors and is context only; it is not the J-space's size, and the design's derived figures (about 947 dimensions at layer 20; a random direction carrying about 37 percent of its variance in the J-space; a remainder of about 63 percent) are withdrawn from every reading. The Head's structural requirement stands and is made exact: every share, rank and readout is reported as a percentile against spectrum-matched random directions at the same layer put through the same k-sparse pursuit, which is the paper's own "same-size random control"; every clamp is compared to the remainder and to a k-matched random control, the pursuit fit of a spectrum-matched random direction with the same k, at matched norm. With that null the paper's precedent of a 6 to 15 percent share is comparable again, and stays context rather than criterion (R35).
- **R42b, WP7(d) layers.** The state probes move to the R41b kind pairs 12/13, 15/16, 19/20, 23/24, 27/28, reported per pair; "13, 16, 20, 24, 28" in the design is superseded. Re-run, not re-analysis, as the design says.
- **R42c, WP9 prerequisite and criterion.** Adopted as written: the intermediate must be present by rank at band layers on at least a third of items before any clamp runs; the 54 percent Haiku figure is discussion, not criterion; the reading is the three-way contrast with the k-matched random control of R42a in place of the dimension-matched subspace.
- **R42d, order and dependencies.** WP7(d), WP7(c), the shared projector and clamp (gradient pursuit at k = 25 against the hosted dictionary; the paper's lens-coordinate patch with the pseudoinverse on the selected vectors), then WP7(a) with WP9, then WP7(b). All after WP1 lands. WP5's run waits on WP3's measured curve; the WP5 re-specification proceeds meanwhile.
- **R42e, the Base fork.** WP7(b)'s pre-written fork is adopted: if the Base shows the same paired differences, the checkpoint-pair contrast is retired on this lineage and Thread 1 is tested on training we control (WP7(c), WP10).

### R38(d) (2026-09-05, evening; general form of three fixture findings, at the Deputy's request)

A fixture must be able to fail in the way the code can fail. Three instances in two days lost that property without anyone noticing: the EXP-002 mask fixture that could not show the mask defect; the fake tokenizer that accepted a system-only prefix the real template refuses; and the read-weight fixture at equal key and value head counts that cannot exercise the h // 2 pairing `mx.repeat` performs. The recurring shapes are equal-sized dimensions where the code distinguishes sizes, single-kind blocks where the code branches on kind, and permissive stand-ins where the real class refuses. Every fixture that guards a branch names, in its docstring, the failure it can show and the shape that makes that failure reachable; a reviewer who cannot find that sentence treats the fixture as absent (R31 form).

### R43 (2026-09-05, evening; the Head's R36 verdicts on WP1 and WP5, `under_review/R36-VERDICTS-WP1-WP5-2026-09-05.md`)

- **R43a, WP1 proceeds with four amendments.** (1) The 42-case table is a validation run, not a fixture: WP1 carries a unit fixture on the seams in R31 form (no model load) and, separately, a validation run compared to the recorded table. (2) No count tolerance: the candidate ordering is deterministic given the same activations, lens file and dtype, so the validation run reproduces the table exactly or names the source of nondeterminism. (3) Layer 32 is reported and excluded from the J-lens Holm family; the lens covers blocks 0 to 30 and layer 32 is the identity, so a claim there is a claim about the output. (4) The quantisation gap is measured: a few tens of prompts through the bfloat16 post-trained model, top-k agreement and rank correlation at band layers against the 4-bit checkpoint; until it exists every WP7 and WP9 reading carries that conditional in one line. The bfloat16 pull is a model-loading run under the standing lift and is queued behind the calibration.
- **R43b, WP5 re-specification.** Full-vocabulary ranks, never a top-k dump, so the paired Wilcoxon on log-rank has no censored observations; the continuous rank is primary; recovery depth is descriptive over the R41b pairs plus "never", with no significance claim; the masked arm on EXP-002's arm-A machinery is carried in the re-specification from the start, because under F4 a flat open-render curve is the prediction and the masked arm is the informative experiment. B2 and B3 recorded as raised and settled by R41b.

### R44 (2026-09-05, late evening; standing rule at the Head's proposal, on the third instance in one day)

Every threshold gets the null its own generative form implies, and the null is stated with the threshold. A uniform or zero null is admissible only when the statistic's generative form makes it the expectation. Three instances on 2026-09-05: the 0.05 retention threshold (not scale-free; the library's scaling bounds the quantity below it), the linear-span variance share (a random direction scores it), and the top-ten source concentration (write strength times random alignment concentrates by itself). Companion to R38(d) and to the per-head rule of the WP3 design section 9.

### R38(e) (2026-09-05, late evening; the Deputy's corollary from the worktree cleanup)

A difference from the branch tip is not evidence of unlanded work: a landed change diverges from the tip whenever a later commit touches the same file. Absence of a line from the tip needs the history checked before it means unlanded: a line superseded by a later fix is absent for a reason. The check that answers "is this landed" asks whether each added line appears in the branch and, for those that do not, whether they appeared in an earlier commit and were superseded since. Recorded after the first-render fix landed as a922013 and the ten scratchpad worktrees were removed on that check.

### R45 (2026-09-05, late evening; the fork-after-Metal crash)

Instance: a Python process in the Deputy's session (report `Python-2026-09-05-202845.ips`, responsible claude 13279, venv under a temporary path with MLX, tokenizers and safetensors loaded) died six seconds after launch in `_posixsubprocess.fork_exec` under `subprocess.Popen`, because the Metal driver's memory-pool-decay thread held a malloc lock at the instant of `fork()` and macOS's fork handler aborted ("Unlock of an os_unfair_lock not owned by current thread"). Any process that has initialised Metal through MLX is multi-threaded by the driver, and a fork from it is unsafe and intermittent.

Rule: no process that may have initialised Metal forks. CPython's `subprocess` takes `posix_spawn`, which runs no fork handlers, only when `close_fds=False`, the executable is an absolute path, `cwd` is None and there is no `preexec_fn`, `start_new_session` or `process_group`; measured on the project interpreter 2026-09-05: default `Popen(sys.executable, ...)` forks, `cwd=` forks, a bare command name forks, and `close_fds=False` with an absolute executable and no `cwd` spawns, with `capture_output` and `env` unaffected. Therefore: one spawn helper in the package resolves the executable with `shutil.which`, passes `close_fds=False`, refuses `cwd` and `preexec_fn`, and is the only permitted way to start a process under `src` and `tests`, enforced by a repository rule; the test suite's conftest monkeypatches `subprocess._fork_exec` to raise so any fork in the suite fails loudly. Sites to migrate: `tests/test_runlock.py` (two Popen tests), `runlock._process_table`, `tests/test_probes.py:2155`, `probes/guard.py`, and the git calls in `runlog`, `integrity` and `provenance` wherever they can run after a model load. Acceptance items on issue 83.

R45, amended on the Deputy's review (2026-09-05, late evening). Both halves of the hazard are the rule: forking is not unsafe in itself; forking in an interpreter where Metal has been initialised is, and in the test suite that initialisation happens at collection through `tests/test_probes.py`, so a module that has never imported `mlx` is not exempt, since the interpreter it runs in already has. The conftest guard that makes any fork raise keys on a private CPython symbol (`subprocess._fork_exec`) and therefore carries its own test: a deliberate fork under the guard must raise, and the test must fail if the symbol it patches no longer exists, so the guard cannot become dead code that is believed. The helper's docstring states that `close_fds=False` is load-bearing (it is what selects `posix_spawn`) and that its cost is inherited descriptors in the child, accepted for that reason, so the flag is not deleted as an oddity. The crash is confirmed as the lock implementer's test run (worktree created 20:06:35, seven Popen sites in its test file; `procPath` is the Homebrew framework interpreter the venv resolves to).

R45 / issue 83, process-check refinements verified on a live instance (the Deputy, 2026-09-05 late evening). Instance: pid 63943, "Python ctxmax.py" from the Research Division session's scratchpad, a model-loading benchmark orphaned to launchd, invisible to every name-based check for twenty-three minutes; lsof shows libmlx.dylib, libjaccl.dylib and the AGX Metal bundle mapped. Two rules for the bridge check and for any later refinement of it: (1) match on any process holding a file whose basename is `libmlx.dylib`, never on the process name, since a scratch script with a shebang and another name slips a name filter; (2) never gate on resident memory: `ps` reports this process at about 25 MB resident against a 17.2 GB physical footprint, because Metal buffer allocations do not appear in RSS, so any refinement that uses a size to tell a real load from an incidental import waves through the very case that motivated the rule. Presence of the mapping is the whole signal. `vmmap -summary` or `footprint` report the real footprint when one is needed for reporting, never for gating.

### R46 (2026-09-05, late evening; benchmark and long-run guard practice, composed from the ctxmax run)

Three lessons from the 2026-09-05 context-length benchmark (pid 63943, Research Division session) compose into one mechanism, adopted for every long model-loading run. (1) The run emits a progress line per chunk or per row segment carrying elapsed time, tokens done, peak memory in GiB and peak as a share of the device's recommended Metal working set (`mx.device_info()["max_recommended_working_set_size"]`, 17.76 GiB on this box); a run that emits this cannot be mistaken for a hang and cannot silently cross the working set, which degrades throughput rather than raising and would otherwise read as a result. (2) An external watchdog tails that line and gates primarily on the working-set share, with swap free as the machine-level backstop at a margin of several hundred megabytes and the system free percentage as lagging confirmation only, never the trigger, since macOS defends that percentage by the very swapping the guard exists to detect. (3) The watchdog's lifetime derives from the expected run time times a multiplier, so it outlives its target by construction rather than by an estimated iteration count. The working-set fraction is one number for the codebase: preflight's, or a stated difference beside it (question to the Deputy, 2026-09-05). Any guard is verified from its own code in the process table, not from its owner's description.

**R46 amendment (2026-09-05, later; the Deputy read `preflight.py` and the Chief confirmed each line).** Three corrections to the paragraph above, which stands otherwise.

(i) *No fraction to share.* Preflight uses no fraction of the working set. Its budget is the minimum of the registry declaration and the device's recommended working set (`preflight.py` line 840), with `budget_source` naming which one bound, and `within_budget` is a plain comparison at line 853 with no margin. The only fraction in the file, 0.10 at line 74, is R32(d)'s estimation headroom: it multiplies a *predicted* training peak at line 1046 to cover the transients the estimate leaves out. It corrects an estimate and is not a ceiling margin. The benchmark measures its peak, so importing the 0.10 would correct an error the benchmark does not have and would create a second definition of "too close" under the appearance of sharing the first. Rule: a measured total gets no margin; an estimate gets its stated headroom; what is shared across the codebase is the budget resolution (minimum of declared and device, source named), never a margin. Any margin on a measured working-set share needs its own stated reason, and that reason cannot be preflight's. The last sentence but one of the paragraph above ("one number for the codebase: preflight's") is answered by this item.

(ii) *Two signals, two consequences.* Crossing the working set degrades throughput rather than failing, so the working-set share is a validity signal, not a safety one. A guard that kills on it destroys the datum it exists to protect. The response to crossing it is to mark the row degraded and keep the number with its label, so the figure survives and is known to be untrustworthy. Swap free is the safety signal, a machine-level concern, and the only one a guard kills on. Item (2) above is superseded where it says the watchdog "gates primarily on the working-set share": it gates on swap free and marks on the working-set share.

(iii) *Lifetime follows.* The degraded marker lives in the row and needs no lifetime. Only the killer needs one, and its lifetime derives as item (3) says.

(iv) *Swap free is read against a pool that grows* (2026-09-05, 21:17, from the Chief's own two readings). macOS sizes the swap pool on demand. At 21:16:59 `vm.swapusage` read 5,120 MB total, 458 MB free; at 21:17:38 it read 6,144 MB total, 1,674 MB free, with no change in the run between the two. A single-sample dip in swap free during a swapfile resize is not memory pressure, and a guard with a single-sample floor kills a run at the finish line for a pool resize: the Research Division's replacement guard had a 400 MB floor, sixty megabytes under that transient. A dip can therefore mean the system is about to give itself more room, not that it is about to run out (the Deputy's reading of the same measurement, 21:20), which inverts the signal's meaning on exactly the sample a floor would act on. So the primary kill signal is the kernel's own judgement, not a proxy we interpret: `sysctl kern.memorystatus_vm_pressure_level` (1 normal, 2 warning, 4 critical) at critical on two consecutive samples thirty seconds apart. Swap free is confirmation under it, read as a level held across consecutive samples with the pool total not having grown between them, never a trigger on its own; the pool's ceiling is disk space, so a low reading on a pool that can still grow is the system working. Warning (2) never fires: it read 2 throughout the benchmark's last quarter with the run healthy. The same persistence applies to any kill signal. Also from the same run: the working-set share of item (ii) is readable only from inside the run, so a run that predates the rule and emits no progress line cannot be marked on it; its rows carry "working-set share not measured, run predates R46" rather than a number an external guard cannot have.


### R48 (2026-09-05, 21:50; the Director's rulings on efficiency probes, verbatim in substance)

(a) *The benchmark's data stands as collected.* The first sweep (4k to 128k at chunk 2048) and the rebuilt run's partial rows (prefill to 98,816 tokens at chunk 512) are the record; no re-run. (b) *Efficiency probes run up to 64k context and no further.* A probe that exists to assess an engineering implication is allowed; one that would only benefit the model at its maximum context length is not worth the machine. The native-window question is closed at its projection. (c) *Standing order:* pursue research that optimises training and inference efficiency, and prioritise training over inference. (d) The first probe under (c) is training-row feasibility at 32k: inference at 32k is measured and feasible (440 tok/s prefill, 25 tok/s decode, 7.4 GiB), and the open question is the training step, whose attention-score term grows with the square of the row and which preflight's linear activation estimate (`preflight.py:818-822`) omits.


### R41d (2026-09-05, 22:10; roles under the band are per pair)

A declared band pair is one unit: both members carry the same role. Under R41b's pairs (12/13, 15/16, 19/20, 23/24, 27/28) roles are assigned per pair, so 27/28 is primary as a pair even though layer 28 lies past EXP-001's 5/6 fraction. EXP-001 stays as recorded on its own selection, pinned in its tests as the six fractions it ran with; any run under the band is an R35 named difference against it, not a supersession. Two facts from issue 80's implementation, recorded because a paraphrase already got each wrong once tonight: the pairs are not uniform in direction (13 pairs up to 12; 16, 20, 24, 28 pair down to 15, 19, 23, 27; derived from the library's layer kinds and R40b's set by the nearest-opposite-kind rule, lower index on a tie), and the pairs' attention members are layers 12, 16, 20, 24, 28, written by blocks 11, 15, 19, 23, 27: five blocks, 80 in-band query heads, not four and 64.


### R41e (2026-09-05, 22:30; R41b's pair for layer 16 corrected to the recorded tie-break: 16/17, not 15/16)

The Head rejected issue 80 as written, and the record upholds the rejection. EXP-003-DISTANCE-CURVE-QWEN35-4B.md lines 99-112 (research division, committed b93f294 at 19:38) rule that 16's partner is 17: both 15 and 17 are linear and equidistant, the choice was measured, not indexed, and the reason was written down "because the next person to extend the family inherits the rule and not the reason". R41b, ratified later the same evening, took 15 from the pattern "the attention layer and its predecessor", which is the Chief's slip; the implementer's derivation test then encoded the code's rule, lower index on a tie, and so agreed with the literal without either matching the record. The band is 12/13, **16/17**, 19/20, 23/24, 27/28; ten layers 12, 13, 16, 17, 19, 20, 23, 24, 27, 28. R42b and R43b read the pairs by reference and follow. On the band set the correction moves the linear members' mean depth from 19.4 to 19.8 against the attention members' 20.0, and attention is the deeper member in three of five pairs rather than four: the residual depth gap that lay in the direction of the kind difference is halved, which is what the pairing exists for. The attention members and their blocks are unchanged (12, 16, 20, 24, 28; blocks 11, 15, 19, 23, 27), so issue 81's capture plan does not move. Two consequences for code: the registry literal and the derivation test in issue 80 change, and the test's tie-break must consult the recorded ruling rather than an index rule; `pipeline/jlens.py` `kind_matched_layer_family` diverges from EXP-003's generalised rule in two clauses, not one (the Head's read of both sites, 22:35): it still says an in-band layer that is itself an attention output takes no partner (EXP-001 §3.5, `LAYER_FAMILY_RULE` at line 202 and the docstring at 764), so under the band it would leave 16, 20, 24 and 28 unpaired, and it breaks ties to the lower index. Both clauses move under issue 86, filed rather than widened into 80. The function is faithful to its own specification and stale against the 19:38 ruling; since the rule string is quoted into artifacts under R34, artifacts from its two callers (`jlens.py:1741`, `jspace_sweep.py:673`) since 19:38 carry the superseded rule as the rule. EXP-001 is unaffected: under its rule 16 took no partner, so the tie never arose.


### R38(f) (2026-09-05, 22:40; a test of a recorded ruling asserts the ruling and says so)

From issue 80's first derivation test, which agreed with the registry literal by luck. A tie-break that was measured is recorded, not derivable: EXP-003's measurement balanced its own family, and carried onto the band the recorded answer ranks above the alternative on both stated criteria without being singled out (ten of the sixteen possible bands reach the same 0.2 imbalance of mean depth). So any test that claims to derive such pairs encodes a rule of its own choosing, and its agreement with the literal proves nothing. Rule: where a value rests on a recorded ruling, the test asserts the ruling, its docstring says it is asserting a ruling rather than deriving a rule, and it names the ruling's location (here EXP-003 lines 99-112) so the test fails if either the code or the ruling moves without the other. The same applies to a function that carries such a value: under issue 86 the family function asserts the tie-break rather than computing it, and because its rule string is quoted into artifacts under R34, the string says so. (The Deputy's caution, 22:40, adopted verbatim in substance.)


### R49 (2026-09-05, 23:00; the shared working tree is append-only for records, and history-moving git runs only in worktrees)

Two losses of written work in the shared tree in one night and one false alarm, with all three mechanisms identified. (1) The Head's section 6a of the WP3 design, a tracked file, was most likely destroyed by the Deputy's `git reset --hard HEAD~1` at 21:43:20 (reflog), run in the shared tree to back out an ungated commit of issue 84 that had been applied there for convenience. (2) The Deputy's staged 84 patch rode into an unrelated commit, the same root cause. (3) **Retracted at 23:10, the Chief's own error.** The Chief reported 23 untracked files and a README append as written at 22:44 and confirmed on disk, then found the folder at its committed 18 files and treated it as a deletion by another session. No deletion occurred. The 22:43 command chained the page build and the copies with `&&`; the build failed on a syntax slip, the chain stopped before any copy, the path variable was never assigned, and the trailing `ls $R | wc -l` counted the scratchpad directory instead, which held 39 files. The transcript search across every session found no checkout, restore, clean or stash in the window, and the four sessions questioned each answered from their own records that they had run none. Everything was restored from the Chief's scratchpad and committed as b8ec4a1 within the hour.

Rule. (a) The shared working tree is append-only for records: a record written by any seat is committed within the hour so it cannot be lost to the next reset. (b) Patches are applied, tested and reset only in worktrees, never in the shared tree; the ungated 84 commit, its backout and the lost section all follow from breaking this clause once. (c) `git reset --hard`, `git clean`, `git stash` with untracked files, and `git checkout` of the whole tree or of paths another seat is writing are never run in the shared tree; `reset --hard` is named because it is the one that destroys uncommitted edits to tracked files and the one actually run. (d) A seat that finds work missing reports the timestamps and the signature (tracked or untracked, reverted or removed) before anything is rewritten, and rewrites from its own scratchpad, which is the source of truth for anything it produced. (f) A confirmation that something is on disk names the absolute path it checked and shows the listing, never a count, and a copy is never chained behind a step that can fail: the false alarm of 22:44 came from a broken `&&` chain and an unset variable, and cost four sessions a question each. (e) Every session keeps its own scratchpad copy of what it writes into the tree until the commit hash is back.


### R50 (2026-09-05, 23:25; adapter depth: the speed lever is free only where the adapter will not be read)

Probe 5 (TRAIN-COST-2026-09-05, variants M and N) measured adapters on the top 8 layers training at 276 tokens per second against 126 for all 32, and the top 16 at 206, at nearly the same memory. The Head's count against R41e's pairs decides where that lever may be used: an adapter that does not exist at a layer has no update direction for WP7(c) to read through the lens. The top 8 cover one of the five pairs end to end (27/28); the top 16 cover three (19/20, 23/24, 27/28) with 16/17 half covered; only the full set covers all five. Rule: (a) any run whose adapter is going to be read, WP10 and every run feeding WP7(c), trains the full set; this matters more if WP7(b)'s Base fork lands as the hosted-lens memo predicts, since WP7(c) then carries Thread 1 on training we control. (b) Throughput-only work, feasibility probes and task-performance runs where nothing is read, may use the top 8. (c) The defensible middle is the top 16, covering three whole pairs and the layers where EXP-001 saw context sensitivity appear and flip; the top 8 covers neither. (d) Before the lever is used at all, top-8 task performance is measured against the full set on the existing evaluation as an R35 named difference, because late-layer-only adaptation tends to remap outputs rather than change representations, which is what the programme observes; if it also trains worse the choice is moot. Issue 88 is scoped to (d).


### R43c (2026-09-05, 23:55; WP5 reports layer 16 separately, and no readout is named as the Holm survivor)

From the quantisation record (QUANT-GAP-2026-09-05): layer 16's paired discordant structure inverts between the 4-bit and bfloat16 checkpoints ([3, 2] against [0, 4]) with its case ranking half preserved (Spearman 0.53), so WP5's paired Wilcoxon on log-rank reports layer 16 separately rather than pooled into the band, since a rank statistic pooled across five pairs inherits the least stable one without showing it; any rank-based result at 16 states the checkpoint it was measured on, and the module records the checkpoint in every per-pair row. Under R40a's Holm family the surviving readout of the ordering test is L20 on 4-bit and L27 on bfloat16, so no artifact names a single layer as the one where the ordering effect lives; it is present at both on both checkpoints. The decomposition, within-checkpoint, is the leg that carries World A.


### R38(g) (2026-09-06, 00:20; a review gate waits for its reviewer)

A review gate waits for the reviewer's answer; silence is not assent and a timeout is not approval. The Head's count of the night: four designs would have produced a confident wrong answer and each catch came from the review, not the run (a threshold that could not fail, a fixture that could not see its own bug, five hooks on the wrong blocks, a null built against the wrong baseline). A timeout converts a busy reviewer into approval. The Chief set one on the retrieval-corpus design at 00:05 because the Head's socket had gone stale and the seat could not be seen to be live; that is the reason, recorded here, and not the rule. When a reviewer cannot be reached, the run waits or the Director decides.


### R38(h) (2026-09-06, 03:50; a corrected selection rule re-scores every prediction written against the old one)

From WP12 stage 1: prediction 1 (the broadcast heads sit in the first half of the band) was scored as confirmed on a set selected by the paper's gain-and-preservation rule, under which layers 16 and 28 held none. When the gain criterion was found not to transfer and the set was re-selected on preservation, layer 28 held the most (nine of sixteen) and the second half of the band held fourteen relays against the first half's nine: the prediction is refuted and the paper's own claim does not reproduce on this model. A prediction that changes verdict when the selection rule changes has to be seen to have been at risk. Rule: when a selection rule or a null is corrected, every prediction written against the old one is re-scored against the new one, both scorings are shown in the record, and no verdict is inherited. Companion rule from the same review: a control that is one realisation (a single fixed rotation, a single sample of control rows) is not a distribution, and a margin against it cannot say whether it is unusual; controls that can be resampled are resampled, at least twenty draws, and criteria are set at a percentile of the draws (the same point as R44 and R42's k-matched nulls, applied to weight-space controls).


### R51 (2026-09-06, 05:30; F10 carried into WP7 and WP9 as design changes, ratified)

The Head's section 0a of the probe designs, ratified as written. (a) Every WP7 and WP9 result is reported per R41e pair with that pair's lens-identity cosine beside it, never pooled, because pooling mixes instruments whose independence from the unembedding differs by a factor of seven across the band (0.10 at layer 4 to 0.69 at 28). This is the second independent reason for per-pair reporting; the first was layer 16's checkpoint instability under R43c; two arguments arriving at one disposition is recorded as such. (b) WP7(a) runs the k-sparse pursuit against the orthogonalised dictionary as well as the raw one and reports both: a component of the assistant axis that survives orthogonalisation at 27/28 is the finding, one that does not is the unembedding wearing the workspace's clothes. (c) WP9 gains an orthogonalised clamp as a fourth arm at the two deepest pairs: causal privilege that survives it is privilege of workspace content; privilege that does not is privilege of the output direction, which a J-against-remainder contrast at depth cannot distinguish. (d) WP7(c) gains a third arm in the blinded rating, adapter directions read through the orthogonalised lens, with the secondary result that raters distinguishing the lens from the plain unembedding at 12/13 but not at 27/28 would be the instrument's profile measured behaviourally, reported beside F10's cosine ramp. (e) WP7(d) reports its layer set per pair for consistency. Framing kept visible: the ramp is not a defect in the lens; a workspace that converges on the output as it approaches the output is what the paper predicts, and the last three layers being motor is on record from the persistence band; what F10 adds is that the convergence starts far earlier and is already substantial at layer 20, in the middle of the band.


### R41e(b) (2026-09-06, 07:00; band membership is read from the registry, never enumerated)

Four times in one night a list of band members was written out by extending the obvious rule and dropped pair 12/13, the one pair whose recurrent member pairs upward: R41b took 15 as 16's partner; the retrieval probe's attention hook named four blocks and missed block 11; the WP12 in-band denominator counted four attention layers and missed 12; and the WP12 ablation script carried the same four-layer list into its in-band set and its readout layers, caught by the Deputy's reading of the denominator (06:55) before the run. That is one property of the band rediscovered four times, not four mistakes. Rule: any list of band members, in any lane, scratch or code, comes from the resolved registry that issue 80 lands, or, until it lands, from the R41e literal asserted as the recorded ruling with its location named (R38(f)), and is never enumerated in prose or typed from the pattern. Any quantity computed "on the band" states its denominator as the registry gives it: ten layers, five attention members, eighty in-band query heads. (The Deputy's proposal, 06:55, adopted.)


### R53 (2026-09-06, 08:12; every preservation statistic carries a copy-map control by default)

Near-identity output maps, heads whose OV product is close to a scaled identity on the residual, confounded four separate measurements in one night: layer 8's zero passes under the broadcast rule, where random MLP rows preserved the population at 0.446; block 19 group 0's energy outlier in the key-value test; layer 20's between-group F, as strong on arbitrary populations as on the lens; and block 19 head 2 in the retrieval set, a copy map at 0.989 against draws at the same level. A copy map preserves any population's labels, captures any population's energy in its row space, and imposes group structure on anything, so it passes every statistic built on preservation unless the statistic is controlled against it. That is a recurring property of this model, not four coincidences. Rule: every statistic built on label preservation, gain, energy capture or their group structure carries a copy-map control by default, either the same statistic computed on the same real maps against arbitrary populations (the Chief's form of the F test: the structure the real heads impose on any population) or against explicit near-identity maps, and a preservation statistic reported without one is reported as uncontrolled. The need is not rediscovered per statistic. (The Head's synthesis, 08:10, adopted.)


### F12 and R54 (2026-09-06, 11:05; no band layer is both wide and independent, and layer 20 is the programme's primary lens readout)

F10 and F11 run in opposite directions across the band. F10, the lens-to-identity cosine, rises 0.29, 0.29, 0.48, 0.63, 0.71 at layers 12, 16, 20, 24, 28: the instrument converges on the output with depth. F11, the share of a token's unembedding row inside the lens's reachable subspace (singular values at or above a hundredth of the largest), rises 0.42, 0.56, 0.82, 0.87, 0.88 for the sixteen tokens measured (`WP12-BROADCAST-HEADS-2026-09-06/out/lens_spectra_concepts.json`; the random-direction proxy gives 0.45, 0.61, 0.94, 0.98, 0.99 and is optimistic by four to twelve points): the instrument is a narrow filter at shallow depth. **F12: no band layer is both wide and independent.** Shallow layers give a narrow instrument genuinely distinct from the output; deep layers give a broad instrument that is substantially the output. Layer 20 is the only one that is wide (0.82 for real tokens) while still mostly independent (0.48), and two independent calculations tonight, the injection calibration that failed at 12 and the spectra that named 20 as the writable layer, arrived at it. This changes what every lens reading in the programme means: WP5's rank statistics, WP7's decompositions and WP9's clamp treated the five pairs as equivalent instruments, and they are not. (The Head, 11:00; recorded as its own finding because it is visible in neither F10 nor F11 alone.)

**R54.** (a) Layer 20 (pair 19/20) is the programme's primary lens readout; the other four pairs are reported as a profile, never as equals and never pooled. (b) Every per-pair row carries the pair's lens-to-identity cosine (F10) and its real-token reachable share (F11) beside the value, so a reader sees which defect the row inherits. (c) A result that appears at 24 or 28 and not at 20 is read with the output-alignment caveat first; one that appears at 12 or 16 and not at 20 is read with the narrow-filter caveat first. (d) The Head's section 0b of the WP7 and WP9 designs carries this into those designs; WP5 adopts it at its next revision. Supersedes nothing: R51 (per-pair reporting with the cosine) stands and gains the second column.

**R54(e) and F11 addendum (2026-09-06, 11:20; the Head).** Reachability is a property of the direction, not only of the layer: across the sixteen measured tokens the minimum reachable share is 0.23, 0.29, 0.43, 0.52, 0.55 at layers 12 to 28 against medians of 0.42, 0.56, 0.82, 0.87, 0.88, a factor of two within layer 20. WP7(a)'s pursuit, WP9's clamp and WP5's rank statistics operate on specific directions, and a clamp on a direction at 0.43 clamps less than half of it, so a null there is a null about the part the lens can see. Rule: every per-direction row carries its own direction's reachable share at that layer, not the layer median; the layer median stays on per-pair rows. F11 addendum: the random-direction proxy was optimistic by three to twelve points and most where it mattered, flattering the deep layers, whose extra width on random directions (0.98, 0.99 against 0.94) partly paid for their contamination; on real tokens they are barely wider (0.87, 0.88 against 0.82) while the cosine climbs to 0.63 and 0.71, and the product of reachable share and one minus the cosine is 0.298, 0.398, 0.426, 0.322, 0.255 across the band, layer 20 first outright. A control biased uniformly is a nuisance; one whose bias grows with the quantity compared changes conclusions.

### R55 (2026-09-07, 18:15; the Metal live-buffer cap is respected by construction, never by byte monitoring)

Issue 90 closes its counter item on the Deputy's search of the installed MLX package and the compiled core's symbol table: every allocator entry point (`get_active_memory`, `get_cache_memory`, `get_peak_memory`, `reset_peak_memory`, `clear_cache`, `set_cache_limit`, `set_memory_limit`, `set_wired_limit`) is in bytes; there is no buffer count to read. Rule: (a) **bytes are not a proxy for the count cap.** The last two deaths peaked at 9.25 and 10.4 GB against a 17.76 GiB working set with seven gigabytes free at the abort, and `train.metal_cache_gib` never helped for the same reason; a launch record that cites byte headroom as evidence against the cap is wrong on its face. (b) **The per-row component is bounded by construction and the bound is stated before launch:** chunk invocations per row = ⌈row tokens / chunk⌉ × recurrent layers × accumulation; at 24 recurrent layers, accumulation 4 and the longest row of 2,764 tokens that is 4,224 at chunk 64, 2,112 at 128 and 1,056 at 256, i.e. 118, 236 and 472 live buffers per invocation before the 499,000 cap — the arithmetic that would have chosen chunk 256 before iteration 51 rather than after 551. Every training launch record carries this number for its configuration. (c) **The accumulation-side component is unbounded and unobservable** with the APIs we have: the deaths at 51, 141, 551 and 780 moved later as the per-row cost fell, and the fourth was in the optimizer's apply step, not the recurrence; a per-row bound predicts a death at the first long row or none, not iteration 780 at a different call site. It is written as the residual risk in every launch record, not as covered. (d) The sweep keeps bytes and time per row and adds the analytic invocation count per configuration in place of the counter it cannot measure. (The Deputy's closure, 18:10, adopted; the Chief's "monitored by bytes" clause of 17:58 withdrawn.)

### R47(b) (2026-09-07, 19:10; a sweep over sizes projects before it runs)

The Deputy's recurrence-exponent sweep breached R47 on its last row: the unrolled backward at 8,192 tokens peaked at 16.78 GiB, 0.94 of the 17.76 GiB working set, with no declared window, because the harness read the peak after measuring it and the pre-registered stop fired one row late. The peaks in hand (2.50, 4.56, 5.85, 8.69 GiB at 1,024 to 4,096) projected 16.5 GiB for the next row. Rule: any sweep over sizes projects the next size's peak from the measured rows before running it (the last ratio, or the fitted exponent), stops if the projection exceeds 0.6 of the working set, and asks for a declared window with the projection quoted; a peak read after the fact is a record of a breach, not a guard. The machine survived this one; it had crashed under load six hours earlier, and a second crash would have taken the running lens work with it.

### R1(b) (2026-09-07, 20:20; the Director's decision: `history` is qwen35-4b's default cache strategy)

`configs/models/qwen35-4b.yaml` now declares `cache.strategy: history`. The strategy landed at 618d41a and was accepted bit-identically against the ordinary runner on the real checkpoint with and without the adapter (`research/records/HISTORY-CACHE-2026-09-07/`). R1's `auto` semantics are unchanged for other models, and its `equivalence_verified` field stays null on qwen35-4b: that field licenses `snapshot`, which failed equivalence on 2026-09-07, and `auto` must never resolve to it on this model. Consequences: (a) every evaluation and rollout on qwen35-4b runs through `HistoryCache` — identical forward schedule, reuse only across whole native chunks on contexts beyond 2,048 tokens; (b) capture-carrying entry points resolve `none` explicitly (`scripts/live_lens_pilot.py` does; the lens replay reader must), because capture requires no reuse and the runner refuses the combination rather than silently disabling it; (c) the registry tests pin the new declaration and name this ruling. (Director, 20:20: "Yes, make it the default.")


### R45(b) (2026-09-08, 22:40; the pre-launch inventory is the lock module's check, never a grep)

The Deputy's shell inventory all evening was `lsof -c python | grep -i libmlx`. `lsof -c` matches the command name case-sensitively, and the venv's interpreter resolves to the framework binary `.../Python.framework/.../MacOS/Python`, so the filter returned "clear" while a holder existed; the lock module's `running_model_processes()` saw the holder and refused the launch, twice. R45's lesson was already that a model process is detected by `libmlx.dylib` being mapped and never by its name; this clause applies it to the shell. Rule: every seat's pre-launch inventory, in the launching command, is `runlock.running_model_processes()` (or the CLI that calls it), whose match is by mapped library and whose exclusion is by the caller's own lineage and process group; `lsof -c python`, `pgrep -f python` and every name-based idiom are not inventories. The announced-window practice of the same evening stands beside this: it covers the gap between a correct inventory and the lock, which is what refused Codex; the blind grep is what refused the Deputy.

**R45(c) — box state is read from disk at the moment of acting, never from a message.** A launch, a teardown, or a suite that can reach MLX reads `outputs/.box-window.json`, the lock, and `running_model_processes()` in the same command that acts, and acts on what it finds. A message about the box, however recent and from whichever seat, is a claim about the past; the file is the present. On the morning of 2026-09-08 a teardown instruction that was correct when written reached the Deputy six minutes later, after the window it named had been cleared and Codex's had opened; the Deputy read the disk first and nothing was torn down. The mirror of this rule for a seat taking a gap: announce your own window in the command that runs the suite, with `end` trapped on the shell's exit, so the gap is held by a file rather than by the belief that it is free. (Proposed by the Deputy, 2026-09-08; adopted by the Chief.) Clause, 2026-09-08 09:05: the disk is the authority on the box's **state**, not on **permission**. R45(c) stops a seat acting on a stale claim that the box is busy or free; an empty file is not a licence that was withheld. A block the Director or the Chief has held stays held while the file says free, until the word that releases it. (The Deputy's clause, adopted.)

### R56 (2026-09-08, 13:50; what makes a gate evidence, from the three ways one failed today)

**R56(a) — a gate whose comparator shares the defect it is gating measures nothing, and it reports
zero.** The Deputy's sentence, kept in their words. On 2026-09-08 the residual gate compared the
hand-run loop against `diagnostic_native_final_residual`, and the entry-transform omission the port
was fixing lived in **both**, so the gate compared a broken loop against itself and passed. The one
place the defect would have been caught is the one place it survived. When choosing a comparator,
ask what it shares with the thing it checks, and prefer a comparator that comes from somewhere
else entirely — the model's own forward, a published artifact, a second implementation — over one
that is nearby and convenient.

**R56(b) — a gate's inputs must exceed every length scale in the model it gates.** The same day's
residual gate ran on a prompt padded to 64 tokens against a model whose sliding window is 1,024.
Below a window, a windowed mask and a global mask are the same mask, so **no mask defect could ever
have appeared in that gate**, and the one that existed cost 95 per cent relative error at the first
layer the moment a real sequence was run. Length scales to check before setting a gate's inputs:
attention window, cache capacity, chunk size, any period, and the shortest sequence that exercises
each. A gate below them is not a weak gate, it is a gate that cannot fail.

**R56(c) — a gate without a negative control that bites is not evidence, and the control must be
shown to bite in the regime the gate runs in.** The port's control — every block handed the first
block's mask — changes nothing below the window and 5.1 per cent above it. Had it been demonstrated
only at 64 tokens it would have passed and proved the gate blind rather than sound. A control
demonstrated in the wrong regime is worse than none, because it is cited.

**R56(d) — an absolute difference is not an acceptance unless the quantity's own scale is quoted
beside it.** The first Gemma acceptance returned 181,895 and nearly read as a failing port; most of
it was a float32 promotion against a bfloat16 forward, and none of it was interpretable without the
residual's norm. Report relative, or report absolute **with** the norm. This applies to every
number this repository calls an agreement, an error or a discrepancy.

**R56(e) — a control must patch the path the gate actually runs through, and a refactor moves that
path without breaking the control visibly.** The port's negative control patched `masks`. Then the
Chief's hoist was applied, so `diagnostic_native_final_residual` observes once and **no longer calls
`masks` at all** — and the control went on passing, silently, because it was patching a method that
had left the path. An optimisation disarmed the instrument that proves the thing it optimised, and
nothing failed. Neither the person who ruled for the hoist nor the person who wrote the control
noticed; it surfaced only because the control's *number* was checked against the uncontrolled one
and found identical to the last bit. So: after any change to the code a control patches — and a
hoist, an inline, a rename or a caching change all count — **re-demonstrate that the control still
bites** before quoting the gate. A control is code and it rots like code, except that its rot is
silent by construction, because a disarmed control reports success.

Together these are why the port was accepted on evidence rather than on a passing number:
`research/records/GEMMA3-PORT-ACCEPTANCE-2026-09-08/`.
