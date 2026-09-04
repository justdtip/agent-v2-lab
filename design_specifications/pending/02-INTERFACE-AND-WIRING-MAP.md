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
