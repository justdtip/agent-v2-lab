# SPEC-001: Model-agnostic backbone for training, evaluation, and probes

> Read first: `01-IMPLEMENTER-BRIEFING.md` (standing rules, traps, hand-off) and `02-INTERFACE-AND-WIRING-MAP.md` (exact shared signatures, file ownership, implementation order, integration checks). Signatures in the wiring map override any looser wording here.

Status: pending. Author: Claude (Chief AI Research Scientist). Date: 2026-09-03.
Depends on: nothing. Blocks: SPEC-002, SPEC-003, SPEC-004.
GPU: none, except the `preflight` stage in §10, which loads a model and is gated by the
deployment prohibition.

## 0. Goal and non-goals

Make every stage that touches a model (data rendering, training, selection, evaluation,
rollouts, cache reuse, activation capture, J-lens, adapter analysis) run unchanged on:

| Registry name | HF id | Layers | Hidden | Attention | Tied unembed | Vocab | Thinking |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen25-coder-3b` | `mlx-community/Qwen2.5-Coder-3B-Instruct-4bit` | 36 | 2048 | dense GQA | yes | 151,936 | unsupported |
| `qwen35-4b` | `mlx-community/Qwen3.5-4B-MLX-4bit` | 32 | 2560 | 24 Gated DeltaNet + 8 full (1 in 4) | yes | 248,320 | on by default |
| `qwen35-9b` | `mlx-community/Qwen3.5-9B-MLX-4bit` | 32 | 4096 | 24 Gated DeltaNet + 8 full | no | 248,320 | on by default |

and on any further architecture that the installed `mlx-lm` (0.31.3) can load. The installed
package already contains `models/qwen3_5.py`; the text model lives at
`model.language_model.model`, its blocks expose `is_linear`, linear blocks take an SSM mask and
an `ArraysCache(size=2)`, full-attention blocks take a causal mask and a `KVCache`, and the
vision tower is stripped by `sanitize`. Nothing in the repository knows any of that today.

Non-goals: vision input; MoE variants (must load, need not be tuned); changing the tool protocol
(the fenced-JSON call format is already model-agnostic and stays).

## 1. Model registry

New: `configs/models/<name>.yaml` and `src/local_llm_lab/models.py`.

```yaml
# configs/models/qwen35-4b.yaml
name: qwen35-4b
hf_id: mlx-community/Qwen3.5-4B-MLX-4bit
family: qwen3_5                # informational; behaviour is discovered, not keyed on this
chat:
  thinking: off                # unsupported | off | inference | trained   (§4)
  template_kwargs: {enable_thinking: false}
  end_of_turn: "<|im_end|>"
  extra_stop_tokens: ["<|endoftext|>"]
lora:
  keys: auto                   # auto | attention+mlp | all-linear | explicit list   (§6)
  rank: 16
  scale: 32.0
  dropout: 0.0
train:
  max_seq_length: 2688
  batch_size: 2
  grad_accumulation_steps: 2
  learning_rate: 3.0e-5
  grad_checkpoint: true
cache:
  strategy: auto               # auto | trim | snapshot | none   (§5; ruling R1 in the wiring map)
  equivalence_verified: null   # {date, sha256} written only after research/cache_equivalence.py passes for this model
probes:
  layer_fractions: [0.167, 0.333, 0.5, 0.667, 0.833, 1.0]   # §8
memory:
  budget_gib: 22
```

`ModelSpec` (frozen dataclass) is built by `load_model_spec(name_or_hf_id)`; a raw HF id yields a
spec with defaults and `name = hf_id`, so existing `configs/agent_v2*.yaml` keep working. After
the model is loaded, `ModelSpec.resolve(model, tokenizer)` fills the derived fields
(`num_layers`, `hidden_size`, `layer_types`, `tie_word_embeddings`, `vocab_size`, resolved LoRA
keys, resolved probe layers, cache strategy, HF snapshot revision) and returns a `ResolvedSpec`
whose `as_dict()` is written into every manifest, evaluation summary, probe JSON, and
`provenance.json` (§9). Nothing downstream may read `num_layers`, layer lists, module names, or
stop tokens from anywhere but the resolved spec.

Every pipeline YAML gains `model: <registry name>`; `pipeline/cli.py` resolves it once and passes
the spec to every stage.

## 2. Architecture view

New: `src/local_llm_lab/arch.py`, class `ArchitectureView`, constructed from a loaded model by
structural inspection, never by `model_type` string:

| Member | Contract |
| --- | --- |
| `text_module` | the module owning `embed_tokens`, `layers`, `norm`; found by walking `model`, `model.model`, `model.language_model.model` |
| `blocks` | `list` of decoder blocks in order |
| `num_layers`, `hidden_size` | from `blocks` and the embedding |
| `layer_kind(i)` | `"attention"` or `"linear_attention"`; from `getattr(block, "is_linear", False)` with an override table for known families |
| `embed(ids)` | float32 embedding output; layer 0 of the residual index |
| `masks(h, cache)` | returns the per-kind masks the model itself would build (`create_attention_mask`, `create_ssm_mask`) so `run_block` uses the same mask as `text_module.__call__` |
| `run_block(i, h, masks, cache_i)` | one block, correct mask by kind, returns float32 |
| `final_norm(h)` | `text_module.norm` |
| `unembed(h)` | `embed_tokens.as_linear` when tied, else `lm_head`; used by J-lens and logit lens |
| `make_cache()` | `model.make_cache()` when present, else `mlx_lm.models.cache.make_prompt_cache(model)` |
| `cache_trimmable` | `all(c.is_trimmable() for c in make_cache())` |
| `lora_targets(policy)` | §6 |

Residual index convention (unchanged from `capture.py`): layer `L` is the residual after block
`L-1`; layer `num_layers` is the pre-final-norm residual. `layer_fractions` map to indices with
`max(1, round(f * num_layers))`.

Contract tests (pure numpy fakes, no checkpoint): a dense Qwen2-like fake and a hybrid fake
whose blocks carry `is_linear` and whose linear blocks require a different mask. For each:
`view.final_norm(view.residual(ids, num_layers))` equals `model(ids)` pre-unembedding to 1e-5,
and every intermediate residual equals the one obtained by running the model's own loop with a
recording wrapper. These replace and generalise `tests/test_probes.py::test_capture_decomposition_matches_the_models_own_forward_pass`.

Rewrite on top of the view, keeping public signatures where possible:

- `probes/capture.py`: `capture_residuals`, `response_mean_activations`, `InjectionHook`,
  `lora_block_mask`. `InjectionHook` must wrap `run_block` rather than `block.__call__` and read
  the absolute position from whichever cache kind is present (`KVCache.offset`; for
  `ArraysCache` track the offset in the view since the cache does not).
- `pipeline/jlens.py`: `residual_at`, `jacobian_vector_product`, `jlens_map`, `readout`,
  `logit_lens`. The tail function `fn(h_L)` must apply per-kind masks for every block in
  `[L, num_layers)`. **Risk**: `mx.jvp` through `gated_delta_update` may be unsupported or
  numerically unstable. `preflight` (§10) tests it on a 64-token prompt; on failure the view
  falls back to a central finite-difference directional derivative in float32
  (`(f(h+eps v) - f(h-eps v)) / 2eps`, `eps = 1e-2 * ||h|| / ||v||`) and every J-lens record
  carries `jvp_method: "forward" | "finite_difference"`. The equivalence test
  `test_jlens_averaging_and_linearity` must pass under both methods on the fakes.
- Efficiency: `residual_at` for all requested layers is computed in one head pass and cached per
  `(context, layers)` inside `jlens_map`; `adapter_delta.readout_update_directions` calls
  `jlens_map` once per direction and derives `-v` by negation (JVP is linear in the tangent).

## 3. Prompt rendering: one renderer, byte-identical everywhere

`protocol.build_prompt(tokenizer, messages, *, spec, keep_last)` becomes the only place a chat
template is applied. It passes `spec.chat.template_kwargs` and asserts the rendered string ends
with the expected generation suffix for the spec's thinking mode (`"<|im_start|>assistant\n"` for
`unsupported`/`inference`/`trained`, `"<|im_start|>assistant\n<think>\n\n</think>\n\n"` for
`off` on Qwen3.5). A mismatch raises: a wrong suffix would silently shift every probe position and
every training target.

Training rows are no longer left to `mlx-lm`'s `ChatDataset`, which applies the template with its
own defaults (no `enable_thinking`, and template-specific handling of the final assistant turn).
Instead `pipeline/data.py` writes rows as

```json
{"prompt": "<rendered context ending in the generation suffix>",
 "completion": "<note>\n```json\n{...}\n```<|im_end|>\n",
 "metadata": {...}}
```

and training consumes them through an in-process dataset (§7) that tokenises `prompt` and
`completion` separately and masks the prompt by token offset. Train and inference contexts are
therefore identical by construction, for any template. A regression test renders one row through
`build_prompt` and through the dataset and asserts token identity; a second test checks that the
first token of `completion` is not merged with the last token of `prompt` by the tokenizer (the
`prefix_mismatch` repair in `response_mean_activations` documents that this can happen).

## 4. Thinking as a mode

`spec.chat.thinking`:

| Mode | Training render | Inference render | Parsing | Budget |
| --- | --- | --- | --- | --- |
| `unsupported` | plain | plain | unchanged | n/a |
| `off` | `enable_thinking=False` | same | unchanged | n/a |
| `inference` | `enable_thinking=False` (rows never contain reasoning) | `enable_thinking=True` | `strip_thinking(raw)` removes the first `<think>…</think>` block before `parse_turn`; `turn_is_complete` ignores fences inside an open think block | `max_think_tokens` (default 512); when reached, the runner appends `\n</think>\n\n` and continues generating the note and call |
| `trained` | rows contain a synthesised reasoning block (SPEC-003 §6, later) | `enable_thinking=True` | as above | as above |

Every `Trajectory.steps[i]` gains `thinking: str | None` and `think_tokens: int`; `summarize`
adds `think_tokens_per_task`. Transcripts print thinking collapsed to its first and last line.
The note remains required in every mode; a turn whose post-think text has no note is a parse
error exactly as today.

## 5. Cross-turn cache strategy

`runner.TurnCache` becomes an interface with two implementations chosen by
`spec.cache.strategy` (`auto`: `trim` if trimmable, else `snapshot` only when `equivalence_verified` is set, else `none`; wiring map ruling R1):

- `TrimCache`: the existing longest-common-prefix trim (dense KV only).
- `SnapshotCache`: after the first turn's prefill, snapshot the cache state at the token boundary
  of the immutable prefix (system message plus task prompt; the boundary is the token length of
  `build_prompt` over those two messages alone, computed once per task). Each later turn restores
  the snapshot (deep copy of `c.state` per layer cache; `ArraysCache` state is a list of arrays,
  `KVCache` state is sliced to the boundary) and encodes only the remainder. Expected saving is
  the 23-26% of prompt tokens measured for the fixed prefix, versus 56-58% for trimming on the
  dense model; that is the honest cost of a recurrent cache.
- `none`: no reuse; the reference for equivalence checks.

`research/cache_equivalence.py` takes `--model` and must report bit-identical raw text, actions,
and verdicts for the chosen strategy against `none` before that strategy may be set in a
registry file; the resolved spec records `cache_equivalence_verified: <date, sha>` or the runner
falls back to `none` with a warning.

## 6. LoRA target discovery

`view.lora_targets(policy)` returns the explicit key list written into the mlx-lm YAML as
`lora_parameters.keys`; never rely on mlx-lm's default (which adapts every linear layer it finds,
embeddings included). Policies:

| Policy | Modules |
| --- | --- |
| `attention+mlp` | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` where present (reproduces the 3B recipe) |
| `all-linear` | the above plus, in linear-attention blocks, every `nn.Linear` the block owns: installed mlx-lm 0.31.3 `qwen3_5.py` uses `in_proj_qkv in_proj_z in_proj_b in_proj_a out_proj`; the `qwen3_next.py` layout uses `in_proj_qkvz in_proj_ba out_proj`; discover by type, accept both name sets (correction 2026-09-03 evening, confirmed against the installed source) |
| `auto` | `attention+mlp` for dense models, `all-linear` for hybrids |
| explicit | validated against the module tree; unknown names abort |

Discovery walks `block.named_modules()` for `nn.Linear`/`nn.QuantizedLinear` and classifies by
the final path component. The resolved key list, the trainable-parameter count, and the count per
module type are written to `provenance.json` and echoed by `agent-pipeline train`. `num_layers`
in the YAML is always `view.num_layers`.

## 7. Training entry

`pipeline/cli.py::stage_train` moves from `subprocess(mlx_lm.lora --config)` to an in-process
call of `mlx_lm.lora.train_model(args, model, train_set, valid_set, training_callback)` (installed 0.31.3 signature; ruling R14) with
`RenderedRowsDataset` objects (tokenised `prompt`/`completion`, prompt masked by offset, sorted by
length for batching as mlx-lm does). The effective mlx-lm args namespace is still dumped to
`lora.yaml` for the record. If `train_model`'s signature differs from the pinned version, abort
with a message rather than degrade; the pin is checked at import.

Memory rules encoded in `ModelSpec.memory`: `preflight` estimates peak memory from the 4-bit
weight size plus `batch × max_seq × hidden × layers` activations under gradient checkpointing and
refuses to start training above `budget_gib`; the 9B model is expected to need
`batch_size: 1, grad_accumulation_steps: 4`.

## 8. Probes on the view

- `probes/state_probe.py`, `probes/assistant_axis.py`, `probes/adapter_delta.py`: `--layers`
  accepts fractions or indices; defaults come from `spec.probes.layer_fractions`; outputs record
  both. All three take `--model <registry name>` and resolve adapters through `policies.py`,
  which becomes a per-model table (`configs/models/<name>.yaml: policies: {A: ..., B: ...}`)
  instead of a hard-coded triple.
- `adapter_delta`: parse any `<path>.lora_a` / `<path>.lora_b` pair; layer index from the first
  integer path segment; module type from the last segment; group tables by the module types
  present. Cross-run comparison asserts identical base `hf_id` and snapshot revision. Base weights
  dequantised through the view.
- `capture.strip_state_fields` is unchanged (it is text-level).

## 9. Provenance

New `src/local_llm_lab/provenance.py::write_provenance(run_dir, spec, extra)` called by `data`,
`train`, `select`, `eval`, `rollout`, and every probe CLI. Contents: per-file SHA-256 of
`src/**/*.py` and `configs/**`, `uv.lock` hash, installed versions of `mlx`, `mlx-lm`,
`transformers`, `numpy`, the resolved `ModelSpec`, the HF snapshot revision, dataset split hashes
and row counts, the effective training config, the resolved LoRA keys, the selection criterion
string, and the command line. Deterministic ordering; a test asserts two calls on an unchanged
tree produce identical JSON.

## 10. Preflight stage (gated)

`agent-pipeline preflight --model <name>` (the only stage in this spec that loads a checkpoint):
loads lazily, prints the view summary (layers by kind, hidden size, tied unembed, vocab, cache
kinds), runs the residual equivalence contract on a 64-token prompt, tests `mx.jvp` through the
tail from the middle layer and records the method, renders one sample prompt per thinking mode
and prints token counts, discovers LoRA keys and counts trainable parameters, estimates memory,
and writes `outputs/preflight/<name>.json`. Required to pass before any `train`, `eval`, or
probe on a model not yet in `outputs/preflight/`.

## 11. Acceptance

- `uv run pytest` passes with the fake dense and fake hybrid models parametrised through the
  capture, J-lens, injection, block-mask, cache-snapshot, rendering, thinking-parse, LoRA-key,
  adapter-delta-parsing, and provenance tests.
- `agent-pipeline data` for `qwen25-coder-3b` with `thinking: unsupported` reproduces the pinned
  hashes of the current `GENERATOR_VERSION` (ruling R5), and the rendered `prompt`/`completion`
  rows tokenise identically to the `messages` rendering of the same rows (migration test).
- `preflight` output exists for `qwen25-coder-3b` and `qwen35-4b` once runs are permitted.
- Provenance records the git commit hash and a dirty-tree patch: the workspace is now a git checkout (branch `codex/agent-v2-specs`, baseline `bda5ff5`), so `write_provenance` must include `git.commit`, `git.branch`, and `git.dirty_patch_sha256` alongside the per-file hashes.

## 12. Review checklist for the implementer

Numerics float32 for activations and tangents; weights stay quantised. No module reads
`model.model.layers` directly. No hard-coded `36`, `2048`, `35`, `<|im_end|>`, or module-name
list outside `configs/models/` and `arch.py`. Every new CLI flag is recorded in the output JSON.
