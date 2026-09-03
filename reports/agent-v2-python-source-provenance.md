# `agent-v2c` is reproducible from the current Python tree; `agent-v2` and `agent-v2b` are not

Assessment date: 2026-09-03 (Australia/Melbourne)

## Scope and conclusion

This report inventories the repository-owned Python source, material third-party Python implementation, datasets, configurations, checkpoints, and selection records associated with every LoRA adapter directory matching `outputs/agent-v2*`.

Three runs exist:

| Run | Training data | Selected checkpoint | Best-adapter SHA-256 | Reproducibility from current source |
|---|---|---:|---|---|
| `outputs/agent-v2` | `data/agent_v2` | 400 | `d4e1f5f3cc4c4fede6c6622d0f7140594e2bfc903b6cc8f5796700498622b29b` | No: regenerated train, valid, and test files all differ from their manifests |
| `outputs/agent-v2b` | `data/agent_v2b` | 400 | `472f5be90964d5d6ed4f51e8a64f25956edffeab78f2aa0c556f217d28b9a80b` | No: regenerated train, valid, and test files all differ from their manifests |
| `outputs/agent-v2c` | `data/agent_v2c` | 400 | `f4f22886cfa93d7362d7db29e8d482e071692988f06f7d30efd7ca4e4f80ab74` | Yes at the dataset level: all three regenerated files match byte-for-byte |

The saved datasets, configurations, weights, logs, and evaluation outputs are intact enough to identify the artifacts precisely. The exact historic first-party Python revisions for runs A and B are not recoverable from this workspace because it has no Git metadata or source snapshot. The current tree is the C-era implementation and exactly regenerates C's dataset.

## Shared training recipe

All three runs used the same effective training recipe; only their generated data and evaluation scope changed.

| Setting | Value |
|---|---|
| Base model | `mlx-community/Qwen2.5-Coder-3B-Instruct-4bit` |
| Method | QLoRA: frozen 4-bit base, `fine_tune_type: lora` |
| Transformer layers | 36 |
| LoRA rank / scale / dropout | 16 / 32.0 / 0.0 |
| Adapted modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Optimizer / learning rate | AdamW / `3e-5` |
| Batch / accumulation | 2 / 2 |
| Iterations | 400 |
| Maximum sequence length | 2,688 |
| Gradient checkpointing | Enabled |
| Prompt masking | Enabled |
| Validation batches | 24 |
| Evaluation / save cadence | Every 100 iterations |
| Resume adapter | None recorded |
| Trainable parameters | 29.934M of 3,085.939M (0.970%) |

Each best adapter is 119,789,457 bytes and contains 504 tensors totaling 29,933,568 elements. Each of the seven target projections contributes 72 tensors: 36 layers multiplied by the `lora_a` and `lora_b` tensors. In every run, `best-adapter/adapters.safetensors`, `adapters/adapters.safetensors`, `adapters/0000400_adapters.safetensors`, and `checkpoints/step-400/adapters.safetensors` are identical copies.

The authoritative effective run configs are:

- `outputs/agent-v2/lora.yaml`
- `outputs/agent-v2b/lora.yaml`
- `outputs/agent-v2c/lora.yaml`

The higher-level inputs are `configs/agent_v2.yaml`, `configs/agent_v2b.yaml`, and `configs/agent_v2c.yaml`.

## Dataset provenance

The manifest hashes agree with the dataset files currently on disk.

| Run | Split | Rows | Composition | SHA-256 |
|---|---|---:|---|---|
| A | train | 1,700 | 1,460 expert + 240 chat | `35e7c7ad414069bbd8d263e2f3e5c26e11429df477a457635ff66917cd4be2c4` |
| A | valid | 284 | 236 expert + 48 chat | `3a6e661c2f4cd2384793ba495817bacca5c978c6daf8bb7cdc477015cc3dc4a3` |
| A | test | 494 | 434 expert + 60 chat | `c4a3291c450edc7fd74727fa972cf00c125ac9e9120165bdc4e243225c3f939c` |
| B | train | 1,915 | 1,675 expert + 240 chat; 165 recovery targets expanded to 367 rows | `c265435f53721ff2909a4be8a2b8b2300d35b86107cf8eb4a5d4cb94522449e4` |
| B | valid | 284 | 236 expert + 48 chat | `00a8512bad32e1971b72a198227860c9b8b3cff24f3db715663e7379cb903a16` |
| B | test | 494 | 434 expert + 60 chat | `43f806f9178908f35e2a057f491d871c93786746fd0e7e7560571a451bbb1c41` |
| C | train | 1,915 | 1,675 expert + 240 chat; 165 recovery targets expanded to 367 rows | `89622429a3861b60be9584d05c5d4e032aa0d0d7e31c6b61dca69448de65fb62` |
| C | valid | 284 | 236 expert + 48 chat | `4d6ce1a724fe182f04891fd07a43741f3088a1e02cc4dfad518853c4a11b8bf1` |
| C | test | 1,362 | 1,302 expert + 60 chat | `9c5672d8be9f10e41a69b44d39da9e269db0fb4fcc8e79f5d8d76b23e254d1e9` |

Run A's training variants were 60 clean, 80 transient, 25 unknown-tool, and 75 wrong-path tasks. Runs B and C used 75 clean, 17 failed-edit, 9 stale-path, 67 transient, 17 unknown-tool, and 55 wrong-path tasks. B and C repeated corrective decisions with multipliers of 1 for transient, 2 for wrong-path and unknown-tool, and 6 for stale-path and failed-edit. No run incorporated extra rollout rows.

C changed B's expert notes for the aggregate-report, batch-update, and conditional-update families and expanded only the clean test set from 60 to 180 tasks. Its train and validation row counts therefore match B, but their bytes do not.

All runs incorporated the same retained-chat dataset:

| Split | Rows | SHA-256 |
|---|---:|---|
| train | 240 | `24fb881521983946ade7a0acdbf27541e47b202d77c572970e1312d142aba915` |
| valid | 48 | `6528c4ba93cba58d1edab638d0834afc775e2bbfe7d80ff72eda904e2a45d485` |
| test | 60 | `ba8d072c8d1580f0d5f4d0fb4f3d7767d180933bbe3897f37c73b5418299d755` |

Its manifest records the same Qwen base as teacher and `outputs/agent-3b/best-adapter` as the teacher adapter.

## Complete first-party Python provenance inventory

The repository contains 43 Python files under `src` and 55 across `src`, `research`, and `tests`. The following 14 files are the behavior-bearing first-party provenance set for the three SFT adapters. Two package initializers are listed separately because they are imported but contain no training behavior.

### Dataset and training orchestration

| Python file | Role | Applies to |
|---|---|---|
| `src/local_llm_lab/pipeline/cli.py` | Loads the high-level YAML, generates data, emits the MLX-LM LoRA YAML, launches training, discovers checkpoints, evaluates them, and promotes the selected adapter | A, B, C |
| `src/local_llm_lab/pipeline/data.py` | Replays expert trajectories, applies observation windowing, mixes chat replay, repeats recovery targets, writes JSONL, and records manifests | A, B, C |
| `src/local_llm_lab/pipeline/tasks.py` | Defines the 12 task families, clean/recovery variants, seeded task contents, expert steps, expected answers, and faults | A, B, C; current revision exactly matches C data only |
| `src/local_llm_lab/pipeline/protocol.py` | Defines the v2 system prompt, textual fenced-JSON tool protocol, state-carrying notes, assistant/tool messages, parsing, and context windowing | A, B, C |
| `src/local_llm_lab/pipeline/env.py` | Implements the deterministic virtual tool environment, schema checking, injected failures, and task verdicts | A, B, C |
| `src/local_llm_lab/agent_protocol.py` | Supplies the shared action type, parser, and tool schemas used by the v2 protocol and simulator | A, B, C |
| `src/local_llm_lab/agent_tasks.py` | Supplies the safe arithmetic evaluator imported by task generation and the simulator | A, B, C |
| `src/local_llm_lab/project.py` | Resolves the project root and configures the repository-local Hugging Face cache | A, B, C |

### Upstream retained-chat generation

| Python file | Role | Applies to |
|---|---|---|
| `src/local_llm_lab/chat_replay.py` | Generates the retained-chat train/valid/test rows later mixed into every agent-v2 dataset | A, B, C |
| `src/local_llm_lab/compare_chat.py` | Defines `CHAT_SYSTEM_PROMPT`, imported by the chat-replay generator | A, B, C |
| `src/local_llm_lab/evaluate_agent.py` | Defines the default Qwen model identifier imported by the chat-replay generator | A, B, C |

### Behavioral checkpoint selection

These files do not change tensor values, but they determine which trained checkpoint is named `best-adapter`.

| Python file | Role | Applies to |
|---|---|---|
| `src/local_llm_lab/pipeline/evaluate.py` | Loads each checkpoint and evaluates the held-out task set | A, B, C |
| `src/local_llm_lab/pipeline/runner.py` | Executes model trajectories, loop detection, tool calls, and verdict collection | A, B, C |
| `src/local_llm_lab/pipeline/transcript.py` | Records trajectory summaries consumed during evaluation and selection | A, B, C |

All three `selection.json` files choose step 400 by the ordered criterion: held-out success rate, clean rate, valid-action rate, then later step.

### Passive imports and dormant CLI branches

The package initializers `src/local_llm_lab/__init__.py` and `src/local_llm_lab/pipeline/__init__.py` are part of import resolution but contain no behavior material to the adapters.

`pipeline/cli.py` imports `pipeline/branch.py`, `pipeline/rollout.py`, `pipeline/prefer.py`, and `pipeline/report.py` at module load time. Their rollout mining, DPO/preference training, and reporting stages were not invoked for these saved SFT adapters. The manifests confirm `extra_rows: 0`, the LoRA configs contain no resume adapter, and no preference-trained adapter is involved. They are therefore loaded/dormant code, not weight-producing inputs.

The probe package, research scripts, tests, post-training reports, and the remaining legacy/experimental training scripts did not produce the three saved SFT adapters.

## Current first-party source identities

These hashes identify the present files, not the unavailable historic A/B revisions.

| File | Current SHA-256 |
|---|---|
| `src/local_llm_lab/__init__.py` | `dcbdef2e02c3757b8f9fc3ba34a5595f7f33d295cf1415ec75d07e516662f45b` |
| `src/local_llm_lab/agent_protocol.py` | `2adfd82e40f22528bef823fe6d257042f83327b0c735784e9432427fb9bd59cf` |
| `src/local_llm_lab/agent_tasks.py` | `7c413b7cb457369b8d4be82a83a2791e360a12bae6454894fa1096a96354e9e8` |
| `src/local_llm_lab/chat_replay.py` | `20d94b55a46974e60c2ea13933c0458fdb9ca15492bce3660134325b93450d23` |
| `src/local_llm_lab/compare_chat.py` | `b0fb19a54d3a2a84c05f92d1dffa884cd15a7abad91b52cd516565e47a055352` |
| `src/local_llm_lab/evaluate_agent.py` | `a32196f4544da72465a5bf9e4cba4b3c26d8551fb6fc7529f5458b808b9b5d93` |
| `src/local_llm_lab/project.py` | `d5302a4a1b68906e1e8b2f814cc016e3be144ecea43ae7370ddc392fdeaf14fe` |
| `src/local_llm_lab/pipeline/__init__.py` | `149384f3721df199454edd7f1bf4542ff00519781f565fc5ababead2c982fcdf` |
| `src/local_llm_lab/pipeline/cli.py` | `c3210fee1e77fb91a70148ac7660e5fa7c2cf44f06dbd90dde717a20d45cb27b` |
| `src/local_llm_lab/pipeline/data.py` | `414672f2c45f7d1243fb1dc2bfe604bcc4e7322c353364848281ca2e26cf61b5` |
| `src/local_llm_lab/pipeline/env.py` | `64811cea22ae5239a1df9fe4292b7471fcf9f97af7ed2a85c143d9a47a7d4235` |
| `src/local_llm_lab/pipeline/evaluate.py` | `a8552a8efdff8a8d8f67ef41a1a8fb9d71b4b7a71a054e705775fc4a21775960` |
| `src/local_llm_lab/pipeline/protocol.py` | `30fdb43bf021655ceab64e12e147a2c5ecf15978bf8753967d56474f40cf6782` |
| `src/local_llm_lab/pipeline/runner.py` | `ee463d6afd908e89bd24cf3722882a125d46920d9eb22164a360453b0d628c18` |
| `src/local_llm_lab/pipeline/tasks.py` | `45aebe0a1c28c905bc8ccc73c18df4b35afbe4c31bcbea29c3de1a95c1458789` |
| `src/local_llm_lab/pipeline/transcript.py` | `9ea18849bfdfd04f5b7deeb1de575feb24313671d18848d8540f5412a1db3c1e` |

## Material third-party Python implementation

Training was delegated by `pipeline/cli.py` to MLX-LM through `mlx_lm.lora --config ...` or `python -m mlx_lm lora --config ...`. The material installed MLX-LM Python path is:

- `.venv/lib/python3.13/site-packages/mlx_lm/lora.py`
- `.venv/lib/python3.13/site-packages/mlx_lm/tuner/datasets.py`
- `.venv/lib/python3.13/site-packages/mlx_lm/tuner/trainer.py`
- `.venv/lib/python3.13/site-packages/mlx_lm/tuner/utils.py`
- `.venv/lib/python3.13/site-packages/mlx_lm/tuner/lora.py`
- `.venv/lib/python3.13/site-packages/mlx_lm/tuner/callbacks.py`
- `.venv/lib/python3.13/site-packages/mlx_lm/utils.py`
- `.venv/lib/python3.13/site-packages/mlx_lm/tokenizer_utils.py`
- `.venv/lib/python3.13/site-packages/mlx_lm/models/qwen2.py`
- `.venv/lib/python3.13/site-packages/mlx_lm/models/base.py`
- `.venv/lib/python3.13/site-packages/mlx_lm/models/rope_utils.py`
- `.venv/lib/python3.13/site-packages/mlx_lm/models/activations.py`

`mlx_lm/tuner/dora.py` and `mlx_lm/tuner/switch_layers.py` exist in the installed package but are dormant for this LoRA/Qwen2 configuration. Python standard-library and MLX framework internals are transitive runtime dependencies rather than repository-owned experiment source.

The installed environment records CPython 3.13.7 on arm64 macOS 26.6.1, `mlx-lm 0.31.3`, `mlx 0.32.2`, `transformers 5.16.1`, and `numpy 2.5.2`. `uv.lock` pins the MLX-LM wheel with SHA-256 `758cfddf1180053b7613db76fad3d246a331a2a905808e1164a275621fc983b8`.

The local Hugging Face cache contains one snapshot for the base model:

`3dd939c621c08e5753d5b89f35a2642cd83b98ca`

The adapter configs record the model name but not this revision, so the cache is strong local evidence rather than a self-contained provenance guarantee.

## Checkpoint identities

Each run has checkpoints at steps 100, 200, 300, and 400. The same checkpoint bytes appear in both `adapters/0000xxx_adapters.safetensors` and `checkpoints/step-xxx/adapters.safetensors`.

| Run | Step | SHA-256 |
|---|---:|---|
| A | 100 | `7679360a767a725447c52299f450ec18a5a40135a14a2a3eadc771cd753481b8` |
| A | 200 | `50c1fa8babb2fc2435a820b48d74e4d4e92a2e9619832a8b00561e85f1da897d` |
| A | 300 | `c742d6e660c11cb3e911d97819cfc53e94dcbbcdf6edf39600486ec4ff08bc28` |
| A | 400 | `d4e1f5f3cc4c4fede6c6622d0f7140594e2bfc903b6cc8f5796700498622b29b` |
| B | 100 | `daee089ba670740263b737b0b398fc73a9f78177aa32ef7bbf755da0d1565ad8` |
| B | 200 | `8afc3326efcacc442a9de4e1565cc6baf7c693f22bb8d7cd77f3c70ed1265dab` |
| B | 300 | `070a93c1776dae62f8124e3a6719c57f0bce9366f616d9269d49f15277264034` |
| B | 400 | `472f5be90964d5d6ed4f51e8a64f25956edffeab78f2aa0c556f217d28b9a80b` |
| C | 100 | `8a3aa51989f92583766612bb432cd923f9679e82ca396830decb143fcfe3a720` |
| C | 200 | `97b46d845e2652ea557916fb3bd4cc1eb7f540a6d775fc37827d910bc6a8adef` |
| C | 300 | `81940b9a932ea5a7d3a9eee392fef25412090561c28613e6fcd8d0178c56da40` |
| C | 400 | `f4f22886cfa93d7362d7db29e8d482e071692988f06f7d30efd7ca4e4f80ab74` |

## Training and selection evidence

| Run | Trained tokens at step 400 | Train loss | Validation loss | Selected validation screen | Native best-adapter test |
|---|---:|---:|---:|---:|---:|
| A | 59,524 | 0.111 | 0.158 | 16/18 (0.8889) | 47/60 (0.7833) |
| B | 59,217 | 0.014 | 0.127 | 17/18 (0.9444) | 47/60 (0.7833) |
| C | 58,246 | 0.014 | 0.116 | 16/18 (0.8889) | 118/180 (0.6556) |

Run A also contains `train-oom.log`. That failed attempt stopped after initial validation and before any saved checkpoint. The successful `train.log` started a fresh run, records no resume adapter, reached step 400, and saved the weights now inventoried.

## Why A and B cannot be tied to exact Python bytes

A deterministic regeneration was performed in a temporary directory using each saved high-level config and the current Python tree. Generated train, valid, and test SHA-256 values were compared with the saved manifests:

- A: all three splits differed.
- B: all three splits differed.
- C: all three splits matched byte-for-byte.

The file chronology is consistent with source evolution between runs. For example, current `pipeline/data.py` and `pipeline/env.py` have 2026-09-02 evening modification times after A's data was created; current `pipeline/tasks.py` was modified at 2026-09-03 10:28:38 +1000, immediately before C's data. Some source files have later modification times but still regenerate C exactly, showing that timestamps alone are not proof of behavioral change.

Because the workspace is not a Git checkout and no source tarball or per-run source hashes were saved, the precise A/B revisions cannot be reconstructed locally. Their generated data remains exact and hash-verified, so the model inputs are preserved even though the generator source history is not.

## Verification boundaries

This assessment used file discovery, source inspection, JSON/YAML inspection, SHA-256 checks, SafeTensors header inspection, log reconciliation, and deterministic dataset regeneration in a deleted temporary directory. It did not modify source, data, adapter, or probe output files.

No runtime test result is used as evidence in this report. MLX/Metal execution and the active probe were excluded from the final verification after the test crash. The report itself was checked using static text and path validation only.

## Preservation recommendation

For future adapters, save the following next to each run before training begins:

- a Git commit plus dirty-tree patch, or a source tarball and per-file SHA-256 manifest;
- the resolved base-model snapshot revision;
- the complete `uv.lock` hash and relevant installed package versions;
- the effective MLX-LM YAML;
- generated split hashes and row counts;
- checkpoint hashes and the exact selection criterion.

That would make later provenance exact at the source, data, runtime, and artifact levels rather than relying on chronology and local cache evidence.
