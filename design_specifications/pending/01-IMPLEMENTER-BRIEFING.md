# Implementer briefing: read this before touching any file

Audience: every AI agent implementing SPEC-001 to SPEC-004. Keep this document in context for
the whole task. It states the standing rules, the facts about this repository that are easy to
get wrong, the traps already found in the installed libraries, and the hand-off procedure.
The interface and wiring map (`02-INTERFACE-AND-WIRING-MAP.md`) is the companion; read both.

## 1. Standing rules (non-negotiable)

1. **No model runs.** Deployment of agents is forbidden in this project until the Research
   Director lifts the ban. For you this means: do not load a checkpoint, do not call
   `mlx_lm.load`, do not run `agent-pipeline train|select|eval|rollout|branch|prefer|preflight`,
   any `agent-v2-*` CLI that loads a model, `research/cache_equivalence.py`, or
   `research/jspace_sweep.py`. Tests run on numpy/MLX fakes only. If a spec section is marked
   "gated", implement it, test it on fakes, and stop.
2. **Do not modify or delete anything under `outputs/`, `data/`, or `reports/`** except the new
   files a spec tells you to write (SPEC-002 §2.4 writes `reports/note-integrity-B-vs-C.md`;
   SPEC-004 §1 writes a `reanalysis` pair next to the saved npz; SPEC-004 §4 writes
   `outputs/probes/axis-corrected/CLOSED.md`). Regenerated datasets go to new directories
   (`data/agent_v2d`, never over `data/agent_v2c`). The three adapters are irreplaceable: runs A
   and B cannot be regenerated from the current source.
3. **Read-only on the research documents.** `research/*.md` are the Research Director's records.
   Do not edit them; propose text in your implementation report instead.
4. **Every result carries its control.** A probe number without its shuffled, position, or
   surface baseline is not reported. A rate without a Wilson interval is not reported.
5. **Float32 for stored activations, tangents, and probe features; native dtype for block
   execution during capture (R18).** Weights stay 4-bit. Float16 overflowed the J-lens tangent
   on the real checkpoint; bfloat16 has float32 range, and running blocks in float32 on a
   bfloat16 model diverges from deployment by up to 5% on the hybrid.
6. **The model's own causal mask, always.** Any code that runs decoder blocks by hand must build
   masks the way the model does (`create_attention_mask` for attention blocks,
   `create_ssm_mask` for linear-attention blocks). Passing `None` gives bidirectional attention
   and silently invalidates every downstream number.
7. **No hard-coded model constants** outside `configs/models/` and `src/local_llm_lab/arch.py`:
   no `36`, `2048`, `35`, `<|im_end|>`, `model.model.layers`, or a list of projection names.
   Grep for them before you hand off.
8. **Determinism.** Every random choice takes a seed from the CLI or config and records it.
   `make_tasks` is seeded by the split name string; the shuffle in `write_dataset` is seeded by
   `f"{seed}:{split}"`. Do not change these derivations; generated data is versioned by
   `tasks.GENERATOR_VERSION` (wiring map ruling R5); bump it and re-pin the reference hashes whenever a change alters any row. Run C's data reproduces only at generator version 1 (commit 97d197c).
9. **Atomic writes** for anything large (temp file in the same directory, fsync, `os.replace`),
   as `state_probe.save_dataset` already does.
10. **Ask by writing, not by waiting.** If a spec is ambiguous, pick the reading that keeps
    existing behaviour for the 3B model unchanged, implement it, and flag the choice in your
    report under "decisions taken".

## 2. What this project is (keep in mind while coding)

- A 3B model (`mlx-community/Qwen2.5-Coder-3B-Instruct-4bit`) is trained with QLoRA to operate a
  six-tool virtual file workspace. The next base is `Qwen/Qwen3.5-4B` (via
  `mlx-community/Qwen3.5-4B-MLX-4bit`), possibly 9B. Everything you write must work for all of
  them; SPEC-001 makes the model a parameter.
- Every assistant turn is a short **state-carrying note** followed by exactly one tool call
  rendered as a fenced JSON block. The harness hides all but the last two tool observations
  (`keep_last = 2`), so the note is the only cross-turn memory. This is verified mechanically:
  without the note the model cannot recover a hidden filename (research/jspace_probe.md).
- Design rule learned from run C: **a note may enumerate what remains and count against a
  stated total; it must never assert completion.** Completion is expressed only as
  `pending: none`.
- Checkpoints are selected by held-out behaviour, not loss. Run C's screen scored 16/18 while
  two families were at 0/15; SPEC-002 fixes the screen.
- Thinking (Qwen3.5) is intra-turn compute; the chat template strips prior reasoning, so it
  cannot replace the note. The note is mandatory in every thinking mode.

## 3. Repository facts that are easy to get wrong

> Anchors verified 2026-09-03 22:55 by the Deputy against the working tree at `6325d3a`. Line
> numbers drift on every insertion: nine of these anchors went stale when SPEC-004 §1 added 874
> lines to `state_probe.py`. Check the named symbol, not the line, and treat a mismatch as drift
> rather than as a missing fact. Two files are being edited right now by the SPEC-004 remediation
> lane, `probes/state_probe.py` and `tests/test_probes.py`, so their anchors carry the symbol
> name and will move again when that work lands.

| Fact | Where | Why it matters |
| --- | --- | --- |
| The pipeline is `src/local_llm_lab/pipeline/`; legacy modules (`agent_*.py`, `complex_*`, `depth_expansion.py`, `train_*.py`, `chat*.py`) are kept for reference and are not on the v2 path except `agent_protocol.py` (action type, parser, tool schemas) and `agent_tasks.py` (`_calculate`) | provenance report | do not refactor legacy modules |
| `protocol.build_prompt` deliberately does not pass `tools=` to the chat template | protocol.py:219-230 | the template's tool section would instruct `<tool_call>` tags, which are untrained on the 3B checkpoint |
| Residual layer index `L` = after block `L-1`; `L = num_layers` = pre-final-norm | capture.py:11-13 | probe layers and J-lens depths depend on it |
| `state_probe` rows come from **expert replay**, never from policy rollouts | state_probe.py:456-469 (`build_probe_dataset`), data.py:20-71 | base and adapters see identical prompts |
| The mixed P2 dataset's `train-` rows (737) are byte-identical to SFT rows; `p2mix` rows are different tasks (R7) | state_probe.py:102-105 (`MIX_PLAN`), data.py:114-118 | SPEC-004 §2 uses new split names |
| `--strip` rewrites assistant messages only | state_probe.py:309-315 (`_strip_messages`) | SPEC-004 §2 adds observation stubbing |
| `capture.InjectionHook` and `capture.lora_block_mask` exist, are tested, and are imported by nothing | capture.py:145-285 | SPEC-004 §3 and §5 wire them |
| `adapter_delta --ablate` is a `parser.error` stub | adapter_delta.py:614-623 | SPEC-004 §3 |
| `assistant_axis` does measurement only; steering is absent by design | assistant_axis.py:8-10 | do not add steering until an axis passes |
| `TurnCache` rebuilds from scratch whenever the cache cannot be trimmed | runner.py:96-112 | correct but zero reuse on hybrid models; SPEC-001 §5 |
| `Trajectory(**record)` is used to reload saved evaluations | assistant_axis.py:958 | every new `Trajectory` field needs a default or old evals stop loading |
| `transcripts.jsonl` is opened in append mode | transcript.py:153-154 | reruns duplicate records; SPEC-002 §5 |
| `checkpoint_dirs` re-copies config and replaces missing or stale weights | cli.py:174-191 | SPEC-002 §5 hardens materialised checkpoints |
| Selection tie-break is "later step wins" | cli.py:247-250 | SPEC-002 §1 changes it |
| Tests reference a saved artifact: `outputs/probes/state/state-base.npz` | tests/test_probes.py:2006-2024 (`_BASE_CAPTURE`) | keep that test passing; do not move the file |
| `tasks.applicable_variants` is cached per `(family, level)` from a same-level probe | tasks.py:54-83 | SPEC-002 §5 |
| Test split is all-clean; recovery variants exist only in `train` and fresh splits | tasks.py:92-103 | `prev_error` has no positives at difficulty 2 |
| `data/chat_replay` rows are mixed into every training set (240/48/60) | data.py:122-129 | keep them in run D |

## 4. Installed-library traps (mlx-lm 0.31.3, mlx 0.32.2, transformers 5.16.1, mlx-tune 0.6.0)

1. `mlx_lm/models/qwen3_5.py`: `Model.language_model.model` owns `embed_tokens/layers/norm`;
   `Model.layers` is a property that forwards to it. Blocks have `is_linear`;
   `TextModel.__call__` builds `fa_mask = create_attention_mask(h, cache[3])` and
   `ssm_mask = create_ssm_mask(h, cache[0])` and passes one or the other per block.
   `make_cache()` returns `ArraysCache(size=2)` for linear blocks and `KVCache` otherwise.
   Tied embeddings on 4B (`embed_tokens.as_linear`), untied on 9B (`lm_head`).
2. `ArraysCache.is_trimmable()` is false; `can_trim_prompt_cache` therefore returns false for
   the whole cache list on hybrids. Snapshot/restore of `c.state` is the only reuse path.
3. `mlx_lm.tuner.utils.linear_to_lora_layers`: when `lora_parameters.keys` is absent it adapts
   **every** `nn.Linear`/`QuantizedLinear`/`Embedding` it finds in the blocks. Always pass
   explicit keys.
4. `mlx_lm.tuner.datasets.ChatDataset` applies the chat template with no kwargs; on Qwen3.5 the
   rendering of a completed assistant turn and of the generation prompt are template-dependent.
   Do not rely on it; SPEC-001 §3 and §7 render rows in-repo and mask by token offset.
5. Qwen3.5 chat template: thinking on by default; `enable_thinking=False` appends
   `<think>\n\n</think>\n\n` to the generation prompt; prior assistant `reasoning_content` is
   stripped; tools render as `<tool_call><function=…>` XML, which the v2 protocol does not use.
   The end-of-turn token is still `<|im_end|>`; the eos id in `config.json` is 248044.
6. Tokenizer boundary merges: the last token of a rendered prompt and the first token of the
   completion can merge when tokenised jointly. `capture.response_mean_activations` already
   repairs this and counts `prefix_mismatch`; SPEC-001 §3 requires a test for the training rows.
7. `mx.jvp` through `gated_delta_update` is unverified. SPEC-001 §2 requires the preflight test
   and a finite-difference fallback recorded in every J-lens record.
8. mlx-tune 0.6.0 (DPO path): `load_in_4bit` is unused, `load_adapter` does not freeze the base,
   `save_pretrained` copies the input adapter, the synthesised `adapter_config.json` has the
   wrong LoRA scale, and the native loop ignores `gradient_accumulation_steps`. `prefer.py`
   works around all five; do not remove those guards.
9. `pgrep`-based GPU guard (`probes/guard.py`) is fail-open on subprocess errors; keep it and
   keep `--allow-busy-gpu`.
10. MLX unified memory on the 24 GB M4 Pro: training at batch 2 × 2,688 tokens needs gradient
    checkpointing (peak about 9.4 GB on the 3B model); the 9B model will need batch 1.

## 5. Measured costs to plan against (3B model, this machine)

| Operation | Cost |
| --- | --- |
| One evaluation task, greedy, cache on | median 5 s, p95 60 s, mean about 15 s |
| 180-task evaluation | 52 minutes |
| 18-task screen | 3.5 to 9.5 minutes |
| 400-iteration QLoRA run | about 70 minutes |
| P2 capture, 360 tasks, six layers | 63 minutes |
| One JVP through half the network | 144 ms |
| Full forward on a 2k-token prompt | 2.5 s |

Assume 1.4× for the 4B model and 2.5× for the 9B model until `preflight` measures them.

## 6. Coding standards for this repository

- Python 3.13, `from __future__ import annotations`, frozen dataclasses for records, explicit
  `__all__` in modules that are imported widely, type hints everywhere, no new third-party
  dependencies without stating why in the report (the probe package is numpy-only by design).
- Public functions keep their current signatures where the spec does not change them; where a
  signature changes, update every caller listed in the wiring map and add a test that imports
  each caller.
- Tests: `uv run pytest` must pass. New behaviour gets a test on the fakes in
  `tests/test_probes.py` (`_ProbeModel`) or a new fake hybrid model you add next to it. Never
  load a real checkpoint in a test. Tests that need a saved artifact must `skip` if it is
  missing, like the existing `state-base.npz` regression test.
- Outputs: JSON plus Markdown for anything a human reads; every JSON records the command line,
  the resolved `ModelSpec`, seeds, and the control it was compared with.
- Logging: the pipeline prints live per-step transcripts; keep that behaviour and the `--quiet`
  flag.
- Docstrings state the contract, not the implementation; where a choice is load-bearing (mask,
  float32, seeds, atomic writes) say why in one sentence, as the existing code does.
- Do not reformat files you are not changing. Keep diffs reviewable.

## 7. Hand-off procedure

1. Move the spec you implemented from `pending/` to `under_review/` and place next to it
   `<SPEC-ID>-IMPLEMENTATION-REPORT.md` containing: files added/changed/removed (with line
   counts), the interface map rows you touched, decisions taken under ambiguity, deviations
   from the spec and why, the full `uv run pytest` output summary, grep results for the banned
   constants (§1.7), the list of callers you updated for each changed signature, and anything
   you could not verify because a model run is required.
2. Do not move anything to `complete/`; that happens after review.
3. If you found a defect outside your spec's scope, record it in the report under "observed,
   not fixed" with file and line; do not fix it silently.
4. Never run a model to "just check". State what a run would verify and leave it to the gated
   stage.

## 8. Glossary

- **Note**: the state-carrying text before the tool call in an assistant turn.
- **Window / `keep_last`**: number of recent tool observations kept verbatim; older ones become
  one-line stubs.
- **Recovery variant**: a task whose expert trajectory contains one deliberate failing step
  (`supervise=False`) followed by the supervised correction.
- **Screen**: the behavioural evaluation of each checkpoint used for selection.
- **Position baseline**: a `(family, step)` lookup table predicting a probe target from
  trajectory position alone; the reportable P2 quantity is the margin over it.
- **Within-position**: probe fit on activations and targets with the training-cell mean removed.
- **World A / World B**: the notes create state the model does not hold / the notes transcribe
  state the model already holds. The evidence says World A.
- **Integrity violation**: a note that copies its predecessor verbatim, drops a required-carry
  value, asserts completion early, miscounts, or states a stale fact (SPEC-002 §2).
- **View**: `ArchitectureView`, the only sanctioned way to run blocks by hand (SPEC-001 §2).

## 9. Addendum 2026-09-04: implementers spawned by the Deputy

If you are an Opus implementation agent spawned by the Deputy Chief, every rule above applies
to you unchanged. In addition: you commit nothing; you hand your change and an implementation
report (R16: every claim cites file:line) to the Deputy; the Chief's ratification on a GitHub
issue gates the commit. Read the wiring map §7 rulings R1 to R18 before starting; they
override spec text. Work only inside the paths your task names.
