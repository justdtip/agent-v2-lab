# Local LLM Lab

A local-first Apple Silicon workspace for:

- chatting with small MLX language models;
- LoRA/QLoRA supervised fine-tuning with Apple's `mlx-lm`;
- writing and testing reward functions;
- small GRPO experiments with `mlx-tune`.

The defaults are deliberately conservative for a 24 GB M4 Pro. Model downloads and
training outputs stay under this directory and are ignored by Git.

## Setup

`uv` and Python 3.13 are already available on this Mac. Recreate or update the environment
with:

```bash
uv sync
uv run lab-check
```

You do not need to activate the virtual environment. Prefix commands with `uv run`, or use
`source .venv/bin/activate` if you prefer an activated shell.

## Try a model

The first run downloads about 350 MB into `.cache/huggingface`:

```bash
uv run lab-chat --run
```

The default is `mlx-community/Qwen3-0.6B-4bit`. Select a larger model with `--model` after
the small workflow is working.

## Supervised fine-tuning (SFT)

Replace the examples in `data/sft/{train,valid,test}.jsonl` with your data. Each line is a
JSON object containing a `messages` array. Keep a held-out validation and test split.

Preview the exact command without downloading or training:

```bash
uv run lab-sft
```

Start the small QLoRA run:

```bash
uv run lab-sft --run
```

Configuration lives in `configs/lora.yaml`. Because its model is quantized, `mlx-lm`
automatically uses QLoRA. Adapters are written to `outputs/sft/adapters`.

## Reward functions and GRPO

Reward functions are ordinary Python callables in `src/local_llm_lab/rewards.py`. Exercise
them without loading a model:

```bash
uv run lab-rewards
uv run pytest
```

`data/rewards/train.jsonl` contains toy prompt/answer pairs. Inspect the GRPO configuration
and reward distribution:

```bash
uv run lab-grpo
```

Start a deliberately tiny experiment:

```bash
uv run lab-grpo --run --steps 5
```

This is reward-driven **QLoRA**: the 4-bit base model remains frozen, LoRA adapters are
inserted into selected attention projections, and GRPO updates only those adapters. The
reward function does not need to be differentiable; it scores generated text and the trainer
turns relative scores into adapter updates. Ordinary SFT cannot consume a reward function—it
uses next-token cross-entropy instead.

GRPO generates several answers for every prompt, so it is much slower than SFT. A useful
reward must create score variation among those answers. For serious work, first use SFT to
teach the response format, then run GRPO on fresh prompts that were not used for SFT.

## Practical progression

1. Get the task and evaluation set right before training.
2. Establish the base model's held-out score.
3. Run SFT and compare against that baseline.
4. Unit-test rewards against good, partial, and adversarial responses.
5. Run a tiny GRPO job and confirm rewards have non-zero variance.
6. Scale model size, examples, sequence length, or steps one variable at a time.

The included data is only a smoke-test fixture. It is far too small to improve a model.

## Agent pipeline v2 (current)

The current approach is documented in `research/agentic_paradigm.md`. In short: every model
turn is a short state-carrying note plus one native Qwen tool call, the harness hides all but
the last two tool observations so the notes must carry the facts, training data includes
executed-but-unsupervised mistakes followed by supervised recoveries, checkpoints are chosen by
held-out behaviour rather than loss, and every stage prints a live per-step transcript.

```bash
uv run agent-pipeline data          # expert + recovery trajectories, manifest with hashes
uv run agent-pipeline train         # QLoRA on the frozen 4-bit base (configs/agent_v2.yaml)
uv run agent-pipeline train --resume-from outputs/agent-v2/adapters/0000400_adapters.safetensors --iters 200
uv run agent-pipeline select        # screen every checkpoint on the validation split
uv run agent-pipeline eval --base   # baseline on the held-out test split
uv run agent-pipeline eval          # best adapter on the held-out test split
uv run agent-pipeline eval --stress # inject a transient tool error into every task
uv run agent-pipeline rollout --split iter1   # verified on-policy data for the next round
uv run agent-pipeline branch --split pref1    # step-level preference pairs for DPO
uv run agent-pipeline prefer                  # DPO from best-adapter on those pairs
uv run agent-pipeline report                  # all evaluation summaries in one table
uv run agent-pipeline all           # data -> train -> select -> eval
```

Transcripts stream to the terminal and are saved under `outputs/agent-v2/transcripts/<stage>/`
as Markdown per task plus a `transcripts.jsonl`. Metrics land in `outputs/agent-v2/evals/`.
Pass `--quiet` to print only pass/fail lines. Turn reuse of the KV cache across a task's turns
is on by default (37% faster, verified output-identical by `research/cache_equivalence.py`);
`--no-cache` disables it.

## Earlier experiments (kept for reference)

## 3B agent training run

The main experiment trains `mlx-community/Qwen2.5-Coder-3B-Instruct-4bit` to operate a
six-tool virtual environment. Agentic behavior is measured as end-to-end held-out task
success, not merely whether an output resembles JSON.

Generate the deterministic, split-isolated trajectories:

```bash
uv run agent-data
```

Benchmark the untouched base model, train QLoRA, then evaluate the adapter:

```bash
uv run agent-eval --split test --limit 40
uv run mlx_lm.lora --config configs/agent_lora.yaml
uv run agent-eval --split test --limit 40 --adapter outputs/agent-3b/best-adapter
```

Every intermediate tool action is expanded into its own supervised row. With prompt
masking enabled, loss is applied to the agent action rather than to system prompts or tool
observations. The evaluator executes actions against a deterministic simulator and verifies
required tool use, final state, exact grounded answers, parse validity, and tool errors.

Compare the models interactively with identical prompts and decoding settings:

```bash
uv run compare-chat
uv run compare-chat --tools
```

The comparison loads the untouched base and the selected step-250 adapter side by side.
Without `--tools`, it is an ordinary multi-turn chat comparison. With `--tools`, both models
receive the exact six function schemas used by the agent harness, so you can compare their raw
tool-call choices. Type `exit` to leave the interactive session, or use `--prompt "..."` for a
single reproducible comparison.

Inspect both models' complete traces on the same held-out agent task:

```bash
uv run agent-compare --task-index 3
```

The trained adapter is in `outputs/agent-3b/best-adapter`. The full held-out results are in
`outputs/agent-3b/base-eval.json` and `outputs/agent-3b/checkpoint-250-eval.json`; the model card
records the training recipe, selection evidence, limitations, and artifact hashes.
