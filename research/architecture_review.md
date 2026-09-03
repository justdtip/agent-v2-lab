# Architecture and training review for a higher-complexity local agent

Date: 2026-09-02

## Objective and constraints

The starting point is the 4-bit `mlx-community/Qwen2.5-Coder-3B-Instruct-4bit`
backbone plus the step-250 agent LoRA adapter. The extension must add genuine trainable
architectural capacity, remain trainable on a 24 GB M4 Pro, preserve the first adapter's behavior
at initialization, and improve unseen long-horizon tool tasks rather than merely reduce token loss.

## Primary literature consulted

- [LLaMA Pro: Progressive LLaMA with Block Expansion](https://arxiv.org/abs/2401.02415)
  expands a pretrained transformer with copied blocks, zero-initializes attention and FFN output
  projections so the inserted blocks are exact identities, freezes inherited blocks, and trains the
  new capacity. This is the closest match to the local constraint and the main architectural basis.
- [SOLAR 10.7B](https://arxiv.org/abs/2312.15166) shows that depth up-scaling plus continued
  training can improve a smaller pretrained model without a complicated mixture-of-experts
  runtime. It supports depth as the first scaling axis to test.
- [Scaling up Test-Time Compute with Latent Reasoning](https://arxiv.org/abs/2502.05171) studies
  repeated latent computation through a recurrent block. It motivates distributing added blocks
  through the backbone and leaves recurrent reuse as a later ablation, but its from-scratch training
  scale makes direct transplantation into an instruction-tuned quantized model risky.
- [Mixture-of-Depths](https://arxiv.org/abs/2404.02258) learns token-dependent depth under a fixed
  compute budget. Routing is a promising efficiency follow-up, but it confounds the initial capacity
  experiment and is therefore not included in run A.
- [ReAct](https://arxiv.org/abs/2210.03629) interleaves reasoning, actions, and observations for
  long-horizon interactive tasks. It motivates action-by-action trajectories with fresh environment
  feedback rather than single-shot imitation.
- [Toolformer](https://arxiv.org/abs/2302.04761) trains when and how to call APIs and how to use
  returned results. It motivates keeping tool selection, arguments, and result-conditioned next
  actions inside the supervised objective.
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) reports stronger complex reasoning
  from process supervision than final-outcome supervision. It supports supervising every action and
  evaluating intermediate validity, not only the final answer.
- [STaR](https://arxiv.org/abs/2203.14465) iteratively retains rationales that produce correct
  answers. It motivates later data iteration from verified successful traces and failure repair.
- [Quiet-STaR](https://arxiv.org/abs/2403.09629) adds learned internal rationales before future
  tokens. It is relevant to latent deliberation, but requires specialized objectives beyond the first
  expansion experiment.
- [LoRA](https://arxiv.org/abs/2106.09685) and
  [QLoRA](https://arxiv.org/abs/2305.14314) justify preserving the quantized backbone and its
  parameter-efficient specialization. The new blocks are the exception: they are full-precision,
  fully trainable capacity layered on top of the frozen QLoRA policy.

## Selected design: identity-preserving interleaved depth expansion

The base has 36 Qwen2 transformer blocks, hidden width 2,048, SwiGLU intermediate width 11,008,
16 query heads, and two key/value heads. An expansion inserts copied full-width Qwen2 blocks at
roughly uniform depth intervals. For each inserted block:

1. Layer norms and the q/k/v, gate, and up projections are copied from the preceding effective
   block, including the already-trained agent LoRA contribution after dequantization.
2. The attention output projection and MLP down projection are initialized to zero.
3. Residual connections therefore make the new block an exact identity before training.
4. All inherited 4-bit backbone and LoRA parameters remain frozen; all parameters in inserted
   blocks are trainable at floating-point precision.
5. Only the extension weights are checkpointed. Loading reconstructs the same identity blocks and
   applies the saved delta, keeping the original base and adapter reusable.

This is intentionally closer to LLaMA Pro than SOLAR: exact functional preservation makes it
possible to attribute any change to training, while interleaving avoids forcing all new computation
into the final representation stage.

## Iteration protocol

- Run A: four interleaved full-width blocks, process-supervised SFT on longer trajectories.
- Evaluate on split-isolated tasks requiring multi-hop retrieval, arithmetic joins, verified edits,
  distractor handling, and longer action horizons. Compare the untouched base, the first adapter,
  and the extension with identical greedy decoding and tool execution.
- Analyze failure classes and memory/time measurements.
- Run B: make a material revision based on evidence. Candidate revisions include six blocks,
  targeted replay of failures, explicit recovery trajectories, recurrent reuse of selected blocks,
  or an additional low-learning-rate backbone LoRA stage.
- Select by end-to-end held-out success, action validity, recovery, state correctness, and retention
  on the original 40-task suite. Validation loss is secondary.

## Rejected first-run alternatives

- Merely increasing LoRA rank adds trainable parameters but no inference depth; it does not satisfy
  the architectural-extension requirement.
- A router/Mixture-of-Depths layer may save compute, but does not directly raise the fixed model's
  representational capacity and introduces an additional failure mode.
- A new external planner would raise system capability without changing the model architecture.
- Full dequantization and end-to-end tuning would consume substantially more memory and risks
  forgetting the already verified policy.
- Reward optimization before a challenging benchmark exists would optimize an underspecified or
  saturated signal. Process SFT establishes competence first; outcome/process rewards are reserved
  for failure-driven iteration.

## Claims this experiment can and cannot support

A positive result will show that the particular expanded model improves on a controlled synthetic
long-horizon agent distribution while retaining the original suite. It will not prove that added
depth universally raises reasoning ability or that the agent is safe for real shell, network, or file
access. Architecture attribution requires at least one capacity or training ablation, which is why
the workflow requires multiple runs.
