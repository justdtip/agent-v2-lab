# WS-B: generation, the `none` cache path, and the golden trajectory harness

**From the Chief, 2026-09-09.** Read the plan §1, §2 (generation), §5 WS-B, §7, §12.1. You can start
today against a stub view; WS-A's real view slots in when its gates pass. Develop on CPU torch in
float32; record what passed at a commit; mark anything unrun as unexecuted.

## Reuse before invention

Everything scientific in the runner is backend-free and stays: `run_task`, `detect_loop`,
`_ThinkingTracker`, `build_prompt`, `window_messages`, `Simulator`, the transcript. The readout
gate, `_logit_hash`, `ForwardLedger.validate` and `residual_source_agreement` are the comparison
tools; the harness composes them, it does not rewrite them.

## What you build

**1. `runner.py`, torch branch of `generate_turn_with_count`.** MLX used `mlx_lm.stream_generate`;
torch gets a hand-rolled greedy loop of ~50 lines — prefill, then argmax, append, single-token
forward with the cache, decode, and the **existing** stop rule (`stop_ids`, `turn_is_complete` on
the decoded text). No HF `generate`: the stop needs a per-token decision and the golden records were
produced under exactly these semantics. Determinism from `device.py` (deterministic algorithms,
pinned attention kernel, dtype path recorded).

**2. `make_turn_cache` for `cache_strategy: none` only.** `trim`, `snapshot` and `history` are
deferred — stage two ran under `none` and the golden records never exercised them.

**3. `research/acceptance/golden_trajectories.py`.** From each stage-two episode's recorded
`prompt_ids` (`begin_turn` rows), regenerate greedily and compare to the `emitted` rows
token-for-token; on mismatch report the **first divergence point** with both tokens and the two
layer-34 distributions there. Then the readout comparison: argmax agreement and max-abs logit error
against the recorded `argmax` and `logits_sha256` per forward.

**4. Gates 5 and 6 of `scripts/acceptance_gates.py`**, the remote diagnosis kit (plan §12.1): the
trajectories, then the lens reads through the hosted lens against the recorded `rank` rows.

## Confounds and tools this stream strengthens while implementing

- **`longest_identical_run` is fooled by a two-cycle** (register): the loop metric becomes
  cycle-aware — longest run of a repeating *period*, period recorded — and both columns are emitted
  so old tables remain readable.
- **Truncation is a distinct outcome**, not a wrong answer (pre-registration §14): a turn that emits
  exactly `max_tokens` and yields no parseable action is recorded as `truncated`.
- **75 of 180 tasks are graded stricter than their prompt states** (§13): the verdict gains a
  `contains_expected` field beside `success`, computed under the existing normaliser, so every pass
  rate carries its bound.
- **The readout-gate tolerance is measured, not chosen**: the CPU-fp32-versus-MLX-4-bit delta on the
  golden episodes is the declared band, and the remote's bf16 result is checked against a projection
  from it.

## Golden tests

- All 15 stage-two trajectories reproduce token-for-token on CPU float32 from recorded `prompt_ids`.
- Layer-34 argmax agreement and max-abs error reported per episode with the quantisation
  attribution (MLX 4-bit against CPU fp32 on the same prefix).
- The four new fields — `truncated`, `contains_expected`, cycle period, longest period run — present
  in the manifest for every episode, and the old fields unchanged.

## Budget

~60 new in the runner, ~250 in the harness, ~40 in the verdict and loop metric; ~300 deferred.
