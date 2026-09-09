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

---

## Corrections from the survey (plan §13), which supersede anything above they contradict

Read plan §13 in full. The items below are the ones that change this order.

- **The three cache strategies are in your first cut, not deferred**: `history` is `qwen35-4b`'s
  declared default with equivalence recorded and `trim` is the default model's; ~35 lines over
  `cache.layers`. `SnapshotCache` restore is broken on MLX today (position not in `.state`); fix it.
- **EOS is never appended by the loop** and three downstream sites depend on that; HF `generate`
  appends it. **EOS ids come from the config's set `[1, 106]`**, never `tokenizer.eos_token_id`.
  The `</tool_call>` stop-set expression stays byte-for-byte; it is a no-op on Gemma by design. The
  decode-loop reference is `mlx_lm/generate.py:424-470` **and 715-753**; the trailing yield at
  741-753 is what delivers EOS on six turns.
- **Prefill in chunks of 2,048**, final token separate, asserted by `ForwardLedger.validate`.
- **The golden corpus has 5,245 emitted tokens, not 4,801.** G-2 drives `replay_record` with
  `logits_sha256` **dropped** from the key and `argmax` armed, and must also exempt the second
  enforcement site at `profiles.py:306-315`. **One argmax flip at P ≥ 0.99 fails the run outright.**
  Then a free-running replay of the three chat episodes.
- G-3 reports decode and prefill **separately**: MLX reads exactly 0.0 on all 5,339 decode forwards
  and 0.375–0.75 on the 100 prefill forwards. `replay.py` needs a `readout_factory=` seam (~14 lines).
- TF32 off; deterministic algorithms on; `CUBLAS_WORKSPACE_CONFIG=:4096:8`; SDPA backend pinned.


## The Research Division's answers (plan §14) supersede the above where they conflict

Read plan §14 and `CUDA-MIGRATION-RESEARCH-BRIEF-ANSWERS-2026-09-09.md` in full.
- **G-2 is two tests now.** (a) CUDA-versus-CUDA exact replay, bit-identical, same build and device.
  (b) MLX-versus-CUDA as tolerance: teacher-forced argmax agreement with the P ≥ 0.99 rule, the
  divergence-index distribution with a floor, top-5 Jaccard, per-step KL percentiles against the
  recorded layer-34 distributions. **Cross-backend token-for-token reproduction is struck**; it is
  not achievable and nobody gates on it.
- **`attn_implementation="eager"`** on the replay path. Batch one, unpadded, chunked at 2,048.
- Cache offsets from the API, never tensor shapes; `activate_past_recording()` on sliding layers at
  construction for `trim`, `snapshot` and `history`; negative arguments to `crop`.

## Amendment, 2026-09-09 late — cache strategies are arms of the gate

The torch cache strategies were built against the real `DynamicCache` and measured what the
deferral was waiting on: arming rollback after a sliding window has filled corrupts silently (now
refused at `enable_rollback`), and an armed sliding layer is unbounded, 2.15x the bounded KV cache
at the map's longest episode on this model, a ratio of a small base (~380 MB against ~180 MB at the
class-default head dimension; read both off the loaded config). Memory does not decide. Ruling:
`make_turn_cache` keeps refusing all three under torch until the tolerance runner has its baseline
at `none` against the view; then each strategy is its own arm of the same gate and must reproduce
the `none` trajectories byte for byte within the backend. A strategy that changes one token is a
defect, not a speed setting. Plan §16.8.

## Amendment, Chief, 2026-09-10: the sampled decoding path, ordered on the Director's ruling

The Director lifted the temperature refusal today, on the Chief's advice and scoped as the advice
was (plan §16.16). SWE-1 builds, on fixtures, device-free:

- **A sampled decode beside the greedy loop**, not inside it: the greedy path's output stays
  byte-identical to before and a test says so. The sampled path draws from the full softmax at
  temperature 1.0 with no top-p or top-k truncation, from a generator seeded through the device
  shim's pinned seed, and refuses any other temperature or any truncation by name: an argument
  for another value is a change to what the device's runs measure and is a separate ruling.
- **Only for runs declared as the state programme.** Every golden gate, the tolerance runner and
  every comparison against an MLX record stay greedy and refuse the sampled mode by name.
- **Provenance in every manifest of a sampled run**: `decoding: {mode: sampled, temperature: 1.0,
  truncation: none, sampler: <the function's name and the backend>, seed}`, with the note that
  sampled draws reproduce only within one backend and kernel set, so a sampled record is compared
  only with a sampled record from the same backend.
- **Tests**: a fixed seed reproduces the draw on CPU; two seeds differ; the greedy path unchanged
  byte for byte; a gate handed the sampled mode refuses; the manifest fields present and typed.

Own pathspec commit on your branch, pushed for the Chief's review and merge as before. The state
programme's script (`STATE-PROGRAMME-RUN-ORDER-2026-09-10.md`) lifts its own refusal of
`--decoding sampled` only once this lands on the branch it runs from.
