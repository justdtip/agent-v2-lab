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

## Review of the sampled path, Chief, 2026-09-10 — `b29c509`, on `cuda-migration` at `ac69ab6`

**Verdict: passes.** SWE-1 commits on the integration branch, so the review is of what landed:
the seat's suite 2,661 passed, 10 skipped, box free, and the Chief's forward merge over it
2,666 passed under the Chief's window at 06:58Z. Read in full: `SampledDecoding` refuses any
temperature but the ruled one and any truncation by name at construction, seeds its own generator
from the pinned seed unless told, refuses to draw when nothing is pinned, and draws from the full
softmax; `torch_sampled_stream` is the greedy loop's sibling with one line different, and a test
holds the greedy function to having no sampled branch and never calling the sampler; the sampled
mode is declared by the object and the bare word is refused; an MLX sampler carrying a temperature
is still refused on torch; one `require_greedy` sits behind the golden generator, the tolerance
runner and both device scripts at the parser, before records, window or weights; the manifest
block carries mode, temperature, truncation, sampler, seed with its source, draws, the kernel set
read back and the note verbatim, and a greedy block carries neither seed nor temperature. The
state programme's script now selects this loop through the policy seam (run order, review of
`fa5fadc`).

**One additive follow-up, SWE-1.** The shared stop rule reports an EOS-ended turn as `token_cap`
on both backends, because only the tool-call id is in its stop set and the stream ends itself on
EOS. Relabelling would move a field the MLX records are compared on, so the label stays; add an
`ended_on_eos` boolean beside the reason on both backends, with a test on each, so a record never
says a turn was capped when the model ended it. Own pathspec commit, as before.

## Review of `ended_on_eos`, Chief, 2026-09-10 — `49de004`, on `cuda-migration`

**Verdict: passes.** The Chief's own run of the full suite on the tip: 2,676 passed, 10 skipped at
nine sites for absent worktree artefacts, under the Chief's window at 07:26Z; the torch, sampler
and state sets 92. Read in full: the fact is computed from the last token consumed against the
model's own terminator set and changes no stop decision; the turn output stays a three-tuple for
every existing caller and carries the reason and the fact as attributes; a generator that cannot
say leaves `None` rather than a guess; the step record and the transcript carry it beside the
unchanged count-based `truncated`. The builder's flagged choice is right and stands: `truncated`
is a compared field and stays count-based, so a turn that ends on its terminator exactly at the
cap reads `truncated: true` beside `ended_on_eos: true`, which is two true statements. Nothing
further for WS-B on the laptop.

## The device's first hour, WS-B, 2026-09-10 — the seam is faithful and the gate fails; a ruling

SWE-1's record `research/records/WSB-DEVICE-2026-09-10/` (`0350db4`, on `cuda-migration`). Two facts
held together: **the port is device-consistent** on the card, CUDA and CPU agreeing to one position
in 5,245 with thirteen of fifteen episodes giving the identical set of confident disagreements on
both and two CUDA runs of one episode byte-identical; and **the pre-registered gate fails**, 24
positions where the MLX recording held its token at P ≥ 0.99 produced a different argmax, on CUDA
and identically on the CPU. So the failure predates the card and was invisible on the laptop because
one episode of fifteen had ever been measured, and that episode is clean everywhere. Peak 7.92 GiB
for an episode and 8.81 GiB for the corpus against a 7.56 GiB projection; 4.0 s per episode on the
card against 83.5 s on the laptop's CPU.

**Ruling.** The gate stays failed as pre-registered; nothing is passed by borrowing a tolerance or
by reading the 0.57% flip rate at confident positions against the 21.2% at unconfident ones as a
pass, however concentrated the disagreement is where precision puts it. Two hypotheses remain and
the record separates neither: a narrow port defect, or the rule's premise, that 4-bit quantisation
cannot move a confident argmax, being too strong for MLX 4-bit against torch bf16. The separating
test is torch bf16 against **MLX bf16** on the same fifteen episodes, and it needs MLX, so it is a
laptop run: authorised, as a teacher-forced pass over 5,245 tokens under an announced window with
its projected peak, each episode written as it completes; not a run that ties up the box. If MLX
bf16 shows zero confident flips against torch bf16, the premise was 4-bit's and the golden records
are re-based on MLX bf16 with that stated in every comparison; if the 24 remain, it is the port,
and the search starts from the eleven above P = 0.999. Gate 5's failure at the first near-tie is
the free-running cross-backend comparison the programme already ruled invalid, and stays so.

**Gate 5 restated, Chief, 2026-09-10.** The free-running cross-backend reproduction that gate 5
asked for is struck, as the tolerance prose already ruled it invalid; it failed on the card at the
first near-tie (position 506, recorded P = 0.5155), which is the expected behaviour of an invalid
comparison, not a finding. Gate 5 is within-backend: two decodes of one prefix on one backend agree
byte for byte, which the two CUDA runs of `calculate-0158` already measured and passed. Gate 6 stays
unavailable until the lens path is ported. The finding at position 506, a single full forward
against chunked prefill with the cache differing at a near-tie, is carried into the cache-strategy
gate arm's notes as unchecked; that arm is the equivalence claim it touches.

## The precision-matched arm, 2026-09-10 — the premise was wrong, not the port; rulings

SWE-1, `df2f0a3`: MLX bfloat16 against torch bfloat16, teacher-forced over the turns carrying the
24 gated positions, 55 s on the laptop. **Twenty-one of the 24 are quantisation**: MLX bf16
produced exactly what torch bf16 produced and only the 4-bit record dissented, including the
position recorded at P = 1.000000. **Three are bfloat16 ties**: the top-two gaps, recovered exactly
from ln(p1/p2) and checked to lie on the 0.25 grid that is the bf16 step at that magnitude, are a
0-ULP gap (two candidates exactly equal), a 2-ULP gap pointing opposite ways in the two frameworks,
and one position where the card's own CPU and CUDA already differed, a device tie. A probability
margin had called all three "not a tie"; the ULP measure is the instrument. No port defect is
implicated at any of the 24, and the record says plainly that this is not evidence the port is
correct, only that these positions were never evidence against it; the positive evidence remains
device consistency to one position in 5,245 and the WS-A gates.

**Rulings.** The rule "a confident flip is a defect" holds only against a precision-matched
reference, with a margin in ULPs of the stored dtype: a position whose top-two gap on the
reference lies within two ULPs is a tie, counted and reported as a tie, never as a flip and never
as agreement; against a differently quantised recording the check is not a defect test and
refuses to be read as one. The golden records are re-based on MLX bf16: the full fifteen episodes
through the MLX bf16 arm on the laptop under one announced window, the new reference recorded with
its provenance beside the 4-bit one, which stays as the record of what the rule was measured
against; then the kit runs once more on the card against the new reference, and the expectation is
zero confident flips outside listed ties. The rule goes into code; the lesson goes into the method
record as the thirty-first entry, in SWE-1's name.
