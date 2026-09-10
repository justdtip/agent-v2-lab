# WS-A, for Codex through the Director: the torch seam, and the graph-once estimator

**From the Chief, 2026-09-09.** Read `CUDA-MIGRATION-PLAN-2026-09-09.md` first, §1, §3, §7, §10–12.
This is the hardest piece and the one everything else is measured against. Nothing CUDA runs until a
device is rented; **you develop and validate on CPU torch in float32 on the laptop**, against the MLX
golden records, and you record what passed gate by gate at a commit. Anything not run is marked
unexecuted. The Director's standing rule for this pivot: **reuse before invention; a new convention
where an implementation exists is a defect.**

## Use upstream directly, modified

The cloned `anthropics/jacobian-lens` at `/Users/daniel.tipton/reference/jacobian-lens` (Apache-2.0)
already contains two of the four things this stream builds. Build on them; do not reimplement beside
them.

| upstream | what it does | what you do with it |
|---|---|---|
| `jlens/hooks.py::ActivationRecorder` | forward hooks on `model.layers[i]`, stores each block's output keyed by index, `start_graph_at` roots autograd at a chosen layer | **`TorchCapture` extends it**: add a pre-hook for the entry residual and the per-layer kwargs, the sink protocol, and the intervention hook |
| `jlens/hf.py::_find_layout`, `HFLensModel` | structural discovery of `layers`/`embed`/`norm`/`lm_head`; `unembed`; `encode` | **the torch view's discovery delegates to `_find_layout`**; our `_TEXT_MODULE_PATHS` becomes the fallback, not the primary |
| `jlens/fitting.py::_check_layer_indices` | the layer convention: sources `0..target−1`, target `n_layers−1` | **the residual convention test asserts against it**, so ours can never drift from upstream's |

## What you build

**1. The shared base.** Extract from `arch.py` the backend-neutral parts — index validation,
`lora_targets` over a tuple of linear types, `_validate_scored_position`, `_validate_hidden_span`,
the docstrings that carry rulings — into `arch_base.py`. The MLX view inherits it unchanged; this
is ~40 lines moved, not written, and it is your first commit so the MLX suite stays green.

**2. `arch_torch.py::TorchArchitectureView`**, same public surface as `ArchitectureView`, no
signature change (plan §1). The specific design points:

- `_observe_forward`: a forward **pre-hook** on every `layers[i]` capturing `hidden_states`,
  `attention_mask`, `position_embeddings` as the model passes them. Observe, never reconstruct — the
  principle that found the entry-scale and boolean-mask defects on MLX (`arch.py:461-509`).
- `masks(h, cache)` returns `{i: {"attention_mask": m_i, "position_embeddings": pe_i}}`, an opaque
  bundle per block; `run_block(i, h, masks, cache_i)` unpacks it and calls
  `layers[i](h, attention_mask=..., position_embeddings=..., past_key_values=...)`. HF Gemma 3
  builds `causal_mask_mapping[layer_type]` and `position_embeddings[layer_type]` and dispatches on
  `config.layer_types[i]` — the same two-mask structure MLX had, so `attention_span(i)` reads
  `config.layer_types[i]` directly.
- The cache adapter: HF's single `DynamicCache` presented as the per-block list the repo expects,
  each entry exposing `offset` (from `get_seq_length`) and nothing else until a strategy other than
  `none` is ported. `make_cache` returns it; `cache_trimmable` is `crop`-backed.
- Every hand-run return is `.float()` (diagnostic dtype promotion, plan §3.3).
- `residuals`, `native_residuals`, `residual_source_agreement`, `cached_logits`, `tail`,
  `native_readout`, `unembed`, `final_norm`: as on MLX, over the torch model.

**3. `TorchCapture`**, extending `ActivationRecorder`: the sink protocol
`residual(layer, offset, h)` / `output(offset, ids, logits)` unchanged so `session.py` does not
know the backend changed; and **`intervene(layer, position, fn)`** where `fn: h → h'`, applied in the
forward hook, covering add (today's injection), replace, and interchange
(`h + D(z' − z)` on a frozen support). Design it for replacement from the start; the state
derivation's §7.1 and §11.3 need it and it should not be a second migration.

**4. After gates 1–4 pass: plan §6.3, the graph-once estimator.** Upstream's
`jacobian_for_prompt` replicates the prompt `dim_batch` times and retains the whole graph
(`fitting.py`: `input_ids.expand(dim_batch, -1)`, `retain_graph=True`); at 2,816 tokens that is
~3.7 GB per layer. Rewrite it as `jacobian_for_prompt_vjp`: one unreplicated forward,
`torch.func.vjp` with cotangents batched by `vmap`, the same one-hot-per-valid-position semantics,
the same `.mean` over sources, **a cotangent selector argument** so §6.2's position bands and §6.4's
spans are a mask and not a fork. Keep upstream's function beside it and **assert the two agree on a
short prompt to float32 tolerance** — that is the golden test for the rewrite. Then sharding:
prompts across devices, `JacobianLens.merge` (upstream, already written) as the reduction, world
size 1 being the single-device case.

## Confounds this stream resolves by construction

- The hand-run-versus-native disagreement that cost a day on MLX: on torch the native path *is*
  hooks, and `residual_source_agreement` becomes a regression test rather than a port gate.
- Head capture "architecturally unavailable" on Gemma: it was unavailable to MLX's emitter; a hook on
  `self_attn` is model-agnostic. Not ported now, but the design leaves the seam.
- The residual-convention ambiguity, settled yesterday by reading upstream's source: now asserted
  in a test against that source.
- The 21 GiB fit projection: not a batching choice — there was no batch — but a retained graph, and
  §6.3 removes it.

## Golden tests, in order, on CPU float32

1. Discovery reports 34 layers, 2,560 hidden, `layer_types` spans matching MLX's `attention_span`.
2. `residual_source_agreement` at 64 and 1,400 tokens: zero at both; the mask-dispatch negative
   control non-zero at 1,400 only. **This is the test that has found every seam defect so far.**
3. Layer-34 identity: native softmax top-1 is the emitted token, every decode step.
4. The readout gate with both negative controls, tolerance set from the measured fp32 distribution
   and recorded with its basis; `None` and `0.0` remain different values.
5. (§6.3) `jacobian_for_prompt_vjp` agrees with upstream on a 64-token prompt, layers 0–33, to
   float32 tolerance; then the memory at 2,816 tokens measured, not projected.

## Budget and what is unexecuted

~450 new lines for 1–3 against 1,120 kept; ~120 for 4. Everything marked unexecuted until the gate
above it passes on CPU, and the manifest of what passed at which commit ships with the branch so the
remote's first hour attributes any failure to the device change.

---

## Corrections from the survey (plan §13), which supersede anything above they contradict

Read plan §13 in full. The items below are the ones that change this order.

- The per-block bundle is `(attention_mask, (cos, sin), position_ids)` with `(cos, sin)` **per layer
  type**; one rope pair gives five of six blocks the wrong base. `cache_position` is not needed.
- `cache.layers[i]` is the per-block handle already. `crop` takes a **negative** count; return the
  trimmed count yourself. `.state` carries no position on either backend — `SnapshotCache` restore
  is broken on MLX today; fix, do not reproduce.
- `create_causal_mask(allow_is_causal_skip=True)` may return `None` meaning *use `is_causal`* — the
  same trap as MLX's `"causal"` sentinel. Pin it `False` or handle the `None` route explicitly.
- `NativeCapture.__enter__` imports `Qwen3NextAttention` unconditionally; make it conditional on head
  capture. The HF model returns `CausalLMOutputWithPast` and takes `past_key_values=`; unwrap
  `.logits` where `arch.py:965` and `session.py:410` expect a tensor.
- Add `view.input_device` and `view.vocab_size` to the contract. `from_model` never calls `.to()`.
- **`residuals()` takes a capture dtype.** Today it is fp32-per-block recomputation; hooks give the
  native path that no probe consumes. Which one the torch view's `residuals()` returns is **Q5**, the
  Director's; build both behind the parameter and default to whatever he rules.
- **TF32 off** before any fp32 matmul: `torch.backends.cuda.matmul.allow_tf32 = False`,
  `torch.set_float32_matmul_precision("highest")`. Set in `device.py`, printed by the diagnosis kit.
- **Prefill in chunks of 2,048** (`NATIVE_PREFILL_STEP_SIZE`), final prompt token separate, or the
  forward rows change shape under an identical trajectory.
- Stable top-k tie-break by ascending token id (`session.py:106-111`); `torch.topk` guarantees no
  order. Competition ranks are one-based (`records.py:62`); `jlens/vis.py::_ranks_of` is zero-based.
- **G-1 has three negative controls**, each biting at 1,400 and not at 64: mask dispatch, hook-site
  off-by-one, entry-transform omission — and the controlled number must not be bit-identical to the
  uncontrolled one. Pre-registered criterion: ≤ 1e-3 relative per layer in bf16, control two orders
  above it.


## The Research Division's answers (plan §14) supersede the above where they conflict

Read plan §14 and `CUDA-MIGRATION-RESEARCH-BRIEF-ANSWERS-2026-09-09.md` in full.
- **`attn_implementation="eager"`** on the model the view wraps, **for determinism**: under `sdpa`
  Gemma runs two attention kernels per forward. The batching-rule argument is withdrawn (plan §14.2,
  `c0e4233`): batched rows are ~1.5x faster than sequential under both implementations at steady
  state; the missing rule costs one first call only.
- **§6.3 uses `torch.autograd.grad(..., is_grads_batched=True)`**, not `vjp` + `vmap`; upstream's
  `ActivationRecorder` then works unchanged. The acceptance test for the rewrite is a **timing ratio**
  ≥ 1 against sequential on a small fixture. Never `output_hidden_states=True`; never
  `register_full_backward_hook`; never `flash_attention_2`.
- Cache offsets from `get_seq_length()` / `get_mask_sizes()`, never `keys.shape[-2]`; call
  `activate_past_recording()` on every sliding layer at construction if anything will rewind.
- The memory claim in your §4 was against a straw man: upstream fits at 128 tokens by default. Restate
  it: the saving is the forward tape saved once, at whatever context length *we* choose.

## Amendment, 2026-09-09 evening — review of `8edb4cd`, and the answer to the memory question

Reviewed in full (plan §16.6). Passes on substance; three edits before it merges into
`cuda-migration`, which now exists on origin:

1. **Reach `jlens` through the seam.** `arch_torch.py` and `torch_capture.py` import it at module
   top level and nothing declares it, so in the shared venv both test files ERROR at collection and
   abort the whole run. Import through `load_upstream()` (being relocated to
   `local_llm_lab/upstream_ref.py`; until then it lives in `pipeline/lens_fitting/upstream.py`) and
   add `pytest.importorskip` for `jlens` in both test files, so a missing clone is a skip with a
   reason. Plan §16.2.
2. **Re-run the evidence on torch 2.14.0**, the plan's floor; the record's isolated runtime was
   2.9.1. The 79 branch tests pass here on 2.14.0 with the clone on the path; the record should say
   the same from your side.
3. **Do not merge `WS-A.patch` / `patch.json`.** A record cites the commit range and its shas; a
   2,584-line copy of the diff is a second source of truth that is stale at the first edit.

**The memory question: no larger host.** Load the checkpoint in its stored bf16 and let
`_call_promoted` upcast per block: the hand-run matmuls are the same arithmetic on the same numbers
as a float32 load (an exact upcast of the same bf16 values) at 7.2 GiB resident. Declare the one
thing that differs — the natively observed rotary tables and masks are bf16 — by reporting the max
deviation between the rotary module evaluated in float32 and the observed tables at the gate's
length, as one number in the record. Unembed scored rows only (`cached_logits` / `native_readout`);
the fp32 `unembed` promotes the tied 2.7 GB head per call. Measure the ~8.5 GiB projection before
believing it, with MLX not loaded. The float32-loaded run is a control for the GPU host. Plan §16.1.

Two standing rules from tonight: in a worktree set `PYTHONPATH=<worktree>/src` or the tests exercise
main's code (§16.4); stage by explicit path and read `git log --stat` over the range before every
push (METHOD-2026-09-08).

## Review of the edit round, 2026-09-09 late — through `e0a05cf`

**Verdict: the three edits pass; the fourth arrived unasked and is accepted; the branch merges into
`cuda-migration` at `6106ea8`.** Checked here, not read off the record: the branch's tests against
its own source through the seam on torch 2.14.0, the rules suite on the actual merge result, the
allowlist line, the patch files gone.

1. `jlens` is reached through `load_upstream()` in the view, the capture and the acceptance helpers;
   a missing clone skips with the seam's message, a second copy refuses. Done.
2. Evidence rerun on torch 2.14.0 (`development-torch214.json`), and the record says which run
   each number came from, including the scanner failure before the allowlist line. Done.
3. `arch_base.py` in the approved-modules list, with `arch_torch.py` asserted as still scanned.
   Done, and the second assertion is the better half.
4. `WS-A.patch` and `patch.json` gone. Done.
5. `input_device` on the view, reporting the embedding's placement without moving the model: not
   asked for, accepted, because WS-D's `CorpusLensModel.register` reads exactly that attribute off
   upstream's model contract, and the two seats now agree on it by construction.

**The precision diagnosis is accepted as the transferable technique**, and it is the answer to the
question §16.1 left open. On a fixture holding the same bf16-rounded weights in both copies: the
float32 loop against float32 native is exactly zero at 64 and 1,400 tokens; the promoted loop
against **float32 native** is below 0.073%, which is the rotary-table and mask rounding §16.1 said to
declare; the promoted loop against **bf16 native** is 0.6–1.1%, which is bf16 block arithmetic and
not the seam. So a cross-precision comparison includes more than rotary rounding, the record says so,
and the 1.24% at 64 tokens on the checkpoint is consistent with it without yet being isolated.

**Ruling on gates 1–4 under bf16 loading, which the record correctly asked for rather than
assumed.** The registered 1e-3 bound was written for a comparison at one precision and stays: it
governs the float32 loop against float32 native (exact on the fixture; the GPU host's control, where
a 14.5 GiB float32 load costs nothing) and every within-backend exact comparison. It does not
govern a cross-precision comparison, and no tolerance chosen after the fact may. On the laptop the
gate has three parts:

- **Seam exactness at the model's own dtype, gated at zero.** Add the arm the diagnosis is missing:
  the hand-run loop in bf16, through `_block` without promotion, against bf16 native, at 64 and
  1,400 tokens on the checkpoint. Same modules, same kwargs, same dtype: the expected value is
  exactly zero, as the float32 arm is on the fixture, and any nonzero is a seam error and not
  precision. This isolates the 1.24% with no extra memory.
- **The precision floor, declared and not gated.** The promoted loop against bf16 native, as
  measured, recorded beside the rotary-table deviation (2^-9 measured) and the fixture's
  decomposition, as the floor beneath which a cross-precision residual means nothing.
- **The controls, gated outside the floor.** Mask dispatch at 1,400 tokens on the checkpoint, and
  the hook-site and entry-transform controls, must each fail by a margin the floor cannot explain;
  on the fixture the mask control was 21% against a 1% floor.

Then the measured peak beside the 9.8 GiB projection and the 10.656 GiB cap, per §10.2, and the
graph-once estimator (§6.3) starts. The refused first attempt is the lock doing its job: another
seat's suite had MLX mapped, and the wrapper stopped before loading anything.

**Next instruction.** (1) The bf16-loop arm above, at both lengths, in `cpu_gates.py`'s report.
(2) The three controls at 1,400 on the checkpoint, each against the declared floor. (3) Peak memory
measured and recorded. (4) Fetch `cuda-migration`; your branch is merged there and WS-B's tolerance
runner will run against your view first. (5) Then §6.3. Reviews continue to land here, under a
dated heading naming your commit.

**Addendum, same evening.** The text-only loader is lifted into the package as
`local_llm_lab.hf_text.load_text_causal_lm` (plan §16.9): your `load_text_model`'s fail-closed
checks, with the Gemma classes and the listed vision prefixes replaced by detection from the
checkpoint's own headers and `AutoConfig`/`AutoModelForCausalLM`. `cpu_gates.py` may import it in
place of its own copy; the report it returns carries the same fields plus the storage dtypes and
the sha256 per file. And the three worktree readers that resolved shared artefacts against the
running checkout (`models/`, the HF cache) now resolve against the git-derived primary,
`runlock.primary_checkout_root()`, never through the box-state override.

**Superseding note, 2026-09-10 early — the next instruction's arms run on the device, not on CPU.**
The Director: "All subsequent runs will use the rented GPU. Do not plan for long runs that will tie
up this box. Test as much as you can, but accept we may need to resolve bugs in the actual
environment." So items (1) to (3) of the next instruction above, the bf16-loop arm, the three
controls at 1,400 tokens and the measured peak, are the device's first hour, with `cpu_gates.py`
becoming the device gate script: the CPU calibration figures you have (1.24% at 64 tokens, the
rotary rounding at 2^-9, the 9.8 GiB projection) are what the device's numbers are compared
against. On this box from here: fixture tests only. Item (4), fetching `cuda-migration`, stands;
item (5), the graph-once estimator, is written and tested here on fixtures and run there. Plan
§16.12.

## Next instruction, 2026-09-10 — three tasks that need no device, in this order

The Director's ruling stands (plan §16.12): nothing model-scale runs on the laptop; fixtures and
tests do. Each task below is a test before it is a feature, and each carries its own record
section in the shape WS-C's record has: the finding, the technique without reference to our code,
the implementation in one line, then the device checklist row with the number it must produce and
the laptop figure it is compared against.

**1. The graph-once estimator, §6.3, on fixtures.** `torch.autograd.grad(..., is_grads_batched=True)`
over upstream's recorder, the forward tape saved once at whatever context length we choose, never
`output_hidden_states=True`, never `register_full_backward_hook`, never `flash_attention_2`. The
acceptance test is the timing ratio ≥ 1 against sequential on the small fixture, under both `eager`
and `sdpa` (Research's `tests/torch/test_batched_cotangents.py` is the shape), plus exactness: the
batched Jacobian equals the sequential one to float32 epsilon on the fixture, per layer. It is
written and tested here and run on the device, where its memory claim (the tape once, not per
cotangent) is measured for the first time.

**2. `cpu_gates.py` becomes the device gate script.** Load through
`local_llm_lab.hf_text.load_text_causal_lm` and delete your own loader copy, so there is one;
`device.select()` and `device.pin(seed, attention="eager")` before the first CUDA use;
`device.describe()`, the checkpoint sha and the source commit in every report's head; every phase
written and flushed as it completes (yours already does; keep it); `basis: measured-here |
laptop-basis | expected` on every number; the bf16-loop arm added (the hand-run loop through
`_block` without promotion against bf16 native, expected exactly zero, at 64 and 1,400 tokens); the
three controls at 1,400 gated outside the declared floor; the float32-loaded control under the 1e-3
bound, which fits on the device. Fixture-tested here against the tiny model, with the fixture
mirroring the snapshot's declared type and key layout (plan §16.9: a fixture built from
`Gemma3TextConfig` declares a shape no registered checkpoint has). Its device run is the first
hour's step 1 in the runbook, `CUDA-DEVICE-FIRST-HOUR-2026-09-10.md`.

**3. The WS-A record as the first hour's checklist**, in your own order, cheapest and most
diagnostic first, each row with what a mismatch means. The 1.24% at 64 tokens, the rotary rounding
at 2^-9, the 9.8 GiB projection and the fixture's precision decomposition are the laptop bases;
say of each which was taken on an idle box.

Reviews continue to land here under a dated heading naming your commit. Two rules from tonight apply
to every seat: commit with `git commit -- <paths>` in a shared checkout, never bare; and a suite
reading is not a claim without its skip count and the box state on the same line.

## Review of the three tasks, 2026-09-10 — `4aaab19`, merged into `cuda-migration` at `6b22d44`

**Verdict: all three pass; merged.** Checked against the merge result itself: the branch's torch
tests through the seam, the estimator's fixtures, the rules suite, the device and loader tests,
162 passed; the record's own gate tests, 88 passed; the merge-tree clean against the integration
tip. Every CUDA measurement remains unexecuted, as the checklist says in its first line.

1. **The graph-once estimator.** `torch_jacobian.jacobian_for_prompt_vjp`: upstream's recorder,
   layer convention and position rule unchanged, one forward, batched one-hot cotangents at every
   selected target position through `torch.autograd.grad(..., is_grads_batched=True)`, source mean
   over the same selection, no target-count division, hooks removed on every failure path. Exact
   against unmodified upstream at float32 epsilon under `eager` and `sdpa`, remainder batches
   preserved, the selector local rather than a patch of upstream's global, and the warmed
   end-to-end batching ratio at least one under both kernels. Its memory claim is the device's.
2. **The device gate script.** Loads through `hf_text.checkpoint_metadata` and
   `load_text_causal_lm`, its own loader copy gone; `device.pin` before the first device use; every
   number carries a basis; the `native_dtype_loop` phase is the seam-exactness arm gated at zero;
   the precision arm declared, the controls gated outside the floor; resume verified.
3. **The checklist.** Prepared on fixtures, every CUDA row marked unexecuted, the technique stated
   without reference to our code, and the laptop bases with their box state.

Nothing further is asked of WS-A on the laptop. The next thing Codex does is on the device, in the
order the checklist gives, and its record comes back here under a dated heading.

## Next instruction, 2026-09-10, second — the decoder intervention wrapper, on fixtures

The derivation `sae_j_lens_state_derivation.md` (§7.1) and its order `SAE-J-BRIDGE-ORDER-2026-09-08.md`
are now requirements for the device phase (plan §16.16). One piece is yours because it sits on
`TorchCapture.intervene`, which you built. Device-free; fixture-tested against the tiny decoder.

**1. The decoder intervention.** `h' = h + D_F (z'_F − E(h)_F)` inside the capture's `intervene`
function at a layer and position: the SAE's encoder `E`, decoder `D` and bias `b` are taken as
plain callables and tensors (SWE-2's Gemma Scope 2 loader lands separately; until then the tests
use a synthetic dictionary with a JumpReLU encoder), the feature set `F` and the target values
`z'_F` are arguments, and the base reconstruction residual is preserved by construction, which is a
test: `h' − b − D z'` restricted off `F` equals the base residual off `F`.

**2. The re-encoding diagnostic.** After the intervention, report `E(h')_F` against `z'_F` and the
change in `E(h')_{−F}` against `E(h)_{−F}`, as numbers in the record, never as an assertion that the
target was attained: the derivation is explicit that attainment must be measured. A test constructs
a dictionary where re-encoding does not return the target and checks the diagnostic says so.

**3. Clamp mode.** Today an intervention applies once per fresh prefill. A clamp re-applies at every
forward, keyed to absolute positions or to every emitted position, and is released explicitly; the
record distinguishes a one-shot patch from a clamp, since the derivation treats them as different
claims (§11.3). Tests: the clamp holds across steps through a cache, the one-shot does not, and a
clamp on a cached position fails closed the way the existing intervention does.

**4. Model-agnostic**, as everything: no layer count, no width, no family class; the layer list and
the feature set come from the caller. Pathspec commit on `codex/cuda-torch-seam`, merged after
review as before. Its device use is the state programme's, ordered after the migration validates.

## Review of the wrapper, 2026-09-10 — `ff3b1b4`, merged into `cuda-migration` at `e8dbcc3`

**Verdict: passes on all four points; merged.** Checked against the merge result itself, not the
branch: the intervention and capture tests, the architecture view, the graph-once estimator, the
upstream seam, the rules suite, the device shim, the loader and the import-tree guard, 218 passed,
0 skipped, under the Chief's own window at 05:43Z; the merge-tree clean against the integration
tip, and no deletion on main since the last integration base for the seat's main merge to carry
forward. The source was read in full.

1. **The decoder intervention** is the derivation's line: encode, take the selected difference,
   synthesise only the selected decoder directions, add to the original. The base reconstruction
   residual is preserved for the whole vector, not only off `F`, and the test states the identity
   exactly in float32 rather than to a tolerance. Bias is checked and never added, since it cancels.
   The decoder is the derivation's `[residual, feature]` matrix or a linear bias-free callable, and
   the module says which; a square dictionary could hide a transposed matrix from the shape check,
   which no real dictionary is.
2. **The re-encoding diagnostic** reports the target error and every unselected feature's change as
   numbers, and the counterexample is the right kind: target 3, achieved 5, another feature moved
   by 2, residual unchanged. The encoder output is cloned before re-encoding, because a reusable
   output buffer rewrote the baseline during Codex's own review and reported a false zero; the
   failing regression is kept.
3. **Clamp mode** reapplies on every forward that recomputes the position, is released explicitly
   through a handle, and the record separates `one_shot` from `clamp` per registration with a
   per-application event carrying forward index, cache offset, absolute position and the
   diagnostic snapshot. "Every emitted position" is implemented as positions the caller declares
   on each forward, and that is the correct reading: the capture cannot see which input tokens
   were sampled, and inferring it from cache shape would have been a guess written as a fact. A
   clamp on a cached position refuses before any block runs, and the cache length is unchanged.
   Both refusals during a forward, and the recursive-forward case in its caught and uncaught forms,
   are tested; the counter that keeps a caught recursive rejection from unlocking the outer forward
   is the subtle piece and it is right.
4. **Model-agnostic**: no layer count, width or family class anywhere; the layer list and the
   feature set are the caller's.

**Two notes for other seats, not edits to this work.** SWE-2's loader hands the wrapper the
decoder in `[residual, feature]` orientation, which for the Gemma Scope 2 parameters means the
transpose of the stored matrix, and the encoder callable returns the residual's dtype and device,
so the dictionary's precision is the loader's declared choice. The full off-target vector per
application is right for fixtures and wrong at corpus scale, where a clamp across an episode
would hold dictionary-width lists per emitted token; the summary form below is ordered so the run
script (`STATE-PROGRAMME-RUN-ORDER-2026-09-10.md`) has it.

## Next instruction, 2026-09-10, third — the diagnostic's summary form, on fixtures

One small addition, device-free. `SAEIntervention` takes `diagnostic="full" | "summary"`, default
`full`. Under `summary` the selected features keep `before`, `target`, `achieved` and
`target_error` in full, and the off-target change is reduced to its count of nonzero entries, its
maximum absolute value, its L1 norm and the eight largest entries by magnitude as
`(feature, change)` pairs; the record names the mode in the diagnostic's `kind`. Tests: the summary
is the reduction of the full form on the existing fixture, exactly; the capture record carries
either form; an off-target change of zero everywhere summarises to zeros, not to an absence. No
change to the edit itself and no change to the full form. Pathspec commit on
`codex/cuda-torch-seam`, merged after review as before. After it, nothing further on the laptop:
the device arms in the checklist are next, in their order.

## Review of the summary form, 2026-09-10 — `fcab5b5`, merged into `cuda-migration` at `3973f95`

**Verdict: passes; merged.** Against the merge result: the intervention, capture, bridge,
architecture, estimator, state-run, rules, import-tree and device-setup sets, 289 passed with user
warnings promoted to errors, under the Chief's window at 06:39Z; no deletion on either side. The
form is what the order asked: `diagnostic="full"` the default and unchanged, `"summary"` keeping
the selected readings whole and reducing the off-target change to its nonzero count, maximum
magnitude, L1 norm and eight signed pairs with ties broken by feature index; the kind names the
mode; an empty or all-zero complement summarises to zeros and an empty list, never to an absence;
the edit and its gradients identical in both modes. The test that no dictionary-width Python list
is ever built, with the full form as its negative control, is the check the order's reason rests
on, and the L1 overflow refusal clears stale evidence rather than keeping it.

Nothing further on the laptop for WS-A. The device arms in the checklist are next, in their order,
and their records come back here under dated headings.

## The device's first hour, WS-A, 2026-09-10 — three runs, one finding, one rule, a pass

Run on the rented RTX PRO 6000 by the Chief, since Codex cannot reach the card; records
`device-gates-01..03` with their logs on `cuda-migration`. **The seam is exact on the card**: the
native-dtype loop against the native capture reads 0 at every site at 64 and at 1,400 tokens,
and the sliding window is applied natively beyond 1,024 (1,024 allowed keys per row in sliding
blocks, 1,400 in global; `device_mask_probe.py`). The float32 control passes at both lengths under
its 1e-3 bound. Device peak 15.66 GiB against a 24 GiB projection, host peak 23.25 GiB, 18 s.

**The finding.** The promoted-float32 floor reads 1.2386% at 64 tokens on both backends, to four
digits, and 69.4% at 1,400 tokens on CUDA against 6.9% on the CPU, while the mask-dispatch control
reads 54.5% on both. The first run failed only because that control is judged against the long
length's own floor, which on the card is not precision noise but the promoted path's own
divergence; the seam it protects was exact. The rule is amended (`bc710e6`): a long control must
lie strictly above the native seam's error and the short length's promoted floor, the one
magnitude that agrees across backends, and the long floor is recorded as a finding. The finding
itself goes to Q5: on CUDA beyond the window, a capture through the promoted path is 69% from
native bf16, so the capture dtype is decided on the card, not inherited from the laptop.

**Three near-misses of the Chief's in the same hour, each caught by a gate.** The amended rule was
committed once without the amendment, because the patch command broke before the edit and the
tests then passed on the unpatched file; the second run therefore ran the old rule, which the
record's own rule string exposed. A chain read a pipe's status and committed past a failing test;
the forward merge's suite refused it. A rerun was launched without the repository root on the
path and without the full commit hash, and the script refused both by name. Gates 3–4 remain
unexecuted by the script's scope and are WS-B's handoff.

## Rows 6 and 7 on the device, 2026-09-10 — the graph-once estimator on the card

Run by the Chief with `device_graph_once.py`, which rebuilds the fixture from Codex's own
definition (seed 0, the tiny Gemma 3 text config, the same frozen ids) with the model placed on
`cuda:0`, since the tests are CPU fixtures with no device knob and the checklist says repeating
them on a GPU host is not a CUDA measurement. Record `device-graph-once-01.json` on
`cuda-migration`. **Row 6 passes**: every source block against upstream's sequential estimator
within float32 epsilon under both kernels, SDPA exactly zero, eager worst 2.4e-7 absolute against
the scale-aware bound. **Row 7 passes**: the warmed end-to-end ratio of sequential over batched is
9.66 with eager attention and 8.17 with SDPA, best of three, against 2.98 and 2.69 on the laptop's
CPU; taken while the D-CRO's fit held the card, so `basis: shared-card`, and re-taken when the card
is idle before it is quoted as a device performance figure. **Row 8**, graph memory at the intended
context on the real checkpoint against upstream's replicated batch, waits for the card alone; it
is the measurement the 12B fits need, since the exact fit's peak on the 4B read 43.9 GiB for one
row at dictionary batch 64.

## Review of the comparability analysis, 2026-09-10 — `f1d934f`, merged into `cuda-migration`

**Verdict: passes as an analysis record; merged.** `research/records/GPU-COMPARABILITY-2026-09-09-0955Z/`
analyses SWE-1's two full-corpus runs on the card, and its four source hashes equal the bytes
SWE-1 committed in `WSB-DEVICE-2026-09-10/`, so the analysis is of the canonical record. Its
conclusions are the record's: 24 reference-gated mismatches on both the card's GPU and its CPU,
with the per-episode counts differing on two episodes so equal totals are not identical
positions; one net agreement difference in 5,245; the gate's probability being the MLX 4-bit
reference's own confidence and not shared ground truth; and the whole labelled instrument
comparability rather than model behaviour, with no significance claim. The script refuses changed
source hashes and forbids model imports, which is the right shape for a file-only analysis. The
figure is honest: the two chat episodes at 19–20% disagreement are the free prose the gate was
never about, and the gated panel shows where the 24 sit.

Two amendments, small. Cite the canonical record path beside the card's directory, since the
record is what survives the rental. And the limitation "flip-position identities are absent" was
true of the sources at the time and is no longer true of the record: SWE-1's `661a453` added
`confident-flips.json`, all 24 positions with turn, position, both tokens and the recorded
probability, after a cap in the flip printer was found to hide half of them.

**Codex may read the card.** The analysis reached `/workspace/wsb/out/` as the laptop's own user;
that is allowed, read-only: no window, no run, no write outside a directory of Codex's own, and
the seats' output directories are never modified. Every rule of the primer applies there too.

**Next instruction, when SWE-1's precision-matched arm lands** (torch bf16 against MLX bf16 at the
24 positions, running on the laptop now): extend this analysis, in a new record directory, by
joining `confident-flips.json` with that arm's per-position outcomes, and draw the figure of
which of the 24 survive precision matching and at what recorded probability, with the same
hash-refusing, model-import-free script. That figure is the one the Director reads to decide
whether the golden records are re-based on MLX bf16 or the port is searched.

## Review, Chief, 2026-09-10 — `7db3336`, `558d424`, `6983d8e` merged into `cuda-migration` at `9e082ab`; next instruction

**The precision-matched join (`7db3336`)** is accepted as a record: the 24 joined exactly on
episode, turn and position; 21 resolved to the change of precision; the three remaining read with
the actual spacing (1, 3, 1 ULP) against the assumed scale (1, 6, 2); and its caution that
"outside the band does not itself prove a port defect, because cross-framework numerical error has
not been bounded" is the same point the Chief ruled in the WS-B order tonight — the port's
resolution is a corpus-wide measurement, and SWE-1 produces it tomorrow.

**The finite-difference diagnosis and protocol (`558d424`) and the saturation audit (`6983d8e`)**
are the Director's consultation and are now the WS-D order: the D-CRO executes the protocol's
§1–§6 on the card; the audit's two corrections to the golden record's §10 are carried in the WS-D
ruling of the same date. Both records passed file-only verification and touched no run.

**Next instruction, through the Director, file-only as before:** (1) when
`research/records/WSD-FD-CALIBRATION-2026-09-10/` lands on `cuda-migration`, audit it against the
protocol's §5 quantities and §6 table — in particular that individual responses were preserved
before reduction, that the boundary check preceded every derivative, and that no execution error
was read as a scientific branch; (2) before the plan-progress pre-registration is sealed, review
the D-CRO's draft against `STATE-PLAN-PROGRESS-ORDER-2026-09-10.md` at f98dbaf — the three corpus
facts with their basis, A1–A4, R2's labels, R3's retry clause, R4's within-pass H2, R5's ladder, the
twelve exploratory episodes and the corpus digest `7fe6e64b…f99120` — and report any rule the
draft does not carry. The rented card stays read-only for Codex.

## Routing, ruled by the Director 2026-09-10: review requests reach Codex through this file

The Chief sends Codex review requests by writing them here and pushing main; Codex answers by
commit on `codex/cuda-torch-seam`; the Research Division's watch on `codex/*` tips notifies the
Chief. No verbal relay in either direction. The card stays read-only for Codex.

**Two standing requests, each file-only, each triggered by a commit on `cuda-ws-d`:**

1. **Re-review of the plan-progress pre-registration** once the D-CRO's commit applies P1–P6 and
   the episode-unit correction (`STATE-PLAN-PROGRESS-ORDER-2026-09-10.md`, the two sections of this
   date): confirm each item is carried as ruled — `forward_batch: 1` and the token index in the
   capture contract; the retrieval candidate set, tie policy and the transport distance with its
   metric; §13 refreshed; the episode-level bound with the subgroup epsilons and the decision-count
   epsilons marked; five-fold episode cross-fitting with folds and seed in the seal; both digests —
   and report any rule the draft still does not carry. The seal waits for this review.
2. **Audit of the width-1 ladder** when `WSD-FD-CALIBRATION-2026-09-10` carries it: against the
   protocol's §5 quantities and §6 table, and against C2 of the WS-D order — individual response
   vectors preserved before reduction, the boundary check preceding every derivative, `anchor_batch`
   and `forward_batch` declared and equal to 1, both precisions on one forward path, no execution
   error read as a scientific branch, and the unexecuted pieces listed as unexecuted.

Report as before: a record directory under `research/records/`, sources pinned by hash, no model
loaded, nothing on the card changed.

*2026-09-10, later:* the width-1 ladder landed on `cuda-ws-d` at `5b0feb7`/`15eddfa`; standing request 2 is live.

*2026-09-10, later:* standing request 2 delivered at `2678218` (`WSA-LADDER-AUDIT-2026-09-09`), accepted in full and ruled in the WS-D order's correction of this date; merged. Standing request 1 stays live until the D-CRO's P1–P6 commit exists.

## Director → Codex, 2026-09-10: the direction reviews audit against from here

The programme's purpose is to replicate what Anthropic found about the model's workspace, on
agentic problems, and then to steer what the agent *does* rather than what it says. The experiments
now being defined (`WORKSPACE-EXPERIMENTS-2026-09-10.md`, forthcoming) ask global-workspace
questions of an agent at its decision positions: ignition and competition when it commits to a tool
call; whether the chosen action lives in one subspace available across all task families or in
family-specific ones; where the decision draws its content from, with Gemma's six global layers as
the only long-range channel; and whether the progress note is a report of a decision already made
or part of the deciding.

Audit each measurement for whether it bears on the workspace property it claims to, not only
whether its arithmetic reproduces. The failure modes to hunt, in this order: an instrument artefact
read as a workspace property — this week that has meant an overcomplete dictionary, a lens read
outside the positions it was fitted at, a carrier re-read from the transcript, a batch or precision
path difference, and a test that certifies its own implementation; a claim about the model made
from a median that hides the tail; and a control that matches size but not structure. A result that
passes names its positions, its precision and width, its lens's identity and fitting positions, its
nulls, and the unit of its confidence bound. Review the design before any capture is read and
audit the result after, as for the calibration. Keep counting the gates that can disagree with their
producers against the ones that cannot.

**Standing requests 3 and 4:** (3) a design review of the workspace-experiments order when it lands
on main, before any capture is read; (4) an audit of each experiment's record when it lands on
`cuda-migration`.

*2026-09-10 UTC:* standing request 1 is live on `cuda-ws-d` at bbdc8dc (P1–P6 applied) and covers, with it, the plan-progress capture code at 144fa1c (`pipeline/state_programme/capture.py`, its 34 tests, and the resume shard numbering the D-CRO found and fixed); both are merged into `cuda-migration`. Standing request 3 (the workspace-experiments design) is live on main through 70c44bb, W-3b included.

*Added to the brief, 2026-09-10 UTC, from method entry thirty-four (the D-CRO):* the one-line
diagnostic that found six of seven broken verifiers this week — ask what the check would say if the
thing it guards were broken, then break it — and the rule that a verdict without the set it was
computed over is half a verdict: every review states what was checked beside whether it passed.

*2026-09-10 UTC, later:* Codex's run-capabilities review (`712f78c`) found the empty current-note cut before its pass ran; ruled and repaired in the workspace order of this date. **Standing request 5, file-only:** audit the repaired `workspace_w3b.py` and `workspace_w3b_analyze.py` at the commit that carries this line (`research/records/WORKSPACE-EXPERIMENTS-2026-09-10/scripts/`) against P1's requirements — the boundary from token offsets, the explicit query cut, non-empty forbidden edges at `P_act`, refusal on a failed check — and say what each check would say if the cut were misplaced; the fixture for a deliberately misplaced cut is owed and may be specified by the review. Request 4 stands for the readings that follow.

*2026-09-10 UTC, later:* standing request 5 delivered at `63d0549` and applied in full (workspace order, rulings on the audit of the repair): placement certified by a self-test on Codex's fixtures, a text oracle per row and per-arm receipts with effective-mask checks; admission and the paired transition metric in the analysis. **Standing request 6, file-only, when the W-3b record lands on main:** audit the actual pass outputs — the receipts, the effective-mask fields and the admission block — against the review's G6–G8 obligations, on the rows as written, not on fixtures; and say which of G9 and G10 the real tokenizer and the digest branch now settle. Request 4 stands for the readings.

*2026-09-10 UTC, later:* Codex's review of the steering pilot (59e6b11, local to its worktree; relayed) found a crash on a valid negative outcome and four reporting defects; all applied before the run (`research/records/STEER-PILOT-2026-09-10/`, producer b3b676a0467c, reporter 7f646b820e82, fixtures beside them). **Standing request 7, file-only, when the pilot's record lands on main with its output:** audit the run's admission block, the paired outcome states and the void cells against the review's obligations, on the rows as written; and say whether the smoke's one-row observation (an operation switch at layer 20 only) survives as a reading over the cohort or dissolves.

*2026-09-10 UTC, 07:15Z:* the 4B W-3b v3.1 pass has run with its real receipts (300 rows; 1,886 arm receipts, every one passing; the analyzer admits 300 of 300 in strict mode), the 4B capture's per-row index is verified against the corpus (7,629 rows, 0 differences), and the workspace record on main carries the first 4B readings from this commit. **Standing request 6 is live.** The steering pilot's first attempt died at model load (an out-of-memory of the Chief's scheduling, recorded in its record, `research/records/STEER-PILOT-2026-09-10/`); standing request 7 waits for the re-run's output.

*2026-09-10 UTC, 08:05Z:* the steering pilot's re-run has completed on the base 4B (24 recipients, admitted in strict mode, no void cell; `research/records/STEER-PILOT-2026-09-10/results/`): the operation donor switches the tool in 24 of 24 recipients at layers 20–32 and nothing at 4–12; the three path donors move nothing at any layer. **Standing request 7 is live** from this commit.

*2026-09-10 UTC, 08:50Z:* Codex's file-only readout (`WORKSPACE-DATA-READOUT-2026-09-10`, its worktree 5b60282) accepted in full: the v3.2 carrier control short in 55 of 187 rows (count-matched 13 of 125 against 5 of 124; verified on the rows), the window restriction (48 against 1), the two narrative discrepancies (the 4B fit's dimension batch is 32, not 16; W-5 is not fully resolved at 26) — corrected in the workspace record — and the argument audit (12 of 24 donor matches, all 24 from the recipient's listed values; verified). v3.3 adds a carrier arm count-matched in every row; the 12B pass runs it and the 4B is repeated. Codex's proposed next measurement (force only `calculate`) is noted for the next window, not authorised here.

*2026-09-10 UTC, 09:35Z, bookkeeping of the queue:* Codex's file-only readout (`WORKSPACE-DATA-READOUT-2026-09-10`) reparsed all 984 pilot calls, regenerated the 768 outcome cells, checked the 192 same-state identities, the 960 single-application counts and the 192 residual differences — **standing request 7 is delivered** by it. The same readout checked the 1,886 v3.1 and 2,186 v3.2 W-3b receipts and effective-mask values, the v3.2 key counts, the carrier and position-0 exclusions and the query intervals on the rows as written — G7 and G8, and G6 as recorded by the producer. **Standing request 6 is re-scoped to what remains, on the v3.3 outputs when they exist (the 4B repeat and the 12B pass): G6 by an oracle of Codex's own over the real rows (decode the masked keys from the corpus and the capture's offsets, not from the producer's receipts), G9 with the real tokenizer (the six first-token identities and P_act = S−2 on the rows), and G10 by confirming the digest branch was the one exercised (`corpus_sha256` in run.json against the capture's).** Standing request 1 (the plan-progress pre-registration re-review at bbdc8dc, with the capture code at 144fa1c) stays open and is the gate to the state-programme seal; the D-CRO's two captures are complete and unread behind it.

*2026-09-10 UTC, 10:40Z:* **standing requests 1 and 6 are delivered** at `89268e3` (`WSA-OUTSTANDING-REVIEWS-2026-09-10`): S1 and S2 ruled in the plan-progress order (cbe307e) and sent to the D-CRO as the pre-registration's owner, the seal waiting on them; M1 accepted and applied as W-3b v3.4 (the corpus digest mandatory from the manifest or the capture-time event, path equality refused; capture v2.2 writes the digest into the manifest) — promoted before the 12B pass. What remains of request 6 is the final acceptance of the v3.4 stores once they exist (the 12B pass and the 4B repeat), against G6–G10 as re-scoped at 09:35Z.
