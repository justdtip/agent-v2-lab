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
