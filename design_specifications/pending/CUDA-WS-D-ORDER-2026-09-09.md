# WS-D, for the D-CRO: un-port the lens to upstream, then extend it

**From the Chief, 2026-09-09.** Read the plan §2, §5 WS-D, §6, §11. The Director's instruction is to
**refine, extend, optimise and make upstream higher resolution**, using its code directly where we
can and resolving a confound wherever the implementation lets us. The stage-two analyses are closed;
WS-D begins when torch is installed and not before, on CPU float32, with your own rule in force —
nothing unrun is handed over as run.

## Use upstream directly, modified

| upstream | use |
|---|---|
| `jlens/fitting.py::jacobian_for_prompt` | **called**, through an ~80-line adapter that feeds it our corpus rows and writes our artefact (npz + sidecar + R57 identity + declared ν) |
| `jlens/fitting.py::valid_position_mask`, `SKIP_FIRST_N_POSITIONS` | generalised to a **cotangent selector** (WS-A builds the graph-once estimator with it; you use it) |
| `jlens/lens.py::JacobianLens.save/load/merge` | the in-memory lens object; our converter already maps `.pt` to npz; `merge` is the reduction for sharded fits |
| `jlens/hooks.py::ActivationRecorder` | via WS-A's `TorchCapture`; the sub-block hooks of §6.5 extend it |

**Deleted from `jacobian.py`:** the finite-difference machinery — `unit_directions`,
`finite_difference_steps`, `reference_responses`, `cached_responses`, `full_jacobian`, `self_check`,
`copy_cache`, `pre_norm_tail`, `PositionState`, `prepare_position`, `_directions`, ~450 lines.
**Kept:** `make_plan`/`freeze_plan`/`read_plan`, `Convergence`, `memory_gate`/`time_gate`/
`WorkloadMemoryGuard` (extended from *stop* to *choose*, plan §10.2), `benchmark`, `write_record`,
`run_jacobian_stage`. `regression.py` ports line for line (§3.7).

## What you build, in order

1. **The adapter and the un-port golden test.** Refit the hosted recipe — WikiText-103 train,
   128 tokens, bf16 on the remote, fp32 on CPU with fewer prompts — and compare per layer against
   `neuronpedia/jacobian-lens` with your own `compare_maps.py` (now on the branch). Agreement
   validates the un-port; **the residual is the finite-difference-versus-exact estimator
   difference, measured for the first time**, and it retroactively bounds every MLX-fitted lens.
2. **§6.1 declared ν.** Endpoint, position weighting, pair weighting, corpus, precision, estimator,
   in the sidecar; the hosted lens's sidecar gains upstream's defaults retroactively.
3. **§6.6 the second moment**, as you verified it: one `einsum('bpi,bpj->ij', g, g)` per pass,
   within-prompt and between-prompt terms **reported separately**, cost and +0.81 GiB in the §10.2
   table.
4. **§6.2 position bands** on the transcript corpus (Codex's task-1 recorder and corpus builder,
   already on their worktree, are the input; the 2,816-token coverage gate is theirs). This is what
   makes amendment 9 computable and the window secondary interpretable.
5. **§6.4 span-conditioned lenses** — the same selector, a different mask, from the labels the
   recorder emits.
6. **§6.5 sub-block and multi-target** — hooks on `self_attn` for 68 read points; `target_layer`
   looped for `J_{L→M}`.

## Confounds this stream resolves

The 21.8x position extrapolation on every published map number (bands); the prose-domain bias
toward notes (spans); the averaging cancellation the derivation warns of (second moment); the
23-to-24 commitment's attribution to attention or MLP (sub-block); the estimator difference nobody
has measured (the un-port test); and the ν that yesterday's hosted-versus-fitted comparison was made
without.

## Golden tests

Step 1's per-layer comparison; for each extension, the **canonical ν recovers the unconditioned lens
as the special case** — a band selector covering all positions reproduces the plain fit to float32
tolerance, a span selector over all spans likewise. That is the regression test that keeps
"higher resolution" from becoming "different instrument".

## Budget

~80 adapter, ~45 regression edits, ~400 extensions; ~450 deleted.

---

## Corrections from the survey (plan §13), which supersede anything above they contradict

Read plan §13 in full. The items below are the ones that change this order.
- **The un-port changes ν and must declare it.** Our MLX fit uses the same-position reduction;
  upstream sums over targets. The adapter supports **both** through the selector, the sidecar
  records which, and `profiles.py` keeps `jacobian` distinct from `hosted-jacobian`. A lens fitted
  under upstream's default is not the same estimator as ours; the un-port golden test compares
  like with like by running upstream's ν.
- **Finite differences versus exact autograd is Q3, the Director's**, decided by the pre-registered
  c-sweep: no plateau means instrument-limited and autograd; a plateau keeps the estimator. The
  saving is ~15.5x from the layer loop, not more.
- The residual dtype your regression fit consumes is **Q5**; `residual_source` in the artefact must
  name it either way.


## The Research Division's answers (plan §14) supersede the above where they conflict

Read plan §14 and `CUDA-MIGRATION-RESEARCH-BRIEF-ANSWERS-2026-09-09.md` in full.
- **Upstream fits at 128 tokens by default** and the published lenses were fitted so. Your golden
  test at 128 tokens is like-for-like. §6.2's transcript-length bands are a **different estimator by
  construction** and are recorded as a declared departure with their own memory model, never
  compared to the hosted lens as if they were one. The position selector exists only at readout
  upstream; the fit-time selector is ours.
- **`attn_implementation="eager"`** on the fitting model, or batched rows regress to sequential.
- Upstream has one commit and is unmaintained; there is nothing to track and no runner to diff.

## Upstream defaults, verified against the clone rather than reported

**Checked at `581d398` before any of this reached code, because three of them are load-bearing for
the adapter and one was not in anyone's message.**

| claim | verified |
|---|---|
| `fit(max_seq_len=128, skip_first=16)` | **yes** — both are defaults of `fit` *and* of `jacobian_for_prompt` |
| no fit-time position selector | **yes** — `jacobian_for_prompt` takes only `skip_first`; there is no `positions` parameter anywhere in the fit path |
| `JacobianLens.apply(positions=)` exists at readout | **yes** |
| upstream sets `attn_implementation` | **no** — it appears nowhere in `jlens/`, so the fitting model inherits transformers' default |

So 128 tokens is the reference estimator's **definition**, not a Neuronpedia choice; the fit-time
selector §6.2 needs is genuinely ours to add; and eager attention must be set by us because upstream
never mentions it.

**A fourth default, in the same class, that follows from reading the two signatures together.**
`HFLensModel.encode` defaults to `max_length=512`, and `JacobianLens.apply` defaults to
`max_seq_len=512`, while `jacobian_for_prompt` passes its own `max_seq_len=128` down into `encode`.
**So upstream fits at 128 and reads out at 512 by default.** Nothing in either signature warns of it;
a caller who fits with `fit(...)` and reads with `apply(...)`, both at their defaults, is applying a
lens four times outside the length it was fitted at and will see no error.

That is our own 21.8x extrapolation arriving from upstream's defaults rather than from our corpus,
and it is a trap for anyone told to "use upstream directly". The adapter passes lengths explicitly
at both ends and never relies on either default.
