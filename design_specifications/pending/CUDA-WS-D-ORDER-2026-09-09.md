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
