# SAE decoder interventions on CPU fixtures

Order: `ae6ea94`, second 2026-09-10 heading in the WS-A order. Work starts from
`add9bc0` on `codex/cuda-torch-seam`. No checkpoint, CUDA, MPS or native MLX run is
part of this task. Device use waits for migration acceptance.

## Implementation plan and acceptance

1. Implement the additive decoder edit with a synthetic JumpReLU dictionary.
   Preserve the full base reconstruction error against the algebraically replaced
   code. Report re-encoding's target error and changes to all other features.
2. Extend capture with explicit, releasable clamps while preserving its existing
   one-shot API. Forwarded generated-token positions are caller-declared; they
   cannot be inferred from a token tensor. Refuse targets already inside a cache.
3. Integrate the callable edit with capture, record each application's mode and
   positions beside its numerical diagnostic, and verify on CPU fixtures.
4. Record fresh acceptance evidence, source commit, skip count and box state;
   commit and push the bounded branch for review.

The exact invariant is `h' - b - D z_replaced = h - b - D E(h)` for the entire
residual vector. The edited code equals `E(h)` outside the selected features. It
is not `E(h')`: that distinction is the diagnostic's purpose. Decoder matrices
use the derivation's `[residual, feature]` convention; callable decoders must be
linear and bias-free, with `b` supplied separately. Bias cancels from the edit.

Status: ready for review. Implementation is committed at `eb601af`. Acceptance:
**196 passed, 0 skipped, CPU; another seat held a CPU bridge window and the desktop
was not certified idle.** Native MLX imports were blocked. No model behavior,
device timing, feature semantics, autonomous retention or on-distribution claim
is made here.

## Finding and transferable technique

This is a finding about the instrument, not about a trained language model.
A decoder-direction edit can preserve the complete base reconstruction error
while failing to attain its requested encoded value and changing other features.
For the recorded counterexample, the selected feature starts at 1, the target is
3, the re-encoding is 5, and the other feature increases by 2. The reconstruction
error remains `[-1, -1, 3]` exactly in the float32 fixture. `example.py` reproduces
these numbers without a model; `counterexample.json` holds the measured record.

The portable method is to encode the current vector, subtract its selected code
from the requested code, synthesize only that decoder-direction difference, and
add it to the original vector. Re-encode the result separately and record both
target error and every other feature's change. Do not replace the original vector
with the SAE reconstruction, and do not silently identify the algebraically edited
code with the encoder's response to the changed vector.

The first encoder output must be snapshotted before re-encoding: a callable that
reuses one output buffer otherwise rewrites the baseline and falsely reports zero
cross-feature change. A failing regression reproduced this during review; the
wrapper now clones the encoder output while retaining its gradient connection.
Diagnostic subtraction uses CPU float64 so finite float32 readings of opposite
sign do not overflow their reported difference. Both encodings and the model edit
stay in the caller's chosen precision.

## Implementation and use

`SAEIntervention` takes encoder and decoder callables (or a decoder matrix), a bias
vector, feature indices and target values. It has no dependency on a dictionary
loader, layer count or model family. A matrix uses only the selected columns for
the edit; a callable receives the full sparse difference vector. A callable decoder
must be linear and bias-free. That is a declared input contract, not a global
linearity claim inferred by testing a few inputs. Residual-shaped values must match `h`; encoder outputs have dictionary width
and targets have length `|F|`. All tensors share the residual's dtype and device;
conversion is the caller's decision.

```python
edit = SAEIntervention(E, D, b, features=F, target_values=desired)
with TorchCapture(view, sink, layers=layers) as capture:
    capture.intervene(layer, position, edit)  # existing one-shot API
    capture(prompt_ids)                      # no reuse in this example
    one_shot_record = capture.intervention_record
```

For sustained interventions, use a separate explicit handle:

```python
with TorchCapture(view, sink, layers=layers) as capture:
    handle = capture.clamp(layer, "emitted", edit)
    capture(prompt_ids, past_key_values=cache, emitted_positions=[])
    # absolute_position is the position of this forwarded generated-token input.
    capture(next_ids, past_key_values=cache, emitted_positions=[absolute_position])
    handle.release()
    record = capture.intervention_record
```

An emitted position means a **forwarded generated-token input**, not the input
position whose logits sample the next token. Sampling provenance is invisible to
the capture, so the caller declares it on every forward while this clamp is active.
An empty declaration is valid for a prompt-only prefill. Several declared positions
can be forwarded together; all must be unique absolute positions inside that
forward. Direct calls to the native model cannot provide this wrapper metadata and
are refused while an emitted clamp is active.

`clamp(layer, absolute_position, edit)` reapplies whenever that position is freshly
computed. If it lies in the existing cache, capture refuses before blocks advance;
it does not claim to update a past residual by editing future tokens. One-shot
registrations still apply once per fresh-prefill sequence and re-arm on a new
prefill. They compose with clamps in registration order. Release is explicit and
idempotent and stops future edits; it does not erase an intervention's effects
already stored in the cache. A clamp is a sequence of interventions, and maintained
control under a clamp is not autonomous retention.

`intervention_record` is a defensive JSON-safe copy. Each registration identifies
its layer, mode (`one_shot` or `clamp`), selection and release state. Every
application supplies forward index, cache offset, absolute position, and a snapshot
of the callable's numerical diagnostic. An event reports an applied hook, not a
successful completion of the entire forward: a later layer can still fail. The
record contains no tensors or autograd graphs. Full off-target vectors and all
application events consume storage proportional to dictionary width and event
count; no constant-memory or device-throughput claim is made.

## Acceptance map

| Requirement | Direct evidence |
| --- | --- |
| Keep original reconstruction error | JumpReLU fixture compares the full before/after error against the algebraically edited code; replacing h with a reconstruction would fail |
| Report re-encoding rather than assert attainment | Explicit target 3 / achieved 5 counterexample, nonzero off-target change, threshold-crossing and reusable-buffer cases |
| Clamp across cached steps; one-shot once | Tiny native Hugging Face Gemma fixture with actual `DynamicCache`, alongside scheduling fixtures |
| Cached absolute position fails closed | Cache length unchanged by refusal; no block forward takes place |
| Explicit release and distinct record modes | Application counts, registration ordering, release state and per-application diagnostic copies |
| Preserve capture seam | Native output containers, donor gradients, layer numbering, architecture and Jacobian regression suite |
| Clean up on failure | Direct and wrapper recursion, failures in callbacks, rejected metadata, hook removal and context re-entry |

## Remaining device work

No real checkpoint, real SAE dictionary, CUDA execution, device memory peak,
on-distribution effect, semantic feature interpretation, or memory/retention result
was tested. Device use remains ordered after migration acceptance. This wrapper
provides the intervention and the measurements needed to assess those claims; the
fixture does not establish them.


## Verification and commit binding

- `326b0a0`: initial decoder wrapper, error-preservation and re-encoding fixtures.
  First focused acceptance: 19 passed, 0 skipped, CPU, desktop not certified idle.
- `eb601af`: complete source and tests, including clamp scheduling, real tiny
  DynamicCache integration, encoder-buffer and recursive-hook corrections.
  Final affected acceptance: **196 passed, 0 skipped**, CPU during SWE-2's CPU
  bridge window; desktop not certified idle, no native MLX modules loaded.
  Ruff and source whitespace checks passed. No source changed between this
  acceptance and the explicit-path source commit.
- `VERIFICATION.json` binds the tested source bytes to `eb601af` and the raw
  acceptance log/XML hashes. The suite started on dirty `326b0a0`; its unchanged
  source became `eb601af`. That distinction is retained in `acceptance.json`,
  rather than replacing the recorded run-start commit after the fact.
- `counterexample.json` was subsequently produced directly at `eb601af`.
  Historical red checks are preserved: missing SAE module, missing clamp API,
  and the encoder-output aliasing defect. Intermediate passes are observations
  of those intermediate trees, not substitutes for final acceptance.

Reproduce the affected acceptance from this worktree, using the installed CPU
Torch environment and the pinned upstream reference:

```sh
PYTHONPATH="$PWD/src:$PWD" LLL_BACKEND=torch LLL_DEVICE=cpu \
  /path/to/python research/records/SAE-DECODER-INTERVENTION-2026-09-10/verify.py rerun \
  tests/test_sae_intervention.py tests/test_torch_capture.py tests/test_arch_torch.py \
  tests/torch/test_torch_jacobian.py tests/test_torch_upstream_imports.py \
  tests/test_upstream_ref.py tests/test_repository_rules.py
```

The evidence runner blocks native MLX imports and records CPU/box state and skip
counts. Raw pytest logs and XML are retained byte for byte, including whitespace
in failure tracebacks; source, documentation and JSON receive whitespace checks.
The existing graph-once timing regressions passed under eager and SDPA, but their
active-desktop timings are regression observations, not new idle-box benchmarks.

The source and method record are ready for the Chief's review and integration.
The review heartbeat remains disabled.
