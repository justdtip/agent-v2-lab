# What passed on CPU, gate by gate, at a commit

Plan §12.1 requires this file: every workstream validates on CPU torch in float32 against the
MLX golden records, and what passes is recorded here at a commit, so that a failure on the
rented device is attributable to the device change and to nothing else.

**Nothing in this file is a CUDA result.** Nothing here has touched a GPU or a checkpoint.

## Environment these results were taken in

| | |
|---|---|
| torch | 2.14.0, pinned in `pyproject.toml` |
| platform | macOS 26.6.2, arm64 |
| CUDA available | no |
| MPS available | yes, unused |
| default dtype | float32 |
| determinism | `device.pin(attention="eager")`, called by the kit before the first gate, printing what `pin` read back rather than what it was asked for |

`device.py` owns the policy and this workstream holds none of its own. The kit calls `pin`
first and prints the reading, so a run whose pin was refused says so rather than looking
configured. `CUBLAS_WORKSPACE_CONFIG` is read by cuBLAS at first use, so `pin` refuses when
CUDA is already initialised without it; that refusal reaching the operator is the point of
pinning at the top of the kit. `pin_torch_determinism` remains as a once-per-process guard on
the generation loop so a loop entered without the kit cannot take a number unpinned.

## Gate status

Run: `python scripts/acceptance_gates.py --records <stage-two records> --keep-going`

| gate | name | owner | status at `38e8504` |
|---:|---|---|---|
| 1 | structural discovery of the decoder | WS-A | unavailable |
| 2 | `residual_source_agreement` at 64 and 1,400 | WS-A | unavailable |
| 3 | layer-34 identity against the emitted token | WS-A | unavailable |
| 4 | readout gate with both negative controls | WS-A | unavailable |
| 5 | golden trajectories | WS-B | **unavailable** — records read and self-consistent, nothing regenerated |
| 6 | golden lens reads | WS-B | unavailable — comparison side built, producing side is WS-A's |
| 7 | lens un-port refitted and compared per layer | WS-D | unavailable |

**0 of 7 passed.** Unavailable is not a pass and is never rendered as one.

## Facts measured from the records, which need no backend

Each of these is a real number about the corpus. None of them is evidence about any port.

| fact | value |
|---|---|
| emissions whose recorded forward argmax matches | 5,245 of 5,245 |
| turns whose forward partition matches the predicted one | 94 of 94 |
| recorded rank at layer 34, horizon 1 | 1 on all 5,245 emissions |
| emissions recorded at P ≥ 0.99 | 4,199 of 5,245 (80.06%) |
| median recorded probability at layer 34 | 0.999998 |
| turns ending on the terminator 106 | 6 of 94 |

**The forward-to-emission join** is bookkeeping and not validation. It asserts that decoding
was greedy, that the record's forward offsets and emitted positions share one convention, and
that this reader joins them the way the writer wrote them. It asserts nothing about any
backend and cannot fail on a correct reader. It is here because it is the only check in the
chain that catches a position convention conflated across two coordinate systems.

**The partition check is not bookkeeping**, and it is the one that changed the code. Against
every recorded turn, the forward offsets and widths the torch loop would produce match the
recorded ones exactly: native prefill chunks of 2,048 with the final prompt token separate,
then one single-token lookahead forward per emission whose argmax is unused on the last one.
A single-chunk prefill produces the same tokens and different forward rows, and
`ForwardLedger.validate` asserts the partition on every forward.

**The terminator finding sized a real defect.** Token 106 is emitted on six of the ninety-four
turns and is the last token of each. The loop had no EOS handling at all, so those six turns
would have run to the 200-token cap. The ids come from the model config's set `[1, 106]`;
`config_eos_ids` refuses to fall back to the tokenizer's single id, which is not that set.

**The saturation figure is a warning, not a result.** A median of 0.999998 means a mean over
this distribution says nothing, which is why the hard rule is a threshold and the rest is
reported rather than averaged.

## What is executed, and what it does and does not assert

### The harness's comparison machinery — `897fba4`

Exercised by a generator that replays the record under test, so its "reproduced" and 1.0000
readout columns **pass by construction**. What makes that run evidence rather than nothing:

- `--self-test` plants a token flip at three indices and a turn one token short, and requires
  the divergence at the exact planted index. All controls report correctly.
- Bit-identity is reported as "not assessed" rather than as a count, because the replay
  generator supplies no logit digests and the record stores only a SHA-256 of each logit
  tensor and never the tensor itself.

### The torch generation loop — `907db88`, corrected at `15cc202`

Proved against a stub view supplying `make_cache` and the model. Both backends stop at the
same index on four scripted piece streams, because both feed one `_consume_stream` rather than
two copies of the stop rule.

Mutation controls, all caught: replacing the incremental piece with the empty string fails six
tests; a single-chunk prefill fails four; removing the EOS stop fails the EOS test.

### The three outcome columns — `cf5b76e`

Cycle-aware loop metric beside the original one, truncation as a distinct outcome, and
`contains_expected` beside `success`. Tested against the shapes that fooled the old columns.

### The measured band — `08e9b36`

`readout_tolerance` refuses an empty sample, a blank basis, and a projection without a stated
reason. A test pins that the 0.999 quantile of a thousand samples is the 999th and not the
largest. **No band has been measured**; that needs both backends live.

### The tolerance half of G-2, and its runner

Teacher-forced agreement, the P ≥ 0.99 hard rule, the divergence profile with a floor, and
top-k Jaccard, with `run_tolerance` driving all of them from one pass. Built and unit-tested;
none has been run against a port, because there is no port to run against.

**Teacher forcing is one forward per turn, not one per token.** Feeding the whole recorded
sequence and reading the argmax at each position gives exactly the prediction that position
would have made with the recorded prefix ahead of it, because attention is causal. The runner
checks the returned row count against the sequence length rather than trusting it: an
off-by-one there shifts every comparison by one position and still produces a plausible
agreement rate, which is the failure this programme has already paid for once.

Only two things gate. A flip where the recorded probability was at least 0.99 fails the run,
and an episode that parts company below the divergence floor fails it. Everything else is
reported. Jaccard is over sets, so producing the right token plus extras is not a perfect
score; a test pins that, so it is not later "fixed" into a top-1 agreement rate wearing a
Jaccard's name.

### Torch cache reuse, and two findings from building it against the real class — `66d14af`

`trim` and `snapshot` on torch, built against `transformers` 5.16.1's own `DynamicCache` on a
four-layer Gemma 3 config, because every rule they obey is a property of that class. `history`
is not implemented and the module says why. **`make_turn_cache` still refuses all three under
torch**; they are not wired in.

**Arming rollback late does not raise, it corrupts.** Never arming makes `crop` raise, which is
safe. Arming after the window has filled succeeds. Rewinding from absolute 12 back to 6, with
each token's key set to its own index:

| | reported offset | stored keys |
|---|---:|---|
| armed at construction | 6 | 0, 1, 2, 3, 4, 5 |
| armed after 12 tokens | 6 | 5 |

Both report offset 6; one holds six tokens and the other holds one, and would attend over a
five-token hole with nothing raising. `enable_rollback` refuses a cache that has already
advanced.

**Arming rollback makes a sliding layer unbounded**, which is the measured need the deferral of
the reuse strategies was waiting on. Unarmed, a sliding layer stores `window - 1` entries at any
context length; armed, it stores everything, because rolling back needs the past. On Gemma 3 4B
that is 29 of 34 layers, and at 2,749 positions against a 1,024 window the whole KV cache goes
to **2.15x**. Below the window it costs nothing, so a short calibration misses it entirely.

Read off the loaded config, `head_dim` is the class default of **256** because the checkpoint's
own config leaves it null. Deriving it from hidden size over attention heads gives 320 and
figures 25% too large, which is why the first version of this section quoted the ratio alone.

| positions | bounded | rollback-armed | ratio |
|---:|---:|---:|---:|
| 1,024 | 142.6 MB | 142.6 MB | 1.000x |
| 2,749 | 177.9 MB | 382.8 MB | 2.152x |

**And the ruling is that memory does not decide this.** 2.15x of 178 MB is inside the laptop's
cap and nothing on an 80 GB card. **The deciding question is fidelity**: each strategy is an arm
of the same acceptance gate as `none` and must reproduce the `none` trajectories byte for byte
within one backend. A strategy that changes a single token is a defect and not a speed setting,
because a rewindable sliding cache is exactly where the stored keys and the attention mask can
part company.

That gate cannot run until the tolerance runner has a `none` baseline against the real view, so
`make_turn_cache` keeps refusing until then.

## Corrections taken from the survey, and one sent back

**Taken.** The readout is no longer the producer: the loop generates from the model's own head
and `native_readout` is the thing compared against it, never substituted for it. The prefill
partition is the native one. EOS comes from the config's id set. Determinism is pinned.

The three cache reuse strategies were briefly moved into the first cut and are **deferred
again**, on the Chief's ruling: a cache strategy changes memory and time, not the tokens,
unless it changes which keys are attended, which is exactly why a sliding cache must not
promise rollback. They are built when a measured need on the device says so. `make_turn_cache`
refuses them under torch meanwhile, because an MLX cache handed to a torch model attends to the
wrong keys rather than raising.

**Sent back, and accepted.** *Per-step KL percentiles against the recorded final-layer
distributions cannot be computed.* **No recorded distribution exists at layer 34.** The records
store the top-k token ids there and the probability of the emitted token, and never a
distribution; a divergence needs both sides in full. `symmetric_kl` is therefore
live-against-live only and raises when handed anything whose mass does not sum to one, with the
message saying that a top-k slice is not a distribution. The sentence is in this file at the
Chief's request so the statistic is not proposed again.

The three that survive are teacher-forced argmax agreement under the P ≥ 0.99 rule, the floored
divergence profile, and top-k Jaccard, which survives precisely because it needs ids only.

## A run under a box window is not a coverage statement

The suite prints a green summary line whether or not it exercised the files that can reach the
model library. When another seat holds the box window, those files are collected and skipped,
and the count that moves is the skip count, which is the last number anyone reads.

Concretely, on this tree and on the same commit:

| run | passed | skipped |
|---|---:|---:|
| window free | 2,161 | 14 |
| WS-A's window held | 1,082 | 1,110 |

Both say "passed" and neither says "failed". The second exercised none of the model-facing
code and is not evidence about it.

**So every test count in this file names the window state it was taken under, and a count
taken under a held window is never carried forward as though it were clean.** Three runs during
this workstream were discarded on that ground and re-run rather than reported. WS-A's window
will be open often over the next day, which makes this the difference between a manifest that
records coverage and one that records the appearance of it.

## Test suite

| commit | passed | skipped | failed | window |
|---|---:|---:|---:|---|
| `41dc6b4` | 2,161 | 14 | 0 | free |
| `66d14af` | not yet taken clean | | | WS-A's held throughout |

**The last clean full-suite run is `41dc6b4`.** `66d14af` adds the torch cache module and its
tests; those tests and the workstream's own files pass, and no clean full-suite count exists
for it yet because WS-A's window has been open since it landed. That is stated here rather
than carrying the `41dc6b4` number forward under a later commit's name.

Two failures reported earlier in this workstream are gone, both fixed on the main line and
brought across by merges: the registry name list that `gemma3-4b-cuda-bf16` made a sixth
entry of, and the two `test_repository_rules` guards.

## Unexecuted, and what each one needs

| item | needs |
|---|---|
| gate 5's reproduction arm | WS-A's torch architecture view |
| gate 6 entirely | WS-A's view and the hosted lens read path |
| every G-2 tolerance statistic | a produced side from a loaded backend |
| the readout band's actual value | both backends live on the same prefix |
| `trim` and `snapshot` wired into `make_turn_cache` on torch | the `none` baseline from the tolerance runner against the real view. Both strategies are built and tested against the real `DynamicCache` at `66d14af`; each then runs as its own arm of the same gate and must reproduce the `none` trajectories byte for byte within one backend. `make_turn_cache` refuses all three meanwhile. |
| `history` on torch | not implemented. Qwen 3.5's strategy, carrying snapshot files and a generation-model proxy, and the golden records never exercise it. |
| `attn_implementation="eager"` on the replay model | WS-A's loading path |

## The seam, as ruled

`logits = generation_model(ids, cache)`, where `generation_model` is what WS-A's capture
context returns. The wrapper owns the backend difference: on HF the model returns
`CausalLMOutputWithPast` and takes `past_key_values`, and the wrapper unwraps `.logits`. The
loop never calls `native_readout`; the capture session does, in the gate, comparing the
readout against the model's own distribution.
