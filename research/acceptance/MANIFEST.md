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

## The first number not true by construction

`agentic-d2-calculate-0158`, teacher-forced, MLX 4-bit records against CPU bfloat16, one box
window, `cache_strategy: none`.

| | |
|---|---:|
| argmax agreement | 98 of 103 (0.9515) |
| flips | 5 |
| flips at P ≥ 0.99 | **0** |
| top-5 Jaccard, mean | 0.7018 |
| top-5 Jaccard, median and worst | 0.6667 and 0.4286 |
| gate | **PASS** |
| wall clock | 83.5 s for the episode, 96 s total |
| peak resident set | 7.88 GiB against a projected 7.56 and an R47 stop of 10.656 |

**The pass is evidence and not a threshold nobody came near.** In this episode 78 of the 103
emissions sit at P ≥ 0.99 and 25 do not. All five flips landed in the 25. If a flip were
independent of confidence, the chance of none reaching the confident bucket is
0.2427⁵ = 8.4 × 10⁻⁴. Put the other way: **0 of 78 confident positions flipped and 5 of 25
unconfident ones did**, which is what a precision difference predicts and not what a mask,
position, entry or norm defect would look like.

**A threshold result is not a claim without the base rate beside it.** "0 flips at P ≥ 0.99"
and "0 of 78 confident positions flipped" are the same fact, and only the second can be read.
The first is indistinguishable from a threshold nothing came near. The report therefore prints
every flip with its recorded probability, not only the gating ones, and the JSON carries the
episode's own count of confident positions. I had to derive that count by hand after the first
run, which is how the gap was found.

**The projection missed by 4.2% and in the safe direction**, 7.88 GiB measured against 7.56
projected. The basis was the snapshot's own header: 7.23 GiB of text tensors with the 0.78 GiB
vision tower discarded, plus a 256 MiB ranking block and 80 MiB of KV. Stating the basis is
what makes the 4.2% a number to learn from rather than a near miss.

**What it does not say.** One episode of fifteen, 103 emissions of 5,245, and the shortest in
the set — the same instance whose cheapness produced an earlier calibration error, chosen here
deliberately so the box could be released quickly. Jaccard at 0.70 means the top-5 sets differ
by more than the argmax does, which is expected between precisions and is reported rather than
gated. The remaining fourteen episodes are unrun.

## The eleven-episode run: stopped, and what it cost

Ordered stopped by the Director part-way, so that no long run ties up this box: subsequent
runs go to the rented device. **Zero of eleven episodes completed.** Ten and a half minutes of
box time, 32 minutes of processor time, and nothing recovered.

**Nothing was recovered because of a defect in this kit, not because of the interrupt.** The
runner accumulated every result and wrote once at the end, and its stdout was block-buffered
because it was not a terminal. So an interrupt at ten minutes produced no JSON and not one
episode line. The interrupt was clean — `SIGINT`, so the interpreter unwound and flushed — and
there was simply nothing in the buffer to flush.

That is survivable on a laptop and is not on a rented device, where an interrupted hour is an
hour paid for and this kit's entire purpose is to say what failed and where. Fixed: each
episode's row is written and flushed the moment it finishes, and progress prints unbuffered.
The property is now a test that fails against the old behaviour.

**One number does survive**, and it is about the box rather than the port:
`aggregate_report-0167`, projected at 667 s on an idle box, had not finished in 630 s while
SWE-2's two `gloo` processes shared the machine at load 6.4. Consistent with the sharing
inflation predicted before the run, and the reason tonight's projections would have used the
idle-box basis.

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

### A check whose answer I already knew, and got backwards — `bb4dba3`

The baseline runner must refuse to load 7.3 GiB of weights unless this process holds the box
window. My first draft asked whether a window **existed**. One did, held by a third seat, so
**the check passed precisely because somebody else was holding the machine**, and the script
would have loaded the model inside their run.

`blocking_window` is the right question, because it returns `None` both when nothing is open
and when the open one is ours; those two cases have to be separated explicitly. The runner now
refuses with the runlock's own text, and I watched it refuse against a live foreign window
rather than assuming it would.

**The only reason it surfaced is that it was run while another seat held the lock.** Run five
minutes after that window closed, it would have passed for the right-looking reason and stayed
inverted indefinitely. That is the shape this repository already has a record about: the check
that cannot fail teaches nothing, except about the one thing nothing else can catch, and only
if it is exercised in the state it exists for.

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
| `66d14af` | never taken clean | | | held throughout |
| `bb4dba3` | 2,265 | 14 | 0 | free |

**`bb4dba3` is the current clean run**, and it is the first that includes WS-A's torch view and
capture. `66d14af` never got a clean count: the window was held for the whole of its life and
the row says so rather than borrowing a neighbour's number. The count was taken by a job that
waited for the window rather than by hand, which took 990 seconds.

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
