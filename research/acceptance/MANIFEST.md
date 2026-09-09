# What passed on CPU, gate by gate, at a commit

Plan §12.1 requires this file: every workstream validates on CPU torch in float32 against the
MLX golden records, and what passes is recorded here at a commit, so that a failure on the
rented device is attributable to the device change and to nothing else.

**Nothing in this file is a CUDA result.** Nothing here has touched a GPU or a checkpoint.

## Environment these results were taken in

| | |
|---|---|
| torch | 2.14.0 |
| platform | macOS 26.6.2, arm64 |
| CUDA available | no |
| MPS available | yes, unused |
| default dtype | float32 |
| determinism | **unpinned** — `device.py` does not exist yet (WS-A/WS-E), so deterministic algorithms are not enabled and no attention kernel is pinned |

The determinism row is the one to read twice. Every number below was taken without the
settings the plan requires, which is acceptable for record-reading checks and is **not**
acceptable for any number produced by a model forward. No such number exists here yet.

## Gate status

Run: `python scripts/acceptance_gates.py --records <stage-two records> --keep-going`

| gate | name | owner | status at `08e9b36` |
|---:|---|---|---|
| 1 | structural discovery of the decoder | WS-A | unavailable |
| 2 | `residual_source_agreement` at 64 and 1,400 | WS-A | unavailable |
| 3 | layer-34 identity against the emitted token | WS-A | unavailable |
| 4 | readout gate with both negative controls | WS-A | unavailable |
| 5 | golden trajectories | WS-B | **unavailable** — records read and self-consistent, nothing regenerated |
| 6 | golden lens reads | WS-B | unavailable — comparison side built, producing side is WS-A's |
| 7 | lens un-port refitted and compared per layer | WS-D | unavailable |

**0 of 7 passed.** Unavailable is not a pass and is never rendered as one.

## What is executed, and what it does and does not assert

### The record's own forward-to-emission join — `897fba4`

The forward at offset *p* predicts position *p+1*, so every emitted token must equal that
forward's last argmax.

| | |
|---|---|
| emissions checked | 5,245 |
| agree | 5,245 |
| disagree | 0 |
| missing forwards | 0 |
| episodes | 15 |
| agentic subset | 4,801 |

The agentic subset matches the plan's own figure for gate 6 independently, so the reader and
the plan agree on what the corpus is.

**This is bookkeeping and not validation.** It asserts that decoding was greedy, that the
record's forward offsets and emitted positions share one convention, and that this reader
joins them the way the writer wrote them. It asserts nothing about any backend and cannot fail
on a correct reader. It is here because it is the only check in the chain that catches a
position convention conflated across two coordinate systems, which is the error class that
produced this programme's worst mistake.

### The harness's comparison machinery — `897fba4`

Exercised by a generator that replays the record under test, so its "reproduced" and 1.0000
readout columns **pass by construction**. What makes that run evidence rather than nothing:

- `--self-test` plants a token flip at index 0, index 3 and the final index, plus a turn one
  token short, and requires the divergence to be reported at the exact planted index. All
  controls report correctly.
- Bit-identity is reported as "not assessed" rather than as a count, because the replay
  generator supplies no logit digests and the record stores only a SHA-256 of each logit
  tensor and never the tensor itself.

### The torch generation loop — `907db88`

Proved against a stub view supplying exactly `make_cache`, `native_readout` and the model.

- Both backends stop at the same index on four scripted piece streams, because both feed one
  `_consume_stream` rather than two copies of the stop rule.
- The forward widths are asserted as `[len(prompt), 1, 1, …]`, so a quadratic re-feed that
  still produced the right tokens cannot pass.
- Mutation control: replacing the loop's incremental piece with the empty string fails six
  tests, including three of the four equivalence cases. The fourth contains no fence, so the
  completeness gate cannot fire in it; that insensitivity is expected and is not a gap.

### The three outcome columns — `cf5b76e`

Cycle-aware loop metric beside the original one, truncation as a distinct outcome, and
`contains_expected` beside `success`. Tested against the shapes that fooled the old columns,
including the two-cycle that reads `longest_identical_run = 1` while the trajectory loops.

### The measured band — `08e9b36`

`readout_tolerance` refuses an empty sample, a blank basis, and a projection without a stated
reason. A test pins that the 0.999 quantile of a thousand samples is the 999th and not the
largest, so a band cannot quietly become the maximum wearing a quantile's name.

**No band has been measured.** The band the readout gate will use is the CPU-float32 against
MLX-4-bit difference on the golden episodes, and that requires both backends live.

## Test suite

| | at `08e9b36` |
|---|---|
| passed | 2,085 |
| skipped | 14 |
| failed | 2 |

Both failures are `tests/test_repository_rules.py::test_every_records_script_that_reaches_the_model_carries_a_refusal_guard`
and `::test_every_guarded_records_script_actually_refuses_when_run`. They fail identically at
clean `9d68c27` with every WS-B change stashed, so they are the branch's and not this
workstream's. Cause: `research/records/GEMMA3-REGRESSION-2026-09-08/run-end.json` is committed,
so that script's refusal guard hits `FileExistsError` before it can refuse.

`ruff check` also fails at `9d68c27` with one import-sort error each in
`pipeline/runner.py` and `pipeline/evaluate.py`, both predating this work. They are left alone
rather than reordered in files WS-A and WS-E are about to edit.

## Unexecuted, and what each one needs

| item | needs |
|---|---|
| gate 5's reproduction arm | WS-A's torch architecture view |
| gate 6 entirely | WS-A's view and the hosted lens read path |
| `golden_trajectories.torch_generator` | the same view; it is wired and marked unexecuted at its definition |
| the readout band's actual value | both backends live on the same prefix |
| `trim`, `snapshot`, `history` caches on torch | deferred by the order; `make_turn_cache` raises rather than falling back |

## The one assumption a stub cannot check

`torch_greedy_stream` calls `model(tokens, cache=cache)` for hidden states and then
`view.native_readout(hidden)`. The readout was chosen over the model's own head so that a
generated token and a captured one come from one readout by construction, which is how the
record's argmax rows were written. If WS-A's view exposes a different call shape, that is one
function to change and the rest of the loop is unaffected.
