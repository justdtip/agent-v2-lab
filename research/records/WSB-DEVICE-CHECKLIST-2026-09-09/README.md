# WS-B on the device: the checklist, the number each gate must produce, and what a mismatch means

**Written on the laptop, to be executed on the rented GPU.** All subsequent runs use the device;
this box is for tests and corpus freezing. So this is not a plan, it is a list of gates with the
number each must produce, ordered cheapest and most diagnostic first, so that an hour which goes
wrong goes wrong early and for a nameable reason.

Every figure below is either **measured here**, and stated with what measured it and the box
state it was taken under, or **to be measured there**, and stated as unknown. A projection says
what it rests on. Nothing is asserted from arithmetic.

The tools carry the same distinction as fields rather than as prose: every number the kit writes
says `measured-here`, `laptop-basis` or `expected`, per cell rather than per row.

---

## 0. Before anything: seconds, no model

| check | expected | if it differs |
|---|---|---|
| `python scripts/acceptance_gates.py --records <dir>` with no `--model` | the environment block, then 0 of 7 with all seven listed as not runnable | the kit is not reading the corpus; nothing below can run |
| the determinism block, from `device.pin(attention="eager")` | `determinism: pinned`, TF32 off, `float32_matmul_precision: highest`, `attn_implementation: eager` | `pin()` ran after something touched the device, and every later number inherits it |
| the corpus, from the records alone | 5,245 emitted tokens over 15 episodes, 4,801 agentic | a different corpus; every laptop basis below is void |
| the forward-to-emission join | 5,245 agree, 0 disagree, 0 missing | **bookkeeping, not validation.** It asserts greedy decoding and one position convention, and nothing about any backend. It cannot fail on a correct reader, which is exactly why it catches a conflated position convention and nothing else does |
| the hard rule's coverage | 4,199 of 5,245 at P ≥ 0.99 (80.06%) | the confidence rows were not read; the gate below degrades to a threshold nobody came near |

---

## 1. The smoke pass: every gate once, on its smallest input

`scripts/acceptance_gates.py --records <dir> --smoke --results <store>`

Purpose: an hour that dies at minute fifty has touched every gate once and can say which of them
was never going to work. It is not a pass on the full input and the resume key says so, because
the selected episode labels are part of the key. A later full pass meets
`the record measured something else: gate input ... against ...` rather than silently inheriting.

| gate | smallest input | expected |
|---|---|---|
| 5, golden trajectories | `agentic-d2-calculate-0158`, 103 emissions | the bookkeeping join at 103 agree, 0 disagree |
| 6, golden lens reads | the same episode, one lens read | **unknown**: the producing side is not ported, so this reports unavailable and that is the correct output, not a failure |

---

## 2. The tolerance gate, in the order to run it

`scripts/tolerance_baseline.py --records <dir> --json <out>` over all fifteen episodes.

Teacher-forced: one causal forward per turn over prompt plus recorded emissions, read at every
deciding position, so a disagreement at one position cannot cascade into the next. Chunked at
2,048 with the final prompt token separate. Ranked in 256-row blocks.

**Free-running token-for-token agreement across backends is not a gate and is not attempted.**
One flipped argmax separates two trajectories completely; over thousands of tokens the
probability of zero flips is effectively zero. Exact reproduction is asked of one backend
against itself.

| step | must produce | laptop basis, and the box state it was taken under | a mismatch means |
|---|---|---|---|
| 1. `calculate-0158` alone, 103 emissions | flips only where precision puts them | **98 of 103 agree; 5 flips; 0 of 78 confident positions flipped and 5 of 25 unconfident ones did.** Idle box, one window, CPU bf16 against MLX 4-bit, 83.5 s | a flip at P ≥ 0.99 fails the run outright: quantisation cannot move an argmax that confident, so it is a mask, position, entry or norm defect |
| 2. the same episode a second time | identical flip positions | same on CPU | the backend is not deterministic; `pin()` did not take, or a kernel is non-deterministic. This is the CUDA-against-CUDA half of G-2 and costs 84 s on CPU |
| 3. the ten remaining short episodes | the same signature | **none: unrun.** The eleven-episode block was stopped by the Director's order at 10.5 minutes with zero complete | — |
| 4. the four long episodes, `update-0028` first, 2,607-position turn | the same signature **beyond the 1,024 window** | **none, and this is the point of the gate rather than its remainder.** These are the only episodes that exercise the sliding window; the mask-dispatch control on the tiny model bit only past the window | a signature that changes past the window is the mask dispatch, not precision |
| 5. peak memory | measured there | 7.88 GiB against 7.56 projected, on CPU, one window. Basis: 7.23 GiB text tensors from the snapshot header, 0.78 GiB vision discarded on load, 256 MiB ranking block, 80 MiB KV | the projection missed by 4.2% in the safe direction here; on the device the allocator differs and this figure is history, not a prediction |
| 6. gates 5 and 6 | PASS with their measured bands | the kit prints them | — |

**Cache strategies stay refused** until each reproduces the `none` trajectories byte for byte on
the device. `trim` and `snapshot` are built and tested against the real `DynamicCache`;
`make_turn_cache` refuses all three and its error names the ruling.

---

## 3. What the numbers rest on, stated once

| figure | measured on | box state |
|---|---|---|
| 98 of 103, 5 flips, 0 confident | CPU bf16, `google/gemma-3-4b-it` text-only, eager | idle, one window, 83.5 s |
| 7.88 GiB peak | the same run | idle |
| 4,199 of 5,245 at P ≥ 0.99 | the MLX records alone | no model |
| 94 of 94 forward partitions match | the MLX records alone | no model |
| 6 of 94 turns end on terminator 106 | the MLX records alone | no model |
| `aggregate_report-0167` unfinished at 630 s against a 667 s idle projection | CPU bf16 | **shared**, load 6.4, another seat's two processes |

---

## 4. Three findings that are about method rather than about the port

**The eleven-episode run lost everything, and the interrupt was not the cause.** Ten and a half
minutes of box time, 32 minutes of processor time, zero of eleven episodes recovered. The runner
accumulated every result and wrote once at the end, with stdout block-buffered because it was not
a terminal. `SIGINT` was clean and the interpreter flushed; there was nothing in the buffer.
Survivable on a laptop, not on a paid hour whose whole purpose is to say what failed and where.
Fixed: a row per episode, written and flushed as it completes, with the property as a test.

**The calibration error has a third direction.** Already recorded: measured on too small an
instance, and measured on the wrong artefact. The third is *measured under conditions the run
will not be in* — a 667-second idle projection against a run that had not finished in 630 seconds
while sharing the box. Every projection here therefore names the box state beside the number.

**A threshold result is not a claim without its base rate.** "0 flips at P ≥ 0.99" and "0 of 78
confident positions flipped" are the same fact, and only the second can be read; the first cannot
be told apart from a threshold nothing came near. The runner prints every flip with its recorded
probability and the episode's confident count. The two halves carry *different* provenance on the
device, because the agreement is re-measured there and the confident count is not, which is why
provenance is per cell.

---

## 5. What the kit guarantees, so the hour can rely on it

- **Resumable**, keyed on the working tree's content, the checkpoint digests, the device reading
  and the gate's own input. A one-line fix on the device keeps every record taken on that same
  edited tree. A refusal names every field that differs, not the first.
- **A result per gate, written and flushed as it completes.** An interrupted hour keeps what
  finished.
- **Only a pass resumes.** A failure is a thing to try again, not to inherit.
- **Stops at the first failing gate**, with what it saw beside what it expected.
- **A gate that has not run is never a pass.** `unavailable` names the workstream that owns it.
- **Refuses to load weights** unless this process holds the box window, and refuses a window
  whose holder would be a discarded shell.
- **Every number carries its kind.** A number whose kind nobody chose cannot be written.
