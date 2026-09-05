# Context-length efficiency on the M4 Pro: two runs, one stopped

**Provenance.** Author: research division session `an-app-cd`. Date: 2026-09-05, evening.
Machine: MacBook Pro, Apple M4 Pro, 24 GiB physical, recommended working set 17.76 GiB read from
`mx.device_info()`. Model: `mlx-community/Qwen3.5-4B-MLX-4bit`, snapshot
`32f3e8ecf65426fc3306969496342d504bfa13f3`. Stack: mlx 0.32.2, mlx-lm 0.31.3, Python 3.13.
Runs: `ctxbench.py` (sweep, prefill chunk 2048) and `ctxmax2.py` (rebuild, chunk 512), both in
this directory as they ran, with `guard.sh` and `launch.sh`. Logs: `rebuild-run.log`,
`rebuild-guard.log`, `rebuild-launch.log`, `sweep-rows.json`. Chart: `curves.svg`.
Authority: test-run standing lift; concurrency rule observed, one model load at a time, verified
by the library test before each launch. No training, no evaluation, no model weights changed.
Prompts were random token identifiers: these runs measure time and memory only and say nothing
about whether the model uses the context.

## 1. The sweep, prefill chunk 2048

| tokens | prefill s | prefill tok/s | decode tok/s | peak GiB | share of working set |
|---|---|---|---|---|---|
| 4096 | 7.4 | 551.9 | 54.71 | 4.796 | 0.270 |
| 8192 | 14.8 | 553.9 | 65.41 | 5.194 | 0.292 |
| 16384 | 31.6 | 518.0 | 34.48 | 5.865 | 0.330 |
| 32768 | 74.6 | 439.5 | 25.29 | 7.379 | 0.415 |
| 65536 | 222.7 | 294.2 | 9.69 | 10.442 | 0.588 |
| 131072 | 667.5 | 196.3 | 5.02 | 16.692 | 0.940 |

**The 4096 row is contaminated and was re-measured.** The first length in any sweep pays kernel
compilation. Re-measured after a discarded warm-up: prefill 571.0 tok/s and decode **67.55**
against the 54.71 above, an under-reading of 19 percent in decode and 3 percent in prefill. Peak
memory was identical to three decimals, so memory was never affected. The corrected decode curve
is monotone; the original was not, and its non-monotonicity was the signal that found this.

**Corrected decode curve:** 67.6, 65.4, 34.5, 25.3, 9.7, 5.0 tok/s at the six lengths. Decode is
flat to 8192 and halves in the single step to 16384. That knee, not memory, is the operating
constraint.

## 2. The rebuild, prefill chunk 512, stopped at 98816 of 262144

Purpose: test whether the full native window fits at a smaller prefill chunk, which the sweep's
memory slope had been read as predicting. Progress lines every sixteen chunks carried elapsed,
tokens, peak and working-set share, so a stop at any point leaves a curve (R46).

Thirteen prefill rows are in `rebuild-run.log`; endpoints are 512 tokens at 3.32 GiB and 98816
tokens at 9.22 GiB, share 0.519, `crossed_working_set` false throughout.

**Stop.** SIGTERM at 21:37:00 by the Chief on the Director's report that the machine was
unusable. Launcher reaped **status 143**, recorded in `rebuild-launch.log` and as the final line
of `rebuild-run.log`. The guard did not fire: `rebuild-guard.log` shows one critical pressure
sample with swap free 678 MB, held for a second sample that never came, then the target exiting.
By the Director's ruling the data stands as collected and there is no re-run.

## 3. The three slopes, and what they answer

| slope | measured | cache plus per-chunk score term predicts |
|---|---|---|
| peak growth, chunk 2048 | 98.3 KiB/token | 96 |
| peak growth, chunk 512 | 62.9 KiB/token | 48 |
| cache alone, architectural | 32 KiB/token | — |

The cache term is 2 x 4 kv-heads x 256 head-dim x 2 B x 8 full-attention layers = 32 KiB/token;
only 8 of 32 layers keep a growing cache, the other 24 hold fixed recurrent state totalling
0.05 GiB. The per-chunk term is 16 query heads x chunk x 2 B, so 64 KiB/token at chunk 2048 and
16 at chunk 512.

**Answer to the native-window question: it does not fit, at either chunk.** Projecting each
measured slope to 262144 gives **27.8 GiB** at chunk 2048 and **19.0 GiB** at chunk 512, that is
1.57 and **1.07** of the working set. Reducing the chunk cuts the slope by a third and does not
close a gap that size.

**Two of my own claims are refuted by this and should not survive in the record.** First, that a
smaller chunk buys the native window: it does not, 19.0 GiB against 17.76 available. Second, a
mid-run reversal claiming the slope was chunk-independent and therefore came from cache
reallocation: the slope is plainly chunk-dependent, 98.3 against 62.9, and that claim came from
comparing noisy per-segment rates rather than the slope across a whole run. Peak memory is a
high-water mark and the cache grows in discrete steps, so a segment rate measures whether a
reallocation fell inside the window, not a rate. The chunk-2048 slope fits the model to 2
percent; the chunk-512 slope is 15 KiB/token above it and that residual is unexplained.

## 4. What this supports, and what it does not

Supported: an operating window of **16k to 32k**. At 32768 the whole thing costs 7.4 GiB of
17.76, prefill is 75 s and decode 25 tok/s. Past 16k each doubling roughly halves decode, so the
cost of a longer window is paid in the wall-clock of every future evaluation, not at a memory
ceiling. Current configuration is 2688 (`configs/agent_v2d_qwen35_4b.yaml`), so 32k is a
twelvefold increase inside every measured constraint.

Not supported: any claim about whether the model can **use** a longer context. These runs used
random tokens. The published figure for a 4B-class model is an effective ceiling near 64k, and it
is from a different model. Measuring ours is a separate experiment of the same shape as the
distance-curve work already specified.

Also measured, incidentally: peak memory during prefill is dominated by neither weights nor
cache at short contexts. At 4096 tokens 2.5 GiB of the 4.8 GiB peak is transient, most of it the
per-chunk logits tensor over a quarter-million-token vocabulary.

## 5. Rulings this record is filed under

R46 and its items (progress line inside long rows carrying the working-set share; share marks
degraded rather than kills; kernel pressure critical across two consecutive samples as the only
kill trigger; target identified by pid; launcher reaps and records exit status). R47: a run
projected above 0.6 of the working set runs only in a Director-declared window. R48: efficiency
probes run to 64k and no further. Both runs here predate R47 and R48; under them, the sweep's
131072 row and the whole of the rebuild would not have been run.

## Guards added at the records commit (Deputy, 2026-09-05)

`ctxbench.py`, `ctxmax2.py` and `launch.sh` refuse to run unless invoked with
`--i-am-a-record`. They were unguarded when handed over, and they load the model directly without
taking the model-run lock of issue #83, so committing them as they stood would have put two working
launchers into the tree — one of them a sibling of the script that ran orphaned for 57 minutes
tonight and made the machine unusable. Successors run as package entry points under R47 and R48, in
a window the Director declares. Nothing else in this directory was modified; the scripts are
otherwise byte-identical to the versions that produced the figures above.
