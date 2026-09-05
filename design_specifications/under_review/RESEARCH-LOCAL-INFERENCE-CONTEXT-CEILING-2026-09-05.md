To the Research Director, from the research division session `an-app-cd`. 2026-09-05.
Subject: preliminary research on how far local context length can be pushed on this machine.
Read-only. Three parallel literature agents; all machine and code figures measured here. No model
was loaded and no run was made.

## The short answer, which is not the answer the question expects

**Memory is not the constraint, and it is not close.** The full native window of 262144 tokens
costs 10.87 GiB all in, against a measured 17.76 GiB budget on this machine. Nothing needs to be
quantized, evicted or tuned to fit it.

**The constraint is what a 4B-class model can actually use, which is about 64K.** Everything
below follows from that inversion. We are currently configured at 2688.

## 1. What the model costs, measured from its own code

`make_cache` in mlx-lm's `qwen3_5.py:305` gives a growing `KVCache` only to the eight
full-attention layers and a fixed-size `ArraysCache` to the other twenty-four. The recurrent
state does not grow with context.

| quantity | value |
|---|---|
| growing layers | 8 of 32, every fourth |
| KV per token | 2 x 4 kv-heads x 256 head-dim x 2 B x 8 layers = **32 KiB** |
| constant recurrent state, all 24 linear layers | 0.05 GiB |
| weights, on disk | 2.83 GiB |
| KV at 262144 tokens | 8.00 GiB |
| **total at the full native window** | **10.87 GiB** |

A dense model of the same size would pay four times the KV, and this memo would reach the
opposite conclusion.

## 2. What this machine allows, measured not assumed

| quantity | value |
|---|---|
| recommended max working set | 17.76 GiB |
| physical memory | 24.00 GiB |
| largest single buffer | 13.32 GiB |
| `iogpu.wired_limit_mb` | 0, meaning default |

That is 74 percent of physical memory. Apple's published tiering implies about 66 percent at this
memory size, and a community report for a 24 GB machine gives 16 GiB, so the measurement is worth
about two gigabytes over either estimate. The single-buffer cap is not binding: the largest
per-layer array at the full window is about 512 MiB.

Raising the wired limit is available and not needed. It is also not a safety boundary: neither
the sysctl nor MLX's own limits prevented the kernel panics reported against them, since wired
pages cannot be reclaimed and the memory pressure system never fires. Bounding the cache is the
only real control, and we do not need to.

## 3. What actually binds, in order

**(a) Model capacity, about 64K.** RULER scores Qwen3-4B at 94.0 at 64K and 85.1 at 128K against
the benchmark's 85.6 threshold, so a 4B sits at the bar around 128K and comfortably above it at
64K. Effective context tracks model size closely: 0.6B and 1.7B are exhausted by about 32K, 8B
reaches 85.6, 14B reaches 90.6. This is capacity, not memory, and it is the number that should
set the target. It is soft — long-context distillation lifted Qwen3-4B's 128K score from 65.8 to
78.5 — but it is not soft by a factor of four.

**(b) The advertised-versus-effective gap, which is large.** RULER finds almost all models fall
below its threshold before their claimed length. NoLiMa, which strips lexical overlap between
question and needle, is harsher still: models advertising 128K hold 85 percent of baseline only
to 8K, 4K, or in some cases 1K. HELMET adds that needle-in-a-haystack scores do not predict
downstream performance, so any single-needle result should be discounted.

**(c) An mlx-lm prefill ceiling that may bite before any of this.** A 35B MoE was reported to
run out of memory at about 176K during prefill on a 128 GB machine, from transient Metal command
buffers rather than from the cache, while llama.cpp completed 262K on the same box. If that
generalises, the native window is unreachable in this stack at any memory size. It is a report
about a much larger model and it needs our own test before we believe it.

**(d) Time, quadratic in the eight attention layers.** Decode throughput on a larger machine
falls from 31 tokens per second at 1K to 8.5 at 128K, with the knee at 16K to 32K; scale down
about threefold for this machine's memory bandwidth. Prefill is superlinear, roughly a fivefold
time cost for a fourfold token increase between 8K and 32K. No M4 Pro long-context curve exists
publicly, so ours would be the measurement.

## 4. Levers, with the two that matter separated from the rest

**Cache quantization works here, contrary to the first reading.** The warning in circulation is
that hybrid models wrap their caches in a `CacheList` that lacks the `to_quantized` hook, so
`--kv-bits` silently does nothing. Checked against the installed source: our model builds a flat
list, not a `CacheList`, and `maybe_quantize_kv_cache` tests each entry with `hasattr`
(`generate.py:303`), so the eight growing caches quantize and the twenty-four constant ones are
skipped. That is the behaviour we want. Keep it at 8-bit: keys carry per-channel outliers and are
the sensitive half, 4-bit costs real accuracy, and text beyond 4K is more sensitive than text
below it. Do not quantize the recurrent state on full-attention evidence; linear attention is
reported to be markedly more precision-sensitive.

**Prompt caching is the large win and it is partly closed to us.** Whole-prompt cache reuse cuts
time-to-first-token by one to two orders of magnitude on this exact hardware class. Incremental
prefix reuse is broken for Qwen3.5-class models, because the recurrent state cannot be split at
an arbitrary boundary, which is the same property EXP-002 is about. Whole-prompt caching works;
prefix extension does not.

**Not available:** `--max-kv-size`. `make_prompt_cache` defers to the model's own `make_cache`
whenever one exists, and its docstring says the rotating path applies only when the model has
none. Ours has one, so the flag is silently ignored. Rotation is not a lever we have.

**Not needed:** YaRN. It is required only past 262144, and the vendor advises against enabling it
when average context is at or below 32768 because it degrades short-context performance. At any
target we would sensibly pick, leave it off.

## 5. What I would do next, in order

1. **Raise the configured window from 2688 toward 32K**, which is inside every constraint above
   with large margins and roughly a twelvefold increase. This costs 1 GiB of cache.
2. **Measure the prefill and decode curve on this machine** at 4K, 8K, 16K, 32K, 64K. No public
   M4 Pro long-context curve exists, the estimates above are scaled from other hardware, and the
   knee is the number that decides the operating point. This loads the model and therefore needs
   a per-run lift under the current gate.
3. **Test the 176K prefill ceiling** on our own model, since a negative result would cap the whole
   programme and a positive one removes a worry. Same lift.
4. **Verify usable context with a recall probe rather than a single needle**, given HELMET's
   finding. We have the instrument for this already.

**One thing worth saying plainly about the hybrid.** The vendor's own table has its hybrid beating
full attention on RULER at every length through 256K and losing at 1M. Two labs have since walked
back hybrid designs, MiniMax citing multi-hop reasoning and agent tasks specifically, which is our
workload rather than an incidental one. No matched-size, matched-data comparison exists at scale.
The architecture that makes our memory answer comfortable is the same one under active doubt for
the tasks we care about, and preliminary research should not bury that.

— research division session `an-app-cd`
