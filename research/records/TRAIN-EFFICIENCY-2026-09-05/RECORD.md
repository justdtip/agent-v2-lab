# Training efficiency on Metal: eight literature searches and one measurement

**Provenance.** Author: research division session `an-app-cd`. Date: 2026-09-05, night.
Commissioned by the Director: a broad literature search on memory optimisation when training
with the Metal API, extended at his request to time optimisation, then a measurement.
Machine: MacBook Pro M4 Pro, **16 GPU cores** (confirmed locally), 24 GiB unified, working set
17.76 GiB, 273 GB/s. mlx 0.32.2, mlx-lm 0.31.3. Method: eight parallel read-only literature
agents, four on memory and four on time, plus one local measurement, `dispatchtest.py` with its
output `dispatchtest.json`, both in this directory. No model weights were loaded; the test
exercises the recurrence op directly on synthetic tensors.

**Compliance exception, declared rather than buried.** The width sweep's largest configuration
reached **17.27 GiB, 0.97 of the working set**, against R47's 0.6 threshold for a declared
window. I did not project the peak before running and should have; the largest case is eight
times our own state size carrying a backward pass. The run completed and the machine was not
disturbed, but it breached the rule and the rule was right.

## 1. The measurement, and it refutes the prediction it was designed to test

The prediction from the literature was that our recurrence is dispatch-bound, so that holding
sequence length fixed and doubling width would leave step time roughly flat. Tested on the exact
op the training path uses, `gated_delta_ops`, at our own shapes (32 value heads, 128 key and
value dimensions).

**Scaling in sequence length, width fixed.**

| T | forward ms | forward+backward ms | forward µs/token |
|---|---|---|---|
| 125 | 9.2 | 43.4 | 73.7 |
| 250 | 18.0 | 119.5 | 71.9 |
| 500 | 35.1 | 368.4 | 70.2 |
| 1000 | 71.4 | 1145.7 | 71.4 |

**The forward is exactly linear** and its per-token cost is constant at about 71 µs, which is the
signature of a per-token dispatch chain. **The backward is not linear.** Each doubling of T
multiplies it by 2.75, 3.08, 3.11, a fitted exponent of **1.57**. A correct implementation should
be linear in T. Extrapolated to our configured 2688 tokens, one layer's forward and backward is
**5.4 s**, and we have 24 such layers.

**Scaling in width, T fixed at 500.**

| relative state size | forward ms | forward+backward ms |
|---|---|---|
| 0.062 | 14.5 | 76.6 |
| 0.125 | 15.4 | 103.9 |
| 0.5 | 23.8 | 196.9 |
| **1.0 (ours)** | **35.2** | **360.2** |
| 4.0 | 140.1 | 969.4 |
| 8.0 | 335.5 | 3005.6 |

At the smallest widths the forward is flat, doubling the work for six percent more time, which is
dispatch-bound as predicted. But **our own width is past that regime**: from 1.0 to 4.0 the
forward scales linearly with work. Taking the small-width plateau as the dispatch floor, launch
overhead is about **41 percent of the forward and 21 percent of the forward-plus-backward** at our
shapes.

**So the diagnosis is a mixture, not the clean answer the literature suggested.** A fused kernel
removes the floor, roughly a fifth of the recurrence's cost. The superlinear backward is the
larger and stranger term, and no source predicted it.

## 2. What the eight searches establish

**Nothing exists to adopt.** Every fast gated-delta implementation across five projects is Triton,
and Triton has no Metal backend. There is no Metal or MLX chunkwise gated-delta kernel with a
backward pass anywhere. The only trainable attention kernel on Metal is a softmax one, a porting
template rather than the algorithm.

**Our chunkwise implementation is the correct form.** A naive materialised chunkwise delta rule is
a factor of the chunk size *worse* than sequential; the representation using Householder products
through a triangular solve is what restores O(LCd + Ld²). Ours builds the Gram matrix on the strict
causal triangle and inverts it as a unit lower triangular system, which is that form
(`training/gated_delta_chunkwise.py`). Recomputing states in the backward, which it also does, is
what every source paper adopts as default.

**The hardware has no matrix datapath.** Matrix multiply on M4 executes on the same shader cores
as everything else. This matters more than it appears: the literature's case for chunked matmul
over a parallel scan rests on datacentre GPUs where matrix units run about fifteen times faster
than general compute. That premise is absent here, so the argument against a scan is much weaker
on our hardware than the papers imply. Peak is about 13.8 TFLOP/s fp16, with a realistic matmul
ceiling near 6.3.

**Parameter-efficient tuning does not reduce activation memory.** LoRA's saving is optimizer state
and gradients; it still retains the full input activations. Once on LoRA over a quantized base,
activations dominate, which is exactly where we are. The quantized base may also be costing time
rather than saving it: reported twenty to thirty percent per step from dequantisation, measured
with kernels Metal does not have.

**A whole category is inapplicable.** Anything that moves bytes off the accelerator, and all
multi-device sharding. On unified memory the destination is the same pool.

**Our fp32 recurrent state is validated by a failure, not caution.** A shipped product wrote that
state back in bfloat16 on this model family and it compounded into degenerate repetition.

**Reference points.** On comparable Apple hardware the same model trains at roughly 115 tokens per
second; ours is 21.7. Roughly fivefold slow, not the thousandfold that a datacentre comparison
suggests. The decisive ambiguity is whether our 46 s covers 997 real tokens or 10,752 padded
slots, and nobody has measured it.

**The closest precedent for a fix.** Replacing a sequential per-token scan in this framework, for
a state-space model, moved prefill from 293 to 5497 tokens per second. Decode, which has no loop
to remove, moved three percent. Our measurement suggests we would land nearer the smaller end of
that range than the larger, because we are only partly dispatch-bound.

## 3. What I would do next, in order

1. **Count padded slots per step.** One number, settles whether we are five times slow or one and
   a half. Everything else is priced against it.
2. **Explain the superlinear backward.** An exponent of 1.57 where the algebra says 1.0 is the
   largest single anomaly and it is ours, not the framework's, until shown otherwise. It should be
   reproduced against the chunkwise path, which may not share it.
3. **Only then consider a fused kernel.** It buys the dispatch floor, about a fifth, and that is
   worth having but is not the main term.
4. **Test the quantized base.** Half-precision LoRA against 4-bit LoRA, same rows, same seed. The
   literature says we may be paying for quantisation in time as well as gaining it in memory.

Hardware is the wrong lever until the above is done. The newer chip's accelerators shorten the
inside of a kernel, and roughly a fifth of our cost is between kernels.
