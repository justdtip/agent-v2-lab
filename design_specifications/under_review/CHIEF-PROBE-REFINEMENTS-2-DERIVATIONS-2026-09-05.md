# Chief's refinements, part 2: what the architecture lets us compute instead of intervene (2026-09-05)

For the Director's discussion. Held under the veto; nothing here is ratified or dispatched.
Architecture facts used (all verified in the code this week): Qwen3.5-4B has 24 Gated DeltaNet
blocks and 8 full-attention blocks; the DeltaNet step per head is `S_t = g_t S_{t-1}`,
`S_t += β_t (v_t − S_t k_t) k_tᵀ`, `y_t = S_t q_t`, with a scalar gate `g_t ∈ (0,1)` per head
(`compute_g` broadcasts `A_log`/`dt_bias` over `[B, T, Hv]`), keys RMS-normalised and scaled
(`qwen3_5.py:180`), state `[Hv, Dv, Dk]` in float32; the recurrent cache is untrimmable; the
runner re-prefills the windowed text each turn. The chunkwise algebra in
`training/gated_delta_chunkwise.py` is the derivation below with different bookkeeping.

## 1. The recurrent read is a linear combination of past values with computable weights

Substituting the update into itself gives the linear recurrence
`S_t = g_t S_{t−1} (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ`. Unrolling from an initial state `S_0`
(zero at the start of a prefill):

    S_t = S_0 Π_{r=1..t} M_r + Σ_{s=1..t} β_s v_s k_sᵀ Π_{r=s+1..t} M_r,   M_r = g_r (I − β_r k_r k_rᵀ).

Reading with `q_t`:

    y_t = Σ_{s≤t} α_{t,s} v_s,   with   α_{t,s} = β_s · k_sᵀ [Π_{r=s+1..t} M_r] q_t   (plus the S_0 term).

So every DeltaNet head is, exactly, a **signed, unnormalised attention** over the past with
weights `α_{t,s}` determined by gates, keys and the query. The `α` are what the WY form
computes: `α_{t,·}` is the row of `(D ⊙ Q Kᵀ)(I + A)^{-1} diag(β)` in the chunkwise notation,
so the machinery to compute them for a whole sequence exists and is tested (`_unit_lower_inverse`
and friends). Cost is one `T × T` matrix per head per layer, fine at `T ≈ 2.7k`.

**What this buys the probes.** At a decision position `t`, the fraction of the recurrent read
that comes from a span `Σ` (the listing observation) is `m_Σ(t) = Σ_{s∈Σ} |α_{t,s}| ‖v_s‖ / Σ_s |α_{t,s}| ‖v_s‖`,
per head and layer, from a **single forward with the cache**, with no intervention. The
attention blocks give the same quantity directly from their attention weights. Together they
decompose the decision's retrieval of the listing into its attention part and its recurrent
part, layer by layer, observationally.

**And it buys the horizon without a grid.** Within one trajectory every earlier token `s` sits
at its own realised distance `t − s`, so one forward yields `|α_{t,s}|` at every distance at
once. Pooling over trajectories gives the recurrent read-weight profile against distance, per
head, from the same points EXP-001 already uses: 42 forwards, not 252 across seven bands.
EXP-003's seven-level grid was designed for an intervention whose output is one number per
run; the exact weights make the intervention unnecessary for the recurrent path.

## 2. The horizon is bounded by two products, both measurable per head from real text

Factor `M_r = g_r (I − β_r k_r k_rᵀ)`. Along the direction of an old key `k_s`, a later step
scales the retained write by `g_r (1 − β_r (k_r·k_s)²)` to first order (exact when `k_r ∥ k_s`,
and the orthogonal component is untouched). Over a gap of `Δ` tokens:

    retained(Δ) ≈ Π_{r} g_r · Π_{r} (1 − β_r ρ_{r,s}²),   ρ_{r,s} = k_r·k_s  (keys unit-norm).

Taking logs and expectations over real text:

    log retained(Δ) ≈ Δ · ( E[log g] − E[β] · E[ρ²] ) ,   so   τ_head ≈ 1 / ( −E[log g] + E[β] E[ρ²] ).

Two consequences that are decisive before any probe runs:

- **The gate term** `−E[log g]` is the pure-forgetting rate. A head whose mean log-gate is
  `−0.001` forgets on a scale of a thousand tokens; `−0.05` on twenty. It is read straight off
  `compute_g` on real trajectories, one forward per trajectory, no probe design at all.
- **The interference term** `E[β] E[ρ²]` is the delta rule overwriting. For keys behaving like
  random unit vectors in `Dk` dimensions, `E[ρ²] = 1/Dk`, so even a head that never forgets
  (`g ≡ 1`) retains a write for only about `Dk / E[β]` tokens: with `Dk = 128` and
  `E[β] = 0.5`, about **250 tokens**. That is the same scale as Khandelwal's 200-token effective
  horizon, derived from the architecture rather than fitted. Structured keys (low overlap for
  distinct content) lengthen it; measured `E[ρ²]` per head on real text says by how much.

**Proposal (near-free, the first thing to run when the hold lifts):** one forward per trajectory
over the EXP-001 points, hooking `g`, `β` and `k` in every DeltaNet layer, and report per head
`E[log g]`, `E[β]`, `E[ρ²]` between keys at gaps of 1, 10, 100 and 1000 tokens, and the implied
`τ_head`. If no head's `τ` reaches the hundreds, the recurrent path cannot be a long-range memory
channel on this model, EXP-002's arm A is predicted null at the ledger distance, and EXP-003's
masked arm (part 1, proposal 1) is answered before it is designed. If some heads have `τ` in
the thousands, those heads are where a persistent-cache regime would live, and the probes
should look at them specifically. This is a pre-registration of a prediction with a mechanism,
which is what an interpretability programme should be producing.

## 3. EXP-002 lacks two cells of its own factorial, and the architecture supplies them cheaply

EXP-002's arms are: A (attention blind to the listing, recurrent has it), B (both have it), C
(neither, but on the **windowed text**, which differs from A and B's), and a foreign-state
control. The comparison A versus C therefore confounds "recurrent path only" with "different
text". Two cells complete the design **on identical text**:

- **D, recurrent-ablated:** run the same unwindowed prefill with `β_s := 0` for every `s` in the
  listing span in every DeltaNet layer (no writes from the listing; gates still applied), while
  attention sees everything. Exact, cheap (the same forward with modified `β`), and it never
  touches the attention path, which is the asymmetry EXP-002 turns on.
- **E, both-blind:** attention masked over the span **and** `β_s := 0` on it. This is the true
  floor on the same text, and it must land at EXP-001's condition; if it does not, the
  identical-text assumption is broken and the reading stops.

Then `P(true)` over `{A, B, D, E}` is a 2 × 2 on one text: main effect of attention, main
effect of the recurrent path, and their interaction, each paired per point with the exact sign
test and the decomposition. Arm C is kept as the reproduction check only. The `β := 0`
ablation is also a second gate on the instrument: `E` must be indistinguishable from the
never-mentioned condition, which is a prediction with no free parameter.

## 4. What each curve decides, stated so nobody over-reads EXP-003

The deployed regime re-prefills the windowed text each turn, so inside the window the listing
is visible to attention until it is stubbed. EXP-003's **open** arm therefore measures the
deployed regime's retrieval reliability against distance, which is what the note contract
needs *for the deployment we run*; its expected shape is near-flat until capacity effects, and
that is a result, not a null. The **recurrent** horizon (parts 1 and 2) matters only for a
persistent-cache regime we might build, which is EXP-002's question. The two must not share a
verdict row. Part 1's weights give both from the same forwards.

## 5. A prediction to pre-register for the J-space sweep's context-constant rows (#78)

Through a DeltaNet layer, the Jacobian of `y_t` with respect to the input at position `s`
carries the same `α_{t,s}` structure, so the `future` readout at a source position `s` is
dominated by `Σ_{t>s} α_{t,s}` through the recurrent blocks plus the attention broadcast. Rows
that were context-constant (L5, L12, L20 `future`) should be exactly the layers and readouts
where `Σ_{t>s} |α_{t,s}|` from the source is negligible on this text. That is checkable from the
part-1 weights and would turn #78's empirical exclusion into a mechanistic one.

## 6. Summary of the amendments proposed

1. Add the exact recurrent read-weight decomposition (`α_{t,s}`) as an instrument in
   `pipeline/`: per head, per layer, from one cached forward; report span mass at the decision
   and the weight-against-distance profile. Reuses the chunkwise algebra.
2. Run the gate-and-interference horizon measurement (`E[log g]`, `E[β]`, `E[ρ²]`, `τ_head`)
   first; pre-register its prediction for EXP-002 arm A and EXP-003's masked arm.
3. Complete EXP-002 to a 2 × 2 on identical text with the `β := 0` write ablation (arms D, E);
   arm C becomes the reproduction check only.
4. Split EXP-003's verdict rows by regime: open arm for the deployed regime, recurrent weights
   for the persistent regime; drop the masked-arm grid if part 1 supplies the curve.
5. Pre-register #78's mechanistic explanation and test it against the weights.
6. Part 1 of this memo (six earlier proposals) stands where not superseded: the note-carried
   arm, the model set and criterion, the visible-at-zero ceiling, graded relevance.

— Chief AI Research Scientist

## 7. The constants, pinned to this model from disk (no model run)

From the cached 4B `config.json`: 32 layers, `full_attention_interval` 4 (24 DeltaNet blocks),
`linear_key_head_dim = linear_value_head_dim = 128`, 32 value heads over 16 key heads (2:1
repeat), attention heads 16 with 4 KV heads at `head_dim` 256, native window 262,144. From the
library: `β = sigmoid(in_proj_b(x))` (`gated_delta.py:274`);
`g = exp(−exp(A_log) · softplus(a + dt_bias))` with `a = in_proj_a(x)` and `dt_bias`
initialised at 1 (`compute_g`, `qwen3_5.py:121-124`). So `log g = −A_h · softplus(a + 1)` with
`A_h = exp(A_log_h)` a learned per-head scale.

**Read from the weights (24 × 32 = 768 heads):** at neutral input (`a = 0`, `softplus(1) =
1.31`) the gate time constant `τ = 1/(A_h · 1.31)` has median **2.4 tokens**, maximum **51.7**
(a head in the first DeltaNet block), and no head reaches 500. Per layer the slowest head sits
between 3.5 and 13.5 tokens in every block after the first. For any head to retain a write for
a thousand tokens, its input gate must hold `a ≲ −5.2 to −6.4` on average across those tokens,
i.e. `softplus(a + 1)` a few thousandths. That is a strong, learned, input-dependent condition,
not a default, and whether real trajectories produce it is the single measurement that decides
whether the recurrent path can be a long-range channel on this model: **`E[softplus(a + 1)]` per
head on real text, one hook, one forward per trajectory.** With the interference bound of part
2 (`Dk = 128`, so about `128 / E[β]` tokens even at `g ≡ 1`), the recurrent horizon on this
model is predicted to be tens of tokens by the gate alone and at most a few hundred by the delta
rule even where the gate is held open. EXP-002's arm A and any masked distance curve are
predicted null beyond that scale, and the prediction is falsifiable by the hook measurement
before either experiment spends lane time.

## 8. From the Director's reading (2026-09-05 late), recorded

- **Verification of the derivations is part of the proposal, not optional:** the read-weight
  identity is checked by reconstructing each DeltaNet layer's output from the weights to float
  tolerance; the horizon bound is checked against measured weight magnitudes at distance on
  the same text; the gate-input measurement can falsify the "tens of tokens" claim outright.
- **Proposal 3 (EXP-002 completed to a 2 × 2 with the `β := 0` write ablation) is approved by
  the Director**, to take effect when the veto lifts; the gate-input measurement (part 2) is the
  first thing the Deputy implements then.
- **Open epistemic item:** EXP-001 did not establish that the model has a J-space in the source
  paper's sense. It showed a Jacobian readout that responds to in-context content at mid-to-late
  layers and is context-constant at early ones; it never tested the paper's stronger claims (a
  workspace band; the J-lens reading intermediate content more faithfully than the logit lens),
  because the sweep artifact carries only the two candidates' probabilities per layer. The
  sweep should record full per-layer top-k distributions and test the J-lens against the logit
  lens and the model's output before the programme keeps calling it a J-space probe.

## 9. Brief for the gate-input measurement (ready for dispatch when the veto lifts; not dispatched)

**Name:** `agent-v2-probe-recurrence` (or a `--recurrence-stats` mode of an existing probe CLI).
**Question:** on real trajectories, what horizon does each of the 768 DeltaNet heads actually
have, and does any head satisfy the thousand-token condition?

**Inputs:** EXP-001's 42 probe points on the `jsweep` split (persistent, unwindowed render with
stripped notes as EXP-002 §3.2 constructs it, so the text is the one the persistent-cache
question is about), `qwen35-4b`, policy `base`, one forward per point, no Jacobians.
**Hooks (R31: the real `GatedDeltaNet`, not a stand-in):** per layer, capture `g = compute_g(...)`
(`[B, T, Hv]`), `β = sigmoid(in_proj_b(x))` (`[B, T, Hv]`) and the normalised keys `k`
(`[B, T, Hk, Dk]`) at the recurrence's input; no residual capture, so R18b's capture dtype does
not apply and the block runs natively.
**Statistics per head (R38: every count as n of N tokens):** `E[log g]` and its distribution
(p10/p50/p90); `E[β]`; `E[ρ²]` between keys at gaps of 1, 10, 100 and 1000 tokens (random-key
baseline `1/Dk = 1/128`); the implied `τ_head = 1/(−E[log g] + E[β]E[ρ²])`; the fraction of
tokens on which `softplus(a + 1)` is below the value a thousand-token horizon needs (part 7,
per layer); and the **measured** retention `|α_{t,s}|` against `t − s` from the read-weight
decomposition (part 1) at the decision position, which is the identity check against the bound.
**Verification (the derivation is falsifiable here):** reconstruct each layer's recurrent
output from `Σ_s α_{t,s} v_s` and compare with the layer's actual output at the decision (float
tolerance stated and recorded, as the chunkwise slice does); compare measured retention with
the predicted product at three gaps; both in the artifact.
**Pre-registered reading:** if no head's `τ` reaches the hundreds on this text, the recurrent
path is not a long-range channel on this model; EXP-002's arm A and any masked distance curve
are predicted null beyond that scale, and EXP-002's factorial (part 3) becomes a confirmation
run rather than a discovery run. If some heads reach the thousands, those heads are named and
the probes target them.
**Cost:** prefill-bound, one forward per point plus `T × T` weight matrices per head at the
decision; well under EXP-002's hour. R26 logging, R34 conformance (what was hooked, at what
dtype, with which convention), R35 comparability, one Director or Proxy lift.

## 10. Verification on random tensors against the library's own reference loop (2026-09-05 late)

Run in-process on random tensors with `mlx_lm.models.gated_delta.gated_delta_ops` as the
reference (no model, no lane; the same route the chunkwise slice's tests use). Two results, one
of which corrects part 2.

1. **The read-weight identity holds.** `y_t = Σ_s α_{t,s} v_s` with `α_{t,s} = β_s · k_sᵀ
   [Π_{r=s+1..t} g_r (I − β_r k_r k_rᵀ)] q_t` reproduces the library's output at every one of 96
   positions to a worst absolute error of **1.2e-6** in float32 (`T = 96`, `Dk = Dv = 128`,
   unit keys, gates in [0.9, 0.999]). Part 1's instrument rests on an identity the library's
   step satisfies.

2. **The retention bound is a bound on the surviving norm, and part 2's interference constant
   was wrong by a factor.** Propagating an old key through the later operators, its norm never
   exceeds the gate product (max ratio 0.985 over 400 positions: the deflation is a
   contraction), but the *read weight* `α` is that norm times the cosine to the query, and the
   deflation rotates the trace as well as shrinking it, so `|α|` can exceed the gate-only figure
   where `k_s ⊥ q_t` (ratio up to 8.3 observed). The horizon statement therefore applies to the
   surviving norm `‖k_sᵀ Π M_r‖`, which is what "the write is still there in some direction"
   means; the read at a given query is that retention times a cosine. Squaring the deflation
   step gives the exact per-token norm factor `√(1 − (2β_r − β_r²) ρ_r²)`, so the rate is
   `E[β − β²/2] · E[ρ²]`, not `E[β] E[ρ²]`: with uniform `β` that is `0.333/Dk` per token, and
   the prediction at a gap of 399 tokens is **0.354 against a measured 0.353**. Corrected bound
   at `g ≡ 1`: `Dk / E[β − β²/2]`, about **384 tokens** for uniform `β` and **256** for `β ≡ 1`,
   with `Dk = 128`. The gate term is unchanged. Part 7's conclusion (tens of tokens by the gate,
   a few hundred at most by interference) stands with the larger constant.

So the derivations have been tested where they can be tested without the model, and the one
constant that was wrong is now the measured one. What remains empirical is the model's own
`E[log g]`, `E[β]` and `E[ρ²]` on real text, which is part 9's measurement.


## 11. Amendments after the Head's WP3 design (2026-09-05 evening, R41a)

- Section 9's threshold ("median |alpha| below 0.05 beyond 256 tokens") is withdrawn: the library scales q by Dk^-1 and k by Dk^-0.5 after RMS normalisation (`mlx_lm/models/qwen3_5.py:178-180`), so |alpha| <= 0.0884 * beta at every gap and the threshold passes at gap 1. Replaced by the design's retention ratio (normalised to bin 1-4) and mass share against a uniform null.
- Section 10's fixture used one key head per value head. The library pairs value head h with key head h // 2 by `mx.repeat` (`gated_delta.py:242-244`); a fixture at Hk == Hv cannot exercise this. Re-verified 2026-09-05 with Hv/Hk = 2: scan and explicit read weights agree to 6e-17; the h % Hk pairing fails on two of four heads.
- Production path for alpha rows: the backward scan w_{s-1} = g_s (w_s - beta_s k_s (k_s^T w_s)) from w_t = q_t, with alpha_{t,s} = beta_s k_s^T w_s, O(T Dk) per head; the explicit product form stays as the oracle.
- Layer convention: the attention blocks are block indices 3, 7, ..., 31 and write layers 4, 8, ..., 32; in-band attention blocks are 15, 19, 23, 27 (layers 16, 20, 24, 28; 64 query heads).
