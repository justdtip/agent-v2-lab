# Live-lens pilot on the running model — 7 September 2026

Written before any pilot record was read (house rule: pre-register before reading). The Director's
instruction, 15:30: "Do the J-one at some point in the next few hours." This is items 1 and 2 of the
plan at pilot scale, on the capture infrastructure landed at a740997; item 3 (the head population in a
real scenario) is a separate, targeted run.

## What runs

`scripts/live_lens_pilot.py --out research/records/LIVE-LENS-PILOT-2026-09-07/` (plan in `plan.json`):
ten agentic episodes from the test split (seed 20260902) at difficulties 0, 1 and 2 across read, list,
pointer_chain, search, synthesis, batch_update, update, cross_reference and aggregate_report, twelve
steps at most; and three chat episodes of two turns each (a short factual question, a reasoning
question, a 1,681-token summary request), each followed by the same one-sentence follow-up. Greedy
decoding; cache strategy `none`; the base model (no adapter); layers read = the band's five attention
members from the registry (12, 16, 20, 24, 28) and the final layer 32 (the native distribution); top-10
per position; head capture off. One record per episode, hash-chained, with the native forward hashes.

## What is measured, and the reading rules, fixed now

1. **Foreknowledge.** For each emitted token at position p and each horizon h in {1, 4, 8}: the
   competition rank of that token in the lens distribution read at position p−h, per layer. Summary:
   median rank and the share of ranks ≤ 10, by layer, by horizon, by episode kind (agentic / chat), by
   difficulty, and by the span the emitted token falls in (note text, call JSON, chat prose). Layer 20 is
   primary; the others are a profile and are never pooled with it. Layer 32 at h = 1 is rank 1 by
   construction under greedy decoding and is the reference, not a result.
2. **Lens–next-token agreement along the prompt.** For prompt positions, whether the layer's lens top-1
   equals the actual next prompt token, by span (system, task, observation, note, call) and by layer.
   Descriptive: this is the picture of what the lens reads on the running model across contexts.
3. **Cost.** Wall time and peak memory per episode, against the native episode times on record.

No hypothesis test is run in the pilot. The comparison the report will state, chosen now: at layer 20
and h = 1, the share of ranks ≤ 10 in agentic call spans against note spans against chat prose. A layer
whose h = 1 share at layer 20 is below the final layer's h = 4 share is read as "not ahead of the
output"; nothing else is inferred from the pilot. Unresolved futures at a turn boundary are censored,
as the session records them.

## Limits declared in advance

Thirteen episodes; one model; no adapter; greedy only; head capture off; prompts under 4,096 tokens;
foreknowledge horizons cross no turn boundary. Per-episode cost is expected at minutes, dominated by the
full-vocabulary lens ranking at every prompt position (benchmark: 13.6× native on a short prefix).
