# Execution lift request: training arm B4 (Qwen3.5-4B on run B's rows)

To the Director, from the Deputy. Ruling R15's seven conditions, one evidence line each, with
artifact paths and SHA-256s (recompute with `shasum -a 256 <path>`). Per the accepted
sequence this follows P6; it is drafted now because every condition except your own
acceptance is already evidenced on origin at `07c6657`.

**Status (re-issued 2026-09-05 evening after R32): condition 6 is ACCEPTED (Director: "Go"); three
attempts failed before iteration one (records on disk); attempt 4 waits on two measured gates
(below) and on condition 1's regeneration. Conditions 2, 3, 5, 7 and 8 are met; 4 is met with two
ruled deviations recorded here.**

## The command

    uv run agent-pipeline --config configs/agent_v2b_qwen35_4b.yaml train

Reads `data/agent_v2b-qwen35-4b`; writes only under `outputs/agent-v2b-qwen35-4b/`
(adapters, checkpoints every 100 iterations, lora.yaml, train.log, provenance). The training
stage is gated by `require_preflight` on the artifact in condition 1, so a stale or failed
preflight refuses the run before any weight loads.

## Conditions

1. **Preflight `passed: true` under R18a — met at schema 2; to be REGENERATED under schema 3
   (R32 b/d) when the calibrated footprint gate lands: budget = min(registry 22, device working
   set 17.8 GiB), training footprint for the longest row (2,257 tokens) under the chosen chunk
   with 10% headroom.** Current artifact: `outputs/preflight/qwen35-4b.json`, schema 2,
   native max_abs 0.0, Frobenius 0.0, JVP finite-difference (finite), memory 3.85 GiB of 22,
   cache `none` (`auto:equivalence_unverified` — correct and slower, not wrong). SHA-256
   `8a6f429809eebdc11d65202e3c9ebeea703b578ad43300c0318b776222464a79`. 3B control passed
   identically (`c536353043eb5d0eb53dcb8ecd1eb275616b398ad089bdd7c3b08cdccae36e04`).
2. **In-repo rendered rows; rendering-equivalence green — met (R14).** `stage_train` consumes
   `load_rendered_splits`; byte-identical migration test for the 3B path; chat rows proven
   byte-identical to the template on all 348 real rows (#24).
3. **Dataset on disk with manifest hashes and provenance — met.** `data/agent_v2b-qwen35-4b/`:
   train 1915 rows `618dcd5006666baa48aaaacf9f1ec0ae6e218acd2bd47e912b36776e2fb7c1b6`, valid
   284 `9a4e23777f81d749b2fcb1ac76e838b6fb32dcd3ffdf147be8d1377499c44a68`, test 494
   `9ab2590e784f15ff43f2de5690ba30181386bc2704d87355a9bfdb5a69f6c1f5`; exact row parity with
   run B; source manifest SHA `f6784deba9dddc50707d46cfe326d46e3fab273ac3f1320cf3b44b91147d06da`;
   `generator_version: null` recorded for pre-versioning B; `manifest.json`
   `8fa5899fb43fea0ca757c4b926242b62c8b7b4100c6e03dc9e6a1736dfff5dd0`, `provenance.json`
   `679520a2452d63893b898f0e7342eae2b127c5dc34d48c551b1f6f1a7467aa7a`. Rendered, not
   regenerated: the guard (R21, `94c920e`) makes regeneration over run B unreachable.
4. **Model-aware loading — met per R15's text; two ruled recipe deviations recorded (R32).**
   (i) `batch_size: 1`, `grad_accumulation_steps: 4` — run B's effective batch of 4 at half the
   per-step activations; the pairwise-recipe test compares effective batch. (ii)
   `train.gated_delta_chunk: 64` — the training-time gated-delta recurrence runs in checkpointed
   chunks (`0b5227e`), bit-exact against the library's loop; hybrid arms only. `max_seq_length`
   stays 2688 (the 7f11288 change was reverted, 112648b). `load_policy` on the registry signature
   (`a2f003c`), `resolve_policy` reads `ModelSpec.policies` (`e53bd81`),
   `branch.build_prompt` threaded (`a2f003c`); the view sees adapter wrappers (`89dfb56`).
   The training-path loader still calls `mlx_lm.load` directly — the R20 debt slice, a bound
   follow-up the Chief scoped outside R15; it does not affect this run's correctness (LoRA
   targets resolve through the view: twelve keys including the five `linear_attn` projections,
   32,464,896 trainable, per the preflight artifact). Note: the B4 config still carries
   `train.num_layers: 36` (the 3B's count) — `lora_config` ignores it and writes
   `resolved.num_layers` (32) into `lora.yaml`, verified at `cli.py` in `lora_config`; harmless,
   misleading, and a one-line cleanup follow-up for the cross-model configs.
5. **Screen present; criteria recorded — met.** `select.screen` in the config; `criteria:`
   block verbatim from SPEC-003 §5 (`7dd251d`): total higher than run B with McNemar p < 0.05
   on paired task ids, or a family-level gain of at least five in one long-horizon family with
   no loss elsewhere. Recorded, never used for selection.
6. **Execution claim free; cost accepted — ACCEPTED ("Go", 2026-09-05).** The cost estimate
   is superseded: the 4B's training-time recurrence is a per-token loop (R32); measured on the
   lane, one 997-token row steps at 11.86 GB peak in 72.6 s including a one-row validation at
   chunk 64. **Attempt-4 gates (Chief, #51):** (a) the 2,257-token row steps under the device
   working set (17.8 GiB) with 10% headroom at the chosen chunk; (b) the clean per-step time
   projects 400 iterations at ≤ 3 h. Both come from the probe after the P6 rerun releases the
   lane; above (b), stage 2 (chunkwise-parallel form) is built first. Memory budget resolves to
   17.8 GiB, not 22. Writes only under `outputs/agent-v2b-qwen35-4b/`; attempts 1–3 kept there.
7. **First arm is B4 — met by R15 itself.** Same data as run B, new base, thinking off,
   `all-linear` keys: the controlled comparison the decision memo §3 asked for.
8. **Run logging in place (R26, Director's amendment; checklist A8) — met.** Lane A
   `c1f7d51` (`runlog.py`, 70 tests) and lane B `88ecac0` (`stage_train` opens the run log
   before the base loads; `health.json` on every exit path; `non_finite_loss` aborts with no
   provenance; `incomplete_run` on a short run or a checkpoint older than the run's start;
   `metrics.jsonl` byte-identical). This run will write `outputs/agent-v2b-qwen35-4b/run.log`,
   `events.jsonl` and `health.json`; the evaluation lift will cite all three by path and
   SHA-256 and list every warning with the Deputy's reading (R26 d).

## After training, before evaluation

Selection uses the family-balanced screen (SPEC-002 §1) with earlier-step tie-break; the
evaluation's `load_policy` path on a Qwen3.5 adapter is the first live use of wave 1's code on
the new base — the bound follow-up (adapter-wrapped standard fixture, due before this eval
lift) should land first, and the eval gets its own per-run lift with this format.
