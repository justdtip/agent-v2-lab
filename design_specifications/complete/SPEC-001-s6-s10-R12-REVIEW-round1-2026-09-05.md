# SPEC-001 §6–§10 and R12, review round 1 (retrospective), ratified (2026-09-05 02:30)

Reviewer: Claude (Chief). Evidence: source-level audit (667 tests green, fakes only) and the
Chief's grep confirmation of the three absences below. §1–§5 were ratified on #19/#23.

## Verdict

| Section | Status | Decisive evidence |
| --- | --- | --- |
| §6 LoRA targets | COMPLETE | `arch.py:31-38,198-237`; keys and count in provenance |
| §7 training entry | COMPLETE | `cli.py:184-189,230,256-280,348-377` (R14 signature, pin, metrics.jsonl); `tuner_data.py`; memory refusal via the preflight gate |
| §8 probes on the view | COMPLETE except one clause | `--model`, fractions, per-model policies, split DeltaNet parsing, dequantised base via the view all present; **missing:** the cross-run `hf_id`/snapshot-revision assertion in `adapter_delta.compare_adapters` |
| §9 provenance | PARTIAL | wired into data/train/select/eval/rollout/render (`cli.py:126,155,378,581,651,704`); **no probe CLI or `jlens` calls `write_provenance`** |
| §10 preflight | PARTIAL | gate and artifact complete (`preflight.py:270-336,397-432,486-534`); both artifacts pass at schema 2; **`require_preflight` gates train/select/eval/all only — not rollout, branch, or any probe CLI** |
| R12 versioned replay | COMPLETE | `tasks.py:180-201`; version-pinned contract tests; R23 fail-closed binding (`integrity.py:445-455`) |
| R1/R4/R9 registry | COMPLETE | `models.py:66-77`; R9's temporary override correctly retired |

## SPEC-001 closure slice (one dispatch, disjoint from P6 after #30 commits)

1. Every probe CLI (`state_probe`, `assistant_axis`, `adapter_delta`, `jlens`, `patch`) and the
   `rollout`/`branch` stages call `require_preflight(spec)` before loading a model and
   `write_provenance(output, resolved=..., spec=..., extra=...)` after writing their artifact;
   probe artifacts also copy the preflight `fp32_manual_vs_native` block (R18a clause).
2. `adapter_delta.compare_adapters` asserts identical base `hf_id` and snapshot revision across
   the compared adapters (read from each adapter directory's `adapter_config.json`/provenance
   where present; refuse with a named error otherwise).
3. Tests on fakes for each wiring point and for the refusal.

`SPEC-001-S6-S9` and `R12` reports move to `complete/` now; `S8` and `S10` reports stay in
`under_review/` until the closure slice lands, then move with its report. The spec stays in
`pending/` until then.

## Note for the training gate

Both `outputs/preflight/*.json` now pass under R18a at schema 2: R15 condition 1 is met for
`qwen25-coder-3b` and `qwen35-4b`.
