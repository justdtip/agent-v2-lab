# R21 write guard and render stage, review round 1, ratified (2026-09-04 20:40)

Reviewer: Claude (Chief). Evidence: issue #24, `R21-RENDER-GUARD-REPORT.md`, and a direct
read of the production diff (`pipeline/data.py`, `pipeline/cli.py`, both configs) because the
slice is safety code. Tests not re-read; the reviewer's real-data verification (348 chat rows
byte-identical to the template; case-alias bypass found and closed by file identity) is adopted.

## Verdict: APPROVED TO COMMIT with three small conditions, all inside the slice's own files.

The guard sits at the write boundary, protected directories refuse unconditionally by
inode identity, the render stage carries `messages`/`metadata` verbatim and provably never
reaches the generator, source hashes and the source's `generator_version` (explicit null for
run B) are carried, and the B4 config physically cannot regenerate. Good work, including the
mid-slice discovery of the 348 chat rows.

## Conditions (K)

| # | Condition | Why |
| --- | --- | --- |
| K1 | `guard_dataset_write` also refuses any target **inside** a protected directory (walk `resolved` and its parents with the same identity check), not only the directory itself | today `data/agent_v2b/anything` passes; pollution of an irreplaceable directory should be impossible, not merely cosmetic |
| K2 | The rollout and branch guards are currently inert: neither stage writes `manifest.json`, so the manifest-triggered guard never fires on their outputs. Fix by having both stages write a `manifest.json` (rows, sha256, seed, source config) alongside `summary.json`, so the guard applies, and keep `override_flag=None` | a guard that can never fire is a false assurance |
| K3 | The verbatim-completion fallback in `render_rows` keys on `source != "expert"`. Make it an explicit allowlist `VERBATIM_COMPLETION_SOURCES = {"pre-expansion-policy-replay"}` (the tag the 240 chat rows actually carry); any other source without a parseable call raises | a rollout row that lost its call would otherwise be silently rendered as chat; a controlled arm must fail loud |

## Rulings

- Flag name `--force-overwrite` accepted; R21 wording amended, no rename.
- Flag-free protected refusal message accepted.

## Follow-ups (record, no dispatch now)

- Legacy writers (`chat_replay.py:140`, `generate_agent_data.py`, `generate_complex_data.py`,
  `build_expanded_data.py`) and the module-level mains of `rollout.py`/`branch.py` bypass the
  CLI-boundary guard. Route through `guard_dataset_write` when next touched; K2 covers the two
  pipeline stages' CLI path.
- `write_jsonl` writes directly; R11 says over 1 MiB uses the atomic pattern. Pre-existing;
  fold into the next data.py touch.

## On commit

Training checklist A3.1 ticks. The authorised B4 render is then
`agent-pipeline render --source data/agent_v2b --output data/agent_v2b-qwen35-4b --model qwen35-4b`,
and its manifest hashes go into the #18 request as R15 condition 3's evidence.
