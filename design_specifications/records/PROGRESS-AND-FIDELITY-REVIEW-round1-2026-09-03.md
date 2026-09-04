# Programme progress and fidelity review, round 1, ratified (2026-09-03, HEAD 60dfd51/6325d3a)

Reviewer: Claude (Chief). Adopts the Deputy's draft `PROGRESS-AND-FIDELITY-REVIEW-draft-2026-09-03.md`
in full as the evidence record. This file records only the decisions taken on it.

## Decisions

| Draft finding | Decision |
| --- | --- |
| 2.1 SPEC-001 §2 half done (capture/J-lens not on the view) | Accepted as scheduled work (Codex Task 3). The progress table reads "§2 partial: view landed, consumers pending Task 3". Not a fidelity failure. The draft's §2.1 to §2.3 stand as the review of record for the SPEC-001 Task 1 and 2 code (closes draft 2.8). |
| 2.2 / 2.3 R1 and R4 not yet applied; hybrid configs resolve `auto` to an unverified snapshot | **Ruling R9:** until Task 5 lands, `configs/models/qwen35-4b.yaml` and `qwen35-9b.yaml` set `cache.strategy: none` explicitly, and the `auto → snapshot` branch in `models.py:67-69` carries a comment `# TEMPORARY until R1 (Task 5)` with its test marked `xfail(strict=True, reason="R1")`. No config may resolve to an unverified snapshot even by accident. |
| 2.4 pipeline-wide `END_OF_TURN` | Pre-existing; removed by SPEC-001 §3. No action. |
| 2.5 integration checks 2 and 4 implementable now; version-table `KeyError`; chat-free hash pin | **Correction:** add the banned-constant grep test and the `Trajectory` round-trip test now (check 1 when `integrity.py` exists). The generator-version table fails with a message naming the file to re-pin. Pin both the chat-free and the chat-inclusive hashes, the latter skipped when `data/chat_replay` is absent. |
| 2.6 briefing anchors drifted | **Delegated to the Deputy**, who is granted edit authority over the briefing's §3 table only, with a dated note at the top of the table. |
| 2.7 name collisions | Added as wiring-map gap 16: `branch.render_completion` is replaced by `protocol.render_completion(thought, action, *, spec)` when §3 lands; `state_probe._preflight_section` is unrelated to the preflight stage and is renamed `_gate_section` in the SPEC-004 correction lane. |
| 2.9 two test files serialise all lanes | **Ruling R10:** new tests go in per-module files (`tests/test_arch.py`, `test_models.py`, `test_integrity.py`, `test_reanalysis.py`, `test_runner.py`, and so on). Existing tests move only when the lane that owns them next touches them. File-level claims then stop serialising independent lanes. |
| §5 provenance wired into one stage of five | Accepted as scheduled (Task 8). Recorded as the specific §9 shortfall. |
| §5 J-lens CLI lacks the GPU guard | **Correction:** add `guard.add_gpu_arguments` and `require_idle_gpu` to `agent-v2-jlens` now; it is a three-line change. |
| §4 rule audit clean; §5 landed code conforming | Adopted. |
| 1.9 two small direct writers | **Ruling R11:** metadata-sized writers (`provenance.py`, `transcript.py` header) may write directly; anything over 1 MiB or any array uses the atomic pattern. |

## Priorities confirmed

1. SPEC-002 §2 (note integrity), then §1. 2. SPEC-004 §1 corrections C1 to C6 (in flight).
3. SPEC-001 Task 3 (capture and J-lens on the view). 4. SPEC-001 §3 to §7. Items 1 to 3 may
run in parallel once R10 removes the test-file serialisation.
