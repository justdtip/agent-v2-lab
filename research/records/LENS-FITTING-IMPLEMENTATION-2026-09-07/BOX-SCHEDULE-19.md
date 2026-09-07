# Lens box-time hold released for issue 88

Recorded 2026-09-07T20:19:20+10:00 following the Director's status request.

Issue 88's latest sizing comment holds its depth ablation until Codex schedules the Jacobian and regression box time. The lens task releases that scheduling hold now. It has no active model run or model reservation. No further native MLX test, calibration or checkpoint fit will start from this task until an explicit handoff supplies a gap in the other work's queue. Pure source work and reviews continue in the persistent worktree. Issue 88 may take the next model slot once its own prerequisites clear; this statement does not change its scientific acceptance or authorise skipping them.

At inspection, issues 86 and 87 both remain open, and 88 says after 87, which says after 86. Their sequencing is separate from this released hold. The sizing in issue 88 is approximately 13 hours in six separable blocks, including projected rather than measured top-16 throughput. That is the issue author's estimate, not a timing measured or revalidated by the lens task.

Lens scheduling is staged: first the registered regression resource probe, then individual regression fits only after its memory and timing result; Jacobian self-check and batch timing before reserving full all-layer fitting time. The requirements' under-one-hour regression target and at-most-25-minutes-per-layer Jacobian target are acceptance targets, not measured reservations. No all-layer duration is promised before calibration. Calibration attempts 01 and 02 were refused before weights loaded; no fit currently exists from this task.

References read directly on GitHub: issue 88 and its sizing comment https://github.com/justdtip/agent-v2-lab/issues/88#issuecomment-5568972603 ; issues 87 and 86. No GitHub comment or issue mutation was made. This record is pushed on codex/lens-fitting for the Chief's handoff; the primary shared branch remains untouched by this task.
