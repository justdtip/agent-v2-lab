To Codex, from the Deputy Chief of AI Research, passed on by the Director. 2026-09-03, night.

Overnight arrangement:

1. I review hourly, roughly on the hour from 01:00 to 08:00; you check issues at 00:30 and
   hourly after. Every finding of mine lands as a GitHub issue before your next check. The
   running record is `design_specifications/under_review/OVERNIGHT-LOG-2026-09-04.md`.
2. Already filed: issue #5, "SPEC-004 §1: ratified-table mismatch". It is addressed to the
   Chief, not to you. Preserve the rerun outputs exactly as computed; do not adjust them toward
   the ratified table. You did the right thing reporting honest numbers.
3. What I verify each hour: new commits mapped to lanes; signatures against wiring map §2;
   each lane inside its claimed paths; banned constants on added lines; protected directories
   untouched; no model loads in tests; the suite green; report and SDD ledger present when a
   lane completes; R10 test moves subtractive, not duplicating.
4. First-pass reviews of your Task 3 slice (`d498599`, `be68a2d`) and the SPEC-004 remediation
   (`b39d6ea`) are running now; any findings will be on the tracker before your 01:30 check.
   On Task 3 I am looking hardest at `response_mean_activations`: any behavioural change to the
   prefix-merge repair breaks comparability with the saved P2 npz and will be raised
   immediately.
5. Standing gates tonight, unchanged: Task 5 stays blocked until your head's explicit unblock;
   no model runs of any kind; documents in `pending/` are not edited by implementers (R3);
   protected directories per briefing §1.2.
6. Priority once your seams free: SPEC-002 §2, note integrity. It is the programme's next
   priority per issues #2 and #4. If no lane has claimed it by 03:00 I will file an issue so
   the monitor assigns it.

Reviews I draft overnight are DRAFT; the Chief ratifies at morning standup. Good night's work
so far: the oracle split was subtractive and the C6 rerun was clean and honest.
