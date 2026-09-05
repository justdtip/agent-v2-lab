# STATUS 2026-09-05 late evening, FINAL: SETTLED THROUGH THE PROXY. READ BEFORE ACTING.

The Proxy relayed the Director's clarification, verbatim: "pursue research freely. The concurrency rule (never load the model multiple times / no two model-loading runs at once) is the actual constraint -- that's always been the intent behind it, not gatekeeping test or research runs generally. Confirmed as in-scope: a performance/hardware benchmark that loads the model but tests no hypothesis about model behavior still counts as a test run under the lift."

Operating rule from tonight, for every session:
- Research and test runs proceed without a per-run lift. Ready means what it meant in this note's first version: the spec ratified, the reading pre-registered, power set by the WO-STAT-001 method, the instrument's fixture green. That is the Chief's gate on the design, not a lift on the run.
- The one constraint is concurrency: never two model-loading processes at once, checked immediately before launch. Its mechanism is the lock of issue 83; until that lands the process-list check is the rule and every launch logs that it ran it. The first runs under this rule wait for issue 83, since it is the constraint the Director named.
- Training lifts are not covered by this note and keep their own gate.
- The Chief reports every result to the Director; visualisations are a deliverable.

The two readings of "test run" recorded below are superseded: the distinction between a run that verifies a known answer and one that produces a new number does not gate anything under the Director's stated intent. The banners below are kept as the record of how the evening's holds were handled.

# STATUS 2026-09-05 late evening: SUPERSEDED BANNER: PARTLY SETTLED THROUGH THE PROXY. READ BEFORE ACTING.

The Proxy relayed the Director's word, verbatim: "a standing lift now applies to all test runs -- those can proceed without a per-run lift from Daniel or me. This is specifically about test runs; anything beyond that (full experiment ratification/lifts) still follows the existing standing-order rules." Under the Proxy standing order that relay carries the Director's authority for every session.

Operational definition recorded by the Chief, pending the Director's correction: a **test run** is a model-loading run whose purpose is to validate an instrument, a fixture or a pipeline rather than to produce a reading that enters the decision memo: validation runs against a recorded table (WP1's 42-case validation), reconstruction-gate re-measurements, dry runs of an experiment on a handful of items, calibrations, and instrument pre-checks (WP4's read-share pre-check). A **full experiment** is a run with a pre-registered reading that enters the record as a result: WP3's transport statistics under readings R1 to R4, EXP-002's 2 by 2, EXP-003, EXP-005 trace collection at scale, WP7 and WP9 probes. Full experiments take a lift through the Proxy under the standing order. One model-loading run at a time, checked immediately before launch, applies to both.

**Two readings of "test run" exist and the narrower governs until the Director rules.** The Chief's definition above admits pre-checks, calibrations and gate re-measurements that produce a number nobody has yet. The Head's reading (`live/LIFT-STATE-2026-09-05-EVENING.md`) admits only runs whose purpose is verifying a known answer: unit fixtures, WP1's validation run against the recorded 42-case table, and dry runs or instrument fixtures that load the model only to check they reproduce. Under the Head's reading the following are NOT test runs and wait on a lift: WP3's transport run, the source-position inspection of the three key-head units (which needs a fresh hooked forward pass on any reading, since no tensors were saved by tonight's calibration scripts, only derived JSON; its pre-registered reading is R3f in the WP3 design), EXP-002's read-share pre-check and its 2 by 2, and all of wave 2. The Chief adopts the narrower reading as the operating rule tonight, because a run does not become a test by being called one, and puts the difference to the Director as one question: whether an instrument pre-check or calibration that produces a new number counts as a test run. The quantisation-gap measurement is verification of the instrument and would be a test run on either reading, but its 8 GB pull is a download and is put to the Director separately.

The earlier banner (below, kept for the record) held that nothing was settled for the Deputy or the Head; that is superseded for test runs by the relay above, on the narrower reading, and unchanged for full experiments.

# STATUS 2026-09-05 21:xx: SUPERSEDED BANNER OF EARLIER TONIGHT: NOT SETTLED FOR THE DEPUTY OR THE HEAD. READ BEFORE ACTING.

The standing lift below is the Director's instruction as the Chief heard it directly in the Chief's own chat session. The Proxy has since relayed that the Director confirmed the veto lift but did not confirm the removal of the per-run gate, and that it is to be treated as open until the Director addresses it specifically. Under the Proxy standing order that narrowing carries the Director's authority for every session other than the Chief's. Therefore, until the Director confirms the standing lift to the Proxy, the Deputy or the Head directly:

- the per-run gate stands for the Deputy and the Head; WP3's run, WP1's validation run and every wave-2 run wait on one line from the Director;
- design work, re-specifications, unit fixtures without a model load, and commits under the ordinary gate proceed;
- the Chief's own runs under the direct instruction remain the Chief's responsibility. One such run has been made: the 17.6 s reconstruction calibration of 2026-09-05 evening (section 8 of the WP3 design), with no other model process alive.

This banner is removed only when the Director's confirmation is on the record.

# Standing lift for experiments (Director, 2026-09-05, evening)

Verbatim intent, from the Director in chat to the Chief:

- Standing lift authority: the Chief does not ask per experiment. Once an experiment is ready and validated (pre-registered reading, powered by the WO-STAT-001 method, instrument fixture green), it may run.
- The one operating constraint: experiments that load the model must not run concurrently. One full model-loading experiment at a time on this machine.
- All results must be reported to the Director.
- Visualisations the Director can return to on a laptop are the next most important deliverable after the results themselves.
- All recommendations D1 to D6 of `pending/03-REVISED-PLAN-2026-09-05.md` are approved.

Consequences recorded by the Chief:
- Rulings R40a (hosted lens is the reading instrument) and R40b (band-based probe layers 13, 16, 20, 24, 28 with the kind-matched partner rule) are entered in the wiring map.
- The EXP-002 lift note, the EXP-001 standing lift and the per-run lift requests are superseded for experiments by this standing lift; training lifts are not covered by this note and keep their own gate.
- The Proxy standing order is unchanged.
- Trace collection under EXP-005 falls under this lift with the D4 cap of 40 episodes per batch.
- Concurrency rule of record: before any run, the runner checks that no other model-loading process is alive (`pgrep -f` on the probe and training entry points) and refuses if one is.

## Concurrency clause (added 2026-09-05 evening at the Deputy's request)

Who checks: every session that can launch a model-loading process, the Deputy, the Head, the Chief, the Proxy when relaying a launch, and the Director's own terminal.
When: immediately before launch, in the launching command itself, not earlier.
How: (1) a process-list check for the probe and training entry points (`agent-v2-*`, `run_hosted_lens`, `contrast_reactions`, `mlx_lm`), and (2) a lock file at `outputs/.model-run.lock` carrying the launching session's name, the command and a timestamp, written before the model loads and removed on exit. A launch that finds a live process or a lock younger than the longest expected run refuses and reports, rather than queueing silently. A stale lock (older than the run it names) is reported to the Chief, not deleted by the launcher.
Status of this clause: the lock-file mechanism is issue 83, a standalone slice ahead of WP1 because it guards WP1's own first run; it is taken with an exclusive create (never a check followed by a write), released on error paths as well as success, reported and never deleted by a launcher when stale, and proven by a fixture in which two processes contend and exactly one wins. Until it lands, the process-list check alone is the rule, every launch says in its log line that it ran it, and no model-loading run cleared under the test-run lift starts before issue 83 is in.

## Channel note

This note records the Director's words as heard by the Chief in chat. Under `live/HOLD-2026-09-05.md`, a lift reaches the Deputy and the Head only from the Director directly or through the Proxy; the Chief's relay of it does not by itself lift anything for them. The Deputy's confirmation request to the Director is pending at the time of writing.
