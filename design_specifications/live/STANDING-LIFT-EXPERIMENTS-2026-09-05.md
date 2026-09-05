# STATUS 2026-09-05 21:xx: NOT SETTLED FOR THE DEPUTY OR THE HEAD. READ BEFORE ACTING.

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
Status of this clause: the lock-file mechanism is a WP1 deliverable; until it lands, the process-list check alone is the rule and every launch says in its log line that it ran it.

## Channel note

This note records the Director's words as heard by the Chief in chat. Under `live/HOLD-2026-09-05.md`, a lift reaches the Deputy and the Head only from the Director directly or through the Proxy; the Chief's relay of it does not by itself lift anything for them. The Deputy's confirmation request to the Director is pending at the time of writing.
