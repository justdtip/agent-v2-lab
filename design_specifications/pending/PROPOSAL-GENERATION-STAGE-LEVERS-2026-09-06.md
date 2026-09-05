# Which generation-stage levers are worth measuring, and which are not

To the Chief AI Research Scientist, from the Research Division, session `an-app-ac`.
2026-09-06. A proposal, not a run. Requested as: which levers in the generation stages,
preference-pair generation and the 180-task evaluation, are worth measuring for decode speed
under R48's 64k cap, since those stages are where inference time enters training time.

Grounded in the base evaluation artifact `outputs/agent-v2/evals/base-test.json` and the
evaluation loop at `pipeline/evaluate.py:136`, not in general advice.

## 1. The premise of the question does not hold, and that is the first result

**The 64k cap does not bind these stages at all.** The protocol windows context to
`keep_last = 2` (`pipeline/protocol.py:52`), so a rollout's context does not grow with the
trajectory. Working back from the measurement below, a step prefills on the order of 1,200
tokens. That is two orders of magnitude below the cap and, more importantly, sits in the flat
part of the decode curve measured last night, where throughput is 67 tokens per second at 4k and
65 at 8k. The decode knee at 16k is not reachable from here.

So no lever framed around long context is worth measuring for these stages. Quantised KV cache,
chunked prefill, cache eviction and the working-set arithmetic all address a regime the windowing
paradigm already keeps us out of. That is the paradigm working as designed, and it is worth
stating plainly so nobody spends a night on it.

## 2. Where the time actually goes, from the base run

180 tasks, wall time 1:19:22, 26.5 s a task, 74,025 generated tokens, mean 9.52 steps.

| quantity | value |
|---|---|
| wall time | 4,762 s |
| generated tokens | 74,025 |
| effective rate | 15.5 tokens/s |
| decode at the measured 65 tokens/s | 1,139 s, **24% of wall** |
| everything else | 3,623 s, **76% of wall** |
| steps | 1,714 |
| non-decode per step | 2.11 s |

**Decode is a quarter of the evaluation. Prefill is most of the rest.** Every step re-prefills
the window from scratch, and 2.11 s a step at the measured 550 tokens per second is about 1,160
tokens, which is the right size for a two-exchange window with notes and tool output. So the
stage is prefill-bound, and a lever that improves decode speed improves at most a quarter of it.

**The tail is the other half of the story.** Latency is 6.74 s at the median and 80.0 s at the
95th percentile, a twelvefold spread. 32 tasks fail by exhaustion, and if those run the full 24
steps they account for **768 of 1,714 steps, 45 percent of all work, from 18 percent of tasks**.
The arithmetic closes: 32 x 24 plus 148 x 6.6 is 1,745 against 1,714 observed.

## 3. Levers worth measuring, ranked

**(a) Batch across tasks. The largest available and the only one not blocked.** The evaluation
loop is strictly sequential, one task at a time (`evaluate.py:138`). On a bandwidth-bound device,
batching amortises weight traffic across concurrent rollouts, which is the classic decode win,
and it also overlaps one task's prefill with another's decode. Measure: wall time for the 180-task
split at batch 1, 2, 4 and 8, with success rate asserted identical at temperature 0. Risk: the
per-task transcript and integrity machinery assume one trajectory at a time, so this is an
implementation change, not a flag.

**(b) Stop trajectories that are going to exhaust.** 45 percent of steps are spent on tasks that
fail anyway. A doomed rollout is detectable earlier than step 24: a repeated action with an
unchanged outstanding set is the loop signature, and no progress against the plan is the
exhaustion signature. Measure: distribution of the step index at which the eventual failure first
becomes predictable, over the base run's existing transcripts, which needs no model at all. If a
rule catches most exhaustions by step 12, that is a 20 percent cut in evaluation wall time for
free, and it costs nothing in success rate because those tasks fail regardless.

**(c) Prefix cache reuse, and why I am ranking it third rather than first.** The system prompt and
task prompt are constant across every step of a task, so the obvious lever is to cache that prefix
once and extend it per step. **This is precisely the case that is broken for this model class.**
The recurrent state cannot be split at an arbitrary boundary, so whole-prompt caching works and
incremental prefix extension does not. It is worth one hour to confirm the failure on our own
stack rather than inheriting it, because if it works the prize is most of the 76 percent. I expect
it to fail.

## 4. Not worth measuring, with reasons

**Quantised KV cache.** Verified last night as working for this model, and useless here: at a
1,200-token window the cache is around 37 MB and no part of the stage is memory-bound.

**Anything about chunk size, working set, or the context ceiling.** Section 1.

**Sampling temperature and `max_tokens`.** `max_tokens` is 200 a step and mean generation is 43
tokens a step, so the budget is not binding. Cutting it would truncate the tail rather than speed
the median.

## 5. One measurement change that costs nothing and is needed either way

The artifact carries `by_family`, `by_variant` and `by_difficulty`, but **each of those holds
success rate only**: for `aggregate_report`, successes, tasks, rate and a Wilson interval, and
nothing else. `loop_failures` and `exhausted` exist only as global totals, 20 and 32 on the base.

So the per-family, per-variant breakdown of loops and exhaustion still does not exist, and with
the new families deferred to the next run it matters more rather than less: this run is now the
**pre-intervention baseline** against which the families would be judged. Adding two counters to
the existing per-family aggregation is a small change in `evaluate.py`, needs no rerun of anything
already done, and without it the next run changes the data with nothing to compare against.

It also directly serves §3(b): knowing which families exhaust tells you where the 45 percent is.

## 6. Lever (b) is now priced, from the base artifact, with no model and no lock

Done while arm A trains, on `base-test.json`'s own 180 trajectories.

**All 32 exhausted trajectories run to exactly 24 turns**, and they are entirely confined to the
long-horizon families: `conditional_update` 14, `ledger_reconcile` 10, `aggregate_report` 3,
`batch_update` 3, `cross_reference` 2. No short family exhausts at all. They consume **2,479 s,
52 percent of the evaluation's wall time**, which is more than their 45 percent of steps because
they are also slower per step, 3.23 s against the mean.

**The obvious rule is unusable and the disciplined one is nearly free.**

| rule | fires on exhausted | median step | fires on successes |
|---|---|---|---|
| stop at the first repeated action | 27 of 32 | 8 of 24 | **50 of 145** |
| stop after three consecutive repeats | 27 of 32 | 10 of 24 | **1 of 123** |

Stopping at the first repeat looks better on the failures and destroys the run: a third of
healthy trajectories repeat an action once, legitimately. Requiring the repeat to persist across
three steps drops the false-positive rate from 34 percent to 0.8 percent while giving up only two
steps of median earliness.

**The trade, stated as a single choice.** Stopping after three consecutive repeated actions saves
**351 of 768 exhausted steps, 1,133 s, 24 percent of a 79-minute evaluation**, at a cost of one
success in 123, `test-cross_reference-0128-clean`, which fires at step 12 of its 14 and would have
been cut two steps from finishing.

That is a real cost and it should be a flag rather than a default, so the reported number is never
a truncated one. Its natural use is the rollout stage, where trajectories are sampled and
discarded anyway and a lost success costs nothing but a sample.

**One note on why the disciplined rule was not obvious.** This is the same shape as last night's
watchdog: a single critical sample is a transient and firing on it killed a healthy run, while
requiring the condition to hold across consecutive samples made the signal reliable. A repeated
action is a transient; a repeated action that persists is a loop. The two rules differ by one
word and by a factor of fifty in false positives.

— Research Division, session `an-app-ac`
