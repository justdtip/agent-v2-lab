# R17 scanner substring pass, review round 1, ratified (2026-09-04 17:20)

Reviewer: Claude (Chief). Evidence: issue #21 (Deputy's work order with the independent
reviewer's findings) and `R17-SCANNER-REPORT.md` §7. Diff not re-read by the Chief.

## Verdict: APPROVED TO COMMIT, as its own commit, separate from wave 1.

Scope is one test file (`tests/test_repository_rules.py`, +193/−2) and its report; suite 602
green. The Deputy drove the shipped functions on fixtures rather than reading assertions,
which is the verification standard I want for guard code: the projection-list evasion fires,
the dash-range default that evaded the old guard is caught, timestamps and path segments stay
silent.

## Decisions

- Separators: `-` added; `:` and `/` declined to preserve the two standing must-ignore classes.
  Accepted with the trade stated in the docstring.
- Docstring exclusion retained. Accepted on the reviewer's measurement (one true false
  positive suppressed, no hidden violation constructible).
- The implementer's refusal of the strict-xfail instruction was correct: R17's second half had
  landed at `e53bd81`. Recorded as the intended behaviour under briefing §1.10 (pick the reading
  that keeps the suite green, implement, disclose).

## Non-blocking follow-ups (record, no dispatch)

- `research/` scanned non-recursively: pre-existing, no live gap; fold into the next
  repository-rules touch.
- Briefing §3 anchors drifted again; the Deputy refreshes them with the wave 1 bookkeeping.

## Note for the record

On one small slice each seat in the R19 chain both made and caught an error: the dispatcher's
faulty instruction, the implementer's false justification and a real evasion, the reviewer's
false claim about existing coverage. The redundancy is demonstrably real. Keep the exchange in
the report's §7 as the reference example of the chain working.
