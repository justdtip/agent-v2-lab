# P6 position groups over the windowed prompt (issue #29), review round 1, ratified (2026-09-05 00:10)

Reviewer: Claude (Chief). Evidence: issue #29, the reviewer's real-tokenizer measurements on
all five cases, and a direct read of `patch.py:409-545` (`_char_span_once`, `_token_span`,
`position_groups`).

## Verdict: APPROVED TO COMMIT with one small fail-closed guard (K1). No new R19 round.

Both causes are closed as described: groups are built over the windowed messages, and every
group is located by character span and mapped to tokens by prefix encoding with the merged
boundary widened by one token. The trailing-edge correction (content's final newline fusing
with `\n</tool_response>`) is consistent with the code: the end repair
`min(lcp + 1, len(ids))` includes the merged token, so each observation span ends on it.

## K1 (code, same file, Deputy verifies directly)

`_token_span` widens a boundary to the longest common prefix of the prefix encoding and the
full ids. If a tokenizer were not prefix-stable, or a span were mislocated, the common prefix
could fall many tokens short and the span would silently widen backwards. Bound it: raise
`boundary repair exceeded one token for <label>` when `len(head) - start > 1` or
`len(body) - (end - 1) > 1`. A fake with a two-token divergence must trip it. This converts a
silent mis-span into the loud failure the rest of the tool already prefers.

## Non-blocking, recorded

- Value spans are matched by plain substring inside the note region; a value whose digits
  occur inside an earlier value (`12` in `112`) aborts as ambiguous (fail-closed, correct) but
  a word-boundary match (`\b<value>\b`) would avoid the false abort. Fold into the R25 slice,
  which touches the same code.
- `_char_span_once` refuses any later duplicate of a located text anywhere in the prompt;
  strict but safe. Sequential location already fixes order, so uniqueness after `start` would
  suffice; leave as is unless a real case aborts.
- The transient `random control candidate pool` red seen twice inside the implementer's edit
  window and not reproduced: recorded; the R25 slice changes control cardinality and must
  re-verify that guard.

## Commit

Own commit referencing #29 and #26; Deputy pushes. The R25 alignment slice lands next.
