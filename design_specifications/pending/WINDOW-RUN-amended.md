# WINDOW-RUN amended (issue 97) — the vacuous test, SIGTERM, and the separator

Patch: `design_specifications/pending/WINDOW-RUN-amended.patch`, against `3e894a8`. Replaces
`WINDOW-RUN.patch`. All three required amendments and all three smaller items are in it.

Verification, in a worktree at HEAD with the patch applied, run against the worktree's own source
(`PYTHONPATH` first, so the venv's installed copy is not what was tested): `tests/test_runlock.py`
57 passed, and with `test_repository_rules.py` and `test_spawn.py` beside it, 119 passed. `ruff
check` clean on both changed files. The `ruff format` diff is unchanged and untouched: every one of
its eighteen hunks is at or before line 1051 of `test_runlock.py`, in pre-existing `with`
statements, and the new code starts at 1366.

## 1. The window assertion was vacuous, and it is proven both ways

You were right about the mechanism and right that my no-op line reads as the intention. The test
now takes `unredirected_window_path`, keeps the `BOX_STATE_DIR_ENV` override, and asserts first
that `runlock.default_window_path()` **is** the path it goes on to assert about, so the two can
never drift apart again silently.

The command is now `/bin/test -e <window>`, so the child's exit status is the evidence that the
window existed while it ran. Without that, `not window.exists()` afterwards passes just as well on
a build that never opened one.

Proven in both directions rather than asserted:

- **The old body cannot fail.** I put it in a throwaway test file, under the suite's own fixtures,
  against a build whose `run` has `end_window` removed from its `finally`. It passed, including an
  added assertion that the window was sitting at the redirected path the whole time, unasserted.
- **The new body fails on that same build**, at `assert not window.exists()`.

## 2. SIGTERM and SIGHUP are relayed and closed over

`run_under_window` now installs the `_install_signal_release` shape: only dispositions still on
`SIG_DFL` are taken, so no caller's handler is displaced; on the signal the child is sent the same
signal, given five seconds, the window is closed, the default restored and the signal re-raised.
The handlers are restored in the `finally` beside the `end_window` that was already there. It uses
`spawn.popen` rather than `spawn.run`, because a handler cannot signal a child it has no handle to.

`test_run_ends_its_window_when_the_wrapper_is_killed` drives the CLI as a subprocess, waits for the
window file and a running holder, sends `SIGTERM` to the wrapper, and asserts the window is gone
and the wrapper died of the signal. With the handler installation disabled it fails on the window
still existing, which is last night in one assertion.

Five seconds is a stated constant with its reason beside it: the alternative to closing the window
on a child that ignores its signal is the failure the verb exists to prevent.

## 3. Only the leading separator is stripped

`argparse.REMAINDER` hands back the one separator and nothing else needs removing.
`test_the_cli_strips_only_the_separator_argparse_left_it` runs both spellings through `_window_cli`
with `run_under_window` captured and asserts the command's own `--` survives. With the old filter
restored it fails.

## The three smaller items

- **`RunLockBusy` at the CLI.** `announce` and `run` both catch it and print the refusal to stderr
  with exit 1. `test_a_refused_window_is_a_refusal_at_the_cli_not_a_traceback` plants another
  seat's window, checks both verbs refuse and name that seat, and checks the other seat's window is
  still there afterwards. It needed `unredirected_window_path` for the same reason the `run` test
  did, which is the second time that fixture was the difference between a test and a decoration.
- **A signalled child's status.** The CLI maps a negative return code to `128 + N`.
  `test_a_child_killed_by_a_signal_becomes_the_shells_own_status` runs `kill -TERM $$` under the
  wrapper through the real CLI and asserts 143 and an ended window. Without the mapping it fails.
- **Pid reuse.** `holder_state`'s docstring now says that `"running"` is evidence of a process at
  that pid and not proof that it is the holder, and names that as one more reason the report never
  removes.

## What I did not change

The `ruff format` diff, as you ruled. Nothing in the format-only hunks is mine, and I checked
rather than assumed: the hunk list stops at line 1051 and the amendment begins at 1366.

---

## Appended after the Chief's approval: the docstring note, taken

The Chief approved for landing with one note and no change required: after the five-second grace a
child that ignores its signal is left running with the window closed, and the docstring called that
outcome worse than either alternative without saying why it is safe.

Taken, because it is the more accurate sentence and the patch had not landed. The `relay` docstring
now says what protects the box in that case: **the child holds the model-run lock itself**, through
`load_weights`, so the interlock is intact whatever happens to the window. The window is a
declaration of intent between seats; the lock is the interlock. Closing an intent whose wrapper is
already dying is honest, and leaving it open is the failure this verb exists to prevent.

The clause about "worse than either" is replaced rather than extended, because it was the wrong
reason for the right design: the signal is relayed because it was meant for the command, not only
for the wrapper around it.

`WINDOW-RUN-amended.patch` is regenerated. **The Chief's review was of the previous file**, so this
one hunk — the `relay` docstring, no executable line touched — is new since it. 57 tests in
`test_runlock.py` pass again on the regenerated patch and `ruff check` is clean.
