# A native residual source for the fitting path: a Gemma regression lens without the port

Patch: `design_specifications/pending/NATIVE-RESIDUALS.patch`. **Apply after `PILOT-PATH.patch`**,
which it is built on; it composes with `LENS-IDENTITY.patch` (both touch `scripts/lens_fit.py`, in
different places). Four files, 205 insertions. **Not landed.**

**876 passed, 1,097 skipped, exit 0**; `ruff check` clean on all four files. **The two new tests
are among the skips** — they live in `test_lens_regression.py`, which reaches the model library,
and the box is the Chief's. They are delivered unrun and I am not reporting them green. They cost
seconds and need no checkpoint, so they can ride in the Chief's suite.

## What it is

`ArchitectureView.native_residuals(ids, layers)` — the same signature and the same contract as
`residuals`, `{layer: (1, positions, hidden_size)}`, no cache, the final layer taken after the last
block and before the final norm. A different producer: one `NativeCapture` forward with a
collecting sink, so the entry transform, the mask construction and the per-block mask selection are
**the model's** rather than this class's.

That is the whole reason it unblocks Gemma. The hand-run loop omits Gemma's `sqrt(hidden_size)`
entry scale and builds one mask where the model builds two; capture wraps the model's own forward
and inherits all of it, exactly as the live-lens pilot does.

## The choice is explicit and it is recorded

`accumulate` and `fit_regression` take `residual_source`, one of `"hand_run"` or `"native"`, with
no inference from the model and a refusal on anything else. `scripts/lens_fit.py` exposes
`--residual-source`, and the value reaches the artifact twice: inside `counts`, by the same route
the sequence and position totals travel, and at the top level of the metadata so a reader deciding
whether two fits are comparable does not have to dig.

**A fit whose residual source changed silently would be unattributable afterwards.** The numbers
are the same shape from either producer and nothing else in the artifact would say which forward
made them. So the caller chooses and the record says.

## The substitution is demonstrated, not argued

`ArchitectureView.residual_source_agreement(ids, layers)` returns the per-layer maximum absolute
difference between the two producers, and the test asserts it is **exactly zero at every layer** on
the hybrid toy model — the family the loop was written for, so both mask kinds and both block kinds
are exercised.

Exact equality rather than a tolerance, deliberately: the same arithmetic in the same order should
give the same bits, and a tolerance would hide the drift this exists to detect.

**On Gemma the two will not agree until the port lands, and that is the point.**
`residual_source_agreement` then reports which layers the loop gets wrong and by how much, which is
the architecture port's acceptance evidence rather than a defect in this method. A fit that needs
residuals can run today; anything that perturbs a residual and re-runs a tail still needs the loop
and still needs the port.

## One change to existing behaviour, and why

The guard `if any(not value["positions"] for value in counts.values())` now iterates the two splits
by name rather than every value of `counts`, because `counts` gained a non-split entry. The intent
is unchanged and now stated: both fit and held require positions.
