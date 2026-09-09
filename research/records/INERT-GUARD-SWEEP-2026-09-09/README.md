# The inert-guard sweep: three shapes, 220 files, one live hit and it was already fixed

**D-CRO, 2026-09-09**, on the Chief's order after a third inert guard turned up in two days. The
scanner is `tests/test_repository_rules.py`; this record is the first run of it and the assessment
of every hit, including the ones that are fine, because "no hits" from a scanner nobody has proved
is exactly the failure being swept for.

## The family

Three shapes have appeared across two streams and they are one thing: **a check that is present and
inert**. Not absent — present, so a reader who greps for a guard finds one, and silence gets read as
evidence.

- SWE-2's: a guard that passed the object it existed to catch, so the early error never fired.
- SWE-2's: an assertion that an artefact *loads*, under a reader that treats missing weights as a
  warning, satisfied by a checkpoint that had learned nothing.
- Mine: an assertion that cannot fail, because `==` binds tighter than `or`.

SWE-2's generalisation is the sharper form and is adopted here: the failure is not an absent check.

## What was scanned, and how

Parsed, not grepped. The first draft of the pin test in `tests/test_fixed_history_seam.py` searched
for the string `or True` and failed on the docstring explaining the shape — prose describing an
inert guard is not one, and a substring search cannot tell them apart.

| shape | detection |
|---|---|
| `assert <cmp> or True` | `Assert` whose `test` is a `BoolOp(Or)` with a truthy `Constant` operand |
| `assert <constant>` | `Assert` whose `test` is a truthy `Constant` |
| `assert (x, y)` | `Assert` whose `test` is a non-empty `Tuple` — ruff's F631 |
| broad `except` that swallows | `ExceptHandler` on `Exception`/`BaseException`/bare whose body is only `pass`, `continue` or `break` |

Ruff was run as an independent second net over the same files: `F631,B011,S110,S112,BLE001`.

**The scanner was proved before it was believed.** Against the pre-fix
`scripts/fixed_history_lens.py` it finds the one hit at line 207. Against a synthetic file carrying
all four shapes it finds all four, and does not flag a broad handler that re-raises.

## The result

| scope | files | hits |
|---|---:|---:|
| `src/`, `scripts/`, `research/*.py` | 100 | **0** |
| `tests/` | 76 | **0** |
| `research/records/` (scanned, reported, not changed) | 44 | **2** |

`research/acceptance` does not exist; the two loose scripts at the top of `research/` were scanned
in its place.

The zero in the first row is a real zero and not an empty scanner: the only live hit in that scope
was `scripts/fixed_history_lens.py:207`, fixed at `5820a33` before this sweep ran, and the scanner
finds it in the pre-fix file.

## Every hit, assessed

**`research/records/TRAIN-COST-2026-09-05/train_cost_probe.py:176` — benign.** The swallowed call is
`mx.get_peak_memory()` for a diagnostic field, inside an outer handler that has already recorded the
exception type, message and traceback tail. What the swallow costs is one absent field, and an
absent field does not read as a passing one. Not changed; it is a record's producer.

**`research/records/jlens-hosted-qwen35-4b-2026-09-05/run_hosted_lens.py:36` — benign in effect, but
the question is not recoverable.** The swallowed call is `mx.set_cache_limit(2 GiB)`, a declared
memory discipline. If it failed, the limit was never applied and nothing said so, which is the R60(c)
shape — a declaration that does not carry its measurement. Measured now: on MLX 0.32.2 the call
succeeds and returns the previous limit (24,481,313,587 bytes), so the handler only fires on a
runtime without the call at all. **The record does not name the MLX version that ran**, so whether it
fired cannot be recovered from the record. The consequence is bounded to memory headroom and touches
no number in that record. A verification note is added there; the script is not changed.

**`research/records/CTX-EFFICIENCY-2026-09-05/ctxbench.py:80` — benign** (ruff's net, not the shape
scanner's). It catches broadly and writes `{"ok": False, "error": "<Type>: <message>"}` into the row.
The failure is recorded, which is the opposite of inert.

**Four broad handlers in `src/`, all reporting — no change made.** Ruff's `BLE001` is a wider net than
the shape scanner and catches handlers that do something. Each of these records its failure in its
return value, so none is inert:

| site | what it does instead of swallowing |
|---|---|
| `pipeline/preflight.py:882` | falls back to finite difference and **reports `method`** in the result |
| `pipeline/preflight.py:1013` | returns `None` with a reason naming the exception type; the registry value stands |
| `probes/patch.py:1232` | returns `ScoredGeneration("parse_error", …)` and keeps the raw head |
| `probes/patch.py:1375` | returns `""` for token text; the artefact records the ids regardless |

**One of them is worth naming as a risk rather than a defect.** `probes/patch.py:1232` catches
`Exception` around `strip_thinking` and `parse_turn`, which raise `ActionParseError` and nothing
else. A `TypeError` or `AttributeError` from a bug in the pipeline would therefore be **scored as the
model failing to parse** — a programming error entering the record as a measurement about the model.
Narrowing it to `ActionParseError` is the fix and it is one line with one test. Not done here: the
plan defers the probes explicitly, and `patch.py`'s donor-difference core is scheduled to migrate
onto the new `intervene` API. It belongs in that migration, and it is written down so it arrives
with it rather than being rediscovered.

## What this sweep does not cover

The three shapes are the ones that have actually occurred. They are not the family — SWE-2's first
two, a guard that passes the object it exists to catch and an assertion satisfied by a tolerant
reader, are **not detectable by parsing**, because whether a check discriminates depends on what it
is checking against. Those stay a reading problem, and the technique for them is the one in the WS-D
record: construct the input the check should fail on, and confirm it does.
