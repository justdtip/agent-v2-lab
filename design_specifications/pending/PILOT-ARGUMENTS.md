# The pilot's pinned constants become arguments

Patch: `design_specifications/pending/PILOT-ARGUMENTS.patch`. **Apply after `PILOT-PATH.patch` and
`LENS-IDENTITY.patch`**, both of which it sits on; it touches two files and nothing they touch.
Work orders, the Deputy's item 2, fourth bullet. **Not landed.**

**884 passed, 1,095 skipped, exit 0** with all three applied, and the four new tests ran — they
reach no model library, which is the point of the last one.

## What changed

`LENS`, `LENS_SHA` and `REGISTRY` were module constants pinned to Qwen, so `--model` moved nothing.

- **`--lens` and `--lens-sha256` are required.** There is no defensible default now that two
  lenses are on disk, and the old constants were the only thing preventing the Qwen lens from
  being loaded onto a model of the same width. Issue 99 makes that a refusal; this makes it a
  question the caller has to answer.
- **`--registry` defaults to `configs/models/<model>.yaml`**, derived from the resolved spec, so a
  run cannot read one model's band while running another's. An explicit override stays for a
  scratch registry, and the manifest records which file was read and its digest.
- **A model with no declared band no longer crashes.** `read_band` raises `KeyError` on a registry
  with no `probes.live_lens_pairs`, which is Gemma today. Where a band is declared it is read and
  validated against the installed block kinds exactly as before; where it is not, `--layers` is
  required and the manifest records `band_declared: false`, so a reader is told rather than left
  to infer it from an empty list.

## The property the tests pin

`test_live_lens_pilot_arguments.py` runs the script as a subprocess in its plan-only path and
asserts that **planning imports no MLX at all** — checked twice, once through the script's
behaviour and once by importing the module in a child and inspecting `sys.modules`.

That is worth pinning rather than assuming. Planning resolves tasks from the factory and renders
prompts to count their tokens, which is exactly the kind of work that acquires a tokenizer import
by convenience, and one such import would make the stage take a model slot it does not need. It
also means the pilot's plan can be built beside another seat's run, which is how this patch was
written.

The tests never open a lens, and the path they pass does not exist. That is deliberate: the
plan-only path returns before the lens is loaded, and a test that needed the real artifact would
be untestable on a fresh checkout and would silently skip, which is how the last set of
never-running tests came about.

## What is still pinned, and why it should be

`SEED = 20260902` stays a constant. It is the evaluation's data seed and therefore a property of
the task set rather than of the model, and every episode label in `AGENTIC` is resolved against it.
Making it an argument would let a run silently draw different tasks under the same labels.
