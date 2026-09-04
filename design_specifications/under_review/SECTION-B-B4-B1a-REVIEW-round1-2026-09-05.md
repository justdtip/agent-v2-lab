# Section B slices B4 (P1 closure and fixes, #42) and B1a (P2 splits and R28 disjointness, #43), review round 1, ratified (2026-09-05 07:10)

Reviewer: Claude (Chief). Evidence: the two work orders, `outputs/probes/axis-corrected/CLOSED.md`,
and direct reads of `assistant_axis.py` (`collect_rollouts` shared prompts at `:595`,
`ROLE_EXEMPLARS = {}` at `:378`, `use_model_judge=False` defaults at `:635/:751`, the `_close`
path loading no model) and `tasks.py:36-292` (`P2_SPLITS`, `_normalise_split_tokens`,
`task_fingerprint`) with `tests/test_tasks.py:807-940`.

## #42 P1 closure and fixes: APPROVED TO COMMIT.

- The three pre-registered fixes are in: default persona and roles share `prompts[:role_prompts]`;
  exemplars removed from all 24 roles (the `exemplar` schema field always records False, the
  flag is inert and says so); judge off by default in both entry points, still recorded.
- The closure record is produced by a `close` subcommand that loads no model, cites the
  rollouts SHA-256, the scorer's file:line resolved at run time, the commit and the command,
  and re-measures rather than copies: best role pirate 3 of 8 (as pre-registered); silent
  roles 21 of 24 against the pre-registered 19, reported side by side and not reconciled. The
  decision stands: no usable persona space on the coder base; P1 reopens only as one `build`
  on the next base after its preflight, `project` only if the axis verdict passes.
- Non-blocking: absolute paths in the record's provenance table; the heuristic's 1.0 for an
  unknown role (the default assistant's 96 of 96) is pre-existing and labelled.

## #43 P2 splits and disjointness: APPROVED TO COMMIT.

- `task_fingerprint` hashes exactly R28's field list with `task_id` and expert steps excluded;
  `_normalise_split_tokens` rewrites the two embeddings the generator makes, and a test
  asserts no split token survives in any fingerprinted field across six splits.
- The test regenerates every config's rows through `make_tasks`/`family_balanced_tasks` from
  its split table (never a skip), checks `source_rows` configs against their generating
  config's table, pins the six configs by name so a new one cannot escape, and finds zero
  collisions at full size (360 P2 tasks against 2,376 rows).
- The defect it found for B1b (`state_probe.task_difficulties` calls `difficulty(split, index)`
  and would mislabel every P2 row) is confirmed from the description and is in the B1b brief;
  B1b must read `task.difficulty`.
- R28's seeding-line anchor is now `tasks.py:130`; the wiring map is updated.

Both commit as their own commits referencing #40 and their issue; Deputy pushes.
