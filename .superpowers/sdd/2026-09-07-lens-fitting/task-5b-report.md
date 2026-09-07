# Task 5b — gated profiles and specificity

Bounded source implementation on `codex/lens-fitting`, persistent home worktree. Began
from reviewed Task5a `793c451`; paused for the producer's tokenizer fix `055f63b`, then
continued with its unchanged capture schema. Parent replay cleanup `ce219f7` is present.
Only the profile module, public profile script, its tests, and four prose-only lines in
the actual legacy atlas script were changed. Parent claims and unrelated work were preserved.

## Implemented behavior

`src/local_llm_lab/pipeline/lens_fitting/profiles.py` and `scripts/lens_profiles.py` emit
one model/domain/kind JSON, `specificity.json`, a standalone `profiles.html`, and an exclusive
`complete.json` with output hashes. All layer statistics stay separated by episode, episode
kind, span and summary/horizon. Every row has numerator, denominator `n`, exact unrounded
share, explicit span/target roles and a matching final-identity control. Zero denominators
are null. No primary layer or aggregate interpretation is chosen.

Foreknowledge targets only actual emitted events, with exact `(turn, source_position,
source_position+h, target_token_id, horizon)` final/nonfinal matching inside each episode.
Missing, extra, duplicate, wrong-turn, wrong-horizon and wrong-target pairs fail. The producer's
EOS convention is retained. Emitted spans use the actual legacy helper after each emission;
prose adds `continuation`. Authored descriptive tails never enter ranks. Prompt agreement
requires `t+1` inside the recorded prompt and exact tokenizer IDs/offsets. It uses the source
prompt-position span, including at a span boundary. Raw prose prompt positions use the
existing shared `chat` content label and explicit `source_prompt_position` role.

Regression h1 rows remain visible as biased by construction and ineligible for the reading
rule. Jacobian eligibility uses recorded per-layer `map_check`; failed/inconclusive rows
remain visible. Existing verdicts and inconclusive concept/output hooks are carried rather
than replaced. No rule-2 or rule-4 aggregate conclusion is inferred. The output conventions
state the distinct h4/h8 reference horizons and retain each horizon separately.

Specificity cells name training domain separately from evaluation material. Both fitted
domains and pilot-agentic/held-prose/pilot-chat material remain visible even when missing.
Chat is not a third fitted domain. Every held-prose cell states the ridge-selection overlap;
pilot evaluation is labeled disjoint generalisation.

## Evidence and input contracts

The CLI takes `--config <json> --out <new-directory>`. The configuration has `sets`, each
with `model` (registry name), `domain` (`agentic` or `prose`), `kind` (`regression`, `jacobian`,
or the explicit hosted comparison label `hosted-jacobian`), `lens` (NPZ path), `records`
(list of evaluation directories), `identity` (Task4 hosted-pilot identity directory), and
`instrument` containing `plan`, `corpus`, `proof`, and `runtime` paths. Paths are resolved
relative to the invoking working directory; absolute paths are recommended. `records: []`
produces explicit unavailable cells, not invented readings. The hosted comparison label
identifies the existing hosted Jacobian artifact; it adds no fitted method or domain.

All lens-set gates run before scientific aggregation. The instrument gate verifies the
actual corpus and upstream hashes, calls `read_plan` to reconstruct its frozen protocol,
verifies physical snapshot asset identity and dimensions, and calls existing pure
`require_self_check`. It also checks numeric receipt fields, seeds, finite positive epsilon,
and the predeclared self/stability bounds. A Jacobian sidecar must carry the same plan and
complete per-layer validation. Regression/hosted proofs establish the shared cached/reference
instrument path on the same snapshot, not regression Jacobian map agreement. Lens NPZ bytes
are hashed against their sidecar; matrices are not loaded by the profile path.

The identity gate anchors to the repository's actual pilot manifest, atlas and thirteen
record hashes. It recomputes exact decoded atlas equality, verifies both atlas hashes,
current summarizer hash, Task4 identity metadata, all source/output chains, current replay
provenance, complete membership/counts, and native forward/emission controls. It additionally
compares hosted replay observations to the canonical pilot events, excluding the intentionally
new record provenance. A supplied success boolean or arbitrary synthetic reference cannot
satisfy the production CLI gate. Test-local patched seams are explicitly synthetic fixtures.

For ordinary replays, new `replay-manifest.json` and `replay-complete.json` authenticate the
current lens, snapshot, layers, domain/kind and output record hashes. Historical `manifest.json`
is retained solely for source metadata/source hashes. Prose replays must trace to a complete
Task5a capture with all 51 held windows accounted for. The original prose adapter verifies
frozen prefix/window hashes, corpus membership, raw prompt alignment, accepted/dropped record
sets and counts, EOS/cap convention and all recorded provenance. Only accepted root episodes
reach statistics; every dropped record remains checked and listed as excluded.

Consumed file bytes, snapshot asset membership and complete record-directory membership are
revalidated before output and before completion. Output creation is exclusive and all JSON
rejects nonfinite values. Source commit, exact consumed file hashes, lens sidecar, current
record metadata, evidence paths, metric conventions and exclusions accompany the products.

## Fresh verification

Final focused command (pure, no MLX/checkpoints/native tests):

```
PYTHONPATH=src '/Users/daniel.tipton/Desktop/An app/.venv/bin/python' -m pytest -o addopts='' tests/test_lens_profiles.py -q
```

**38 passed in 3.56s**, exit 0 for pytest. A test autouse import guard rejects `mlx`/`mlx_lm`
and verifies that MLX is absent. CLI subprocess checks use the repository's spawn helper.
Tests cover independently calculable pairs, episode isolation, zero denominators, boundary
roles, exclusions/eligibility, corpus-plan/numeric-proof/snapshot/lens drift, current replay
hash authority, invalid prose completion, forged identity refusal, gate ordering, exclusive
output and output hash membership, and script-termination-safe HTML data. The actual embedded
page JavaScript ran under a pure Node DOM fixture: episode selection changes the displayed
rows, one episode stays selected, and the matched final base-rate line is drawn. This is
functional script evidence, not a browser screenshot or visual-appearance review.

Owned-file `ruff check --no-cache`, `ruff format --check`, and `git diff --check` pass.
The four-line atlas change was intentionally not reformatted. Early test runs demonstrated
the absent module, then exposed and fixed a JSON string-path hashing error and completion
self-hashing bug. Fixture-only corrections included the approved spawn helper, complete toy
plan inputs and tokenizer-only offline snapshot patterns.

### Separate actual legacy control

This is distinct from synthetic source tests and from the still-pending native replay gate.
A fresh exclusive control is preserved at:

- `.superpowers/sdd/2026-09-07-lens-fitting/task-5b-legacy-control/atlas.json`
- `.superpowers/sdd/2026-09-07-lens-fitting/task-5b-legacy-control/evidence.json`

The real 13 decompressed pilot records were checked against their canonical compressed
sources and hashes. The actual updated `scripts/live_lens_atlas.py` ran with the already
cached tokenizer whose asset hashes are in the evidence. Its result equals the original
atlas as decoded JSON **and bytes**, SHA-256
`28f5f4d558bf9a501e1b489419367145b71e904fe8f7450f046bd0e3be3fca41`.
The 64,152 rank rows and 76,628 reading rows are the original legacy control only. Original
pilot records, previous control and tokenizer files are unchanged. An import guard confirms
zero MLX imports and no checkpoint load. The evidence names the new summarizer SHA and every
consumed control/source/tokenizer file hash.

## Remaining acceptance and handoff

No native self-check, native hosted replay identity, fitted-lens replay, new prose generation,
scientific profile run, model load, full suite, download, merge or push was performed here.
The native gates must genuinely pass before the CLI can publish scientific profiles. No
scientific result was fabricated to fill those cells. Parent owns that serialized runtime
queue, integrated verification and backup push; independent review owns the source decision.
The source/report and real legacy-control evidence are ready to freeze for review.
