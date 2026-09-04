# P6 position-groups fix — windowed view and char-offset token spans

Date: 2026-09-04. Branch `codex/agent-v2-specs`, base `07c6657`, uncommitted. Paths touched:
`src/local_llm_lab/probes/patch.py`, `tests/test_patch.py`, this report. No model execution
(one tokenizer load in the smoke test); nothing written under `outputs/`, `data/`, `reports/`,
or `pending/`.

## 1. Root cause — confirmed, both parts, on the real tokenizer

Reproduced offline with the cached 3B tokenizer only, on the first selected case
`test-ledger_reconcile-0031-clean` (decision step 7, 16 replayed messages, 7 tool observations,
2287 prompt tokens):

| text | in rendered prompt | standalone ids are a substring |
| --- | --- | --- |
| system, task, notes 0–6 | yes | yes |
| observations 0–4 | **no** (hidden stubs) | no |
| observations 5–6 | yes | **no** |

- **Cause A (windowing).** `_message_contents` read every tool observation from the full
  replayed list, while `build_prompt` renders `window_messages(messages, keep_last)`
  (`src/local_llm_lab/pipeline/protocol.py:157-173`, applied at `:271`), which replaces all but
  the last `keep_last` observations with `hidden_observation` stubs. The first observation
  raised `missing token span for observation`.
- **Cause B (context-dependent tokenisation).** The template renders a tool message as
  `\n<tool_response>\n{content}\n</tool_response>`; a boundary token of the content merges with
  an adjacent template newline, so `encode(content)` is not an id-substring of `encode(prompt)`.
  *Corrected after R19 review (measured by the reviewer and the Deputy on case 0031): on both
  real observations the start boundary is clean and the merge is at the trailing edge — the
  content's final newline fuses with the closing tag's leading newline — so the recorded span
  ends one character past the content (`'\n'`), which is the merged token and cannot be split;
  the earlier "first token merges with the preceding newline" wording was wrong.* The
  other groups only resolved because their boundaries happened not to merge (briefing trap 6).

Refinement: on the real prompts, only `last_two_observations` merges (2 repairs per prompt);
system/task/notes/values need 0 repairs. The fix applies the repair to all six groups anyway.

## 2. The fix (`src/local_llm_lab/probes/patch.py`)

1. **Group over the windowed view.** `_message_contents(messages, *, keep_last)`
   (`patch.py:701-731`) builds `window_messages(...)` (`:710`) and returns as observations only
   tool messages whose content is verbatim in the windowed view (`:725-730`); stubs are not
   observations. `_groups_for(tokenizer, ids, messages, *, prompt_text, keep_last)`
   (`:766-790`) threads `keep_last` through (`:775`) and returns `(groups, repairs)`.
   `run_patch_probe` passes the same `keep_last` to `build_prompt` (`:920-921`) and to
   `_groups_for` (`:924-929`), so the searched view is the rendered view.
2. **Char offsets, then tokens.** `_find_once` (id search) is replaced by
   `_char_span_once` (`:416-425`, unique character span, same "missing"/"ambiguous" refusals)
   and `_token_span` (`:437-468`): the prompt prefix up to the span start and up to the span
   end is encoded and its length taken as the token index; when the prefix ids are not a prefix
   of the full prompt's ids, the boundary is widened to the merged token via the longest common
   prefix (`:459`, `:464`) and a repair is counted (`:460`, `:465`). `position_groups`
   (`:471-544`) applies this to all six groups, takes `prompt_text=` and an out-param `stats=`
   for per-group `boundary_repairs`; `note_value_tokens` are located inside the last note's
   character span (`:515-527`); `last_two_observations` is the last two verbatim observations
   (`:529-533`). The at-least-one-token refusal stays (`:789-790`).
3. **Artifact.** Each headline case record carries
   `boundary_repairs: {failing: {group: n}, counterfactual: {group: n}}` (`:930-935`).

No offset-mapping shortcut is used; the prefix-count method is the only path and is the one
tested against the merging fake. `_encode` (`:409-413`) keeps the `add_special_tokens=False`
fallback the old `encoded` helper had.

Decisions taken (briefing rule 1.10): `build_prompt` in `run_patch_probe` now receives the
CLI's `keep_last` instead of the module default (identical at the Director's `--keep-last 2`);
the char-span search for observations restarts after the task prompt (`:529`), as before.

## 3. Tests (`tests/test_patch.py`) — fake-only, red first

Fakes: `_MergingTokenizer` (`:1689-1721`; token rule `\n[^\n]|\n|[^\n]` at `:1692`, so
`encode("\n"+x) != encode("\n")+encode(x)`; template wraps every message in newlines and
appends `generation_suffix(spec)` rather than a literal), `_CharacterTokenizer` (`:1883-1886`,
context-free, same template), `_windowed_messages` (`:1724-1744`, three observations >
`keep_last`=2, counterfactual differs only in the last note's value), `_covering_positions`
(`:1747-1754`, expected tokens from the fake's own offsets).

| requirement | test | lines |
| --- | --- | --- |
| (a) more observations than `keep_last`, through `run_patch_probe` with real `build_prompt`/`window_messages`/`position_groups` | `test_run_patch_probe_resolves_groups_over_the_windowed_prompt` (parametrised on both fakes; asserts `boundary_repairs` in the payload) | `:1888-1956` |
| (b) merging fake defeats id search, char mapping succeeds with repairs > 0 | `test_groups_for_uses_the_windowed_view_and_char_offsets_with_a_merging_tokenizer` (id-substring assertion `:1772-1773`; repairs `:1779-1786`) | `:1757-1815` |
| (c) six groups equal the tokens covering each text | same test, `:1788-1815` | |
| (d) ambiguity, missing, note-value refusals; hidden observation refused as missing | `test_position_groups_refuses_ambiguous_and_missing_text_on_the_rendered_prompt` | `:1818-1864` |
| `keep_last` threading (1 → one observation; 0 → fail closed) | `test_groups_for_windows_with_the_threaded_keep_last` | `:1867-1880` |
| real-tokenizer smoke (skip-guarded on the saved evals `:1969-1970` and the cached snapshot `:1972-1974`; `cli._load_data_tokenizer(spec.hf_id)` at `:1978`) | `test_real_tokenizer_resolves_the_six_groups_on_the_first_selected_case` | `:1959-2000` |

Existing tests updated for the new signatures: `test_position_groups_are_exact_and_fail_closed`
(`:722`, adds `prompt_text=`), `test_groups_for_extracts_canonical_family_value_tokens` (`:778`,
tuple return, zero repairs), `test_patch_probe_routes_every_group_and_control_with_exact_trace`
(`:1388-1405`, case records now carry zero `boundary_repairs` because `position_groups` is
faked there).

**Red-first proof.** With `patch.py` stashed at HEAD and the new tests in place,
`uv run pytest tests/test_patch.py -k run_patch_probe_resolves_groups` failed both
parametrisations: `_CharacterTokenizer` with `ValueError: missing token span for observation`
(the Director's live error, cause A alone) and `_MergingTokenizer` with
`ValueError: missing token span for system_prompt` (cause B trips the id search on the very
first group). The other new tests failed on HEAD with `TypeError` on the new `prompt_text=`
keyword.

## 4. Verification

- `uv run pytest -p no:cacheprovider -rs`, run bare: **exit status 0, 653 passed, 0 skipped**
  (the smoke test ran, not skipped). The `-q` in `pyproject.toml:54` is the only addopt.
- Banned-constant scan (`tests/test_repository_rules.py:222-239`) passes; no model literal
  was added to `patch.py`.
- Director's command (`ISSUE-22-P6-PROVENANCE-REPORT.md:163-170`) parsed with the same
  argument set and its selection stage run read-only: **5 cases**,
  `test-ledger_reconcile-0031/0127/0139/0163/0175-clean`, unchanged.
- Real-tokenizer result on all five cases, failing and counterfactual prompts (no weights): all
  six groups resolve on every prompt. First case, failing prompt (2287 tokens):

  | group | tokens | boundary repairs |
  | --- | --- | --- |
  | system_prompt | 359 | 0 |
  | task_prompt | 69 | 0 |
  | previous_notes | 659 | 0 |
  | note_value_tokens | 5 | 0 |
  | last_two_observations | 967 | 2 |
  | final_token | 1 | 0 |

  Counterfactual prompt of the same case: 2291 tokens; previous_notes 663, note_value_tokens 7,
  other groups identical; repairs identical.

## 5. Next blocker for the Director's run — not fixed here, out of this slice

`run_patch_probe` requires treatment source and target groups to have equal cardinality
(`patch.py:955-956`). On the real cases the counterfactual note is longer than the failing note
(first case: `previous_notes` 659 vs 663 tokens, `note_value_tokens` 5 vs 7; every case shows
the same pattern in the verification above), so the live run will now stop at
`treatment source and target group cardinality must match` for the `previous_notes` and
`note_value_tokens` cells. This is a design question (how to patch unequal note spans:
truncate, `_match_rows`-style resampling as the unrelated control does at `:964-967`, or
value-token alignment) and needs a ruling before implementation. Recommend opening the issue now
so the run is not blocked twice.

## Addendum (Deputy, ~20:20) — ratified and committed `ac9c27a` (#29)

The Chief's one condition is in: `_token_span` raises `token span for … moved more than one
token at a boundary` when either edge moves by more than one token from the naive prefix
length, so a non-prefix-stable tokenizer or a mislocated span fails loudly instead of widening
backwards. Red-first with `_ParityTokenizer` (ids depend on text-length parity, common prefix
collapses to zero). After the guard, all five selected cases resolve on both prompts with two
repairs each and the guard never fires. Suite 654 passed, exit 0.
