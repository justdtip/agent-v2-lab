# R17 first half: substring pass for the banned-constant scanner

Implementer: Opus subagent, 2026-09-04. Revised same day after the independent review round
(§7). Paths owned and touched: `tests/test_repository_rules.py` only (+193 / −2). No file under
`src/` was read for edit or written. No model was loaded. Nothing was committed; the change is
in the working tree.

## 1. What was added

| Symbol | Location | Role |
| --- | --- | --- |
| `_FORBIDDEN_MODEL_CONSTANTS` | `tests/test_repository_rules.py:11` | Briefing §1.7 tuple, hoisted to module scope so the AST pass and the substring pass cannot drift apart |
| `_NUMERIC_TOKEN_SEPARATORS` | `tests/test_repository_rules.py:19` | the neighbour whitelist that defines "whole number" |
| `_docstring_constant_ids` | `tests/test_repository_rules.py:97` | identifies module/class/function docstring nodes so prose is skipped |
| `_contains_number_token` | `tests/test_repository_rules.py:113` | the boundary rule for numeric constants |
| `_is_projection_name_list` | `tests/test_repository_rules.py:128` | recognises a projection name standing as a cell of a comma-separated list |
| `_banned_model_substrings` | `tests/test_repository_rules.py:140` | the substring pass itself; its docstring states the rule |

The AST pass `_banned_model_assumptions` (`tests/test_repository_rules.py:74`) is unchanged and
still runs; the substring pass is additive. Its only edit is at
`tests/test_repository_rules.py:222-227`, where the inline `forbidden` tuple was replaced by the
shared `_FORBIDDEN_MODEL_CONSTANTS`.

## 2. The rule the substring pass uses

Stated in full in the docstring at `tests/test_repository_rules.py:141-168`. In summary:

1. **String literals only, docstrings excluded.** The pass walks `ast.Constant` string nodes, so
   comments and identifiers are invisible; docstrings are skipped because prose cannot become a
   runtime value. This exclusion is load-bearing on the live tree: `ridge_fit`'s docstring at
   `src/local_llm_lab/probes/state_probe.py:823` reads "a few hundred rows against 2048
   dimensions" while the function itself is dimension-agnostic (`n, d = features.shape`,
   `src/local_llm_lab/probes/state_probe.py:826`). Without the exclusion that line is the single
   repository-wide false positive.
2. **Exact matches are left to the AST pass** (`tests/test_repository_rules.py:180`), which
   reports them with more context and also covers non-string constants.
3. **A banned number counts only as a whole number token.** Both neighbours must be the string
   boundary or a character in `_NUMERIC_TOKEN_SEPARATORS` — list separators, brackets, quotes,
   whitespace, and (since the review round) the dash of a range spec such as `"1-35"`. Every
   other neighbour (digit, ASCII letter, `_`, `.`, `:`, `/`, `%`, `\`) means the digits belong
   to a longer token. The dash is a separator because a comma-free range default is a real
   evasion: `_hard_coded_probe_layer_defaults` (`tests/test_repository_rules.py:190`) skips any
   default without a comma, so nothing else sees `"1-35"` (verified in §7). Dates such as
   `2035-09-04` remain protected because their digits guard each other; `:` and `/` remain
   excluded so timestamps (`12:35:07`) and paths (`run/35/x`) never fire.
4. **Non-numeric banned constants match plainly.** `<|im_end|>` and `model.model.layers` are
   self-delimiting, so no boundary rule is needed.
5. **A projection name is reported when it stands as an exact cell of a comma-separated,
   multi-cell literal.** Any cell counts — the same any-member rule as the AST pass's
   `names & _PROJECTION_NAMES` (`tests/test_repository_rules.py:86-93`) — so an unknown cell
   such as `lm_head` cannot launder the list (§7 finding 1). Prose survives on cell shape, not
   on an all-cells rule: the live help string at `src/local_llm_lab/probes/adapter_delta.py:1018`
   ("Comma-separated blocks whose down_proj/o_proj update directions are read out.") contains no
   comma character at all — "Comma-separated" is a hyphenated word — so it is a single cell and
   fails the two-cell minimum; and a prose fragment between commas is never *exactly* a
   projection name.

## 3. Boundary cases tested, both ways

Catching:

- `test_banned_substring_scanner_catches_the_layer_list_default`
  (`tests/test_repository_rules.py:274`): the exact historical violation
  `parser.add_argument("--layers", default="6,12,18,24,30,35")` is asserted to yield `[]` from
  the AST pass and `["example.py:1:35"]` from the substring pass — the blind spot and its
  closure in one assertion.
- `test_banned_substring_scanner_catches_the_dash_range_default`
  (`tests/test_repository_rules.py:286`): `default="1-35"` yields `[]` from
  `_hard_coded_probe_layer_defaults` (no comma, so that guard skips it) and `["example.py:1:35"]`
  from the substring pass — the gap the dash separator closes.
- `test_banned_substring_scanner_reports_every_embedded_constant_kind`
  (`tests/test_repository_rules.py:295`): all five embedded kinds (`36` in a layer list, `2048`
  in `"hidden=2048"`, `<|im_end|>` in a longer string, `model.model.layers` inside a `getattr`
  string, a comma-joined projection list) plus the reviewer's measured evasion
  `"q_proj,k_proj,v_proj,o_proj,lm_head"`, which the earlier all-cells rule let through and
  which now fires (`example.py:6:projection-list`).

Not firing (`test_banned_substring_scanner_ignores_legitimate_text`,
`tests/test_repository_rules.py:321`), asserted to yield `[]`:

| Class | Fixture literal | Why it is ignored |
| --- | --- | --- |
| docstring prose | `"""A few hundred rows against 2048 dimensions; 36 blocks."""` | docstring exclusion (rule 1) |
| version strings | `"mlx-lm 0.31.3 needs 0.35.1"` | `.` neighbour |
| hex digest | `"a35f36c2048deadbeef1234567890abcdef2048ab"` | letter or digit neighbours |
| ANSI colour codes | `"\x1b[35m\x1b[36mwarn\x1b[0m"` | trailing `m` is a letter |
| dates and timestamps | `"2035-09-04T12:35:07"` | digit neighbour inside `2035`, `:` around `35` |
| longer numbers | `"20480 tokens, 1350 rows, 2360 steps"` | digit neighbours |
| inside a word | `"layer35 layer36 x2048y _35"` | letter or `_` neighbours |
| path segment | `"outputs/probes/run/35/x"` | `/` neighbours |
| percentage prose | `"35% of rows"` | `%` neighbour |
| fractional layer spec | `"0.35,0.5"` | `.` neighbour — required, since R17's second half makes `--layers` accept fractions |
| projection prose, no comma | `"Blocks whose down_proj/o_proj directions are read out."` | single cell, fails the two-cell minimum (rule 5) |
| projection prose, with comma | `"gate_proj is adapted everywhere, and the rest is frozen"` | no cell is exactly a projection name (rule 5, guards the any-cell change) |

The same fixture ends with a bare `bare = "2048"`, and the test asserts the AST pass still reports
it as `example.py:13:2048` (`tests/test_repository_rules.py:347`) — proof the substring pass's
exact-match skip did not create a new hole.

Existing allowances are inherited unchanged: `_discover_modern_python_sources`
(`tests/test_repository_rules.py:42`) already excludes `arch.py` and `models.py`
(`tests/test_repository_rules.py:21-26`, covering R2's `_default_spec` allowance), the legacy set
(`tests/test_repository_rules.py:27-39`), and everything outside `src/local_llm_lab/` and
`research/` — so `configs/models/` and `tests/` are out of scope by construction.

## 4. The expected-failure I was told to leave, and why I did not

**The instruction could not be carried out as written, because the violation it names no longer
exists.** Disclosed per briefing §1.10; the independent review endorsed the refusal (§7).

The task brief stated that the scanner would newly fail on a live `"6,12,18,24,30,35"` default in
`state_probe.py` and `assistant_axis.py`, and asked for `xfail(strict=True)`. R17's second half
landed first, in commit `e53bd81` "Implement model-aware probe defaults", whose diff removes both
literals (`- build.add_argument("--layers", default="6,12,18,24,30,35")` and
`- parser.add_argument("--layers", default="6,12,18,24,30,35")`). The three CLIs now declare the
flag with no default and resolve it from the registry:
`src/local_llm_lab/probes/state_probe.py:2866-2868`,
`src/local_llm_lab/probes/assistant_axis.py:1336-1338`,
`src/local_llm_lab/pipeline/jlens.py:506-508`, with parsing and validation in
`src/local_llm_lab/probes/policies.py:66-116`. A repository-wide grep for the literal returns hits
only in prose and in a saved manifest recording a past invocation
(`reports/probes-and-assessments-manifest.json:173`), never in source. The substring pass finds
zero violations across the whole modern source set, working-tree changes included.

A `strict=True` xfail on a passing test is `XPASS(strict)`, which pytest reports as `FAILED`. I
verified this against this repository's own helpers in a throwaway file: the marked test reported
`[XPASS(strict)] R17 second half: --layers defaults from the registry` and the run ended
`1 failed`. Adding it would therefore turn the suite red, breaking R13's green gate and the
task's own definition of done, and would assert a violation that is not there.

Both stated goals are met without the marker: the suite is green, and the violation stays visible
rather than silently tolerated, because
`test_banned_substring_scanner_catches_the_layer_list_default`
(`tests/test_repository_rules.py:274`) pins the exact historical literal as a regression fixture.
Reintroducing that default anywhere in `src/local_llm_lab/` or `research/` now fails
`test_banned_model_constants_are_not_hidden_inside_string_literals`
(`tests/test_repository_rules.py:230`).

**So: this report claims a pass, not an expected-failure. No `xfail` marker was added.**

## 5. Verification

```
$ uv run pytest
602 passed in 7.98s
exit code 0
```

Scanner file alone: `uv run pytest tests/test_repository_rules.py -q` → `10 passed`
(5 pre-existing, 5 added). Zero failures, errors, xfail, xpass, or skips.

`uv run ruff check tests/test_repository_rules.py` → `All checks passed!`. `ruff format --diff` on
the file reports one hunk, entirely inside the pre-existing `_banned_model_assumptions` body at
`tests/test_repository_rules.py:87-91`, which I did not touch (briefing §6: do not reformat what
you are not changing); the repository is not format-clean overall (39 files would be reformatted).

Grep for the banned constants (briefing §1.7, wiring-map §6 check 2) over `src/` and `research/`:
`<|im_end|>` — no hits outside `arch.py`/`models.py`; `model.model.layers` — no hits;
projection-name lists — only `src/local_llm_lab/train_grpo.py:92` and
`src/local_llm_lab/depth_expansion.py:109-115`, both legacy and both already excluded at
`tests/test_repository_rules.py:31` and `:37`; `36`/`2048`/`35` as embedded substrings — no hits
under the rule in §2 (re-measured after the §7 changes: 42 modern sources, zero findings).

## 6. Observed, not fixed

- **The working tree is live.** Other lanes were writing during this task:
  `tests/test_preflight.py` gained a test between two full-suite runs (600 → 601 passed, seconds
  apart; 602 includes my added test), and `git status` shows uncommitted work across ~20 files
  under `src/` and `tests/`. Every suite run was green; the count in §5 is the latest.
- **Two briefing §3 anchors have drifted further**, as that section's own header warns. The row
  citing `state_probe.py:456-469 (build_probe_dataset)` now resolves to
  `src/local_llm_lab/probes/state_probe.py:380`, and the row citing `capture.py:145-285` for
  `InjectionHook` and `lora_block_mask` now spans `src/local_llm_lab/probes/capture.py:155` and
  `:368`. The named symbols are still correct, which is what the header says to trust. The
  `MIX_PLAN` (`state_probe.py:105`) and `_strip_messages` (`state_probe.py:309`) anchors still
  hold. Not in my scope to edit (`pending/` is read-only under R3).
- **Wiring-map §6 check 2 and R2 still disagree** on whether tests may hold the end-of-turn
  literal; the Deputy's 2026-09-04 00:20 amendment reconciling them was promoted at 09:00 but §6's
  text was not updated. The scanner never scans `tests/`, so nothing is enforced either way.

## 7. Review round 2 (2026-09-04, after the independent review)

All reviewer measurements were re-verified independently before acting; two were confirmed, one
was not.

1. **Blocking finding, fixed.** `_is_projection_name_list` used `all(...)` over the cells, so
   `"q_proj,k_proj,v_proj,o_proj,lm_head"` evaded all three guards — confirmed by running the
   string through the AST pass, the substring pass and the layer-defaults pass (all returned
   `[]`). The rule is now any-cell (`tests/test_repository_rules.py:136-137`), matching the AST
   pass's set-intersection semantics, and the evasion string is a fixture that must fire
   (`tests/test_repository_rules.py:305`, expected at `:317`). Re-measured over all 42 live
   modern sources: zero new findings. A comma-containing prose fixture
   (`"gate_proj is adapted everywhere, and the rest is frozen"`,
   `tests/test_repository_rules.py:338`) guards the false-positive side of the change.
2. **Report correction, applied.** The previous revision justified the safety of
   `adapter_delta.py:1018` by the all-cells clause. That was false: the string contains no comma
   character, so it short-circuits on the two-cell minimum and the cell clause never runs. §2
   rule 5 and the §3 table now state the correct mechanism, which also survives the switch to
   any-cell.
3. **Optional separator widening, taken in part.** `-` was added to `_NUMERIC_TOKEN_SEPARATORS`
   (`tests/test_repository_rules.py:19`): re-measured at zero findings over the live tree, and it
   closes a real gap — the review's supporting claim that `_hard_coded_probe_layer_defaults`
   "already covers the --layers dash forms" is **incorrect**: that guard skips any default
   without a comma (`tests/test_repository_rules.py:210`) and rejects non-float cells like
   `"12-18"` (`:212-216`), so `default="1-35"` and `default="6,12-18"` both return `[]` from it,
   verified by direct execution. The new dash-range test (`tests/test_repository_rules.py:286`)
   pins both halves: that guard misses it, the substring pass catches it. `:` and `/` were
   **declined**: each measured zero on today's live tree, but whitelisting them flips two of the
   contract's enumerated innocent classes — `"12:35:07"` (timestamps) and
   `"outputs/probes/run/35/x"` (paths), both standing fixtures at
   `tests/test_repository_rules.py:331` and `:334` — into findings, trading documented
   false-positive classes for speculative evasions. The docstring records the choice
   (`tests/test_repository_rules.py:153-160`).
