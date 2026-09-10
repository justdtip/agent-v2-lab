"""Check that every figure the pre-registration states is a figure the scripts computed.

The failure this guards against is the week's own: a document that summarises correctly-computed
numbers and summarises them wrongly. Each claim names a path into one of the JSON artefacts and a
**line pattern** the document must match — the label and the value on the same line. The check fails
if the JSON does not hold the value, and it fails again if no line of the document carries the value
in its own context.

The first version of this file tested only that the value appeared *somewhere* in the document, and
a deliberate corruption of one table cell passed, because the same figure appeared seven times
elsewhere. That is the twenty-seventh entry's family — a check that cannot fail — inside the tool
written to guard against exactly that. So `--self-test` corrupts each pinned line in turn and asserts
that this script rejects the corrupted document; a claim whose corruption survives is reported and
fails the run. The guard is exercised rather than assumed.

    python .../check_prereg.py             # check the document
    python .../check_prereg.py --self-test # check the check

Exits nonzero on any disagreement, and prints every claim with its verdict either way.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: (claim, artefact, path into it, expected value, a regex that must match one line of the document).
#: The pattern carries the label and the value together, so a figure changed in its own row cannot
#: hide behind the same figure written correctly somewhere else.
CLAIMS: tuple[tuple[str, str, str, object, str], ...] = (
    ("rendered rows", "inputs", "rows.total", 8907, r"rendered rows \| 8,907"),
    ("agentic rows", "inputs", "rows.agentic", 8559, r"agentic rows.*\| 8,559"),
    ("chat replay rows", "inputs", "rows.chat_replay", 348, r"chat replay rows.*\| 348"),
    ("distinct decisions", "inputs", "decisions.total", 7629, r"distinct decisions\*\* \| \*\*7,629\*\*"),
    ("episodes", "inputs", "tasks.total", 1128, r"episodes \| 1,128 \(840 train, 48 valid, 240 test\)"),
    ("extra rows from oversampling", "inputs", "decisions.extra_rows_from_oversampling", 930,
     r"930 extra rows"),
    ("recovery decisions", "inputs", "decisions.recovery", 578, r"578 rows flagged `recovery`"),
    ("duplicated twice", "inputs", "decisions.rendered_rows_per_decision.2", 490,
     r"490 decisions duplicated twice"),
    ("duplicated six times", "inputs", "decisions.rendered_rows_per_decision.6", 88,
     r"88 duplicated six times"),
    ("repeats are identical rows", "inputs",
     "decisions.repeated_keys_that_are_not_identical_rows", {}, r"byte-identical row"),
    ("tasks with a dropped decision", "inputs",
     "a2_decisions_without_a_rendered_row.tasks_affected", 334, r"one short on 334 tasks"),
    ("transitions", "inputs", "e2_transition_census.total", 6501, r"6,501 transitions:"),
    ("ordinary transitions", "inputs", "e2_transition_census.by_kind.ordinary", 5948,
     r"ordinary \(step advances by one\) \| 5,948"),
    ("contiguous recovery transitions", "inputs", "e2_transition_census.by_kind.into_recovery", 244,
     r"\| into the corrective decision \| 244"),
    ("recovery across a gap", "inputs",
     "e2_transition_census.by_kind.into_recovery_across_a_gap", 309, r"no rendered row \| 309"),
    ("train ordinary transitions", "inputs", "e2_transition_census.by_split.train.ordinary", 4122,
     r"\| 5,948 \| 4,122 \|"),
    ("test ordinary transitions", "inputs", "e2_transition_census.by_split.test.ordinary", 1541,
     r"\| 285 \| 1,541 \|"),
    ("valid ordinary transitions", "inputs", "e2_transition_census.by_split.valid.ordinary", 285,
     r"\| 4,122 \| 285 \|"),
    ("test-split decisions", "inputs", "decisions.per_split.test", 1781,
     r"test split's 1,781 decisions"),
    ("exploratory episodes", "inputs", "r6_exploratory_episodes.count", 12,
     r"twelve pre-registered episodes"),
    ("capture set digest, the triples", "inputs", "decisions.capture_set_sha256",
     "1a7cfbdd4e21fc203eafbcc3ec50b96afcb214c169f21c7ba968ca83dfc9709a",
     r"\| the set \| `1a7cfbdd4e21fc203eafbcc3ec50b96afcb214c169f21c7ba968ca83dfc9709a`"),
    ("capture set digest, the file", "file", "capture-set.jsonl",
     "8bbc8062249ce1cc0e15050a866cd726a2c205a969f3eeab3840fcb5520cee3e",
     r"\| the file \| `8bbc8062249ce1cc0e15050a866cd726a2c205a969f3eeab3840fcb5520cee3e`"),
    ("E1 evaluation episodes", "inputs", "tasks.per_split.test", 240,
     r"E1 on the test split \| episodes \| 240 \| \*\*0\.23\*\* \| 234 \| 6"),
    ("E2 ordinary episodes", "inputs", "tasks.per_split.train", 840,
     r"train split \| episodes \| 840 \| \*\*0\.13\*\* \| 731 \| 109"),
    ("archive digest", "inputs", "corpus_archive_sha256",
     "7fe6e64b89749b997854638b260d7654316636da3d8eaa59924ad9ed62f99120",
     r"`7fe6e64b89749b997854638b260d7654316636da3d8eaa59924ad9ed62f99120`"),
    ("4B layers", "budget", "models.gemma3-4b-cuda-bf16.layers", 34,
     r"`gemma3-4b-cuda-bf16` \| 34 \| 2,560 \| 35 \| 175\.0 \| 1\.273"),
    ("4B hidden", "budget", "models.gemma3-4b-cuda-bf16.hidden", 2560,
     r"`gemma3-4b-cuda-bf16` \| 34 \| 2,560"),
    ("12B layers", "budget", "models.gemma3-12b-cuda-bf16.layers", 48,
     r"`gemma3-12b-cuda-bf16` \| 48 \| 3,840 \| 49 \| 367\.5 \| 2\.674"),
    ("12B hidden", "budget", "models.gemma3-12b-cuda-bf16.hidden", 3840,
     r"`gemma3-12b-cuda-bf16` \| 48 \| 3,840"),
    ("4B main stratum GiB", "budget", "main_stratum.per_model_gib.gemma3-4b-cuda-bf16", 1.273,
     r"175\.0 \| 1\.273"),
    ("12B main stratum GiB", "budget", "main_stratum.per_model_gib.gemma3-12b-cuda-bf16", 2.674,
     r"367\.5 \| 2\.674"),
    ("both, main stratum GiB", "budget", "main_stratum.total_gib", 3.947,
     r"542\.5 \| \*\*3\.947\*\*"),
    # The token figures are now measured from the 4B capture's own manifest, so they are pinned to
    # that artefact rather than to the budget script's character-based projection. The projection is
    # still in the document beside them, being scored, which is why the assumed numbers appear too.
    ("tokens measured, median", "capture", "median", 1310, r"\| median \| ~1,185 \| \*\*1,310\*\*"),
    ("tokens measured, max", "capture", "max", 4274, r"\| longest \| ~3,579 \| \*\*4,274\*\*"),
    ("tokens measured, min", "capture", "min", 416, r"\| minimum \| 456 \| \*\*416\*\*"),
    ("cells captured", "capture", "cells", 7629,
     r"in \*\*7,629 of 7,629\*\* cells"),
    ("exploratory positions measured", "capture", "exploratory_positions", 18018,
     r"18,018 positions\*\*, against the 15,776 assumed"),
    # The precondition the whole comparison rests on, pinned to the artefacts that will be read.
    ("identical ids across the two models", "both", "identical_token_ids", 7629,
     r"\*\*identical at 7,629 of 7,629 decisions\*\*"),
    ("identical read position", "both", "identical_read_position", 7629,
     r"read position is identical at"),
    ("terminal chain decisions", "chain", "terminal_with_marker", 94,
     r"all \*\*94\*\* terminal `pointer_chain` decisions"),
    ("terminal chain in test", "chain", "terminal_in_test", 20,
     r"\(20 of them in E1's test split\)"),
    # Numerator and denominator bound together, on the joined paragraph. The first version offered
    # alternatives, so 575 → 999 and 0 → 1 each passed by satisfying the other half (Codex C1).
    ("non-terminal without the marker", "chain", "non_terminal_decisions", 575,
     r"\*\*0 of 575\*\* non-terminal ones do"),
    ("12B peak reserved", "both", "peak_reserved_gib_12b", 26.453,
     r"\| 12B \| 2\.674 GiB \| \*\*2\.7 GB\*\* \| 42\.2 min \| 331\.6 \(shared\) \| 26\.45"),
    # The cross-fitting folds. The digest is what the seal fixes, so it is pinned to the line that
    # states it and the fold table is pinned row by row: a fold table that drifts from the
    # assignment would be a description of folds nobody used.
    ("fold assignment digest", "folds", "assignment_sha256",
     "c5e9622f7bef663c96614a99bab7e120873ad9eb79617912526c02e48016372c",
     r"`c5e9622f7bef663c96614a99bab7e120873ad9eb79617912526c02e48016372c`"),
    ("fold seed", "folds", "seed", 20260910, r"Seed \*\*20260910\*\*"),
    ("folds", "folds", "folds", 5, r"K = 5 folds, assigned by"),
    ("fold 0 size", "folds", "per_fold.0", 220, r"\| 0 \| 220 \| 108 \| 47 \| 38 \| 10 \| 11 \| 6 \|"),
    ("fold 1 size", "folds", "per_fold.1", 232, r"\| 1 \| 232 \| 111 \| 50 \| 38 \| 15 \| 12 \| 6 \|"),
    ("fold 2 size", "folds", "per_fold.2", 226, r"\| 2 \| 226 \| 110 \| 50 \| 34 \| 15 \| 11 \| 6 \|"),
    ("fold 3 size", "folds", "per_fold.3", 224, r"\| 3 \| 224 \| 112 \| 48 \| 34 \| 12 \| 12 \| 6 \|"),
    ("fold 4 size", "folds", "per_fold.4", 226, r"\| 4 \| 226 \| 109 \| 49 \| 38 \| 12 \| 12 \| 6 \|"),
    ("episodes in the folds", "folds", "episodes", 1128, r"K = 5 folds"),
    # The tolerance rows. Each pins its n beside the n it needs, because that pairing is the whole
    # point of the table and the place two roundings were caught.
    ("eps_sub row", "inputs", "e2_transition_census.total", 6501,
     r"ε_sub \| E2 corrective transitions \| episodes \(one each\) \| 553 \| \*\*0\.15\*\* \| 549 \| 4"),
    ("contiguous subgroup row", "inputs", "e2_transition_census.by_kind.into_recovery", 244,
     r"contiguous \(`transient`\) \| 244 \| 0\.2250 \| \*\*0\.23\*\* \| 234 \| 10"),
    ("across a gap row", "inputs", "e2_transition_census.by_kind.into_recovery_across_a_gap", 309,
     r"across a gap \| 309 \| 0\.1999 \| \*\*0\.20\*\* \| 309 \| \*\*0\*\*"),
    ("the paired range width is declared", "inputs", "decisions.total", 7629,
     r"needs `2 ln\(2M/α\)/ε²`"),
)


def at(payload: object, path: str) -> object:
    """Walk a dotted path, but try the whole path as a key first.

    A file name is a legitimate key and contains a dot, so splitting first turned
    `capture-set.jsonl` into `["capture-set", "jsonl"]` and reported the digest absent. The check
    then failed for a reason that had nothing to do with the digest, which is the worst kind of
    failure a checker can have: it was right to fail and wrong about why.
    """
    if isinstance(payload, dict) and path in payload:
        return payload[path]
    for part in path.split("."):
        payload = payload[int(part)] if isinstance(payload, list) else payload[part]
    return payload


def artefacts() -> dict[str, object]:
    """The two computed JSONs, plus the digests of the files in this directory.

    The capture set has two digests and they answer different questions: the digest *of the logical
    triples*, which is invariant to formatting and is what a re-run on another machine must
    reproduce, and the digest *of the file*, which is what says the bytes on the card are the bytes
    written here. Quoting one under a label that means the other is how a check like this gives
    false assurance, so both are pinned and each is checked against its own source.
    """
    import hashlib

    return {
        "inputs": json.loads((HERE / "prereg-inputs.json").read_text()),
        "budget": json.loads((HERE / "capture-budget.json").read_text()),
        "folds": json.loads((HERE / "folds.json").read_text()),
        "capture": json.loads((HERE / "capture-4b-measured.json").read_text()),
        "both": json.loads((HERE / "capture-12b-measured.json").read_text()),
        "chain": json.loads((HERE / "pointer-chain-terminal.json").read_text()),
        "file": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(HERE.iterdir()) if path.is_file()},
    }


#: The contradictory edits Codex demonstrated, kept as permanent regressions. Each is appended to a
#: clean document and **must** be rejected; the clean document must pass. They live here rather than
#: in a review because the detector that missed them was written to catch exactly this class.
CONTRADICTION_COUNTEREXAMPLES = {
    "a duplicated table row with a changed epsilon":
        "\n| ε_sub | E2 corrective transitions | episodes (one each) | 553 | **0.11** | 549 | 4 |\n",
    "a declaration wrapped across two lines":
        "\nThe operative ε_main =\n0.01.\n",
    "a current declaration inside a paragraph that mentions supersession":
        "\nThe earlier values are superseded. The operative ε_main = 0.01.\n",
    "a subgroup word beside a different tolerance's name":
        "\nFor contiguous cases, the operative ε_ord = 0.23.\n",
    "an ordinary standalone declaration":
        "\nThe bound is tight: ε_main = 0.01 throughout.\n",
    "a duplicated table row without emphasis (Codex R1)":
        "\n| ε_sub | E2 corrective transitions | episodes (one each) | 553 | 0.11 | 549 | 4 |\n",
}

#: Edits that must **not** fail: they discriminate a live rule from a dead one. A decimal placed in a
#: non-declared column of an operative row is ignored by the header-column rule and would be a false
#: "competing tolerance" under the fail-closed fallback — the D-CRO found the rule inert and the suite
#: green, because every other case gets the same verdict whichever path runs.
MUST_PASS_EDITS = {
    "a decimal in the spare column of an operative row (the column rule, not the fallback)": (
        "| ε_ord | E2 ordinary transitions, train split | episodes | 840 | **0.13** | 731 | 109 |",
        "| ε_ord | E2 ordinary transitions, train split | episodes | 840 | **0.13** | 731 | 109 | 0.109 |",
    ),
}

#: Edits that replace text rather than append it; each must be rejected with a "veto form" failure.
VETO_COUNTEREXAMPLES = {
    "the old unconditional veto restored in §12 (Codex R1)": (
        "**The unknown-horizon control of §4.1 is a\n  diagnostic, not a falsifier.**",
        "The instrument is falsified by any decoding of `pointer_chain` steps-remaining above the\n"
        "  permutation null.",
    ),
    "the old veto in its original wording (Codex S2 paraphrase)": (
        "**The unknown-horizon control of §4.1 is a\n  diagnostic, not a falsifier.**",
        "Any above-permutation decoding of pointer-chain remaining steps identifies a leak and\n"
        "  invalidates the instrument.",
    ),
    "the old veto added elsewhere while the new text stays (a contradiction, not a replacement)": (
        "read *not measured* because they are.",
        "read *not measured* because they are.\n\nThe instrument is falsified by any decoding of\n"
        "`pointer_chain` steps-remaining above the permutation null.",
    ),
    "'above the null' without the word permutation (D-CRO)": (
        "read *not measured* because they are.",
        "read *not measured* because they are.\n\nDecoding above the null on `pointer_chain` invalidates the instrument.",
    ),
    "'better than chance' (D-CRO)": (
        "read *not measured* because they are.",
        "read *not measured* because they are.\n\nA steps-remaining decoder better than chance identifies a leak.",
    ),
    "'if the null is exceeded' (D-CRO)": (
        "read *not measured* because they are.",
        "read *not measured* because they are.\n\nIf the null is exceeded by the steps-remaining decoder the instrument is falsified.",
    ),
    "the pinned wording beside a stray 'never' (D-CRO's escape hatch)": (
        "read *not measured* because they are.",
        "read *not measured* because they are.\n\nThe instrument is falsified by any decoding above the permutation null, and this was never in doubt.",
    ),
    "the passive wording, 'the instrument is invalid' (D-CRO)": (
        "read *not measured* because they are.",
        "read *not measured* because they are.\n\nIf the null is exceeded by the steps-remaining decoder, the instrument is invalid.",
    ),
    "the §4.1 deferral removed": ("**Status: DEFERRED, its veto disabled", "**Status: ARMED"),
    "the §10 state changed": ("**deferred, not measured**, its **veto disabled**", "**measured**"),
}


def check(document: str, *, verbose: bool = True) -> list[str]:
    """Every validation this checker performs, so both entry points run the same path.

    `competing_tolerances` used to run only in `main`, so `--self-test` never exercised it: the 56
    pinned-claim corruptions all rejected and gave no coverage at all to the newly added detector,
    and a copy carrying the very defect the normal command rejects passed the self-test (Codex C1).
    A verifier with two entry points that validate different things has one entry point nobody
    tests.
    """
    sources = artefacts()
    lines = document.splitlines()
    # Paragraph text with newlines collapsed, so a claim the markdown wrapped is still one claim.
    paragraphs = [" ".join(block.split()) for block in document.split("\n\n")]
    failures = [f"competing tolerance ({p})" for p in competing_tolerances(document)]
    failures += [f"veto form ({p})" for p in veto_form(document)]
    for claim, source, path, expected, pattern in CLAIMS:
        try:
            got = at(sources[source], path)
        except (KeyError, IndexError, TypeError):
            got = "<absent>"
        in_json = got == expected
        matched = (any(re.search(pattern, line) for line in lines)
                   or any(re.search(pattern, block) for block in paragraphs))
        if not (in_json and matched):
            failures.append(claim)
        if verbose:
            verdict = ("ok" if in_json and matched
                       else f"JSON has {got!r}" if not in_json else "no line matches")
            print(f"{claim:32s} {str(expected)[:40]:42s} {verdict}")

    # Every one of the twelve must be named in the document in its own table row.
    for family, task in at(sources["inputs"], "r6_exploratory_episodes.by_family").items():
        pattern = rf"\| {re.escape(family)} \| `{re.escape(task)}` \|"
        if not any(re.search(pattern, line) for line in lines):
            failures.append(f"episode {task}")
            if verbose:
                print(f"{'episode ' + task:32s} {'':42s} not in its own row")
    return failures


#: The operative value of each tolerance, keyed by **population** — the complete key, subgroup
#: included. Keying by bare name was the first version's mistake: a subgroup word anywhere on a line
#: added its number to the allowed set of *every* tolerance, so `the operative ε_ord = 0.23` passed
#: because "contiguous" appeared in the sentence (Codex C1).
OPERATIVE_TOLERANCES = {
    "ε_main": "0.23",
    "ε_ord": "0.13",
    "ε_sub": "0.15",
    "ε_sub/contiguous": "0.23",
    "ε_sub/gap": "0.20",
}

#: Words that make a **declaration** historical. Applied to the declaration's own sentence or table
#: row, never to its paragraph: exempting a whole paragraph let an explicitly current declaration
#: ride inside a paragraph that mentioned supersession, which is precisely the edit a careless
#: author makes.
HISTORICAL_MARKERS = (
    "superseded", "SUPERSEDED", "not operative", "coverage is not supported",
    "kept for the comparison", "kept for the reasoning", "stood here", "stood there",
    "the first draft", "earlier table", "at range width 1", "needs 428", "needs 1,021",
)

#: Which subgroup a declaration belongs to, by the words that name it.
SUBGROUP_WORDS = {"contiguous": "contiguous", "transient": "contiguous", "across a gap": "gap",
                  "across-a-gap": "gap"}


def _population(text: str, name: str) -> str:
    """The complete key a declaration is about: the tolerance name, plus its subgroup if named."""
    if name == "ε_sub":
        for word, key in SUBGROUP_WORDS.items():
            if word in text.lower():
                return f"ε_sub/{key}"
    return name


def _declarations(document: str) -> list[tuple[int, str, str, str]]:
    """Every tolerance declaration in the document, as (line, population, value, the text).

    Two forms are parsed rather than pattern-matched at a distance. A **table row** is split on its
    pipes and read positionally, so the format's own 61-character row can never slip past a
    60-character window. A **prose declaration** is read from a sentence with the newlines collapsed,
    so a declaration wrapped across two lines is one declaration and not two halves that each look
    harmless.
    """
    found: list[tuple[int, str, str, str]] = []
    all_lines = document.splitlines()
    declared_column: int | None = None  # of the table the scan is inside, by its header
    for number, line in enumerate(all_lines, 1):
        stripped = line.strip()
        if not stripped.startswith("|"):
            declared_column = None
            continue
        # Emphasis stripped: asterisks and backticks only — the tolerance names carry underscores.
        fields = [re.sub(r"[*`]", "", f).strip() for f in stripped.strip("|").split("|")]
        following = all_lines[number] if number < len(all_lines) else ""
        if re.fullmatch(r"\|?(\s*:?-+:?\s*\|)+\s*", following.strip().rstrip("|") + "|") and "---" in following:
            # (The D-CRO found the first version appended a pipe to a separator that already ended in
            # one, so the rule never matched and every row took the fail-closed branch.)
            # A header row: the declared-ε column is the one whose header says so. Emphasis is
            # formatting, not semantics (Codex R1): a value is a declaration by its column, never
            # by whether it is bold.
            hits = [i for i, f in enumerate(fields) if "declared" in f.lower()]
            declared_column = hits[0] if len(hits) == 1 else None
            continue
        if "---" in stripped and set(stripped) <= set("|-: "):
            continue
        names = [n for n in OPERATIVE_TOLERANCES if n in fields[0]]
        if not names:
            continue
        name = max(names, key=len).split("/")[0]
        if declared_column is not None and declared_column < len(fields):
            values = re.findall(r"\b0\.\d+\b", fields[declared_column])
        else:
            # No declared column to read: every ε-shaped number in the row counts, so an ambiguous
            # tolerance row fails closed rather than passing by omission.
            values = [v for f in fields[1:] for v in re.findall(r"\b0\.\d+\b", f)]
        for value in values:
            found.append((number, _population(fields[0], name), value, line))
    # Prose, on sentences with the newlines collapsed.
    joined, index = [], 1
    for raw in document.split("\n\n"):
        text = " ".join(raw.split())
        joined.append((index, text))
        index += raw.count("\n") + 2
    for start_line, block in joined:
        for sentence in re.split(r"(?<=[.;])\s+", block):
            for name in ("ε_main", "ε_ord", "ε_sub"):
                if name not in sentence:
                    continue
                for match in re.finditer(
                    rf"{re.escape(name)}[^0-9\n]{{0,80}}?(\*\*)?\b(0\.\d+)\b", sentence
                ):
                    found.append((start_line, _population(sentence, name), match.group(2),
                                  sentence))
    return found


#: The single statement of the unknown-horizon control's consequence, pinned in both places it is
#: referenced, and the form that must not come back. Codex R1/R2: restoring the old unconditional
#: veto passed both entry points, and §12 said "falsified" where §4.1 said "investigate".
VETO_PINS = (
    ("§4.1 status", "**Status: DEFERRED, its veto disabled"),
    ("§10 state", "**deferred, not measured**, its **veto disabled**"),
    ("§12 consequence", "is a\n  diagnostic, not a falsifier.**"),
)
#: A sentence that says decoding above the permutation null falsifies or invalidates the
#: instrument, unless it is the negation ("does not", "not by itself").
#: A backstop against the wordings that have been seen or proposed, not a parser of intent: the three
#: exact-string pins above are the load-bearing check and fail closed. Negation is honoured only in
#: the clause that carries the verdict word (within sixty characters before it), never anywhere in the
#: sentence — a sentence-wide exemption was an escape hatch (the D-CRO, on 6ee2563).
_NULL = (r"(above[- ](the )?(unconditional )?(permutation )?null|(exceed\w*|beat\w*) the (unconditional )?"
         r"(permutation )?null|(the )?null (is|was|were) exceeded|better than chance|above chance)")
_VERDICT = r"(falsif\w*|invalid\w*|identif\w* a leak|is a leak|a leak)"
FORBIDDEN_VETO = re.compile(rf"{_VERDICT}[^.]{{0,160}}?{_NULL}|{_NULL}[^.]{{0,160}}?{_VERDICT}", re.IGNORECASE)
_NEGATED = re.compile(r"\b(does|do|did) not\b|\bnot by itself\b|\bnever\b|\bnot\s+(a|an|the)?\s*(leak|falsif|invalid)|\bno\b", re.IGNORECASE)


def veto_form(document: str) -> list[str]:
    """One veto, deferred and disabled, stated as a diagnostic in every place it is named."""
    problems = []
    flat = " ".join(document.split())
    for name, text in VETO_PINS:
        if " ".join(text.split()) not in flat:  # a re-wrap is not a restored veto
            problems.append(f"veto pin missing: {name}")
    for block in document.split("\n\n"):
        text = " ".join(block.split())
        for sentence in re.split(r"(?<=[.;])\s+", re.sub(r"[*`]", "", text)):
            match = FORBIDDEN_VETO.search(sentence)
            if match:
                verdict = re.search(_VERDICT, sentence[match.start():match.end()], re.IGNORECASE)
                at = match.start() + (verdict.start() if verdict else 0)
                if not _NEGATED.search(sentence[max(0, at - 60): at + 12]):
                    problems.append(f"veto restored in unconditional form: {sentence[:90]!r}")
    return sorted(set(problems))


def competing_tolerances(document: str) -> list[str]:
    """Refuse a second operative value for one population.

    Recomputing §7 at the paired range width left a whole second table behind, unmarked, declaring
    values that under the new rule are unattainable. Every individual claim passed, because each
    pinned figure was present and correct; nothing asked whether the document contradicted itself.
    A checker that verifies every claim and cannot see a contradiction between two of them is
    checking the sentences and not the document.
    """
    problems = []
    for number, population, value, text in _declarations(document):
        if any(marker in text for marker in HISTORICAL_MARKERS):
            continue
        operative = OPERATIVE_TOLERANCES.get(population)
        if operative is not None and value != operative:
            problems.append(
                f"line {number}: {population} declared {value}, operative is {operative}"
            )
    return sorted(set(problems))


def main() -> int:
    document = (HERE / "PREREGISTRATION.md").read_text()
    failures = check(document)
    print()
    if failures:
        print(f"{len(failures)} claim(s) do not check out: {', '.join(failures)}")
        return 1
    print(f"all {len(CLAIMS) + 12} claims check out against the computed artefacts.")
    print("  (coverage is what these claims pin, not the whole document; a section with no claim "
          "here is unchecked, not verified)")
    return 0


def corrupt(line: str, span: tuple[int, int]) -> str:
    """Break the claim's *own* figure, inside the text its pattern matched.

    Corrupting the first digit of the whole line is not enough: a table row carries several figures,
    and changing a neighbouring one leaves the pinned pattern matching. That was this self-test's
    first version, and it reported nine false survivors — the same mistake one level up.
    """
    start, end = span
    matched = line[start:end]
    if re.search(r"\d", matched):
        matched = re.sub(r"\d", lambda m: "8" if m.group() != "8" else "7", matched, count=1)
    else:
        matched = ""  # a claim pinned by wording, not by a figure: delete the wording
    return line[:start] + matched + line[end:]


def _corrupt_wrapped(document: str, pattern: str, expected: object) -> str | None:
    """Break a claim whose pattern only matches across a line wrap, by changing its own figure.

    Line-level corruption cannot reach these: the pattern matches the paragraph with its newlines
    collapsed and no single line satisfies it. Rather than loosen the pattern — which is what binds
    the numerator to its denominator — the corruption targets the value the claim asserts, inside the
    paragraph that asserts it.
    """
    rendered = {str(expected)}
    if isinstance(expected, int):
        rendered.add(f"{expected:,}")
    offset = 0
    for block in document.split("\n\n"):
        if re.search(pattern, " ".join(block.split())):
            for form in sorted(rendered, key=len, reverse=True):
                if form in block:
                    changed = block.replace(form, form[:-1] + ("8" if form[-1] != "8" else "7"), 1)
                    return document[:offset] + changed + document[offset + len(block):]
            return None
        offset += len(block) + 2
    return None


def self_test() -> int:
    """Corrupt each pinned claim in turn and assert the check rejects the corrupted document.

    A guard that has never been seen to fail is not known to be a guard.
    """
    document = (HERE / "PREREGISTRATION.md").read_text()
    lines = document.splitlines(keepends=True)
    if check(document, verbose=False):
        print("self-test cannot run: the document does not pass the check as it stands")
        return 1

    survived, exercised = [], 0
    # The contradiction detector, exercised by the same command that exercises the claims.
    for name, edit in CONTRADICTION_COUNTEREXAMPLES.items():
        exercised += 1
        if not any(f.startswith("competing tolerance") for f in check(document + edit, verbose=False)):
            survived.append(f"contradiction: {name}")
    for name, (old, new) in MUST_PASS_EDITS.items():
        exercised += 1
        if old not in document:
            survived.append(f"must-pass: {name} (anchor text absent)")
        elif check(document.replace(old, new), verbose=False):
            survived.append(f"must-pass edit wrongly rejected: {name}")
    for name, (old, new) in VETO_COUNTEREXAMPLES.items():
        exercised += 1
        if old not in document:
            survived.append(f"veto: {name} (anchor text absent)")
            continue
        if not any(f.startswith("veto form") for f in check(document.replace(old, new), verbose=False)):
            survived.append(f"veto: {name}")
    for claim, _source, _path, expected, pattern in CLAIMS:
        hits = [(i, re.search(pattern, line)) for i, line in enumerate(lines)]
        hits = [(i, match) for i, match in hits if match]
        if not hits:
            # A claim the markdown wrapped matches only the joined paragraph, so there is no single
            # line to corrupt. Corrupt the expected value where it sits in that paragraph instead:
            # the point is to break the pinned figure, and the figure is what the claim asserts.
            broken = _corrupt_wrapped(document, pattern, expected)
            if broken is None:
                survived.append(f"{claim} (pattern matches nothing to corrupt)")
                continue
            exercised += 1
            if claim not in check(broken, verbose=False):
                survived.append(claim)
            continue
        # Every matching line must be corrupted at once, or a claim that pins a figure appearing
        # twice would be caught by the copy rather than by the line under test.
        broken = lines[:]
        for index, match in hits:
            broken[index] = corrupt(broken[index], match.span())
        exercised += 1
        if claim not in check("".join(broken), verbose=False):
            survived.append(claim)
    print(f"self-test: {exercised} pinned claim(s) corrupted; "
          f"{len(survived)} survived corruption")
    if survived:
        print("  not caught: " + ", ".join(survived))
        return 1
    print("  every pinned claim's corruption was caught by the claim that pins it.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true",
                        help="corrupt each pinned line in turn and assert the check rejects it")
    sys.exit(self_test() if parser.parse_args().self_test else main())
