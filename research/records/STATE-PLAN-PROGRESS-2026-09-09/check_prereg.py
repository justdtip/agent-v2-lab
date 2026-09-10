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
        "file": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(HERE.iterdir()) if path.is_file()},
    }


def check(document: str, *, verbose: bool = True) -> list[str]:
    """Return the names of the claims that do not check out. Empty means the document is consistent."""
    sources = artefacts()
    lines = document.splitlines()
    failures = []
    for claim, source, path, expected, pattern in CLAIMS:
        try:
            got = at(sources[source], path)
        except (KeyError, IndexError, TypeError):
            got = "<absent>"
        in_json = got == expected
        matched = any(re.search(pattern, line) for line in lines)
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
    for claim, _source, _path, _expected, pattern in CLAIMS:
        hits = [(i, re.search(pattern, line)) for i, line in enumerate(lines)]
        hits = [(i, match) for i, match in hits if match]
        if not hits:
            survived.append(f"{claim} (pattern matches no line)")
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
