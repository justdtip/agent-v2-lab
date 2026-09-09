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
    ("capture set digest", "inputs", "decisions.capture_set_sha256",
     "1a7cfbdd4e21fc203eafbcc3ec50b96afcb214c169f21c7ba968ca83dfc9709a",
     r"`1a7cfbdd4e21fc203eafbcc3ec50b96afcb214c169f21c7ba968ca83dfc9709a`"),
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
    ("prompt chars, median", "budget", "prompt_length_chars.median", 4738,
     r"minimum 1,824, median 4,738, maximum 14,316"),
    ("prompt chars, max", "budget", "prompt_length_chars.max", 14316,
     r"median 4,738, maximum 14,316"),
    ("prompt chars, min", "budget", "prompt_length_chars.min", 1824, r"minimum 1,824"),
    ("exploratory positions", "budget", "exploratory_stratum.positions_total", 15776,
     r"About 15,776 positions"),
)


def at(payload: object, path: str) -> object:
    for part in path.split("."):
        payload = payload[int(part)] if isinstance(payload, list) else payload[part]
    return payload


def artefacts() -> dict[str, object]:
    return {
        "inputs": json.loads((HERE / "prereg-inputs.json").read_text()),
        "budget": json.loads((HERE / "capture-budget.json").read_text()),
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
