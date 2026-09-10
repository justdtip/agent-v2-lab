"""Check that Amendment 1 states each measured figure in one place.

The failure this guards against is Codex's F3 on revision 4: §7 still mandated the withdrawn form's
headline capability figures after §5's table had been re-measured, because the numbers had been
copied into prose. The rule the amendment now states is that a §5 table figure appears in §5's
table, may be quoted in §10 (history), and appears nowhere else; and that §7 carries no decimal
figure at all — it refers to §5's current table.

    python .../check_amendment.py             # check the document
    python .../check_amendment.py --self-test # corrupt §7 with a §5 figure and assert rejection

Exits nonzero on any disagreement.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOC = HERE / "AMENDMENT-1-TRANSPORT-RULE.md"
FIGURE = re.compile(r"(?<![\d.])\d\.\d{3}(?![\d])")


def sections(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    current = "preamble"
    for line in text.splitlines():
        m = re.match(r"^## (\d+)\.", line)
        if m:
            current = m.group(1)
        out[current] = out.get(current, "") + line + "\n"
    return out


def table_figures(section: str) -> set[str]:
    return {f for line in section.splitlines() if line.startswith("|") for f in FIGURE.findall(line)}


def check(text: str) -> list[str]:
    secs = sections(text)
    problems: list[str] = []
    owned = table_figures(secs.get("5", ""))
    if not owned:
        problems.append("§5 carries no table figures; the gate table is missing")
    for number, body in secs.items():
        if number in ("5", "10"):
            continue
        for lineno, line in enumerate(body.splitlines(), 1):
            for f in FIGURE.findall(line):
                if f in owned:
                    problems.append(f"§{number} line {lineno} repeats §5 table figure {f}: {line.strip()[:80]}")
    for lineno, line in enumerate(secs.get("7", "").splitlines(), 1):
        for f in FIGURE.findall(line):
            problems.append(f"§7 line {lineno} carries a decimal figure {f}; §7 refers to §5, it does not copy it")
    return problems


def main(argv: list[str]) -> int:
    text = DOC.read_text()
    if "--self-test" in argv:
        secs = sections(text)
        figure = sorted(table_figures(secs["5"]))[0]
        corrupted = text.replace("## 7. What is scored, and once\n", f"## 7. What is scored, and once\n\nbeside {figure} at the headline rank.\n", 1)
        assert corrupted != text
        if not check(corrupted):
            print(f"self-test FAILED: a §5 figure ({figure}) planted in §7 was not rejected")
            return 1
        print(f"self-test passed: planting {figure} in §7 is rejected")
    problems = check(text)
    for p in problems:
        print("FAIL", p)
    print("amendment figures: " + ("clean" if not problems else f"{len(problems)} problem(s)"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
