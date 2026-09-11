"""Amendment 1's addendum seal, against its own baseline.

The parent seal fixes the pre-registration. This fixes the amendment that extends it: the document,
the rule's implementation, the readers and checkers that carry it out, and the capability table with
the artefact that produced it.

Same discipline as `make_seal.py`, for the same reason. Every value is **recomputed and then checked
against what the amendment declares** rather than copied from a constant here (`METHOD-2026-09-08`,
entry 34): the capability table is read back out of `gate-table.json` and must reproduce §5's printed
figures to the digit they are printed at; the parent seal must still verify; the amendment must pass
its own checker; and every sealed file must match its bytes at the baseline commit — the commit the
review actually read. Each disagreement refuses by name and nothing is written.

`--write` is not the default. Without it the payload is printed and its digest reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from local_llm_lab import spawn  # noqa: E402
from local_llm_lab.pipeline.state_programme.read_gate import require_seal  # noqa: E402

import check_amendment  # noqa: E402

#: Paths relative to the repository root. The amendment's procedure is exactly these files.
SEALED_FILES = (
    "research/records/STATE-PLAN-PROGRESS-2026-09-09/AMENDMENT-1-TRANSPORT-RULE.md",
    "research/records/STATE-PLAN-PROGRESS-2026-09-09/check_amendment.py",
    "research/records/STATE-PLAN-PROGRESS-2026-09-09/measure_gate.py",
    "research/records/STATE-PLAN-PROGRESS-2026-09-09/gate-table.json",
    "research/records/STATE-PLAN-PROGRESS-2026-09-09/read_e2.py",
    "src/local_llm_lab/pipeline/state_programme/transport.py",
    "tests/test_state_transport.py",
)

PARENT_SEAL = "seal.json"
ADDENDUM = "addendum-1.json"


class Refused(RuntimeError):
    """An addendum that cannot be sealed honestly is not sealed."""


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def printed_table(document: str) -> dict[str, list[float]]:
    """§5's capability table, read out of the document as rows rather than as prose."""
    rows: dict[str, list[float]] = {}
    for line in document.splitlines():
        hit = re.match(r"\|\s*(4B|12B), layer (\d+)\s*\|(.+)\|\s*$", line.strip())
        if not hit:
            continue
        values = [float(re.sub(r"[*\s]", "", f)) for f in hit.group(3).split("|")]
        rows[f"layer {hit.group(2)}"] = values
    if len(rows) != 2:
        raise Refused(f"§5's capability table has {len(rows)} model rows, not 2")
    return rows


def blob_at(commit: str, name: str) -> str | None:
    shown = spawn.run(["git", "-C", str(ROOT), "show", f"{commit}:{name}"], capture_output=True)
    return None if shown.returncode else hashlib.sha256(shown.stdout).hexdigest()


def build(directory: Path, baseline: str) -> tuple[dict, list[str]]:
    refusals: list[str] = []
    parent = require_seal(directory)
    document = (directory / "AMENDMENT-1-TRANSPORT-RULE.md").read_text(encoding="utf-8")

    if check_amendment.check(document):
        refusals.append("the amendment does not pass check_amendment.py")

    # The capability table, read back out of the artefact that produced it.
    table = json.loads((directory / "gate-table.json").read_text())
    if table["seal_sha256"] != parent["verified"]["seal_sha256"]:
        refusals.append("gate-table.json was measured against a different seal")
    if table["assignment_sha256"] != parent["folds"]["assignment_sha256"]:
        refusals.append("gate-table.json was measured against a different fold assignment")
    printed = printed_table(document)
    for model, row in table["models"].items():
        key = f"layer {row['layer']}"
        if key not in printed:
            refusals.append(f"{model}: §5 prints no row for {key}")
            continue
        measured = [row["by_rank"][str(rank)] for rank in (1, 2, 4, 8, 16, 32)]
        for rank, got, shown in zip((1, 2, 4, 8, 16, 32), measured, printed[key]):
            if got is None or round(got, 3) != shown:
                refusals.append(f"{model} r={rank}: §5 prints {shown}, gate-table.json holds {got}")

    resolved = spawn.run(["git", "-C", str(ROOT), "rev-parse", "--verify", f"{baseline}^{{commit}}"],
                         capture_output=True, text=True)
    if resolved.returncode:
        raise Refused(f"{baseline!r} does not name a commit in this repository")
    baseline = resolved.stdout.strip()

    files = {}
    for name in SEALED_FILES:
        path = ROOT / name
        if not path.exists():
            refusals.append(f"{name} does not exist")
            continue
        files[name] = sha256_of(path)
        was = blob_at(baseline, name)
        if was is None:
            refusals.append(f"{name} does not exist at {baseline[:12]}…")
        elif was != files[name]:
            refusals.append(f"{name} differs from its bytes at {baseline[:12]}…")

    dirty = spawn.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                      capture_output=True, text=True).stdout.strip()
    if dirty:
        refusals.append(f"the tree has uncommitted changes; an addendum fixes committed bytes:\n{dirty}")

    payload = {
        "schema_version": 1,
        "amends": {"seal": PARENT_SEAL, "sha256": parent["verified"]["seal_sha256"],
                   "baseline_commit": parent.get("baseline_commit")},
        "amendment": "Amendment 1 — E2's transport rule",
        "baseline_commit": baseline,
        "commit": spawn.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip(),
        "capability_table": {model: {"layer": row["layer"],
                                     "by_rank": {r: row["by_rank"][r] for r in ("1", "2", "4", "8", "16", "32")}}
                             for model, row in table["models"].items()},
        "files": files,
        "refusals": {
            "reader": "no corrective stratum is scored until the gate of §5 passes",
            "table": "§5's printed table must reproduce gate-table.json to the digit it prints",
            "parent": "the addendum is void if the parent seal does not verify",
        },
    }
    return payload, refusals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=HERE)
    parser.add_argument("--baseline", required=True, help="the commit the review read")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    try:
        payload, refusals = build(args.record, args.baseline)
    except Refused as exc:
        print(f"refused: {exc}")
        return 2

    print(f"amends seal       {payload['amends']['sha256']}")
    print(f"baseline          {payload['baseline_commit']}")
    print(f"commit            {payload['commit']}")
    for name, value in payload["files"].items():
        print(f"  {Path(name).name:<32} {value[:16]}…")
    body = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    digest = hashlib.sha256(body.encode()).hexdigest()
    print(f"addendum sha256   {digest}")

    if refusals:
        print(f"\nREFUSED, {len(refusals)} reason(s); nothing written:")
        for reason in refusals:
            print(f"  - {reason}")
        return 1
    if not args.write:
        print("\nprepared, not written. Re-run with --write to create the addendum.")
        return 0
    target = args.record / ADDENDUM
    if target.exists():
        print(f"\nrefused: {target} already exists; an addendum is not rewritten")
        return 2
    target.write_text(body, encoding="utf-8")
    print(f"\nsealed: {target} ({digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
