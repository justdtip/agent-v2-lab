"""Assemble the seal for the plan-progress pre-registration (§7, §7.1, §11 precondition 5).

Every value in the payload is **recomputed from an artefact** and then checked against what the
document declares. Nothing is copied from a constant in this file into the payload: a writer that
supplies its own answers checks only itself (`METHOD-2026-09-08`, entry 34, seventh rule). Each
disagreement refuses by name and no seal is written.

Writing is not the default. Without `--write` the payload is printed and its digest reported, which
is what preparing the seal means; `--write` creates `seal.json` and refuses to overwrite one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))

import check_prereg  # noqa: E402
import folds as folds_module  # noqa: E402
from local_llm_lab.pipeline.state_programme.tolerances import required_n  # noqa: E402

#: Files whose bytes the seal fixes. A reader that finds a different digest is reading a different
#: study, and says so rather than proceeding.
SEALED_FILES = (
    "PREREGISTRATION.md",
    "check_prereg.py",
    "prereg_inputs.py",
    "prereg-inputs.json",
    "capture-set.jsonl",
    "folds.py",
    "folds.json",
    "carrier_ablation.py",
    "carrier-ablation.json",
    "capture_budget.py",
    "capture-budget.json",
    "count_corpus.py",
    "corpus-count.json",
)

TABLE_HEADER = "| | evaluation set | unit | n | ε declared | needs n ≥ | spare |"


class Refused(RuntimeError):
    """The seal was not written, and the reason names itself."""


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def declared_in(document: str, pattern: str, name: str) -> str:
    """One value read out of the document's own prose, so the seal is checked against the text."""
    hits = re.findall(pattern, document)
    if not hits:
        raise Refused(f"the document declares no {name}; the seal has nothing to check against")
    if len(set(hits)) != 1:
        raise Refused(f"the document declares {len(set(hits))} different values for {name}: {sorted(set(hits))}")
    return hits[0]


def operative_rows(document: str) -> list[dict]:
    """The operative tolerance table, read as rows rather than as a paragraph."""
    lines = document.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == TABLE_HEADER)
    except StopIteration as exc:
        raise Refused("the operative tolerance table's header is not where §7 puts it") from exc
    rows = []
    for line in lines[start + 2:]:
        if not line.strip().startswith("|"):
            break
        fields = [re.sub(r"[*`]", "", f).strip() for f in line.strip().strip("|").split("|")]
        rows.append({
            "name": fields[0], "evaluation_set": fields[1], "unit": fields[2],
            "n": int(fields[3].replace(",", "")),
            "epsilon_declared": float(fields[4]),
            "needs_n_at_least": int(fields[5].replace(",", "")),
            "spare": int(fields[6].replace(",", "")),
        })
    if not rows:
        raise Refused("the operative tolerance table has no rows")
    return rows


def recompute_folds(directory: Path) -> dict:
    """Run the assignment again rather than read the digest someone wrote down."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "folds.json"
        code = folds_module.main([
            "--decisions", str(directory / "capture-set.jsonl"),
            "--out", str(out),
        ])
        if code != 0:
            raise Refused(f"folds.py exited {code}; the assignment could not be recomputed")
        return json.loads(out.read_text())


def blob_at(directory: Path, commit: str, name: str) -> str | None:
    """The digest of one sealed file as that commit has it, so a later commit cannot move it."""
    rel = subprocess.run(["git", "-C", str(directory), "ls-files", "--full-name", name],
                         capture_output=True, text=True, check=True).stdout.strip()
    if not rel:
        return None
    shown = subprocess.run(["git", "-C", str(directory), "show", f"{commit}:{rel}"],
                           capture_output=True, check=False)
    if shown.returncode != 0:
        return None
    return hashlib.sha256(shown.stdout).hexdigest()


def build(directory: Path, baseline: str | None = None) -> tuple[dict, list[str]]:
    document = (directory / "PREREGISTRATION.md").read_text(encoding="utf-8")
    refusals: list[str] = []

    # 1. The document must pass its own checker. A seal over a document that fails its claims would
    #    fix the failure in place.
    failures = check_prereg.check(document, verbose=False)
    if failures:
        refusals.append(f"the document does not pass check_prereg.py: {len(failures)} failure(s), first {failures[0]!r}")

    # 2. The folds, recomputed. Three values must agree: the fresh run, the stored artefact, the text.
    fresh = recompute_folds(directory)
    stored = json.loads((directory / "folds.json").read_text())
    declared_folds = declared_in(document, r"\b([0-9a-f]{64})\b(?=`,\s*which goes into the seal)", "folds assignment digest")
    if fresh["assignment_sha256"] != stored["assignment_sha256"]:
        refusals.append("the recomputed fold assignment differs from folds.json")
    if fresh["assignment_sha256"] != declared_folds:
        refusals.append("the recomputed fold assignment differs from the digest §7.1 declares")

    # 3. The capture set, recomputed against both the artefact that enumerated it and the text.
    capture_path = directory / "capture-set.jsonl"
    capture_sha = sha256_of(capture_path)
    capture_lines = sum(1 for line in capture_path.read_text().splitlines() if line.strip())
    inputs = json.loads((directory / "prereg-inputs.json").read_text())
    # §2 makes two different claims and the seal carries both: the digest over the enumerated
    # triples, and the digest of the bytes on disk. One cannot stand in for the other.
    triples_sha = inputs["decisions"]["capture_set_sha256"]
    declared_file = declared_in(document, r"\| the file \| `([0-9a-f]{64})`", "capture-set file digest")
    declared_set = declared_in(document, r"\| the set \| `([0-9a-f]{64})`", "capture-set triples digest")
    if capture_sha != declared_file:
        refusals.append("capture-set.jsonl's bytes differ from the file digest §2 declares")
    if triples_sha != declared_set:
        refusals.append("prereg-inputs.json's triples digest differs from the one §2 declares")
    declared_lines = int(declared_in(document, r"`capture-set\.jsonl`, ([\d,]+) lines", "capture set line count").replace(",", ""))
    if declared_lines != capture_lines:
        refusals.append(f"the document says {declared_lines} capture-set lines; the file has {capture_lines}")

    # 4. The tolerances, recomputed from the bound rather than read. M and alpha come from the text.
    m = int(declared_in(document, r"\*\*M = (\d+) pre-registered quantities\*\*", "M"))
    alpha = float(declared_in(document, r"with α = ([\d.]+)", "alpha"))
    rows = operative_rows(document)
    for row in rows:
        recomputed = required_n(m, row["epsilon_declared"], alpha=alpha, range_width=2.0)
        if recomputed != row["needs_n_at_least"]:
            refusals.append(
                f"{row['name']}: the table says it needs n ≥ {row['needs_n_at_least']}, "
                f"the bound at M={m}, α={alpha}, width 2 gives {recomputed}"
            )
        if row["n"] - row["needs_n_at_least"] != row["spare"]:
            refusals.append(f"{row['name']}: n − needs does not equal the spare column")
        if row["n"] < recomputed:
            refusals.append(f"{row['name']}: declared ε {row['epsilon_declared']} is unmet at n = {row['n']}")

    commit = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(directory), "status", "--porcelain", "--", "."],
        capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        refusals.append(f"the record directory has uncommitted changes; a seal fixes committed bytes:\n{dirty}")

    if baseline:
        # Resolve to the full object name, so the seal's own digest does not depend on how many
        # characters of the commit someone typed on the command line.
        resolved = subprocess.run(["git", "-C", str(directory), "rev-parse", "--verify", f"{baseline}^{{commit}}"],
                                  capture_output=True, text=True)
        if resolved.returncode != 0:
            raise Refused(f"{baseline!r} does not name a commit in this repository")
        baseline = resolved.stdout.strip()

    files = {name: sha256_of(directory / name) for name in SEALED_FILES}
    # The seal may be created at a later commit than the one whose text was reviewed, so long as no
    # sealed file has moved since. That is checked here rather than asserted.
    if baseline:
        for name, value in files.items():
            was = blob_at(directory, baseline, name)
            if was is None:
                refusals.append(f"{name} does not exist at {baseline}; it cannot be sealed against it")
            elif was != value:
                refusals.append(f"{name} differs from its bytes at {baseline}")

    payload = {
        "schema_version": 1,
        "baseline_commit": baseline,
        "programme": "state-plan-progress",
        "record": directory.name,
        "commit": commit,
        "seed": fresh["seed"],
        "folds": {
            "k": fresh["folds"],
            "basis": fresh["basis"],
            "episodes": fresh["episodes"],
            "assignment_sha256": fresh["assignment_sha256"],
            "per_fold": fresh["per_fold"],
        },
        "capture_set": {
            "path": "capture-set.jsonl",
            "file_sha256": capture_sha,
            "triples_sha256": triples_sha,
            "lines": capture_lines,
        },
        "tolerances": {"m": m, "alpha": alpha, "range_width": 2.0, "rows": rows},
        "deferred": {
            "unknown_horizon_control": {
                "quantities": "11-12 of M",
                "state": "deferred, not measured, veto disabled",
                "released_by": "an amendment Codex reviews before any of its results are read",
            }
        },
        "files": files,
        # The instrument, recorded but not sealed: it is not part of the reviewed study, and it
        # postdates the baseline, so a reader re-running it later can tell whether it has the same
        # builder without the seal claiming the builder was reviewed.
        "produced_by": {"path": "make_seal.py", "sha256": sha256_of(Path(__file__).resolve())},
        "refusals": {
            "reader": "the reader refuses to read any capture without this file (§11 precondition 5)",
            "folds": "the assignment digest is fixed here so the folds cannot move afterwards (§7.1)",
            "tolerances": "every ε here was recomputed from the bound at the paired range width, not copied",
        },
    }
    return payload, refusals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=HERE)
    parser.add_argument("--baseline", help="commit whose sealed bytes this seal must match")
    parser.add_argument("--write", action="store_true", help="create seal.json; without it, prepare only")
    args = parser.parse_args(argv)

    try:
        payload, refusals = build(args.directory, args.baseline)
    except Refused as exc:
        print(f"refused: {exc}")
        return 2

    body = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    seal_sha = hashlib.sha256(body.encode()).hexdigest()

    print(f"commit            {payload['commit']}")
    print(f"baseline          {payload['baseline_commit'] or '(none given)'}")
    print(f"seed              {payload['seed']}")
    print(f"folds             K={payload['folds']['k']}, {payload['folds']['episodes']} episodes, "
          f"{payload['folds']['assignment_sha256']}")
    print(f"capture set       {payload['capture_set']['lines']} lines\n                  file    {payload['capture_set']['file_sha256']}\n                  triples {payload['capture_set']['triples_sha256']}")
    print(f"tolerances        M={payload['tolerances']['m']}, α={payload['tolerances']['alpha']}, "
          f"{len(payload['tolerances']['rows'])} rows, all recomputed at width 2")
    for name, value in payload["files"].items():
        print(f"  {name:<26} {value}")
    print(f"seal sha256       {seal_sha}")

    if refusals:
        print(f"\nREFUSED, {len(refusals)} reason(s); no seal written:")
        for reason in refusals:
            print(f"  - {reason}")
        return 1

    if not args.write:
        print("\nprepared, not written. Re-run with --write to create seal.json.")
        return 0

    target = args.directory / "seal.json"
    if target.exists():
        print(f"\nrefused: {target} already exists; a seal is not rewritten")
        return 2
    target.write_text(body, encoding="utf-8")
    print(f"\nsealed: {target} ({seal_sha})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
