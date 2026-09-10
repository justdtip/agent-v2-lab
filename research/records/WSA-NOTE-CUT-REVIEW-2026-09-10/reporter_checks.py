"""Model-free reporter checks using frozen source and wholly synthetic records.

The caller verifies the record's source hashes before calling ``run_checks``.
No experiment output is read, no native package is imported, and reporter writes
are redirected to an in-memory filesystem. All counterfactual rows are retained
in the returned object so that every observed acceptance can be reviewed.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
from typing import Any


def _compile_source(source: str) -> Any:
    """Verify every import, then remove imports from the actual module AST."""
    tree = ast.parse(source)
    allowed_imports = {"json", "sys", "statistics", "argparse"}
    allowed_from = {"pathlib": {"Path"}, "collections": {"defaultdict"}}
    top_imports = {
        id(node) for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if id(node) not in top_imports:
                raise ValueError("nested source import is not permitted")
            for name in node.names:
                if name.name not in allowed_imports or name.asname is not None:
                    raise ValueError(f"unapproved source import: {name.name}")
        elif isinstance(node, ast.ImportFrom):
            if id(node) not in top_imports or node.level != 0:
                raise ValueError("nested or relative source import is not permitted")
            permitted = allowed_from.get(node.module, set())
            for name in node.names:
                if name.name not in permitted or name.asname is not None:
                    raise ValueError(f"unapproved source import: {node.module}.{name.name}")
    tree.body = [
        node for node in tree.body
        if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    return compile(ast.fix_missing_locations(tree), "<frozen-w3b-reporter>", "exec")


def _execute(code: Any, rows: list[dict], manifest: dict) -> dict:
    files = {
        "fixture/w3b.jsonl": "\n".join(json.dumps(row) for row in rows) + "\n",
        "fixture/manifest.json": json.dumps(manifest),
    }
    accesses: list[dict[str, str]] = []

    class VirtualPath:
        def __init__(self, value: object):
            self.value = str(value)

        def __truediv__(self, part: object) -> VirtualPath:
            return VirtualPath(self.value + "/" + str(part))

        def __str__(self) -> str:
            return self.value

        def read_text(self) -> str:
            accesses.append({"operation": "read", "path": self.value})
            return files[self.value]

        def write_text(self, value: str) -> int:
            if self.value != "report.json":
                raise ValueError("reporter attempted an unexpected virtual write")
            accesses.append({"operation": "write", "path": self.value})
            files[self.value] = value
            return len(value)

    class FixtureArgumentParser(argparse.ArgumentParser):
        def __init__(self, *args: Any, **kwargs: Any):
            kwargs.setdefault("prog", "workspace_w3b_analyze.py")
            super().__init__(*args, **kwargs)

        def parse_args(self, args: Any = None, namespace: Any = None) -> Any:
            return super().parse_args(
                ["fixture", "report.json"] if args is None else args, namespace
            )

    namespace = {
        "json": json,
        "sys": sys,
        "statistics": statistics,
        "argparse": SimpleNamespace(ArgumentParser=FixtureArgumentParser),
        "Path": VirtualPath,
        "defaultdict": defaultdict,
        "__name__": "__synthetic_reporter_check__",
    }
    stdout = io.StringIO()
    completed = False
    error = None
    with contextlib.redirect_stdout(stdout):
        try:
            exec(code, namespace)
            completed = True
        except (Exception, SystemExit) as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
    return {
        "completed_without_exception": completed,
        "error": error,
        "stdout": stdout.getvalue(),
        "report": json.loads(files["report.json"]) if "report.json" in files else None,
        "virtual_file_accesses": accesses,
    }


def run_checks(root: Path) -> dict:
    """Run the frozen reporter at ``root/source`` against deterministic fixtures."""
    source_bytes = (root / "source" / "workspace_w3b_analyze.py").read_bytes()
    code = _compile_source(source_bytes.decode("utf-8"))
    six = [0.8, 0.04, 0.04, 0.04, 0.04, 0.04]
    unmasked = {
        "model_six_act": [six, 1.0],
        "lens_act": {"1": {"six": six, "mass": 1.0}},
    }
    masked = copy.deepcopy(unmasked)
    masked.update({
        "gate": {
            "attention_on_masked_keys": 0.0,
            "attention_on_masked_edges_all_queries": 0.0,
            "local_mass_beyond_window": 0.0,
            "n_blocked_edges_at_P_act": 1,
            "n_blocked_edges_total": 2,
        },
        "p_note_identical_to_unmasked": True,
    })
    baseline_row = {
        "i": 7,
        "task_id": "synthetic-episode-1",
        "step": 0,
        "family": "synthetic-family-a",
        "tool_idx": 0,
        "note_prose_tokens": 2,
        "note_syntax_tokens": 3,
        "arms": {"unmasked": unmasked, "current_note": masked},
        "skipped_arms": {},
    }
    baseline_manifest = {
        "schema_version": 2,
        "rows_written": 1,
        "sample": [7],
        "queries_masked_from": "synthetic declared positions",
        "control": "synthetic declared control",
        "corpus_sha256": "synthetic-expected-corpus",
        "checkpoint": "synthetic-expected-checkpoint",
    }
    cases: list[dict] = []

    def add(name: str, purpose: str, rows: list[dict], manifest: dict) -> None:
        input_rows, input_manifest = copy.deepcopy((rows, manifest))
        cases.append({
            "name": name,
            "purpose": purpose,
            "inputs": {"rows": input_rows, "manifest": input_manifest},
            "execution": _execute(code, input_rows, input_manifest),
        })

    add("positive_baseline", "Valid synthetic schema-2 row and passing gate values.",
        [baseline_row], baseline_manifest)
    for name, field, value in (
        ("failed_masked_edge_gate", "attention_on_masked_edges_all_queries", 0.125),
        ("failed_window_gate", "local_mass_beyond_window", 0.25),
        ("failed_nonempty_cut_gate", "n_blocked_edges_at_P_act", 0),
    ):
        row = copy.deepcopy(baseline_row)
        row["arms"]["current_note"]["gate"][field] = value
        add(name, "One producer gate fails; observe whether the reporter blocks results.",
            [row], baseline_manifest)
    row = copy.deepcopy(baseline_row)
    row["arms"]["current_note"]["p_note_identical_to_unmasked"] = False
    add("failed_p_note_identity_gate", "P_note identity is false, with other gates passing.",
        [row], baseline_manifest)
    row = copy.deepcopy(baseline_row)
    del row["arms"]["current_note"]["gate"]
    add("missing_gate", "A scientific arm has no gate record.", [row], baseline_manifest)
    manifest = copy.deepcopy(baseline_manifest)
    manifest["schema_version"] = 1
    add("schema_1", "Schema version is legacy, although this fixture retains newer fields.",
        [baseline_row], manifest)
    manifest = copy.deepcopy(baseline_manifest)
    manifest.update(sample=[7, 8], rows_written=2)
    add("missing_expected_id", "Expected row 8 is absent; manifest still declares two rows.",
        [baseline_row], manifest)
    add("duplicate_id_replaces_expected_id", "Row 7 is duplicated instead of expected row 8.",
        [baseline_row, baseline_row], manifest)
    extra = copy.deepcopy(baseline_row)
    extra["i"] = 8
    extra["task_id"] = "synthetic-episode-2"
    add("extra_id", "An additional row is not in the manifest sample.",
        [baseline_row, extra], baseline_manifest)
    unexpected = copy.deepcopy(baseline_row)
    unexpected["i"] = 999
    add("unexpected_id_with_matching_count", "Row count matches but requested identity does not.",
        [unexpected], baseline_manifest)
    legacy_first = copy.deepcopy(baseline_row)
    del legacy_first["arms"]["current_note"]["gate"]["attention_on_masked_edges_all_queries"]
    leaking_later = copy.deepcopy(extra)
    leaking_later["arms"]["current_note"]["gate"]["attention_on_masked_edges_all_queries"] = 0.5
    manifest = copy.deepcopy(baseline_manifest)
    manifest.update(sample=[7, 8], rows_written=2)
    add("first_legacy_gate_hides_later_leak", "The later all-query leak is absent from the summary when the first gate lacks that field.",
        [legacy_first, leaking_later], manifest)
    gained = copy.deepcopy(baseline_row)
    gained["arms"]["unmasked"]["model_six_act"] = [[0.1, 0.8, 0.025, 0.025, 0.025, 0.025], 1.0]
    add("gained_winner_reported_still_top", "Expert tool 0 becomes top only after masking; this is gain, not retention.",
        [gained], baseline_manifest)
    unresolved = copy.deepcopy(baseline_row)
    unresolved["arms"]["unmasked"]["model_six_act"][1] = 1e-6
    add("unresolved_baseline_reported_still_top", "The unmasked model reading is below the declared floor.",
        [unresolved], baseline_manifest)
    empty = copy.deepcopy(baseline_row)
    empty["arms"]["current_note"]["model_six_act"][1] = 1e-9
    empty["arms"]["current_note"]["lens_act"]["1"]["mass"] = 1e-9
    add("empty_denominator_reported_zero_share", "No masked model reading is resolved; inspect n=0 and share=0.0.",
        [empty], baseline_manifest)
    return {
        "method": "Full frozen source AST; verified standard-library imports stripped and supplied explicitly; in-memory inputs and output; actual argparse parser with fixed fixture arguments.",
        "source_relative_path": "source/workspace_w3b_analyze.py",
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "scientific_outcome_files_read": False,
        "model_or_native_imports": False,
        "fixture_count": len(cases),
        "fixtures": cases,
    }
