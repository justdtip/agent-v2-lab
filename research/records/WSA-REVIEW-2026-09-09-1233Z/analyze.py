"""Reproduce this review from frozen metadata only; standard library, no model imports."""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASIS = "computed from pinned producer-branch records; no new device execution"
def value(x, basis=BASIS):
    return {"value": x, "basis": basis}

def read(name):
    return json.loads((ROOT / "sources" / name).read_text())

def verify_bytes(data, expected, name):
    got = hashlib.sha256(data).hexdigest()
    if got != expected:
        raise ValueError(f"source hash mismatch: {name}")

def analyze():
    manifest = json.loads((ROOT / "SOURCE-MANIFEST.json").read_text())
    for name, entry in manifest["sources"].items():
        verify_bytes((ROOT / "sources" / name).read_bytes(), entry["sha256"], name)
    boundary = read("calibration/boundary.json")["results"]
    native = read("calibration/batch-invariance.json")
    unchanged = [r for r in boundary if r["check"] == "unchanged_residual"]
    width_one = [r for r in unchanged if r["width"] == 1]
    wide = [r for r in unchanged if r["width"] != 1]
    assert len(width_one) == 6 and all(r["bitwise_identical"] for r in width_one)
    assert len(wide) == 12 and all(not r["bitwise_identical"] for r in wide)
    by_width = defaultdict(list)
    for r in native["results"]:
        by_width[r["width"]].append(r)
    batch_summary = {}
    for width, rows in sorted(by_width.items()):
        rows.sort(key=lambda r: r["repo_layer"])
        assert len(rows) == 34 and all(not r["bitwise_identical"] for r in rows)
        assert all(r["rows_identical_to_each_other"] for r in rows)
        batch_summary[str(width)] = {
            "differing_layers": value(len(rows)),
            "first_layer_MAE": value(rows[0]["mean_abs_difference"]),
            "target_layer_MAE": value(rows[-1]["mean_abs_difference"]),
            "units": "absolute residual activation units, not a percentage",
        }
    tree = ast.parse((ROOT / "sources/calibration/batch_invariance.py").read_text())
    hook_fields = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value == "hook_is_implicated":
                    hook_fields.append(v)
    assert len(hook_fields) == 1
    expr = hook_fields[0]
    assert isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.And)
    assert isinstance(expr.values[-1], ast.Constant) and expr.values[-1].value is False
    # Known outcomes: neither possible truth value of the preceding expression can change it.
    assert [x and False for x in (True, False)] == [False, False]
    raw = (ROOT / "sources/plan/capture-set.jsonl").read_bytes()
    decisions = [json.loads(line) for line in raw.splitlines()]
    assert len({(d["task_id"], d["step"]) for d in decisions}) == len(decisions)
    triples = [[d["task_id"], d["step"], d["prompt_sha256"]] for d in decisions]
    logical_hash = hashlib.sha256(json.dumps(triples, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    inputs = read("plan/prereg-inputs.json")
    assert logical_hash == inputs["decisions"]["capture_set_sha256"]
    tasks = defaultdict(list)
    for d in decisions:
        tasks[d["task_id"]].append(d)
    transitions = Counter()
    by_split = defaultdict(Counter)
    ordinary_train_per_episode = Counter()
    corrective_per_episode = Counter()
    for task, rows in tasks.items():
        rows.sort(key=lambda d: d["step"])
        for before, after in zip(rows, rows[1:]):
            delta = after["step"] - before["step"]
            assert delta > 0
            if after["recovery"]:
                kind = "corrective_gap" if delta > 1 else "corrective_contiguous"
                corrective_per_episode[task] += 1
            else:
                assert delta == 1, "an unregistered ordinary gap appeared"
                kind = "ordinary"
                if after["split"] == "train":
                    ordinary_train_per_episode[task] += 1
            transitions[kind] += 1
            by_split[after["split"]][kind] += 1
    assert dict(transitions) == {"ordinary": 5948, "corrective_contiguous": 244, "corrective_gap": 309}
    assert sum(ordinary_train_per_episode.values()) == 4122
    assert len(corrective_per_episode) == 553 and max(corrective_per_episode.values()) == 1
    e1_counts = Counter(d["task_id"] for d in decisions if d["split"] == "test")
    def concentration_units(counts):
        total = sum(counts.values())
        squares = sum(n*n for n in counts.values())
        return {"observations": value(total), "episodes": value(len(counts)),
                "sum_squared_episode_counts": value(squares),
                "inverse_squared_episode_weights": value(total*total/squares),
                "interpretation": "arithmetic illustration for independently sampled bounded episode means; not an estimated effective sample size or validated confidence bound"}
    assert sum(e1_counts.values()) == 1781 and len(e1_counts) == 240
    illustration = "analytic Bernoulli(1/2) counterexample; not model data"
    counterexample = {"independent_episodes": value(1, illustration), "identical_rows_per_episode": value(8, illustration),
                      "actual_mean_variance": value(0.25, illustration), "false_independent_row_variance": value(0.25/8, illustration),
                      "basis_note": "analytic Bernoulli(1/2) illustration; not model data"}
    return {"basis": BASIS, "model_execution": False,
      "calibration": {"width_one_entries_passed": value(len(width_one)),
        "distinct_zero_step_source_interventions": value(len({r["repo_layer"] for r in width_one})),
        "cross_width_entries_failed": value(len(wide)), "native_by_width": batch_summary,
        "hook_is_implicated_field_always_false": value(True),
        "full_derivative_protocol_status": "unexecuted"},
      "corpus": {"decisions": value(len(decisions)), "episodes": value(len(tasks)),
        "decisions_per_split": {k:value(v) for k,v in sorted(Counter(d["split"] for d in decisions).items())},
        "transitions": {k:value(v) for k,v in sorted(transitions.items())},
        "transitions_per_split": {k:{j:value(n) for j,n in sorted(v.items())} for k,v in sorted(by_split.items())},
        "corrective_episodes": value(len(corrective_per_episode)),
        "capture_set_logical_triples_sha256": logical_hash,
        "capture_set_file_sha256": hashlib.sha256(raw).hexdigest()},
      "confidence_review": {"E1_test_decisions": concentration_units(e1_counts),
        "E2_ordinary_train": concentration_units(ordinary_train_per_episode),
        "approved_epsilon_arithmetic_only": {str(eps):value(math.ceil(math.log(2*12/0.05)/(eps*eps))) for eps in (0.06,0.04,0.11)},
        "dependence_counterexample": counterexample,
        "approved_tolerances_modified": False}}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = analyze()
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    path = ROOT / "analysis.json"
    if args.check:
        if path.read_text() != text:
            raise ValueError("analysis.json does not reproduce")
        manifest = json.loads((ROOT / "SOURCE-MANIFEST.json").read_text())
        # Every source hash check must reject a known corruption, not only the first source.
        rejected = 0
        for name, entry in manifest["sources"].items():
            data = (ROOT / "sources" / name).read_bytes()
            try:
                verify_bytes(data + b"x", entry["sha256"], name)
            except ValueError:
                rejected += 1
            else:
                raise AssertionError("corrupted source accepted: " + name)
        print(json.dumps({"status":"passed", "reproduced":True, "corrupted_sources_rejected":rejected, "model_execution":False}))
    else:
        path.write_text(text)
        print(text)

if __name__ == "__main__":
    main()
