"""Audit frozen ladder cells with the standard library; no producer code or model imports."""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
BASIS = "recomputed from pinned device results; no new model execution"
def val(x, basis=BASIS):
    return {"value": x, "basis": basis}

def verify_source(data, sha):
    if hashlib.sha256(data).hexdigest() != sha:
        raise ValueError("source hash mismatch")

def load_cells():
    rows = []
    for name in ("ladder.jsonl", "ladder-deep.jsonl"):
        data = (ROOT / "sources/calibration/artefacts" / name).read_bytes()
        if not data.endswith(b"\n"):
            raise ValueError("incomplete JSONL")
        rows.extend(json.loads(line) for line in data.splitlines())
    unique = {}
    duplicates = 0
    for row in rows:
        key = tuple(row[k] for k in ("precision", "repo_layer", "position", "direction", "cotangent", "k"))
        if key in unique:
            if unique[key] != row:
                raise ValueError("conflicting repeat: " + str(key))
            duplicates += 1
        unique[key] = row
        assert row["basis"] == "measured-here"
        assert row["h"] == row["h0"] * 2**(-row["k"])
        assert row["absolute_error"] == abs(row["d_h"] - row["a_requested"])
        if row["a_requested"]:
            assert row["relative_error"] == row["absolute_error"] / abs(row["a_requested"])
    assert len(rows) == 864 and len(unique) == 828 and duplicates == 36
    return rows, list(unique.values()), duplicates

def boundary_guard_counterexample():
    # Evaluate only the small gating expression, extracted as syntax. Never execute the script.
    text = (ROOT / "sources/calibration/boundary.py").read_text()
    tree = ast.parse(text)
    candidates = [n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == "anchor == 'width' and (not entry['bitwise_identical'])"]
    assert len(candidates) == 1
    expr = ast.Expression(candidates[0].test)
    allowed = (ast.Expression, ast.BoolOp, ast.And, ast.Compare, ast.Eq, ast.UnaryOp,
               ast.Not, ast.Subscript, ast.Name, ast.Load, ast.Constant)
    assert all(isinstance(n, allowed) for n in ast.walk(expr))
    code = compile(expr, "<extracted-boundary-condition>", "eval")
    def gate(anchor, passed):
        return eval(code, {"__builtins__": {}}, {"anchor": anchor, "entry": {"bitwise_identical": passed}})
    actual = gate("one", False)
    wide = gate("width", False)
    assert actual is False and wide is True
    return {"source_line": candidates[0].lineno,
            "corrupted_width_one_result_appended_to_failures": actual,
            "corrupted_matched_width_64_result_appended_to_failures": wide,
            "basis": "source-expression mutation check; fixture only, no model"}

def analyze():
    manifest = json.loads((ROOT / "SOURCE-MANIFEST.json").read_text())
    for name, item in manifest["sources"].items():
        verify_source((ROOT / "sources" / name).read_bytes(), item["sha256"])
    rows, cells, repeats = load_cells()
    groups = defaultdict(list)
    for row in cells:
        groups[(row["precision"], row["repo_layer"], row["k"])].append(row)
    summaries = []
    for (precision, layer, k), g in sorted(groups.items()):
        assert len(g) == 18
        requested = [row["a_requested"] for row in g]
        errors = [row["relative_error"] for row in g]
        worst = max(g, key=lambda row: row["relative_error"])
        directions = [row for row in g if row["cotangent"] == "coordinate:0"]
        assert len(directions) == 6
        # The same forward diagnostics appear under three cotangents; do not count them thrice.
        for row in directions:
            tied = [x for x in g if x["direction"] == row["direction"]]
            for field in ("h", "h0", "realized_plus_norm", "realized_minus_norm", "midpoint_shift_norm",
                          "unchanged_input_fraction_in_support", "equal_output_fraction", "odd_norm"):
                assert all(x[field] == row[field] for x in tied)
        summaries.append({"precision": precision, "layer": layer, "k": k, "basis": BASIS,
            "h": g[0]["h"], "epsilon_scale": 0.01 * 2**(-k), "projection_count": len(g),
            "direction_count": len(directions), "median_relative_error": statistics.median(errors),
            "min_relative_error": min(errors), "max_relative_error": max(errors),
            "projected_set_L2_error": math.sqrt(sum((r["d_h"]-r["a_requested"])**2 for r in g) / sum(a*a for a in requested)),
            "worst_projection": {key:worst[key] for key in ("direction", "cotangent", "a_requested", "d_h", "absolute_error", "relative_error")},
            "unchanged_input_min": min(r["unchanged_input_fraction_in_support"] for r in directions),
            "unchanged_input_max": max(r["unchanged_input_fraction_in_support"] for r in directions),
            "equal_summed_output_min": min(r["equal_output_fraction"] for r in directions),
            "equal_summed_output_median": statistics.median(r["equal_output_fraction"] for r in directions),
            "equal_summed_output_max": max(r["equal_output_fraction"] for r in directions)})
    best = []
    for precision in ("float32", "native"):
        for layer in (1,17,33):
            choices = [g for g in summaries if g["precision"] == precision and g["layer"] == layer]
            best.append(min(choices, key=lambda g:g["median_relative_error"]))
    k16 = [r for r in cells if (r["precision"],r["repo_layer"],r["k"],r["cotangent"]) == ("native",1,16,"coordinate:0")]
    assert len(k16) == 6
    coordinate = [r for r in k16 if r["direction"].startswith("coordinate")]
    dense = [r for r in k16 if r["direction"].startswith("dense")]
    assert len(coordinate) == 4 and len(dense) == 2
    assert all(r["realized_plus_norm"] == r["realized_minus_norm"] == r["odd_norm"] == 0 for r in coordinate)
    assert all(r["odd_norm"] > 0 and r["equal_output_fraction"] < 1 for r in dense)
    assert all(r["unchanged_input_fraction_in_support"] > 0.99 for r in dense)
    raw_boundary = json.loads((ROOT / "sources/calibration/artefacts/boundary-anchored.json").read_text())
    boundaries = [r for r in raw_boundary["results"] if r["check"] == "unchanged_residual"]
    counts = Counter((r["anchor"],r["width"],r["bitwise_identical"]) for r in boundaries)
    assert counts == Counter({("one",1,True):6, ("one",64,False):6, ("width",64,True):6,
                              ("one",256,False):6, ("width",256,True):6})
    fp1 = next(r for r in best if (r["precision"],r["layer"]) == ("float32",1))
    fp33 = next(r for r in best if (r["precision"],r["layer"]) == ("float32",33))
    # Exact algebraic cancellation example: equal position-sums need not mean equal positions.
    plus, minus = [1, -1], [0, 0]
    assert sum(plus) == sum(minus) and all(a != b for a,b in zip(plus,minus))
    return {"basis": BASIS, "model_execution": False,
        "raw_cells": val(len(rows)), "unique_cells": val(len(cells)), "identical_repeat_cells": val(repeats),
        "summaries": summaries, "best_medians": best,
        "native_layer1_k16_directions": [{key:r[key] for key in ("direction","a_requested","a_realized_direction","d_h","realized_plus_norm","realized_minus_norm","unchanged_input_fraction_in_support","equal_output_fraction","odd_norm")} for r in k16],
        "k16_interpretation": "4 coordinate inputs are exact no-ops; 2 dense inputs mostly round away and retain nonzero summed responses",
        "source_guard_counterexample": boundary_guard_counterexample(),
        "boundary_counts": [{"anchor":a,"forward_width":w,"equal":b,"count":n,"basis":BASIS} for (a,w,b),n in sorted(counts.items())],
        "best_scale_ratios": {"normalized_epsilon_late_over_early":val(fp33["epsilon_scale"]/fp1["epsilon_scale"]),
          "absolute_h_late_over_early":val(fp33["h"]/fp1["h"])},
        "cancellation_counterexample": {"plus_by_position":plus,"minus_by_position":minus,
          "sums_equal":True,"positions_equal":False,"basis":"exact analytic illustration, not model data"},
        "unexecuted": ["preserved individual target response vectors before position reduction",
          "ladder boundary comparison in both precisions before the VJPs",
          "same-anchor arithmetic triangle", "graph-once versus sequential checkpoint VJP control",
          "nonzero identity-derivative control", "flipped even-remainder check",
          "measured per-component representable spacing", "full-map validation at chosen steps"]}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = analyze()
    encoded = json.dumps(result,indent=2,sort_keys=True) + "\n"
    path = ROOT / "analysis.json"
    if args.check:
        assert path.read_text() == encoded, "analysis does not reproduce"
        sources = json.loads((ROOT / "SOURCE-MANIFEST.json").read_text())["sources"]
        rejected = 0
        for name,item in sources.items():
            b = (ROOT / "sources" / name).read_bytes()
            try:
                verify_source(b + b"x", item["sha256"])
            except ValueError:
                rejected += 1
            else:
                raise AssertionError("corrupted source accepted")
        print(json.dumps({"status":"passed","corrupted_sources_rejected":rejected,
              "row_arithmetic_verified":result["raw_cells"]["value"],"boundary_guard_blind_spot_reproduced":True,
              "cancellation_counterexample_verified":True,"model_execution":False}))
    else:
        path.write_text(encoded)
        print(json.dumps({"status":"written","raw":result["raw_cells"]["value"],"unique":result["unique_cells"]["value"]}))

if __name__ == "__main__":
    main()
