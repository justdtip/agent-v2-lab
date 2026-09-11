"""Reproduce this confirming review against exactly the recorded source hashes.

Only allowlisted source/metadata files and synthetic arrays are opened. Model runtimes are blocked.
No real addendum is created and the original worktree is not modified.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import contextlib
import hashlib
import importlib
import importlib.abc
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import numpy as np


class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "mlx", "mlx_lm", "transformers"}:
            raise RuntimeError("file-only review forbids " + fullname)
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/Users/daniel.tipton/worktrees/cuda-ws-d"))
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("reviewed-files.json"))
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("evidence.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    sys.meta_path.insert(0, NoModels())
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    evidence = {"reviewed_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                "base_commit": manifest["base_commit"], "python": platform.python_version(),
                "numpy": np.__version__, "checks": {}, "real_capture_reads": 0}
    checks = evidence["checks"]
    def verify_source():
        for path, digest in manifest["source_sha256"].items():
            assert hashlib.sha256((args.source / path).read_bytes()).hexdigest() == digest, path
    verify_source()

    with tempfile.TemporaryDirectory(prefix="codex-e2-confirm-run-") as tmp:
        root = Path(tmp)
        for path in manifest["source_sha256"]:
            dst = root / path; dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes((args.source / path).read_bytes())
            assert hashlib.sha256(dst.read_bytes()).hexdigest() == manifest["source_sha256"][path]
        record = root / "research/records/STATE-PLAN-PROGRESS-2026-09-09"
        sys.path[:0] = [str(root / "src"), str(record)]

        import pytest
        suite = [root / "tests" / name for name in [
            "test_e2_review_regressions.py", "test_state_transport_certified.py",
            "test_state_transport.py", "test_state_read_gate.py", "test_state_tolerances.py"]]
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code = pytest.main([*(str(p) for p in suite), "-q", "--confcutdir", str(root),
                                "--rootdir", str(root), "-o", "addopts=", "-p", "no:cacheprovider"])
        checks["affected_suite"] = {"exit_code": int(code), "output": output.getvalue(),
                                    "files": [str(p.relative_to(root)) for p in suite]}
        assert code == 0, output.getvalue()

        import types
        e1 = types.ModuleType("read_e1")
        def forbidden(*a, **kw):
            raise AssertionError("a review may not open a residual capture")
        e1.load_cells = e1.load_layers = forbidden
        sys.modules["read_e1"] = e1
        reader = importlib.import_module("read_e2")
        producer = importlib.import_module("check_findings")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = producer.main()
        assert code == 0, output.getvalue()
        checks["producer_witnesses"] = {"exit_code": code, "output": output.getvalue()}
        gate = importlib.import_module("local_llm_lab.pipeline.state_programme.read_gate")
        parent = gate.require_seal(record)
        for name, digest in parent["files"].items():
            raw = subprocess.check_output(["git", "-C", str(args.source), "show",
                  parent["baseline_commit"] + ":research/records/STATE-PLAN-PROGRESS-2026-09-09/" + name])
            assert hashlib.sha256(raw).hexdigest() == digest
        checks["sealed_baseline_file_hashes_rederived"] = len(parent["files"])
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                reader.main(["--record", str(record), "--captures", str(root / "absent-captures"),
                             "--out", str(root / "must-not-exist.json")])
        except gate.NotSealed as exc:
            checks["actual_reader_refuses_without_new_addendum"] = str(exc).replace(str(record), "<record>")
        else:
            raise AssertionError("changed reader admitted without a new addendum")
        assert not (root / "must-not-exist.json").exists()

        # Independent reduction oracle: unequal transitions per episode, duplicate draws,
        # fitting-only episodes, several strata and fixed episode folds. Numerical fit is the
        # unchanged reference; retrieval and episode/draw reduction are independently computed.
        features = np.random.default_rng(891).normal(size=(24, 3)).astype(np.float32)
        ordinary, subset = [], []
        folds = {f"e{i}": (i // 2) % 2 for i in range(8)}
        strata = {f"e{i}": ("train", f"family{i // 2}", "v") for i in range(8)}
        for i in range(8):
            for j in range(2):
                move = dict(task_id=f"e{i}", source=3*i+j, target=3*i+j+1, m=1,
                            fold=folds[f"e{i}"], candidates=[3*i, 3*i+1, 3*i+2])
                ordinary.append(move)
                if i < 6 and (j == 0 or i % 2 == 0):
                    subset.append(move)
        from local_llm_lab.pipeline.state_programme.transport import fit
        members = defaultdict(list)
        for task in sorted(folds):
            members[strata[task]].append(task)
        rng = np.random.default_rng(31)
        expected = []
        trace = []
        for draw in range(32):
            weights = Counter()
            for key in sorted(members):
                pool = members[key]
                weights.update(pool[i] for i in rng.integers(0, len(pool), len(pool)))
            selected = [m for m in subset if weights[m["task_id"]]]
            rules = {}
            for fold in sorted({folds[m["task_id"]] for m in selected}):
                train = [m for m in ordinary if folds[m["task_id"]] != fold
                         for _ in range(weights[m["task_id"]])]
                rules[fold] = fit(np.array([features[m["source"]] for m in train]),
                                  np.array([features[m["target"]] for m in train]), 1)
                assert rules[fold].max_rank == 1
            per_episode = defaultdict(list)
            for move in selected:
                prediction = np.asarray(rules[folds[move["task_id"]]].apply(
                    features[move["source"]], rank=1), np.float32)
                distances = [float(np.linalg.norm(features[k] - prediction)) for k in move["candidates"]]
                smallest = min(distances)
                winners = [k for k, d in zip(move["candidates"], distances) if d == smallest]
                hit = float(winners == [move["target"]])
                per_episode[move["task_id"]].append(hit - 1 / len(move["candidates"]))
            tasks = sorted(per_episode)
            mean = sum(weights[t] * np.mean(per_episode[t]) for t in tasks) / sum(weights[t] for t in tasks)
            expected.append(float(mean))
            if draw < 2:
                trace.append({"weights": dict(weights), "per_episode": {
                    t: float(np.mean(per_episode[t])) for t in tasks}, "mean": float(mean)})
        actual = reader.refitting_bootstrap(features, subset, ordinary, folds, strata,
                                             rank=1, resamples=32, seed=31, alpha=.05)
        assert actual["complete"] and actual["resamples"] == 32
        want = np.quantile(expected, [.05, .95])
        assert np.allclose([actual["low"], actual["high"]], want, rtol=0, atol=1e-14)
        checks["independent_joint_population_and_episode_weight_oracle"] = {
            "draws": 32, "within_episode_transition_counts": {t: sum(m["task_id"] == t for m in subset) for t in folds},
            "first_two_draws": trace, "expected_endpoints": want.tolist(), "actual": actual,
            "endpoint_convention": "sealed one-sided alpha and 1-alpha, not alpha/2"}

    verify_source()
    checks["source_hashes_unchanged_during_review"] = True
    evidence["verdict"] = "PASS for the reviewed F2/F3 and C1-C3 corrections; not a release"
    args.out.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
