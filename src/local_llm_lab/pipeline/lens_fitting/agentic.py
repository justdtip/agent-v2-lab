"""Bounded, restartable agentic Jacobian fits. Importing this module loads no model."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


class RowStore:
    """A single-writer store: receipts commit complete rows, orphan files never count.

    Runlock owns process exclusion. The immutable contract binds inputs, code, precision,
    sample and schedule; receipts bind each array. Resume never trusts a file name alone.
    """

    def __init__(self, root, contract):
        self.root, self.contract = Path(root), contract
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "contract.json"
        if path.exists():
            if canonical(json.loads(path.read_text())) != canonical(contract):
                raise ValueError("resume contract differs")
        else:
            if any(self.root.iterdir()):
                raise ValueError("nonempty store lacks contract")
            atomic_json(path, contract)
        self.receipts = {}
        for path in sorted(self.root.glob("row-*.json")):
            record = json.loads(path.read_text())
            i = record["index"]
            if i not in contract["rows"] or path.name != f"row-{i}.json":
                raise ValueError("row outside contract")
            archive = self.root / f"row-{i}.npz"
            if not archive.exists() or sha(archive) != record["sha256"]:
                raise ValueError("row archive digest mismatch")
            with np.load(archive, allow_pickle=False) as arrays:
                self._validate({int(k[1:]): arrays[k] for k in arrays.files})
            self.receipts[i] = record

    @property
    def completed(self):
        return [i for i in self.contract["rows"] if i in self.receipts]

    def _validate(self, maps):
        if set(maps) != set(self.contract["layers"]):
            raise ValueError("row layer set differs")
        d = self.contract["hidden"]
        if any(
            m.shape != (d, d) or m.dtype != np.float32 or not np.isfinite(m).all()
            for m in maps.values()
        ):
            raise ValueError("row map shape, dtype or finiteness differs")

    def put(self, index, maps, metadata):
        if index in self.receipts:
            raise ValueError("row already committed")
        if index not in self.contract["rows"]:
            raise ValueError("row outside contract")
        self._validate(maps)
        tmp = self.root / f"row-{index}.npz.tmp"
        with tmp.open("wb") as stream:
            np.savez(stream, **{f"J{k}": v for k, v in maps.items()})
            stream.flush()
            os.fsync(stream.fileno())
        path = self.root / f"row-{index}.npz"
        os.replace(tmp, path)
        record = {"index": index, "sha256": sha(path), "metadata": metadata}
        atomic_json(self.root / f"row-{index}.json", record)
        self.receipts[index] = record

    def mean(self):
        if self.completed != self.contract["rows"]:
            raise ValueError("incomplete requested population cannot become a lens")
        d = self.contract["hidden"]
        total = {k: np.zeros((d, d), np.float64) for k in self.contract["layers"]}
        for i in self.completed:
            with np.load(self.root / f"row-{i}.npz", allow_pickle=False) as arrays:
                for k in total:
                    total[k] += arrays[f"J{k}"]
        return {k: (v / len(self.completed)).astype(np.float32) for k, v in total.items()}


def ordered_candidates(rows):
    """One episode per row; deterministic family rounds alternate context-length extremes."""
    if not rows or len({r["task_id"] for r in rows}) != len(rows):
        raise ValueError("candidate episodes must be nonempty and unique")
    groups = {}
    for row in rows:
        groups.setdefault(row["family"], []).append(row)
    for family, group in groups.items():
        group.sort(key=lambda r: (len(r["ids"]), r["task_id"], r["index"]))
        alternating = []
        while group:
            alternating.append(group.pop())
            if group:
                alternating.append(group.pop(0))
        groups[family] = alternating
    out = []
    for i in range(max(map(len, groups.values()))):
        for family in sorted(groups):
            if i < len(groups[family]):
                out.append(groups[family][i])
    return out


def select_sample(rows, *, seconds_per_row, budget_seconds=4 * 3600):
    if not math.isfinite(seconds_per_row) or seconds_per_row <= 0 or budget_seconds <= 0:
        raise ValueError("budget calibration must be positive and finite")
    ordered = ordered_candidates(rows)
    families = len({r["family"] for r in rows})
    n = int(budget_seconds * 0.8 // seconds_per_row)
    n = min(len(ordered), n - n % families)
    if n < families:
        raise ValueError("budget cannot cover every eligible family")
    return ordered[:n]


def fit_selected(
    model,
    rows,
    root,
    *,
    binding,
    layers,
    dim_batch,
    max_seq_len,
    budget_seconds,
    seconds_per_row=0,
    progress=None,
):
    """Fit the frozen population; incomplete populations never produce admitted maps."""
    from . import upstream as U

    if not rows or any(len(r["ids"]) > max_seq_len for r in rows):
        raise ValueError("empty population or context truncation requested")
    if (
        layers != sorted(set(layers))
        or not layers
        or not 1 <= layers[0] <= layers[-1] < model.n_layers
    ):
        raise ValueError("invalid source layers")
    if len({r["index"] for r in rows}) != len(rows):
        raise ValueError("duplicate fit rows")
    if budget_seconds <= 0 or not math.isfinite(budget_seconds):
        raise ValueError("invalid time budget")
    root = Path(root)
    contract = {
        "schema_version": 1,
        "binding": binding,
        "rows": [r["index"] for r in rows],
        "row_data_sha256": hashlib.sha256(canonical(rows)).hexdigest(),
        "layers": layers,
        "hidden": model.d_model,
        "depth": model.n_layers,
        "dim_batch": dim_batch,
        "max_seq_len": max_seq_len,
        "budget_seconds": budget_seconds,
        "seconds_per_row": seconds_per_row,
        "estimator_schedule": "graph-once",
    }
    store = RowStore(root / "rows", contract)
    up = U.load_upstream()
    completed_seconds = sum(r["metadata"]["seconds"] for r in store.receipts.values())
    started = time.monotonic()
    for row in rows:
        if row["index"] in store.completed:
            continue
        used = completed_seconds + time.monotonic() - started
        if used + seconds_per_row >= budget_seconds:
            status = {
                "status": "budget_exhausted",
                "completed": store.completed,
                "requested": contract["rows"],
                "compute_seconds": used,
            }
            atomic_json(root / "status.json", status)
            return status
        if progress:
            progress({"event": "before_row", "index": row["index"]})
        t0 = time.monotonic()
        fitted = U.fit_upstream_jacobian(
            model,
            [row],
            source_layers=[i - 1 for i in layers],
            max_seq_len=max_seq_len,
            dim_batch=dim_batch,
            device=str(model.input_device),
            dtype="float32",
            estimator_schedule="graph-once",
            upstream=up,
        )
        if fitted.skipped or fitted.n_prompts != 1:
            raise ValueError("requested row was not fitted")
        from dataclasses import asdict

        metadata = asdict(fitted)
        metadata.pop("jacobians")
        metadata["seconds"] = time.monotonic() - t0
        store.put(row["index"], {k + 1: v for k, v in fitted.jacobians.items()}, metadata)
        if progress:
            progress(
                {
                    "event": "row",
                    "index": row["index"],
                    "seconds": metadata["seconds"],
                    "completed": len(store.completed),
                }
            )
    maps = store.mean()
    records = [store.receipts[i]["metadata"] for i in store.completed]
    final = dict(records[-1])
    final.pop("seconds")
    final.update(
        jacobians={k - 1: v for k, v in maps.items()},
        n_prompts=len(rows),
        per_prompt=[r["per_prompt"][0] for r in records],
        elapsed_s=sum(r["seconds"] for r in records),
    )
    # Recompute the selector descriptor on EVERY fitted length, not merely the last row.
    final["selector"] = final["selector"] | U.selector_descriptor(
        U.default_position_selector(upstream=up),
        probe_lengths=sorted({len(r["ids"]) for r in rows}),
    )
    fitted = U.UpstreamJacobianFit(**final)
    corpus = {
        "manifest": binding.get("corpus_manifest", "fixture"),
        "manifest_sha256": binding.get("corpus_sha256"),
        "split": "fit",
        "domain": "agentic",
        "rows": contract["rows"],
    }
    nu = U.declare_nu(fitted, num_layers=model.n_layers, corpus=corpus)
    manifest = {
        "schema_version": 1,
        "n_layers": model.n_layers,
        "d_model": model.d_model,
        "n_rows": len(rows),
        "source_layers_repo": layers,
        "corpus": corpus,
        "precision": "float32",
        "forward_batch": 1,
        "anchor_batch": 1,
        "dim_batch": dim_batch,
        "max_seq_len": max_seq_len,
        "estimator_schedule": "graph-once",
        "load_report_sha256": binding.get("checkpoint"),
        "binding": binding,
        "contract_sha256": sha(root / "rows" / "contract.json"),
    }
    archive = root / "exact-maps.npz"
    if archive.exists():
        with np.load(archive, allow_pickle=False) as stored:
            if set(stored.files) != {f"J{k}" for k in maps} or any(
                stored[f"J{k}"].tobytes() != maps[k].tobytes() for k in maps
            ):
                raise ValueError("existing final archive differs from row store")
    else:
        tmp = root / "exact-maps.npz.tmp"
        with tmp.open("wb") as stream:
            np.savez(stream, **{f"J{k}": v for k, v in maps.items()})
        os.replace(tmp, archive)
    for name, value in (("manifest.json", manifest), ("nu.json", nu)):
        path = root / name
        if path.exists() and canonical(json.loads(path.read_text())) != canonical(value):
            raise ValueError(f"existing {name} differs from row store")
        if not path.exists():
            atomic_json(path, value)
    status = {
        "status": "complete",
        "completed": store.completed,
        "compute_seconds": fitted.elapsed_s,
        "archive_sha256": sha(archive),
    }
    atomic_json(root / "status.json", status)
    return status


def check_exactness(reference, candidate):
    """Predeclared float32 scale-aware comparison, not a bit-identity claim."""
    if set(reference) != set(candidate):
        raise ValueError("exactness layer mismatch")
    checks = {}
    for layer, ref in reference.items():
        other = candidate[layer]
        if ref.shape != other.shape or not np.isfinite(ref).all() or not np.isfinite(other).all():
            raise ValueError("exactness shape or finite-value failure")
        scale = max(1.0, float(np.max(np.abs(ref))))
        bound = float(8 * np.finfo(np.float32).eps * scale)
        error = float(np.max(np.abs(ref - other)))
        checks[str(layer)] = {"max_abs_error": error, "bound": bound}
        if error > bound:
            raise ValueError(f"exactness failed at layer {layer}: {error} > {bound}")
    return {"passed": True, "layers": checks}


def calibration_rows(rows):
    ordered = sorted(rows, key=lambda r: (len(r["ids"]), r["index"]))
    return list(
        {r["index"]: r for r in (ordered[0], ordered[len(ordered) // 2], ordered[-1])}.values()
    )


def freeze_plan(rows, measurements, *, binding, dim_batch, budget_seconds, peak_limit_gib):
    expected = {r["index"] for r in calibration_rows(rows)}
    if {m["index"] for m in measurements} != expected or len(measurements) != len(expected):
        raise ValueError("calibration must cover short, median and longest full contexts")
    if (
        type(dim_batch) is not int
        or dim_batch < 1
        or not math.isfinite(peak_limit_gib)
        or peak_limit_gib <= 0
        or not math.isfinite(budget_seconds)
        or budget_seconds <= 0
    ):
        raise ValueError("calibration resource bounds must be positive and finite")
    lookup = {r["index"]: r for r in rows}
    for m in measurements:
        if (
            not math.isfinite(m.get("peak_gib", float("nan")))
            or not 0 <= m["peak_gib"] <= peak_limit_gib
            or not math.isfinite(m.get("seconds", float("nan")))
            or m["seconds"] <= 0
            or m.get("context_tokens") != len(lookup[m["index"]]["ids"])
            or m.get("dim_batch") != dim_batch
            or m.get("forward_batch") != 1
            or m.get("layers") != [18, 24]
        ):
            raise ValueError("calibration timing, memory, context or schedule differs")
    seconds = max(m["seconds"] for m in measurements)
    selected = select_sample(rows, seconds_per_row=seconds, budget_seconds=budget_seconds)
    return {
        "schema_version": 1,
        "binding": binding,
        "rows": [r["index"] for r in selected],
        "sample_sha256": hashlib.sha256(canonical(selected)).hexdigest(),
        "layers": [18, 24],
        "dim_batch": dim_batch,
        "forward_batch": 1,
        "max_seq_len": max(len(r["ids"]) for r in selected),
        "budget_seconds": budget_seconds,
        "seconds_per_row": seconds,
        "peak_limit_gib": peak_limit_gib,
        "calibration": measurements,
        "headroom_fraction": 0.2,
        "predicted_fit_seconds": seconds * len(selected),
    }


def validate_plan(plan, rows, *, binding, peak_limit_gib, calibration_path):
    """Rebuild the frozen recipe from its bound calibration receipt before model loading."""
    if not isinstance(plan, dict) or plan.get("schema_version") != 1:
        raise ValueError("unsupported calibration plan")
    path = Path(calibration_path)
    if not path.is_file() or sha(path) != plan.get("calibration_sha256"):
        raise ValueError("calibration receipt digest differs")
    receipt = json.loads(path.read_text())
    if (
        receipt.get("binding") != binding
        or plan.get("binding") != binding
        or plan.get("peak_limit_gib") != peak_limit_gib
    ):
        raise ValueError("plan input/code identity or memory ceiling changed")
    checks = receipt.get("exactness", {})
    if (
        checks.get("passed") is not True
        or set(checks.get("layers", {})) != {"17", "23"}
        or plan.get("exactness") != checks
    ):
        raise ValueError("calibration exactness was not passed for both source layers")
    for entry in checks["layers"].values():
        error, bound = entry.get("max_abs_error"), entry.get("bound")
        if (
            not isinstance(error, (float, int))
            or not isinstance(bound, (float, int))
            or not math.isfinite(error)
            or not math.isfinite(bound)
            or error < 0
            or bound <= 0
            or error > bound
        ):
            raise ValueError("calibration exactness contains invalid bounds")
    recipe = receipt["recipe"]
    if recipe.get("layers") != [18, 24]:
        raise ValueError("calibration source layers differ from the 4B pilot")
    rebuilt = freeze_plan(
        rows,
        receipt["measurements"],
        binding=binding,
        dim_batch=recipe["dim_batch"],
        budget_seconds=recipe["budget_seconds"],
        peak_limit_gib=recipe["peak_limit_gib"],
    )
    rebuilt.update(exactness=checks, calibration_sha256=sha(path))
    if canonical(rebuilt) != canonical(plan):
        raise ValueError("plan differs from the measured calibration recipe")
    return plan


def source_binding(corpus, snapshot, checkpoint_hashes):
    """Bind code bytes as well as commit: dirty source is not mislabelled committed."""
    from local_llm_lab import hf_text, torch_jacobian

    from . import corpus as C
    from . import upstream as U
    from . import workspace_corpus as W

    paths = [Path(x.__file__) for x in (hf_text, torch_jacobian, C, U, W)] + [Path(__file__)]
    from local_llm_lab import spawn

    commit = spawn.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parents[4],
    ).stdout.strip()
    return {
        "corpus_manifest": str(Path(corpus).resolve()),
        "corpus_sha256": sha(corpus),
        "snapshot": str(Path(snapshot).resolve()),
        "checkpoint": checkpoint_hashes,
        "commit": commit,
        "source_sha256": {str(p): sha(p) for p in paths},
        "upstream": U.load_upstream().provenance(),
    }


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["calibrate", "fit"])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--dim-batch", type=int, default=1)
    parser.add_argument("--peak-limit-gib", type=float, required=True)
    parser.add_argument("--budget-seconds", type=float, default=4 * 3600)
    args = parser.parse_args(argv)
    from .corpus import read_corpus

    rows = read_corpus(args.corpus)
    if not rows or any(r.get("domain") != "agentic" for r in rows):
        parser.error("a verified agentic corpus is required")
    if args.dim_batch < 1 or not 0 < args.peak_limit_gib < float("inf"):
        parser.error("positive finite resource limits are required")
    if not args.device.startswith("cuda"):
        parser.error("checkpoint pilot requires CUDA; use the injectable fitter for CPU fixtures")
    if args.stage == "calibrate" and args.output.exists():
        parser.error("calibration output must be new")
    if args.stage == "fit" and args.plan is None:
        parser.error("fit requires a frozen calibration plan")
    # File checks happen before taking a slot or mapping weights.
    from local_llm_lab import device, hf_text, runlock
    from local_llm_lab.models import load_model_spec

    spec = load_model_spec("gemma3-4b-cuda-bf16")
    if "models--" + spec.base.replace("/", "--") not in args.snapshot.resolve().parts:
        parser.error(f"pilot requires the official cached snapshot for {spec.base}")
    checkpoint = hf_text.checkpoint_metadata(args.snapshot)["sha256"]
    binding = source_binding(args.corpus, args.snapshot, checkpoint)
    plan = json.loads(args.plan.read_text()) if args.plan else None
    corpus_manifest = json.loads(args.corpus.read_text())
    tokenizer_record = corpus_manifest.get("sources", {}).get("tokenizer")
    if not isinstance(tokenizer_record, dict) or sha(
        args.snapshot / "tokenizer.json"
    ) != tokenizer_record.get("sha256"):
        parser.error("snapshot tokenizer differs from the frozen workspace corpus")
    if plan is not None:
        validate_plan(
            plan,
            rows,
            binding=binding,
            peak_limit_gib=args.peak_limit_gib,
            calibration_path=args.plan.parent / "calibration.json",
        )
    runlock.hold_model_run_lock(command="agentic lens " + args.stage, session="codex-agentic")
    device.pin(seed=0)
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    free, total = torch.cuda.mem_get_info(args.device)
    if args.peak_limit_gib * 2**30 > free * device.budget_fraction():
        raise ValueError("declared peak exceeds the currently available device budget")
    torch.cuda.set_per_process_memory_fraction(args.peak_limit_gib * 2**30 / total, args.device)
    model, report = hf_text.load_text_causal_lm(
        args.snapshot, dtype="bfloat16", attn_implementation="eager", device=args.device
    )
    model.to(torch.float32).eval().requires_grad_(False)
    if report["sha256"] != checkpoint:
        raise ValueError("loaded checkpoint differs from preflight")
    from . import upstream as U

    up = U.load_upstream()
    from jlens.hf import HFLensModel

    wrapped = HFLensModel(model, tokenizer=None)
    if wrapped.n_layers <= max([18, 24]):
        raise ValueError("requested source layers are outside the model")

    def memory():
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_reserved() / 2**30
            if peak > args.peak_limit_gib:
                raise ValueError(f"memory ceiling exceeded: {peak:.3f} GiB")
            return peak
        raise ValueError("checkpoint pilot requires CUDA; CPU is reserved for fixtures")

    if args.stage == "fit":
        lookup = {r["index"]: r for r in rows}
        selected = [lookup[i] for i in plan["rows"]]
        if hashlib.sha256(canonical(selected)).hexdigest() != plan["sample_sha256"]:
            raise ValueError("frozen sample changed")

        def progress(event):
            event["peak_gib"] = memory()
            print(json.dumps(event), flush=True)

        result = fit_selected(
            wrapped,
            selected,
            args.output,
            binding=binding,
            layers=plan["layers"],
            dim_batch=plan["dim_batch"],
            max_seq_len=plan["max_seq_len"],
            budget_seconds=plan["budget_seconds"],
            seconds_per_row=plan["seconds_per_row"],
            progress=progress,
        )
        print(json.dumps(result), flush=True)
        return 0 if result["status"] == "complete" else 2
    args.output.mkdir(parents=True)
    atomic_json(
        args.output / "bounds.json",
        {
            "binding": binding,
            "device": device.describe(),
            "peak_limit_gib": args.peak_limit_gib,
            "dim_batch": args.dim_batch,
            "exactness": "max_abs <= 8*float32_epsilon*max(1,max_abs(reference))",
            "status": "unexecuted",
        },
    )
    samples = calibration_rows(rows)
    measured = []
    # Begin at width one. A wider requested batch gets its own correctness and memory checks.
    reference = U.fit_upstream_jacobian(
        wrapped,
        [samples[0]],
        source_layers=[17, 23],
        dim_batch=1,
        max_seq_len=len(samples[0]["ids"]),
        device=args.device,
        upstream=up,
    )
    memory()
    checks = None
    for row in samples:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        fit = U.fit_upstream_jacobian(
            wrapped,
            [row],
            source_layers=[17, 23],
            dim_batch=args.dim_batch,
            max_seq_len=len(row["ids"]),
            device=args.device,
            upstream=up,
            estimator_schedule="graph-once",
        )
        peak = memory()
        measured.append(
            {
                "index": row["index"],
                "context_tokens": len(row["ids"]),
                "seconds": time.monotonic() - start,
                "peak_gib": peak,
                "dim_batch": args.dim_batch,
                "forward_batch": 1,
                "layers": [18, 24],
            }
        )
        if row is samples[0]:
            checks = check_exactness(reference.jacobians, fit.jacobians)
        atomic_json(
            args.output / "calibration.json",
            {
                "binding": binding,
                "exactness": checks,
                "measurements": measured,
                "recipe": {
                    "dim_batch": args.dim_batch,
                    "budget_seconds": args.budget_seconds,
                    "peak_limit_gib": args.peak_limit_gib,
                    "layers": [18, 24],
                },
            },
        )
        print(json.dumps(measured[-1]), flush=True)
    plan = freeze_plan(
        rows,
        measured,
        binding=binding,
        dim_batch=args.dim_batch,
        budget_seconds=args.budget_seconds,
        peak_limit_gib=args.peak_limit_gib,
    )
    plan["exactness"] = checks
    plan["calibration_sha256"] = sha(args.output / "calibration.json")
    atomic_json(args.output / "plan.json", plan)
    print(
        json.dumps(
            {
                "event": "plan",
                "rows": len(plan["rows"]),
                "predicted_fit_seconds": plan["predicted_fit_seconds"],
            }
        ),
        flush=True,
    )
    return 0
