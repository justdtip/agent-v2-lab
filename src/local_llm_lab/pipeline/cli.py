from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from local_llm_lab.models import LoraSpec, ResolvedSpec, load_model_spec
from local_llm_lab.pipeline.branch import run_branch_mining
from local_llm_lab.pipeline.data import SplitSpec, write_dataset
from local_llm_lab.pipeline.evaluate import run_evaluation, wilson
from local_llm_lab.pipeline.prefer import run_prefer
from local_llm_lab.pipeline.report import load_summaries, render
from local_llm_lab.pipeline.rollout import run_rollout
from local_llm_lab.pipeline.tasks import GENERATOR_VERSION
from local_llm_lab.pipeline.transcript import Transcript
from local_llm_lab.project import PROJECT_ROOT, configure_local_cache
from local_llm_lab.provenance import write_provenance

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "agent_v2.yaml"


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("output", "data", "chat_replay"):
        if key in config and config[key]:
            config[key] = (PROJECT_ROOT / config[key]).resolve()
    return config


def _log(message: str) -> None:
    print(f"\n### {time.strftime('%H:%M:%S')} {message}", flush=True)


def dataset_splits(config: dict[str, Any]) -> dict[str, SplitSpec]:
    """Convert one supported data schema to role-aware deterministic split specifications."""
    has_tasks = "tasks" in config
    has_splits = "splits" in config
    if has_tasks == has_splits:
        raise ValueError("data config must define exactly one of tasks or splits")
    if has_tasks:
        raw_tasks = config["tasks"]
        if not isinstance(raw_tasks, dict):
            raise ValueError("tasks must be a mapping")
        result: dict[str, SplitSpec] = {}
        for name, count in raw_tasks.items():
            if name not in {"train", "valid", "test"}:
                raise ValueError(f"legacy task split {name!r} has no standard role")
            result[name] = SplitSpec(count, role=name)
        return result
    raw_splits = config["splits"]
    if not isinstance(raw_splits, dict):
        raise ValueError("splits must be a mapping")
    result = {}
    for name, value in raw_splits.items():
        if not isinstance(value, dict):
            raise ValueError(f"split {name!r} must be a mapping")
        result[name] = SplitSpec(
            value.get("count"),
            difficulty=value.get("difficulty"),
            perturb=value.get("perturb"),
            role=value.get("role", "train"),
        )
    return result


# --------------------------------------------------------------------------- stages


def stage_data(config: dict[str, Any], extra: list[Path]) -> None:
    _log("data: generating expert trajectories with state-carrying notes")
    manifest = write_dataset(
        config["data"],
        dataset_splits(config),
        seed=config["seed"],
        keep_last=config["keep_last"],
        chat_dir=config.get("chat_replay"),
        chat_repeats=config.get("chat_repeats", 1),
        recovery_repeats=config.get("recovery_repeats", 1),
        extra_dirs=extra,
    )
    write_provenance(
        config["output"],
        resolved=None,
        spec=load_model_spec(config["model"]),
        extra={
            "stage": "data",
            "generator_version": GENERATOR_VERSION,
            "dataset_manifest": manifest,
        },
    )
    for split, info in manifest["splits"].items():
        print(
            f"{split:5s}: {info['tasks']:3d} tasks -> {info['rows']:4d} rows "
            f"({info['expert_rows']} expert, {info['chat_rows']} chat, {info['extra_rows']} extra); "
            f"horizon {info['min_horizon']}-{info['max_horizon']}; variants {info['variants']}"
        )
    print(f"Wrote {config['data']}")


def _load_training_base(hf_id: str) -> tuple[Any, Any]:
    """Lazily load the registry's base model only while resolving training targets."""
    configure_local_cache()
    from mlx_lm import load

    return load(hf_id)


def _clear_model_cache() -> None:
    """Release the temporary base-model allocation after architecture resolution."""
    import mlx.core as mx

    mx.clear_cache()


def _resolve_training_spec(config: dict[str, Any]) -> ResolvedSpec:
    """Resolve LoRA targets from the declared base architecture and training overrides."""
    spec = load_model_spec(config["model"])
    train = config["train"]
    keys = "auto" if "lora_keys" not in train else tuple(train["lora_keys"])
    effective = replace(
        spec,
        lora=LoraSpec(
            keys=keys,
            rank=train["rank"],
            scale=train["scale"],
            dropout=train.get("dropout", 0.0),
        ),
    )
    model: Any | None = None
    tokenizer: Any | None = None
    try:
        model, tokenizer = _load_training_base(effective.hf_id)
        return effective.resolve(model, tokenizer)
    finally:
        del model, tokenizer
        _clear_model_cache()


def lora_config(
    config: dict[str, Any],
    resolved: ResolvedSpec,
    iters: int | None = None,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    """Build the mlx-lm ``lora`` YAML payload from the pipeline config (pure; no side effects).

    ``resume_from`` is a saved ``*.safetensors`` adapter file; when given it is written as
    mlx-lm's ``resume_adapter_file`` so training continues from those weights.
    """
    train = dict(config["train"])
    if iters is not None:
        train["iters"] = iters
    output: Path = config["output"]
    lora: dict[str, Any] = {
        "model": resolved.spec.hf_id,
        "train": True,
        "fine_tune_type": "lora",
        "optimizer": "adamw",
        "data": str(config["data"]),
        "seed": config["seed"],
        "num_layers": resolved.num_layers,
        "batch_size": train["batch_size"],
        "grad_accumulation_steps": train["grad_accumulation_steps"],
        "iters": train["iters"],
        "learning_rate": train["learning_rate"],
        "max_seq_length": train["max_seq_length"],
        "grad_checkpoint": bool(train.get("grad_checkpoint", True)),
        "mask_prompt": True,
        "val_batches": train["val_batches"],
        "steps_per_report": train.get("steps_per_report", 10),
        "steps_per_eval": train["steps_per_eval"],
        "save_every": train["save_every"],
        "adapter_path": str(output / "adapters"),
        "test": False,
        "lora_parameters": {
            "keys": list(resolved.lora_keys),
            "rank": train["rank"],
            "scale": train["scale"],
            "dropout": train.get("dropout", 0.0),
        },
    }
    if resume_from is not None:
        lora["resume_adapter_file"] = str(Path(resume_from).resolve())
    return lora


def stage_train(config: dict[str, Any], iters: int | None, resume_from: Path | None = None) -> None:
    output: Path = config["output"]
    adapters = output / "adapters"
    checkpoints = output / "checkpoints"
    if checkpoints.exists():
        shutil.rmtree(checkpoints)
    adapters.mkdir(parents=True, exist_ok=True)
    resolved = _resolve_training_spec(config)
    lora = lora_config(config, resolved, iters=iters, resume_from=resume_from)
    config_path = output / "lora.yaml"
    config_path.write_text(yaml.safe_dump(lora, sort_keys=False), encoding="utf-8")
    print(f"Resolved {len(resolved.lora_keys)} LoRA targets: {list(resolved.lora_keys)}")
    configure_local_cache()
    command = [sys.executable, "-m", "mlx_lm", "lora", "--config", str(config_path)]
    if shutil.which("mlx_lm.lora"):
        command = ["mlx_lm.lora", "--config", str(config_path)]
    _log("train: " + " ".join(command))
    started = time.monotonic()
    log_path = output / "train.log"
    with log_path.open("w", encoding="utf-8") as log:
        # Without PYTHONUNBUFFERED the child block-buffers stdout when it is a pipe rather than a
        # terminal, so training progress stays invisible (and train.log stays empty) for many
        # minutes at a time even though the run is healthy.
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    if code != 0:
        raise SystemExit(f"training failed with exit code {code}; see {log_path}")
    write_provenance(
        output,
        resolved=resolved,
        spec=resolved.spec,
        extra={
            "stage": "train",
            "training_config": yaml.safe_load(config_path.read_text(encoding="utf-8")),
        },
    )
    print(
        f"Training finished in {(time.monotonic() - started) / 60:.1f} min; checkpoints in {adapters}"
    )


def checkpoint_dirs(config: dict[str, Any]) -> list[tuple[int, Path]]:
    """Materialise each saved checkpoint as a loadable adapter directory."""
    output: Path = config["output"]
    adapters = output / "adapters"
    adapter_config = adapters / "adapter_config.json"
    if not adapter_config.is_file():
        raise SystemExit(f"no adapters found in {adapters}; run the train stage first")
    found = []
    for weights in sorted(adapters.glob("*_adapters.safetensors")):
        step = int(weights.name.split("_")[0])
        target = output / "checkpoints" / f"step-{step}"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(adapter_config, target / "adapter_config.json")
        target_weights = target / "adapters.safetensors"
        if not target_weights.is_file() or _sha256(weights) != _sha256(target_weights):
            shutil.copy2(weights, target / "adapters.safetensors")
        found.append((step, target))
    return found


def _sha256(path: Path) -> str:
    """Return the digest used to reject a stale materialised checkpoint weight file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_VAL_LOSS = re.compile(r"Iter\s+(\d+):\s+Val loss\s+([0-9.eE+-]+)")


def _validation_losses(output: Path) -> dict[int, float]:
    """Read lightweight saved training-log loss values without loading any model artifact."""
    path = output / "train.log"
    if not path.is_file():
        return {}
    return {
        int(match.group(1)): float(match.group(2))
        for match in _VAL_LOSS.finditer(path.read_text(encoding="utf-8"))
    }


def _counts(summary: dict[str, Any], name: str, fallback: tuple[str, str]) -> tuple[int, int]:
    record = summary.get("rate_counts", {}).get(name, {})
    if isinstance(record, dict) and {"numerator", "denominator"} <= record.keys():
        return (int(record["numerator"]), int(record["denominator"]))
    return (int(summary[fallback[0]]), int(summary[fallback[1]]))


def _selection_components(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate screen cells from exact counts so ranking never uses rounded rates."""
    success = [0, 0]
    clean = [0, 0]
    valid = [0, 0]
    families: dict[str, list[int]] = {}
    for summary in summaries:
        for target, source, fallback in (
            (success, "success", ("successes", "tasks")),
            (clean, "clean", ("clean_successes", "tasks")),
            (valid, "valid_actions", ("valid_turns", "turns")),
        ):
            numerator, denominator = _counts(summary, source, fallback)
            target[0] += numerator
            target[1] += denominator
        for family, stats in summary.get("by_family", {}).items():
            bucket = families.setdefault(str(family), [0, 0])
            bucket[0] += int(stats["successes"])
            bucket[1] += int(stats["tasks"])
    family_rates = [numerator / denominator for numerator, denominator in families.values() if denominator]
    return {
        "tasks": success[1],
        "successes": success[0],
        "family_macro_success": sum(family_rates) / len(family_rates) if family_rates else 0.0,
        "micro_success": success[0] / success[1] if success[1] else 0.0,
        "clean_rate": clean[0] / clean[1] if clean[1] else 0.0,
        "valid_action_rate": valid[0] / valid[1] if valid[1] else 0.0,
        "by_family": {
            family: {"successes": values[0], "tasks": values[1]}
            for family, values in sorted(families.items())
        },
    }


def _selection_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, int]:
    """Rank behavior first, then lower loss and an earlier checkpoint deterministically."""
    components = row["components"]
    loss = row["val_loss"]
    return (
        float(components["family_macro_success"]),
        float(components["micro_success"]),
        float(components["clean_rate"]),
        float(components["valid_action_rate"]),
        -float("inf") if loss is None else -float(loss),
        -int(row["step"]),
    )


def stage_select(config: dict[str, Any], limit: int | None, quiet: bool) -> Path:
    output: Path = config["output"]
    select = config["select"]
    checkpoints = checkpoint_dirs(config)
    if not checkpoints:
        raise SystemExit(
            f"no checkpoint directories found in {output / 'adapters'}; run the train stage first"
        )
    screen = select["screen"]
    losses = _validation_losses(output)
    results = []
    for step, adapter in checkpoints:
        summaries = []
        for cell in screen:
            split = cell["split"]
            transcript_dir = output / "transcripts" / f"select-step-{step}-{split}"
            Transcript.start_run(transcript_dir)
            _log(f"select: screening step-{step} on {split}")
            summaries.append(
                run_evaluation(
                    model_name=config["model"],
                    adapter=adapter,
                    label=f"step-{step}-{split}",
                    split=split,
                    limit=limit,
                    difficulty=cell["difficulty"],
                    family_quotas=cell["per_family"],
                    output=output / "evals" / f"select-step-{step}-{split}.json",
                    transcript_dir=transcript_dir,
                    max_steps=config["eval"]["max_steps"],
                    max_tokens=config["eval"]["max_tokens"],
                    keep_last=config["keep_last"],
                    quiet=quiet,
                    seed=config["seed"],
                )
            )
        components = _selection_components(summaries)
        results.append(
            {
                "step": step,
                "adapter": str(adapter),
                "components": components,
                "wilson_95": {"success": wilson(components["successes"], components["tasks"])},
                "val_loss": losses.get(step),
            }
        )
    best = max(results, key=_selection_key)
    loss_rows = [row for row in results if row["val_loss"] is not None]
    loss_best = min(loss_rows, key=lambda row: (float(row["val_loss"]), int(row["step"]))) if loss_rows else None
    best_dir = output / "best-adapter"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.copytree(best["adapter"], best_dir)
    selection = {
        "selected_step": best["step"],
        "behavior_best_step": best["step"],
        "loss_best_step": None if loss_best is None else loss_best["step"],
        "disagreement": loss_best is not None and best["step"] != loss_best["step"],
        "model": config["model"],
        "data_seed": config["seed"],
        "screen": screen,
        "criterion": "family macro success, micro success, clean rate, valid actions, lower validation loss, earlier step",
        "checkpoints": results,
    }
    selection_path = output / "selection.json"
    selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    write_provenance(
        output,
        resolved=None,
        spec=load_model_spec(config["model"]),
        extra={
            "stage": "select",
            "selection": json.loads(selection_path.read_text(encoding="utf-8")),
        },
    )
    print("\nCheckpoint screen:")
    for row in results:
        mark = "<- selected" if row["step"] == best["step"] else ""
        components = row["components"]
        print(
            f"  step-{row['step']:<5} {components['successes']:2d}/{components['tasks']} success  clean {components['clean_rate']:.0%}  {mark}"
        )
    print(f"Best adapter copied to {best_dir}")
    return best_dir


def stage_eval(
    config: dict[str, Any],
    adapter: Path | None,
    *,
    base: bool,
    split: str | None,
    limit: int | None,
    stress: bool,
    quiet: bool,
) -> None:
    output: Path = config["output"]
    evaluation = config["eval"]
    split = split or evaluation["split"]
    limit = limit or evaluation["limit"]
    policies: list[tuple[str, Path | None]] = []
    if base:
        policies.append(("base", None))
    if adapter is not None:
        policies.append(
            (adapter.name if adapter.name != "best-adapter" else "best-adapter", adapter)
        )
    elif not base:
        best = output / "best-adapter"
        if not best.is_dir():
            raise SystemExit(f"{best} does not exist; run select or pass --adapter/--base")
        policies.append(("best-adapter", best))
    summaries = []
    for label, path in policies:
        stem = f"{label}-{split}{'-stress' if stress else ''}"
        transcript_dir = output / "transcripts" / stem
        Transcript.start_run(transcript_dir)
        _log(f"eval: {label} on {split} ({limit} tasks{', stress' if stress else ''})")
        summaries.append(
            run_evaluation(
                model_name=config["model"],
                adapter=path,
                label=label,
                split=split,
                limit=limit,
                output=output / "evals" / f"{stem}.json",
                transcript_dir=transcript_dir,
                stress=stress,
                max_steps=evaluation["max_steps"],
                max_tokens=evaluation["max_tokens"],
                keep_last=config["keep_last"],
                quiet=quiet,
                seed=config["seed"],
            )
        )
    write_provenance(
        output,
        resolved=None,
        spec=load_model_spec(config["model"]),
        extra={"stage": "eval", "evaluations": summaries},
    )


def stage_rollout(
    config: dict[str, Any],
    adapter: Path | None,
    split: str,
    limit: int | None,
    samples: int | None,
    quiet: bool,
) -> None:
    output: Path = config["output"]
    rollout = config["rollout"]
    adapter = adapter or output / "best-adapter"
    transcript_dir = output / "transcripts" / f"rollout-{split}"
    Transcript.start_run(transcript_dir)
    _log(f"rollout: sampling {adapter.name} on fresh split '{split}'")
    summary = run_rollout(
        model_name=config["model"],
        adapter=adapter,
        label=f"rollout-{split}",
        split=split,
        limit=limit or rollout["limit"],
        samples=samples or rollout["samples"],
        temperature=rollout["temperature"],
        output=PROJECT_ROOT / "data" / "rollouts" / split,
        transcript_dir=transcript_dir,
        keep_per_task=rollout.get("keep_per_task", 2),
        max_steps=config["eval"]["max_steps"],
        max_tokens=config["eval"]["max_tokens"],
        keep_last=config["keep_last"],
        quiet=quiet,
        seed=config["seed"],
    )
    write_provenance(
        output,
        resolved=None,
        spec=load_model_spec(config["model"]),
        extra={"stage": "rollout", "summary": summary},
    )


# --------------------------------------------------------------------------- entry point


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agent pipeline v2: data -> train -> select -> eval -> rollout -> branch -> prefer, with live transcripts.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print pass/fail lines only; transcripts still go to disk.",
    )
    stages = parser.add_subparsers(dest="stage", required=True)

    data = stages.add_parser(
        "data", help="Generate expert + recovery trajectories and mix retention data."
    )
    data.add_argument(
        "--extra",
        type=Path,
        action="append",
        default=[],
        help="Rollout directory whose train.jsonl is mixed in.",
    )

    train = stages.add_parser("train", help="QLoRA-train the base model on the generated data.")
    train.add_argument("--iters", type=int)
    train.add_argument(
        "--resume-from",
        type=Path,
        help="Saved adapter weights (*.safetensors) to continue training from.",
    )

    select = stages.add_parser(
        "select", help="Screen every checkpoint behaviourally on the validation split."
    )
    select.add_argument("--limit", type=int)

    evaluate = stages.add_parser(
        "eval", help="Evaluate on the held-out test split with transcripts."
    )
    evaluate.add_argument("--adapter", type=Path)
    evaluate.add_argument(
        "--base", action="store_true", help="Also (or only) evaluate the untouched base model."
    )
    evaluate.add_argument("--split")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--stress", action="store_true")

    rollout = stages.add_parser(
        "rollout", help="Sample the policy on fresh tasks and keep verified trajectories."
    )
    rollout.add_argument("--adapter", type=Path)
    rollout.add_argument("--split", default="iter1")
    rollout.add_argument("--limit", type=int)
    rollout.add_argument("--samples", type=int)

    branch = stages.add_parser(
        "branch", help="Mine step-level preference pairs by branching verified trajectories."
    )
    branch.add_argument("--adapter", type=Path)
    branch.add_argument("--split", default="pref1")
    branch.add_argument("--limit", type=int, default=48)
    branch.add_argument("--branches", type=int, default=3)

    prefer = stages.add_parser(
        "prefer",
        help="DPO-train the best adapter on mined preference pairs (frozen SFT reference).",
    )
    prefer.add_argument("--pairs", type=Path)
    prefer.add_argument("--adapter", type=Path)
    prefer.add_argument("--max-steps", type=int)
    prefer.add_argument("--beta", type=float)
    prefer.add_argument("--learning-rate", type=float)

    stages.add_parser("report", help="Tabulate every evaluation summary in outputs/<run>/evals.")

    everything = stages.add_parser(
        "all", help="data -> train -> select -> eval the best adapter (and optionally base)."
    )
    everything.add_argument("--iters", type=int)
    everything.add_argument("--base", action="store_true")
    everything.add_argument("--stress", action="store_true")
    everything.add_argument("--limit", type=int, default=180)

    args = parser.parse_args()
    config = load_config(args.config.resolve())
    if args.stage == "data":
        stage_data(config, [path.resolve() for path in args.extra])
    elif args.stage == "train":
        resume_from: Path | None = args.resume_from
        if resume_from is not None:
            if not resume_from.is_file():
                parser.error(f"--resume-from {resume_from} does not exist")
            if resume_from.suffix != ".safetensors":
                parser.error(f"--resume-from {resume_from} must be a .safetensors adapter file")
        stage_train(config, args.iters, resume_from=resume_from)
    elif args.stage == "select":
        stage_select(config, args.limit, args.quiet)
    elif args.stage == "eval":
        stage_eval(
            config,
            args.adapter,
            base=args.base,
            split=args.split,
            limit=args.limit,
            stress=args.stress,
            quiet=args.quiet,
        )
    elif args.stage == "rollout":
        if args.split in {"train", "valid", "test"}:
            parser.error("rollouts must use a fresh split name, e.g. iter1")
        stage_rollout(config, args.adapter, args.split, args.limit, args.samples, args.quiet)
    elif args.stage == "branch":
        if args.split in {"train", "valid", "test"}:
            parser.error("branch mining must use a fresh split name, e.g. pref1")
        transcript_dir = config["output"] / "transcripts" / f"branch-{args.split}"
        Transcript.start_run(transcript_dir)
        run_branch_mining(
            model_name=config["model"],
            adapter=args.adapter or config["output"] / "best-adapter",
            split=args.split,
            limit=args.limit,
            branches=args.branches,
            temperature=config.get("branch", {}).get("temperature", 0.9),
            output=PROJECT_ROOT / "data" / "preferences" / args.split,
            transcript_dir=transcript_dir,
            max_steps=config["eval"]["max_steps"],
            max_tokens=config["eval"]["max_tokens"],
            keep_last=config["keep_last"],
            quiet=args.quiet,
            seed=config["seed"],
        )
    elif args.stage == "prefer":
        prefer_cfg = config.get("prefer") or {}
        for flag in ("max_steps", "beta", "learning_rate"):
            value = getattr(args, flag)
            if value is not None and value <= 0:
                parser.error(f"--{flag.replace('_', '-')} must be positive")
        _log("prefer: DPO on mined pairs with the SFT adapter as frozen reference")
        run_prefer(
            model_name=config["model"],
            adapter=args.adapter or config["output"] / "best-adapter",
            pairs_path=args.pairs
            or PROJECT_ROOT / "data" / "preferences" / "pref1" / "pairs.jsonl",
            output=config["output"] / "prefer",
            beta=args.beta or prefer_cfg.get("beta", 0.1),
            learning_rate=args.learning_rate or prefer_cfg.get("learning_rate", 5e-6),
            max_steps=args.max_steps or prefer_cfg.get("max_steps", 200),
            batch_size=prefer_cfg.get("batch_size", 1),
            max_seq_length=prefer_cfg.get("max_seq_length", 3072),
            grad_accumulation=prefer_cfg.get("grad_accumulation", 4),
            max_chars=prefer_cfg.get("max_chars"),
            seed=config["seed"],
        )
    elif args.stage == "report":
        print(render(load_summaries(config["output"] / "evals")))
    elif args.stage == "all":
        stage_data(config, [])
        stage_train(config, args.iters)
        stage_select(config, None, args.quiet)
        stage_eval(
            config,
            config["output"] / "best-adapter",
            base=args.base,
            split=None,
            limit=args.limit,
            stress=args.stress,
            quiet=args.quiet,
        )
