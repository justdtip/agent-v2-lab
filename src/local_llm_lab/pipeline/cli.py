from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from local_llm_lab.pipeline.branch import run_branch_mining
from local_llm_lab.pipeline.data import write_dataset
from local_llm_lab.pipeline.evaluate import run_evaluation
from local_llm_lab.pipeline.prefer import run_prefer
from local_llm_lab.pipeline.report import load_summaries, render
from local_llm_lab.pipeline.rollout import run_rollout
from local_llm_lab.pipeline.transcript import Transcript
from local_llm_lab.project import PROJECT_ROOT, configure_local_cache

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "agent_v2.yaml"


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("output", "data", "chat_replay"):
        if key in config and config[key]:
            config[key] = (PROJECT_ROOT / config[key]).resolve()
    return config


def _log(message: str) -> None:
    print(f"\n### {time.strftime('%H:%M:%S')} {message}", flush=True)


# --------------------------------------------------------------------------- stages


def stage_data(config: dict[str, Any], extra: list[Path]) -> None:
    _log("data: generating expert trajectories with state-carrying notes")
    manifest = write_dataset(
        config["data"],
        config["tasks"],
        seed=config["seed"],
        keep_last=config["keep_last"],
        chat_dir=config.get("chat_replay"),
        chat_repeats=config.get("chat_repeats", 1),
        recovery_repeats=config.get("recovery_repeats", 1),
        extra_dirs=extra,
    )
    for split, info in manifest["splits"].items():
        print(
            f"{split:5s}: {info['tasks']:3d} tasks -> {info['rows']:4d} rows "
            f"({info['expert_rows']} expert, {info['chat_rows']} chat, {info['extra_rows']} extra); "
            f"horizon {info['min_horizon']}-{info['max_horizon']}; variants {info['variants']}"
        )
    print(f"Wrote {config['data']}")


LORA_KEYS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


def lora_config(
    config: dict[str, Any], iters: int | None = None, resume_from: Path | None = None
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
        "model": config["model"],
        "train": True,
        "fine_tune_type": "lora",
        "optimizer": "adamw",
        "data": str(config["data"]),
        "seed": config["seed"],
        "num_layers": train["num_layers"],
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
            "keys": list(LORA_KEYS),
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
    lora = lora_config(config, iters=iters, resume_from=resume_from)
    config_path = output / "lora.yaml"
    config_path.write_text(yaml.safe_dump(lora, sort_keys=False), encoding="utf-8")
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


def stage_select(config: dict[str, Any], limit: int | None, quiet: bool) -> Path:
    output: Path = config["output"]
    select = config["select"]
    limit = limit or select["limit"]
    results = []
    for step, adapter in checkpoint_dirs(config):
        transcript_dir = output / "transcripts" / f"select-step-{step}"
        Transcript.start_run(transcript_dir)
        _log(f"select: screening step-{step} on {select['split']} ({limit} tasks)")
        summary = run_evaluation(
            model_name=config["model"],
            adapter=adapter,
            label=f"step-{step}",
            split=select["split"],
            limit=limit,
            output=output / "evals" / f"select-step-{step}.json",
            transcript_dir=transcript_dir,
            max_steps=config["eval"]["max_steps"],
            max_tokens=config["eval"]["max_tokens"],
            keep_last=config["keep_last"],
            quiet=quiet,
            seed=config["seed"],
        )
        results.append(
            {
                "step": step,
                "adapter": str(adapter),
                **{
                    k: summary[k]
                    for k in (
                        "successes",
                        "tasks",
                        "success_rate",
                        "clean_rate",
                        "valid_action_rate",
                        "tool_errors",
                    )
                },
            }
        )
    if not results:
        raise SystemExit(
            f"no checkpoint directories found in {output / 'adapters'}; run the train stage first"
        )
    best = max(
        results,
        key=lambda r: (r["success_rate"], r["clean_rate"], r["valid_action_rate"], r["step"]),
    )
    best_dir = output / "best-adapter"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.copytree(best["adapter"], best_dir)
    selection = {
        "selected_step": best["step"],
        "criterion": "held-out success, then clean rate, valid actions, later step",
        "screen": results,
    }
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print("\nCheckpoint screen:")
    for row in results:
        mark = "<- selected" if row["step"] == best["step"] else ""
        print(
            f"  step-{row['step']:<5} {row['successes']:2d}/{row['tasks']} success  clean {row['clean_rate']:.0%}  {mark}"
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
    for label, path in policies:
        stem = f"{label}-{split}{'-stress' if stress else ''}"
        transcript_dir = output / "transcripts" / stem
        Transcript.start_run(transcript_dir)
        _log(f"eval: {label} on {split} ({limit} tasks{', stress' if stress else ''})")
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
    run_rollout(
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
        "all", help="data -> train -> select -> eval (base and best adapter)."
    )
    everything.add_argument("--iters", type=int)
    everything.add_argument("--limit", type=int)

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
            config, None, base=True, split=None, limit=args.limit, stress=False, quiet=args.quiet
        )
        stage_eval(
            config, None, base=False, split=None, limit=args.limit, stress=False, quiet=args.quiet
        )
