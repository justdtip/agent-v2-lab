from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import inspect
import json
import math
import shutil
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TextIO

import yaml

from local_llm_lab.models import LoraSpec, ModelSpec, ResolvedSpec, load_model_spec
from local_llm_lab.pipeline.branch import run_branch_mining
from local_llm_lab.pipeline.data import (
    SplitSpec,
    guard_dataset_write,
    render_dataset,
    write_dataset,
)
from local_llm_lab.pipeline.evaluate import run_evaluation, wilson
from local_llm_lab.pipeline.prefer import run_prefer
from local_llm_lab.pipeline.preflight import require_preflight, run_preflight
from local_llm_lab.pipeline.report import load_summaries, render
from local_llm_lab.pipeline.rollout import run_rollout
from local_llm_lab.pipeline.tasks import GENERATOR_VERSION
from local_llm_lab.pipeline.transcript import Transcript
from local_llm_lab.project import PROJECT_ROOT, configure_local_cache
from local_llm_lab.provenance import write_provenance
from local_llm_lab.runlock import load_weights
from local_llm_lab.runlog import (
    HealthThresholds,
    RunLog,
    Tee,
    TrainingAborted,
    TrainingHealth,
    git_commit,
    sha256_of,
    write_text_atomic,
)
from local_llm_lab.tuner_data import load_rendered_splits

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "agent_v2.yaml"
_PINNED_MLX_LM_VERSION = "0.31.3"
# R32: the two approved forms of the training-time gated-delta recurrence, and the installer
# each one enters. "checkpointed" is stage 1 (bit-exact against mlx-lm's loop) and stays the
# default, so an arm that names no mode trains exactly as it did before stage 2 existed.
GATED_DELTA_CHECKPOINTED = "checkpointed"
GATED_DELTA_CHUNKWISE = "chunkwise"
_GATED_DELTA_INSTALLERS = {
    GATED_DELTA_CHECKPOINTED: "install_chunked_gated_delta",
    GATED_DELTA_CHUNKWISE: "install_chunkwise_gated_delta",
}
# K6(b): what the run record says about the fallback counters on an arm that never installed
# the chunkwise recurrence. Only the chunkwise installer resets that counter, so reading it on
# any other arm returns whatever an earlier chunkwise run in the same process left behind — a
# number that looks like evidence and is not. These strings cannot be mistaken for a count.
GATED_DELTA_NO_RECURRENCE = "not applicable: no recurrence installed"
_TRAIN_MODEL_PARAMETERS = ("args", "model", "train_set", "valid_set", "training_callback")


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("output", "data", "chat_replay", "source_rows"):
        if key in config and config[key]:
            config[key] = (PROJECT_ROOT / config[key]).resolve()
    return config


def _log(message: str) -> None:
    print(f"\n### {time.strftime('%H:%M:%S')} {message}", flush=True)


def _require_config_preflight(config: dict[str, Any], *, skip: bool) -> None:
    """Validate the declared base model before a stage can load it."""
    require_preflight(load_model_spec(config["model"]), skip=skip)


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


def stage_data(config: dict[str, Any], extra: list[Path], *, overwrite: bool = False) -> None:
    if config.get("source_rows"):
        # Ruling R21(c): a config with source_rows names a cross-model arm whose dataset is
        # the source rows re-rendered, never regenerated; the data stage delegates to render.
        if extra:
            raise SystemExit(
                "source_rows configs re-render existing rows; --extra mixing does not apply"
            )
        stage_render(
            source=config["source_rows"],
            output=config["data"],
            model=config["model"],
            overwrite=overwrite,
        )
        return
    _log("data: generating expert trajectories with state-carrying notes")
    spec = load_model_spec(config["model"])
    tokenizer = _load_data_tokenizer(spec.hf_id)
    manifest = write_dataset(
        config["data"],
        dataset_splits(config),
        tokenizer=tokenizer,
        spec=spec,
        seed=config["seed"],
        keep_last=config["keep_last"],
        chat_dir=config.get("chat_replay"),
        chat_repeats=config.get("chat_repeats", 1),
        recovery_repeats=config.get("recovery_repeats", 1),
        extra_dirs=extra,
        overwrite=overwrite,
    )
    write_provenance(
        config["output"],
        resolved=None,
        spec=spec,
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


def stage_render(*, source: Path, output: Path, model: str, overwrite: bool = False) -> None:
    """Re-render existing rows for the named model's template; never regenerates tasks (R21).

    Loading the registered tokenizer here is not model execution (issue #12): no weights are
    touched, and the guarded ``render_dataset`` write carries the source rows through verbatim.
    """
    _log(f"render: applying the {model} template to existing rows from {source}")
    spec = load_model_spec(model)
    tokenizer = _load_data_tokenizer(spec.hf_id)
    manifest = render_dataset(source, output, tokenizer, spec=spec, overwrite=overwrite)
    write_provenance(
        output,
        resolved=None,
        spec=spec,
        extra={"stage": "render", "dataset_manifest": manifest},
    )
    for role, info in manifest["outputs"].items():
        print(f"{role:5s}: {info['rows']:4d} rows re-rendered")
    print(f"Wrote {output} (source rows carried verbatim from {manifest['source']['directory']})")


def _write_stage_manifest(target: Path, payload: dict[str, Any]) -> None:
    """Stamp a stage's data output with manifest.json after it succeeds.

    The R21 guard fires only on this file, so without it the rollout/branch guard calls
    would be inert; the stamp also gives those outputs the provenance they were missing.
    """
    target.mkdir(parents=True, exist_ok=True)
    (target / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _load_training_base(hf_id: str) -> tuple[Any, Any]:
    """Lazily load the registry's base model only while resolving training targets.

    The second of the two doors on the v2 surface (``evaluate.load_policy`` is the other), and
    the one the training stage goes through, so the model-run lock is taken here too: the
    trainer holds these weights for the whole 70-minute run (issue 83).
    """
    configure_local_cache()
    return load_weights(hf_id)


def _import_pinned_mlx_lm() -> Any:
    package = importlib.import_module("mlx_lm")
    installed = getattr(package, "__version__", None)
    if installed != _PINNED_MLX_LM_VERSION:
        raise SystemExit(f"mlx-lm {_PINNED_MLX_LM_VERSION} required; found {installed}")
    return package


def _load_data_tokenizer(hf_id: str) -> Any:
    configure_local_cache()
    _import_pinned_mlx_lm()
    try:
        utils = importlib.import_module("mlx_lm.utils")
    except ImportError as error:
        raise SystemExit(f"cannot import pinned mlx_lm.utils: {error}") from error
    return utils.load_tokenizer(hf_id)


def _clear_model_cache() -> None:
    """Release the temporary base-model allocation after architecture resolution."""
    import mlx.core as mx

    mx.clear_cache()


DEFAULT_METAL_CACHE_GIB = 2.0


def _limit_metal_cache(train_config: dict[str, Any]) -> int:
    """Cap the Metal allocator's cache before training and return the limit in bytes.

    Without a cap the allocator keeps every freed temporary, so a step's transient peaks (the
    full-vocabulary logits above all) accumulate in the cache and the working set climbs
    until Metal refuses a command buffer with "Insufficient Memory". The training-cost probe
    of 2026-09-05 ran this trainer's own step at 4,096 tokens under a 2 GiB cache limit and
    fitted; the pipeline set no limit and died at the first optimizer step three times on
    2026-09-06 at caps of 4,096, 3,072 and 2,688 tokens (heartbeat 13:20 to 14:05). The
    limit is ``train.metal_cache_gib`` in the config, default 2; recorded in the run log.
    """
    import mlx.core as mx

    gib = float(train_config.get("metal_cache_gib", DEFAULT_METAL_CACHE_GIB))
    if gib <= 0:
        raise ValueError("train.metal_cache_gib must be positive")
    limit = int(gib * 2**30)
    mx.set_cache_limit(limit)
    return limit


def _effective_training_spec(config: dict[str, Any]) -> ModelSpec:
    spec = load_model_spec(config["model"])
    train = config["train"]
    keys = "auto" if "lora_keys" not in train else tuple(train["lora_keys"])
    return replace(
        spec,
        lora=LoraSpec(
            keys=keys,
            rank=train["rank"],
            scale=train["scale"],
            dropout=train.get("dropout", 0.0),
        ),
    )


def _load_training_entry() -> tuple[Any, dict[str, Any]]:
    try:
        _import_pinned_mlx_lm()
        module = importlib.import_module("mlx_lm.lora")
    except ImportError as error:
        raise SystemExit(f"cannot import pinned mlx_lm.lora: {error}") from error
    trainer = module.train_model
    parameters = tuple(inspect.signature(trainer).parameters.values())
    actual = tuple(parameter.name for parameter in parameters)
    expected_kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD,) * len(_TRAIN_MODEL_PARAMETERS)
    valid_defaults = len(parameters) == len(_TRAIN_MODEL_PARAMETERS) and all(
        parameter.default is inspect.Parameter.empty for parameter in parameters[:-1]
    ) and parameters[-1].default is None
    if (
        actual != _TRAIN_MODEL_PARAMETERS
        or tuple(parameter.kind for parameter in parameters) != expected_kinds
        or not valid_defaults
    ):
        raise SystemExit(
            f"mlx-lm train_model signature mismatch: expected {_TRAIN_MODEL_PARAMETERS} "
            f"with training_callback=None; found {actual}"
        )
    defaults = module.CONFIG_DEFAULTS
    if not isinstance(defaults, dict):
        raise SystemExit("mlx_lm.lora.CONFIG_DEFAULTS must be a mapping")
    return trainer, dict(defaults)


def _effective_lora_args(lora: dict[str, Any], defaults: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**{**defaults, **lora})


def _flag_detail(flag: dict[str, Any]) -> dict[str, Any]:
    """A flag's detail as log fields, minus the two names the log line already carries."""
    detail = flag.get("detail") or {}
    return {key: value for key, value in detail.items() if key not in {"flag", "iteration"}}


def _health_flags(log: RunLog, health: TrainingHealth, names: list[str], iteration: int) -> None:
    """Put every newly raised health flag on stderr with the detail that raised it (R26)."""
    if not names:
        return
    latest = {flag["flag"]: flag for flag in health.flags}
    for name in names:
        log.warn("health flag", flag=name, iteration=iteration, **_flag_detail(latest.get(name, {})))


def _finish_health(
    log: RunLog,
    health: TrainingHealth,
    callback: _TrainingMetrics | None,
    *,
    adapters: Path,
    since: float,
) -> None:
    """Close the run's health record, whichever way the training stage is leaving (R26(c)).

    The rule is the same on every exit path, so the record is the same whether the trainer
    returned, the trainer raised, the loader raised, or the stage returned early: iterations
    done comes from the last report the callback saw (none at all is zero) and the final
    checkpoint from the fresh-checkpoint check. ``TrainingHealth.on_finish`` is idempotent,
    so the normal path's call and the ``finally``'s fallback record one flag between them,
    and a run that crashed before iteration one can never read ``healthy``.
    """
    iteration = callback.last_iteration if callback is not None else 0
    _health_flags(
        log,
        health,
        health.on_finish(
            iterations_done=iteration,
            final_checkpoint=_fresh_checkpoint(adapters / "adapters.safetensors", since=since),
        ),
        iteration,
    )


class _TrainingMetrics:
    """Write ``metrics.jsonl`` and mirror every trainer report to the run log and health rules.

    ``_validation_losses`` reads this file back for checkpoint selection, so its keys, their
    order, and the ``iteration + 1`` step convention for validation records are frozen; the
    run log and the health state machine are additive and never change what is written here.
    """

    def __init__(
        self,
        target: TextIO,
        started: float,
        *,
        log: RunLog,
        health: TrainingHealth,
        iters: int,
        budget_gib: float | None,
    ) -> None:
        self.target = target
        self.started = started
        self.log = log
        self.health = health
        self.iters = iters
        self.budget_gib = budget_gib
        self.last_iteration = 0
        self.best: float | None = None
        self.best_iteration: int | None = None

    def _write(self, *, step: int, train_loss: Any, val_loss: Any, tokens: Any) -> None:
        self.target.write(
            json.dumps(
                {
                    "step": step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "tokens": tokens,
                    "elapsed": time.monotonic() - self.started,
                }
            )
            + "\n"
        )
        self.target.flush()

    def on_val_loss_report(self, metrics: dict[str, Any]) -> None:
        self._write(step=int(metrics["iteration"]) + 1, train_loss=None, val_loss=metrics["val_loss"], tokens=None)
        iteration = int(metrics["iteration"])
        val_loss = metrics["val_loss"]
        _health_flags(
            self.log,
            self.health,
            self.health.on_val_report(iteration=iteration, val_loss=val_loss),
            iteration,
        )
        if self.best is None or val_loss < self.best:
            self.best, self.best_iteration = val_loss, iteration
        self.log.metric(
            "val",
            iteration=iteration,
            val_loss=val_loss,
            best=self.best,
            best_iteration=self.best_iteration,
        )

    def on_train_loss_report(self, metrics: dict[str, Any]) -> None:
        self._write(step=int(metrics["iteration"]), train_loss=metrics["train_loss"], val_loss=None, tokens=metrics["trained_tokens"])
        iteration = int(metrics["iteration"])
        self.last_iteration = iteration
        # mlx-lm 0.31.3 always sends the five metric fields; the defaults keep a partial
        # report (a fake, or a future trainer) from turning logging into the failure.
        _health_flags(
            self.log,
            self.health,
            self.health.on_train_report(
                iteration=iteration,
                train_loss=metrics["train_loss"],
                learning_rate=metrics.get("learning_rate", 0.0),
                tokens_per_second=metrics.get("tokens_per_second", 0.0),
                trained_tokens=metrics.get("trained_tokens", 0),
                peak_memory_gb=metrics.get("peak_memory", 0.0),
            ),
            iteration,
        )
        self.log.progress(
            iteration,
            self.iters,
            "train",
            loss=metrics["train_loss"],
            lr=metrics.get("learning_rate"),
            tok_s=metrics.get("tokens_per_second"),
            tokens=metrics["trained_tokens"],
            peak_gb=metrics.get("peak_memory"),
            budget_gib=self.budget_gib,
        )


def _resolve_training_spec(config: dict[str, Any]) -> ResolvedSpec:
    """Resolve LoRA targets from the declared base architecture and training overrides."""
    effective = _effective_training_spec(config)
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
    # R32(c): the chunk length of the training-time gated-delta recurrence is an arm-config
    # field, recorded here so lora.yaml and provenance carry the number the run actually used.
    # Absent on a dense backbone, which has no recurrence to chunk.
    if "gated_delta_chunk" in train:
        lora["gated_delta_chunk"] = int(train["gated_delta_chunk"])
        # R32 stage 2: which form of the recurrence the chunk length belongs to. Recorded
        # beside it because the two have different step time and different numerics — stage 1
        # is bit-exact against the library's loop, stage 2 agrees to a tolerance.
        lora["gated_delta_mode"] = str(train.get("gated_delta_mode", GATED_DELTA_CHECKPOINTED))
    if resume_from is not None:
        lora["resume_adapter_file"] = str(Path(resume_from).resolve())
    return lora


@contextlib.contextmanager
def _kernel_path_validation() -> Iterator[None]:
    """Run the trainer's in-loop validation with the model in evaluation mode (R32, item 1).

    A hybrid backbone dispatches on ``self.training``: ``GatedDeltaNet.__call__`` asks for
    ``use_kernel=not self.training`` (``mlx_lm/models/qwen3_5.py:192``), so training mode
    takes the differentiable Python recurrence and evaluation mode the exact Metal kernel.
    Validation needs no gradient, and the two agree in the forward, so the pass belongs on
    the kernel and the loss it reports is unchanged.

    mlx-lm 0.31.3 already calls ``model.eval()`` on entry to ``evaluate``
    (``mlx_lm/tuner/trainer.py:186``) and ``model.train()`` after it returns
    (``trainer.py:299``). This wrapper makes the guarantee ours rather than a property of the
    pinned release, and restores the mode the model was actually in even when ``evaluate``
    raises, which the library's own path does not.
    """
    trainer = importlib.import_module("mlx_lm.tuner.trainer")
    original = trainer.evaluate

    def evaluate(*args: Any, **kwargs: Any) -> Any:
        # ``train`` calls it by keyword (``trainer.py:290``); positional is accepted too.
        model = kwargs["model"] if "model" in kwargs else args[0]
        training_mode = bool(getattr(model, "training", False))
        model.eval()
        try:
            return original(*args, **kwargs)
        finally:
            model.train(training_mode)

    trainer.evaluate = evaluate
    try:
        yield
    finally:
        trainer.evaluate = original


@dataclass
class _BackboneReport:
    """What the backbone block installed, and what that recurrence actually did (K6(b)).

    ``fallbacks`` is written when the block exits: the chunkwise recurrence's own counts on a
    chunkwise arm, and one of the not-applicable strings on any other. The block hands this
    out rather than letting the caller read ``training.fallback_counts()`` itself, because
    only the block knows which installer it entered, and that global is meaningless — stale,
    not empty — on an arm whose installer never resets it.
    """

    fallbacks: dict[str, int] | str


@contextlib.contextmanager
def _training_backbone(
    chunk: int | None, mode: str = GATED_DELTA_CHECKPOINTED
) -> Iterator[_BackboneReport]:
    """Library patches that live exactly as long as the trainer call (R32(a), stage 2).

    ``chunk`` installs a gated-delta recurrence for training-mode calls; ``None`` — a dense
    backbone, with no recurrence to chunk — installs nothing. ``mode`` chooses the form:

    * ``"checkpointed"`` (R32(a) stage 1, the default so an arm that names no mode is
      unchanged): the library's own per-token loop, run in chunks under ``mx.checkpoint``, so
      the autograd graph retains chunk-boundary states instead of one per token. Bit-exact.
    * ``"chunkwise"`` (R32 stage 2): the chunkwise-parallel form, matmul-bound inside a chunk
      with states passed between chunks. Exact in exact arithmetic, so it agrees with the
      library's loop to a tolerance rather than bit for bit.

    The validation wrapper is unconditional: on a backbone whose forward does not branch on the
    training flag it changes nothing.

    Yields a ``_BackboneReport`` whose ``fallbacks`` field the block fills in on the way out,
    so the caller records the recurrence the run TOOK and not merely the one it configured.
    """
    if mode not in _GATED_DELTA_INSTALLERS:
        raise ValueError(
            f"train.gated_delta_mode must be one of "
            f"{sorted(_GATED_DELTA_INSTALLERS)}; got {mode!r}"
        )
    report = _BackboneReport(fallbacks=GATED_DELTA_NO_RECURRENCE)
    # Bound only on the chunkwise branch, and the only route to the counter: on every other
    # arm there is nothing to call, so a stale global cannot reach the record (K6(b)).
    counts: Callable[[], dict[str, int]] | None = None
    with contextlib.ExitStack() as stack:
        if chunk is not None:
            from local_llm_lab import training

            name = _GATED_DELTA_INSTALLERS[mode]
            # A named mode whose installer is not in the tree: R32 stage 2 is a separate slice,
            # so ``chunkwise`` reaches here before it lands. Say which function is missing and
            # where it comes from, rather than letting a bare AttributeError surface at train
            # start. Falling back to another mode's installer is not an option — an arm that
            # asked for the chunkwise form may not silently train under the chunked one.
            installer = getattr(training, name, None)
            if installer is None:
                raise ValueError(
                    f"train.gated_delta_mode={mode!r} needs local_llm_lab.training.{name}, "
                    f"which this tree does not provide; it lands with R32 stage 2"
                )
            stack.enter_context(installer(int(chunk)))
            # An installed recurrence that is not the chunkwise one has no counter of its own;
            # say which form ran rather than leaving the field looking like an empty count.
            report.fallbacks = f"not applicable: {mode}"
            if mode == GATED_DELTA_CHUNKWISE:
                counts = training.fallback_counts
        stack.enter_context(_kernel_path_validation())
        try:
            yield report
        finally:
            # Read while the installer is still in place, so the counts are the ones this
            # block accumulated and nothing between here and the caller can reset them. The
            # exception path runs it too: a run that died mid-training is the one whose next
            # attempt most needs to know which recurrence it was actually running.
            if counts is not None:
                report.fallbacks = counts()


def _fresh_checkpoint(path: Path, *, since: float) -> bool:
    """True when ``path`` exists and was written by THIS run.

    A stale ``adapters.safetensors`` left in the same output directory by an earlier run must
    not count as this run's final checkpoint (Chief's condition on issue #37; the same class of
    defect SPEC-002 closed in checkpoint selection). ``since`` is wall-clock seconds at stage
    start, floored to the second so a coarse-resolution filesystem cannot truncate a fresh
    file's mtime to just before it.
    """
    if not path.is_file():
        return False
    return path.stat().st_mtime >= math.floor(since)


def _split_size(split: Any) -> int | None:
    """Row count when the loaded split exposes one; ``None`` for anything unmeasurable."""
    try:
        return len(split)
    except TypeError:
        return None


def _training_identity(config: dict[str, Any], spec: ModelSpec) -> dict[str, Any]:
    """R26(e): what the start event names so a lift request is checkable from the file alone."""
    manifest = Path(config["data"]) / "manifest.json"
    return {
        "model": spec.name,
        "hf_id": spec.hf_id,
        "config": config.get("config_path"),
        "data_manifest_sha256": sha256_of(manifest) if manifest.is_file() else None,
        "git_commit": git_commit(),
    }


def _recorded_fallbacks(backbone: _BackboneReport | None) -> dict[str, int] | str:
    """What the backbone block recorded; a run that never reached it installed no recurrence."""
    return GATED_DELTA_NO_RECURRENCE if backbone is None else backbone.fallbacks


def _health_record(
    health: TrainingHealth,
    *,
    elapsed: float,
    status: str,
    fallbacks: dict[str, int] | str,
) -> dict[str, Any]:
    """The health summary plus the recurrence the run ACTUALLY took (K6(a)).

    ``TrainingHealth`` watches losses and memory and knows nothing about the backbone, but
    ``health.json`` is the record the next attempt's memory gate reads. A run that configured
    the chunkwise form and fell back to stage 1's checkpointed loop carries stage 1's memory
    and step time, so the gate has to be able to read that here rather than infer it from the
    mode the arm asked for — which is the one thing the record already said.
    """
    return {
        **health.summary(elapsed=elapsed, status=status),
        "gated_delta_fallbacks": fallbacks,
    }


def _log_fallbacks(runlog: RunLog, fallbacks: dict[str, int] | str) -> None:
    """Put the recurrence the run took on the run log; a fallback that fired is a warning."""
    if isinstance(fallbacks, str):
        runlog.info("gated delta recurrence", fallbacks=fallbacks)
    elif not fallbacks:
        runlog.info("gated delta recurrence", fallbacks="none: chunkwise throughout")
    else:
        # One field per reason, named and counted rather than merely announced: which fallback
        # fired decides whether the cause is a mask reaching the recurrence or a gate shape the
        # chunkwise form cannot express, and the count decides whether it was every call or one
        # layer. Prefixed so a reason can never collide with a field this logger already has.
        runlog.warn(
            "gated delta fell back to the checkpointed loop",
            **{f"fallback_{reason}": count for reason, count in sorted(fallbacks.items())},
        )


def stage_train(config: dict[str, Any], iters: int | None, resume_from: Path | None = None) -> None:
    from local_llm_lab.pipeline.live_lens.preflight import training_preflight

    launch = training_preflight(config['train'], load_model_spec(config['model']), iters=iters)
    output: Path = config["output"]
    adapters = output / "adapters"
    checkpoints = output / "checkpoints"
    trainer, defaults = _load_training_entry()
    effective = _effective_training_spec(config)
    # Both are resolved before anything loads, so a typo under train.health costs no weights.
    # The planned iteration count is lora["iters"] by the same override lora_config applies.
    thresholds = HealthThresholds.from_config(
        config["train"].get("health"), memory_budget_gib=effective.memory_budget_gib
    )
    health = TrainingHealth(
        thresholds, iters=config["train"]["iters"] if iters is None else iters
    )
    model: Any | None = None
    tokenizer: Any | None = None
    train_set: Any | None = None
    valid_set: Any | None = None
    test_set: Any | None = None
    callback: _TrainingMetrics | None = None
    # Bound by the backbone block's ``as`` clause, so it survives an exception raised inside
    # it and the finally below can still record what the recurrence did (K6(a)).
    backbone: _BackboneReport | None = None
    aborted: TrainingAborted | None = None
    summary: dict[str, Any] | None = None
    status = "ok"
    started = time.monotonic()
    started_wall = time.time()
    with RunLog.open(
        output,
        name="train",
        command=list(sys.argv),
        identity=_training_identity(config, effective),
    ) as runlog:
        try:
            if launch is not None:
                runlog.info('training launch preflight', **launch)
            cache_limit = _limit_metal_cache(config["train"])
            runlog.info("metal cache limit", bytes=cache_limit, gib=round(cache_limit / 2**30, 2))
            runlog.info("loading base", model=effective.hf_id)
            model, tokenizer = _load_training_base(effective.hf_id)
            resolved = effective.resolve(model, tokenizer)
            runlog.info(
                "resolved lora targets",
                count=len(resolved.lora_keys),
                keys=list(resolved.lora_keys),
            )
            lora = {
                **defaults,
                **lora_config(config, resolved, iters=iters, resume_from=resume_from),
            }
            args = _effective_lora_args(lora, {})
            adapters.mkdir(parents=True, exist_ok=True)
            config_path = output / "lora.yaml"
            config_path.write_text(yaml.safe_dump(lora, sort_keys=False), encoding="utf-8")
            # ``lora["test"]`` is False: this stage never trains on or evaluates the test
            # split, so it does not load one. Reading it costs a full tokenization pass and
            # fails closed on rows the trainer would never see — the B4 start failure, where
            # one over-long test row ended the run before iteration one. ``test_set`` stays
            # None, and the splits line below reports it as such.
            train_set, valid_set = load_rendered_splits(
                config["data"],
                tokenizer,
                max_seq_length=args.max_seq_length,
                splits=("train", "valid"),
            )
            runlog.info(
                "splits",
                train=_split_size(train_set),
                valid=_split_size(valid_set),
                test=_split_size(test_set),
            )
            if checkpoints.exists():
                shutil.rmtree(checkpoints)
            runlog.info(
                "trainer",
                entry="mlx_lm.lora.train_model",
                iters=lora["iters"],
                batch_size=lora["batch_size"],
                grad_accumulation_steps=lora["grad_accumulation_steps"],
                max_seq_length=lora["max_seq_length"],
                steps_per_report=lora["steps_per_report"],
                steps_per_eval=lora["steps_per_eval"],
                save_every=lora["save_every"],
                gated_delta_chunk=lora.get("gated_delta_chunk"),
                gated_delta_mode=lora.get("gated_delta_mode"),
            )
            _log("train: mlx_lm.lora.train_model")
            with (output / "train.log").open("w", encoding="utf-8") as log, (
                output / "metrics.jsonl"
            ).open("w", encoding="utf-8") as metrics:
                callback = _TrainingMetrics(
                    metrics,
                    started,
                    log=runlog,
                    health=health,
                    iters=lora["iters"],
                    budget_gib=thresholds.memory_budget_gib,
                )
                # The backbone patches wrap the trainer call and nothing else, so no other
                # stage — and no later inference — ever sees them (R32(a)).
                with contextlib.redirect_stdout(
                    _Tee(runlog.tee(sys.stdout), log)
                ), _training_backbone(
                    lora.get("gated_delta_chunk"),
                    mode=lora.get("gated_delta_mode", GATED_DELTA_CHECKPOINTED),
                ) as backbone:
                    trainer(args, model, train_set, valid_set, callback)
            # R26(c): a trainer that returned early or saved nothing produced an adapter that
            # nothing downstream may select from; on_finish records that and never raises.
            # Called here, before provenance, so the copy provenance carries is the final one.
            _finish_health(runlog, health, callback, adapters=adapters, since=started_wall)
            summary = _health_record(
                health,
                elapsed=time.monotonic() - started,
                status="ok",
                fallbacks=_recorded_fallbacks(backbone),
            )
            # R26(f): provenance carries the same health record health.json does, written
            # after on_finish so the verdict recorded there is the final one.
            write_provenance(
                output,
                resolved=resolved,
                spec=resolved.spec,
                extra={
                    "stage": "train",
                    "training_config": lora,
                    "health_thresholds": thresholds.as_dict(),
                    "health": summary,
                },
            )
            runlog.info(
                "training finished",
                minutes=round((time.monotonic() - started) / 60, 1),
                checkpoints=str(adapters),
                verdict=health.verdict,
            )
        except TrainingAborted as error:
            # R26(c): an aborted run's adapters are not eligible for selection, so the stage
            # records the health verdict and stops without writing provenance. A fatal flag
            # is raised rather than returned, so it is the one flag _health_flags never sees.
            status, aborted = "aborted", error
            runlog.error(
                "training aborted",
                flag=error.flag["flag"],
                iteration=error.flag["iteration"],
                **_flag_detail(error.flag),
            )
        except BaseException:
            status = "error"
            raise
        finally:
            # R26(c) on the error path: an exception before or during training leaves a run
            # that did fewer iterations than it planned, so the finish rule must run here too
            # or a crashed run's health.json reads ``healthy`` beside its ``error`` status.
            _finish_health(runlog, health, callback, adapters=adapters, since=started_wall)
            # K6(a): here rather than after the trainer call, so the line is emitted once on
            # every path. The run that died mid-training is the one whose next attempt most
            # needs to read which recurrence it was actually running, and that run never
            # reaches the success path at all. Outside the stdout redirect on every path too,
            # so it lands in run.log alone and not in the trainer's verbatim train.log.
            _log_fallbacks(runlog, _recorded_fallbacks(backbone))
            if summary is None or status != "ok":
                summary = _health_record(
                    health,
                    elapsed=time.monotonic() - started,
                    status=status,
                    fallbacks=_recorded_fallbacks(backbone),
                )
            (output / "health.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            runlog.close(status=status, verdict=health.verdict, flags=len(health.flags))
            del callback, train_set, valid_set, test_set, model, tokenizer
            _clear_model_cache()
        if aborted is not None:
            raise SystemExit(
                f"training aborted: {aborted.flag['flag']} at iteration "
                f"{aborted.flag['iteration']}; see {output / 'health.json'}"
            )


# RunLog.tee already mirrors the console side into run.log, so the training stage nests one
# Tee inside another; the private alias keeps the old name for anything that imports it.
_Tee = Tee


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


def _validation_losses(output: Path) -> dict[int, float]:
    """Read only typed validation records from the structured in-process metrics stream."""
    path = output / "metrics.jsonl"
    if not path.is_file():
        return {}
    losses = {}
    required = {"step", "train_loss", "val_loss", "tokens", "elapsed"}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
        if not isinstance(record, dict) or set(record) != required:
            raise ValueError(f"{path}:{line_number}: invalid metrics fields")
        if isinstance(record["step"], bool) or not isinstance(record["step"], int):
            raise ValueError(f"{path}:{line_number}: invalid step")
        for key in ("train_loss", "val_loss", "elapsed"):
            if record[key] is not None and (isinstance(record[key], bool) or not isinstance(record[key], (int, float))):
                raise ValueError(f"{path}:{line_number}: invalid {key}")
        if record["tokens"] is not None and (isinstance(record["tokens"], bool) or not isinstance(record["tokens"], int)):
            raise ValueError(f"{path}:{line_number}: invalid tokens")
        if record["val_loss"] is not None:
            losses[record["step"]] = float(record["val_loss"])
    return losses


def _counts(
    summary: dict[str, Any], name: str, fallback: tuple[str, str] | None
) -> tuple[int, int]:
    """Exact numerator and denominator for one rate, from ``rate_counts`` or a flat pair.

    ``fallback`` names the pre-``rate_counts`` keys carrying the same two numbers, for the
    evaluation JSONs written before that block existed. ``None`` means no such pair was ever
    written: the artifact must carry ``rate_counts`` or it is not one this can score.
    """
    record = summary.get("rate_counts", {}).get(name, {})
    if isinstance(record, dict) and {"numerator", "denominator"} <= record.keys():
        return (int(record["numerator"]), int(record["denominator"]))
    if fallback is None:
        raise ValueError(
            f"evaluation summary carries no rate_counts[{name!r}] and no flat equivalent "
            "was ever written; refusing to score a checkpoint on counts it does not have"
        )
    return (int(summary[fallback[0]]), int(summary[fallback[1]]))


def _selection_components(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate screen cells from exact counts so ranking never uses rounded rates.

    A summary with no ``by_family`` raises rather than scoring zero. ``family_macro_success``
    is the FIRST key :func:`_selection_key` compares on, so a block one level from where
    ``summarize`` writes it would give every checkpoint the same 0.0, drop the comparison
    through to micro success, and leave ``selection.json`` recording a macro-first
    ``criterion`` for a ranking that never used it. The recorded criterion and the comparison
    actually made must not be able to disagree.
    """
    success = [0, 0]
    clean = [0, 0]
    valid = [0, 0]
    families: dict[str, list[int]] = {}
    for summary in summaries:
        for target, source, fallback in (
            (success, "success", ("successes", "tasks")),
            (clean, "clean", ("clean_successes", "tasks")),
            # No flat fallback: ``summarize`` has never written ``valid_turns``/``turns`` at
            # the top level, and no evaluation JSON under outputs/ carries either, so the
            # pair named here before raised KeyError on exactly the legacy files it was for.
            (valid, "valid_actions", None),
        ):
            numerator, denominator = _counts(summary, source, fallback)
            target[0] += numerator
            target[1] += denominator
        if "by_family" not in summary:
            raise ValueError(
                "evaluation summary carries no by_family block; refusing to rank checkpoints "
                "on a family macro success of zero while recording a macro-first criterion"
            )
        for family, stats in summary["by_family"].items():
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
    spec = load_model_spec(config["model"])
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
                    spec=spec,
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
    # Atomic because this file is a stage sentinel like the dataset manifests: `best-adapter/`
    # has already been copied by the time it is written, so a truncated selection.json is the
    # one artifact saying WHICH checkpoint that copy is, left unparseable next to a directory
    # that looks finished. Whole or absent, never half. This module's own _write_stage_manifest
    # and health.json writes are still plain write_text; see DEBT(R21) in pipeline/data.py.
    write_text_atomic(selection_path, json.dumps(selection, indent=2) + "\n")
    write_provenance(
        output,
        resolved=None,
        spec=spec,
        extra={
            "stage": "select",
            # Read back from disk rather than embedded from `selection`: provenance records
            # what was PERSISTED, not what was intended, so the block and the file cannot
            # disagree. It is not redundant and must not be simplified to `selection` -- if
            # the write above ever failed to land what it was given, this raises here instead
            # of stamping a provenance that quietly outranks the file it describes.
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
    spec = load_model_spec(config["model"])
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
                spec=spec,
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
        spec=spec,
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
    rollout_output = PROJECT_ROOT / "data" / "rollouts" / split
    guard_dataset_write(rollout_output, override_flag=None)
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
        output=rollout_output,
        transcript_dir=transcript_dir,
        keep_per_task=rollout.get("keep_per_task", 2),
        max_steps=config["eval"]["max_steps"],
        max_tokens=config["eval"]["max_tokens"],
        keep_last=config["keep_last"],
        quiet=quiet,
        seed=config["seed"],
    )
    _write_stage_manifest(
        rollout_output,
        {
            "stage": "rollout",
            "generator_version": GENERATOR_VERSION,
            "model": config["model"],
            "split": split,
            "seed": config["seed"],
            "adapter": str(adapter),
            "summary": summary,
        },
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
    data.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Replace an existing non-protected dataset; recorded in the new manifest.",
    )

    render = stages.add_parser(
        "render",
        help="Re-render existing dataset rows for a model's template; never regenerates tasks.",
    )
    render.add_argument("--source", type=Path, help="Dataset directory holding the existing rows.")
    render.add_argument("--output", type=Path, help="New dataset directory for the rendered rows.")
    render.add_argument("--model", help="Registry model whose template renders the rows.")
    render.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Replace an existing non-protected dataset; recorded in the new manifest.",
    )

    train = stages.add_parser("train", help="QLoRA-train the base model on the generated data.")
    train.add_argument("--iters", type=int)
    train.add_argument("--skip-preflight-check", action="store_true")
    train.add_argument(
        "--resume-from",
        type=Path,
        help="Saved adapter weights (*.safetensors) to continue training from.",
    )

    select = stages.add_parser(
        "select", help="Screen every checkpoint behaviourally on the validation split."
    )
    select.add_argument("--limit", type=int)
    select.add_argument("--skip-preflight-check", action="store_true")

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
    evaluate.add_argument("--skip-preflight-check", action="store_true")

    preflight = stages.add_parser("preflight", help="Inspect a registered base model before loading stages.")
    preflight.add_argument("--model", required=True)
    # R32(d): without a row count the training-footprint gate records why it was skipped, so
    # `preflight --model <name>` keeps working and an arm's gate is one flag away.
    preflight.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Dataset directory whose longest trained row sizes the training footprint.",
    )
    preflight.add_argument(
        "--max-row-tokens",
        type=int,
        default=None,
        help="Longest training row in tokens, when the dataset is not on disk to be read.",
    )
    preflight.add_argument(
        "--gated-delta-chunk",
        type=int,
        default=None,
        help="The arm's train.gated_delta_chunk; without it the library's unrolled loop is estimated.",
    )
    # Issue #51 lane 2: the estimate is calibrated per recurrence form, and which form runs is
    # a configuration choice, so the gate has to be told the same thing the trainer is told.
    preflight.add_argument(
        "--gated-delta-mode",
        default=None,
        help="The arm's train.gated_delta_mode; the calibrated estimate for that form is what gates.",
    )

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
    everything.add_argument("--skip-preflight-check", action="store_true")

    args = parser.parse_args()
    if args.stage == "preflight":
        run_preflight(
            args.model,
            data_dir=args.data,
            max_row_tokens=args.max_row_tokens,
            gated_delta_chunk=args.gated_delta_chunk,
            gated_delta_mode=args.gated_delta_mode,
        )
        return
    if args.stage == "render" and args.source and args.output and args.model:
        # R21(b) form: render --source/--output/--model needs no run configuration.
        stage_render(
            source=args.source.resolve(),
            output=args.output.resolve(),
            model=args.model,
            overwrite=args.force_overwrite,
        )
        return
    run_config_path = args.config.resolve()
    config = load_config(run_config_path)
    # R26(e): the run log's start event names the recipe file this run came from. Recorded
    # here rather than in load_config so reading a config for comparison stays value-only.
    config["config_path"] = str(run_config_path)
    if args.stage == "render":
        source = args.source.resolve() if args.source else config.get("source_rows")
        if source is None:
            parser.error("render needs --source/--output/--model or a config with source_rows")
        stage_render(
            source=source,
            output=args.output.resolve() if args.output else config["data"],
            model=args.model or config["model"],
            overwrite=args.force_overwrite,
        )
    elif args.stage == "data":
        stage_data(
            config,
            [path.resolve() for path in args.extra],
            overwrite=args.force_overwrite,
        )
    elif args.stage == "train":
        _require_config_preflight(config, skip=args.skip_preflight_check)
        resume_from: Path | None = args.resume_from
        if resume_from is not None:
            if not resume_from.is_file():
                parser.error(f"--resume-from {resume_from} does not exist")
            if resume_from.suffix != ".safetensors":
                parser.error(f"--resume-from {resume_from} must be a .safetensors adapter file")
        stage_train(config, args.iters, resume_from=resume_from)
    elif args.stage == "select":
        _require_config_preflight(config, skip=args.skip_preflight_check)
        stage_select(config, args.limit, args.quiet)
    elif args.stage == "eval":
        _require_config_preflight(config, skip=args.skip_preflight_check)
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
        branch_output = PROJECT_ROOT / "data" / "preferences" / args.split
        guard_dataset_write(branch_output, override_flag=None)
        transcript_dir = config["output"] / "transcripts" / f"branch-{args.split}"
        Transcript.start_run(transcript_dir)
        branch_adapter = args.adapter or config["output"] / "best-adapter"
        branch_summary = run_branch_mining(
            model_name=config["model"],
            adapter=branch_adapter,
            split=args.split,
            limit=args.limit,
            branches=args.branches,
            temperature=config.get("branch", {}).get("temperature", 0.9),
            output=branch_output,
            transcript_dir=transcript_dir,
            max_steps=config["eval"]["max_steps"],
            max_tokens=config["eval"]["max_tokens"],
            keep_last=config["keep_last"],
            quiet=args.quiet,
            seed=config["seed"],
        )
        _write_stage_manifest(
            branch_output,
            {
                "stage": "branch",
                "generator_version": GENERATOR_VERSION,
                "model": config["model"],
                "split": args.split,
                "seed": config["seed"],
                "adapter": str(branch_adapter),
                "summary": branch_summary,
            },
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
        _require_config_preflight(config, skip=args.skip_preflight_check)
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
