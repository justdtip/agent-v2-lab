"""Inspect registration, capture one training split, or freeze exact native-replay IDs.

Default inspection never loads a tokenizer or model, and never hashes weight files.
Capture requires --execute and a current window owned by this process's inherited token.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from local_llm_lab.forward import encode_prompt
from local_llm_lab.models import load_model_spec as model_spec
from local_llm_lab.pipeline.evaluate import run_evaluation
from local_llm_lab.pipeline.lens_fitting.runtime import (
    TOKENIZER_ASSET_PATTERNS,
    primary_worktree,
    snapshot_identity,
    tokenizer_asset_hashes,
)
from local_llm_lab.pipeline.lens_fitting.transcript import (
    TranscriptCapture,
    TranscriptWriter,
    build_transcript_corpus,
    checked_bytes,
    digest,
    file_record,
    read_transcript,
)
from local_llm_lab.pipeline.tasks import GENERATOR_VERSION, make_tasks, task_fingerprint
from local_llm_lab.runlock import (
    BOX_STATE_DIR_ENV,
    WINDOW_HOLDER_ENV,
    WINDOW_RELATIVE_PATH,
    read_window,
)
from local_llm_lab.spawn import run

FORMAT = "gemma-transcript-lens-registration-v1"
FIT_MODELS = ("gemma3-4b-bf16", "gemma3-4b")
EVALUATION = {
    "max_steps": 24,
    "max_tokens": 200,
    "temperature": 0.0,
    "keep_last": 2,
    "use_cache": False,
}


def plan_cohorts():
    return [
        {
            "split": split,
            "difficulty": difficulty,
            "count": 24,
            "seed": 20260902,
            "generator_version": GENERATOR_VERSION,
            "tasks": [
                {"task_id": task.task_id, "fingerprint": task_fingerprint(task)}
                for task in make_tasks(split, 24, 20260902, difficulty=difficulty)
            ],
        }
        for split, difficulty in zip(("train", "train1", "train2"), (0, 1, 2), strict=True)
    ]


def tokenizer_identity_from_directory(directory):
    """Hash tokenization assets only, excluding config.json and all checkpoint shards."""
    directory = Path(directory).resolve(strict=True)
    files = sorted(
        {p for pattern in TOKENIZER_ASSET_PATTERNS for p in directory.glob(pattern) if p.is_file()}
    )
    if not files:
        raise ValueError("no local tokenizer assets")
    config_path = directory / "tokenizer_config.json"
    config = json.loads(config_path.read_bytes()) if config_path.is_file() else {}
    template_path = directory / "chat_template.jinja"
    template = template_path.read_text() if template_path.is_file() else config.get("chat_template")
    if isinstance(template, list):
        template = {row["name"]: row["template"] for row in template}
    if not template:
        raise ValueError("local tokenizer chat template is required")
    return {
        "files": [file_record(path) for path in files],
        "assets": [
            {"name": p.relative_to(directory).as_posix(), "sha256": file_record(p)["sha256"]}
            for p in files
        ],
        "chat_template_sha256": digest(template),
    }


def verify_rendering_gate(gate):
    root = primary_worktree()
    commits = {}
    for key, prefix in (("landed_commit", "7d2a18a"), ("ruling_commit", "a1ff0b0")):
        supplied = gate.get(key, "")
        if not isinstance(supplied, str) or not supplied.startswith(prefix):
            raise ValueError("registration must bind the landed rendering fix and amended ruling")
        result = run(
            ["git", "-C", str(root), "rev-parse", supplied + "^{commit}"],
            capture_output=True,
            text=True,
            check=True,
        )
        commits[key] = result.stdout.strip()
        result = run(
            ["git", "-C", str(root), "merge-base", "--is-ancestor", commits[key], "HEAD"],
            capture_output=True,
        )
        if result.returncode:
            raise ValueError("rendering gate has not landed in the primary worktree")
    records = gate.get("files", [])
    if not records:
        raise ValueError("rendering gate requires committed confirming evidence hashes")
    suffixes = set()
    for record in records:
        if record.get("storage") != "git_blob":
            raise ValueError("rendering evidence must explicitly identify git_blob storage")
        path = Path(record["path"]).resolve()
        relative = path.relative_to(root).as_posix()
        suffixes.add(relative)
        commit = record.get("commit")
        if commit not in commits.values():
            raise ValueError("rendering evidence commit is not one of the pinned gate commits")
        blob = run(
            ["git", "-C", str(root), "show", f"{commit}:{relative}"],
            capture_output=True,
            check=True,
        ).stdout
        if hashlib.sha256(blob).hexdigest() != record["sha256"]:
            raise ValueError("rendering evidence committed blob hash mismatch")
        if relative.startswith(("src/", "configs/")):
            if path.read_bytes() != blob:
                raise ValueError("primary checkout differs from landed rendering evidence")
            running = Path(__file__).resolve().parents[1] / relative
            if running.read_bytes() != blob:
                raise ValueError("running checkout differs from landed rendering evidence")
    required = {
        "design_specifications/pending/CODEX-TASKS-2026-09-08.md",
        "research/records/GEMMA3-JSPACE-MAP-2026-09-08/DIAGNOSTIC-RERUN.md",
        "src/local_llm_lab/pipeline/protocol.py",
        "configs/models/gemma3-4b.yaml",
        "configs/models/gemma3-4b-bf16.yaml",
    }
    if not required <= suffixes:
        raise ValueError(
            "rendering gate omits the fix, model declarations, ruling, or confirming record"
        )


def read_registration(path):
    record = file_record(path)
    registration = json.loads(checked_bytes(record))
    if (
        registration.get("schema_version") != 1
        or registration.get("format") != FORMAT
        or registration.get("fitting_context_tokens") != 2048
        or type(registration.get("fitting_context_tokens")) is not int
        or registration.get("producer_model") != "gemma3-4b"
        or registration.get("fit_models") != list(FIT_MODELS)
        or registration.get("evaluation") != EVALUATION
    ):
        raise ValueError("unsupported transcript registration or evaluation settings")
    if registration.get("cohorts") != plan_cohorts():
        raise ValueError("registered cohort IDs/fingerprints differ from the fixed training cohort")
    ident = registration.get("model_identity")
    for name in FIT_MODELS:
        spec = model_spec(name)
        if (
            ident != {"base": spec.base, "training": None, "num_layers": 34}
            or spec.training is not None
        ):
            raise ValueError(
                "registration requires the untrained canonical Gemma lineage at depth34"
            )
        snapshot = registration["snapshots"][name]
        directory = Path(snapshot["snapshot_path"]).resolve(strict=True)
        if directory != Path(spec.hf_id).resolve() or snapshot["hf_id"] != spec.hf_id:
            raise ValueError("registered snapshot path differs from registered model artifact")
        config = json.loads((directory / "config.json").read_bytes())
        config = config.get("text_config", config)
        if config.get("num_hidden_layers") != ident["num_layers"]:
            raise ValueError("snapshot configuration depth differs from canonical identity")
        if tokenizer_asset_hashes(snapshot) != registration["tokenizer"]["assets"]:
            raise ValueError("precision snapshots do not declare identical tokenization assets")
        if not snapshot.get("snapshot_sha256") or not snapshot.get("files"):
            raise ValueError("registration requires a preexisting snapshot inventory")
    actual = tokenizer_identity_from_directory(registration["tokenizer_directory"])
    if actual != registration["tokenizer"]:
        raise ValueError("registered tokenizer metadata or exact asset bytes changed")
    limits = registration.get("capture_limits", {})
    prompt, forward = limits.get("max_prompt_tokens"), limits.get("max_forward_tokens")
    if (
        type(prompt) is not int
        or not 0 < prompt <= 65536 - 201
        or type(forward) is not int
        or forward != prompt + 201
    ):
        raise ValueError(
            "capture limits require positive prompt bound plus200emissions and1lookahead"
        )
    projection = registration.get("capture_projection", {})
    peak = projection.get("peak_gib")
    if (
        isinstance(peak, bool)
        or not isinstance(peak, (int, float))
        or not math.isfinite(peak)
        or peak <= 0
        or not isinstance(projection.get("basis"), str)
        or not projection["basis"].strip()
    ):
        raise ValueError("capture projection requires a finite positive peak and explicit basis")
    verify_rendering_gate(registration.get("rendering_gate", {}))
    return registration, record


def require_owned_window():
    """Read the primary window explicitly; never clear or inherit a foreign reservation."""
    root = primary_worktree()
    window = read_window(root / WINDOW_RELATIVE_PATH)
    token = os.environ.get(WINDOW_HOLDER_ENV)
    if (
        window is None
        or not token
        or window.nonce != token
        or window.holder_state != "running"
        or window.expected_end_epoch is None
        or window.expected_end_epoch <= time.time()
    ):
        raise ValueError("capture requires a current owned R61 window at the primary worktree")
    # Both the model-run lock and its loader must use this same primary authority.
    os.environ[BOX_STATE_DIR_ENV] = str(root)
    return root


def verify_snapshot(registration, name):
    spec = model_spec(name)
    expected = registration["snapshots"][name]
    observed = snapshot_identity(Path(expected["snapshot_path"]), hf_id=spec.hf_id)
    if (
        observed["snapshot_sha256"] != expected["snapshot_sha256"]
        or observed["files"] != expected["files"]
    ):
        raise ValueError("checkpoint snapshot differs from registered inventory")


class BoundedTranscriptCapture(TranscriptCapture):
    def __init__(self, emit, *, max_prompt_tokens, max_forward_tokens):
        super().__init__(emit)
        self.max_prompt_tokens, self.max_forward_tokens = max_prompt_tokens, max_forward_tokens

    @contextmanager
    def generation(self, model, tokenizer, prompt, *, turn_cache):
        if len(encode_prompt(tokenizer, prompt)) > self.max_prompt_tokens:
            raise ValueError("captured prompt exceeds the registered pre-forward memory bound")
        with super().generation(model, tokenizer, prompt, turn_cache=turn_cache) as proxy:
            self.ledger.max_tokens = self.max_forward_tokens
            yield proxy


def capture(registration_path, *, split, record, output, execute=False):
    if not execute:
        raise ValueError("capture requires explicit --execute; inspect does not launch work")
    registration, binding = read_registration(registration_path)
    cohort = next((c for c in registration["cohorts"] if c["split"] == split), None)
    if cohort is None:
        raise ValueError("capture split is not a registered training cohort")
    record, output = Path(record).absolute(), Path(output).absolute()
    if record == output or record.exists() or output.exists():
        raise FileExistsError("capture and evaluation outputs must be distinct new files")
    require_owned_window()
    verify_snapshot(registration, "gemma3-4b")  # permitted hashing begins only inside owned window
    require_owned_window()  # hashing cannot silently consume the launch window
    spec = replace(model_spec("gemma3-4b"), cache_strategy="none")
    provenance = {
        "registration": binding,
        "fitting_context_tokens": registration["fitting_context_tokens"],
        "model_identity": registration["model_identity"],
        "tokenizer": registration["tokenizer"],
        "producer_snapshot": registration["snapshots"]["gemma3-4b"],
        "cohort": cohort,
        "evaluation": registration["evaluation"],
        "evaluation_output": str(output),
        "capture_limits": registration["capture_limits"],
        "capture_projection": registration["capture_projection"],
        "rendering_gate": registration["rendering_gate"],
        "capture_script": file_record(__file__),
    }
    record.parent.mkdir(parents=True, exist_ok=True)
    with TranscriptWriter(record, provenance) as writer:
        sink = BoundedTranscriptCapture(writer, **registration["capture_limits"])
        run_evaluation(
            spec=spec,
            adapter=None,
            label="gemma3-4b-transcript-training",
            split=split,
            limit=cohort["count"],
            output=output,
            transcript_dir=None,
            stress=False,
            seed=cohort["seed"],
            difficulty=cohort["difficulty"],
            quiet=True,
            capture=sink,
            **registration["evaluation"],
        )
        payload = json.loads(output.read_bytes())
        if [t["task_id"] for t in payload["trajectories"]] != [
            t["task_id"] for t in cohort["tasks"]
        ]:
            raise ValueError("evaluation output cohort differs from registration")
    return {
        "status": "captured",
        "split": split,
        "record": file_record(record),
        "evaluation": file_record(output),
    }


def validate_captured_cohort(events, cohort):
    tasks = make_tasks(
        cohort["split"], cohort["count"], cohort["seed"], difficulty=cohort["difficulty"]
    )
    expected = {task.task_id: task for task in tasks}
    observed = []
    for event in events:
        if event["kind"] != "begin_turn":
            continue
        context = event["context"]
        task = expected.get(context["task_id"])
        if (
            task is None
            or len(context["messages"]) < 2
            or context["messages"][1] != {"role": "user", "content": task.prompt}
            or context.get("keep_last") != EVALUATION["keep_last"]
        ):
            raise ValueError("captured task content differs from registered task fingerprint")
        if context["step"] == 0:
            observed.append(task.task_id)
    if observed != list(expected):
        raise ValueError("capture omitted or changed registered task identities")


def load_local_tokenizer(directory):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(directory), local_files_only=True, trust_remote_code=False, use_fast=True
    )


def freeze(registration_path, *, captures, output):
    registration, binding = read_registration(registration_path)
    if len(captures) != 3 or len({str(Path(p).resolve()) for p in captures}) != 3:
        raise ValueError("freeze requires exactly three distinct registered split captures")
    ordered = {}
    for path in captures:
        events = read_transcript(path)
        provenance = events[0]["provenance"]
        if provenance.get("registration") != binding:
            raise ValueError("capture belongs to a different registration")
        cohort = provenance.get("cohort")
        if cohort not in registration["cohorts"] or cohort["split"] in ordered:
            raise ValueError("capture has duplicate or unregistered training cohort")
        validate_captured_cohort(events, cohort)
        if (
            provenance.get("fitting_context_tokens") != registration["fitting_context_tokens"]
            or provenance.get("producer_snapshot") != registration["snapshots"]["gemma3-4b"]
            or provenance.get("evaluation") != registration["evaluation"]
            or provenance.get("capture_limits") != registration["capture_limits"]
            or provenance.get("rendering_gate") != registration["rendering_gate"]
        ):
            raise ValueError("capture execution provenance differs from registration")
        ordered[cohort["split"]] = Path(path)
    tokenizer = load_local_tokenizer(registration["tokenizer_directory"])
    if (
        not getattr(tokenizer, "is_fast", False)
        or digest(tokenizer.chat_template) != registration["tokenizer"]["chat_template_sha256"]
    ):
        raise ValueError("local fast tokenizer template differs from frozen metadata")
    manifest = build_transcript_corpus(
        [ordered[c["split"]] for c in registration["cohorts"]],
        tokenizer,
        model_spec("gemma3-4b"),
        Path(output),
        tokenizer_identity=registration["tokenizer"],
        model_identity=registration["model_identity"],
        max_tokens=registration["fitting_context_tokens"],
    )
    return {
        "status": manifest["acceptance"]["status"],
        "manifest": str(Path(output).absolute()),
        "manifest_sha256": manifest["manifest_sha256"],
        "acceptance": manifest["acceptance"],
    }


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        argv.insert(0, "inspect")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--registration", type=Path, required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--registration", type=Path, required=True)
    cap.add_argument("--split", choices=("train", "train1", "train2"), required=True)
    cap.add_argument("--record", type=Path, required=True)
    cap.add_argument("--output", type=Path, required=True)
    cap.add_argument("--execute", action="store_true")
    frozen = sub.add_parser("freeze")
    frozen.add_argument("--registration", type=Path, required=True)
    frozen.add_argument("--capture", type=Path, action="append", required=True)
    frozen.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            registration, binding = read_registration(args.registration)
            result = {
                "status": "inspected",
                "registration": binding,
                "cohorts": registration["cohorts"],
                "capture_projection": registration["capture_projection"],
                "capture_limits": registration["capture_limits"],
                "model_runtime_loaded": False,
                "weight_files_hashed": False,
            }
        elif args.command == "capture":
            result = capture(
                args.registration,
                split=args.split,
                record=args.record,
                output=args.output,
                execute=args.execute,
            )
        else:
            result = freeze(args.registration, captures=args.capture, output=args.output)
    except (ValueError, OSError, KeyError) as error:
        print(json.dumps({"status": "refused", "reason": str(error)}), flush=True)
        return 2
    print(json.dumps(result, allow_nan=False), flush=True)
    return 2 if result["status"] == "ruling_required" else 0


if __name__ == "__main__":
    raise SystemExit(main())
