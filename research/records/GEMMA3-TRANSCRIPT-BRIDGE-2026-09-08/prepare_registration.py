"""Prepare the source-only transcript registration; never import a model runtime.

The BF16 snapshot inventory is reused from the completed fit's committed record.
The 4-bit inventory is streamed through SHA256, not loaded as tensors. Inspection
and capture recheck their respective assets before accepting them.
"""

import hashlib
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RECORD = Path(__file__).resolve().parent
PRIMARY = Path("/Users/daniel.tipton/Desktop/An app")


def main():
    driver = runpy.run_path(str(ROOT / "scripts/gemma_transcript_lens.py"))
    from local_llm_lab.spawn import run

    bf16 = json.loads(
        Path(
            "/Users/daniel.tipton/worktrees/gemma-lens-fitting/research/records/"
            "GEMMA3-REGRESSION-2026-09-08/checkpoint-identity.json"
        ).read_bytes()
    )
    model = driver["model_spec"]("gemma3-4b")
    fourbit = driver["snapshot_identity"](Path(model.hf_id), hf_id=model.hf_id)
    tokenizer = driver["tokenizer_identity_from_directory"](model.hf_id)
    landed = run(
        ["git", "-C", str(PRIMARY), "rev-parse", "7d2a18a^{commit}"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    ruling = run(
        ["git", "-C", str(PRIMARY), "rev-parse", "a1ff0b0^{commit}"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    gates = {"landed_commit": landed, "ruling_commit": ruling}
    for key, revision in (
        ("environment_commit", "020aa89"),
        ("second_amendment_commit", "ec4b9d1"),
        ("span_commit", "2edad6e"),
    ):
        gates[key] = run(
            ["git", "-C", str(PRIMARY), "rev-parse", revision + "^{commit}"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    evidence = []
    for name in (
        "design_specifications/pending/CODEX-TASKS-2026-09-08.md",
        "research/records/GEMMA3-JSPACE-MAP-2026-09-08/DIAGNOSTIC-RERUN.md",
        "src/local_llm_lab/pipeline/protocol.py",
        "configs/models/gemma3-4b.yaml",
        "configs/models/gemma3-4b-bf16.yaml",
        "src/local_llm_lab/pipeline/env.py",
        "src/local_llm_lab/agent_tasks.py",
        "src/local_llm_lab/pipeline/live_lens/session.py",
    ):
        if "CODEX-TASKS" in name:
            commit = gates["second_amendment_commit"]
        elif name.endswith(("pipeline/env.py", "agent_tasks.py")):
            commit = gates["environment_commit"]
        elif name.endswith("live_lens/session.py"):
            commit = gates["span_commit"]
        else:
            commit = landed
        blob = run(
            ["git", "-C", str(PRIMARY), "show", f"{commit}:{name}"], capture_output=True, check=True
        ).stdout
        evidence.append(
            {
                "path": str(PRIMARY / name),
                "commit": commit,
                "sha256": hashlib.sha256(blob).hexdigest(),
                "storage": "git_blob",
            }
        )
    registration = {
        "schema_version": 1,
        "format": driver["FORMAT"],
        "producer_model": "gemma3-4b",
        "fit_models": list(driver["FIT_MODELS"]),
        "model_identity": {"base": "google/gemma-3-4b-it", "training": None, "num_layers": 34},
        "fitting_context_tokens": 2048,
        "tokenizer_directory": model.hf_id,
        "tokenizer": tokenizer,
        "snapshots": {"gemma3-4b": fourbit, "gemma3-4b-bf16": bf16},
        "cohorts": driver["plan_cohorts"](),
        "evaluation": driver["EVALUATION"],
        "rendering_gate": {**gates, "files": evidence},
        "amends": "TRANSCRIPT-REGISTRATION.json; unlaunched v1, preserved",
        "capture_limits": {"max_prompt_tokens": 8192, "max_forward_tokens": 8393},
        "capture_projection": {
            "peak_gib": 16.0,
            "basis": (
                "Conservative, unmeasured source bound for the registered 8192-token prompt cap "
                "and native 2048-token prefill chunks. Decimal GB: 4-bit weight data about2.1 "
                "+ transient load copy2.1 + FP32 KV cache1.07 (5 global8393 and29 local3072; "
                "4 KV heads,256 head dimensions,two K/V arrays) + two FP32 prefill logits4.30 "
                "+ two FP32 attention buffers1.10 + FFN intermediates0.17 + possible FP32 "
                "readout workspace2.69 = about13.53GB. Projection16GiB includes >3GB margin. "
                "FP32 and full attention/readout scratch are conservative allowances; "
                "the actual fused implementation may use less. Oversize prompts refuse "
                "before a forward; no truncation to make the bound pass."
            ),
        },
        "residual_source": "native",
        "initial_fit_calibration_bound_gib": 14.5,
        "provisional_full_fit_peak_gib": 18.5,
        "fit_projection_basis": (
            "Prior BF16 native128 fit measured12.44GiB. At2048, conservatively add about0.67GB "
            "for all34 FP32 residual streams,0.35GB for masked row/cast arrays,0.27GB for "
            "two full attention matrices,0.16GB FFN growth,and4.03GB for two FP32 logits "
            "buffers beyond the128 baseline. Total about17.55GiB; round up18.5GiB. "
            "Some output graphs are pruned and fused buffers may not coexist; this is an "
            "unmeasured upper estimate, not claimed actual peak. The256/512 calibration "
            "starts at14.5GiB using the same growth calculation at512 with margin. "
            "Native masked calibration then measures512/1024/2048 and projects each "
            "larger size before launch under the22GiB registry budget. Its measured "
            "projection is mandatory before the combined two-precision fit window."
        ),
        "queue": (
            "after CRO confound audit and Deputy stage-two all-layer map; "
            "not an active reservation"
        ),
    }
    path = RECORD / "TRANSCRIPT-REGISTRATION-v2.json"
    with path.open("x") as stream:
        json.dump(registration, stream, indent=2, allow_nan=False)
        stream.write("\n")
    driver["read_registration"](path)
    print(
        json.dumps(
            {
                "registration": str(path),
                "sha256": driver["file_record"](path)["sha256"],
                "tasks": sum(row["count"] for row in registration["cohorts"]),
                "capture_peak_gib": registration["capture_projection"]["peak_gib"],
            }
        )
    )


if __name__ == "__main__":
    main()
