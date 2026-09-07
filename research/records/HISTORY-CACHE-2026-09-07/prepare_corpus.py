"""Freeze completed pilot histories and the eight native step-0 controls, without weights."""

import hashlib
import json
from pathlib import Path

from local_llm_lab.forward import encode_prompt
from local_llm_lab.models import load_model_spec
from local_llm_lab.pipeline.live_lens.session import read_record
from local_llm_lab.pipeline.protocol import SYSTEM_PROMPT, build_prompt

PRIMARY = Path("/Users/daniel.tipton/Desktop/An app")
OUT = Path(__file__).resolve().parent
PILOT = PRIMARY / "research/records/LIVE-LENS-PILOT-2026-09-07"
TOKENIZER = (
    PRIMARY
    / ".cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots"
    / "32f3e8ecf65426fc3306969496342d504bfa13f3"
)


def main():
    manifest_path = PILOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    sources = {str(manifest_path): hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    episodes = []
    for item in manifest["episodes"]:
        if item["kind"] != "agentic":
            continue
        path = PILOT / item["record"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["record_sha256"]:
            raise ValueError(f"pilot record changed: {path}")
        sources[str(path)] = digest
        turns = []
        current = None
        for row in read_record(path):
            if row["kind"] == "begin_turn":
                current = dict(prompt_ids=row["prompt_ids"], continuation_ids=[])
            elif row["kind"] == "emitted":
                if current is None:
                    raise ValueError("emission without prompt")
                current["continuation_ids"].append(row["token_id"])
            elif row["kind"] == "end_turn":
                if row["status"] != "complete" or current is None:
                    raise ValueError("incomplete pilot turn")
                if len(current["continuation_ids"]) != row["emitted_count"]:
                    raise ValueError("pilot emitted count differs")
                turns.append(current)
                current = None
        if current is not None:
            raise ValueError("unterminated pilot turn")
        episodes.append(dict(id=item["label"], turns=turns))
        print(
            json.dumps(
                {
                    "episode": item["label"],
                    "prompt_lengths": [len(t["prompt_ids"]) for t in turns],
                    "continuations": [len(t["continuation_ids"]) for t in turns],
                }
            ),
            flush=True,
        )

    # Local tokenizer assets only. No model loader, checkpoint weights or lock is used.
    from mlx_lm.tokenizer_utils import load as load_tokenizer

    tokenizer = load_tokenizer(TOKENIZER)
    spec = load_model_spec("qwen35-4b")
    native_path = (
        PRIMARY / "research/records/ARM-A-DIVERGENCE-2026-09-07/cache/cache_runner_paths.json"
    )
    base_path = PRIMARY / "outputs/agent-v2/evals/base-test.json"
    native = json.loads(native_path.read_text())
    base = {t["task_id"]: t for t in json.loads(base_path.read_text())["trajectories"]}
    for path in (native_path, base_path):
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for case in native["cases"]:
        task = base[case["task_id"]]
        prompt = build_prompt(
            tokenizer,
            [dict(role="system", content=SYSTEM_PROMPT), dict(role="user", content=task["prompt"])],
            spec=spec,
            generation=True,
        )
        ids = encode_prompt(tokenizer, prompt)
        if len(ids) != case["prompt_tokens"]:
            raise ValueError("native control prompt token count changed")
        episodes.append(
            dict(
                id="step0-" + case["task_id"],
                turns=[dict(prompt_ids=ids, continuation_ids=case["tokens"]["a"])],
            )
        )
    result = dict(
        schema_version=1,
        model=spec.hf_id,
        source_hashes=sources,
        episodes=episodes,
        selection=(
            "All completed agentic pilot turns, plus all eight native step-0 controls; "
            "full recorded continuations"
        ),
    )
    target = OUT / "fixed-history.json"
    with target.open("x") as stream:
        json.dump(result, stream, separators=(",", ":"))
        stream.write("\n")
    print(
        json.dumps(
            {
                "corpus": str(target),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "episodes": len(episodes),
                "turns": sum(len(e["turns"]) for e in episodes),
                "continuation_tokens": sum(
                    len(t["continuation_ids"]) for e in episodes for t in e["turns"]
                ),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
