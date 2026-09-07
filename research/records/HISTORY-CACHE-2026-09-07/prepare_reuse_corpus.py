"""Add deterministic long-history cases using token structure, before model comparisons."""

import hashlib
import json

from prepare_corpus import OUT, PRIMARY, TOKENIZER

from local_llm_lab.agent_protocol import Action
from local_llm_lab.forward import encode_prompt
from local_llm_lab.models import load_model_spec
from local_llm_lab.pipeline.protocol import (
    SYSTEM_PROMPT,
    assistant_message,
    build_prompt,
    tool_message,
)


def main():
    # Tokenizer assets only: this does not load the checkpoint.
    from mlx_lm.tokenizer_utils import load as load_tokenizer

    tokenizer = load_tokenizer(TOKENIZER)
    spec = load_model_spec("qwen35-4b")
    source = PRIMARY / "outputs/agent-v2/evals/base-test.json"
    trajectories = json.loads(source.read_text())["trajectories"]
    original = OUT / "fixed-history.json"
    corpus = json.loads(original.read_text())
    corpus["source_hashes"][str(original)] = hashlib.sha256(original.read_bytes()).hexdigest()
    if hashlib.sha256(source.read_bytes()).hexdigest() != corpus["source_hashes"][str(source)]:
        raise ValueError("source trajectories changed after the first corpus was frozen")

    selected = []
    for family in ("ledger_reconcile", "batch_update", "aggregate_report"):
        for task in sorted(trajectories, key=lambda task: task["task_id"]):
            if task["family"] != family or task["variant"] != "clean":
                continue
            messages = [
                dict(role="system", content=SYSTEM_PROMPT),
                dict(role="user", content=task["prompt"]),
            ]
            turns = []
            for step in task["steps"]:
                if "action" not in step:
                    break
                ids = encode_prompt(tokenizer, build_prompt(tokenizer, messages, spec=spec))
                # These are declared teacher-forcing IDs retokenized from existing text,
                # not a claim about the original generator's exact token segmentation.
                continuation = tokenizer.encode(step["raw"], add_special_tokens=False)[:24]
                turns.append(dict(prompt_ids=ids, continuation_ids=continuation))
                action = Action(step["action"]["name"], step["action"]["arguments"])
                messages.append(assistant_message(step["thought"], action))
                if action.name == "finish":
                    break
                messages.append(tool_message(action.name, step["observation"]))

            eligible = next(
                (
                    i
                    for i in range(1, len(turns))
                    if len(turns[i - 1]["prompt_ids"]) > 2048
                    and len(turns[i]["prompt_ids"]) > 2048
                    and turns[i - 1]["prompt_ids"][:2048] == turns[i]["prompt_ids"][:2048]
                ),
                None,
            )
            if eligible is None:
                continue
            start, stop = eligible - 1, min(eligible + 3, len(turns))
            item = dict(
                id="long-" + task["task_id"],
                turns=turns[start:stop],
                original_turn_indices=list(range(start, stop)),
                continuation_source="retokenized recorded raw text, first 24 IDs",
            )
            corpus["episodes"].append(item)
            selected.append(
                dict(
                    id=item["id"],
                    original_turn_indices=item["original_turn_indices"],
                    prompt_lengths=[len(t["prompt_ids"]) for t in item["turns"]],
                )
            )
            break
    if not selected:
        raise ValueError("no real conversation history offers a native 2048-token reuse boundary")
    corpus["selection"] += (
        "; plus the first clean task by task ID in each of ledger_reconcile, batch_update, "
        "aggregate_report with adjacent prompts sharing a full native 2048-token chunk. "
        "Keep four turns starting just before the first eligible transition (or the remaining "
        "turns if shorter), and force the first 24 locally retokenized raw-text IDs. "
        "Selection uses input structure only, before this patch's model comparisons."
    )
    target = OUT / "fixed-history-native-reuse.json"
    with target.open("x") as stream:
        json.dump(corpus, stream, separators=(",", ":"))
        stream.write("\n")
    print(
        json.dumps(
            dict(
                selected=selected,
                corpus=str(target),
                sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                episodes=len(corpus["episodes"]),
                turns=sum(len(e["turns"]) for e in corpus["episodes"]),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
