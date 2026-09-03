"""Does the mid-layer J-lens actually know the hidden filename, or does it just like some digits?

The single-task probe in `jspace_probe.md` left one result unexplained: at layers 18 and 30 the
Jacobian lens ranked the correct suffix above a previously-seen suffix above an unrelated one,
*including* in the condition where the model's own output shows it does not know the answer.

Two explanations, and they make opposite predictions under averaging:

- **Weak internal trace.** The ordering tracks the task, so it survives averaging over many tasks
  whose correct and incorrect digits differ.
- **Token-frequency artifact.** The ordering tracks the digits themselves (some digit tokens are
  simply likelier in this readout), so it washes out once the digit assignment varies, and it
  persists when the candidates are paired with a context from a *different* task.

So we run a paired sign test. For each ledger task we take the probe point where the directory
listing is hidden, strip the `pending:` lists so the filename is genuinely unavailable, force the
context to end mid-note at `Reading invoice-K-`, and compare the J-lens probability of the true
suffix's first token against a wrong-but-previously-seen suffix from the same task. Then we repeat
the identical comparison with each candidate pair evaluated against a *mismatched* context taken
from another task, which is the null: under the artifact hypothesis the win rate is unchanged.

Run with: uv run python research/jspace_sweep.py
"""

from __future__ import annotations

import re
from typing import Any

TASK_SPLIT = "jsweep"
TASK_COUNT = 720
LAYERS = (18, 24, 30)
CORPUS_SIZE = 8
PROBE_STEP = 3  # the first read whose directory listing has fallen out of the window


def note_prefix(thought: str, short_name: str) -> str | None:
    """The expert note truncated to just before the filename's random suffix."""
    stem = short_name.rsplit("-", 1)[0] + "-"  # "invoice-2-537.txt" -> "invoice-2-"
    if short_name not in thought:
        return None
    return thought[: thought.index(short_name)] + stem


def strip_pending(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove the `pending:` lists, reproducing the run-A condition."""
    out = []
    for message in messages:
        content = message["content"]
        if message["role"] == "assistant":
            content = re.sub(r"; pending: [^\n]*", ".", content)
        out.append({**message, "content": content})
    return out


def suffix_of(path: str) -> str:
    return path.rsplit("-", 1)[-1].split(".")[0]


def main() -> None:
    import mlx.core as mx
    from mlx_lm import load

    from local_llm_lab.pipeline.data import build_rows
    from local_llm_lab.pipeline.jlens import (
        DEFAULT_CORPUS,
        _distribution,
        _encode,
        jlens_map,
        residual_at,
    )
    from local_llm_lab.pipeline.tasks import make_tasks
    from local_llm_lab.project import configure_local_cache

    configure_local_cache()
    model, tok = load(
        "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
        adapter_path="outputs/agent-v2/best-adapter",
    )
    corpus = [_encode(tok, text) for text in DEFAULT_CORPUS[:CORPUS_SIZE]]

    tasks = [
        task
        for task in make_tasks(TASK_SPLIT, TASK_COUNT, perturb=False)
        if task.family == "ledger_reconcile"
    ]
    cases = []
    for task in tasks:
        rows = build_rows(task)
        if len(rows) <= PROBE_STEP:
            continue
        step = task.steps[PROBE_STEP]
        if step.action.name != "read_file":
            continue
        target_path = step.action.arguments["path"]
        short = target_path.rsplit("/", 1)[-1]
        prefix = note_prefix(step.thought, short)
        if prefix is None:
            continue
        earlier = [
            s.action.arguments["path"]
            for s in task.steps[:PROBE_STEP]
            if s.action.name == "read_file"
        ]
        if not earlier:
            continue
        true_suffix = suffix_of(target_path)
        false_suffix = suffix_of(earlier[-1])
        if true_suffix == false_suffix:
            continue
        messages = strip_pending(rows[PROBE_STEP]["messages"][:-1])
        prompt = (
            tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) + prefix
        )
        if true_suffix in prompt:
            continue  # the answer leaked into the context; not a memory test
        cases.append(
            {
                "task_id": task.task_id,
                "prompt": prompt,
                "true": true_suffix,
                "false": false_suffix,
            }
        )

    print(f"{len(cases)} usable ledger probe points\n")
    if len(cases) < 4:
        raise SystemExit("too few probe points to test anything")

    def first_token(text: str) -> int:
        return _encode(tok, text)[0]

    def probabilities(prompt: str, candidates: dict[str, int]) -> dict[str, dict[str, float]]:
        """J-lens probability per layer, plus the model's real output distribution."""
        ids = tok.encode(prompt)
        result: dict[str, dict[str, float]] = {}
        for layer in LAYERS:
            probe = residual_at(model, ids, layer)[0, -1]
            distribution = _distribution(model, jlens_map(model, layer, probe, corpus))
            probs = distribution.tolist()
            result[f"jlens_L{layer}"] = {
                name: float(probs[tid]) for name, tid in candidates.items()
            }
        final = residual_at(model, ids, len(model.model.layers))[0, -1]
        probs = _distribution(model, final).tolist()
        result["model_output"] = {name: float(probs[tid]) for name, tid in candidates.items()}
        return result

    matched: dict[str, list[bool]] = {}
    mismatched: dict[str, list[bool]] = {}
    output_wins: list[bool] = []
    for index, case in enumerate(cases):
        candidates = {"true": first_token(case["true"]), "false": first_token(case["false"])}
        if candidates["true"] == candidates["false"]:
            continue
        # matched: this task's candidates against this task's context
        own = probabilities(case["prompt"], candidates)
        # null: the SAME candidates against a different task's context
        other = cases[(index + 1) % len(cases)]
        alien = probabilities(other["prompt"], candidates)
        for key, values in own.items():
            matched.setdefault(key, []).append(values["true"] > values["false"])
        for key, values in alien.items():
            mismatched.setdefault(key, []).append(values["true"] > values["false"])
        output_wins.append(own["model_output"]["true"] > own["model_output"]["false"])
        mx.clear_cache()
        if (index + 1) % 10 == 0:
            print(f"  {index + 1}/{len(cases)} probed", flush=True)

    def sign_test(wins: int, total: int) -> float:
        """Two-sided binomial p-value against a fair coin, computed exactly."""
        from math import comb

        if total == 0:
            return 1.0
        tail = [comb(total, i) for i in range(total + 1)]
        observed = tail[wins]
        return min(1.0, sum(t for t in tail if t <= observed) / 2**total)

    print("\nHow often is the TRUE suffix more probable than a wrong, previously-seen one?")
    print(f"{'readout':16s} {'matched context':>18s} {'mismatched (null)':>20s}")
    for key in matched:
        m = matched[key]
        n = mismatched[key]
        print(
            f"{key:16s} {sum(m):>8d}/{len(m):<9d} {sum(n):>10d}/{len(n):<9d}"
            f"   ({sum(m) / len(m):.0%} vs {sum(n) / len(n):.0%})"
            f"   matched p={sign_test(sum(m), len(m)):.3f}"
        )
    print(
        "\nIf a readout tracks the task, its matched rate is well above both 50% and its own "
        "mismatched rate.\nIf it only tracks digit frequency, the two columns agree."
    )


if __name__ == "__main__":
    main()
