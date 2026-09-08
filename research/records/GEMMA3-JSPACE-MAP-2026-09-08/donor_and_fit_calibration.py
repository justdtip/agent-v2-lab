"""Two measurements that each gate a decision, in one model load.

**The donor probe.** `pointer_chain-0018` answers with a fabricated path where the Result was
wanted, and no position in the episode puts the correct token first — so an intervention has no
in-episode donor. The Chief's test: ask the model directly, against the same context. If it cannot
produce the Result under a direct question, that is a larger result than the intervention it was
meant to enable, and it costs one short generation.

**The fit calibration.** A transcript-length lens fit projects 21 GiB at 2,816 tokens, which is
1.97x R47's stop threshold, and the Director has not authorised it. The projection was made the way
this repository's last two were: from the cheapest instance. This measures the real peak for one
2,816-token window with every layer's residual retained in float32 — the floor the batch multiplies.

    python donor_and_fit_calibration.py --out <dir> --i-am-a-record
"""
from __future__ import annotations

import sys as _sys

if __name__ == "__main__" and "--i-am-a-record" not in _sys.argv:
    _sys.exit(
        "refusing to run: this file is a record of two measurements taken on 2026-09-08, not a "
        "launcher. It loads a 4B checkpoint and takes the model-run lock. Re-run it deliberately "
        "with --i-am-a-record, inside an announced box window."
    )

import argparse
import json
from dataclasses import replace
from pathlib import Path

_sys.path.insert(0, "/Users/daniel.tipton/Desktop/An app/src")

RECORD = (
    "/private/tmp/claude-501/-Users-daniel-tipton-Desktop-An-app/"
    "18df6241-b6a2-4c3c-8b24-22cad2612a51/scratchpad/pilot/stage2/"
    "agentic-d2-pointer_chain-0018.jsonl"
)
QUESTION = (
    "Ignore the task instructions above. Answer this one question directly: in the file you most "
    "recently read, what is the value of the Result field? Reply with exactly that value and "
    "nothing else."
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="gemma3-4b")
    parser.add_argument("--fit-window", type=int, default=2816)
    parser.add_argument("--i-am-a-record", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "donor_and_fit.json"
    if destination.exists():
        raise SystemExit(f"{destination} exists; this script never overwrites a result")

    import mlx.core as mx
    import numpy as np

    from local_llm_lab.arch import ArchitectureView
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.pipeline.evaluate import load_policy, make_sampler
    from local_llm_lab.pipeline.protocol import build_prompt
    from local_llm_lab.pipeline.runner import generate_turn_with_count

    out: dict = {}
    spec = replace(load_model_spec(args.model), cache_strategy="none")
    mx.set_cache_limit(2 * 2**30)
    model, tokenizer, view, resolved = load_policy(spec, None)
    model.eval()

    # ---- 1. the donor probe
    events = [json.loads(line)["event"] for line in open(RECORD) if line.startswith("{")]
    turns = [e for e in events if e.get("kind") == "begin_turn"]
    context = turns[-1]["context"]["messages"]
    # Appended to the last turn's content rather than added as a new turn. Gemma's template
    # enforces strict user/model alternation and the context already ends with a user turn,
    # because tool observations render under the user role on this family — the same constraint
    # that produced the observation-rendering work. Merging also keeps the question at very nearly
    # the position the fork sat at, which is the point of asking it against this context.
    messages = [dict(m) for m in context]
    messages[-1]["content"] = f"{messages[-1]['content']}\n\n{QUESTION}"
    prompt = build_prompt(tokenizer, messages, spec=spec, keep_last=8, generation=True)
    mx.reset_peak_memory()
    text, count, _ = generate_turn_with_count(
        model, tokenizer, prompt, make_sampler(0.0), 64, None, spec=spec
    )
    out["donor_probe"] = {
        "prompt_tokens": len(tokenizer.encode(prompt)),
        "generated_tokens": count,
        "answer": text,
        "contains_result": "artifact-93330" in text,
        "peak_gib": round(mx.get_peak_memory() / 2**30, 3),
    }
    print(json.dumps(out["donor_probe"], indent=1))

    # ---- 2. the fit calibration: one window, every layer's residual retained in float32
    captured: dict[int, object] = {}

    class Sink:
        """Retain every layer's residual in float32, which is what a fit holds."""

        def residual(self, layer, offset, hidden):
            captured[layer] = np.array(hidden.astype(mx.float32))

        def attention(self, *a, **k):
            pass

        def output(self, *a, **k):
            pass

    ids = mx.array([[1] * args.fit_window])
    mx.reset_peak_memory()
    from local_llm_lab.arch import NativeCapture

    layers = tuple(range(1, view.num_layers))
    with NativeCapture(view, Sink(), layers=layers, attention_blocks=()) as captured_forward:
        captured_forward(ids, cache=view.make_cache())
    residual_bytes = sum(a.nbytes for a in captured.values())
    out["fit_calibration"] = {
        "window_tokens": args.fit_window,
        "layers_retained": len(captured),
        "residual_bytes_gib": round(residual_bytes / 2**30, 3),
        "mlx_peak_gib": round(mx.get_peak_memory() / 2**30, 3),
        "r47_stop_threshold_gib": 10.66,
    }
    print(json.dumps(out["fit_calibration"], indent=1))
    with destination.open("x") as handle:
        json.dump(out, handle, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
