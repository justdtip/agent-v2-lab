"""The port's acceptance, like for like: the native-dtype loop against the model's own forward.

`residuals` casts to float32 at every block, so comparing it against the model's bfloat16 forward
measures the promotion as well as the port and cannot separate them -- which is why the first
acceptance run returned a large number that said nothing on its own.

`diagnostic_native_final_residual` is the same traversal without the promotion: the view's own
loop, in the model's dtype, which is exactly what the residual-equivalence gate exists to compare.
Run above the sliding window it is the port's acceptance, and run at 64 tokens it is the control
for that, because below the window Gemma's two masks are identical and no mask defect can show.

The negative control breaks the dispatch and must move the long number and leave the short one.
"""
from __future__ import annotations

import sys as _sys

if __name__ == "__main__" and "--i-am-a-record" not in _sys.argv:
    _sys.exit(
        "refusing to run: this file is the record of the architecture-view port's acceptance on "
        "2026-09-08, not a launcher. It loads a 4B checkpoint and takes the model-run lock. "
        "Re-run it deliberately with --i-am-a-record, inside an announced box window."
    )

import json
import sys

sys.path.insert(0, "/Users/daniel.tipton/Desktop/An app/src")


def main() -> int:
    import mlx.core as mx

    from local_llm_lab.arch import ArchitectureView
    from local_llm_lab.runlock import load_weights

    model, _ = load_weights("/Users/daniel.tipton/Desktop/An app/models/gemma-3-4b-it-4bit")
    view = ArchitectureView.from_model(model)
    out: dict = {}

    def compare(length: int) -> dict:
        ids = mx.arange(length, dtype=mx.int32)[None, :] % 1000 + 10
        ours = view.diagnostic_native_final_residual(ids).astype(mx.float32)
        theirs = view.text_module(ids).astype(mx.float32)
        diff = float(mx.max(mx.abs(ours - theirs)).item())
        scale = float(mx.max(mx.abs(theirs)).item())
        return {"tokens": length, "max_abs": diff, "max_rel": diff / (scale or 1.0),
                "scale": scale, "dtype": str(view.diagnostic_native_final_residual(ids).dtype)}

    for length in (64, 1400):
        row = compare(length)
        out[f"native_{length}"] = row
        print(json.dumps({"event": "native_gate", **row}), flush=True)

    # Break the path the gate actually runs. After the hoist the diagnostic observes once and
    # never calls `masks`, so patching `masks` — which is what the first control did — patches a
    # method that is no longer on the path and proves nothing. A control that cannot fail is the
    # thing this whole exercise is about.
    original = ArchitectureView._observe_forward

    def broken(self, ids, cache=None):
        entry, observed = original(self, ids, cache)
        return entry, dict.fromkeys(observed, observed[0])

    ArchitectureView._observe_forward = broken
    try:
        for length in (64, 1400):
            row = compare(length)
            out[f"control_{length}"] = row
            print(json.dumps({"event": "negative_control", **row}), flush=True)
    finally:
        ArchitectureView._observe_forward = original

    out["peak_gib"] = mx.get_peak_memory() / 2**30
    from pathlib import Path
    Path(sys.argv[1] if len(sys.argv) > 1 else "native.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"event": "done", "peak_gib": round(out["peak_gib"], 3)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
