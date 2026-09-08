"""The architecture port's acceptance on the real Gemma 3, not on a toy.

`residual_source_agreement` compares this repository's hand-run decoder loop against the model's
own forward, per layer. Before the port it was expected to be large on Gemma, because the loop
omitted a sqrt(hidden) entry scale and built one mask where the model builds two. The port has
landed, so the number is now evidence rather than a prediction.

**Run above the sliding window or it proves nothing.** Gemma's global and windowed masks are
identical for any sequence shorter than 1,024, so agreement at 64 tokens is agreement about a case
where the defect cannot appear. Both lengths are measured here and the short one is the control
for the long one.

**And a negative control**, which is the part a passing number cannot supply on its own: the mask
dispatch is deliberately broken -- every block handed the first block's mask -- and the same
comparison must then be non-zero **at the long length and zero at the short one**. That is the
only demonstration that this instrument can see what the pre-port one could not.
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
from pathlib import Path

REPO = Path(__file__).resolve().parents[4] if "scratchpad" in str(__file__) else Path.cwd()
sys.path.insert(0, "/Users/daniel.tipton/Desktop/An app/src")

DEFAULT = "/Users/daniel.tipton/Desktop/An app/models/gemma-3-4b-it-4bit"
CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
SHORT, LONG = 64, 1400  # 1,400 > the 1,024 window, so the two masks genuinely differ


def main() -> int:
    import mlx.core as mx

    from local_llm_lab.arch import ArchitectureView
    from local_llm_lab.runlock import load_weights

    model, tokenizer = load_weights(CHECKPOINT)
    view = ArchitectureView.from_model(model)
    layers = tuple(range(1, view.num_layers + 1))
    out: dict = {
        "checkpoint": CHECKPOINT,
        "num_layers": view.num_layers,
        "hidden_size": view.hidden_size,
        "sliding_window": getattr(getattr(view, "args", None), "sliding_window", None),
        "spans": {index: view.attention_span(index) for index in range(view.num_layers)},
    }
    print(json.dumps({"event": "loaded", **{k: out[k] for k in ("num_layers", "hidden_size")}}),
          flush=True)

    for name, length in (("short", SHORT), ("long", LONG)):
        ids = mx.arange(length, dtype=mx.int32)[None, :] % 1000 + 10
        agreement = view.residual_source_agreement(ids, layers)
        worst = max(agreement.values())
        out[f"{name}_tokens"] = length
        out[f"{name}_agreement"] = agreement
        out[f"{name}_worst"] = worst
        print(json.dumps({"event": "agreement", "length": name, "tokens": length,
                          "worst_abs_diff": worst,
                          "nonzero_layers": [k for k, v in agreement.items() if v != 0.0][:12]}),
              flush=True)
        del agreement

    # ---------------------------------------------------------------- the negative control
    original = ArchitectureView.masks

    def broken(self, h, cache, *, hidden_spans=None, record=None):
        """Every block handed the first block's mask: the defect the old gate could not see."""
        proper = original(self, h, cache, hidden_spans=hidden_spans, record=record)
        first = proper[0]
        return dict.fromkeys(proper, first)

    ArchitectureView.masks = broken
    try:
        for name, length in (("short", SHORT), ("long", LONG)):
            ids = mx.arange(length, dtype=mx.int32)[None, :] % 1000 + 10
            agreement = view.residual_source_agreement(ids, layers)
            worst = max(agreement.values())
            out[f"control_{name}_worst"] = worst
            print(json.dumps({"event": "negative_control", "length": name, "tokens": length,
                              "worst_abs_diff": worst}), flush=True)
            del agreement
    finally:
        ArchitectureView.masks = original

    out["peak_gib"] = mx.get_peak_memory() / 2**30
    Path(sys.argv[2] if len(sys.argv) > 2 else "agreement.json").write_text(
        json.dumps(out, indent=1)
    )
    print(json.dumps({"event": "done", "peak_gib": round(out["peak_gib"], 3)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
