"""Rows 6 and 7 of the WS-A checklist on the card: the graph-once estimator's exactness against
upstream's sequential estimator, and the warmed batching ratio, with the fixture's tensors on CUDA.

Codex's tests are CPU fixtures with no device knob, and the checklist says repeating them on a GPU
host is not a CUDA measurement. This rebuilds the fixture from the same definition (seed 0, the same
tiny Gemma-3 text config, the same frozen token ids) on the selected device and records what ran.
Row 8, graph memory at the intended context on the real checkpoint against upstream's replicated
batch, is not here: it needs the card alone.
"""

import json
import subprocess
import sys
import time
from importlib import import_module
from types import SimpleNamespace

import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

from local_llm_lab import device
from local_llm_lab.torch_jacobian import jacobian_for_prompt_vjp
from local_llm_lab.upstream_ref import load_upstream

selected = sys.argv[1]
out_path = sys.argv[2]
shared = sys.argv[3] == "shared"
pinned = device.pin(seed=0, attention="eager")
UPSTREAM = load_upstream()
HF = import_module("jlens.hf")
WIDTH, BLOCKS, VOCAB = 64, 6, 128
SEQ, TIMING_SEQ, DIM_BATCH, REPEATS = 64, 48, 16, 3
PROMPT = "fixed fixture token IDs"
EPSILON = torch.finfo(torch.float32).eps


def make_model(attention, *, seq_len):
    torch.manual_seed(0)
    torch.set_num_threads(1)
    config = Gemma3TextConfig(
        num_hidden_layers=BLOCKS, hidden_size=WIDTH, num_attention_heads=4, num_key_value_heads=1,
        head_dim=16, intermediate_size=128, vocab_size=VOCAB, sliding_window=16,
        sliding_window_pattern=2, attn_implementation=attention,
    )
    model = Gemma3ForCausalLM(config).float().eval().requires_grad_(False).to(selected)
    frozen_ids = torch.randint(0, VOCAB, (1, seq_len))

    class FixedTokenizer:
        def __call__(self, prompt, *, return_tensors, truncation, max_length):
            assert prompt == PROMPT and return_tensors == "pt" and truncation is True
            return SimpleNamespace(input_ids=frozen_ids[:, :max_length].clone())

    return HF.HFLensModel(model, FixedTokenizer(), force_bos=False)


def maps_close(actual, expected):
    assert actual.keys() == expected.keys()
    errors = {}
    for layer in actual:
        ours, reference = actual[layer], expected[layer]
        assert ours.dtype == torch.float32 and ours.device.type == "cpu"
        assert ours.shape == (WIDTH, WIDTH) and bool(torch.isfinite(ours).all())
        difference = float((ours - reference).abs().max())
        scale = float(reference.abs().max())
        errors[str(layer)] = {
            "max_abs": {"value": difference, "basis": "measured-here"},
            "max_norm_relative": {"value": difference / scale if scale else 0.0, "basis": "measured-here"},
            "within_float32_epsilon": bool(difference <= EPSILON + EPSILON * scale),
        }
    return errors


def timed(fn):
    seconds, result = [], None
    for _ in range(REPEATS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - start)
    return seconds, result


basis_t = "shared-card" if shared else "measured-here"
report = {
    "tool": "device_graph_once.py",
    "rows": "6 (exactness) and 7 (timing); row 8 not here",
    "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
    "device": device.describe(),
    "device_selected": selected,
    "upstream": UPSTREAM.provenance() if hasattr(UPSTREAM, "provenance") else None,
    "fixture": {"width": WIDTH, "blocks": BLOCKS, "vocab": VOCAB, "tokens": SEQ, "timing_tokens": TIMING_SEQ, "dim_batch": DIM_BATCH, "seed": 0},
    "exactness": {},
    "timing": {},
}
for attention in ("eager", "sdpa"):
    model = make_model(attention, seq_len=SEQ)
    assert model.input_device.type == "cuda", model.input_device
    sources = list(range(BLOCKS - 1))
    expected, seq_len, count = UPSTREAM.fitting.jacobian_for_prompt(model, PROMPT, sources, dim_batch=1, max_seq_len=SEQ)
    actual, actual_len, actual_count = jacobian_for_prompt_vjp(model, PROMPT, sources, dim_batch=DIM_BATCH, max_seq_len=SEQ)
    assert (actual_len, actual_count) == (seq_len, count)
    errors = maps_close(actual, expected)
    report["exactness"][attention] = {
        "tokens": SEQ, "source_blocks": sources, "target_block": BLOCKS - 1,
        "tensors_on": str(model.input_device), "errors_by_layer": errors,
        "all_within_float32_epsilon": all(e["within_float32_epsilon"] for e in errors.values()),
    }
    worst = max(e["max_abs"]["value"] for e in errors.values())
    print(f"exactness {attention}: worst max_abs {worst:.3g}; all within float32 epsilon: {report['exactness'][attention]['all_within_float32_epsilon']}")
    model = make_model(attention, seq_len=TIMING_SEQ)
    args = {"target_layer": 4, "max_seq_len": TIMING_SEQ}

    def batched():
        return jacobian_for_prompt_vjp(model, PROMPT, [1, 3], dim_batch=DIM_BATCH, **args)

    def sequential():
        return UPSTREAM.fitting.jacobian_for_prompt(model, PROMPT, [1, 3], dim_batch=1, **args)

    batched()
    sequential()
    b_s, a = timed(batched)
    s_s, e = timed(sequential)
    assert a[1:] == e[1:]
    t_errors = maps_close(a[0], e[0])
    ratio = min(s_s) / min(b_s)
    report["timing"][attention] = {
        "tokens": TIMING_SEQ, "source_blocks": [1, 3], "target_block": 4, "dim_batch": DIM_BATCH,
        "batched_seconds": {"value": b_s, "basis": basis_t},
        "sequential_seconds": {"value": s_s, "basis": basis_t},
        "sequential_over_batched": {"value": ratio, "basis": basis_t},
        "best_of": REPEATS,
        "all_within_float32_epsilon": all(x["within_float32_epsilon"] for x in t_errors.values()),
        "note": "taken while another seat held the card" if shared else "card otherwise idle",
    }
    print(f"timing {attention}: sequential/batched = {ratio:.3f} (best of {REPEATS}; {basis_t}); batched best {min(b_s):.4f}s, sequential best {min(s_s):.4f}s")
ok = all(v["all_within_float32_epsilon"] for v in report["exactness"].values()) and all(
    v["all_within_float32_epsilon"] for v in report["timing"].values()
)
report["status"] = "pass" if ok else "fail"
json.dump(report, open(out_path, "w"), indent=2)
print("status:", report["status"], "->", out_path)
