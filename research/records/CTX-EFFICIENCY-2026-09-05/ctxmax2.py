"""Native-window attempt, rebuilt to R46. Not launched until the Director says so.

Changes from the run the guard killed:
  - a progress line per chunk carrying elapsed, tokens done, peak GiB and peak as a
    share of the device's recommended working set. The share is only readable inside
    the process, so this line is the external guard's only sensor for it (R46 ii).
  - the share is a validity signal, not a safety one: crossing the working set marks
    the row degraded and keeps the number. Only the guard kills, and only on the
    kernel's pressure level held across two samples (R46 iv).
  - partial progress survives a kill, which the previous version's one-line-per-row
    printing did not allow.
"""
# RECORD, NOT A LAUNCHER (2026-09-05). This script produced the figures in RECORD.md, cited by
# the Chief's efficiency findings. It loads the model directly and does not take the model-run lock
# of issue 83, so it must not be started by hand again: an earlier sibling of it ran orphaned for
# 57 minutes and made the machine unusable. Successors run under R47 and R48 as package entry
# points, in a window the Director declares.
import sys as _sys
if __name__ == "__main__" and "--i-am-a-record" not in _sys.argv:
    _sys.exit("refusing to run: this file is a record of the 2026-09-05 context-efficiency "
              "measurements, not a launcher; it takes no model-run lock (issue 83) and runs only "
              "in a window the Director declares (R47)")

import json, time, sys
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

GiB = 1024**3
N, CHUNK, DECODE_STEPS = 262144, 512, 16
WS = mx.device_info()["max_recommended_working_set_size"]

model, tokenizer = load("mlx-community/Qwen3.5-4B-MLX-4bit")
mx.eval(model.parameters())
print(json.dumps({"event": "loaded", "working_set_gib": round(WS/GiB, 2)}), flush=True)

mx.clear_cache(); mx.reset_peak_memory()
ids = mx.random.randint(0, tokenizer.vocab_size, (N,))
cache = make_prompt_cache(model)
crossed = False
t0 = time.perf_counter()
for i in range(0, N, CHUNK):
    logits = model(ids[None, i:i+CHUNK], cache=cache)
    mx.eval(logits)
    peak = mx.get_peak_memory()
    share = peak / WS
    crossed = crossed or share >= 1.0
    if (i // CHUNK) % 16 == 0 or share >= 0.95:
        print(json.dumps({"event": "prefill", "done": i + CHUNK, "of": N,
                          "elapsed_s": round(time.perf_counter()-t0, 1),
                          "peak_gib": round(peak/GiB, 2),
                          "ws_share": round(share, 3),
                          "crossed_working_set": crossed}), flush=True)
prefill_s = time.perf_counter() - t0

tok = mx.argmax(logits[:, -1, :], axis=-1)[None]; mx.eval(tok)
t = time.perf_counter()
for _ in range(DECODE_STEPS):
    logits = model(tok, cache=cache)
    tok = mx.argmax(logits[:, -1, :], axis=-1)[None]
    mx.eval(tok)
decode_s = time.perf_counter() - t

peak = mx.get_peak_memory()
row = {"event": "row", "label": "native-window-chunk512", "tokens": N, "chunk": CHUNK,
       "prefill_s": round(prefill_s, 1), "prefill_tok_s": round(N/prefill_s, 1),
       "decode_tok_s": round(DECODE_STEPS/decode_s, 2),
       "peak_gib": round(peak/GiB, 3), "ws_share": round(peak/WS, 3),
       "crossed_working_set": crossed,
       "validity": "degraded - crossed working set" if crossed else "clean"}
print(json.dumps(row), flush=True)
json.dump(row, open("ctxmax2.json", "w"), indent=1)
