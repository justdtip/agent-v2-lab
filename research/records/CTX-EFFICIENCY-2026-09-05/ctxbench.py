"""Prefill/decode curve for Qwen3.5-4B-4bit on this machine.

One model load, all lengths swept inside this process (concurrency rule of record).
Writes results incrementally so a kill at any point still leaves a usable curve.
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

MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072]
CHUNK = 2048
DECODE_STEPS = 32
OUT = "ctxbench.json"
GiB = 1024**3

def log(*a):
    print(*a, flush=True)

info = mx.device_info()
log(f"device {info['device_name']}  working set {info['max_recommended_working_set_size']/GiB:.2f} GiB")

t0 = time.perf_counter()
model, tokenizer = load(MODEL)
mx.eval(model.parameters())
log(f"model loaded in {time.perf_counter()-t0:.1f}s  resident {mx.get_active_memory()/GiB:.2f} GiB")

vocab = tokenizer.vocab_size
results = []

for n in LENGTHS:
    mx.clear_cache()
    mx.reset_peak_memory()
    base = mx.get_active_memory()
    try:
        ids = mx.random.randint(0, vocab, (n,))
        cache = make_prompt_cache(model)

        # prefill, chunked
        t = time.perf_counter()
        for i in range(0, n, CHUNK):
            logits = model(ids[None, i:i+CHUNK], cache=cache)
            mx.eval(logits, *[x for c in cache for x in (c.state if hasattr(c,'state') else [])])
        mx.eval(logits)
        prefill_s = time.perf_counter() - t

        # decode
        tok = mx.argmax(logits[:, -1, :], axis=-1)[None]
        mx.eval(tok)
        t = time.perf_counter()
        for _ in range(DECODE_STEPS):
            logits = model(tok, cache=cache)
            tok = mx.argmax(logits[:, -1, :], axis=-1)[None]
            mx.eval(tok)
        decode_s = time.perf_counter() - t

        peak = mx.get_peak_memory()
        row = {
            "tokens": n,
            "prefill_s": round(prefill_s, 3),
            "prefill_tok_s": round(n / prefill_s, 1),
            "decode_tok_s": round(DECODE_STEPS / decode_s, 2),
            "peak_gib": round(peak / GiB, 3),
            "kv_gib_predicted": round(n * 32 * 1024 / GiB, 3),
            "ok": True,
        }
        log(json.dumps(row))
    except Exception as e:
        row = {"tokens": n, "ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
        log(json.dumps(row))
    results.append(row)
    del cache
    mx.clear_cache()
    json.dump(results, open(OUT, "w"), indent=1)
    if not row["ok"]:
        log("stopping at first failure")
        break

log("done")
