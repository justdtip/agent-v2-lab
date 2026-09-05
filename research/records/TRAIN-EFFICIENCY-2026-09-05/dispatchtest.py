"""Is the training-path recurrence dispatch-bound? Two scaling tests, no model load.

Test 1, scaling in T at fixed width: dispatch-bound gives a straight line through the
origin, slope = ops-per-token x per-dispatch cost, independent of width.
Test 2, scaling in width at fixed T: flat means the GPU is idle between launches
(dispatch-bound); linear means bandwidth-bound; quadratic means compute-bound.

Exercises mlx_lm.models.gated_delta.gated_delta_ops, which is the path training takes
because the fused kernel is disabled whenever the module is in training mode.
"""
# RECORD, NOT A LAUNCHER (2026-09-05). This script allocates on the GPU: its largest width case
# reached 17.27 GiB, 0.97 of the working set, which breached R47's 0.6 threshold for a declared
# window. It must not be started by hand. Re-runs go through a declared window under R47.
import sys as _sys
if __name__ == "__main__" and "--i-am-a-record" not in _sys.argv:
    _sys.exit("refusing to run: this file is a record of the 2026-09-05 dispatch measurement, not "
              "a launcher; its largest case reached 0.97 of the working set and needs a declared "
              "R47 window")
import json, time
import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_ops

B = 1
REAL_HV, REAL_D = 32, 128   # the model's own shapes

def make(T, hv, d):
    sh = (B, T, hv, d)
    return (mx.random.normal(sh), mx.random.normal(sh), mx.random.normal(sh),
            mx.random.uniform(0.9, 1.0, (B, T, hv)), mx.random.uniform(0.0, 1.0, (B, T, hv)))

def timed(T, hv, d, backward, reps=3):
    q, k, v, g, beta = make(T, hv, d)
    mx.eval(q, k, v, g, beta)
    if backward:
        def loss(q, k, v, g, beta):
            y, s = gated_delta_ops(q, k, v, g, beta)
            return y.sum()
        fn = mx.grad(loss, argnums=(0, 1, 2, 3, 4))
    else:
        fn = lambda *a: gated_delta_ops(*a)[0]
    out = fn(q, k, v, g, beta); mx.eval(out)          # warm
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        out = fn(q, k, v, g, beta)
        mx.eval(out)
        best = min(best, time.perf_counter() - t)
    return best

rows = []
print("=== test 1: scaling in T, width fixed at the model's own shapes ===", flush=True)
for T in (125, 250, 500, 1000):
    f = timed(T, REAL_HV, REAL_D, False)
    b = timed(T, REAL_HV, REAL_D, True)
    r = {"test": "T", "T": T, "heads": REAL_HV, "dim": REAL_D,
         "fwd_ms": round(f*1000, 1), "fwd_bwd_ms": round(b*1000, 1),
         "fwd_us_per_token": round(f*1e6/T, 1), "fwd_bwd_us_per_token": round(b*1e6/T, 1)}
    rows.append(r); print(json.dumps(r), flush=True)

print("=== test 2: scaling in width, T fixed at 500 ===", flush=True)
for hv, d in ((8, 64), (16, 64), (16, 128), (32, 128), (32, 256), (64, 256)):
    work = hv * d * d          # per-token state elements, the compute/bandwidth proxy
    f = timed(500, hv, d, False)
    b = timed(500, hv, d, True)
    r = {"test": "width", "T": 500, "heads": hv, "dim": d,
         "state_elems": work, "rel_work": round(work / (REAL_HV*REAL_D*REAL_D), 3),
         "fwd_ms": round(f*1000, 1), "fwd_bwd_ms": round(b*1000, 1)}
    rows.append(r); print(json.dumps(r), flush=True)

json.dump(rows, open("dispatchtest.json", "w"), indent=1)
print(json.dumps({"event": "done", "peak_gib": round(mx.get_peak_memory()/1024**3, 2)}), flush=True)
