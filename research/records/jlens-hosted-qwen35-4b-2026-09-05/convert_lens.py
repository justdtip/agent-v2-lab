"""Convert an anthropics/jacobian-lens .pt file to a plain .npz for MLX.

Runs under the system python3 (torch 2.9.1). The pickle is loaded with weights_only=True,
torch's restricted unpickler: only tensors, dicts, lists and primitives are admitted, any
other global aborts the load. Before loading, the pickle's global references are listed
from the zip's data.pkl so the record shows what the file asked for.
"""
# RECORD, NOT A LAUNCHER (2026-09-05). This script produced figures cited in
# design_specifications/pending/WP3-TRANSPORT-WEIGHTS-DESIGN-2026-09-05.md and the hosted-lens results memo.
# It loads the model directly with an ad hoc process check and must not be started by hand again: the
# one operating constraint is one model load at a time, enforced by the loader wrapper and model-run lock
# of issue 83. Runnable versions arrive as package entry points under issues 79 (hosted lens) and 81 (transport).
import sys as _sys
if __name__ == "__main__" and "--i-am-a-record" not in _sys.argv:
    _sys.exit("refusing to run: this file is a record of the 2026-09-05 measurements, not a launcher; "
              "re-run through the package entry points of issues 79 and 81 once the issue 83 lock has landed")

import json, sys, zipfile, pickletools, time
import numpy as np
import torch

src, dst = sys.argv[1], sys.argv[2]
t0 = time.time()
with zipfile.ZipFile(src) as z:
    pkl = [n for n in z.namelist() if n.endswith("data.pkl")][0]
    data = z.read(pkl)
strings = []
globals_seen = []
for op, arg, pos in pickletools.genops(data):
    if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
        strings.append(arg)
    if op.name == "STACK_GLOBAL":
        globals_seen.append(tuple(strings[-2:]))
    if op.name == "GLOBAL":
        globals_seen.append(tuple(arg.split()))
print("pickle globals referenced:", sorted(set(globals_seen)))
obj = torch.load(src, map_location="cpu", weights_only=True)
print("top-level keys:", list(obj.keys()) if isinstance(obj, dict) else type(obj))
J = obj["J"]
layers = sorted(int(k) for k in J.keys())
meta = {
    "source_file": src,
    "n_prompts": int(obj.get("n_prompts", -1)),
    "source_layers": [int(x) for x in obj.get("source_layers", layers)],
    "d_model": int(obj.get("d_model", next(iter(J.values())).shape[0])),
    "dtype_on_disk": str(next(iter(J.values())).dtype),
    "layer_convention": "file index l = output of decoder block l (0-based); repo layer L = l + 1",
}
arrays = {}
for l in layers:
    t = J[l]
    assert t.shape == (meta["d_model"], meta["d_model"]), (l, t.shape)
    arrays[f"J{l}"] = t.to(torch.float16).numpy()
    if l in (layers[0], layers[len(layers)//2], layers[-1]):
        a = t.to(torch.float32).numpy()
        meta[f"stats_J{l}"] = {
            "fro_norm": float(np.linalg.norm(a)),
            "identity_distance": float(np.linalg.norm(a - np.eye(a.shape[0])) / np.linalg.norm(np.eye(a.shape[0]))),
            "diag_mean": float(np.mean(np.diag(a))),
            "max_abs": float(np.abs(a).max()),
        }
np.savez(dst, **arrays)
json.dump(meta, open(dst.replace(".npz", ".json"), "w"), indent=1)
print(json.dumps(meta, indent=1))
print("layers:", layers[0], "..", layers[-1], "count", len(layers), "elapsed", round(time.time() - t0, 1), "s")
