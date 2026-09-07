"""Convert a neuronpedia/jacobian-lens ``.pt`` file to a plain float16 ``.npz`` plus a JSON sidecar.

Loads no model, so it is safe to run beside a training or evaluation process. Runs under a python
that has torch (the venv does not; Homebrew python3 has torch 2.9.1):

    /opt/homebrew/bin/python3 scripts/convert_jlens.py <src.pt> <dst.npz>

The pickle is loaded with ``torch.load(weights_only=True)`` (restricted unpickler). The npz keys
are ``J<l>`` with the file's own index: ``l`` = output of decoder block ``l`` (0-based), so repo
layer ``L`` reads ``J{L-1}``. The sidecar records the source digest, the layer convention, and
per-layer identity distances so a reader can verify the copy without torch.
"""
import hashlib, json, pickletools, sys, time, zipfile
from pathlib import Path

import numpy as np
import torch

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
t0 = time.time()
sha = hashlib.sha256()
with open(src, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 24), b""):
        sha.update(chunk)
with zipfile.ZipFile(src) as z:
    data = z.read([n for n in z.namelist() if n.endswith("data.pkl")][0])
strings, globals_seen = [], []
for op, arg, pos in pickletools.genops(data):
    if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
        strings.append(arg)
    if op.name == "STACK_GLOBAL":
        globals_seen.append(".".join(strings[-2:]))
    if op.name == "GLOBAL":
        globals_seen.append(".".join(arg.split()))
obj = torch.load(src, map_location="cpu", weights_only=True)
J = obj["J"]
layers = sorted(int(k) for k in J.keys())
d = int(next(iter(J.values())).shape[0])
arrays, per_layer = {}, {}
eye_norm = float(np.sqrt(d))
for l in layers:
    t = J[l]
    assert tuple(t.shape) == (d, d), (l, tuple(t.shape))
    a16 = t.to(torch.float16).numpy()
    arrays[f"J{l}"] = a16
    a = a16.astype(np.float32)
    per_layer[str(l)] = {
        "fro_norm": round(float(np.linalg.norm(a)), 4),
        "identity_distance": round(float(np.linalg.norm(a - np.eye(d)) / eye_norm), 6),
        "diag_mean": round(float(np.mean(np.diag(a))), 6),
        "max_abs": round(float(np.abs(a).max()), 4),
    }
dst.parent.mkdir(parents=True, exist_ok=True)
tmp = dst.with_suffix(".npz.tmp")
with open(tmp, "wb") as fh:
    np.savez(fh, **arrays)
tmp.replace(dst)
out_sha = hashlib.sha256(open(dst, "rb").read()).hexdigest()
meta = {
    "source_file": str(src.resolve()),
    "source_sha256": sha.hexdigest(),
    "source_bytes": src.stat().st_size,
    "pickle_globals": sorted(set(globals_seen)),
    "top_level_keys": sorted(obj.keys()) if isinstance(obj, dict) else str(type(obj)),
    "n_prompts": int(obj.get("n_prompts", -1)) if isinstance(obj, dict) else None,
    "d_model": d,
    "layers_in_file": layers,
    "dtype_on_disk": str(next(iter(J.values())).dtype),
    "dtype_saved": "float16",
    "layer_convention": "npz key J<l>: l = output of decoder block l (0-based); repo layer L reads J{L-1}; the last index is the identity (final-layer target)",
    "npz_sha256": out_sha,
    "npz_bytes": dst.stat().st_size,
    "per_layer": per_layer,
    "converted_with": f"torch {torch.__version__}, numpy {np.__version__}, python {sys.version.split()[0]}",
    "elapsed_s": round(time.time() - t0, 1),
}
json.dump(meta, open(dst.with_suffix(".json"), "w"), indent=1)
print(json.dumps({k: v for k, v in meta.items() if k != "per_layer"}, indent=1))
print("identity distance by layer:", {l: per_layer[l]["identity_distance"] for l in per_layer})
