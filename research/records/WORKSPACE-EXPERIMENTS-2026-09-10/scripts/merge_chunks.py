"""Merge chunked float32 lens fits into one lens, weighting each chunk by its prompt count, exactly
as upstream's JacobianLens.merge does; write exact-maps.npz, nu.json (the chunks' declarations and
the weights) and manifest.json under OUT.

usage: merge_chunks.py OUT CHUNK_DIR [CHUNK_DIR ...]
"""
import json, sys, hashlib
from pathlib import Path
import numpy as np
import torch
from jlens.lens import JacobianLens

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
chunks = [Path(p) for p in sys.argv[2:]]
lenses, decls = [], []
for c in chunks:
    z = np.load(c / "exact-maps.npz"); nu = json.loads((c / "nu.json").read_text()); man = json.loads((c / "manifest.json").read_text())
    n = int(man["n_rows"]); d = int(man["d_model"])
    # keys are repo layers J{r}; upstream indexing is repo-1
    jac = {int(k[1:]) - 1: torch.from_numpy(z[k]) for k in z.files}
    lenses.append(JacobianLens(jacobians=jac, n_prompts=n, d_model=d))
    decls.append({"chunk": str(c), "n_rows": n, "rows_from": man.get("rows_from"), "rows_to": man.get("rows_to"),
                  "dim_batch": man["dim_batch"], "nu_sha256": hashlib.sha256(json.dumps(nu, sort_keys=True).encode()).hexdigest()})
merged = JacobianLens.merge(lenses)
np.savez_compressed(out / "exact-maps.npz", **{f"J{i + 1}": merged.jacobians[i].numpy().astype(np.float32) for i in merged.jacobians})
h = hashlib.sha256((out / "exact-maps.npz").read_bytes()).hexdigest()
(out / "manifest.json").write_text(json.dumps({"schema_version": 1, "seat": "chief", "stage": "merge of chunked float32 exact lenses (JacobianLens.merge, weighted by n_prompts)",
    "chunks": decls, "n_prompts": merged.n_prompts, "d_model": merged.d_model, "layers": sorted(i + 1 for i in merged.jacobians),
    "exact_maps_sha256": h}, indent=2) + "\n")
print(json.dumps({"event": "merged", "n_prompts": merged.n_prompts, "layers": len(merged.jacobians), "sha256": h[:16]}))
