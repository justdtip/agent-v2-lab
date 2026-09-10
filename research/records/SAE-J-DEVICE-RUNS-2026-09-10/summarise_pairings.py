"""Per-layer summary of a T5 pairing store (measure_pairings.py output): the relative displacement
||h_fit - h_capture|| / ||h_capture|| per cell, its median, 90th percentile and maximum per layer, and
the counts above three lines. The store itself stays on the card (49.5 MB); its sha256 is recorded here.

usage: summarise_pairings.py STORE OUT_JSON
"""
import hashlib, json, statistics, sys
from collections import Counter
from pathlib import Path
store = Path(sys.argv[1]); d = json.loads(store.read_text())
(base,) = [k for k in d if k != "_provenance"]; m = d[base]["measured_pairings"]; prov = d["_provenance"]
layers = {}
for L in sorted(m, key=int):
    rs = sorted(e["relative"] for e in m[L]); kinds = Counter(c["cell"]["position"] for c in prov["measurements"][L])
    widths = sorted({json.dumps(e["pair"]["fit_width"]) for e in m[L]})  # a scalar for one chunk, a list for a merged lens
    layers[L] = {"n": len(rs), "median": statistics.median(rs), "p90": rs[int(0.9 * len(rs)) - 1], "max": rs[-1],
                 "above_1e-3": sum(r > 1e-3 for r in rs), "above_1e-2": sum(r > 1e-2 for r in rs), "above_1e-1": sum(r > 1e-1 for r in rs),
                 "cells_by_position": dict(kinds), "fit_widths": widths, "capture_width": sorted({e["pair"]["capture_width"] for e in m[L]})}
out = {"store": store.name, "store_sha256": hashlib.sha256(store.read_bytes()).hexdigest(), "model_base": base,
       "lens_sha256": prov["lens_sha256"], "nu_sha256": prov["nu_sha256"], "corpus_sha256": prov["corpus_sha256"],
       "implementation_sha256": prov["implementation_sha256"], "fit_widths": prov["fit_widths"],
       "cells": sum(v["n"] for v in layers.values()), "layers": layers,
       "basis": "relative = max over the lens's fit-width chunks of ||h_fit - h_capture||_2 / ||h_capture||_2 at the cell's position and layer; the width-1 path reproduced the stored capture residual bit for bit for every cell or the tool would have refused"}
Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({"cells": out["cells"], "worst_layer_max": max((v["max"], L) for L, v in layers.items()), "any_above_1e-3": sum(v["above_1e-3"] for v in layers.values())}))
