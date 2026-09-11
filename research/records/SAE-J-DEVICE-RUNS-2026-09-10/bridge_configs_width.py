"""Configs for the dictionary width/sparsity control: the device config of each model with only the dictionary
repo folder changed (site, width, sparsity), the declared basis carried over, and a note naming the control."""
import json, sys
from pathlib import Path
base = {"4b": "/workspace/chief/bridge/config-4b.json", "12b": "/workspace/chief/bridge/config-12b.json"}
model, folder, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
if out.exists(): sys.exit(0)
c = json.loads(Path(base[model]).read_text()); c["dictionary_folder"] = folder
c["declared_basis"]["dictionary_control"] = f"width/sparsity control, 2026-09-10: the same thresholds, lens, cells and budget as the device runs with the dictionary {folder}; the control layer and the eight check features are unchanged (feature indices are arbitrary for the two-products identity)"
out.write_text(json.dumps(c, indent=1) + "\n"); print("wrote", out)
