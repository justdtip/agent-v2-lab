"""Write the Chief's bridge configs for the 4B and 12B A1/A2 runs. No digest is typed: checkpoint hashes come
from the captures' hash manifests, lens digests from the admission sidecars. Thresholds are declared here,
before any run, with their basis recorded inside the config so the result carries it."""
import json, sys
from pathlib import Path
def load(p): return json.loads(Path(p).read_text())
FEATURES = [61, 403, 1452, 1694, 7616, 9546, 13022, 16363]
BASIS = {
  "declared": "Chief, 2026-09-10, before the device A1/A2 runs; every threshold is a refusal condition, none admits by default",
  "k": "10, the laptop A1's k (SAE-J-BRIDGE-2026-09-09), so overlaps compare with that record",
  "chunk": "256 features per score chunk, the laptop A1's chunk; memory only, no effect on values",
  "control_layer": "a strong control from the laptop control curve (overlap@10 with the hooked layer: 0.047 at layer 2 for the 4B); for the 12B the analogous early layer 3; that curve was measured with the hosted bf16 lens and is the declared basis, not a prediction for the float32 device lens",
  "maximum_control_overlap": "0.5: the laptop curve shows a neighbouring layer shares 0.7-0.8 of top-10 sets, so a control sharing half or more is not a control; 0.5 sits below the neighbour band and far above the strong-control values",
  "check_features": "the laptop A1's eight named features, indices carried over unchanged to the 12B dictionary as arbitrary indices; the two-products check is a numerical identity, not a semantic choice",
  "two_product_tolerance": "1e-3 absolute on scores of order 1-40 (laptop top-10 scores 0.26-37.8): float32 chunked accumulation differs at ~1e-5, an orientation error at order 1",
  "convention_threshold": "0.5, the second amendment's table (laptop: raw arm 0.245, gain arm 0.995)",
  "error_budget": "raw_reconstruction_threshold 0.5 and lens_score_error_threshold 0.5: a decomposition that leaves half the residual, or half the score, unexplained is not read as a description of the site; denominator_floor 1e-6 against float32 zeros; near-zero refuses. These are refusal lines for out-of-domain agent transcripts whose reconstruction error no one has measured; the run reports the shares whatever they are",
  "a2.identity_tolerance": "1e-3 relative on the emitted token's score: the identity L h = L b + sum z_i L d_i + L e holds to float32 rounding (~1e-6) when the orientation is right and fails at order 1 when it is wrong",
}
def config(base, label, hashes, lens_meta, repo, folder, control_layer):
    return {
      "schema_version": 1, "model_base": base, "capture_model": label,
      "checkpoint_sha256": load(hashes), "lens_sha256": load(lens_meta)["npz_sha256"],
      "dictionary_repo": repo, "dictionary_folder": folder,
      "error_budget": {"raw_reconstruction_threshold": 0.5, "lens_score_error_threshold": 0.5, "denominator_floor": 1e-6, "near_zero_policy": "refuse"},
      "a1": {"k": 10, "chunk": 256, "control_layer": control_layer, "check_features": FEATURES, "two_product_tolerance": 1e-3, "maximum_control_overlap": 0.5, "convention_threshold": 0.5},
      "a2": {"k": 10, "identity_tolerance": 1e-3},
      "declared_basis": BASIS,
    }
out = Path("/workspace/chief/bridge")
out.mkdir(exist_ok=True)
c4 = config("google/gemma-3-4b-it", "4b", "/workspace/chief/captures/4b/checkpoint-hashes.json", "/workspace/chief/out/lens4b-f32/admitted-maps.json", "google/gemma-scope-2-4b-it", "resid_post_all/layer_17_width_16k_l0_small", 2)
c12 = config("google/gemma-3-12b-it", "12b", "/workspace/chief/captures/12b/checkpoint-hashes.json", "/workspace/chief/out/lens12b-f32-remerged/admitted-maps.json", "google/gemma-scope-2-12b-it", "resid_post_all/layer_23_width_16k_l0_small", 3)
for name, c in (("config-4b.json", c4), ("config-12b.json", c12)):
    p = out / name
    if p.exists(): sys.exit(f"refused: {p} exists")
    p.write_text(json.dumps(c, indent=1) + "\n"); print("wrote", p, "lens", c["lens_sha256"][:12], "ckpt files", len(c["checkpoint_sha256"]))
