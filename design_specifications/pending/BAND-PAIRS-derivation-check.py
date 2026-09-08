"""Issue 86's two acceptance derivations, exercised without MLX while the box is held.

`tests/test_jlens.py` imports `mlx.core` at module scope, so nothing in it can run beside
another seat's window. `local_llm_lab.pipeline.jlens` imports no MLX at all, and the derivation
is pure Python over layer indices, so the two acceptance claims can be checked now and the
file's own tests re-run after the end line. The kinds here come from the block-kind rule
(layer L is attention-written exactly when L % 4 == 0), which is what the library's
`DecoderLayer.is_linear` reports and what `test_jlens.py` asserts against for real.
"""

import sys

from local_llm_lab.models import load_model_spec
from local_llm_lab.pipeline import jlens

spec = load_model_spec("qwen35-4b")
ties = spec.probes.partner_tie_breaks
kinds = {layer: ("attention" if layer % 4 == 0 else "linear_attention") for layer in range(1, 33)}
selection = tuple(max(1, round(f * 32)) for f in spec.probes.layer_fractions)

family = jlens.kind_matched_layer_family(
    selection, num_layers=32, period=4, kind_of=kinds.get, tie_breaks=ties
)
recurrent = tuple(
    sorted({layer for pair in spec.probes.live_lens_pairs for layer in pair} - set(range(4, 33, 4)))
)
band = jlens.kind_matched_layer_family(
    recurrent, num_layers=32, period=4, kind_of=kinds.get, tie_breaks=ties
)
fallback = jlens.kind_matched_layer_family(selection, num_layers=32, period=4, kind_of=kinds.get)

checks = [
    ("EXP-003's ten-layer family", family.layers, (5, 11, 12, 16, 17, 20, 21, 27, 28, 32)),
    ("its pairs", family.pairs, {11: 12, 16: 17, 21: 20, 27: 28}),
    ("no unrecorded tie", family.unrecorded_ties, {}),
    ("the band's recurrent members", recurrent, (13, 17, 19, 23, 27)),
    (
        "the band's five pairs, same function",
        tuple(sorted(tuple(sorted(pair)) for pair in band.pairs.items())),
        spec.probes.live_lens_pairs,
    ),
    ("no tie in the band at all", band.unrecorded_ties, {}),
    ("the recorded ruling, one entry", ties, {16: 17}),
    ("without it, the old wrong answer", fallback.pairs[16], 15),
    ("and it is reported", fallback.unrecorded_ties, {16: (15, 17)}),
]
bad = 0
for name, got, want in checks:
    ok = got == want
    bad += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {got!r}" + ("" if ok else f"  expected {want!r}"))
print(f"\nmlx imported: {'mlx' in sys.modules}")
sys.exit(1 if bad else 0)
