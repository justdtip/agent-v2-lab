from pathlib import Path

import pytest

from local_llm_lab.depth_expansion import ExpansionSpec


def test_evenly_spaced_expansion() -> None:
    assert ExpansionSpec.evenly_spaced(36, 4).insertion_after == (8, 17, 26, 35)
    assert ExpansionSpec.evenly_spaced(36, 6).insertion_after == (5, 11, 17, 23, 29, 35)


def test_expansion_spec_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "expansion.json"
    expected = ExpansionSpec.evenly_spaced(36, 4)
    expected.save(path)
    assert ExpansionSpec.load(path) == expected


@pytest.mark.parametrize(
    ("base_layers", "added_layers"),
    [(0, 1), (3, 0), (3, 4)],
)
def test_invalid_even_spacing(base_layers: int, added_layers: int) -> None:
    with pytest.raises(ValueError):
        ExpansionSpec.evenly_spaced(base_layers, added_layers)
