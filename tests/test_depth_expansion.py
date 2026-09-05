import json
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
    # The writer's three fields, at the level it writes them: what the move test below moves.
    assert json.loads(path.read_text(encoding="utf-8")).keys() == {
        "insertion_after",
        "initialization",
        "version",
    }


@pytest.mark.parametrize("key", ["insertion_after", "initialization", "version"])
def test_expansion_spec_load_refuses_a_field_that_moved_off_its_level(
    tmp_path: Path, key: str
) -> None:
    """R38 step four (issue #70): the round trip above passes with two of these three gone.

    ``initialization`` and ``version`` were read with the dataclass defaults, and those
    defaults are the only values either field has ever held -- so renaming one left ``load``
    returning a spec equal to the one ``save`` was given, and the round trip stayed green
    while the file it read had stopped saying what the reader reported. ``version`` is the
    one that matters: it exists to mark a recipe this code may not know how to apply.
    """
    path = tmp_path / "expansion.json"
    ExpansionSpec.evenly_spaced(36, 4).save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[f"moved_{key}"] = payload.pop(key)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=key):
        ExpansionSpec.load(path)


@pytest.mark.parametrize(
    ("base_layers", "added_layers"),
    [(0, 1), (3, 0), (3, 4)],
)
def test_invalid_even_spacing(base_layers: int, added_layers: int) -> None:
    with pytest.raises(ValueError):
        ExpansionSpec.evenly_spaced(base_layers, added_layers)


def test_an_unsupported_recipe_version_is_refused_by_name(tmp_path) -> None:
    """#75: requiring ``version`` is not checking it.

    The field's whole job is to say a recipe is NOT the one this code applies, so it is compared
    against ``SUPPORTED_VERSIONS`` rather than merely read. Written through ``save`` and then
    edited, because a spec carrying an unsupported version cannot be constructed -- which is the
    point: the check lives in ``__post_init__``, so ``load`` is covered by construction and a
    direct constructor call cannot bypass what a file cannot.
    """
    from local_llm_lab.depth_expansion import SUPPORTED_VERSIONS

    path = tmp_path / "expansion.json"
    ExpansionSpec(insertion_after=(1, 3)).save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] in SUPPORTED_VERSIONS, "the writer emits a version this code applies"

    payload["version"] = max(SUPPORTED_VERSIONS) + 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported expansion recipe version"):
        ExpansionSpec.load(path)

    with pytest.raises(ValueError, match="unsupported expansion recipe version"):
        ExpansionSpec(insertion_after=(1, 3), version=max(SUPPORTED_VERSIONS) + 1)
