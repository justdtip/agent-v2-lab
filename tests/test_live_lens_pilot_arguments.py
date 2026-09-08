"""The live-lens pilot's argument surface, and the property that makes its planning off-box.

No model and no model library: every case here runs the script as a subprocess in its plan-only
path, which the last assertion pins as importing no MLX at all. That property is why the plan can
be built beside another seat's run, and it is easy to lose to one convenient import.
"""

from __future__ import annotations

import json
import sys

from local_llm_lab import spawn
from local_llm_lab.project import PROJECT_ROOT

_SCRIPT = PROJECT_ROOT / "scripts/live_lens_pilot.py"
#: Never opened: the plan-only path returns before the lens is loaded, which is the
#: property these tests rely on and the reason planning needs no artifact on disk.
_LENS = PROJECT_ROOT / "models/jlens/absent-by-design.npz"
_TIMEOUT = 120.0


def _run(*arguments: str, env: dict[str, str] | None = None):
    environment = {**_child_env(), **(env or {})}
    return spawn.run(
        [sys.executable, str(_SCRIPT), *arguments],
        env=environment,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
    )


def _child_env() -> dict[str, str]:
    import os

    environment = dict(os.environ)
    source = str(PROJECT_ROOT / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source if not existing else f"{source}{os.pathsep}{existing}"
    return environment


def test_the_lens_and_its_digest_must_be_named_because_two_lenses_are_on_disk(tmp_path) -> None:
    """They were module constants pinned to Qwen, so `--model` moved nothing.

    That was also the only thing preventing the Qwen lens from being loaded onto another model of
    the same width (issue 99). A default here would put the accident back.
    """
    completed = _run("--out", str(tmp_path / "a"), "--plan-only")

    assert completed.returncode != 0
    assert "--lens" in completed.stderr and "--lens-sha256" in completed.stderr


def test_the_registry_defaults_to_the_model_named_and_cannot_name_another(tmp_path) -> None:
    """`--registry` follows `--model`, so a run cannot read one model's band for another's."""
    completed = _run(
        "--out", str(tmp_path / "b"),
        "--plan-only",
        "--lens", str(_LENS),
        "--lens-sha256", "0" * 64,
        "--model", "no-such-model-anywhere",
    )

    assert completed.returncode != 0
    assert "configs/models/no-such-model-anywhere.yaml" in completed.stderr


def test_the_plan_is_built_without_importing_mlx_so_it_can_run_beside_another_seat(
    tmp_path,
) -> None:
    """Planning resolves tasks and renders prompts for token counts, and loads nothing.

    Asserted rather than assumed, because the plan reaches the tokenizer for prompt lengths and
    one convenient import would make this stage take a model slot it does not need.
    """
    out = tmp_path / "c"
    completed = _run(
        "--out", str(out),
        "--plan-only",
        "--lens", str(_LENS),
        "--lens-sha256", "0" * 64,
    )

    assert completed.returncode == 0, completed.stderr
    plan = json.loads((out / "plan.json").read_text())
    assert plan["plan"], "the plan has episodes"
    assert plan["seed"] == 20260902
    events = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
    assert any(event.get("event") == "plan_only" for event in events)


def test_the_pilot_module_scope_reaches_no_model_library() -> None:
    """The import closure, checked directly rather than through the script's behaviour."""
    probe = (
        "import sys, importlib.util as u;"
        f"s=u.spec_from_file_location('pilot', {str(_SCRIPT)!r});"
        "m=u.module_from_spec(s);s.loader.exec_module(m);"
        "print('mlx' in sys.modules or 'mlx_lm' in sys.modules)"
    )
    completed = spawn.run(
        [sys.executable, "-c", probe],
        env=_child_env(),
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False", "importing the pilot must not reach MLX"
