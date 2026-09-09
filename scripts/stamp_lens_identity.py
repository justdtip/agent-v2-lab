"""Stamp a model identity onto a lens converted before the field existed (issue 99).

Writes into the **sidecar**, never the archive. The two hosted lenses in ``models/jlens/`` have
digests pinned in ``scripts/live_lens_pilot.py`` and published in
``research/records/LIVE-LENS-PILOT-2026-09-07/``; rewriting the ``.npz`` to carry the identity
inside it would invalidate every one of those records to fix a defect that never affected them.
So the stamp goes beside the file and is bound to it by the sidecar's own ``npz_sha256``, which
this script verifies against the file before writing anything.

Lenses fitted from here on carry the identity inside the archive instead
(``lens_fitting.artifacts.write_lens``), where the digest covers it. This script is for the
retroactive case only, and it records that the stamp was retroactive and on what evidence.

    python scripts/stamp_lens_identity.py models/jlens/<name>.npz \
        --name qwen35-4b --hf-id mlx-community/Qwen3.5-4B-MLX-4bit --num-layers 32 \
        --evidence "converted from neuronpedia/jacobian-lens <path>, whose repository names it"

Loads no model and reads no weights, so it is safe to run beside anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from local_llm_lab.pipeline.live_lens.instruments import (  # noqa: E402
    LensIdentity,
    file_sha256,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path)
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "the model the lens was fitted on, by its canonical upstream name — not a registry "
            "entry name and not a local path, both of which describe a file rather than a model"
        ),
    )
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument(
        "--evidence",
        required=True,
        help="why this identity is the right one, in a sentence a reader can check",
    )
    args = parser.parse_args(argv)

    npz = args.npz.resolve()
    sidecar = npz.with_suffix(".json")
    if not npz.exists() or not sidecar.exists():
        parser.error(f"both {npz.name} and {sidecar.name} must exist")
    meta = json.loads(sidecar.read_text())
    sha = file_sha256(npz)
    if meta.get("npz_sha256") != sha:
        parser.error(
            f"{sidecar.name} records npz_sha256 {meta.get('npz_sha256')} and {npz.name} hashes "
            f"to {sha}: the sidecar describes a different file"
        )
    if "model" in meta:
        parser.error(f"{sidecar.name} already carries a model identity: {meta['model']}")

    identity = LensIdentity(args.model, args.num_layers)
    meta["model"] = identity.as_dict()
    meta["model_stamp"] = {
        "retroactive": True,
        "stamped_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "evidence": args.evidence,
        "reason": (
            "issue 99: the lens carried no model identity, and two models of one hidden width are "
            "now on this box. Stamped in the sidecar rather than the archive so the published "
            "npz digest stays valid."
        ),
    }
    sidecar.write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({"stamped": str(sidecar), "model": meta["model"], "npz_sha256": sha}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
