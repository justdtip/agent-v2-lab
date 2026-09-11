"""Position/length identity alone cannot identify a prompt's measured pairing."""

import copy
import hashlib
import json

import numpy as np
import pytest

from local_llm_lab.pipeline.lens_fitting.agentic_compare import validate_cell_pairings


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


@pytest.fixture
def paired(monkeypatch):
    from local_llm_lab.probes import sae_bridge

    monkeypatch.setattr(sae_bridge, "_supplied_pairing_table", lambda t: t)
    identity = {"positions": [2], "context_tokens": 4}
    cells = [
        dict(
            cell=dict(row=i, position="P_act", token_index=2, context_tokens=4),
            input_ids=[1, i + 2, 3, 4],
            prompt_identity={"task_id": str(i)},
            prompt_sha256=digest(str(i)),
            identity=identity,
            capture_forward={"output_attentions": False, "logits_to_keep": [1, 2]},
            residuals={24: np.array([i, 1], dtype=np.float32)},
        )
        for i in range(2)
    ]
    provenance = {
        k: k
        for k in (
            "lens_sha256",
            "nu_sha256",
            "corpus_sha256",
            "positions_sha256",
            "files_sha256",
            "checkpoint_sha256",
            "lens_sidecar_sha256",
        )
    }
    provenance["fit_widths"] = [{"forward_batch": 1}]
    metadata = {"model": {"base": "tiny"}, "fit_widths": provenance["fit_widths"]}
    registrations = [dict(pair=identity, relative=0.0, basis="fixture") for c in cells]
    measurements = [
        dict(
            cell=c["cell"],
            capture_forward=c["capture_forward"],
            prompt_identity=c["prompt_identity"],
            prompt_sha256=c["prompt_sha256"],
            input_ids_sha256=digest(c["input_ids"]),
            capture_residual_sha256=hashlib.sha256(c["residuals"][24].tobytes()).hexdigest(),
            chunk_terms=[{"forward_batch": 1, "relative": 0.0}],
        )
        for c in cells
    ]
    table = {
        "tiny": {"measured_pairings": {"24": registrations}},
        "_provenance": dict(
            provenance,
            measurements={"24": measurements},
            source_commit="a" * 40,
            implementation_sha256={"producer.py": "b" * 64},
        ),
    }
    return table, cells, metadata, provenance


def test_distinct_rows_same_position_identity_pass(paired):
    validate_cell_pairings(*paired, [24], 0.01)


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_row",
        "wrong_ids",
        "wrong_residual",
        "wrong_prompt",
        "missing_receipts",
        "duplicate",
        "wrong_terms",
        "wrong_capture",
    ],
)
def test_wrong_cell_or_missing_evidence_refuses(paired, mutation):
    table, cells, metadata, provenance = paired
    table = copy.deepcopy(table)
    m = table["_provenance"]["measurements"]["24"]
    if mutation == "wrong_row":
        m[0]["cell"]["row"] = 9
    elif mutation == "wrong_ids":
        m[0]["input_ids_sha256"] = "x"
    elif mutation == "wrong_residual":
        m[0]["capture_residual_sha256"] = "x"
    elif mutation == "wrong_prompt":
        m[0]["prompt_sha256"] = "x"
    elif mutation == "missing_receipts":
        table.pop("_provenance")
    elif mutation == "duplicate":
        m[1] = copy.deepcopy(m[0])
    elif mutation == "wrong_terms":
        m[0]["chunk_terms"][0]["relative"] = 0.5
    elif mutation == "wrong_capture":
        table["_provenance"]["files_sha256"] = "wrong"
    with pytest.raises(ValueError):
        validate_cell_pairings(table, cells, metadata, provenance, [24], 0.01)
