"""Focused regressions for the ratified SPEC-004 §1 offline reanalysis remediation."""

from __future__ import annotations

from test_probes import _offline_reanalysis_dataset

from local_llm_lab.probes import state_probe


def test_reanalysis_markdown_applies_r8_and_defines_holm_support() -> None:
    results = state_probe.reanalyse_dataset(
        _offline_reanalysis_dataset(),
        split_seeds=(3, 5),
        bootstrap_resamples=12,
        data_seed=20260902,
        logistic_steps=20,
    )

    markdown = state_probe.render_reanalysis_markdown(results, "fixture")

    assert "## all_rows (reportable for base)" in markdown
    assert "## sft_disjoint (paired adapter comparisons, within difficulty only)" in markdown
    assert "| difficulty | rows | tasks |" in markdown
    assert "| 0 | 200 | 36 |" in markdown
    assert "Holm support means" in markdown
    assert "margin vs position and margin vs surface" in markdown
    assert "| 0 | overall | all eligible cells | 38 | 19 | 0.994" in markdown
    assert "n/a (n=1)" in markdown


def test_r8_suppression_requires_24_test_rows_and_5_cells() -> None:
    eligible = state_probe._within_cell([0.25, 0.25], [0.75], [24], [5])
    too_few_rows = state_probe._within_cell([0.25, 0.25], [0.75], [23], [5])
    too_few_cells = state_probe._within_cell([0.25, 0.25], [0.75], [24], [4])

    assert eligible == {
        "estimate": 0.75,
        "median": 0.25,
        "lower": 0.25,
        "upper": 0.25,
        "n_test": 24,
        "n_cells": 5,
        "eligible": True,
    }
    assert state_probe._format_within_cell(eligible) == "0.750 [0.250, 0.250]"
    assert state_probe._format_within_cell(too_few_rows) == "n/a (n=23)"
    assert state_probe._format_within_cell(too_few_cells) == "n/a (n=24)"


def test_gate_section_uses_report_gate_naming() -> None:
    results = {
        "meta": {
            "deconfounding_preflight": {
                "status": "PASS",
                "difficulty_levels": [0, 1, 2],
                "targets": {},
                "failures": [],
            }
        }
    }

    assert state_probe._gate_section(results)[0] == "## deconfounding gate"
