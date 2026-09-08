"""Export the full depth profile from frozen numeric records; no model runtime."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def main():
    comparison = json.loads((ROOT / "comparison.json").read_text())
    meta = json.loads((ROOT / "lens-metadata.json").read_text())
    rows = comparison["rows"]
    layers = [r["layer"] for r in rows]
    assert layers == list(range(1, 34))
    blue, ink, gray = "#2563a6", "#242a30", "#dfe3e7"
    plt.rcParams.update(
        {
            "font.size": 11,
            "text.color": ink,
            "axes.labelcolor": ink,
            "xtick.color": ink,
            "ytick.color": ink,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
            "svg.hashsalt": "gemma-native-128",
        }
    )
    fig, axes = plt.subplots(3, 1, figsize=(9, 10.2), sharex=True)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.855, bottom=0.16, hspace=0.46)
    fig.text(0.13, 0.96, "Gemma 3 4B: lens comparison by depth", size=18, weight="bold")
    fig.text(
        0.13,
        0.918,
        "Native BF16 regression versus hosted Jacobian maps | 33 nonfinal layers",
        size=11,
    )
    fig.text(
        0.13,
        0.894,
        "Both fits: 128 tokens, below the 1,024-token window. No sliding/global claim.",
        size=10,
    )
    panels = [
        ("Flattened-map cosine", [r["cosine"] for r in rows], (0, 1.06), "Cosine", 1),
        (
            "Matrix difference relative to hosted norm",
            [r["relative_difference_to_hosted"] for r in rows],
            (0, 4.8),
            "||Reg - Hosted||F / ||Hosted||F",
            0,
        ),
        (
            "Regression residual error on ridge-selection windows",
            [100 * meta["per_layer"][str(layer)]["held_out_relative_error"] for layer in layers],
            (0, 6.3),
            "SSE / target squared norm (%)",
            0,
        ),
    ]
    for ax, (title, values, limits, ylabel, baseline) in zip(axes, panels, strict=True):
        ax.plot(layers, values, color=blue, marker="o", markersize=3, linewidth=1.7)
        ax.scatter(
            [34],
            [baseline],
            marker="D",
            facecolors="white",
            edgecolors=ink,
            s=35,
            zorder=4,
            clip_on=False,
        )
        ax.set_title(title, loc="left", size=12, pad=9)
        ax.set_ylabel(ylabel, size=10)
        ax.set_ylim(*limits)
        ax.set_xlim(0.5, 34.8)
        ax.grid(axis="y", color=gray, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 34])
        ax.tick_params(axis="x", labelbottom=True)
    axes[-1].set_xlabel(
        "Residual layer (output of block L; final layer 34 is identity)", labelpad=10
    )
    fig.text(
        0.13,
        0.091,
        "Open diamonds: exact final identity, not fitted evidence. "
        "Lines join adjacent measured layers.",
        size=9,
    )
    fig.text(
        0.13,
        0.069,
        "Regression: 1,608 fit / 402 selection windows. "
        "All 33 layers chose the grid floor (alpha = 0.001).",
        size=9,
    )
    fig.text(
        0.13,
        0.047,
        "Selection error is uncentered and is not token accuracy or independent generalization.",
        size=9,
    )
    fig.text(
        0.13,
        0.025,
        "Sources: comparison.json and lens-metadata.json | GEMMA3-REGRESSION-2026-09-08",
        size=8.5,
    )
    for suffix in ("png", "svg"):
        fig.savefig(
            ROOT / ("depth-profile." + suffix),
            dpi=180,
            facecolor="white",
            metadata={"Creator": "plot_profile.py"},
        )
    svg = ROOT / "depth-profile.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    print("Wrote depth-profile.png and depth-profile.svg")


if __name__ == "__main__":
    main()
