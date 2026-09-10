"""Plot recorded projection errors. No model imports; no acceptance threshold inferred."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
x = json.loads((ROOT / "analysis.json").read_text())
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.spines.top":False,
                     "axes.spines.right":False, "pdf.fonttype":42})
fig, axs = plt.subplots(3,1,figsize=(7.2,10.2))
colors={"float32":"#1465a5", "native":"#b7610e"}
for ax,layer in zip(axs,(1,17,33)):
    for precision in ("float32","native"):
        rows=sorted((r for r in x["summaries"] if r["precision"]==precision and r["layer"]==layer),key=lambda r:r["k"])
        label="float32" if precision=="float32" else "native bf16"
        ax.plot([r["k"] for r in rows],[r["median_relative_error"] for r in rows],
                color=colors[precision], marker="o",markersize=3,label=label+" median")
        ax.plot([r["k"] for r in rows],[r["max_relative_error"] for r in rows],
                color=colors[precision], linestyle=":",alpha=.9,label=label+" worst projection")
    ax.set_yscale("log")
    ax.set_title(f"Layer {layer}",loc="left",fontweight="bold",fontsize=12)
    ax.set_ylabel("Relative error (log scale)")
    ax.set_xlabel("k: larger k means a smaller step (h = h₀ × 2⁻ᵏ)")
    ax.grid(axis="y",alpha=.17)
    ax.set_xticks(sorted({r["k"] for r in x["summaries"] if r["layer"]==layer}))
axs[0].legend(ncol=2,fontsize=8,loc="upper left")
fig.suptitle("Good medians do not resolve every projection",x=.1,ha="left",fontsize=15,fontweight="bold",y=.985)
fig.text(.1,.942,"Recorded device measurements · one row, source position 8 · width one\n18 scalar checks per rung; target positions 8 and 127 summed",fontsize=9,va="top")
fig.text(.1,.015,"Source: D-CRO 15eddfa / 5b0feb7; recomputed by Codex, no new GPU run.\nWorst relative error can reflect a small reference derivative. See the report for absolute errors.\nThese are projection diagnostics, not full-matrix error bounds or a declared acceptance interval.",fontsize=8,va="bottom")
fig.subplots_adjust(top=.88,bottom=.13,left=.15,right=.97,hspace=.58)
fig.savefig(ROOT/'ladder-audit.png',dpi=180)
fig.savefig(ROOT/'ladder-audit.pdf')
print("Saved ladder-audit.png and ladder-audit.pdf")
