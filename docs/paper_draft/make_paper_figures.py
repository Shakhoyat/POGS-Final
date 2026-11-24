"""Paper figures from data_results (G1 dose-response, G1 curvature shares). Same style as docs/report/make_figures.py.

    .venv/bin/python docs/paper_draft/make_paper_figures.py
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "docs/paper_draft/figs"
os.makedirs(OUT, exist_ok=True)
S1, S2, GREY, INK2, GRID = "#2a78d6", "#eb6834", "#8a8d93", "#4e5157", "#e6e5e0"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8, "axes.edgecolor": "#b9b8b2",
    "axes.linewidth": 0.6, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})
G1 = "data_results/g1_generality"


def dose():
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    files = sorted(glob.glob(f"{G1}/dose_seed*.json"))
    for j, p in enumerate(files):
        d = json.load(open(p))
        s = [r["shadow_strength"] for r in d["rows"]]
        mk = ["o", "s", "^"][j % 3]
        ax.plot(s, [r["argmin"]["L"] for r in d["rows"]], color=S2, lw=1.5, marker=mk, ms=4,
                label="release (background vs. empty render)" if j == 0 else None)
        ax.plot(s, [r["argmin"]["bg_rendered"] for r in d["rows"]], color=S1, lw=1.5, marker=mk, ms=4,
                label="background rendered" if j == 0 else None)
    ax.axhline(0, color=GREY, lw=0.8, ls="--")
    ax.axvline(1.0, color=GREY, lw=0.6, ls=":")
    ax.text(1.02, 3.5, "renderer\ndefault", fontsize=6.5, color=INK2, va="bottom")
    ax.set_xlabel("cast-shadow strength $s$ (0 = no shadow)")
    ax.set_ylabel("roll at loss minimum (°)")
    ax.set_ylim(-12.5, 6.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1)
    fig.savefig(f"{OUT}/dose_response.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    dose()
    print("figures ->", OUT)
