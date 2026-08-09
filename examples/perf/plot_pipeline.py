#!/usr/bin/env python3
"""Pipeline-cost decomposition figure (build / JIT compile / warm integrate).

Reads output/bench_pipeline.csv (from gri30_benchmark.jl) and produces a stacked
horizontal bar chart: each mechanism is one bar, segmented into build (lowering),
jit_compile (one-time Julia compilation), and warm (integration). Visually shows
that JIT compilation dominates and scales with mechanism size.

Usage: python3 examples/perf/plot_pipeline.py [--out-dir examples/perf/output]
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.linewidth": 0.6, "axes.spines.right": False, "axes.spines.top": False,
    "legend.frameon": False,
})

COLORS = {"build": "#2166ac", "jit_compile": "#b2182b", "warm": "#1b7837"}
LABELS = {"build": "build (lowering)", "jit_compile": "JIT compile", "warm": "warm integrate"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    d = args.out_dir
    csv = os.path.join(d, "bench_pipeline.csv")
    if not os.path.exists(csv):
        raise SystemExit(f"no {csv} — run gri30_benchmark.jl first")
    df = pd.read_csv(csv).sort_values("n_states")

    mechs = df["mech"].tolist()
    y = np.arange(len(mechs))

    fig, ax = plt.subplots(figsize=(4.5, 2.5))
    left = np.zeros(len(mechs))
    for stage in ("build", "jit_compile", "warm"):
        vals = df[f"{stage}_s"].values
        ax.barh(y, vals, left=left, height=0.5, color=COLORS[stage], label=LABELS[stage],
                edgecolor="white", linewidth=0.3)
        if stage == "jit_compile":
            for i, v in enumerate(vals):
                ax.text(left[i] + v / 2, y[i], f"{v:.0f}s", ha="center", va="center",
                        fontsize=5, color="white", fontweight="bold")
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels([f"{m}\n({r} sp)" for m, r in zip(mechs, df["n_species"])])
    ax.set_xlabel("time (s)")
    ax.set_title("Pipeline cost: JIT compilation dominates at scale", fontsize=7, fontweight="bold")
    ax.legend(fontsize=5, loc="lower right")
    ax.invert_yaxis()

    for ext in ("svg", "pdf", "png"):
        p = os.path.join(d, f"fig_pipeline.{ext}")
        fig.savefig(p, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"  saved {p}")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
