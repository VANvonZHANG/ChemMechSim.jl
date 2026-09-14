#!/usr/bin/env python3
"""Pipeline-cost decomposition figure (parse / lowering / build_problem / JIT / warm).

Reads output/bench_pipeline.csv (from gri30_benchmark.jl) and produces a stacked
horizontal bar chart with TWO bars per mechanism:
  - coarse bar: build (total) / JIT compile / warm integrate
  - fine bar:   parse / lowering / build_problem / JIT compile / warm integrate
Both bars have the same total height (build total = parse + lowering +
build_problem); the fine bar splits the build segment into its three stages.
Visually shows that JIT compilation dominates and scales with mechanism size.

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

COLORS = {
    "build":         "#4393c3",   # total (coarse bar only)
    "parse":         "#d1e5f0",   # build split: three shades of blue
    "lower":         "#92c5de",
    "build_problem": "#2166ac",
    "jit_compile":   "#b2182b",
    "warm":          "#1b7837",
}
LABELS = {
    "build":         "build (total)",
    "parse":         "  parse",
    "lower":         "  lowering + mtkcompile",
    "build_problem": "  build_problem (codegen)",
    "jit_compile":   "JIT compile",
    "warm":          "warm integrate",
}


def stacked(ax, y, df, stages, height):
    left = np.zeros(len(df))
    for stage in stages:
        vals = df[f"{stage}_s"].values
        ax.barh(y, vals, left=left, height=height, color=COLORS[stage],
                label=LABELS[stage], edgecolor="white", linewidth=0.3)
        left += vals
    return left


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
    off, h = 0.20, 0.34   # bar pair: coarse above (y-off), fine below (y+off)

    fig, ax = plt.subplots(figsize=(4.5, 2.8))

    # coarse bar: build total / JIT / warm; label the JIT seconds (the headline number)
    total = stacked(ax, y - off, df, ("build", "jit_compile", "warm"), h)
    jit = df["jit_compile_s"].values
    for i in range(len(df)):
        ax.text(total[i] - jit[i] / 2, y[i] - off, f"{jit[i]:.0f}s", ha="center",
                va="center", fontsize=5, color="white", fontweight="bold")

    # fine bar: parse / lowering / build_problem / JIT / warm
    stacked(ax, y + off, df, ("parse", "lower", "build_problem", "jit_compile", "warm"), h)

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
