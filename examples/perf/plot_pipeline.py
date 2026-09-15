#!/usr/bin/env python3
"""Pipeline-cost decomposition figure, two panels, no log scale.

Reads output/bench_pipeline.csv (from gri30_benchmark.jl):
  (a) stacked horizontal bars, linear scale with a BROKEN x axis (0–85 s and
      470–530 s) so GRI-Mech 3.0 (~11 s) and FFCM 2.0 (~64 s) keep real width
      next to Aramco 3.0 (~511 s, Julia 1.12 medians); segments: build/JIT/warm;
  (b) build breakdown only, linear scale (0–60 s): each mechanism's build bar
      split into parse / lowering / build_problem.

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

MECH_LABEL = {"gri30": "GRI-Mech 3.0", "ffcm2": "FFCM 2.0", "aramco": "Aramco 3.0"}
A_COLORS = {"build": "#4393c3", "jit_compile": "#b2182b", "warm": "#1b7837"}
A_LABELS = {"build": "build (total)", "jit_compile": "JIT compile", "warm": "warm integrate"}
B_STAGES = ("parse", "lower", "build_problem")
B_COLORS = {"parse": "#d1e5f0", "lower": "#92c5de", "build_problem": "#2166ac"}
B_LABELS = {"parse": "parse", "lower": "lowering + mtkcompile", "build_problem": "build_problem (codegen)"}


def stacked_barh(ax, df, stages, colors, labels, height=0.5):
    y = np.arange(len(df))
    left = np.zeros(len(df))
    for stage in stages:
        vals = df[f"{stage}_s"].values
        ax.barh(y, vals, left=left, height=height, color=colors[stage],
                label=labels[stage], edgecolor="white", linewidth=0.3)
        left += vals
    ax.invert_yaxis()
    return y, left


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
    ylabels = [f"{MECH_LABEL[m]}\n({r} sp)" for m, r in zip(mechs, df["n_species"])]

    fig, (ax1, ax2, axb) = plt.subplots(
        1, 3, figsize=(6.6, 2.5), sharey=True,
        gridspec_kw={"width_ratios": [2.6, 1.1, 2.2], "wspace": 0.06})

    # ---- (a) broken-axis stacked bars: same bars drawn on both axes ----
    y, totals = stacked_barh(ax1, df, ("build", "jit_compile", "warm"), A_COLORS, A_LABELS)
    stacked_barh(ax2, df, ("build", "jit_compile", "warm"), A_COLORS, A_LABELS)
    ax1.set_xlim(0, 85)    # GRI (~11 s) and FFCM (~64 s) live fully here
    ax2.set_xlim(470, 530)  # Aramco JIT tail + warm (~511 s total)
    ax1.set_title("(a) stacked, linear", loc="left", fontsize=7)
    ax1.set_xlabel("time (s)", fontsize=6)

    jit = df["jit_compile_s"].values
    # JIT labels: on the left range where the segment lives inside it
    for i in range(len(df)):
        c = totals[i] - jit[i] / 2
        if 4 < c < 80 and jit[i] > 8:
            ax1.text(c, y[i], f"{jit[i]:.0f}s", ha="center", va="center",
                     fontsize=5, color="white", fontweight="bold")
    # Aramco's JIT tail + total end live on the right range
    for i in range(len(df)):
        if totals[i] > 470:  # only Aramco crosses into the right range
            ax2.text((470 + totals[i] - df["warm_s"].values[i]) / 2, y[i],
                     f"{jit[i]:.0f}s", ha="center", va="center",
                     fontsize=5, color="white", fontweight="bold")
            ax2.text(totals[i] + 4, y[i], f"{totals[i]:.0f}s", ha="left",
                     va="center", fontsize=5, color="dimgray")
        else:
            ax1.text(totals[i] + 1.5, y[i], f"{totals[i]:.1f}s", ha="left",
                     va="center", fontsize=5, color="dimgray")

    # break marks on the junction
    kw = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=9, linestyle="none",
              color="k", mec="k", mew=1, clip_on=False)
    for yy in (0.3, 0.7):
        ax1.plot([1], [yy], transform=ax1.transAxes, **kw)
        ax2.plot([0], [yy], transform=ax2.transAxes, **kw)

    ax1.set_yticks(np.arange(len(mechs)))
    ax1.set_yticklabels(ylabels)
    ax1.legend(fontsize=5, loc="upper right")

    # ---- (b) build breakdown, linear, zoomed to the build range ----
    yb, btot = stacked_barh(axb, df, B_STAGES, B_COLORS, B_LABELS)
    axb.set_xlim(0, 62)
    axb.set_title("(b) build breakdown", loc="left", fontsize=7)
    axb.set_xlabel("build time (s)", fontsize=6)
    for i in range(len(df)):
        axb.text(btot[i] + 1, yb[i], f"{btot[i]:.1f}s", ha="left",
                 va="center", fontsize=5, color="dimgray")
    # label the two dominant segments of the largest mechanism
    low, bpr = df["lower_s"].values, df["build_problem_s"].values
    for i in range(len(df)):
        if low[i] > 6:
            axb.text(low[i] / 2, yb[i], f"{low[i]:.0f}s", ha="center", va="center",
                     fontsize=5, color="black")
        if bpr[i] > 6:
            axb.text(low[i] + bpr[i] / 2, yb[i], f"{bpr[i]:.0f}s", ha="center",
                     va="center", fontsize=5, color="white")
    axb.legend(fontsize=5, loc="upper right")

    for ext in ("svg", "pdf", "png"):
        p = os.path.join(d, f"fig_pipeline.{ext}")
        fig.savefig(p, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"  saved {p}")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
