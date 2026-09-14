#!/usr/bin/env python3
"""Pipeline-cost decomposition figure, two panels.

Reads output/bench_pipeline.csv (from gri30_benchmark.jl):
  (a) stacked horizontal bars, linear scale — build (total) / JIT compile / warm
      integrate per mechanism; shows JIT visually dominating at scale;
  (b) per-stage grouped bars, log10 scale — parse / lowering / build_problem /
      JIT compile / warm, one bar per mechanism per stage. The within-mechanism
      dynamic range (0.09 s parse .. 680 s JIT) and the JIT→warm cliff are both
      only readable on the log panel, while (a) keeps the part-of-whole story.

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
STAGE_COLORS = {"build": "#4393c3", "jit_compile": "#b2182b", "warm": "#1b7837"}
STAGE_LABELS = {"build": "build (total)", "jit_compile": "JIT compile", "warm": "warm integrate"}
STAGES_B = ("parse", "lower", "build_problem", "jit_compile", "warm")
STAGES_B_LABELS = ("parse", "lowering", "build\nproblem", "JIT", "warm")
# mechanism identity for panel (b): purple ramp, distinct from panel (a) stage colors
MECH_COLORS = {"gri30": "#c2a5cf", "ffcm2": "#9970ab", "aramco": "#762a83"}


def panel_a(ax, df):
    mechs = df["mech"].tolist()
    y = np.arange(len(mechs))
    left = np.zeros(len(df))
    for stage in ("build", "jit_compile", "warm"):
        vals = df[f"{stage}_s"].values
        ax.barh(y, vals, left=left, height=0.5, color=STAGE_COLORS[stage],
                label=STAGE_LABELS[stage], edgecolor="white", linewidth=0.3)
        left += vals
    # JIT seconds inside the red segment, only when the segment is wide enough
    xmax = left.max()
    jit = df["jit_compile_s"].values
    for i in range(len(df)):
        if jit[i] > 0.15 * xmax:
            ax.text(left[i] - jit[i] / 2, y[i], f"{jit[i]:.0f}s", ha="center",
                    va="center", fontsize=5, color="white", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{MECH_LABEL[m]}\n({r} sp)" for m, r in zip(mechs, df["n_species"])])
    ax.set_xlabel("time (s)")
    ax.set_title("(a) stacked, linear", loc="left", fontsize=7)
    ax.legend(fontsize=5, loc="upper right")
    ax.invert_yaxis()


def panel_b(ax, df):
    mechs = df["mech"].tolist()
    x = np.arange(len(STAGES_B))
    w = 0.26
    for k, m in enumerate(mechs):
        vals = [df.loc[df["mech"] == m, f"{s}_s"].iloc[0] for s in STAGES_B]
        pos = x + (k - 1) * w
        ax.bar(pos, vals, width=w, color=MECH_COLORS[m], label=MECH_LABEL[m],
               edgecolor="white", linewidth=0.3)
        for xp, v in zip(pos, vals):
            ax.text(xp, v * 1.25, f"{v:g}", ha="center", va="bottom", fontsize=4.5)
    ax.set_yscale("log")
    ax.set_ylim(0.05, 2000)
    ax.set_yticks([0.1, 1, 10, 100, 1000])
    ax.set_xticks(x)
    ax.set_xticklabels(STAGES_B_LABELS)
    ax.set_ylabel("time (s, log scale)")
    ax.set_title("(b) per stage, log scale", loc="left", fontsize=7)
    ax.legend(fontsize=5, loc="upper left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    d = args.out_dir
    csv = os.path.join(d, "bench_pipeline.csv")
    if not os.path.exists(csv):
        raise SystemExit(f"no {csv} — run gri30_benchmark.jl first")
    df = pd.read_csv(csv).sort_values("n_states")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.6),
                                     gridspec_kw={"width_ratios": [1.1, 1.0]})
    panel_a(ax_a, df)
    panel_b(ax_b, df)
    fig.suptitle("Pipeline cost: JIT compilation dominates at scale",
                 fontsize=7, fontweight="bold", x=0.1, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    for ext in ("svg", "pdf", "png"):
        p = os.path.join(d, f"fig_pipeline.{ext}")
        fig.savefig(p, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"  saved {p}")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
