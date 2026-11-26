#!/usr/bin/env python3
"""Plot pipeline stage times and the composition of model build time.

Reads bench_pipeline.csv and writes fig_pipeline.{svg,pdf,png} in --out-dir.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
import csv


OUT_DIR = Path(__file__).resolve().parent / "output"
MECH_LABEL = {"gri30": "GRI-Mech 3.0", "ffcm2": "FFCM 2.0", "aramco": "Aramco 3.0"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    directory = parser.parse_args().out_dir
    with (directory / "bench_pipeline.csv").open(newline="") as source:
        rows = sorted(csv.DictReader(source), key=lambda row: int(row["n_states"]))
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.linewidth": 0.6, "axes.spines.right": False,
        "axes.spines.top": False, "axes.spines.left": False, "legend.frameon": False,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig, (time, build) = plt.subplots(
        1, 2, sharey=True, figsize=(183 / 25.4, 65 / 25.4),
        gridspec_kw={"width_ratios": [1.5, 1]},
    )
    fig.subplots_adjust(left=0.135, right=0.98, bottom=0.20, top=0.72, wspace=0.14)
    y = np.arange(len(rows))
    time.set_yticks(y, [f"{MECH_LABEL[row['mech']]}\n{row['n_species']} species"
                        for row in rows])
    time.set_ylim(len(rows) - 0.54, -0.54)
    time.set_xscale("log")
    stages = [
        ("warm_s", "Warm solve", "#64707b", "o", -0.20),
        ("build_s", "Build", "#447598", "s", 0),
        ("jit_compile_s", "JIT estimate", "#a1532d", "D", 0.20),
    ]
    values = np.array([[float(row[column]) for row in rows] for column, *_ in stages])
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("The logarithmic time axis requires finite, positive stage times")
    time.set_xlim(values.min() / 1.5, values.max() * 3.2)
    time.xaxis.set_major_locator(LogLocator(base=10, numticks=8))
    time.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:g}"))
    time.xaxis.set_minor_formatter(NullFormatter())
    time.grid(axis="x", which="major", color="#dddddd", linewidth=0.5)
    time.set_axisbelow(True)
    for stage_values, (_, label, color, marker, offset) in zip(values, stages):
        time.plot(stage_values, y + offset, linestyle="none", marker=marker,
                  markersize=4, color=color, label=label)
        for value, row in zip(stage_values, y + offset):
            time.annotate(f"{value:.2f}", (value, row), xytext=(5, 0),
                          textcoords="offset points", va="center", fontsize=6.5)
    time.set_xlabel("Elapsed time (s, log scale)")

    parts = np.array([[float(row[c]) for c in ("parse_s", "lower_s", "build_problem_s")]
                      for row in rows])
    shares = 100 * parts / parts.sum(axis=1, keepdims=True)
    left = np.zeros(len(rows))
    for width, (column, label, color) in zip(shares.T, [
        ("parse_s", "Parse", "#d8dfe2"),
        ("lower_s", "Lowering", "#447598"),
        ("build_problem_s", "Build problem", "#a9c5d5"),
    ]):
        build.barh(y, width, left=left, height=0.42, color=color, label=label)
        for row, start, share in zip(y, left, width):
            if share >= 15:
                build.text(start + share / 2, row, f"{share:.0f}%", ha="center",
                           va="center", color="white" if column == "lower_s" else "#20272b")
        left += width
    build.set_xlim(0, 100)
    build.set_xticks([0, 25, 50, 75, 100])
    build.set_xlabel("Share of build time (%)")
    for ax, letter, title in [(time, "a", "Pipeline stage times"), (build, "b", "Build composition")]:
        # tag at the panel's top-left; the title sits a fixed gap to the right of
        # the tag's own bounding box, so the spacing is the same for panels of
        # different widths
        tag = ax.text(-0.01, 1.36, f"({letter})", transform=ax.transAxes,
                      weight="bold", fontsize=10, va="center")
        fig.canvas.draw()
        box = tag.get_window_extent()
        origin = ax.transAxes.transform((0, 0))
        x_points = (box.x1 - origin[0]) * 72 / fig.dpi + 6
        ax.annotate(title, xy=(0, 1.36), xycoords=ax.transAxes,
                    xytext=(x_points, 0), textcoords="offset points",
                    fontsize=8, va="center")
        ax.tick_params(axis="y", length=0, pad=5)
        ax.tick_params(axis="x", width=0.6, length=3)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.04), ncol=3, fontsize=6.5,
                  handlelength=1, columnspacing=0.9, handletextpad=0.4, borderaxespad=0)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(directory / f"fig_pipeline.{ext}", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
