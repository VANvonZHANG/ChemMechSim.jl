#!/usr/bin/env python3
"""Plot the mechanism × linear-solver benchmark matrix.

Reads the CSVs written by bench_matrix.jl (examples/perf/output/) and produces:
  Fig 1 (fig_bench_e2e):   end-to-end FBDF solve time vs mechanism size (N states),
                           one line per linear solver; median marker + IQR error bars.
  Fig 2 (fig_bench_micro): standalone linear-solve per-call cost (factorize+solve on
                           W = αI − J) vs N states, one line per linear solver.
  stdout / bench_accuracy_readable.txt: trajectory accuracy (Δt_ign, max ΔT) vs umfpack.

Usage:  python3 examples/perf/plot_bench.py [--out-dir examples/perf/output]
Requires pandas + matplotlib (Cantera not needed). NaN/CRASH rows are dropped; a solver
that fails at some size just shows a gap in its line.
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# publication rcParams (compact style, matches validation/plot_validation.py)
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.linewidth": 0.6, "axes.spines.right": False, "axes.spines.top": False,
    "legend.frameon": False, "lines.linewidth": 1.0, "lines.markersize": 4.0,
})

COLORS  = {"klu": "#1b7837", "umfpack": "#2166ac", "sparspak": "#5aae61",
           "mumps": "#762a83", "pardiso": "#ff7f00"}
MARKERS = {"klu": "o", "umfpack": "s", "sparspak": "D", "mumps": "P", "pardiso": "X"}
LABELS  = {"klu": "KLU", "umfpack": "UMFPACK", "sparspak": "Sparspak",
           "mumps": "MUMPS", "pardiso": "Pardiso"}


def _save(fig, prefix, out_dir):
    for ext in ("svg", "pdf", "png"):
        p = os.path.join(out_dir, f"{prefix}.{ext}")
        fig.savefig(p, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"  saved {p}")


def _style(ax, xlabel, ylabel):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(fontsize=6, loc="best")
    ax.grid(which="both", linewidth=0.3, alpha=0.5)


def plot_e2e(df, out_dir):
    """median end-to-end (warm) solve time vs N states, with IQR error bars."""
    ns = df.groupby("mech")["n_states"].first()
    g = (df.groupby(["mech", "linsolve"])["wall_s"]
           .agg(["median", "count", lambda s: np.nanpercentile(s, 25), lambda s: np.nanpercentile(s, 75)])
           .reset_index())
    g.columns = ["mech", "linsolve", "median", "count", "q1", "q3"]
    g["n_states"] = g["mech"].map(ns)
    g = g.sort_values("n_states")
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for ls, sub in g.groupby("linsolve"):
        sub = sub.sort_values("n_states")
        lo = (sub["median"] - sub["q1"]).clip(lower=0)
        hi = (sub["q3"] - sub["median"]).clip(lower=0)
        ax.errorbar(sub["n_states"], sub["median"], yerr=[lo, hi],
                    marker=MARKERS.get(ls, "o"), color=COLORS.get(ls, "#333"),
                    linestyle="-", capsize=2, elinewidth=0.6, label=LABELS.get(ls, ls))
    _style(ax, "mechanism size (N states)", "end-to-end solve time (s, warm, median)")
    _save(fig, "fig_bench_e2e_warm", out_dir); plt.close(fig)


def plot_compile(df, out_dir):
    """one-time first-solve cost (compile-dominated) vs N states — the single-shot user cost.
    Mech-level (one row per mech), ~solver-independent — the reaction-sharded Jacobian codegen
    compile, paid once per process. Annotates each point with its value."""
    df = df[df["first_solve_s"].notna()]
    if df.empty:
        print("  (no bench_compile.csv data — skip)"); return
    df = df.sort_values("n_states")
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot(df["n_states"], df["first_solve_s"], marker="o", color="#2166ac",
            linestyle="-", label="first solve (compile)")
    for _, row in df.iterrows():
        ax.annotate(f"{row['first_solve_s']:.0f}s", (row["n_states"], row["first_solve_s"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=5)
    _style(ax, "mechanism size (N states)", "first-solve time (s, one-time compile)")
    _save(fig, "fig_bench_compile", out_dir); plt.close(fig)


def plot_micro(df, out_dir):
    """per-call linear-solve cost (factorize+solve on W=αI−J) vs N states."""
    df = df[df["per_call_s"].notna()].copy()
    if df.empty:
        print("  (no micro-bench data — all skipped)"); return
    df["per_call_ms"] = df["per_call_s"] * 1e3
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for ls, sub in df.groupby("linsolve"):
        sub = sub.sort_values("n_states")
        ax.plot(sub["n_states"], sub["per_call_ms"], marker=MARKERS.get(ls, "o"),
                color=COLORS.get(ls, "#333"), linestyle="-", label=LABELS.get(ls, ls))
    _style(ax, "mechanism size (N states)", "linear-solve per call (ms)")
    _save(fig, "fig_bench_micro", out_dir); plt.close(fig)


def accuracy_table(path, out_dir):
    if not os.path.exists(path):
        print("  (no bench_accuracy.csv)"); return
    df = pd.read_csv(path)
    if df.empty:
        print("  (bench_accuracy.csv empty — need umfpack + ≥1 other solver)"); return
    print("\nTrajectory accuracy vs umfpack reference:")
    with pd.option_context("display.width", 120, "display.max_columns", None):
        print(df.to_string(index=False))
    with open(os.path.join(out_dir, "bench_accuracy_readable.txt"), "w") as f:
        f.write("Trajectory accuracy vs umfpack reference (per bench_matrix.jl)\n\n")
        with pd.option_context("display.width", 120):
            f.write(df.to_string(index=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    d = args.out_dir
    mtx = os.path.join(d, "bench_matrix.csv")
    if not os.path.exists(mtx):
        sys.exit(f"no {mtx} — run `julia --project=. examples/perf/bench_matrix.jl ...` first")
    df = pd.read_csv(mtx)
    df = df[df["wall_s"].notna()]                      # drop CRASH / WARMUP_CRASH rows
    if df.empty:
        sys.exit(f"{mtx} has no successful rows — nothing to plot")
    print(f"warm end-to-end: {len(df)} rows; mechs={sorted(df['mech'].unique())}, "
          f"solvers={sorted(df['linsolve'].unique())}")
    plot_e2e(df, d)
    print("Fig: end-to-end WARM solve time vs N states")
    compile_path = os.path.join(d, "bench_compile.csv")
    if os.path.exists(compile_path):
        print("Fig: one-time compile (first solve) vs N states")
        plot_compile(pd.read_csv(compile_path), d)
    micro_path = os.path.join(d, "bench_linsolve_micro.csv")
    if os.path.exists(micro_path):
        print("Fig: linear-solve per-call cost vs N states (warm)")
        plot_micro(pd.read_csv(micro_path), d)
    accuracy_table(os.path.join(d, "bench_accuracy.csv"), d)
    print("\nDone.")


if __name__ == "__main__":
    main()
