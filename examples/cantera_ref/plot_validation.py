#!/usr/bin/env python3
"""Publication-grade ChemMechSim vs Cantera validation figures.

Reads CSVs (time_s, T_K, X_CH4, X_O2, X_CO2, X_OH, X_H2O) from both
ChemMechSim (Julia) and Cantera (Python ref), produces three figures:

  Fig 1: GRI30  — 3×2 panels (T + 5 species), Cantera line + CMS hollow markers
  Fig 2: FFCM2  — same layout
  Fig 3: combined — same 6 panels, both mechanisms overlaid; double-encoded
          legend (color = mechanism, line/marker = solver).

Each panel: Cantera = solid line, ChemMechSim = hollow circles placed at equal
arc length (sparse on plateaus, denser on the ignition front).
All panels equal size. Publication rcParams (7pt, compact, no suptitle).

Quantitative errors (Δt_ign, max ΔX per species) are written to
validation_errors.txt rather than annotated inside the panels.

Usage:
  python3 examples/cantera_ref/plot_validation.py
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Publication rcParams (compact, journal-grade) ─────────────────────
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.linewidth": 0.6,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.frameon": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.0,
    "lines.markersize": 2.0,
    "figure.constrained_layout.w_pad": 0.03,
    "figure.constrained_layout.h_pad": 0.03,
    "figure.constrained_layout.wspace": 0.02,
    "figure.constrained_layout.hspace": 0.02,
})

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PANELS = [
    ("T_K",   r"$T$ (K)"),
    ("X_CH4", r"$X_{\mathrm{CH_4}}$"),
    ("X_O2",  r"$X_{\mathrm{O_2}}$"),
    ("X_CO2", r"$X_{\mathrm{CO_2}}$"),
    ("X_OH",  r"$X_{\mathrm{OH}}$"),
    ("X_H2O", r"$X_{\mathrm{H_2O}}$"),
]


def _read(path):
    d = np.genfromtxt(path, delimiter=",", names=True, deletechars="")
    return {n.strip(): d[n] for n in (d.dtype.names or ())}


def _ign_delay(t, T):
    """Ignition-delay time: t at the steepest temperature rise (argmax |dT/dt|)."""
    dTdt = np.abs(np.diff(T) / np.diff(t))
    return t[int(np.argmax(dTdt)) + 1]


def _ign_diff(t_cms, T_cms, t_ref, T_ref):
    """Relative ignition-delay difference (%) of ChemMechSim vs Cantera."""
    ti = _ign_delay(t_cms, T_cms)
    tj = _ign_delay(t_ref, T_ref)
    return 100.0 * abs(ti - tj) / tj


def _max_dx(a, b):
    return float(np.max(np.abs(a - b)))


def _arc_sample(t, y, n=150, aspect=1.4):
    """Equal-arc-length resampling via differential-geometry arc-length
    reparametrization.

    The plotted curve γ=(t̂,ŷ) lives in data-normalized [0,1]² coordinates
    (each axis scaled to [0,1] by its own range) equipped with the (diagonal)
    metric tensor g = diag(aspect², 1), where `aspect` ≈ panel width/height
    rescales the horizontal axis so arc length matches the perceived display
    length.  Cumulative arc length

        s(τ) = ∫ √(aspect²·dt̂² + dŷ²)

    is accumulated over the dense samples; γ is then resampled at n equally
    spaced values of s by linear interpolation.  Interpolation (rather than
    snapping to the original sample grid) yields *exact* equal arc-length
    spacing, so markers are neither too sparse on the steep ignition front
    nor too dense on flat plateaus."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    span_t = t[-1] - t[0]
    tn = (t - t[0]) / span_t if span_t > 0 else np.zeros_like(t)
    span_y = y.max() - y.min()
    yn = (y - y.min()) / span_y if span_y > 0 else np.zeros_like(y)
    seg = np.sqrt((aspect * np.diff(tn)) ** 2 + np.diff(yn) ** 2)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return t, y
    pos = np.interp(np.linspace(0.0, s[-1], n), s, np.arange(len(s)))
    i0 = np.clip(np.floor(pos).astype(int), 0, len(t) - 2)
    frac = pos - i0
    i1 = i0 + 1
    return t[i0] + frac * (t[i1] - t[i0]), y[i0] + frac * (y[i1] - y[i0])


def _interp(t_new, t_orig, y_orig):
    return np.interp(t_new, t_orig, y_orig)


def _stylish_ax(ax, i, ylabel):
    ax.set_ylabel(ylabel, fontsize=6)
    if i >= 4:
        ax.set_xlabel("time (ms)", fontsize=6)
    else:
        ax.tick_params(labelbottom=False)  # only the bottom row carries x ticks
    ax.tick_params(direction="out", labelsize=5)


def plot_single(cms_path, ref_path, prefix,
                cms_color="#2166AC", ref_color="#333333"):
    cms = _read(cms_path)
    ref = _read(ref_path)
    t_cms = cms["time_s"] * 1e3
    t_ref = ref["time_s"] * 1e3

    fig, axes = plt.subplots(3, 2, figsize=(4.5, 5.0), constrained_layout=True)
    flat = axes.flatten()

    for i, (col, ylabel) in enumerate(PANELS):
        ax = flat[i]
        ax.plot(t_ref, ref[col], color=ref_color, lw=1.1, zorder=2)
        t_m, y_m = _arc_sample(t_cms, cms[col], n=60, aspect=1.4)
        ax.plot(t_m, y_m, "x", color=cms_color, ms=2.4, mew=0.6, zorder=3)
        _stylish_ax(ax, i, ylabel)

    handles = [
        mlines.Line2D([], [], color=ref_color, lw=1.1, label="Cantera"),
        mlines.Line2D([], [], color=cms_color, lw=0, marker="x",
                      ms=3.5, mew=0.8, label="ChemMechSim.jl"),
    ]
    # Legend inside the figure's upper-right corner (the CH4 panel, whose
    # upper-right stays empty because CH4 is consumed over time).
    flat[1].legend(handles=handles, loc="upper right", fontsize=5, frameon=False)
    _save(fig, prefix)
    plt.close(fig)


def plot_combined(specs, prefix):
    datas = [(_read(c), _read(r), lbl, col) for c, r, lbl, col in specs]

    fig, axes = plt.subplots(3, 2, figsize=(4.5, 5.0), constrained_layout=True)
    flat = axes.flatten()

    for i, (col, ylabel) in enumerate(PANELS):
        ax = flat[i]
        for cms_d, ref_d, _lbl, col_hex in datas:
            t_ref = ref_d["time_s"] * 1e3
            t_cms = cms_d["time_s"] * 1e3
            # color encodes mechanism; line vs hollow-circle encodes solver
            ax.plot(t_ref, ref_d[col], color=col_hex, lw=1.0, alpha=0.9)
            t_m, y_m = _arc_sample(t_cms, cms_d[col], n=50, aspect=1.4)
            ax.plot(t_m, y_m, "x", color=col_hex, ms=2.0, mew=0.5)
        _stylish_ax(ax, i, ylabel)

    # Two separated, framed legends in the figure's upper-right corner (over
    # the CH4 panel, whose upper-right stays empty): "mechanism" (color
    # swatches) stacked above "solver" (gray line / ×).  Bold titles, thin
    # frames, both left-edge aligned so the boxes line up neatly.
    def _boxed(handles, title, y):
        leg = fig.legend(handles=handles, loc="upper left",
                         bbox_to_anchor=(0.735, y), title=title,
                         title_fontsize=5, fontsize=5, handlelength=1.5,
                         labelspacing=0.3, borderpad=0.4, frameon=True)
        leg.get_title().set_fontweight("bold")
        fr = leg.get_frame()
        fr.set_edgecolor("#666666")
        fr.set_linewidth(0.6)
        fr.set_facecolor("white")
        box = getattr(leg, "_legend_box", None)
        if box is not None:
            box.align = "left"  # title + entries flush left inside the frame
        return leg

    mech_handles = [mpatches.Patch(facecolor=c, edgecolor="none", label=lbl)
                    for _, _, lbl, c in datas]
    _boxed(mech_handles, "mechanism", 0.96)
    solver_handles = [
        mlines.Line2D([], [], color="#555555", lw=1.1, label="Cantera"),
        mlines.Line2D([], [], color="#555555", lw=0, marker="x",
                      ms=3.5, mew=0.8, label="ChemMechSim.jl"),
    ]
    _boxed(solver_handles, "solver", 0.87)
    _save(fig, prefix)
    plt.close(fig)


def _save(fig, prefix):
    for ext in ("svg", "pdf", "png"):
        p = os.path.join(OUT_DIR, f"{prefix}.{ext}")
        fig.savefig(p, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"  saved {p}")


def write_error_table(specs, out_path):
    """Write a compact validation-error table to a text file (for paper insertion).

    One row per mechanism: relative ignition-delay difference (Δt_ign, %) and
    the max absolute mole-fraction difference (max ΔX) for each species, taken
    over the whole trajectory after interpolating Cantera onto the ChemMechSim
    time grid."""
    species = [col[2:] for col, _ in PANELS[1:]]  # CH4, O2, CO2, OH, H2O
    col_w = [11] * len(species)
    head_species = "  ".join(f"maxΔX_{s}".rjust(w) for s, w in zip(species, col_w))

    lines = [
        "Validation errors — ChemMechSim.jl vs Cantera",
        "Conditions: T0 = 1500 K, P0 = 1 atm, phi = 1.0 (CH4-air),",
        "            constant-volume adiabatic, t in [0, 5] ms.",
        "",
        "Δt_ign : relative difference of ignition-delay time "
        "(time of max |dT/dt|), in %.",
        "max ΔX : max absolute mole-fraction difference over the trajectory.",
        "",
        f"{'Mechanism':<9} {'Δt_ign(%)':>10}  {head_species}",
        "-" * (9 + 1 + 10 + 2 + len(head_species)),
    ]

    for cms_path, ref_path, lbl, _ in specs:
        cms, ref = _read(cms_path), _read(ref_path)
        dt = _ign_diff(cms["time_s"], cms["T_K"], ref["time_s"], ref["T_K"])
        dxs = []
        for col, _ in PANELS[1:]:
            yi = _interp(cms["time_s"], ref["time_s"], ref[col])
            dxs.append(_max_dx(cms[col], yi))
        row_species = "  ".join(f"{v:.2e}".rjust(w) for v, w in zip(dxs, col_w))
        lines.append(f"{lbl:<9} {dt:>10.3f}  {row_species}")

    text = "\n".join(lines) + "\n"
    with open(out_path, "w") as fh:
        fh.write(text)
    print(f"  wrote {out_path}")
    print(text)


if __name__ == "__main__":
    ALL = [
        (f"{OUT_DIR}/gri30_cms_species.csv", f"{OUT_DIR}/gri30_ref_species.csv",
         "GRI30", "#2166AC"),
        (f"{OUT_DIR}/ffcm2_cms_species.csv", f"{OUT_DIR}/ffcm2_ref_species.csv",
         "FFCM2", "#D6604D"),
    ]
    print("Fig 1: GRI30")
    plot_single(ALL[0][0], ALL[0][1], "fig_gri30_validation")
    print("Fig 2: FFCM2")
    plot_single(ALL[1][0], ALL[1][1], "fig_ffcm2_validation")
    print("Fig 3: combined")
    plot_combined(ALL, "fig_combined_validation")
    print("Error table")
    write_error_table(ALL, f"{OUT_DIR}/validation_errors.txt")
    print("All figures generated.")
