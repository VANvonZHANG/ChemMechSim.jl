"""Figure 1 — species time series for the frozen-photolysis atmospheric box.

Core conclusion: with photolysis frozen at solar zenith angle chi0 = 0, the
box relaxes toward a photochemical steady state — O3 declines while the
radical pool builds from zero — and NO3 never accumulates because a
perpetual day has no night chemistry.

Panel a: stable species (O3, NO, NO2, CH4) as mixing ratios, ppbv, log axis.
Panel b: radicals (OH, HO2) as number densities, molec/cm^3, log axis.
NO3 is deliberately NOT drawn as a line: on the shared molec/cm^3 axis it
sits ~7 decades below OH, so plotting it would stretch the log axis over
empty decades. It is reported as an explicit annotation instead (its
suppression is the frozen-day limitation made visible).

Run:  python3 examples/atmospheric/tools/figures/fig1_series.py
"""

import sys

import numpy as np
import pandas as pd

try:
    from . import style  # package execution
except ImportError:
    import style  # direct script execution


# --- Unit conversions (mol/m^3 concentrations in series.csv) ------------------
def to_ppbv(c):
    """mol/m^3 -> volume mixing ratio in ppbv (c / 41.516 * 1e9)."""
    return c / 41.516 * 1e9


def to_molec_cm3(c):
    """mol/m^3 -> number density in molecules/cm^3."""
    return c * 6.02214076e23 / 1e6


def positive(t, y):
    """Drop non-positive values so log axes skip exact zeros.

    OH, HO2, NO and NO3 start at exactly 0; their curves begin at the first
    saved point after the radicals build up. Guarded: the mask must be a
    leading run — an interior non-positive would silently bridge the polyline
    across the gap (e.g. a solver residual dipping below zero mid-run).
    """
    m = np.asarray(y) > 0
    interior = np.flatnonzero(~m)
    if interior.size and m.any():
        # non-positives must all precede the first positive sample
        first_pos = int(np.flatnonzero(m)[0])
        if interior.max() >= first_pos:
            raise ValueError(
                f"interior non-positive value at index {int(interior[interior >= first_pos][0])} "
                "would be silently bridged on the log axis")
    return np.asarray(t)[m], np.asarray(y)[m]


def decade_limits(values, pad_lo=0.20, pad_hi=0.20, snap=False):
    """Log-axis limits tightened to the data range plus a fractional pad.

    Pads in units of decades (not whole decades), so the axis never spans
    empty decades beyond a small margin around the data. With ``snap`` the
    limits are rounded outward to whole decades — use it when the data spans
    only 2-3 decades, so every labeled major tick lands inside the view.
    """
    lo = np.log10(np.min(values)) - pad_lo
    hi = np.log10(np.max(values)) + pad_hi
    if snap:
        lo, hi = np.floor(lo), np.ceil(hi)
    return 10.0**lo, 10.0**hi


def end_label(ax, x_end, y, text, color, dx=0.05):
    """Direct label at the right end of a line (no legend needed).

    ``y`` is the label's data coordinate (the series' final value, possibly
    nudged to stagger crowded labels). Text sits just outside the axes in
    the right margin (clip_on=False); bbox_inches='tight' keeps it in the
    exported files.
    """
    ax.text(x_end + dx, y, text, color=color, fontsize=6,
            ha="left", va="center", clip_on=False)


def build_figure(df):
    """Assemble the two-panel figure; returns (fig, ax_a, ax_b)."""
    import matplotlib.pyplot as plt

    t = df["time_s"].to_numpy() / 86400.0  # days

    # (column, label, color, y-position of the end label as a multiple of
    # the series' final value). The NOx multipliers stagger NO and NO2,
    # whose end values sit only 0.08 decades apart on the log axis.
    stable = [
        ("CH4", "CH₄", style.PALETTE["teal"], 1.0),
        ("O3", "O₃", style.PALETTE["blue_main"], 1.0),
        ("NO2", "NO₂", style.PALETTE["red_strong"], 1.8),
        ("NO", "NO", style.PALETTE["neutral_dark"], 0.70),
    ]
    radicals = [
        ("HO2", "HO₂", style.PALETTE["blue_secondary"], 1.1),
        ("OH", "OH", style.PALETTE["violet"], 1.0),
    ]

    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(3.5, 4.5), sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0]},
    )

    all_a = []
    for col, label, color, dy in stable:
        tt, yy = positive(t, to_ppbv(df[col].to_numpy()))
        all_a.append(yy)
        ax_a.plot(tt, yy, color=color, lw=1.1)
        end_label(ax_a, tt[-1], yy[-1] * dy, label, color)
    # Extra room below so the staggered NO label clears the axis bottom.
    lo, hi = decade_limits(np.concatenate(all_a), pad_lo=0.42, pad_hi=0.18)
    ax_a.set_yscale("log")  # set scale BEFORE limits, else autoscale resets them
    ax_a.set_ylim(lo, hi)
    ax_a.set_ylabel("Mixing ratio (ppbv)")
    style.add_panel_label(ax_a, "a", x=-0.13, y=1.06)

    all_b = []
    for col, label, color, dy in radicals:
        tt, yy = positive(t, to_molec_cm3(df[col].to_numpy()))
        all_b.append(yy)
        ax_b.plot(tt, yy, color=color, lw=1.1)
        end_label(ax_b, tt[-1], yy[-1] * dy, label, color)
    # Radicals span < 3 decades: snap to whole decades so 10^6..10^9 all
    # carry labeled major ticks inside the view.
    lo_b, hi_b = decade_limits(np.concatenate(all_b), snap=True)
    ax_b.set_yscale("log")  # scale before limits (autoscale-order gotcha)
    ax_b.set_ylim(lo_b, hi_b)
    ax_b.set_ylabel("Number density (molec cm⁻³)")
    ax_b.set_xlabel("Time (days)")
    ax_b.set_xlim(0.0, 3.0)
    ax_b.set_xticks([0, 1, 2, 3])
    style.add_panel_label(ax_b, "b", x=-0.13, y=1.06)

    # NO3: annotated, not plotted (7 decades below OH on the same axis).
    no3 = to_molec_cm3(df["NO3"].to_numpy())
    no3_max, no3_end = no3.max(), no3[-1]
    ax_b.text(
        0.03, 0.52,
        "NO₃ not plotted: it peaks at\n"
        f"{no3_max:.1f} and ends at {no3_end:.2f} molec cm⁻³\n"
        "— effectively zero, because the box\n"
        "has no night. Radicals build from 0.",
        transform=ax_b.transAxes, fontsize=5.5,
        color=style.PALETTE["neutral_dark"], va="center", ha="left",
    )

    # Figure-level caption: the frozen-photolysis caveat must travel with
    # the figure. No diurnal wording beyond denying the cycle.
    caption = (
        "Photolysis rates are frozen at χ₀ = 0: the box is a perpetual day and no diurnal\n"
        "cycle is represented. OH and HO₂ build from zero initial values; NO₃ stays below\n"
        "3 molec cm⁻³ (0.04 at t = 3 d) because nighttime NO₃ accumulation never occurs."
    )
    fig.text(0.01, 0.005, caption, fontsize=5.5,
             color=style.PALETTE["neutral_dark"], va="bottom", ha="left")

    fig.subplots_adjust(hspace=0.34, bottom=0.14, left=0.17, right=0.85)
    return fig, ax_a, ax_b


def main():
    style.apply_style()
    df = pd.read_csv(style.DATA / "series.csv")
    fig, _, _ = build_figure(df)
    style.save(fig, "fig1_series")

    # --- Physics QA: print the end states the figure is eyeballed against -----
    print("\nQA (start = first row, end = t = 3 d):")
    for col in ("O3", "NO", "NO2", "CH4"):
        v = to_ppbv(df[col].to_numpy())
        print(f"  {col:>3}: {v[0]:10.3f} -> {v[-1]:10.3f} ppbv")
    for col in ("OH", "HO2", "NO3"):
        v = to_molec_cm3(df[col].to_numpy())
        print(f"  {col:>3}: {v[0]:10.3e} -> {v[-1]:10.3e} molec/cm^3"
              f"  (peak {v.max():.3e})")


if __name__ == "__main__":
    sys.exit(main())
