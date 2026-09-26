"""Species time-series figures — ONE script, TWO figures (frozen + diurnal runs).

    series_frozen.png  : the frozen run  (photolysis at its defaults = perpetual noon)
    series_diurnal.png : the diurnal run (photolysis driven by the zenith clock)

Both figures share one panel builder — same six panels, same axes, same quantity per
panel — so the only difference is the forcing and the two can be compared panel by
panel. The runs are deliberately NOT overlaid in one figure: each panel shows its own
run only, and panel (a) shows the forcing that run actually applied (read from the
run's own exported `cz` column; frozen is identically 1).

Panels:
  a: the forcing, cos(χ)   — diurnal: the clock; frozen: flat 1 ("perpetual noon")
  b: O₃ alone, LINEAR ppbv — the decline the shared log axis used to hide
  c: CH₄ alone, tight LINEAR ppbv — the small decline + its computed statistics
  d: NO₂ + NO (ppbv, log)  — NOx exhaustion; NO's night values floored (bounds)
  e: OH + HO₂ (molec/cm³, log) — sun-synchronous pulses (diurnal) / steady level
     (frozen); the dashed line is the solver resolution floor
  f: NO₃ alone (molec/cm³, log, OWN range) — below solver resolution in this
     scenario: drawn at the floor as an upper bound, never as physics

The figures carry almost no text on purpose: axes, panel letters, species end-labels,
one short forcing label and the one-word "abstol" tag on the resolution floor. Every
number that used to live in on-figure grey blocks is printed by the QA pass instead —
a figure is for shape, the printout and README are for numbers.

Run:  python3 examples/atmospheric/tools/figures/fig_series.py
"""

import sys

import numpy as np
import pandas as pd

try:
    from . import style
except ImportError:
    import style

NIGHT_EDGES = (21600.0, 64800.0)      # 06:00 / 18:00 in seconds-of-day
# Resolution floor = the runs' flat abstol (1e-12 mol/m^3), asserted at load. Values
# below it are UNRESOLVED: the solver may return anything up to it, so floored parts
# of curves are upper bounds, never physics.
FLOOR_MOLEC_CM3 = 1e-12 * 6.02214076e23 / 1e6   # 602.2 molec/cm^3
FLOOR_PPBV = 1e-12 / 41.516 * 1e9                # 2.41e-5 ppbv


# --- unit conversions & plot helpers (single definitions; fig2 imports none) ---
def to_ppbv(c):
    """mol/m^3 -> volume mixing ratio in ppbv (c / 41.516 * 1e9)."""
    return c / 41.516 * 1e9


def to_molec_cm3(c):
    """mol/m^3 -> number density in molecules/cm^3."""
    return c * 6.02214076e23 / 1e6


def positive(t, y):
    """Drop non-positive values so log axes skip exact zeros.

    OH, HO2, NO and NO3 start at exactly 0; their curves begin at the first saved
    point after the radicals build up. Guarded: the mask must be a leading run — an
    interior non-positive would silently bridge the polyline across the gap.
    """
    m = np.asarray(y) > 0
    interior = np.flatnonzero(~m)
    if interior.size and m.any():
        first_pos = int(np.flatnonzero(m)[0])
        if interior.max() >= first_pos:
            raise ValueError(
                f"interior non-positive value at index {int(interior[interior >= first_pos][0])} "
                "would be silently bridged on the log axis")
    return np.asarray(t)[m], np.asarray(y)[m]


def floored(y, floor):
    """Clip to the resolution floor; returns (values, n_clipped). Flooring — unlike
    dropping — keeps the polyline continuous on the log axis, and the floor is an
    upper bound, not a value: anything at it means 'at or below solver resolution'."""
    y = np.asarray(y, dtype=float)
    return np.maximum(y, floor), int(np.sum(y < floor))


def decade_limits(values, pad_lo=0.20, pad_hi=0.20, snap=False):
    """Log-axis limits tightened to the data range plus a fractional pad (decades).
    With ``snap`` the limits round outward to whole decades — for data spanning only
    2-3 decades, so every labeled major tick lands inside the view."""
    lo = np.log10(np.min(values)) - pad_lo
    hi = np.log10(np.max(values)) + pad_hi
    if snap:
        lo, hi = np.floor(lo), np.ceil(hi)
    return 10.0**lo, 10.0**hi


def end_label(ax, x_end, y, text, color, dx=0.05):
    """Direct label at the right end of a line (no legend needed)."""
    ax.text(x_end + dx, y, text, color=color, fontsize=6,
            ha="left", va="center", clip_on=False)


def shade_nights(ax, t_days):
    """Grey bands over 18:00->06:00, geometric edges (the cz floor makes thresholds
    lie). Applied to every panel of the DIURNAL figure only — the frozen run has
    no nights."""
    for d in range(int(np.floor(t_days.max())) + 1):
        ax.axvspan(d + NIGHT_EDGES[1] / 86400.0,      # 18:00 of day d
                   d + 1 + NIGHT_EDGES[0] / 86400.0,   # 06:00 of day d+1
                   color=style.PALETTE["neutral_dark"], alpha=0.08, lw=0, zorder=0)


def load_run(mode):
    """Read one run's series + meta; assert the flags the figure's framing assumes."""
    run_dir = style.DIURNAL if mode == "diurnal" else style.FROZEN
    df = pd.read_csv(run_dir / "series.csv")
    meta = {}
    for line in (run_dir / "run_meta.txt").read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            meta[key.strip()] = value.strip()
    if meta.get("mode") != mode:
        raise ValueError(f"{run_dir}/run_meta.txt says mode={meta.get('mode')!r} — wrong run?")
    if meta.get("abstol") != "1e-12":
        raise ValueError(f"run_meta abstol={meta.get('abstol')!r} — the resolution floor "
                         "and below-resolution framing assume 1e-12")
    # Cross-language check (the standing contract): the exported cz column must equal
    # the clock recomputed here — or be identically 1 for the frozen run.
    tod = df["time_s"].to_numpy() % 86400.0
    if mode == "diurnal":
        chi = np.minimum(np.deg2rad(89.5), np.abs(2 * np.pi * tod / 86400.0 - np.pi))
        cz = np.maximum(0.0, np.cos(chi))
    else:
        cz = np.ones_like(tod)
    err = np.max(np.abs(cz - df["cz"].to_numpy()))
    if err > 1e-12:
        raise ValueError(f"cz column disagrees with formula recomputation (max {err:.2e})")
    return df, float(meta["span_days"])


# --- the shared six-panel builder ------------------------------------------------------------
def build_figure(df, span_days, mode):
    """Assemble the six-panel figure for one run; returns (fig, axes)."""
    import matplotlib.pyplot as plt

    t = df["time_s"].to_numpy() / 86400.0
    cz = df["cz"].to_numpy()
    diurnal = (mode == "diurnal")

    fig, axes = plt.subplots(
        6, 1, figsize=(3.5, 8.6), sharex=True,
        gridspec_kw={"height_ratios": [0.5, 1.0, 1.0, 0.9, 1.0, 0.85]})

    # (a) the forcing as the run applied it: cz(t) from the run's own export
    ax = axes[0]
    ax.plot(t, cz, color=style.PALETTE["gold"], lw=1.0)
    if diurnal:
        shade_nights(ax, t)
        # the night floor is physics-of-the-port (the 89.5° clamp): one short label
        ax.annotate("cos 89.5°", xy=(t[-1] / 2, 0.0087), xytext=(t[-1] / 2, 0.35),
                    fontsize=5.5, color=style.PALETTE["neutral_dark"], ha="center",
                    arrowprops={"arrowstyle": "-", "color": style.PALETTE["neutral_dark"],
                                "lw": 0.5})
    else:
        ax.text(0.5, 0.5, "perpetual noon", transform=ax.transAxes, fontsize=5.5,
                ha="center", va="center", color=style.PALETTE["neutral_dark"])
    ax.set_ylabel("cos χ")
    ax.set_ylim(0.0, 1.05)
    style.add_panel_label(ax, "a", x=-0.13, y=1.12)

    # (b) O3 alone, LINEAR ppbv — the decline a shared log axis hides
    ax = axes[1]
    o3v = to_ppbv(df["O3"].to_numpy())
    ax.plot(t, o3v, color=style.PALETTE["blue_main"], lw=1.1)
    end_label(ax, t[-1], o3v[-1], "O₃", style.PALETTE["blue_main"])
    ax.set_ylim(o3v.min() - 0.05 * (o3v[0] - o3v.min()), o3v.max() + 1.5)
    ax.set_ylabel("O₃ (ppbv)")
    if diurnal:
        shade_nights(ax, t)
    style.add_panel_label(ax, "b", x=-0.13, y=1.14)

    # (c) CH4 alone, tight LINEAR ppbv — every number in the annotation is computed
    ax = axes[2]
    ch4v = to_ppbv(df["CH4"].to_numpy())
    ax.plot(t, ch4v, color=style.PALETTE["teal"], lw=1.1)
    end_label(ax, t[-1], ch4v[-1], "CH₄", style.PALETTE["teal"])
    pad = 0.25 * (ch4v[0] - ch4v[-1])
    ax.set_ylim(ch4v.min() - pad, ch4v[0] + pad)
    ax.set_ylabel("CH₄ (ppbv)")
    if diurnal:
        shade_nights(ax, t)
    style.add_panel_label(ax, "c", x=-0.13, y=1.14)

    # (d) NOx: NO2 decay (+ NO daylight pulses when there are days and nights)
    ax = axes[3]
    nox = [
        ("NO2", "NO₂", style.PALETTE["red_strong"], 2.2),
        ("NO", "NO", style.PALETTE["neutral_dark"], 0.55),
    ]
    all_d = []
    for col, label, color, dy in nox:
        # values below the resolution floor are floored: upper bounds, not physics
        yy, _ = floored(to_ppbv(df[col].to_numpy()), FLOOR_PPBV)
        all_d.append(yy)
        ax.plot(t, yy, color=color, lw=1.1)
        end_label(ax, t[-1], yy[-1] * dy, label, color)
    lo, hi = decade_limits(np.concatenate(all_d), pad_lo=0.30, pad_hi=0.30)
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.set_ylabel("NO₂, NO (ppbv)")
    if diurnal:
        shade_nights(ax, t)
    style.add_panel_label(ax, "d", x=-0.13, y=1.14)

    # (e) radicals, molec/cm^3, log — floored at the resolution line
    ax = axes[4]
    radicals = [
        ("HO2", "HO₂", style.PALETTE["blue_secondary"], 1.1),
        ("OH", "OH", style.PALETTE["violet"], 1.0),
    ]
    all_c = []
    for col, label, color, dy in radicals:
        yy, _ = floored(to_molec_cm3(df[col].to_numpy()), FLOOR_MOLEC_CM3)
        all_c.append(yy)
        ax.plot(t, yy, color=color, lw=1.1)
        end_label(ax, t[-1], yy[-1] * dy, label, color)
    lo_c, hi_c = decade_limits(np.concatenate(all_c), snap=True)
    ax.set_yscale("log")
    ax.set_ylim(lo_c, hi_c)
    ax.set_ylabel("Number density\n(molec cm⁻³)")
    ax.axhline(FLOOR_MOLEC_CM3, color=style.PALETTE["neutral_mid"], lw=0.5, ls=(0, (2, 2)))
    ax.text(0.99, 0.04, "abstol", transform=ax.transAxes, fontsize=5, ha="right",
            va="bottom", color=style.PALETTE["neutral_dark"])
    if diurnal:
        shade_nights(ax, t)
    style.add_panel_label(ax, "e", x=-0.13, y=1.14)

    # (f) NO3 alone, log, on ITS OWN range (not the radicals') — below resolution in
    # this scenario either way: floored at the line as an upper bound, with the
    # NOx-starvation framing (a single 0.1-ppb NO2 pulse, HNO3 terminal, no source).
    ax = axes[5]
    no3_raw = to_molec_cm3(df["NO3"].to_numpy())
    no3_max = no3_raw.max()
    no3_plot, _ = floored(no3_raw, FLOOR_MOLEC_CM3)
    ax.plot(t, no3_plot, color=style.PALETTE["red_strong"], lw=1.1)
    end_label(ax, t[-1], no3_plot[-1] * 1.6, "NO₃", style.PALETTE["red_strong"])
    ax.axhline(FLOOR_MOLEC_CM3, color=style.PALETTE["neutral_mid"], lw=0.5, ls=(0, (2, 2)))
    ax.text(0.99, 0.04, "abstol", transform=ax.transAxes, fontsize=5, ha="right",
            va="bottom", color=style.PALETTE["neutral_dark"])
    lo_d, hi_d = decade_limits(no3_plot, pad_lo=0.30, pad_hi=0.30, snap=True)
    ax.set_yscale("log")
    ax.set_ylim(lo_d, hi_d)
    ax.set_ylabel("NO₃\n(molec cm⁻³)")
    ax.set_xlabel("Time (days)")
    ax.set_xlim(0.0, span_days)
    ax.set_xticks(range(0, int(span_days) + 1))
    if diurnal:
        shade_nights(ax, t)
    style.add_panel_label(ax, "f", x=-0.13, y=1.14)

    fig.subplots_adjust(hspace=0.36, bottom=0.085, left=0.17, right=0.86)
    return fig, axes


def qa_print(df, span_days, mode):
    """Print the numbers the figure's annotations are computed from — the same
    arithmetic, surfaced for eyeballing."""
    t_d = df["time_s"].to_numpy() / 86400.0
    diurnal = (mode == "diurnal")
    print(f"\nQA [{mode}] (start = first row, end = t = {span_days:.0f} d):")
    for col in ("O3", "NO2", "CH4"):
        v = to_ppbv(df[col].to_numpy())
        print(f"  {col:>3}: {v[0]:10.3f} -> {v[-1]:10.3f} ppbv")
    oh = to_molec_cm3(df["OH"].to_numpy())
    no3 = to_molec_cm3(df["NO3"].to_numpy())
    print(f"  OH : peak {oh.max():.3e}  end {oh[-1]:.3e} molec/cm^3")
    print(f"  NO3: peak {no3.max():.3e}  end {no3[-1]:.3e} molec/cm^3"
          + ("  — below resolution (floor 6.02e2): upper bound only"
             if no3.max() < FLOOR_MOLEC_CM3 else ""))
    # CH4 closure: the decline implies a mean OH; compare with the OH trajectory
    ch4 = df["CH4"].to_numpy()
    k_ch4_oh = 6.3e-15                         # cm3/molec/s, IUPAC at 298 K
    span_s = df["time_s"].to_numpy()[-1] - df["time_s"].to_numpy()[0]
    implied = np.log(ch4[0] / ch4[-1]) / span_s / k_ch4_oh
    direct = to_molec_cm3(df["OH"].to_numpy()).mean()
    print(f"  CH4: −{(ch4[0] - ch4[-1]) / ch4[0] * 100:.3f} % over {span_days:.0f} d; implied ⟨OH⟩ "
          f"{implied:.2e} vs direct {direct:.2e} molec/cm^3 "
          f"({abs(implied / direct - 1) * 100:.1f}% off — d[CH4]/dt = −k[OH][CH4] closure)")
    if diurnal:
        # per-calendar-day O3 extremes: the cycle amplitude
        o3 = to_ppbv(df["O3"].to_numpy())
        for d in range(int(span_days)):
            m = (t_d >= d) & (t_d < d + 1)
            if m.any():
                print(f"  O3 day {d}: min {o3[m].min():8.3f}  max {o3[m].max():8.3f} ppbv")


def main():
    style.apply_style()
    for mode, outname in (("frozen", "series_frozen"), ("diurnal", "series_diurnal")):
        df, span = load_run(mode)
        fig, _ = build_figure(df, span, mode)
        style.save(fig, outname)
        qa_print(df, span, mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
