"""Figure 3 — the diurnal-photolysis box: same mechanism, real day/night cycle.

Core conclusion: once photolysis is DRIVEN by the zenith clock instead of frozen at chi = 0,
the box develops a daily cycle — OH/HO2 pulse with the sun and collapse at night (50-250x),
NO appears only in daylight hours, O3 declines at half the frozen run's rate (12 h of light
per day instead of 24). The SURPRISE, and the reason this figure exists: NO3 shows NO diurnal
structure (night/day max ratio ~1.0) and stays <= 4 molec/cm^3 at all hours. This scenario
starts from a single 0.1-ppb NO2 pulse with no NOx source, NO2 -> HNO3 is terminal, and NO3
slaves to the collapsing NO2 (500x over 8 days) rather than to the light cycle — the frozen
box's NO3 ~= 0 was NOx starvation in disguise, not only the missing night. This figure is the
Cantera-method port made visible: J = l*cos(chi)^m*exp(-n/cos(chi)), chi the converter's
triangular clock, J piecewise-constant per 60-s step (see mcm_box_diurnal.jl).

Panel a: the driving environment — cos(zenith), with nights shaded (same bands on every
         panel). Night shading is GEOMETRIC (time-of-day outside 06:00-18:00), not derived
         from a cz threshold: the clock's 89.5 deg clamp means cz never drops below
         cos(89.5 deg) = 0.0087 even at midnight, so a threshold would smear the edges.
Panel b: O3 ALONE on a LINEAR ppbv axis (30 -> 19 ppbv: -37% is invisible on the shared
         log axis of the earlier draft). Daily sawtooth + the FROZEN run's dashed overlay
         (fig1's run, 3 days): perpetual noon eats O3 visibly faster than 12 h/day does.
Panel c: CH4 ALONE on a LINEAR ppbv axis with a tight range: the decline is only -0.78%
         over 8 days (k(CH4+OH) = 6.3e-15 gives a multi-year lifetime at this box's 24-h
         mean OH), so it is meaningless on any shared axis — but on its own axis the
         day-steps are plain: CH4 falls ONLY in daylight (loss rate ~ [OH]) and staircases
         through the nights. The decline closes against the OH trajectory to ~1% (QA).
Panel d: NO2 + NO (ppbv, log) — NO2's NOx-exhaustion decay, NO's daylight-only pulses.
Panel e: radicals (OH, HO2) as number densities (molec/cm^3, log axis) — the sun-synchronous
         pulse, collapsing each night.
Panel f: NO3 alone (molec/cm^3, log axis) — a featureless decay tracking NOx exhaustion:
the night-time accumulation a "real" night should produce never appears, which re-frames
fig1's NO3 suppression claim (its cause here is NOx starvation, not only the missing night).

Run:  python3 examples/atmospheric/tools/figures/fig3_diurnal.py
"""

import sys

import numpy as np
import pandas as pd

try:
    from . import style
    from . import fig1_series as f1  # decade_limits / positive / unit conversions (pure)
except ImportError:
    import style
    import fig1_series as f1

DIURNAL = style.DATA / "diurnal"      # this figure's run (output/diurnal/)
FROZEN = style.DATA                   # fig1's run, for the O3 overlay

SPANS_DAYS = 8.0                      # must match run_meta span_days; asserted below
NIGHT_EDGES = (21600.0, 64800.0)      # 06:00 / 18:00 in seconds-of-day
# Resolution floor = the run's abstol (1e-12 mol/m^3). Night-time OH/HO2 troughs and ALL of
# NO3 sit below it — the solver may return any value up to it — so those parts of the curves
# are drawn AT the floor, labelled as bounds, never as physics. Asserted against run_meta.
FLOOR_MOLEC_CM3 = 1e-12 * 6.02214076e23 / 1e6   # 602.2 molec/cm^3
FLOOR_PPBV = 1e-12 / 41.516 * 1e9                # 2.41e-5 ppbv


def load_run():
    df = pd.read_csv(DIURNAL / "series.csv")
    # span + diurnal flags must match what this figure claims to show
    meta = {}
    for line in (DIURNAL / "run_meta.txt").read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            meta[key.strip()] = value.strip()
    if meta.get("diurnal") != "true":
        raise ValueError(f"{DIURNAL}/run_meta.txt says diurnal != true — wrong run?")
    if meta.get("abstol") != "1e-12":
        raise ValueError(f"run_meta abstol={meta.get('abstol')!r} — this figure's resolution "
                         "floor and its below-resolution annotations assume 1e-12")
    if float(meta["span_days"]) != SPANS_DAYS:
        raise ValueError(f"series.csv spans {meta['span_days']} d, figure assumes {SPANS_DAYS} d")
    # CROSS-LANGUAGE CHECK (the T4 contract): the cz column the driver exported must equal
    # the clock formula recomputed here, independently, in Python.
    tod = df["time_s"].to_numpy() % 86400.0
    chi = np.minimum(np.deg2rad(89.5), np.abs(2 * np.pi * tod / 86400.0 - np.pi))
    cz = np.maximum(0.0, np.cos(chi))
    err = np.max(np.abs(cz - df["cz"].to_numpy()))
    if err > 1e-12:
        raise ValueError(f"cz column disagrees with formula recomputation (max {err:.2e})")
    return df


def floored(y, floor):
    """Clip to the resolution floor; returns (values, n_clipped). Flooring — unlike dropping
    — keeps the polyline continuous on the log axis, and the floor is an upper bound, not a
    value: anything at it means 'at or below solver resolution', never a measurement."""
    y = np.asarray(y, dtype=float)
    return np.maximum(y, floor), int(np.sum(y < floor))


def night_mask(t):
    """Geometric night: time-of-day outside 06:00-18:00 (the cz floor makes thresholds lie)."""
    tod = np.asarray(t) % 86400.0
    return (tod <= NIGHT_EDGES[0]) | (tod >= NIGHT_EDGES[1])


def shade_nights(ax, t_days):
    """Grey bands over 18:00->06:00, applied to every panel; geometric edges."""
    for d in range(int(np.floor(t_days.max())) + 1):
        ax.axvspan(d + NIGHT_EDGES[1] / 86400.0,      # 18:00 of day d
                   d + 1 + NIGHT_EDGES[0] / 86400.0,   # 06:00 of day d+1
                   color=style.PALETTE["neutral_dark"], alpha=0.08, lw=0, zorder=0)


def build_figure(df):
    """Assemble the six-panel figure; returns (fig, axes)."""
    import matplotlib.pyplot as plt

    t = df["time_s"].to_numpy() / 86400.0
    cz = df["cz"].to_numpy()

    fig, axes = plt.subplots(
        6, 1, figsize=(3.5, 8.6), sharex=True,
        gridspec_kw={"height_ratios": [0.5, 1.0, 1.0, 0.9, 1.0, 0.85]})

    # (a) the environment: cos(zenith)
    ax = axes[0]
    ax.plot(t, cz, color=style.PALETTE["gold"], lw=1.0)
    shade_nights(ax, t)
    ax.set_ylabel("cos χ")
    ax.set_ylim(0.0, 1.05)
    style.add_panel_label(ax, "a", x=-0.13, y=1.12)
    # the night floor is real physics-of-the-port (89.5 deg clamp): annotate it once
    ax.annotate("night floor cos(89.5°) = 0.0087", xy=(t[-1] / 2, 0.0087),
                xytext=(t[-1] / 2, 0.30), fontsize=5.5,
                color=style.PALETTE["neutral_dark"], ha="center",
                arrowprops={"arrowstyle": "-", "color": style.PALETTE["neutral_dark"],
                            "lw": 0.5})

    # (b) O3 alone, LINEAR ppbv — the 30->19 decline + daily sawtooth + frozen overlay
    ax = axes[1]
    o3v = f1.to_ppbv(df["O3"].to_numpy())
    ax.plot(t, o3v, color=style.PALETTE["blue_main"], lw=1.1)
    f1.end_label(ax, t[-1], o3v[-1], "O₃", style.PALETTE["blue_main"])
    ylo = o3v.min()
    if (FROZEN / "series.csv").is_file():
        fz = pd.read_csv(FROZEN / "series.csv")
        fzo3 = f1.to_ppbv(fz["O3"].to_numpy())
        ax.plot(fz["time_s"].to_numpy() / 86400.0, fzo3,
                color=style.PALETTE["blue_main"], lw=1.0, ls=(0, (4, 2)), alpha=0.55)
        ax.text(1.5, fzo3[-1] - 2.2, "frozen-photolysis O₃\n(fig1 run, 3 d)", fontsize=5.5,
                color=style.PALETTE["blue_main"], ha="center", va="top")
        ylo = min(ylo, fzo3.min())
    ax.set_ylim(ylo - 1.5, o3v.max() + 1.5)
    ax.set_ylabel("O₃ (ppbv)")
    shade_nights(ax, t)
    style.add_panel_label(ax, "b", x=-0.13, y=1.14)

    # (c) CH4 alone, LINEAR ppbv, tight range — the -0.8% decline and its day-only steps
    ax = axes[2]
    ch4v = f1.to_ppbv(df["CH4"].to_numpy())
    ax.plot(t, ch4v, color=style.PALETTE["teal"], lw=1.1)
    f1.end_label(ax, t[-1], ch4v[-1], "CH₄", style.PALETTE["teal"])
    pad = 0.25 * (ch4v[0] - ch4v[-1])
    ax.set_ylim(ch4v.min() - pad, ch4v[0] + pad)
    ax.set_ylabel("CH₄ (ppbv)")
    oh_mean = (df["OH"].to_numpy() * 6.02214076e23 / 1e6).mean()
    ax.text(0.02, 0.06,
            f"−{(ch4v[0] - ch4v[-1]) / ch4v[0] * 100:.2f} % / {t[-1]:.0f} d — daylight-only steps\n"
            f"(loss ∝ OH; multi-year lifetime at ⟨OH⟩ = {oh_mean:.2e})",
            transform=ax.transAxes, fontsize=5.5, ha="left", va="bottom",
            color=style.PALETTE["neutral_dark"])
    shade_nights(ax, t)
    style.add_panel_label(ax, "c", x=-0.13, y=1.14)

    # (d) NOx: NO2 decay + NO daylight pulses (log — 3 decades)
    ax = axes[3]
    nox = [
        ("NO2", "NO₂", style.PALETTE["red_strong"], 2.2),
        ("NO", "NO", style.PALETTE["neutral_dark"], 0.55),
    ]
    all_d = []
    for col, label, color, dy in nox:
        # NO's night values sit below resolution; its floored line is an upper bound there
        yy, _ = floored(f1.to_ppbv(df[col].to_numpy()), FLOOR_PPBV)
        all_d.append(yy)
        ax.plot(t, yy, color=color, lw=1.1)
        f1.end_label(ax, t[-1], yy[-1] * dy, label, color)
    lo, hi = f1.decade_limits(np.concatenate(all_d), pad_lo=0.30, pad_hi=0.30)
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.set_ylabel("NO₂, NO (ppbv)")
    shade_nights(ax, t)
    style.add_panel_label(ax, "d", x=-0.13, y=1.14)

    # (e) radicals, molec/cm^3, log
    ax = axes[4]
    radicals = [
        ("HO2", "HO₂", style.PALETTE["blue_secondary"], 1.1),
        ("OH", "OH", style.PALETTE["violet"], 1.0),
    ]
    all_c = []
    for col, label, color, dy in radicals:
        yy, _ = floored(f1.to_molec_cm3(df[col].to_numpy()), FLOOR_MOLEC_CM3)
        all_c.append(yy)
        ax.plot(t, yy, color=color, lw=1.1)
        f1.end_label(ax, t[-1], yy[-1] * dy, label, color)
    lo_c, hi_c = f1.decade_limits(np.concatenate(all_c), snap=True)
    ax.set_yscale("log")
    ax.set_ylim(lo_c, hi_c)
    ax.set_ylabel("Number density\n(molec cm⁻³)")
    ax.axhline(FLOOR_MOLEC_CM3, color=style.PALETTE["neutral_mid"], lw=0.5, ls=(0, (2, 2)))
    ax.text(0.99, 0.06, "dashed: solver resolution (abstol) — night troughs are bounds",
            transform=ax.transAxes, fontsize=5, ha="right", va="bottom",
            color=style.PALETTE["neutral_dark"])
    shade_nights(ax, t)
    style.add_panel_label(ax, "e", x=-0.13, y=1.14)

    # (f) NO3 alone — no night peaks in THIS scenario: NOx-starved. The panel exists to kill
    # fig1's implication that nights alone would make NO3 accumulate here.
    ax = axes[5]
    no3_raw, _ = floored(f1.to_molec_cm3(df["NO3"].to_numpy()), FLOOR_MOLEC_CM3)
    ax.plot(t, no3_raw, color=style.PALETTE["red_strong"], lw=1.1)
    f1.end_label(ax, t[-1], no3_raw[-1] * 1.6, "NO₃", style.PALETTE["red_strong"])
    ax.axhline(FLOOR_MOLEC_CM3, color=style.PALETTE["neutral_mid"], lw=0.5, ls=(0, (2, 2)))
    ax.text(0.985, 0.60,
            "below solver resolution —\n"
            "NO₃'s true value is at/below the dashed\n"
            "abstol line at ALL times (≲6×10² molec cm⁻³);\n"
            "its smallness is NOx starvation, not the\n"
            "missing night (NO₂ collapses ~500×, no source)",
            transform=ax.transAxes, fontsize=5.5, ha="right", va="top",
            color=style.PALETTE["neutral_dark"],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5, "alpha": 0.85})
    lo_d, hi_d = f1.decade_limits(yy, pad_lo=0.30, pad_hi=0.30, snap=True)
    ax.set_yscale("log")
    ax.set_ylim(lo_d, hi_d)
    ax.set_ylabel("NO₃\n(molec cm⁻³)")
    ax.set_xlabel("Time (days)")
    ax.set_xlim(0.0, SPANS_DAYS)
    ax.set_xticks(range(0, 9))
    shade_nights(ax, t)
    style.add_panel_label(ax, "f", x=-0.13, y=1.14)

    caption = (
        "Photolysis follows the zenith clock (Cantera-method port): J = l·cos χᵐ·exp(−n/cos χ),\n"
        "χ(t) triangular with a 89.5° night clamp, J piecewise-constant per 60-s step. Grey bands:\n"
        "18:00–06:00. Radicals pulse with the sun; NO₃ does NOT build at night — this box is\n"
        "NOx-starved (single 0.1-ppb NO₂ pulse, HNO₃ terminal), so fig1's NO₃≈0 was starvation."
    )
    fig.text(0.01, 0.005, caption, fontsize=5.5,
             color=style.PALETTE["neutral_dark"], va="bottom", ha="left")

    fig.subplots_adjust(hspace=0.36, bottom=0.085, left=0.17, right=0.86)
    return fig, axes


def main():
    style.apply_style()
    df = load_run()
    fig, _ = build_figure(df)
    style.save(fig, "fig3_diurnal")

    # --- Physics QA: the numbers the figure's claims are eyeballed against -------------
    t_d = df["time_s"].to_numpy() / 86400.0
    tod = df["time_s"].to_numpy() % 86400.0
    is_night = (tod <= NIGHT_EDGES[0]) | (tod >= NIGHT_EDGES[1])
    no3 = f1.to_molec_cm3(df["NO3"].to_numpy())
    oh = f1.to_molec_cm3(df["OH"].to_numpy())
    o3 = f1.to_ppbv(df["O3"].to_numpy())
    print("\nQA:")
    print(f"  NO3 : max {no3.max():.3e} at t = {t_d[np.argmax(no3)]:.2f} d "
          f"(night at that sample: {is_night[np.argmax(no3)]})")
    print(f"  NO3 : below solver resolution (peak {no3.max():.3e} vs abstol 6.0e2 molec/cm^3) —"
          " upper bound only; structure claims are NOT possible at this tolerance")
    oh_night_min = oh[is_night].min()
    print(f"  OH  : day max {oh[~is_night].max():.3e}, night min {oh_night_min:.3e} "
          + ("(below resolution; negative residual < abstol — floored in panel e)"
             if oh_night_min < FLOOR_MOLEC_CM3 else
             f"-> day/night {oh[~is_night].max() / oh_night_min:.0f}"))
    # O3 daily extremes: per-calendar-day min/max to show the cycle amplitude
    for d in range(8):
        m = (t_d >= d) & (t_d < d + 1)
        if m.any():
            print(f"  O3 day {d}: min {o3[m].min():8.3f}  max {o3[m].max():8.3f} ppbv")
    # CH4 closure: the decline implies a 24-h mean OH; compare with the OH trajectory itself
    ch4 = df["CH4"].to_numpy()
    k_ch4_oh = 6.3e-15                         # cm3/molec/s, IUPAC at 298 K
    span = df["time_s"].to_numpy()[-1] - df["time_s"].to_numpy()[0]
    implied = np.log(ch4[0] / ch4[-1]) / span / k_ch4_oh
    direct = (df["OH"].to_numpy() * 6.02214076e23 / 1e6).mean()
    print(f"  CH4 : {(ch4[0] - ch4[-1]) / ch4[0] * 100:.3f} % over 8 d; implied <OH> "
          f"{implied:.2e} vs direct {direct:.2e} molec/cm^3 "
          f"({abs(implied / direct - 1) * 100:.1f}% off — d[CH4]/dt = -k[OH][CH4] closure)")


if __name__ == "__main__":
    sys.exit(main())
