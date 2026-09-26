"""Figure 2 — Jacobian-strategy efficiency for the atmospheric box model.

Core conclusion: the reaction-sharded analytic Jacobian costs ~4.3x more to
build but solves 4.4-12.2x faster at every measured span, so the total-cost
break-even sits at ~1-4 simulated days on this shared box — a range, not a
number, because co-tenant load noise swamps the span-dependence of the solve.

Panel a — measured costs (bench_jac.csv): grouped bars per strategy for the
one-time build and ONE 0.5-day solve. Build and solve are separate categories,
never stacked on one bar, because they are paid on different schedules: build
once, solve per run.
Panel b — measured TOTALS as points (build + each solve). The linear-solve
assumption was tested by the bench and FALSIFIED on this box: longer spans
measured faster, which warm linear scaling forbids, so no fitted line is drawn.
Instead the break-even is shown as the range implied by the warm measured
pairs (each pair's per-span rates cross at its own span), with this example's
3-day run marked on the span axis.

Cold starts: the first solve call of a strategy pays one-time solver
compilation, inflating the 0.25-d row; those points are marked x and excluded
from the pair-based break-even (Julia's own @btime discards the first call for
the same reason).

Load caveat (central, on-figure): timings are single-run wall-clock on a
shared machine and absolute seconds move 3-6x with co-tenant load; the
transferable quantities are the per-span solve ratios and the build ratio.

Grayscale/print safety: the two strategies differ by hue AND by hatch (bars) /
marker fill, so the figure survives a grayscale print.

Run:  python3 examples/atmospheric/tools/figures/perf.py
"""

import sys
import textwrap

import numpy as np
import pandas as pd

try:
    from . import style  # package execution
except ImportError:
    import style  # direct script execution


# Strategy -> color, encoding. Same mapping in both panels (never remap a method
# between panels). Baseline gets the neutral family, the analytic path the hero
# blue; hatch/marker fill carry the distinction in grayscale.
STRATEGIES = [
    # (csv name, display name, color, hatch)
    ("finite-difference", "finite-difference", style.PALETTE["neutral_dark"],
     "///"),
    ("reaction-sharded", "reaction-sharded (analytic)", style.PALETTE["blue_main"],
     None),
]
C_CROSS = style.PALETTE["red_strong"]    # threshold cue: the break-even band
C_BAND = style.PALETTE["red_1"]          # light fill of the band
C_MARK3 = style.PALETTE["neutral_dark"]  # the 3-day example marker

LINEARITY_R2 = 0.95   # below this, or max relative deviation above ...
LINEARITY_DEV = 0.20  # ... this, the line model is withdrawn
X_EXAMPLE = 3.0       # this example's simulated span (mcm_box.jl), days


def load_bench(path):
    df = pd.read_csv(path, comment="#")
    df["strategy"] = df["strategy"].astype(str).str.strip()
    return df


def fit_strategy(df, name):
    """One strategy's bench rows -> build, per-span solves, through-origin rate fit.

    Cold-start handling: the FIRST solve call of a strategy pays one-time solver
    compilation on top of the integration. A longer span solving strictly FASTER
    than the first (shortest) span is impossible under warm linear scaling, so
    that signature identifies a contaminated first row; it is excluded from the
    rate fit (still plotted, marked x) exactly like @btime discards Julia's
    first call. `cold_compile_s` quantifies the one-time excess when detected.
    """
    sub = df[df["strategy"] == name].sort_values("span_days")
    if sub.empty:
        raise ValueError(f"bench_jac.csv has no rows for strategy {name!r}")
    builds = sub["build_s"].to_numpy(dtype=float)
    if builds.max() - builds.min() > 1e-9:
        raise ValueError(f"{name}: build_s differs across spans — the one-build "
                         "assumption of this figure is violated")
    spans = sub["span_days"].to_numpy(dtype=float)
    solves = sub["solve_s"].to_numpy(dtype=float)
    cold = len(solves) > 2 and solves[0] > np.max(solves[1:])
    fit_sl = slice(1, None) if cold else slice(None)
    wspans, wsolves = spans[fit_sl], solves[fit_sl]
    # Least squares through the origin: solve_s ~ rate * span_days. The model is
    # proportional by construction (zero span must cost zero solve), so no intercept.
    rate = float(wspans @ wsolves / (wspans @ wspans))
    pred = rate * wspans
    ss_res = float(((wsolves - pred) ** 2).sum())
    ss_tot = float(((wsolves - wsolves.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    max_rel = float(np.max(np.abs(wsolves - pred) / wsolves))
    return {
        "name": name, "spans": spans, "solves": solves,
        "build": float(builds[0]), "rate": rate, "r2": r2, "max_rel": max_rel,
        "linear": (r2 >= LINEARITY_R2) and (max_rel <= LINEARITY_DEV),
        "cold": bool(cold),
        "cold_compile_s": (float(solves[0] - rate * spans[0]) if cold else 0.0),
    }


def solve_at(fit, days):
    """Measured solve_s at an exact span, else the through-origin prediction."""
    hit = fit["spans"] == days
    return float(fit["solves"][hit][0]) if hit.any() else fit["rate"] * days


def warm_spans(fit):
    """Spans whose solve rows are warm (first-call compile excluded when cold)."""
    return fit["spans"][1:] if fit["cold"] else fit["spans"]


def break_even(fd, sh):
    """Single crossing span of the two total-cost LINES (linear regime only)."""
    if not (fd["linear"] and sh["linear"]):
        return None
    denom = fd["rate"] - sh["rate"]
    if denom <= 0:
        return None
    t = (sh["build"] - fd["build"]) / denom
    return t if 0 < t <= X_EXAMPLE else None


def break_even_range(fd, sh):
    """Crossing spans implied by each WARM measured pair, as (lo, hi).

    With load noise swamping the span-dependence there is no single fitted
    solve rate, so each warm span s contributes its own crossing: the span
    where the build + (solve(s)/s)·x curves cross. The spread of those
    crossings is the honest statement of the break-even.
    """
    shared = np.intersect1d(warm_spans(fd), warm_spans(sh))
    if shared.size == 0:
        return None
    ts = []
    for s in shared:
        r_fd = solve_at(fd, s) / s
        r_sh = solve_at(sh, s) / s
        denom = r_fd - r_sh
        if denom > 0:
            t = (sh["build"] - fd["build"]) / denom
            if t > 0:
                ts.append(t)
    return (min(ts), max(ts)) if ts else None


def span_ratios(fd, sh):
    """solve_fd / solve_sh at each measured span — the transferable ratios."""
    out = []
    for s in fd["spans"]:
        hit = sh["spans"] == s
        if hit.any():
            out.append(float(solve_at(fd, s) / solve_at(sh, s)))
    return out


def build_figure(bench):
    """Assemble the two-panel figure; returns (fig, ax_a, ax_b, meta)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    fits = {name: fit_strategy(bench, name) for name, *_ in STRATEGIES}
    fd, sh = fits["finite-difference"], fits["reaction-sharded"]
    t_cross = break_even(fd, sh)
    ber = None if t_cross is not None else break_even_range(fd, sh)
    ratios = span_ratios(fd, sh)
    speedup = solve_at(fd, 0.5) / solve_at(sh, 0.5)
    build_ratio = sh["build"] / fd["build"]

    # --- Caption composed BEFORE the figure so its line count sizes the canvas
    # (it is bottom-anchored; every extra line eats upward into the axes). Each
    # line is held under ~128 chars (~405 pt at 5.5 pt) so the caption can never
    # widen the exported page past the 7.1 in design width.
    if fd["linear"] and sh["linear"]:
        lines = [
            "b: measured totals, build + solve (open circles). Solve time scaled linearly "
            f"with span (worst-case through-origin fit R² = {min(fd['r2'], sh['r2']):.3f}),",
            "so the straight-line model is drawn; the crossing is annotated at its measured "
            "position. Cold first solves pay one-time",
            "solver compilation (marked x) and are excluded from the fit.",
            "Timings are single-run wall-clock on a shared machine: co-tenant load moves "
            "absolute times by 3–6×, so the",
            "transferable quantities are the per-span solve ratios "
            f"({min(ratios):.1f}–{max(ratios):.1f}×) and the build ratio "
            f"({build_ratio:.1f}×), not the seconds.",
            "Mechanism: MCM alkanes/alkenes, 1842 species, 5600 reactions; one build per "
            "strategy, solves by remake.",
        ]
    else:
        lines = [
            "b: measured totals, build + solve (open circles; x marks each strategy's first "
            "solve, which also pays one-time",
            "solver compilation). Solve time did NOT scale linearly with span — longer spans "
            "measured FASTER, which warm linear",
            "scaling forbids — so no line model is drawn. Shaded band: break-even range "
            "implied by the warm measured pairs",
            "(each pair's per-span rates cross at its own span); dotted line: this "
            f"example's {X_EXAMPLE:.0f}-day run.",
            "Timings are single-run wall-clock on a shared machine: co-tenant load moves "
            "absolute times by 3–6×, so the",
            "transferable quantities are the per-span solve ratios "
            f"({min(ratios):.1f}–{max(ratios):.1f}×) and the build ratio "
            f"({build_ratio:.1f}×), not the seconds.",
            "Mechanism: MCM alkanes/alkenes, 1842 species, 5600 reactions; one build per "
            "strategy, solves by remake.",
        ]
    assert max(len(l) for l in lines) <= 128, "caption line would widen the page"
    caption = "\n".join(lines)
    n_cap = len(lines)
    fig_h = 3.05 + 0.092 * max(n_cap - 5, 0)   # room for extra caption lines
    bottom_f = min((0.88 + 0.092 * max(n_cap - 5, 0)) / fig_h, 0.45)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.1, fig_h), gridspec_kw={"width_ratios": [1.0, 1.3]},
    )

    # --- Panel a: measured costs, grouped bars -----------------------------------
    # Build is paid once per mechanism; solve is paid per run. Different payment
    # schedules -> different x categories, never one stacked bar.
    cats = ["Build\n(one-time)", "Solve\n(one 0.5-day run)"]
    vals = {name: [fits[name]["build"], solve_at(fits[name], 0.5)]
            for name, *_ in STRATEGIES}
    top_a = max(v for pair in vals.values() for v in pair) * 1.34  # label room
    x = np.arange(len(cats))
    w = 0.34
    for i, (name, disp, color, hatch) in enumerate(STRATEGIES):
        bars = ax_a.bar(x + (i - 0.5) * w, vals[name], width=w, color=color,
                        edgecolor="black", linewidth=0.5, hatch=hatch,
                        label=disp)
        for b, v in zip(bars, vals[name]):
            ax_a.text(b.get_x() + b.get_width() / 2, b.get_height() + top_a * 0.02,
                      f"{v:.1f}", ha="center", va="bottom", fontsize=6)
    ax_a.set_xticks(x, cats)
    ax_a.set_ylabel("Wall-clock (s)")
    ax_a.set_ylim(0, top_a)
    ax_a.yaxis.set_major_locator(
        MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]))  # sparse, data-scaled
    ax_a.legend(fontsize=6, loc="upper left", handlelength=1.4,
                borderaxespad=0.2, borderpad=0.2)
    # The headline ratio(s), stated where they are measured (the solve category),
    # below the legend band (top-left) to keep the panel clash-free.
    head = (f"{speedup:.1f}× faster solve" if fd["linear"] and sh["linear"]
            else f"{min(ratios):.1f}–{max(ratios):.1f}× faster solve")
    ax_a.text(1.0, top_a * 0.84, head, ha="center", va="top", fontsize=6,
              color=style.PALETTE["blue_main"])
    style.add_panel_label(ax_a, "a", x=-0.22, y=1.05)

    # --- Panel b: total cost vs span ---------------------------------------------
    for name, disp, color, _hatch in STRATEGIES:
        f = fits[name]
        if f["linear"]:
            xs = np.linspace(0.0, 3.0, 121)
            ax_b.plot(xs, f["build"] + f["rate"] * xs, color=color, lw=1.1)
            y_end = f["build"] + solve_at(f, 3.0)
            ax_b.text(3.06, y_end, "\n".join(textwrap.wrap(disp, 16)),
                      color=color, fontsize=6, ha="left", va="center",
                      clip_on=False, linespacing=1.25)
        # measured totals, always shown, as open markers; a cold first solve
        # (carries one-time compilation) gets a distinct x over the open circle
        ax_b.plot(f["spans"], f["build"] + f["solves"], marker="o", ls="none",
                  markerfacecolor="white", markeredgecolor=color, markersize=3.2,
                  markeredgewidth=0.9)
        if f["cold"]:
            ax_b.plot([f["spans"][0]], [f["build"] + f["solves"][0]], marker="x",
                      ls="none", color=color, markersize=5.5,
                      markeredgewidth=1.1, zorder=4)

    totals = [f["build"] + solve_at(f, 3.0) for f in (fd, sh)]
    y_top = max(totals) * 1.30
    if t_cross is not None:
        ax_b.set_xlim(0.0, 3.0)
        ax_b.set_xticks([0, 1, 2, 3])
        y_cross = fd["build"] + fd["rate"] * t_cross
        ax_b.plot([t_cross], [y_cross], marker="o", markersize=3.5, color=C_CROSS,
                  markerfacecolor=C_CROSS, zorder=5)
        ax_b.annotate(f"break-even\n{t_cross:.2f} days",
                      xy=(t_cross, y_cross), xytext=(t_cross - 0.22, y_cross * 1.34),
                      fontsize=6, color=C_CROSS, ha="right", va="bottom",
                      arrowprops=dict(arrowstyle="-", lw=0.7, color=C_CROSS))
    else:
        # Points-only regime: room to the right for the band and the example mark
        x_max = 4.3 if ber is not None else 3.0
        ax_b.set_xlim(0.0, x_max)
        ax_b.set_xticks(list(range(0, int(x_max) + 1)))
        if ber is not None:
            lo, hi = ber
            hi_view = min(hi, x_max - 0.05)
            ax_b.axvspan(lo, hi_view, color=C_BAND, alpha=0.55, lw=0, zorder=1)
            ax_b.text((lo + hi_view) / 2, y_top * 0.045,
                      f"break-even ≈ {lo:.1f}–{hi:.1f} days\n(measured pairs)",
                      fontsize=6, color=C_CROSS, ha="center", va="bottom",
                      zorder=5)
        if x_max > X_EXAMPLE:
            ax_b.axvline(X_EXAMPLE, ls=":", lw=0.9, color=C_MARK3, zorder=2)
            ax_b.text(X_EXAMPLE, y_top * 0.90, f"{X_EXAMPLE:.0f}-day\nexample",
                      fontsize=5.5, color=C_MARK3, ha="center", va="top",
                      zorder=5)
        # no lines to label at their ends -> compact marker legend instead
        handles = [Line2D([], [], marker="o", ls="none", markerfacecolor="white",
                          markeredgecolor=c, markeredgewidth=0.9, label=disp)
                   for _n, disp, c, _h in STRATEGIES]
        ax_b.legend(handles=handles, fontsize=6, loc="upper left",
                    handletextpad=0.3, borderaxespad=0.2, borderpad=0.2)
    ax_b.set_ylim(0, y_top)
    ax_b.set_xlabel("Simulated span (days)")
    ax_b.set_ylabel("Total wall-clock (s)")
    style.add_panel_label(ax_b, "b", x=-0.16, y=1.05)

    fig.text(0.01, 0.005, caption, fontsize=5.5,
             color=style.PALETTE["neutral_dark"], va="bottom", ha="left")
    fig.subplots_adjust(left=0.075, right=0.85, bottom=bottom_f, top=0.88,
                        wspace=0.34)
    meta = {"fits": fits, "t_cross": t_cross, "ber": ber, "ratios": ratios,
            "speedup": speedup, "build_ratio": build_ratio}
    return fig, ax_a, ax_b, meta


def main():
    style.apply_style()
    bench = load_bench(style.BENCH_CSV)
    fig, _, _, meta = build_figure(bench)
    style.save(fig, "perf")

    # --- QA printout -------------------------------------------------------------
    print("\nQA (from bench_jac.csv):")
    for name, f in meta["fits"].items():
        print(f"  {name:>18}: build {f['build']:7.1f} s | rate {f['rate']:7.1f} s/day"
              f" | R2 {f['r2']:.4f} | max dev {f['max_rel'] * 100:5.1f}%"
              f" | linear={f['linear']} | cold={f['cold']}"
              + (f" (compile ~{f['cold_compile_s']:.0f} s)" if f["cold"] else ""))
    r = meta["ratios"]
    print(f"  solve ratio per span:   {', '.join(f'{v:.2f}x' for v in r)}")
    print(f"  build ratio:            {meta['build_ratio']:.2f}x")
    if meta["t_cross"]:
        print(f"  break-even (fitted):    {meta['t_cross']:.2f} days")
    elif meta["ber"]:
        print(f"  break-even (measured pairs): {meta['ber'][0]:.2f}"
              f"–{meta['ber'][1]:.2f} days")
    else:
        print("  break-even:             not resolvable")


if __name__ == "__main__":
    sys.exit(main())
