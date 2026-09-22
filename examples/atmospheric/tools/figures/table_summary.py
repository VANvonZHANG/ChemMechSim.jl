"""Summary table for the atmospheric box-model example.

Three blocks, all derived from the run artifacts in output/:
1. Final state   — series.csv's last row for every monitored species, in mol/m^3,
                   molec/cm^3 and ppbv, with the initial value alongside.
2. Rate budget   — budget.csv's top 3 entries per kind, NETTED BY EQUATION FIRST.
                   The converter splits one MCM rate expression into duplicate
                   per-term rows (including signed negative twins), so the same
                   equation appears several times; quoting one row alone (e.g.
                   the positive CH3O2 + HO2 => CH3OOH entry, +1.395e-12 against
                   its -1.251e-13 twin) overstates that sink by ~9%.
3. Jacobian cost — bench_jac.csv's build/solve split per strategy, the derived
                   speedups and the data-derived break-even span.

Emits output/summary_table.md (for pasting into a report) and
output/summary_table.csv (long format, one metric per row, for reuse).

Run:  python3 examples/atmospheric/tools/figures/table_summary.py
"""

import sys

import numpy as np
import pandas as pd

try:
    from . import style  # package execution
    from . import fig2_efficiency as f2  # shared fit + break-even (pure functions)
except ImportError:
    import style  # direct script execution
    import fig2_efficiency as f2


MONITOR = ["O3", "NO", "NO2", "NO3", "OH", "HO2", "CH4"]  # mcm_box.jl's list
KINDS = ["O3_production", "O3_loss", "HOx_source", "HOx_sink"]
N_A = 6.02214076e23
T0, P0, R_GAS = 298.0, 102858.0, 8.314
C_AIR = P0 / (R_GAS * T0)  # 41.516 mol/m^3 — same scenario density fig1 divides by

SUBSCRIPT = {"O3": "O₃", "NO2": "NO₂", "NO3": "NO₃", "HO2": "HO₂", "CH4": "CH₄",
             "NO": "NO", "OH": "OH"}


def to_ppbv(c):
    return c / C_AIR * 1e9


def to_molec_cm3(c):
    return c * N_A / 1e6


def fmt(c):
    """3 significant digits, plain or scientific as magnitude dictates."""
    return f"{c:.3g}"


def t_lower_seconds():
    """t_lower_s parsed from output/run_meta.txt.

    The header quotes this number, so it must come from the artifact — a
    future mcm_box.jl run rewrites run_meta.txt and the summary follows it.
    A missing file or key is an error, never a silent hardcoded fallback."""
    path = style.DATA / "run_meta.txt"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}: re-run mcm_box.jl to regenerate the run "
            "artifacts this summary quotes")
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == "t_lower_s":
            return float(value)
    raise ValueError(f"no t_lower_s= line in {path}")


def run_span_days():
    """span_days parsed from output/run_meta.txt (same contract as t_lower_seconds)."""
    path = style.DATA / "run_meta.txt"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}: re-run mcm_box.jl to regenerate the run artifacts")
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == "span_days":
            return float(value)
    raise ValueError(f"no span_days= line in {path}")


def span_vs_breakeven_clause(span, lo, hi):
    """Computed, never hardcoded: where the example's span sits relative to the
    measured break-even RANGE. A prior version hardcoded 'sits inside that
    range', which became false the moment the bench moved (0.8-1.0 d puts a
    3-day span past it — the opposite verdict).

    Direction: break-even is the span beyond which the analytic Jacobian's
    cheaper solves have repaid its costlier build — so a span ABOVE the range
    means the analytic path wins outright, BELOW it the finite-difference path
    is cheaper. (The first draft of this helper inverted those two labels and
    contradicted the per-pair projections printed right below; caught on
    regeneration review.)"""
    if span > hi:
        return (f"The example's {span:g}-day span is PAST the break-even, so the "
                "analytic Jacobian wins outright at this span.")
    if span < lo:
        return (f"The example's {span:g}-day span is BELOW the break-even, so the "
                "finite-difference path is cheaper at this span.")
    return (f"The example's {span:g}-day span sits inside that range, so no "
            "total-cost verdict at this span.")


# --- Block 1: final state ------------------------------------------------------
def final_state_rows():
    df = pd.read_csv(style.DATA / "series.csv")
    first, last = df.iloc[0], df.iloc[-1]
    rows = []
    for sp in MONITOR:
        c0, c1 = float(first[sp]), float(last[sp])
        rows.append({
            "species": SUBSCRIPT[sp], "raw": sp,
            "init_ppbv": to_ppbv(c0), "final_ppbv": to_ppbv(c1),
            "final_molec_cm3": to_molec_cm3(c1), "final_mol_m3": c1,
        })
    return rows, float(df["time_s"].iloc[-1]) / 86400.0


# --- Block 2: budget, netted by equation --------------------------------------
def budget_blocks(top=3):
    """Top-`top` equations per kind after summing duplicate rows of the equation."""
    df = pd.read_csv(style.DATA / "budget.csv")
    n_rows = len(df)
    net = (df.groupby(["kind", "equation"], as_index=False)["rate_mol_m3_s"]
             .sum())
    blocks = {}
    for kind in KINDS:
        sub = net[net["kind"] == kind].copy()
        sub = sub.reindex(sub["rate_mol_m3_s"].abs()
                                 .sort_values(ascending=False).index)
        blocks[kind] = sub.head(top).reset_index(drop=True)
    # provenance note: how much netting happened, and the signed-twin example
    dup_groups = n_rows - len(net)
    twin = df[df["equation"].str.startswith("CH3O2 + HO2 => CH3OOH")]
    twin_note = ""
    if len(twin) > 1:
        pos = twin.loc[twin["rate_mol_m3_s"] > 0, "rate_mol_m3_s"].sum()
        netv = twin["rate_mol_m3_s"].sum()
        twin_note = (f"e.g. {twin['equation'].iloc[0]}: rows "
                     f"{' and '.join(f'{v:+.3g}' for v in twin['rate_mol_m3_s'])} "
                     f"net to {netv:.3g}, {abs(1 - netv / pos) * 100:.0f}% below "
                     "the positive row alone")
    return blocks, n_rows, dup_groups, twin_note


# --- Block 3: Jacobian cost ----------------------------------------------------
# The fit and the break-even are the SAME functions the figure uses, imported
# from fig2_efficiency (they are pure — no matplotlib at import time), so the
# figure's annotated crossing and this table can never disagree.
fit_strategy = f2.fit_strategy
solve_at = f2.solve_at


def cost_block():
    bench = pd.read_csv(style.DATA / "bench_jac.csv", comment="#")
    bench["strategy"] = bench["strategy"].astype(str).str.strip()
    fits = {n: fit_strategy(bench, n) for n in ("finite-difference",
                                                "reaction-sharded")}
    fd, sh = fits["finite-difference"], fits["reaction-sharded"]
    t_cross = f2.break_even(fd, sh)
    # Load-noise regime: no single fitted rate, so each WARM measured pair
    # implies its own crossing; the spread is the honest break-even.
    ber = None if t_cross is not None else f2.break_even_range(fd, sh)
    ratios = f2.span_ratios(fd, sh)
    build_ratio = sh["build"] / fd["build"]
    total3 = {n: (f["build"] + 3.0 * f["rate"] if f["linear"] else None)
              for n, f in fits.items()}
    # Per-warm-pair projections of the total cost at the example's 3-day span:
    # with no valid single rate, the pairs bracket the answer instead.
    proj3 = {"finite-difference": [], "reaction-sharded": []}
    for s in np.intersect1d(f2.warm_spans(fd), f2.warm_spans(sh)):
        for name, f in fits.items():
            proj3[name].append(f["build"] + (f2.solve_at(f, s) / s) * 3.0)
    speedup_solve = solve_at(fd, 0.5) / solve_at(sh, 0.5)
    speedup_total3 = (total3["finite-difference"] / total3["reaction-sharded"]
                      if total3["finite-difference"] and total3["reaction-sharded"]
                      else None)
    return (bench, fits, t_cross, ber, ratios, build_ratio, total3, proj3,
            speedup_solve, speedup_total3)


# --- Assembly ------------------------------------------------------------------
def build_md(fs_rows, t_end, blocks, n_rows, dup_groups, twin_note,
             bench, fits, t_cross, ber, ratios, build_ratio, total3, proj3,
             speedup_solve, speedup_total3):
    L = []
    L.append("# Atmospheric box model — run summary")
    L.append("")
    L.append("MCM alkanes/alkenes (1842 species, 5600 reactions), T = 298 K, "
             "P = 102 858 Pa, photolysis frozen at χ₀ = 0 (perpetual day, no "
             "diurnal cycle), isothermal :kinetic mode, FBDF(autodiff=false), "
             f"reltol 1e-6 / abstol 1e-12. Reference run: {t_end:.0f}-day span "
             f"(`output/run_meta.txt`: lowering {t_lower_seconds():.1f} s; "
             "its `t_simulate_s` is "
             "build+solve COMBINED, not pure solve — the split below is the "
             "authoritative one from `bench_jac.csv`).")
    L.append("")

    L.append(f"## 1 Final state (t = {t_end:.0f} days)")
    L.append("")
    L.append("| Species | Initial (ppbv) | Final (ppbv) | Final (molec cm⁻³) "
             "| Final (mol m⁻³) |")
    L.append("|---|---:|---:|---:|---:|")
    for r in fs_rows:
        L.append(f"| {r['species']} | {fmt(r['init_ppbv'])} | {fmt(r['final_ppbv'])} "
                 f"| {fmt(r['final_molec_cm3'])} | {fmt(r['final_mol_m3'])} |")
    L.append("")

    L.append("## 2 Rate budgets at the final state (top 3 per kind, net by equation)")
    L.append("")
    L.append("Rates are budget-species molecule fluxes (reaction rate × |net "
             "stoichiometric change|), mol m⁻³ s⁻¹.")
    L.append("")
    for kind in KINDS:
        sub = blocks[kind]
        L.append(f"**{kind}**")
        L.append("")
        L.append("| Rank | Equation | Net flux (mol m⁻³ s⁻¹) |")
        L.append("|---:|---|---:|")
        for i, row in sub.iterrows():
            L.append(f"| {i + 1} | {row['equation']} | {fmt(row['rate_mol_m3_s'])} |")
        L.append("")
    if dup_groups:
        L.append(f"*Netting provenance: {n_rows} budget rows collapsed to "
                 f"{n_rows - dup_groups} unique (kind, equation) groups "
                 f"({dup_groups} duplicate rows summed), {twin_note}.*")
        L.append("")

    L.append("## 3 Jacobian-strategy cost (from bench_jac.csv)")
    L.append("")
    fd, sh = fits["finite-difference"], fits["reaction-sharded"]
    fitted = fd["linear"] and sh["linear"]
    any_cold = any(f["cold"] for f in fits.values())
    cold_col = " First-solve compile (s) |" if any_cold else ""
    L.append("| Strategy | Build (s) | Solve 0.25 d (s) | Solve 0.5 d (s) "
             "| Solve 1.0 d (s) | Solve rate (s/day) | Solve speedup (0.5 d) |"
             + cold_col)
    L.append("|---|---:|---:|---:|---:|---:|---:|" + ("---:|" if any_cold else ""))
    for name in ("finite-difference", "reaction-sharded"):
        f = fits[name]
        up = f"{speedup_solve:.1f}×" if name == "reaction-sharded" else "—"
        rate_v = f"{f['rate']:.1f}" if fitted else "n/a†"
        cold_v = (f"{f['cold_compile_s']:.0f}" if f["cold"] else "—") if any_cold else ""
        L.append(f"| {name} | {f['build']:.1f} | {solve_at(f, 0.25):.1f} "
                 f"| {solve_at(f, 0.5):.1f} | {solve_at(f, 1.0):.1f} "
                 f"| {rate_v} | {up} |" + (f" {cold_v} |" if any_cold else ""))
    L.append("")
    if any_cold:
        L.append("*The 0.25-d solve row of each strategy includes one-time "
                 "first-call solver compilation (quantified in the compile "
                 "column); the 0.25-d measurement is shown raw.*")
        L.append("")
    if t_cross is not None:
        L.append(f"**Break-even: {t_cross:.2f} simulated days** — the span where "
                 "the two fitted total-cost lines cross, computed from the "
                 "measured rates (not a hardcoded value).")
    elif ber is not None:
        L.append(f"**Break-even: ~{ber[0]:.1f}–{ber[1]:.1f} simulated days** — a "
                 "RANGE, not a number: solve time did not scale linearly with "
                 "span under this box's co-tenant load (longer spans measured "
                 "faster), so each warm measured pair implies its own crossing "
                 f"({ber[0]:.2f} d from the 0.5-d pair, {ber[1]:.2f} d from the "
                 f"1.0-d pair). {span_vs_breakeven_clause(run_span_days(), ber[0], ber[1])}")
    else:
        L.append("**Break-even: not resolvable** from this run — the measured "
                 "pairs do not imply a consistent crossing.")
    if fitted and speedup_total3:
        direction = "faster" if speedup_total3 >= 1.0 else "slower"
        L.append(f"At the reference 3-day span the analytic Jacobian is "
                 f"{(speedup_total3 if speedup_total3 >= 1.0 else 1 / speedup_total3):.1f}× "
                 f"{direction} overall "
                 f"({total3['finite-difference']:.0f} s vs "
                 f"{total3['reaction-sharded']:.0f} s including builds).")
    elif proj3["finite-difference"]:
        L.append(f"Per-pair projections to the 3-day span disagree (finite "
                 f"difference {min(proj3['finite-difference']):.0f}–"
                 f"{max(proj3['finite-difference']):.0f} s vs analytic "
                 f"{min(proj3['reaction-sharded']):.0f}–"
                 f"{max(proj3['reaction-sharded']):.0f} s), so no single "
                 "total-cost verdict at 3 days is quoted; the analytic solve "
                 f"advantage holds at every measured span ({min(ratios):.1f}–"
                 f"{max(ratios):.1f}×) against a {build_ratio:.1f}× build cost.")
    L.append("")
    L.append(f"*Load caveat: timings are single-run wall-clock on a shared "
             "machine under co-tenant load (qualitative: co-tenants were "
             "running throughout); absolute times move "
             "3–6× with that load. The transferable quantities are the per-span "
             f"solve ratios ({', '.join(f'{v:.1f}×' for v in ratios)}) and the "
             f"build ratio ({build_ratio:.1f}×), not the seconds.*")
    if not fitted:
        L.append("")
        L.append("*†No single solve rate is quoted: the measured solves are "
                 "non-monotonic in span, so a fitted rate would be spurious.*")
    L.append("")
    return "\n".join(L)


def build_csv_rows(fs_rows, blocks, fits, t_cross, ber, ratios, build_ratio,
                   total3, speedup_solve):
    rows = []
    for r in fs_rows:
        for metric, value in (
            ("initial_ppbv", r["init_ppbv"]), ("final_ppbv", r["final_ppbv"]),
            ("final_molec_cm3", r["final_molec_cm3"]),
            ("final_mol_m3", r["final_mol_m3"]),
        ):
            rows.append(["final_state", r["raw"], metric, repr(value),
                         "mol/m^3" if metric.endswith("mol_m3")
                         else "molec/cm^3" if "molec" in metric else "ppbv"])
    for kind in KINDS:
        for _i, row in blocks[kind].iterrows():
            rows.append(["budget", kind, row["equation"],
                         repr(row["rate_mol_m3_s"]), "mol/m^3/s (species flux)"])
    for name, f in fits.items():
        rows.append(["cost", name, "build_s", repr(f["build"]), "s"])
        for d in (0.25, 0.5, 1.0):
            rows.append(["cost", name, f"solve_s_{d}d", repr(solve_at(f, d)), "s"])
        rows.append(["cost", name, "solve_rate_s_per_day_warm_fit",
                     repr(f["rate"]),
                     "s/day" if f["linear"] else "s/day (invalid: non-monotonic)"])
        if f["cold"]:
            rows.append(["cost", name, "cold_compile_s", repr(f["cold_compile_s"]),
                         "s (one-time, first solve)"])
        if total3[name]:
            rows.append(["cost", name, "total_s_3d", repr(total3[name]), "s"])
    fd_spans = fits["finite-difference"]["spans"][:len(ratios)]
    for s, v in zip(fd_spans, ratios):
        rows.append(["cost", "reaction-sharded", f"solve_speedup_{s}d_vs_fd",
                     repr(v), "ratio"])
    rows.append(["cost", "reaction-sharded", "build_ratio_vs_fd",
                 repr(build_ratio), "ratio"])
    if t_cross is not None:
        rows.append(["cost", "meta", "break_even_days", repr(t_cross), "days"])
    if ber is not None:
        rows.append(["cost", "meta", "break_even_days_lo",
                     repr(float(ber[0])), "days"])
        rows.append(["cost", "meta", "break_even_days_hi",
                     repr(float(ber[1])), "days"])
    return rows


def main():
    fs_rows, t_end = final_state_rows()
    blocks, n_rows, dup_groups, twin_note = budget_blocks(top=3)
    (bench, fits, t_cross, ber, ratios, build_ratio, total3, proj3,
     speedup_solve, speedup_total3) = cost_block()

    md = build_md(fs_rows, t_end, blocks, n_rows, dup_groups, twin_note,
                  bench, fits, t_cross, ber, ratios, build_ratio, total3,
                  proj3, speedup_solve, speedup_total3)
    md_path = style.OUT / "summary_table.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"wrote {md_path}")

    csv_rows = build_csv_rows(fs_rows, blocks, fits, t_cross, ber, ratios,
                              build_ratio, total3, speedup_solve)
    csv_path = style.OUT / "summary_table.csv"
    pd.DataFrame(csv_rows,
                 columns=["block", "item", "metric", "value", "unit"]
                 ).to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")

    print()
    print(md)


if __name__ == "__main__":
    sys.exit(main())
