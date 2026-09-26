# Atmospheric chemistry example — MCM alkanes/alkenes

An atmospheric box model built on the MCM alkanes/alkenes subset: 1843 species (one a
phantom `M`), 5600 reactions, isothermal at 298 K, fixed pressure — integrated for 8 days
in two photolysis modes, both reading the 24 MB source mechanism **directly** (there is no
preprocessor, no derived mechanism, no sidecar):

* `frozen` — photolysis parameters at their defaults = J at overhead sun (perpetual day)
* `diurnal` — the same parameters driven by the zenith clock (the Cantera-method port)

## Why `:kinetic` mode

The reactor is built with `BatchReactor(mech; mode=:kinetic, checks=false)`. `:kinetic` is the
`MechanismConfig()` zero point, and for an atmospheric box model it is not merely sufficient —
the other modes are wrong:

| `:kinetic` field | why it matches an atmospheric box |
|---|---|
| `energy = :isothermal` | T is given by the scenario (298 K), never solved from an energy balance |
| `constraint = :none` | there is no volume to solve for; no constraint layer is wanted |
| `eos = :off` | P is given (102858 Pa) and the state basis is concentration, so no pressure is ever derived |
| `reverse_rate = :irreversible` | MCM emits only forward rates (every reaction is `=>`), and its thermo data is uniformly zero, so `K_c` could not be formed anyway |
| `state_basis = :concentration` | the chemistry is parameterised in number density |

`:fixedT`, `:adiabatic_constV` and `:adiabatic_constP` each inject an energy equation and/or an
EOS that this problem does not have.

## How to run it

The 24 MB source mechanism is **not committed**. Copy it in from the converter:

```bash
cp <kpp-cantera-converter>/examples/mcm/mcm_alkanes_alkenes_converted.yaml \
   examples/atmospheric/
```

Then — three steps; everything lands in `examples/atmospheric/output/` (gitignored, like
the source), and each step reads what the previous one wrote:

```bash
# 1. the box, either mode (span_days defaults to 8 for both):
#    frozen  -> output/frozen/{series.csv, final_state.csv, run_meta.txt}
#    diurnal -> output/diurnal/{...} — photolysis driven by the zenith clock, 60-s grid
julia --project=. examples/atmospheric/mcm_box.jl frozen
julia --project=. examples/atmospheric/mcm_box.jl diurnal
# 2. analyses on the frozen run: O3/HOx rate budgets at the final state -> budget.csv;
#    Jacobian-strategy bench (one build per strategy, solves at 0.25/0.5/1 day) -> bench_jac.csv
julia --project=. examples/atmospheric/tools/budget.jl
julia --project=. examples/atmospheric/tools/bench_jac.jl
# 3. figures + summary table (matplotlib/numpy/pandas, see tools/requirements-figures.txt)
python3 examples/atmospheric/tools/figures/fig_series.py   # -> series_frozen.png + series_diurnal.png
python3 examples/atmospheric/tools/figures/perf.py         # -> perf.png
python3 examples/atmospheric/tools/figures/table_summary.py
```

The two MCM rate types ChemMechSim's parser does not know (`zenith-angle-photolysis`,
`sigmoid-branching`) are parsed by example-side kinetics types handed to `load_mechanism`
through its `rate_type_handlers` registry (`tools/mcm_rate_types.jl`) — the sigmoid as an
EXACT closed form in T, not a frozen constant. Each photolysis A-factor materializes as the
symbolic parameter `k_{j}_A` (default J at overhead sun): that one parameter IS what makes a
single mechanism serve both modes — the frozen mode solves with its default, the diurnal mode
drives it from the clock. `series.csv` uses one schema for both modes
(`time_s,cz,J_NO2,<monitored species>`; frozen emits cz = 1 and the noon J_NO2).

After step 3, `output/` holds `frozen/` and `diurnal/` (each `series.csv` — the 7 monitored
species at 1200-s cadence — `final_state.csv` — all 1843 species, which the budget needs
because a rate law wants every reactant, not just the monitored ones — and `run_meta.txt`),
the analysis CSVs (`budget.csv`, `bench_jac.csv`), the figures
(`series_frozen`/`series_diurnal` from one script, `perf`) and `summary_table.{md,csv}`.

**Expected cost, so you don't think it hung.** The Jacobian strategy is MODE-DEPENDENT
(measured as back-to-back pairs on this box, 2026-09-24): at the frozen mode's tight
`reltol 1e-6` the reaction-sharded analytic Jacobian wins big (8-day solve ~820 s FD →
~61 s analytic, and `tools/bench_jac.jl`'s paired spans show 2–10× per-span solve
speedups at 10.9× the build cost) — but at the diurnal mode's loose `reltol 1e-4` with
its 11520 forced 60-s ticks the analytic path LOSES ~6.6× on total time (a paired A/B:
analytic 343+442 s vs FD 38+81 s; the loose tolerance needs so few Newton iterations
that the cheap-to-form FD Jacobian amortizes better). So the driver builds frozen with
`jac=true` and diurnal with `jac=false` (an optional third CLI arg `jac=true|jac=false`
overrides, for paired probing). The solver is `FBDF(autodiff = false)` — ForwardDiff
must not also run once an analytic Jacobian is supplied — and the analytic arm pins
`linsolve = LUFactorization()` (dense LU): this Jacobian is 76% dense, and FBDF's DEFAULT
sparse linsolve pays a per-linear-solve `dropzeros` COPY of the Newton matrix — measured
69 GiB/day of pure allocation — which dense LU cuts to 2.70 GiB while also running
fastest (Sparspak allocates least but loses 9× to BLAS at this density).

| stage | measured (order of magnitude) |
|---|---|
| parse the source (direct, incl. both MCM types) | ~22 s |
| lower it (`checks=false`) | ~52 s (~3.5–5.6 GiB) |
| `build_problem` frozen (`jac=true`, analytic) | ~268 s |
| `build_problem` diurnal (`jac=false`, FD) | ~38 s |
| solve 8 days, frozen (`reltol 1e-6`, analytic + dense LU) | ~56 s |
| solve 8 days, diurnal (`reltol 1e-4`, 11520 ticks, FD) | ~81 s |

**Absolute seconds move 3–6× with co-tenant load.** This box is shared, and another user
running WRF benchmarks at load average 70–100 sits between the quiet and busy columns,
differently every run — treat every number above as an order of magnitude, not a benchmark,
and never compare seconds across sessions. The load-robust, transferable quantities are the
**ratios** (same-session, from `tools/bench_jac.jl`), and even the ratios' *values* move
between sessions. Trust the per-span solve speedup, hold no single break-even day or span
verdict.

`checks=false` is **required**, not an optimisation: with `checks=true` MTK's unit validator
cannot fold this mechanism's equations and lowering did not finish in 16 minutes. The equations
are dimensionally correct; the checker just cannot prove it.

## The limits — read these before drawing conclusions

1. **The frozen mode is a perpetual noon.** Photolysis runs at J(cz = 1) for all 8 days; there
   is no day/night cycle, so night-only chemistry (NO3 / N2O5 accumulation) does not appear in
   `output/frozen/` — everything produced from `output/frozen/` (fig1, fig2, the budget, the
   summary table) is a perpetual-day result. The diurnal mode (same mechanism, same scenario)
   is the fix; its artifacts live in `output/diurnal/`. In the diurnal run, night-time trace
   species sit below the flat `abstol` and are drawn in fig3 as bounds at the resolution line —
   see the tolerance policy in the diurnal section below.

2. **Every species has `molecular_weight == 0`** — the converter emits `composition: {}` for
   nearly all of them (the phantom `M` aside; see below). Harmless in
   `state_basis=:concentration`, but mass-based output is unavailable. Concentrations only.

3. **The scenario contains no alkanes or alkenes.** `X_INIT` is copied verbatim from the
   reference config, and it initialises only N2/O2/H2O/O3/NO2/CH4 — so despite the mechanism's
   name, the box is effectively a CH4–NOx–O3 system and the mechanism's organic degradation
   chemistry stays largely idle. This is faithful to the reference scenario (which is why it
   is kept), but it is not a demonstration of the alkane/alkene chemistry. Initialise a VOC
   (e.g. `C2H6`, `C3H8`, `BUT1ENE`) to exercise that.

   What the runs *do* demonstrate, and are checked: OH/HO2 rise from zero and NO appears from
   NO2 photolysis — so the photolysis parameters really are driving the radical chemistry
   rather than being a silent no-op. In the frozen run O3 falls 1.245e-6 → 1.94e-7 mol/m³
   over 8 days (OH peaks at 2.26e-11); in the diurnal run O3 reaches ~19 ppbv at day 8 —
   perpetual noon eats O3 visibly faster than 12 h/day does (fig3's overlay). The frozen O3
   decline is the HOx cycle — O3 photolysis (`O3 => O1D`) plus the 1% H2O, then OH + O3 —
   photolysis-driven loss, which is the point the example is making.

4. **The phantom `M`.** The parser strips the literal `M` from reaction equations, but the
   source's species list keeps it, so the direct load carries 1843 species — one of which is
   an inert all-zero state that appears in `final_state.csv` and nowhere else.

## The diurnal mode — the Cantera method, ported

The converter repo does not freeze photolysis at all: its Cantera driver keeps a module-level
`ENV_STATE['zenith']` that its simulator updates once per 60-s step, and a custom
`ExtensibleRate` class (`custom_rates/generic.py`) evaluates J = l·cos(χ)^m·exp(−n/cos χ) from
it — piecewise-constant zenith, solver never restarted. Its repro config defaults to that
diurnal environment, so the reference run has a day/night cycle; the frozen box matches only
its first half-day. The diurnal mode of `mcm_box.jl` is the direct analog with **no src
change**: the flattened A-factor of every photolysis reaction is the symbolic parameter
`k_{j}_A`, so a `DiscreteCallback` on the same 60-s grid writes the same J into the same
parameters through `SymbolicIndexingInterface.setp` setters. The zenith clock
(`tools/diurnal_env.jl`) is a verbatim port, unit-tested against independently pinned values.

Three API traps found by toy-system probes before touching the real mechanism (all documented
in the driver): the sharded-jac problem's `prob.p` is an `MTKParameters` buffer with no
symbolic index — setters must be built from the bare system; a literally-true discrete-callback
condition fires at every internal step and at init, so the grid condition is `iszero(mod(t, 60))`
and t = 0 needs an explicit pre-solve J init; and the clock's 89.5° clamp pins cos χ at 0.0087
all night — night is identified geometrically (18:00–06:00), never by a cos-χ threshold.

De-risking: all 1041 parameters verified by name against their handler-materialized defaults;
the callback path agrees with an independent 360-chunk remake loop to 1e-5 (majors) / 1.5e-3
(trace, which is the restart path's own error accumulation); a J≡0 run leaves OH at 2e-37.

**Tolerance policy, stated plainly.** The diurnal run uses `reltol 1e-4` / flat `abstol 1e-12`
(the frozen run: 1e-6). Night-time OH/HO₂ troughs (~10⁻¹⁷ mol m⁻³) and all of NO₃ (peak ~10⁻¹⁸)
sit **below** `abstol`, where the solver may return anything up to ~10⁻¹² — those parts of
fig3 are drawn at the resolution line as bounds, never as physics. Resolving them needs a
per-state `abstol` of 1e-20..1e-22, which measured at >94 min of solve (vs 7.5 min) without
finishing and was abandoned. At 1e-4 the majors still match a 1e-6 run to 4–5 significant
digits, and the CH₄ decline still closes against the OH trajectory to 1.1% — but do not quote
fine percentages off this run without re-running tighter.

## What the figures show

`series_frozen.png` and `series_diurnal.png` come from ONE script (`tools/figures/
fig_series.py`) sharing a single six-panel builder — same panels, same axes, only the
forcing differs — so the two runs compare panel by panel. The runs are never overlaid in
one figure; the figures carry almost no text (axes, panel letters, species end-labels,
one forcing label, the one-word "abstol" tag on the resolution floor) — every number the
old on-figure grey blocks carried is printed by the script's QA pass instead.

**`series_frozen.png` — the frozen box relaxes toward a photochemical steady state.** The
forcing panel is flat cos χ = 1 ("perpetual noon"); O₃ falls monotonically 30 → 4.7 ppbv
over 8 days, CH₄ −1.7 % (steady loss at the perpetual-noon OH level), NO₂ exhausts, the
radical pool builds from zero to a steady level — nothing oscillates, because there is no
day/night cycle in this mode. NO₃ stays **below solver resolution** (dashed abstol line =
upper bound) — and the diurnal run shows the deeper cause: this scenario is NOx-starved
(a single 0.1-ppb NO₂ pulse, no source, HNO₃ terminal), so nights alone would not
accumulate it either.

**`series_diurnal.png` — the same box with photolysis driven by the zenith clock.** The
cos-χ forcing with nights shaded (night floor = the 89.5° clamp, cos 89.5° = 0.0087);
O₃ 30 → 18.9 ppbv with a daily sawtooth — compare `series_frozen.png`'s O₃ panel:
perpetual noon eats O₃ to 4.7 ppbv, twice the rate of 12 h/day; CH₄ −0.78 %/8 d as
daylight-only staircase steps (loss ∝ [OH]), closing against the OH trajectory to ~1 %
(implied ⟨OH⟩ 1.79×10⁶ vs direct 1.77×10⁶ molec cm⁻³); NO₂'s NOx-exhaustion decay with
NO's daylight-only pulses; the OH/HO₂ sun-synchronous pulses (day max ~10⁷, night at the
resolution floor); and NO₃, **below solver resolution** at this tolerance — the honest
conclusion is NOx starvation, not the missing night, and night-time radical troughs need a
~1e-22 per-state tolerance (see the tolerance policy above).

**`perf.png` — the analytic Jacobian pays for itself within days.** The
reaction-sharded Jacobian costs several times more to build but solves faster at every
measured span, so the total-cost break-even is shown as a range — not a number: solve
time measured non-monotonically in span on this shared box, so the figure plots measured
points, draws no fitted line, and quotes no single span verdict. The absolute seconds on it
are single-run wall-clock and move 3–6× with co-tenant load; the ratios are the transferable
quantities.

**`summary_table.md` — the run on one page, and why O₃ falls.** The final state of every
monitored species; the dominant O₃/HOₓ budget reactions *netted by equation* (the converter
splits one MCM rate expression into duplicate per-term rows, including signed negative twins,
that must be summed before any entry is quoted); and the Jacobian build/solve split. Both
budgets close — production vs loss agree to fraction-of-a-percent for O₃ and HOₓ, the sharper
test since source − sink *is* d[HOₓ]/dt — over 5600/5600 reactions covered. The budget
describes a perpetual-noon box, not a diurnal average: the ranking of the photolysis channels
is real, their absolute rates are noon rates.

## Unrelated known trap: pass a *complete* `u0`

`build_problem` does not default omitted species to zero. If you pass a partial `u0`, MTK treats
the missing initial conditions as an underdetermined initialization system and fills them by
**least squares**, handing unlisted species arbitrary values in ~[0, 1). On this mechanism that
produced a `NaN` RHS and `retcode = DtNaN` on the first solve.

Build the full dict, as the driver does:

```julia
u0 = Dict(String(sp.name) => get(X_INIT, String(sp.name), 0.0) * c_air for sp in mech.species)
```

## Upstream dependencies

Both prerequisites landed on `main` in September 2026 (PRs #36 and #35, merged via rebase):

- **PR #36** (`atmospheric-parser-and-jac`) — `load_mechanism` reading the KPP/MCM YAML
  dialect (`activation-energy: K`, `quantity: molec`, `constant-cp` thermo), and `:kinetic`
  gaining the reaction-sharded analytic Jacobian.
- **PR #35** (`meff-sharing`) — reactions with identical third-body efficiencies sharing one
  `M_eff` algebraic variable; without it, lowering this mechanism took ~33 minutes instead of
  under a minute.

The direct load itself rides `load_mechanism`'s rate-type registry (`rate_type_handlers`
keyword): the parser dispatches YAML reaction `type` strings through a default table that the
caller can extend or override, and skips unknown types with one aggregated warning naming the
keyword — so a mechanism with unhandled rate types announces itself instead of silently
loading a chemically wrong subset.
