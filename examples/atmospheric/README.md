# Atmospheric chemistry example — MCM alkanes/alkenes

An atmospheric box model built on the MCM alkanes/alkenes subset: 1842 species, 5600
reactions, isothermal at 298 K, fixed pressure — integrated for 3 days with frozen photolysis
and 8 days with a diurnal photolysis cycle (the Cantera-method port).

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

Then — six steps in this order; everything lands in `examples/atmospheric/output/` (gitignored,
like the source), and each step reads what the previous one wrote:

```bash
# 1. preprocessor: freeze photolysis at χ0 (default 0, overhead sun) -> derived mechanism
#    (~20 s), plus photolysis_params.csv — the l/m/n sidecar the diurnal run (step 2b) drives
julia --project=. examples/atmospheric/tools/flatten_photolysis.jl
# 2. the frozen box: 3 simulated days, jac=true -> series.csv, final_state.csv, run_meta.txt
julia --project=. examples/atmospheric/mcm_box.jl
# 2b. the diurnal box: same mechanism + scenario, photolysis driven by the zenith clock
#     (Cantera-method port), 8 simulated days -> output/diurnal/{series.csv, final_state.csv, run_meta.txt}
julia --project=. examples/atmospheric/mcm_box_diurnal.jl
# 3. O3 / HOx rate budgets at the final state -> budget.csv
julia --project=. examples/atmospheric/tools/budget.jl
# 4. Jacobian-strategy bench: one build per strategy, solves at 0.25 / 0.5 / 1 day -> bench_jac.csv
julia --project=. examples/atmospheric/tools/bench_jac.jl
# 5. figures + summary table (needs matplotlib / numpy / pandas, see tools/requirements-figures.txt)
python3 examples/atmospheric/tools/figures/fig1_series.py
python3 examples/atmospheric/tools/figures/fig3_diurnal.py
python3 examples/atmospheric/tools/figures/fig2_efficiency.py
python3 examples/atmospheric/tools/figures/table_summary.py
```

After step 5, `output/` holds the derived mechanism (`mcm_alkanes_alkenes_frozen.yaml`) and
its photolysis sidecar (`photolysis_params.csv` — the l/m/n the diurnal run drives), the
frozen run's exports (`series.csv` — the 7 monitored species at 1200 s cadence — and
`final_state.csv` — all 1842 species, which the budget needs because a rate law wants every
reactant, not just the monitored ones) plus `run_meta.txt`, the diurnal run's exports under
`output/diurnal/` (same three files, plus `cz`/`J_NO2` columns in its `series.csv`), the
analysis CSVs (`budget.csv`, `bench_jac.csv`), the three figures and the table
(`summary_table.md` + `.csv`). `flatten_photolysis.jl` takes an optional solar zenith in
degrees (default `0`, overhead sun); a different χ₀ gives a different frozen-noon box, not a
better one.

**Expected cost, so you don't think it hung.** The box runs `jac=true`: `build_problem`
builds the **reaction-sharded analytic Jacobian** instead of letting FBDF finite-difference
the RHS (~1842 RHS evaluations per Jacobian call at 1842 states), and the solver is
`FBDF(autodiff = false)` — ForwardDiff must not also run once an analytic Jacobian is
supplied. The cost is therefore a **build/solve split**, not one number:

| stage | measured (order of magnitude) |
|---|---|
| parse the derived mechanism | ~24 s quiet, ~100 s busy |
| lower it (`checks=false`) | ~56 s quiet (peak ~3.5 GiB), ~340 s busy (~3.7 GiB) |
| `build_problem`, `jac=true` | ~45 s quiet, ~300 s busy — ~4× the finite-difference build |
| solve 3 days (frozen box) | ~40 s quiet, ~300 s busy |
| solve 8 days (diurnal box, `reltol 1e-4`) | ~450 s busy (≈56 s/day — the 11520 forced 60-s step ends cost little vs the frozen box's ~100 s/day) |

**Absolute seconds move 3–6× with co-tenant load.** This box is shared, and another user
running WRF benchmarks at load average 70–100 sits between the two columns, differently every
run — treat every number above as an order of magnitude, not a benchmark, and never compare
seconds across sessions. The load-robust, transferable quantities are the **ratios**
(same-session, from `tools/bench_jac.jl`), and even the ratios' *values* move between
sessions: across two measured sessions the build cost was **4.0–4.3×** and the per-span solve
speedup **4.4–36×**. The total-cost break-even is always a **range, not a number**, because
solve time measured *non-monotonically* in span under this load (longer spans sometimes
measured faster) — observed **1.05–3.76 days** in one session and **0.8–1.0 days** in
another. Those two sessions genuinely disagree on whether this example's 3-day span has a
clear winner: inside the first range (no verdict), past the second (analytic wins outright).
Trust the per-span solve speedup, hold no single break-even day or 3-day verdict.

`checks=false` is **required**, not an optimisation: with `checks=true` MTK's unit validator
cannot fold this mechanism's equations and lowering did not finish in 16 minutes. The equations
are dimensionally correct; the checker just cannot prove it.

## The limits — read these before drawing conclusions

1. **The frozen run has no diurnal cycle — the diurnal run (step 2b) is the fix.** The
   preprocessor evaluates the 1041 `zenith-angle-photolysis` reactions once at a fixed zenith
   χ₀ and emits constant Arrhenius rates **whose A-factors are symbolic parameters**; that is
   what makes the frozen file the carrier for BOTH modes. `mcm_box.jl` solves it as a
   perpetual day; `mcm_box_diurnal.jl` drives the same parameters from the zenith clock (see
   the next section). Everything produced from `output/` at the top level (fig1, fig2, the
   budget, the summary table) is a perpetual-day result; the diurnal artifacts live in
   `output/diurnal/`. Re-running the frozen box with a different `χ₀` gives a different
   frozen-noon box, not a better one.

2. **The 2 `sigmoid-branching` entries are exact only at T₀ = 298 K.** Each is flattened to a
   constant, evaluated at T₀. That is exact at the operating point, but it is not a temperature
   law — if you change T, re-run the preprocessor.

   Note these two entries are *not* a standalone pair. They are the signed correction and the
   σ channel of `CH3O2 + HO2`; the third, plain-Arrhenius entry carrying `k_total` is passed
   through untouched, and the three sum to MCM's own rate. (An earlier version of the
   preprocessor paired the two sigmoids and invented a `k_total` term, running that reaction at
   2× MCM's rate. See the warning in `_flatten_sigmoid`.)

3. **Every species has `molecular_weight == 0`** — the converter emits `composition: {}` for all
   of them. Harmless in `state_basis=:concentration`, but it means mass-based or mixing-ratio
   output is unavailable. Concentrations only.

4. **The derived mechanism is a build artifact, not a source of truth.** It is gitignored, and
   it drops the source file's ~16 800 provenance comment lines. Regenerate it rather than
   editing it.

5. **The scenario contains no alkanes or alkenes.** `X_INIT` is copied verbatim from the
   reference config, and it initialises only N2/O2/H2O/O3/NO2/CH4 — so despite the mechanism's
   name, the box is effectively a CH4–NOx–O3 system and the mechanism's organic degradation
   chemistry stays largely idle. This is faithful to the reference scenario (which is why it is
   kept), but it is not a demonstration of the alkane/alkene chemistry. Initialise a VOC (e.g.
   `C2H6`, `C3H8`, `BUT1ENE`) to exercise that.

   What the run *does* demonstrate, and is checked: OH/HO2 rise from zero and NO appears from NO2
   photolysis — so the flattened photolysis really is driving the radical chemistry rather than
   being a silent no-op.

   O3 also **falls** (1.245e-6 → 6.188e-7 mol/m³ over 3 days; OH peaks at 2.26e-11). That is
   *not* NOx titration: total NOx is 1e-10 mole fraction ≈ 4.2e-9 mol/m³, some 150× smaller than
   the O3 decline. It is the HOx cycle — O3 photolysis (`O3 => O1D` at 3.78e-5 s⁻¹, ~12% of
   which gives OH) plus the 1% H2O, then OH + O3. Photolysis-driven loss, which is the point the
   example is making.

## The diurnal run (step 2b) — the Cantera method, ported

The converter repo does not freeze photolysis at all: its Cantera driver keeps a module-level
`ENV_STATE['zenith']` that its simulator updates once per 60-s step, and a custom
`ExtensibleRate` class (`custom_rates/generic.py`) evaluates J = l·cos(χ)^m·exp(−n/cos χ) from
it — piecewise-constant zenith, solver never restarted. Its repro config defaults to that
diurnal environment, so the reference run has a day/night cycle; the frozen box matches only
its first half-day. `mcm_box_diurnal.jl` is the direct analog with **no src change**: the
flattened A-factor of every photolysis reaction is the symbolic parameter `k_{j}_A`, so a
`DiscreteCallback` on the same 60-s grid writes the same J into the same parameters through
`SymbolicIndexingInterface.setp` setters. The zenith clock (`tools/diurnal_env.jl`) is a
verbatim port, unit-tested against independently pinned values.

Three API traps found by toy-system probes before touching the real mechanism (all documented
in the driver): the sharded-jac problem's `prob.p` is an `MTKParameters` buffer with no
symbolic index — setters must be built from the bare system; a literally-true discrete-callback
condition fires at every internal step and at init, so the grid condition is `iszero(mod(t, 60))`
and t = 0 needs an explicit pre-solve J init; and the clock's 89.5° clamp pins cos χ at 0.0087
all night — night is identified geometrically (18:00–06:00), never by a cos-χ threshold.

De-risking: all 1041 parameters verified by name against their frozen defaults; the callback
path agrees with an independent 360-chunk remake loop to 1e-5 (majors) / 1.5e-3 (trace, which
is the restart path's own error accumulation); a J≡0 run leaves OH at 2e-37.

**Tolerance policy, stated plainly.** The diurnal run uses `reltol 1e-4` / flat `abstol 1e-12`
(the frozen run: 1e-6). Night-time OH/HO₂ troughs (~10⁻¹⁷ mol m⁻³) and all of NO₃ (peak ~10⁻¹⁸)
sit **below** `abstol`, where the solver may return anything up to ~10⁻¹² — those parts of
fig3 are drawn at the resolution line as bounds, never as physics. Resolving them needs a
per-state `abstol` of 1e-20..1e-22, which measured at >94 min of solve (vs 7.5 min) without
finishing and was abandoned. At 1e-4 the majors still match the 1e-6 run to 4–5 significant
digits, and the CH₄ decline still closes against the OH trajectory to 1.1% — but do not quote
fine percentages off this run without re-running tighter.

## What the figures show

Everything below — figures and table alike — is a **frozen-photolysis (perpetual-day)
result**: photolysis is frozen at χ₀ = 0, so there is no diurnal cycle (limit 1 above). A
figure that gets separated from this README must not be mistaken for a diurnal run, which is
why the same caveat is printed in each figure caption.

**Figure 1, `fig1_series.png` — the box relaxes to a photochemical steady state.** O₃ falls
from its initial 30 ppbv toward ~15 ppbv while the radical pool (OH, HO₂) builds from zero
and levels off; nothing oscillates, because there is no day/night cycle to follow in this
setup. NO₃ never accumulates and is drawn as an annotation rather than a curve — though fig3's
diurnal run later showed the deeper cause: this scenario is NOx-starved (a single 0.1-ppb NO₂
pulse, no source, HNO₃ terminal), so NO₃ would stay ≲4 molec cm⁻³ even with nights. The
frozen day suppressed the *cycle*, not the NO₃.

**Figure 2, `fig2_efficiency.png` — the analytic Jacobian pays for itself within days.** The
reaction-sharded Jacobian costs ~4–4.3× more to build (session-dependent) but solves
4.4–36× faster at every measured span, so the total-cost break-even is shown as a range
(e.g. 1.05–3.76 simulated days in one session, 0.8–1.0 in another) — not a number: solve
time measured non-monotonically in span on this shared box, so the figure plots measured
points, draws no fitted line, and quotes no single 3-day verdict. The absolute seconds on it
are single-run wall-clock and move 3–6× with co-tenant load; the ratios are the transferable
quantities.

**Figure 3, `fig3_diurnal.png` — the same box with photolysis driven by the zenith clock
(8 days).** Six panels: the cos-χ forcing with nights shaded; O₃ alone on a linear axis
(30 → 19 ppbv with a daily sawtooth, and the frozen run overlaid as a dashed line — perpetual
noon eats O₃ visibly faster than 12 h/day); CH₄ alone on a tight linear axis, where the
−0.78 %/8 d decline appears as daylight-only staircase steps (loss ∝ [OH]) and closes against
the OH trajectory to 1.1% — ⟨OH⟩ implied 1.79×10⁶ vs 1.77×10⁶ direct; NO₂'s NOx-exhaustion
decay with NO's daylight-only pulses; the OH/HO₂ sun-synchronous pulses (day max ~10⁷ vs
night at the resolution floor); and NO₃, which is **below solver resolution** at this
tolerance — drawn at the abstol line as an upper bound. The honest NO₃ conclusion: this
scenario's NOx starvation, not the missing night, is what keeps it small; and night-time
radical troughs cannot be resolved without a ~1e-22 per-state tolerance (see the tolerance
policy above).

**`summary_table.md` — the run on one page, and why O₃ falls.** The final state of every
monitored species; the dominant O₃/HOₓ budget reactions *netted by equation* (the converter
splits one MCM rate expression into duplicate per-term rows, including signed negative twins,
that must be summed before any entry is quoted); and the Jacobian build/solve split. Both
budgets close — production vs loss agree to −0.60% for O₃ and −0.005% for HOₓ, the sharper
test since source − sink *is* d[HOₓ]/dt — over 5600/5600 reactions covered. The budget
describes a perpetual-noon box, not a diurnal average: the ranking of the photolysis channels
is real, their absolute rates are noon rates.

## Unrelated known trap: pass a *complete* `u0`

`build_problem` does not default omitted species to zero. If you pass a partial `u0`, MTK treats
the missing initial conditions as an underdetermined initialization system and fills them by
**least squares**, handing unlisted species arbitrary values in ~[0, 1). On this mechanism that
produced a `NaN` RHS and `retcode = DtNaN` on the first solve.

Build the full dict, as the other large-mechanism examples already do
(`gri30_ignition.jl:28`, `aramco_ignition.jl:39`, `ffcm2_ignition.jl:28`):

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
