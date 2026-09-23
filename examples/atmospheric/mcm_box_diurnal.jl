# Atmospheric box model with a DIURNAL photolysis cycle: MCM alkanes/alkenes, isothermal,
# fixed pressure, 8 days.
#
#   julia --project=. examples/atmospheric/mcm_box_diurnal.jl [span_days]
#
# This is the Cantera-method counterpart of mcm_box.jl (which freezes photolysis at χ0 = 0, a
# perpetual day). The converter repo drives photolysis through custom rate classes that read a
# module-level ENV_STATE['zenith'], updated by its simulator once per 60-s step — piecewise-
# constant zenith, solver NOT restarted across steps (custom_rates/generic.py
# ZenithAnglePhotolysisRate; cantera_sim/environment.py MCMDiurnalEnvironment; cantera_sim/
# simulator.py run()). Its own repro config defaults to that diurnal environment, so the
# reference run HAS a day/night cycle; the frozen box matches only its first half-day.
#
# The ChemMechSim analog needs NO src change: flattening materializes each photolysis reaction
# as an elementary Arrhenius whose A-factor is the SYMBOLIC PARAMETER k_{j}_A (b = Ea = 0), so
# J IS the parameter value. This driver updates those 1041 parameters from the same zenith
# formula, on the same 60-s cadence, through a DiscreteCallback — ENV_STATE's exact
# counterpart. Mode, u0 convention and solver match mcm_box.jl; tolerances are LOOSER here
# (reltol 1e-4 vs the frozen run's 1e-6) — see the tolerance-policy note at the solve.

using ChemMechSim
using ModelingToolkit
using ModelingToolkit: getname, parameters, setp
using OrdinaryDiffEq
using Printf

include(joinpath(@__DIR__, "tools", "diurnal_env.jl"))

const MECH  = joinpath(@__DIR__, "output", "mcm_alkanes_alkenes_frozen.yaml")
const SIDE  = joinpath(@__DIR__, "output", "photolysis_params.csv")
const T0    = 298.0          # K
const P0    = 102858.0       # Pa
const DT_STEP = 60.0         # s — the converter's simulator step (ENV_STATE update cadence)
const R_GAS = 8.314          # J/(mol·K)
# 8 days shows spin-up plus several repeated cycles. ARGS[1] lets probes run short spans.
const T_END = (isempty(ARGS) ? 8.0 : parse(Float64, ARGS[1])) * 86400.0

# Same scenario as mcm_box.jl / the converter's repro config.
const X_INIT = Dict("N2" => 0.78, "O2" => 0.21, "H2O" => 0.01,
                    "O3" => 3.0e-8, "NO2" => 1.0e-10, "CH4" => 1.8e-6)
const MONITOR = ["O3", "NO", "NO2", "NO3", "OH", "HO2", "CH4"]

isfile(MECH) || error("mcm_box_diurnal: run the preprocessor first (flatten_photolysis.jl):\n  $MECH")
isfile(SIDE) || error("mcm_box_diurnal: photolysis sidecar not found at\n  $SIDE\n" *
                      "Re-run tools/flatten_photolysis.jl (it emits the sidecar).")

# --- photolysis sidecar: (reaction index, equation, l, m, n) ---------------------------------
struct PhotoRate
    j::Int          # 1-based position in the mechanism's reaction list
    equation::String
    l::Float64
    m::Float64
    n::Float64
end

photos = PhotoRate[]
for (i, line) in enumerate(eachline(SIDE))
    i == 1 && continue                                   # header
    parts = split(strip(line), ',')
    length(parts) == 5 ||
        error("mcm_box_diurnal: malformed sidecar line $i: ", line)
    push!(photos, PhotoRate(parse(Int, parts[1]), String(parts[2]),
                            parse(Float64, parts[3]), parse(Float64, parts[4]),
                            parse(Float64, parts[5])))
end
println("photolysis sidecar: ", length(photos), " reactions (l/m/n for J = l·cz^m·exp(−n/cz))")

# --- mechanism -------------------------------------------------------------------------------
println("loading ", basename(MECH), " ...")
t_parse = @elapsed mech = load_mechanism(MECH)
@printf("  %d species, %d reactions  (%.1f s)\n",
        length(mech.species), length(mech.reactions), t_parse)

t_low = @elapsed r = BatchReactor(mech; mode=:kinetic, checks=false, name=:mcm_diurnal)
@printf("  lowered in %.1f s\n", t_low)

sys = extract_system(r)
ps = parameters(sys)

# --- photolysis parameter mapping + ORDER GUARD ----------------------------------------------
# Resolve each k_{j}_A BY NAME (the sharded-Jacobian lesson: parameter ordering is not something
# to assume — commit 8277716 fixed a 20-50-orders-of-magnitude bug from exactly that). j is the
# 1-based enumerate index over mech.reactions (src/lowering/core.jl), which is the sidecar's
# reaction_index by construction. The value guard then closes the loop end-to-end: the frozen
# YAML's A IS J(χ0=0) = l·exp(−n), so if names, ordering, or values drift, this errors BEFORE
# the solve rather than producing a plausible-but-wrong run.
pindex = Dict(String(getname(p)) => i for (i, p) in enumerate(ps))
photo_syms = Any[]
for pr in photos
    name = "k_$(pr.j)_A"
    haskey(pindex, name) ||
        error("mcm_box_diurnal: no parameter $name — reaction index j is not the parameter ",
              "index for ", pr.equation, "; the sidecar contract broke")
    push!(photo_syms, ps[pindex[name]])
end
length(unique(photo_syms)) == length(photo_syms) ||
    error("mcm_box_diurnal: duplicate k_{j}_A mapping across ", length(photo_syms),
          " sidecar rows")

# Value guard against the SYMBOLIC DEFAULTS (prob.p is an MTKParameters buffer, not a flat
# vector — never index it positionally; the sharded-Jacobian param-order lesson again). The
# frozen YAML's A IS J(χ0=0) = l·exp(−n), so if names or values drifted, this errors BEFORE
# the solve rather than producing a plausible-but-wrong run.
for pr in photos
    sym = ps[pindex["k_$(pr.j)_A"]]
    expected = photolysis_J(pr.l, pr.m, pr.n, 1.0)
    isapprox(ModelingToolkit.getdefault(sym), expected; rtol = 1e-12) ||
        error("mcm_box_diurnal: k_$(pr.j)_A defaults to ", ModelingToolkit.getdefault(sym),
              " but the frozen A for ", pr.equation, " is ", expected,
              " — parameter mapping is wrong; not solving")
end
@printf("  all %d photolysis parameters verified against their frozen defaults\n",
        length(photo_syms))

# Setters built from the BARE SYSTEM (SymbolicIndexingInterface.setp): the sharded-jac path
# wraps the problem in a hand-built ODEProblem that carries no symbolic index, but a
# ParameterIndex from `sys` applies to any target whose parameter buffer shares the layout —
# the problem AND its integrators alike (~0.2-0.4 us per call ⇒ 1041 writes × 11520 ticks ≈
# 3-5 s across the whole run; mini-verified on a toy system before trusting it here).
const KSETTERS = [setp(sys, sym) for sym in photo_syms]

# --- initial conditions (COMPLETE u0 — same convention and reason as mcm_box.jl) -------------
const C_AIR = P0 / (R_GAS * T0)
u0 = Dict(String(sp.name) => get(X_INIT, String(sp.name), 0.0) * C_AIR
          for sp in mech.species)
Tparam = ps[findfirst(p -> String(getname(p)) == "T", ps)]

println("building problem (jac=true) ...")
t_build = @elapsed prob = build_problem(r, u0, (0.0, T_END); params=[Tparam => T0], jac=true)
@printf("  built in %.1f s\n", t_build)

# Initial J's at t = 0 (MIDNIGHT: cz = cos(89.5°)). Pre-solve setter writes on the problem DO
# reach the solve (mini-verified) — and the mod-grid callback below does NOT fire at t = 0
# (discrete callbacks fire at tstops; only a literally-true condition is evaluated during
# initialization), so without this block the first 60 s would run at the frozen NOON defaults.
cz0 = cos_zenith(0.0)
for i in eachindex(photos)
    KSETTERS[i](prob, photolysis_J(photos[i].l, photos[i].m, photos[i].n, cz0))
end

# --- the diurnal callback (ENV_STATE's counterpart) -------------------------------------------
# Condition `iszero(mod(t, DT_STEP))`: discrete callbacks with a literally-TRUE condition are
# evaluated after EVERY step AND during initialization (mini-verified: 143 fires on a 2-s toy
# problem), which would update J at solver-chosen times; the mod test restricts firing to
# exactly the 60-s grid — tstops force step ends there and the solver lands on a tstop exactly,
# so the float test is safe (and t = 0 is handled by the explicit pre-solve init above). This
# matches the converter's ENV_STATE cadence: J piecewise-constant per 60-s step, solver NOT
# restarted across ticks (error control absorbs the small jumps). save_positions=(false,false):
# the affect is a parameter change, not a state event — nothing to save.
const N_PHOTO = length(photos)
function apply_photolysis!(integ)
    cz = cos_zenith(integ.t)
    @inbounds for i in 1:N_PHOTO
        pr = photos[i]
        KSETTERS[i](integ, photolysis_J(pr.l, pr.m, pr.n, cz))
    end
    return nothing
end
cb = DiscreteCallback((u, t, integ) -> iszero(mod(t, DT_STEP)), apply_photolysis!;
                      save_positions = (false, false))
tstops = DT_STEP:DT_STEP:T_END

# Maps a species name to its row in the state vector (needed by the tolerance vector below
# and by the exports/report after the solve).
state_index = Dict(String(getname(u)) => i
                   for (i, u) in enumerate(ModelingToolkit.unknowns(sys)))
length(state_index) == length(mech.species) ||
    error("mcm_box_diurnal: state has $(length(state_index)) unknowns vs ",
          "$(length(mech.species)) species — exports would be incomplete")

# --- solve -----------------------------------------------------------------------------------
# TOLERANCE POLICY — the trade-off that was actually taken, not the ideal one. Night-time
# trace species sit BELOW the flat abstol (OH's night trough ~7e-17 mol/m^3, NO3's peak
# ~7e-18, both vs abstol 1e-12): the solver is free to return anything up to ~1e-12 for them,
# so their night values (and all of NO3) are NOT resolved and the figure floors them at the
# tolerance line instead. Resolving them needs a per-state abstol ~1e-20..1e-22 — measured:
# 1e-22 ran >94 min of solve (8x+ the flat-tolerance solve) without finishing, so it was
# abandoned for this fast-iteration configuration. reltol is loosened to 1e-4 (the frozen run
# used 1e-6) to keep the solve in the few-minute range; the visible-cycle quantities (OH/HO2
# pulse shapes, O3 sawtooth, CH4 day-steps) are insensitive at display scale, but do NOT
# quote fine percentages off this run without re-running tighter.
@printf("solving %.1f days (dt_step = %.0f s, reltol 1e-4, abstol 1e-12) ...\n",
        T_END / 86400, DT_STEP)

t_solve = @elapsed sol = solve(prob, FBDF(autodiff = false);
                               reltol = 1e-4, abstol = 1e-12, saveat = 1200.0,
                               callback = cb, tstops = tstops)
@printf("  solved in %.1f s, retcode = %s\n", t_solve, sol.retcode)

# --- exports (output/diurnal/ — the frozen artifacts are consumed by fig1/fig2) ---------------
const OUTDIR = joinpath(@__DIR__, "output", "diurnal")
mkpath(OUTDIR)

# J_NO2 = the sidecar's NO2 photolysis row (the reference J for figures and cross-checks).
i_jno2 = findfirst(pr -> pr.equation == "NO2 => NO + O", photos)
i_jno2 === nothing && error("mcm_box_diurnal: no 'NO2 => NO + O' photolysis row in the sidecar")
jno2 = photos[i_jno2]

# 1200 s is a whole multiple of the 60-s tick, so the formula evaluated AT a save time IS the
# applied piecewise value — no interpolation needed.
const OUT_CSV = joinpath(OUTDIR, "series.csv")
open(OUT_CSV, "w") do io
    println(io, join(vcat("time_s", "cz", "J_NO2", MONITOR), ","))
    for (k, t) in enumerate(sol.t)
        cz = cos_zenith(t)
        row = Any[t, cz, photolysis_J(jno2.l, jno2.m, jno2.n, cz)]
        for name in MONITOR
            push!(row, sol.u[k][state_index[name]])
        end
        println(io, join(row, ","))
    end
end
println("wrote ", OUT_CSV, "  (", length(sol.t), " rows)")

const OUT_STATE = joinpath(OUTDIR, "final_state.csv")
open(OUT_STATE, "w") do io
    println(io, "species,concentration_mol_m3")
    for sp in mech.species
        println(io, String(sp.name), ",", sol.u[end][state_index[String(sp.name)]])
    end
end
println("wrote ", OUT_STATE, "  (", length(mech.species), " species)")

open(joinpath(OUTDIR, "run_meta.txt"), "w") do io
    println(io, "diurnal=true")
    println(io, "dt_step_s=", DT_STEP)
    println(io, "reltol=1e-4")      # the frozen run used 1e-6; see the tolerance-policy note
    println(io, "abstol=1e-12")     # trace species below this are UNRESOLVED (floored in fig3)
    println(io, "span_days=", T_END / 86400)
    println(io, "t_lower_s=", round(t_low, digits = 1))
    # Same caveat as mcm_box.jl: build+solve COMBINED, and solve here includes the callback's
    # 11520 parameter ticks. Not comparable to bench_jac.csv's split measurements.
    println(io, "t_simulate_s=", round(t_build + t_solve, digits = 1))
    println(io, "jac=true")
    # Process-LIFETIME peak (Sys.maxrss high-water mark), dominated by lowering+codegen — same
    # caveat as mcm_box.jl's run_meta.
    println(io, "peak_rss_gib=", round(Sys.maxrss() / 2^30, digits = 2))
end

# --- report + chemistry checks ---------------------------------------------------------------
series(name) = [u[state_index[name]] for u in sol.u]
println("\nspecies        initial [mol/m^3]      final [mol/m^3]")
for name in MONITOR
    v = series(name)
    @printf("  %-8s %18.6e %18.6e\n", name, v[1], v[end])
end

# Hard check (same role as mcm_box.jl's): days have light, so OH must build from zero. A silent
# callback no-op or a midnight-only run would leave every radical at exactly zero.
oh = series("OH")
maximum(oh) > 0.0 ||
    error("mcm_box_diurnal: OH stayed at zero — the photolysis callback is not wired ",
          "(check apply_photolysis! and the parameter guard above)")
@printf("\nOH peak = %.3e mol/m^3\n", maximum(oh))

# Soft report — the headline of this run. The frozen box's NO3 peaked at ~3 molec/cm³ BECAUSE it
# had no night; the diurnal box must accumulate NO3 during the dark half-cycles.
no3 = series("NO3")
no3_max, no3_at = maximum(no3), sol.t[argmax(no3)]
# Night test is GEOMETRIC, not a cz threshold: the clock's 89.5° clamp keeps cz at 0.0087 all
# night, so any threshold above that also catches dusk/dawn, and any below it never fires.
tod = no3_at % 86400.0
@printf("NO3 peak = %.3e mol/m^3 (%.2f molec/cm^3) at t = %.2f d [%s]\n",
        no3_max, no3_max * 6.02214076e23 / 1e6, no3_at / 86400,
        (tod <= 21600.0 || tod >= 64800.0) ? "night ✓" : "DAY — unexpected")
