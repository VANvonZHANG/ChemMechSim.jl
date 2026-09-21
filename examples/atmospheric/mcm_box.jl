# Atmospheric box model: MCM alkanes/alkenes, isothermal, fixed pressure, 3 days.
#
#   julia --project=. examples/atmospheric/mcm_box.jl
#
# Run tools/flatten_photolysis.jl first — it produces the mechanism this reads.
#
# Mode is :kinetic (the MechanismConfig() zero point), which is the right — and the only
# correct — choice for an atmospheric box model:
#   * T is given by the scenario (298 K), never solved from an energy balance;
#   * P is given (102858 Pa) and the state basis is concentration, so no EOS is needed;
#   * MCM emits only forward rates (every reaction is `=>`) and its thermo data is uniformly
#     zero, so a reverse rate could not be formed anyway.
# The other convenience modes (:fixedT, :adiabatic_constV, :adiabatic_constP) would each
# inject an energy equation and/or an EOS that this problem does not have.
#
# Scenario values are copied verbatim from the converter's own
# mcm_alkanes_alkenes_repro_config.yaml so this run is comparable to its Cantera reference.
#
# Caveat this example cannot escape: photolysis is FROZEN (see the README). The diurnal
# cycle is absent, so this is a perpetual-day box — night-only chemistry (NO3 / N2O5
# accumulation) will not appear, and the species listed below settle to a steady daytime
# state rather than cycling.

using ChemMechSim
using ModelingToolkit
using ModelingToolkit: getname, parameters
using OrdinaryDiffEq
using Printf

const MECH  = joinpath(@__DIR__, "output", "mcm_alkanes_alkenes_frozen.yaml")
const T0    = 298.0          # K
const P0    = 102858.0       # Pa
const T_END = 259200.0       # s = 3 days
const R_GAS = 8.314          # J/(mol·K)

# Mole fractions, from the reference config.
const X_INIT = Dict("N2" => 0.78, "O2" => 0.21, "H2O" => 0.01,
                    "O3" => 3.0e-8, "NO2" => 1.0e-10, "CH4" => 1.8e-6)

# The reference config's monitor list.
const MONITOR = ["O3", "NO", "NO2", "NO3", "OH", "HO2", "CH4"]

isfile(MECH) ||
    error("mcm_box: derived mechanism not found at\n  $MECH\n" *
          "Run the preprocessor first:\n" *
          "  julia --project=. examples/atmospheric/tools/flatten_photolysis.jl")

# --- initial conditions -------------------------------------------------------------------
# X_INIT is mole fractions; the state basis is concentration [mol/m^3], so c = X·P/(R·T).
# EVERY species is listed explicitly — the 1836 not in X_INIT get exactly 0.0. Passing a
# partial u0 and relying on MTK to default the rest is NOT safe here: observed doing so, the
# unlisted species came back with arbitrary non-zero values (e.g. TBUTCO3 = 0.827) and the
# integration blew up with NaN on the first solve.
const C_AIR = P0 / (R_GAS * T0)                       # total concentration, mol/m^3
@printf("T = %.1f K, P = %.0f Pa, c_air = %.3f mol/m^3\n", T0, P0, C_AIR)

# --- mechanism ----------------------------------------------------------------------------
println("loading ", basename(MECH), " ...")
t_parse = @elapsed mech = load_mechanism(MECH)
@printf("  %d species, %d reactions  (%.1f s)\n",
        length(mech.species), length(mech.reactions), t_parse)

# checks=false is REQUIRED here, not an optimisation: with checks=true, MTK's unit validator
# cannot fold this mechanism's equations and lowering did not finish in 16 minutes. The
# equations are dimensionally correct; the check just cannot prove it.
t_low = @elapsed r = BatchReactor(mech; mode=:kinetic, checks=false, name=:mcm_box)
@printf("  lowered in %.1f s  (peak %.2f GiB)\n", t_low, Sys.maxrss() / 2^30)
println("  ", r)

# Built after the mechanism is loaded, so it can name every species.
u0 = Dict(String(sp.name) => get(X_INIT, String(sp.name), 0.0) * C_AIR
          for sp in mech.species)

# --- solve ---------------------------------------------------------------------------------
sys = extract_system(r)
Tparam = parameters(sys)[findfirst(p -> String(getname(p)) == "T", parameters(sys))]

println("solving ", T_END / 86400, " days ...")
t_solve = @elapsed sol = simulate(
    r, (0.0, T_END);
    u0 = u0, params = [Tparam => T0],
    # jac=true uses the reaction-sharded analytic Jacobian instead of the finite-difference one
    # FBDF would otherwise form (~1842 RHS evaluations per Jacobian at 1842 states). Measured on
    # this mechanism: build_problem 42.5 s -> 300.3 s, but solve 145.3 s -> 31.1 s per 0.5
    # simulated days. Break-even is ~1.13 days, so a 3-day run is ~1.9x faster overall.
    # autodiff=false stays: an analytic Jacobian is supplied, so ForwardDiff must not also run.
    jac = true,
    solver = FBDF(autodiff = false),
    reltol = 1e-6, abstol = 1e-12, saveat = 1200.0)
@printf("  solved in %.1f s, retcode = %s\n\n", t_solve, sol.retcode)

# Maps a species name to its row in the state vector; used by the export and the report below.
state_index = Dict(String(getname(u)) => i
                   for (i, u) in enumerate(ModelingToolkit.unknowns(sys)))

# --- export for the Python figures ---------------------------------------------------------
# Long format would be tidier, but wide matches how the figures are drawn (one line per species
# against time), and there are only 7 monitored species.
const OUT_CSV = joinpath(@__DIR__, "output", "series.csv")
mkpath(dirname(OUT_CSV))
open(OUT_CSV, "w") do io
    println(io, join(vcat("time_s", MONITOR), ","))
    for (k, t) in enumerate(sol.t)
        row = [t]
        for name in MONITOR
            push!(row, sol.u[k][state_index[name]])
        end
        println(io, join(row, ","))
    end
end
println("wrote ", OUT_CSV, "  (", length(sol.t), " rows)")

open(joinpath(@__DIR__, "output", "run_meta.txt"), "w") do io
    println(io, "t_lower_s=", round(t_low, digits = 1))
    # NOT pure solve time: `simulate` runs build_problem (which with jac=true builds the
    # reaction-sharded Jacobian, ~300 s on this mechanism) and then solve. tools/bench_jac.jl
    # measures the two separately — use that for anything comparing strategies.
    println(io, "t_simulate_s=", round(t_solve, digits = 1))
    println(io, "jac=true")
    println(io, "span_days=", T_END / 86400)
    # NOT the lowering peak: Sys.maxrss() is process-LIFETIME peak RSS, a high-water mark since
    # process start, and this is written after the solve — so it is dominated by the solve and
    # the Jacobian codegen. Same process printed ~3.5 GiB during lowering (the value in
    # examples/atmospheric/README.md); this number is strictly larger and not comparable to it.
    println(io, "peak_rss_gib=", round(Sys.maxrss() / 2^30, digits = 2))
end

# --- report --------------------------------------------------------------------------------
# Read the trajectory straight out of sol.u. The DE solution's `sol[i, j]` indexes
# (component, timestep) — the opposite order from the intuitive reading — so `sol[1:end, k]`
# is every species at time k, not the k-th species over time. Reading sol.u avoids the trap.
series(name) = [u[state_index[name]] for u in sol.u]

println("species        initial [mol/m^3]      final [mol/m^3]")
for name in MONITOR
    haskey(state_index, name) || continue
    v = series(name)
    @printf("  %-8s %18.6e %18.6e\n", name, v[1], v[end])
end

# The 1041 photolysis reactions are the radical source. Without them the box would sit at its
# initial zeros forever — so assert the chemistry actually ran rather than printing a table of
# zeros and calling it success.
oh = series("OH")
maximum(oh) > 0.0 ||
    error("mcm_box: OH stayed at zero. Either the photolysis transform silently no-opped " *
          "(re-run tools/flatten_photolysis.jl and check its reported counts), or the run is " *
          "at a zenith where every J is zero — check the preprocessor's χ0 argument.")
@printf("\nOH peak = %.3e mol/m^3  (radical source is active)\n", maximum(oh))
