# Phase 5b perf benchmark for GRI30. Run: julia --project=. examples/gri30_benchmark.jl
using ChemMechSim
using ChemMechSim: lower_to_mtk, convenience_config, generate_jacobian
using ModelingToolkit: calculate_jacobian, unknowns
using OrdinaryDiffEq: FBDF, ReturnCode
using SparseArrays: nnz

const YAML = joinpath(@__DIR__, "data", "gri30.yaml")
const R, T0, P0 = 8.314, 1500.0, 101325.0
const c_tot = P0 / (R * T0)
const X0 = Dict("CH4" => 1.0/10.52, "O2" => 2.0/10.52, "N2" => 7.52/10.52)

mech = load_mechanism(YAML)
println("GRI30: $(length(mech.species)) species, $(length(mech.reactions)) reactions\n")

# 1. lowering + mtkcompile (adiabatic const-V)
t_lower = @elapsed sys = lower_to_mtk(mech; config=convenience_config(:adiabatic_constV))
n_st = length(unknowns(sys))
println("lower_to_mtk + mtkcompile (:adiabatic_constV): $(round(t_lower, digits=2)) s  ($n_st states)")

# 2. dense Jacobian (symbolic calc + codegen)
t_jc = @elapsed calculate_jacobian(sys; sparse=false)
t_jg = @elapsed generate_jacobian(sys; sparse=false)
jac_sp = calculate_jacobian(sys; sparse=true)
println("Jacobian dense calc:  $(round(t_jc, digits=2)) s")
println("Jacobian dense codegen: $(round(t_jg, digits=2)) s")
println("Jacobian nnz=$(nnz(jac_sp)), density=$(round(nnz(jac_sp)/n_st^2*100, digits=1))% (sparse calc only — do NOT sparse-codegen)\n")

# 3. CH4-air ignition solve (FBDF)
#   The solve is the FIRST simulate() in a fresh process, so a cold run is
#   dominated by one-time RHS+Jacobian function compilation (~tens of seconds).
#   We measure BOTH cold and warm to separate compilation from integration.
reactor = BatchReactor(mech; mode=:adiabatic_constV)
u0 = Dict(sp.name => get(X0, sp.name, 0.0) * c_tot for sp in mech.species); u0["T"] = T0

# Cold first solve (one-time RHS+Jacobian compilation):
t_cold = @elapsed simulate(reactor, (0.0, 5.0e-3); u0=u0, solver=FBDF(), reltol=1e-8, abstol=1e-12)
# Warm steady-state solve (functions already compiled):
sol = nothing
t_solve = @elapsed sol = simulate(reactor, (0.0, 5.0e-3); u0=u0, solver=FBDF(), reltol=1e-8, abstol=1e-12)
println("FBDF solve (5 ms):     $(round(t_solve, digits=2)) s warm  (retcode=$(sol.retcode), steps=$(length(sol)))")
println("  (cold first solve: $(round(t_cold, digits=2)) s — one-time RHS+Jacobian compilation)")

println("\nBenchmark complete.")
