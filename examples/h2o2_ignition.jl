# Phase 5a example: H2-O2 ignition — ChemMechSim vs Cantera (if ref CSV present).
# Run: julia --project=. examples/h2o2_ignition.jl
# Requires examples/cantera_ref/h2o2_ref_{constV,constP}.csv (run the .py first).
using ChemMechSim
using OrdinaryDiffEq: Rodas5P
using ModelingToolkit: unknowns, getname
using DelimitedFiles

# YAML copy is in examples/data/ (kept self-contained; fallback to test/data/ if absent).
const YAML_PATH         = joinpath(@__DIR__, "data", "h2o2.yaml")
const YAML_PATH_FALLBACK = joinpath(@__DIR__, "..", "test", "data", "h2o2.yaml")
const yaml = isfile(YAML_PATH) ? YAML_PATH : YAML_PATH_FALLBACK

"Look up a named unknown on a simplified system."
_var(sys, name) = unknowns(sys)[findfirst(s -> String(getname(s)) == name, unknowns(sys))]

"Build u0 for H2:O2:N2 = 2:1:4 at T, P. `prefix` is \"\" (const-V, concentration) or \"n_\" (const-P, moles)."
function _u0(mech, prefix, T0=1000.0, P0=101325.0, V0=1.0)
    R = 8.314; c_tot = P0 / (R * T0)
    u0 = Dict{String,Float64}()
    for sp in mech.species
        X = sp.name == "H2" ? 2/7 : sp.name == "O2" ? 1/7 : sp.name == "N2" ? 4/7 : 0.0
        u0[prefix * sp.name] = X * c_tot * V0
    end
    u0["T"] = T0
    return u0
end

"Time of maximum |dT/dt| (ignition delay)."
function _t_ignition(sol, sys, n=2001)
    ts = range(0, stop=sol.t[end], length=n)
    Ts = [Float64(sol(t; idxs=_var(sys, "T"))) for t in ts]
    dTdt = diff(Ts) ./ diff(ts)
    return ts[argmax(abs.(dTdt)) + 1]
end

mech = load_mechanism(yaml)
println("Loaded mechanism: $(length(mech.species)) species, $(length(mech.reactions)) reactions")

for (mode, prefix, ref_csv) in (
        (:adiabatic_constV, "",  "examples/cantera_ref/h2o2_ref_constV.csv"),
        (:adiabatic_constP, "n_", "examples/cantera_ref/h2o2_ref_constP.csv"))
    reactor = BatchReactor(mech; mode=mode)
    sys = extract_system(reactor)
    sol = simulate(reactor, (0.0, 1e-3); u0=_u0(mech, prefix),
                   solver=Rodas5P(), reltol=1e-8, abstol=1e-12)
    t_ign = _t_ignition(sol, sys)
    T_max = Float64(sol(1e-3; idxs=_var(sys, "T")))
    println("\n[$mode] retcode=$(sol.retcode), t_ignition=$(round(t_ign*1e6, digits=2)) μs, T_max=$(round(T_max, digits=1)) K")
    if isfile(ref_csv)
        # Cantera t_ignition (max dT/dt on the CSV grid)
        data, header = readdlm(ref_csv, ',', header=true)
        t_col = data[:, 1]; T_col = data[:, 2]
        dTdt = diff(T_col) ./ diff(t_col)
        t_ign_ct = t_col[argmax(abs.(dTdt)) + 1]
        rel = abs(t_ign - t_ign_ct) / t_ign_ct
        tol = mode == :adiabatic_constV ? 0.05 : 0.08
        println("  Cantera t_ignition=$(round(t_ign_ct*1e6, digits=2)) μs, rel diff=$(round(rel*100, digits=2))% (tol $(tol*100)%)")
        println("  ", rel < tol ? "PASS" : "FAIL")
    else
        println("  (no Cantera ref at $ref_csv — run examples/cantera_ref/h2o2_ignition.py first)")
    end
end
