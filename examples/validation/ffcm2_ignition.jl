# Large-mech T4: FFCM2 CH4-air ignition — ChemMechSim vs Cantera.
# Run: julia --project=. examples/validation/ffcm2_ignition.jl
# Requires examples/validation/ffcm2_ref_constV.csv (run the .py first).
#
# Status (2026-07-18): SOLVED via opaque PLOG + P differential state. FFCM2 exercises
# same-pressure PLOG nodes (Cantera sums channels per pressure); lowering uses
# checks=false for the same K_c-unit reason as aramco_ignition.jl (see that file's
# header). FBDF solves cleanly.
using ChemMechSim
using OrdinaryDiffEq: FBDF
using ModelingToolkit: unknowns, getname
using DelimitedFiles

const YAML_PATH = joinpath(@__DIR__, "..", "mechanism", "FFCM2.yaml")
const REF_CSV = joinpath(@__DIR__, "output", "ffcm2_ref_constV.csv")

_var(sys, name) = unknowns(sys)[findfirst(s -> String(getname(s)) == name, unknowns(sys))]

const R, T0, P0 = 8.314, 1500.0, 101325.0
const c_tot = P0 / (R * T0)
const X0 = Dict("CH4" => 1.0/10.52, "O2" => 2.0/10.52, "N2" => 7.52/10.52)

mech = load_mechanism(YAML_PATH)
println("Loaded FFCM2: $(length(mech.species)) species, $(length(mech.reactions)) reactions")

reactor = BatchReactor(mech; mode=:adiabatic_constV, checks=false)
sys = extract_system(reactor)
u0 = Dict(sp.name => get(X0, sp.name, 0.0) * c_tot for sp in mech.species)
u0["T"] = T0
sol = simulate(reactor, (0.0, 5.0e-3); u0=u0, solver=FBDF(), reltol=1e-8, abstol=1e-12)

# ignition delay (max |dT/dt|)
ts = range(0.0, 5.0e-3; length=2001)
Ts = [Float64(sol(t; idxs=_var(sys, "T"))) for t in ts]
dTdt = diff(Ts) ./ diff(ts)
t_ign = ts[argmax(abs.(dTdt)) + 1]
T_end = Ts[end]
println("ChemMechSim: retcode=$(sol.retcode), t_ignition=$(round(t_ign*1e3,digits=3)) ms, T_end=$(round(T_end,digits=1)) K")

has_ref = isfile(REF_CSV)
if has_ref
    data, _ = readdlm(REF_CSV, ',', header=true)
    ct_t, ct_T = vec(data[:,1]), vec(data[:,2])
    ct_t_ign = ct_t[argmax(abs.(diff(ct_T) ./ diff(ct_t))) + 1]
    rel = abs(t_ign - ct_t_ign) / ct_t_ign
    println("Cantera:     t_ignition=$(round(ct_t_ign*1e3,digits=3)) ms, T_end=$(round(ct_T[end],digits=1)) K")
    println("ignition-delay relative diff: $(round(rel*100,digits=2))%  (tol 2%)  →  ", rel < 0.02 ? "PASS" : "FAIL")
else
    println("(no Cantera ref at $REF_CSV — run examples/validation/ffcm2_ref.py first)")
end
