# Phase 5b example: GRI30 CH4-air ignition — ChemMechSim vs Cantera + CairoMakie plot.
# Run: julia --project=. examples/validation/gri30_ignition.jl
# Requires examples/validation/gri30_ref_constV.csv (run the .py first).
using ChemMechSim
using OrdinaryDiffEq: FBDF
using ModelingToolkit: unknowns, getname
using DelimitedFiles
using CairoMakie

const YAML_PATH = joinpath(@__DIR__, "..", "mechanism", "gri30.yaml")
const YAML_FALLBACK = joinpath(@__DIR__, "..", "..", "test", "data", "gri30.yaml")
const yaml = isfile(YAML_PATH) ? YAML_PATH : YAML_FALLBACK
const OUT = joinpath(@__DIR__, "output")
const REF_CSV = joinpath(OUT, "gri30_ref_constV.csv")
const PNG_OUT = joinpath(OUT, "gri30_ignition.png")
mkpath(OUT)

_var(sys, name) = unknowns(sys)[findfirst(s -> String(getname(s)) == name, unknowns(sys))]

const R, T0, P0 = 8.314, 1500.0, 101325.0
const c_tot = P0 / (R * T0)
const X0 = Dict("CH4" => 1.0/10.52, "O2" => 2.0/10.52, "N2" => 7.52/10.52)

mech = load_mechanism(yaml)
println("Loaded GRI30: $(length(mech.species)) species, $(length(mech.reactions)) reactions")

reactor = BatchReactor(mech; mode=:adiabatic_constV)
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
    println("ignition-delay relative diff: $(round(rel*100,digits=2))%  (tol 10%)  →  ", rel < 0.10 ? "PASS" : "FAIL")
else
    println("(no Cantera ref at $REF_CSV — run examples/validation/gri30_ref.py first)")
end

# 2-panel plot: T(t) vs Cantera + key species (CH4, O2, CO2, OH)
fig = Figure(size=(1000, 700))
axT = Axis(fig[1, 1:2]; xlabel="time (ms)", ylabel="T (K)", title="GRI30 CH4-air ignition (const-V adiabatic)")
lines!(axT, ts .* 1e3, Ts; label="ChemMechSim", linewidth=2)
has_ref && lines!(axT, ct_t .* 1e3, ct_T; linestyle=:dash, label="Cantera", linewidth=2)
axislegend(axT; position=:lt)
totals = [sum(Float64(sol(t; idxs=_var(sys, sp.name))) for sp in mech.species) for t in ts]
axX = Axis(fig[2, 1:2]; xlabel="time (ms)", ylabel="mole fraction", title="key species")
for name in ("CH4", "O2", "CO2", "OH")
    vals = [Float64(sol(t; idxs=_var(sys, name))) for t in ts] ./ totals
    lines!(axX, ts .* 1e3, vals; label=name, linewidth=1.5)
end
axislegend(axX; position=:rt)
save(PNG_OUT, fig)
println("Saved plot: $PNG_OUT")
