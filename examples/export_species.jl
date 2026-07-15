# Export ChemMechSim T + species mole fractions to CSV for validation figures.
# Run: julia --project=. examples/export_species.jl
# Outputs: examples/cantera_ref/gri30_cms_species.csv + ffcm2_cms_species.csv
using ChemMechSim
using OrdinaryDiffEq: FBDF
using ModelingToolkit: unknowns, getname

const SP = ["CH4", "O2", "CO2", "OH", "H2O"]
const R, T0, P0 = 8.314, 1500.0, 101325.0
const c_tot = P0 / (R * T0)
const X0 = Dict("CH4" => 1.0/10.52, "O2" => 2.0/10.52, "N2" => 7.52/10.52)

_var(sys, name) = unknowns(sys)[findfirst(s -> String(getname(s)) == name, unknowns(sys))]

function export_mech(yaml, out_csv, label)
    println("Loading $label from $yaml ...")
    mech = load_mechanism(yaml)
    reactor = BatchReactor(mech; mode=:adiabatic_constV)
    sys = extract_system(reactor)
    u0 = Dict(sp.name => get(X0, sp.name, 0.0) * c_tot for sp in mech.species)
    u0["T"] = T0
    println("  solving $(length(mech.species)) sp / $(length(mech.reactions)) rxn ...")
    sol = simulate(reactor, (0.0, 5.0e-3); u0=u0, solver=FBDF(), reltol=1e-8, abstol=1e-12)
    println("  retcode=$(sol.retcode), exporting ...")
    ts = range(0.0, 5.0e-3; length=2000)
    open(out_csv, "w") do f
        println(f, "time_s,T_K,X_CH4,X_O2,X_CO2,X_OH,X_H2O")
        for t in ts
            Tv = Float64(sol(t; idxs=_var(sys, "T")))
            tot = sum(Float64(sol(t; idxs=_var(sys, sp.name))) for sp in mech.species)
            xs = [begin
                idx = findfirst(s -> s.name == n, mech.species)
                idx === nothing ? 0.0 : Float64(sol(t; idxs=_var(sys, n))) / tot
            end for n in SP]
            println(f, join([t, Tv, xs...], ","))
        end
    end
    println("  wrote $out_csv")
end

export_mech("test/data/gri30.yaml", "examples/cantera_ref/gri30_cms_species.csv", "GRI30")
export_mech("examples/data/FFCM2_model.yaml", "examples/cantera_ref/ffcm2_cms_species.csv", "FFCM2")
println("Done.")
