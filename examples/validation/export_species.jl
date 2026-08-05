# Export ChemMechSim T + species mole fractions to CSV for validation figures.
# Run: julia --project=. examples/validation/export_species.jl
# Outputs: examples/validation/{gri30,ffcm2,aramco}_cms_species.csv
using ChemMechSim
using OrdinaryDiffEq: FBDF
using LinearSolve: UMFPACKFactorization
using ModelingToolkit: unknowns, getname

const SP = ["CH4", "O2", "CO2", "OH", "H2O"]
const R, T0, P0 = 8.314, 1500.0, 101325.0
const c_tot = P0 / (R * T0)
const X0 = Dict("CH4" => 1.0/10.52, "O2" => 2.0/10.52, "N2" => 7.52/10.52)

_var(sys, name) = unknowns(sys)[findfirst(s -> String(getname(s)) == name, unknowns(sys))]

function export_mech(yaml, out_csv, label)
    println("Loading $label from $yaml ...")
    mech = load_mechanism(yaml)
    reactor = BatchReactor(mech; mode=:adiabatic_constV, checks=false)  # checks=false: Aramco/FFCM2 K_c unit-fold; safe for GRI30
    sys = extract_system(reactor)
    u0 = Dict(sp.name => get(X0, sp.name, 0.0) * c_tot for sp in mech.species)
    u0["T"] = T0
    println("  solving $(length(mech.species)) sp / $(length(mech.reactions)) rxn ...")
    # jac=true + UMFPACK sparse linsolve, uniform across all mechs. Required for Aramco
    # (581 sp) where the default dense ForwardDiff Jacobian OOMs; see
    # examples/aramco_linsolve_probe.jl (UMFPACK 35x over KLU; ~45s for Aramco).
    sol = simulate(reactor, (0.0, 5.0e-3); u0=u0, jac=true,
                    solver=FBDF(linsolve=UMFPACKFactorization()), reltol=1e-8, abstol=1e-12)
    println("  retcode=$(sol.retcode), exporting ...")
    # jac=true builds the ODEFunction via the sharded path (ODEFunction{true,NoSpecialize}),
    # which drops MTK symbolic solution indexing — so read by NUMERIC state index, not by
    # `sol(t; idxs=<symbol>)` (that throws "Invalid symbol T(t) for getsym"). `sol(t)`
    # returns the full interpolated state vector (one interp per t — fast even for Aramco's
    # 581 states; the old symbolic form did ~Nspecies sol() calls per t); index into it.
    unks = unknowns(sys)
    idxof(name) = findfirst(s -> String(getname(s)) == name, unks)
    T_i = idxof("T")
    sp_i = Dict(sp.name => idxof(sp.name) for sp in mech.species if idxof(sp.name) !== nothing)
    ts = range(0.0, 5.0e-3; length=2000)
    open(out_csv, "w") do f
        println(f, "time_s,T_K,X_CH4,X_O2,X_CO2,X_OH,X_H2O")
        for t in ts
            u = sol(t)
            Tv = Float64(u[T_i])
            tot = sum(Float64(u[sp_i[sp.name]]) for sp in mech.species if haskey(sp_i, sp.name))
            xs = [haskey(sp_i, n) ? Float64(u[sp_i[n]]) / tot : 0.0 for n in SP]
            println(f, join([t, Tv, xs...], ","))
        end
    end
    println("  wrote $out_csv")
end

# @__DIR__-relative paths so the script runs from any CWD (repo root, examples/, …).
const HERE = @__DIR__
const REFS = joinpath(HERE, "output")
mkpath(REFS)
export_mech(joinpath(HERE, "..", "..", "test", "data", "gri30.yaml"), joinpath(REFS, "gri30_cms_species.csv"), "GRI30")
export_mech(joinpath(HERE, "..", "mechanism", "FFCM2.yaml"),         joinpath(REFS, "ffcm2_cms_species.csv"), "FFCM2")
export_mech(joinpath(HERE, "..", "mechanism", "AramcoMech3.0.yaml"), joinpath(REFS, "aramco_cms_species.csv"), "Aramco")
println("Done.")
