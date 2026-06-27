# Phase 2.5b demo: H2-O2 subset lowered through MIXED paths (Catalyst + direct MTK), unit-aware.
# Run:  julia --project=. examples/h2o2_subset.jl
using ChemMechSim
using ModelingToolkit: equations, unknowns, getname, parameters
using OrdinaryDiffEq
H2=SpeciesData(id=1,name="H2"); O2=SpeciesData(id=2,name="O2"); H2O=SpeciesData(id=3,name="H2O")
H=SpeciesData(id=4,name="H"); O=SpeciesData(id=5,name="O"); OH=SpeciesData(id=6,name="OH"); HO2=SpeciesData(id=7,name="HO2")
allM = Dict(sid => 1.0 for sid in 1:7)
mech = Mechanism(species=[H2,O2,H2O,H,O,OH,HO2], reactions=[
    ReactionData(reactants=Dict(1=>1.0,2=>1.0), products=Dict(6=>2.0), kinetics=ElementaryArrhenius(1.0e6,0.0,0.0)),            # catalyst path
    ReactionData(reactants=Dict(4=>1.0,2=>1.0), products=Dict(7=>1.0),
                 kinetics=TroeFalloff(ElementaryArrhenius(1.0e9,0.0,0.0), ElementaryArrhenius(1.0e6,0.0,0.0),                   # low, high
                                      allM, TroeParams(0.5,1.0e-30,1.0e30,1.0e30))),                                            # direct path
])
r = BatchReactor(mech; name=:h2o2)
println(r); println("equations:"); for eq in equations(extract_system(r)); println("  ", eq); end
sys = extract_system(r)
T = parameters(sys)[findfirst(p -> String(getname(p)) == "T", parameters(sys))]
sol = simulate(r, (0.0, 1.0e-5); u0=Dict("H2"=>2.0,"O2"=>1.5,"H2O"=>0.0,"H"=>1e-3,"O"=>0.0,"OH"=>0.0,"HO2"=>0.0),
               params=[T => 1000.0], reltol=1e-9, abstol=1e-12)
for n in ["H2","O2","HO2","OH"]
    s = unknowns(sys)[findfirst(x -> String(getname(x)) == n, unknowns(sys))]
    println("  ", rpad(n,4), "(1e-5) = ", round(sol(1e-5; idxs=s), digits=6))
end
