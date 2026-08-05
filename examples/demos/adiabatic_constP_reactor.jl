# Phase 4b example: adiabatic constant-pressure batch reactor (path A, pure ODE).
# Exothermic A → B with T-dependent Arrhenius; T rises, V grows at const P, H conserved.
using ChemMechSim
using OrdinaryDiffEq: Rodas5P, ReturnCode
using ModelingToolkit: unknowns, getname

"Look up a named unknown on a simplified system (mtkcompile reorders states)."
_var(sys, name) = unknowns(sys)[findfirst(s -> String(getname(s)) == name, unknowns(sys))]

const R = 8.314
a1 = 2.5; a6A = -a1 * 298.15; a6B = a6A - 10000.0 / R              # B is 10 kJ/mol lower (exothermic)
nA = NASA7((a1,0,0,0,0,a6A,0.0),(a1,0,0,0,0,a6A,0.0), 200.0, 1000.0, 3500.0)
nB = NASA7((a1,0,0,0,0,a6B,0.0),(a1,0,0,0,0,a6B,0.0), 200.0, 1000.0, 3500.0)
spA = SpeciesData(id=1, name="A", thermo=nA); spB = SpeciesData(id=2, name="B", thermo=nB)
rxn = ReactionData(reactants=Dict(1=>1.0), products=Dict(2=>1.0),
                   kinetics=ElementaryArrhenius(1.0e2, 0.0, R * 3000.0))   # θ = Ea/R = 3000 K
mech = Mechanism(species=[spA, spB], reactions=[rxn])

reactor = BatchReactor(mech; mode=:adiabatic_constP)
sol = simulate(reactor, (0.0, 5.0);
               u0=Dict("n_A"=>1.0, "n_B"=>0.0, "T"=>800.0),
               solver=Rodas5P(), reltol=1e-8, abstol=1e-10)
println("retcode = ", sol.retcode)
sys = extract_system(reactor)
println("T : ", round(Float64(sol(0.0; idxs=_var(sys,"T"))), digits=1), " K → ",
                   round(Float64(sol(5.0; idxs=_var(sys,"T"))), digits=1), " K")
println("n_A: ", round(Float64(sol(0.0; idxs=_var(sys,"n_A"))), digits=3), " → ",
                   round(Float64(sol(5.0; idxs=_var(sys,"n_A"))), digits=3))
println("n_B: ", round(Float64(sol(0.0; idxs=_var(sys,"n_B"))), digits=3), " → ",
                   round(Float64(sol(5.0; idxs=_var(sys,"n_B"))), digits=3))
