# Phase 4a demo: adiabatic constant-volume batch reactor.
# Exothermic A -> B releases heat; with no heat loss (adiabatic) and fixed V, temperature rises
# and internal energy U = V·Σ cᵢ·ūᵢ(T) is conserved. Pressure (observed via ideal-gas EOS) rises with T.
using ChemMechSim
using OrdinaryDiffEq: Rodas5P
using ModelingToolkit: getname, unknowns

const R = 8.314
a1 = 2.5; a6A = -a1 * 298.15; HEAT = 30_000.0; a6B = a6A - HEAT / R   # B is 30 kJ/mol lower in h̄
nA = NASA7((a1,0,0,0,0,a6A,0.0),(a1,0,0,0,0,a6A,0.0), 200.0, 1000.0, 3500.0)
nB = NASA7((a1,0,0,0,0,a6B,0.0),(a1,0,0,0,0,a6B,0.0), 200.0, 1000.0, 3500.0)
spA = SpeciesData(id=1, name="A", thermo=nA); spB = SpeciesData(id=2, name="B", thermo=nB)
rxn = ReactionData(reactants=Dict(1=>1.0), products=Dict(2=>1.0),
                   kinetics=ElementaryArrhenius(0.5, 0.0, 0.0))         # k = 0.5 s⁻¹
mech = Mechanism(species=[spA, spB], reactions=[rxn])

reactor = BatchReactor(mech; mode=:adiabatic_constV)
sol = simulate(reactor, (0.0, 10.0);
               u0=Dict("A"=>1.0, "B"=>0.0, "T"=>300.0),
               solver=Rodas5P(), reltol=1e-8, abstol=1e-10)

sys = extract_system(reactor)
Tvar = unknowns(sys)[findfirst(s -> String(getname(s)) == "T", unknowns(sys))]
Avar = unknowns(sys)[findfirst(s -> String(getname(s)) == "A", unknowns(sys))]
Bvar = unknowns(sys)[findfirst(s -> String(getname(s)) == "B", unknowns(sys))]
println("retcode          = ", sol.retcode)
println("T  300 K  →  ", round(Float64(sol(10.0; idxs=Tvar)), digits=1), " K  (exothermic rise)")
println("A  1.0    →  ", round(Float64(sol(10.0; idxs=Avar)), digits=4), " mol/m³")
# Total internal energy U/V = Σᵢ cᵢ·ūᵢ(T) is the adiabatic-const-V invariant (ūᵢ = h̄ᵢ − RT).
Tf = Float64(sol(10.0; idxs=Tvar)); Af = Float64(sol(10.0; idxs=Avar)); Bf = Float64(sol(10.0; idxs=Bvar))
Uinit = 1.0 * u_molar(nA, 300.0)                       # only A present initially
Ufin  = Af * u_molar(nA, Tf) + Bf * u_molar(nB, Tf)
println("ΔU/V (energy conservation, should be ≈ 0): ", round(Ufin - Uinit, digits=3), " J/m³")
