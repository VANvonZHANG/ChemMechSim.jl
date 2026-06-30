# Phase 3 demo: :fixedT isothermal reactor with thermodynamic reverse rate + EOS pressure output.
# Run:  julia --project=. examples/fixedT_reactor.jl
using ChemMechSim
using ModelingToolkit: equations, unknowns, parameters, getname, observed
using OrdinaryDiffEq: Rodas5P

# H2 + OH <-> H + H2O  (Δν=0 isomerization-like, ExplicitReverse; T-dependent forward)
H2  = SpeciesData(id=1, name="H2")
OH  = SpeciesData(id=2, name="OH")
H   = SpeciesData(id=3, name="H")
H2O = SpeciesData(id=4, name="H2O")
rxn = ReactionData(reactants=Dict(1=>1.0, 2=>1.0), products=Dict(3=>1.0, 4=>1.0),
                   kinetics=ElementaryArrhenius(1.0e6, 1.5, 2.0e4),     # T-dependent forward
                   reverse_policy=ExplicitReverse(ElementaryArrhenius(2.0e5, 1.0, 1.5e4)))
mech = Mechanism(species=[H2,OH,H,H2O], reactions=[rxn])
r = BatchReactor(mech; mode=:fixedT, name=:fixedT_demo)
println(r)
sys = extract_system(r)
println("observed: ", [getname(o.lhs) for o in observed(sys)])
Tp = parameters(sys)[findfirst(p -> String(getname(p))=="T", parameters(sys))]
sol = simulate(r, (0.0, 1e-3); u0=Dict("H2"=>2.0,"OH"=>1.0,"H"=>0.0,"H2O"=>0.0),
               params=[Tp => 1200.0], solver=Rodas5P(), reltol=1e-9, abstol=1e-12)
Pvar = [o.lhs for o in observed(sys) if getname(o.lhs)==:P][1]
for n in ["H2","OH","H","H2O"]
    s = unknowns(sys)[findfirst(x -> String(getname(x))==n, unknowns(sys))]
    println("  ", rpad(n,4), "(1e-3) = ", round(sol(1e-3; idxs=s), digits=5))
end
println("  P(1e-3) = ", round(sol(1e-3; idxs=Pvar), digits=1), " Pa")
