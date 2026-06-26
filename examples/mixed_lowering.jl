# Phase 2.5a demo: mixed Catalyst/direct lowering + T-dependent Arrhenius.
# Run:  julia --project=. examples/mixed_lowering.jl
using ChemMechSim
using ModelingToolkit: equations, unknowns, getname, parameters

# A T-dependent elementary reaction (lowered via the Catalyst backend path).
a = SpeciesData(id=1, name="A"); b = SpeciesData(id=2, name="B")
mech = Mechanism(species=[a, b], reactions=[
    ReactionData(reactants=Dict(1 => 1.0), products=Dict(2 => 1.0),
                 kinetics=ElementaryArrhenius(2.0, 1.0, 8314.0)),   # A·T^b·exp(-Ea/RT)
])
r = BatchReactor(mech; name=:td)
println(r)
println("equations:")
for eq in equations(extract_system(r)); println("  ", eq); end

sys = extract_system(r)
T = parameters(sys)[findfirst(p -> String(getname(p)) == "T", parameters(sys))]
for Tval in (500.0, 1000.0, 1500.0)
    sol = simulate(r, (0.0, 0.01); u0=Dict("A" => 1.0, "B" => 0.0),
                   params=[T => Tval], reltol=1e-9, abstol=1e-9)
    A = unknowns(sys)[findfirst(s -> String(getname(s)) == "A", unknowns(sys))]
    println("T=", Tval, " K  ->  A(0.01)=", round(sol(0.01; idxs=A), digits=5),
            "  (hotter T -> faster decay)")
end
