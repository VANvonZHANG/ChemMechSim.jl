# BatchReactor Layer-1 demo for ChemMechSim (Phase 2).
# Run:  julia --project=examples examples/demos/batch_reactor.jl
#
# Shows the script API: build a reactor from a Mechanism, choose a convenience
# mode, and simulate. This demo uses :kinetic (the MechanismConfig zero point);
# the other three modes are demonstrated by fixedT_reactor.jl,
# adiabatic_reactor.jl and adiabatic_constP_reactor.jl — all four are implemented.
using ChemMechSim
using ModelingToolkit: equations, unknowns, getname

mech = Mechanism(
    species=[SpeciesData(id=1, name="A"), SpeciesData(id=2, name="B")],
    reactions=[ReactionData(reactants=Dict(1 => 1.0), products=Dict(2 => 1.0),
                            kinetics=ElementaryArrhenius(2.0, 0.0, 0.0))])

reactor = BatchReactor(mech; mode=:kinetic, name=:decay)
println(reactor)
println("equations:")
for eq in equations(extract_system(reactor)); println("  ", eq); end

sol = simulate(reactor, (0.0, 2.0); u0=Dict("A" => 1.0, "B" => 0.0))
a_idx = findfirst(s -> String(getname(s)) == "A", unknowns(extract_system(reactor)))
println("\nA(2.0) = ", round(sol.u[end][a_idx], digits=5),
        "   (analytic exp(-4) = ", round(exp(-4.0), digits=5), ")")

# The other convenience modes (:fixedT, :adiabatic_constV, :adiabatic_constP) are
# shown in the sibling demos — see examples/demos/ and the "Reactor modes" docs page.
