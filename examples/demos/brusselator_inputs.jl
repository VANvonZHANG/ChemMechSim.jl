# One system, three input routes -> one Mechanism. The Brusselator oscillator
# (2 species, 4 constant-rate reactions, A=1/B=3 limit cycle) is entered (1)
# programmatically as plain Julia data types, (2) imported from a Catalyst
# ReactionSystem, (3) loaded from a minimal Cantera-YAML file. All three routes
# produce the SAME data-layer Mechanism -- the intermediate representation the
# rest of the framework (reactor, lowering, Jacobian, simulate) works from --
# and therefore the same solution.
# Run:  julia --project=. examples/demos/brusselator_inputs.jl
using ChemMechSim
using Catalyst: @reaction_network
using ModelingToolkit: equations, unknowns, getname

# -- Route 1: programmatic construction (plain Julia types, no DSL / no file) --
mech_programmatic = Mechanism(
    species=[SpeciesData(id=1, name="X"), SpeciesData(id=2, name="Y")],
    reactions=[
        ReactionData(reactants=Dict{Int,Float64}(),  products=Dict(1=>1.0),        kinetics=ElementaryArrhenius(1.0,0.0,0.0)),  # none -> X
        ReactionData(reactants=Dict(1=>2.0, 2=>1.0), products=Dict(1=>3.0),        kinetics=ElementaryArrhenius(1.0,0.0,0.0)),  # 2X + Y -> 3X
        ReactionData(reactants=Dict(1=>1.0),         products=Dict(2=>1.0),        kinetics=ElementaryArrhenius(3.0,0.0,0.0)),  # X -> Y
        ReactionData(reactants=Dict(1=>1.0),         products=Dict{Int,Float64}(), kinetics=ElementaryArrhenius(1.0,0.0,0.0)),  # X -> none
    ])

# -- Route 2: Catalyst ReactionSystem import (mass-action numeric subset) --
rn = @reaction_network begin
    1.0, ∅ → X
    1.0, 2*X + Y → 3*X
    3.0, X → Y
    1.0, X → ∅
end
mech_catalyst = import_from_catalyst(rn)

# -- Route 3: Cantera-YAML mechanism file (subset notes in the fixture header) --
mech_yaml = load_mechanism(joinpath(@__DIR__, "..", "mechanism", "brusselator.yaml"))

# -- All routes converge on the same intermediate representation --
routes = [("programmatic", mech_programmatic), ("catalyst import", mech_catalyst), ("yaml file", mech_yaml)]
println("Route              -> Mechanism (intermediate representation)")
for (tag, m) in routes
    println(rpad(tag, 18), "-> ", length(m.species), " species, ", length(m.reactions), " reactions")
end

println("\nSpecies balance from route 3 (YAML):")
for eq in equations(extract_system(ChemPhaseSystem(mech_yaml))); println("  ", eq); end

# -- Same representation => same physics: identical limit cycle from all routes --
println("\nSame IR, same solve -- X(t) at t = 40 from each route:")
X40 = Float64[]
for (tag, m) in routes
    phase = ChemPhaseSystem(m)
    sys   = extract_system(phase)
    xv    = unknowns(sys)[findfirst(s -> String(getname(s)) == "X", unknowns(sys))]
    sol   = simulate(phase, (0.0, 40.0); u0=Dict("X"=>1.0, "Y"=>0.5), reltol=1e-9, abstol=1e-9)
    x40   = Float64(sol(40.0; idxs=xv))
    push!(X40, x40)
    println("  ", rpad(tag, 18), "X(40) = ", round(x40, digits=6))
end
spread = maximum(X40) - minimum(X40)
println(spread < 1e-6 ? "PASS: three routes agree (same Mechanism => same limit cycle)" :
                        "FAIL: routes diverge -- X(40) spread = $spread")
