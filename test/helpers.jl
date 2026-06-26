# Shared test fixtures for the ChemMechSim test suite. Included once by runtests.jl
# BEFORE any test file, so fixtures like _brusselator_mech / _var live in one place
# and do not trigger Julia method-redefinition warnings from duplicate per-file
# definitions.
using ChemMechSim
using ModelingToolkit: unknowns, getname

"Brusselator mechanism (programmatic path): ∅→X, 2X+Y→3X, X→Y, X→∅."
function _brusselator_mech()
    X = SpeciesData(id=1, name="X"); Y = SpeciesData(id=2, name="Y")
    rxns = [
        ReactionData(reactants=Dict{Int,Float64}(),  products=Dict(1=>1.0), kinetics=ElementaryArrhenius(1.0,0,0)),
        ReactionData(reactants=Dict(1=>2.0, 2=>1.0), products=Dict(1=>3.0), kinetics=ElementaryArrhenius(1.0,0,0)),
        ReactionData(reactants=Dict(1=>1.0),         products=Dict(2=>1.0), kinetics=ElementaryArrhenius(3.0,0,0)),
        ReactionData(reactants=Dict(1=>1.0),         products=Dict{Int,Float64}(), kinetics=ElementaryArrhenius(1.0,0,0)),
    ]
    Mechanism(species=[X, Y], reactions=rxns)
end

"Look up a named unknown on a simplified system (order may be reordered)."
_var(sys, name) = unknowns(sys)[findfirst(s -> String(getname(s)) == name, unknowns(sys))]
