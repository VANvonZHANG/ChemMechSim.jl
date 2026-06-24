using Test
using ChemMechSim
using Catalyst
using ModelingToolkit: unknowns

_is_default(c) = c.energy === :isothermal && c.constraint === :none && c.eos === :off &&
                 c.thermo_data === :none && c.reverse_rate === :irreversible &&
                 c.state_basis === :concentration

function _brusselator_mech()
    X = SpeciesData(id=1, name="X"); Y = SpeciesData(id=2, name="Y")
    rxns = [
        ReactionData(reactants=Dict{Int,Float64}(),   products=Dict(1=>1.0), kinetics=ElementaryArrhenius(1.0,0,0)),
        ReactionData(reactants=Dict(1=>2.0, 2=>1.0),  products=Dict(1=>3.0), kinetics=ElementaryArrhenius(1.0,0,0)),
        ReactionData(reactants=Dict(1=>1.0),          products=Dict(2=>1.0), kinetics=ElementaryArrhenius(3.0,0,0)),
        ReactionData(reactants=Dict(1=>1.0),          products=Dict{Int,Float64}(), kinetics=ElementaryArrhenius(1.0,0,0)),
    ]
    Mechanism(species=[X, Y], reactions=rxns)
end

@testset "ChemPhaseSystem: from Mechanism" begin
    mech = _brusselator_mech()
    phase = ChemPhaseSystem(mech)
    @test phase.mech === mech
    @test _is_default(phase.config)
    @test length(unknowns(phase.sys)) == 2
    @test extract_system(phase) === phase.sys
end

@testset "ChemPhaseSystem: from Catalyst ReactionSystem" begin
    rn = @reaction_network begin
        1.0, ∅ → X
        1.0, 2*X + Y → 3*X
        3.0, X → Y
        1.0, X → ∅
    end
    phase = ChemPhaseSystem(rn)
    @test length(unknowns(phase.sys)) == 2
    @test extract_system(phase) === phase.sys
end
