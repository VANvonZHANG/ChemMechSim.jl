using Test
using ChemMechSim
using Catalyst
using ModelingToolkit: unknowns

_is_default(c) = c.energy === :isothermal && c.constraint === :none && c.eos === :off &&
                 c.thermo_data === :none && c.reverse_rate === :irreversible &&
                 c.state_basis === :concentration

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
