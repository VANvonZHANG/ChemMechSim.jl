using Test
using ChemMechSim

@testset "data/reaction" begin
    # Reverse-rate policies
    @test Irreversible() isa ReverseRatePolicy
    er = ExplicitReverse(ElementaryArrhenius(1.0, 0.0, 0.0))
    @test er.rate isa ElementaryArrhenius
    @test ThermoReverse() isa ReverseRatePolicy

    # ReactionMeta defaults
    @test ReactionMeta().duplicate == false
    @test isempty(ReactionMeta().orders)

    # ReactionData keyword constructor + defaults
    rxn = ReactionData(
        reactants = Dict(1 => 2.0),
        products  = Dict(2 => 1.0),
        kinetics  = ElementaryArrhenius(1.0e10, 0.0, 0.0),
    )
    @test rxn.reactants[1] == 2.0
    @test rxn.products[2] == 1.0
    @test rxn.kinetics isa AbstractKinetics
    @test rxn.reverse_policy isa Irreversible   # default
    @test rxn.meta.duplicate == false            # default

    # Explicit reverse policy overrides default
    rxn2 = ReactionData(
        reactants = Dict(1 => 1.0),
        products  = Dict(2 => 1.0),
        kinetics  = ElementaryArrhenius(1.0, 0.0, 0.0),
        reverse_policy = ThermoReverse(),
    )
    @test rxn2.reverse_policy isa ThermoReverse
end
