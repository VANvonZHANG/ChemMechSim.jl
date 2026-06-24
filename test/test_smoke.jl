using Test
using ChemMechSim
using DynamicQuantities

@testset "smoke: data layer composes end-to-end" begin
    # Two species, one irreversible first-order reaction A -> B,
    # assembled entirely through exported names (no submodule qualification).
    a = SpeciesData(id=1, name="A", molecular_weight=0.030u"kg/mol")
    b = SpeciesData(id=2, name="B", molecular_weight=0.030u"kg/mol")
    @test a.molecular_weight ≈ 0.030           # canonicalized

    rxn = ReactionData(
        reactants = Dict(1 => 1.0),
        products  = Dict(2 => 1.0),
        kinetics  = ElementaryArrhenius(1.0, 0.0, 0.0),
        reverse_policy = Irreversible(),
    )

    mech = Mechanism(species=[a, b], reactions=[rxn], elements=["A", "B"])
    @test length(mech.species) == 2
    @test length(mech.reactions) == 1
    @test mech.reactions[1].kinetics isa AbstractKinetics

    cfg = MechanismConfig()
    @test cfg.reverse_rate === :irreversible

    # Lowering is a stub: it must error cleanly (framework scope).
    @test_throws ErrorException lower_to_mtk(mech; config=cfg)
    @test_throws ErrorException simulate(mech)
end
