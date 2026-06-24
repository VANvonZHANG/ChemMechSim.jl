using Test
using ChemMechSim

@testset "data/mechanism" begin
    a = SpeciesData(id=1, name="A")
    b = SpeciesData(id=2, name="B")
    rxn = ReactionData(
        reactants = Dict(1 => 1.0),
        products  = Dict(2 => 1.0),
        kinetics  = ElementaryArrhenius(1.0, 0.0, 0.0),
    )

    mech = Mechanism(species=[a, b], reactions=[rxn])
    @test length(mech.species) == 2
    @test length(mech.reactions) == 1
    @test isempty(mech.thermo.entries)
    @test isempty(mech.elements)

    mech2 = Mechanism(species=[a, b], reactions=[rxn], elements=["A", "B"])
    @test mech2.elements == ["A", "B"]
end
