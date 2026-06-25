using Test
using ChemMechSim
using Catalyst

@testset "import_from_catalyst: Brusselator" begin
    rn = @reaction_network begin
        1.0, ∅ → X
        1.0, 2*X + Y → 3*X
        3.0, X → Y
        1.0, X → ∅
    end
    mech = import_from_catalyst(rn)

    @test length(mech.species) == 2
    @test sort([s.name for s in mech.species]) == ["X", "Y"]
    @test length(mech.reactions) == 4

    name2id = Dict(s.name => s.id for s in mech.species)
    @test name2id["X"] == 1 && name2id["Y"] == 2

    # reaction 3 (X -> Y) carries rate B = 3.0, first-order in X
    r3 = mech.reactions[3]
    @test collect(keys(r3.reactants)) == [name2id["X"]]
    @test r3.reactants[name2id["X"]] == 1.0
    @test r3.products == Dict(name2id["Y"] => 1.0)
    @test r3.kinetics.A == 3.0
    @test r3.kinetics.b == 0.0 && r3.kinetics.Ea == 0.0
end

@testset "import_from_catalyst: rejects non-numeric (parameter) rates" begin
    # k is not declared as a species → Catalyst makes it a parameter → symbolic rate.
    rn = @reaction_network begin
        k, ∅ → X
    end
    @test_throws ErrorException import_from_catalyst(rn)
end
