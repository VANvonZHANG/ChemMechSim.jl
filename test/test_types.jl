using Test
using ChemMechSim

@testset "data/types" begin
    # SpeciesID is an integer alias usable as a Dict key
    @test SpeciesID === Int
    d = Dict{SpeciesID,Float64}(1 => 2.0, 3 => 1.0)
    @test d[1] == 2.0

    # SpeciesRole is a Symbol with documented roles
    @test SpeciesRole === Symbol
    for r in (:dynamic, :algebraic_qssa, :constant_pool, :bath_gas)
        @test isa(r, SpeciesRole)
    end
end
