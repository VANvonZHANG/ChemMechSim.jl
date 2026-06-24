using Test
using ChemMechSim

@testset "data/thermo" begin
    # ThermoModel is abstract (concrete NASA7/NASA9 deferred to a later phase)
    @test ThermoModel isa Type
    @test !isconcretetype(ThermoModel)

    # ThermoDatabase default-constructs empty, typed Dict{String,ThermoModel}
    db = ThermoDatabase()
    @test isempty(db.entries)
    @test db.entries isa Dict{String,ThermoModel}
end
