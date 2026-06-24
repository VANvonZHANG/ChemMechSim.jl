using Test
using ChemMechSim
using DynamicQuantities

@testset "data/species" begin
    # Pure-kinetics species: MW/elements/thermo all optional
    s = SpeciesData(id=1, name="X")
    @test s.id == 1
    @test s.name == "X"
    @test isempty(s.elements)
    @test isnan(s.molecular_weight)   # unspecified
    @test s.thermo === nothing
    @test s.role === :dynamic

    # Bare number MW is taken as already-canonical (kg/mol)
    s2 = SpeciesData(id=2, name="H2O", molecular_weight=0.018015)
    @test s2.molecular_weight ≈ 0.018015

    # Quantity MW is converted + stripped
    s3 = SpeciesData(id=3, name="H2O", molecular_weight=18.015u"g/mol")
    @test s3.molecular_weight ≈ 0.018015 atol = 1e-9

    # Elements + role + thermo defaults
    s4 = SpeciesData(id=4, name="CO2",
                     elements=Dict("C"=>1,"O"=>2),
                     molecular_weight=0.04401,
                     role=:constant_pool)
    @test s4.elements["C"] == 1
    @test s4.role === :constant_pool
    @test s4.thermo === nothing
end
