using Test
using ChemMechSim
using DynamicQuantities
const CU = ChemMechSim.ChemUnits

@testset "ChemUnits" begin
    @test CU.conc == u"mol/m^3"
    @test CU.molmass == u"kg/mol"

    # canonical: Real passes through unchanged
    @test CU.canonical(0.018015, CU.molmass) == 0.018015

    # canonical: Quantity converted to reference unit, stripped to number
    @test CU.canonical(18.015u"g/mol", CU.molmass) ≈ 0.018015 atol = 1e-9

    # NaN Real passes through (unspecified value)
    @test isnan(CU.canonical(NaN, CU.molmass))

    # Wrong-dimension quantity is rejected
    @test_throws ErrorException CU.canonical(1.0u"m", CU.molmass)
end
