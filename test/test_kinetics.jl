using Test
using ChemMechSim

@testset "data/kinetics" begin
    arr = ElementaryArrhenius(1.0e9, 0.0, 0.0)
    @test arr isa AbstractKinetics
    @test arr.A == 1.0e9

    tb = ThirdBodyArrhenius(arr, Dict(1 => 1.0, 2 => 3.0))
    @test tb isa AbstractKinetics
    @test tb.efficiencies[2] == 3.0

    troe = TroeFalloff(arr, arr, Dict(1 => 1.0), TroeParams(0.5, 1.0, 1.0e6, 1.0e3))
    @test troe isa AbstractFalloff
    @test troe isa AbstractKinetics
    @test troe.troe.α == 0.5

    sri = SRIFalloff(arr, arr, Dict{SpeciesID,Float64}(), SRIParams(1.0, 100.0, 1000.0))
    @test sri isa AbstractFalloff

    lin = LindemannFalloff(arr, arr, Dict{SpeciesID,Float64}())
    @test lin isa AbstractFalloff
    # Lindemann has no extra center-broadening params
    @test propertynames(lin) == (:low_rate, :high_rate, :efficiencies)

    # PLOG / Chebyshev are minimal concrete subtypes (full fields deferred)
    @test PlogRate() isa AbstractKinetics
    @test ChebyshevRate() isa AbstractKinetics
end
