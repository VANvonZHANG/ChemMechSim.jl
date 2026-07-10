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

@testset "Plan A T1: generic bodies + numeric rate_constant (MTK-free)" begin
    using ChemMechSim: _arrhenius_body, _troe_F_body, rate_constant, paramspec, body, needs_T
    using ChemMechSim: AFactor, KTemp, Plain, afactor, ktemp, plain, numeric_value
    # Arrhenius body works for plain Real
    @test _arrhenius_body(1e9, 0.0, 3000.0, 1000.0) ≈ 1e9 * exp(-3000.0 / 1000.0)
    # a custom law defined ONLY in the test, via paramspec/body (no lowering code):
    struct _T1Custom <: ChemMechSim.AbstractKinetics
        A::Float64; b::Float64; Ea::Float64
    end
    ChemMechSim.paramspec(kin::_T1Custom) = (afactor(:A,"",kin.b), plain(:b), ktemp(:Ea,""))
    ChemMechSim.body(kin::_T1Custom) = (A, b, θ, T) -> A * T^b * exp(-θ / T)
    ChemMechSim.needs_T(kin::_T1Custom) = !(iszero(kin.b) && iszero(kin.Ea))
    kin = _T1Custom(1e9, 0.5, 5000.0)
    @test rate_constant(kin, 1000.0) ≈ 1e9 * 1000.0^0.5 * exp(-(5000.0 / 8.314) / 1000.0)
    @test needs_T(kin) == true
    @test needs_T(_T1Custom(1.0, 0, 0)) == false
end
