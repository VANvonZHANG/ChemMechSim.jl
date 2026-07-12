using Test
using ChemMechSim
using ModelingToolkit: getname, get_variables, getdefault, value, @parameters
import ModelingToolkit: substitute, Num

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

@testset "Plan A T6: generic paramspec-driven symbolic_kf" begin
    import ChemMechSim: symbolic_kf, RateCtx, rate_param, paramspec, body, afactor, ktemp, plain
    import ChemMechSim: rate_constant
    @parameters T_test = 1000.0
    # a law with ONLY paramspec+body (no explicit symbolic_kf)
    struct _T6Custom <: ChemMechSim.AbstractKinetics
        A::Float64; b::Float64; Ea::Float64
    end
    ChemMechSim.paramspec(kin::_T6Custom) = (afactor(:A,"",kin.b), plain(:b), ktemp(:Ea,""))
    ChemMechSim.body(kin::_T6Custom) = (A, b, θ, T) -> A * T^b * exp(-θ / T)
    kin = _T6Custom(1e9, 0.5, 5000.0)
    ctx = RateCtx(ChemMechSim.Mechanism(ChemMechSim.SpeciesData[], ChemMechSim.ReactionData[],
                                        ChemMechSim.ThermoDatabase(), String[]),
                  Dict{Int,Any}(), T_test, 1, 1.0, nothing, nothing, Dict{Int,Any}())
    kf_expr = symbolic_kf(kin, ctx)            # should dispatch to the generic materializer
    @test kf_expr isa Num
    # Structural check: the materialized params (k_1_A, k_1_theta) and T appear in the expression
    var_names = Set(getname.(get_variables(kf_expr)))
    @test :k_1_A in var_names                 # AFactor role → unit-bearing param
    @test :k_1_theta in var_names             # KTemp role → unit-bearing param
    @test :T_test in var_names                # temperature symbol
    # Numeric check: substitute all params+T to their defaults and fold to Float64.
    # (Brief's plain substitute(kf, Dict(T=>1000)) doesn't fold unit-bearing params in this MTK
    # version; fold=Val(true) forces full numeric evaluation.)
    sub = Dict{Any,Float64}()
    for v in get_variables(kf_expr)
        sub[v] = getname(v) === :T_test ? 1000.0 : getdefault(v)
    end
    numeric_kf = Float64(value(substitute(kf_expr, sub; fold=Val(true))))
    @test numeric_kf ≈ 1e9 * 1000.0^0.5 * exp(-(5000.0/8.314)/1000.0)  rtol=1e-9
    # Cross-check: the symbolic materializer agrees with the numeric rate_constant path
    @test rate_constant(kin, 1000.0) ≈ numeric_kf  rtol=1e-12
end
