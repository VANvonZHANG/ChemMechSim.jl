using Test
using ChemMechSim
using ModelingToolkit: getname, get_variables, getdefault, value, @parameters, parameters, unknowns, observed
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

    # PLOG / Chebyshev are concrete subtypes
    @test PlogRate(ChemMechSim.PlogPoint[]) isa AbstractKinetics
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
                  Dict{Int,Any}(), T_test, 1, 1.0, nothing, nothing, Dict{Int,Any}(), nothing, Any[])
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

@testset "Phase 6 T1: PLOG numeric interpolation (MTK-free)" begin
    using ChemMechSim: PlogPoint, PlogRate, plog_rate, _plog_interp_segment
    # segment: multiplicative form, dimensionless ratio
    @test _plog_interp_segment(2.0, 8.0, 0.5) ≈ 2.0 * (8.0 / 2.0)^0.5   # = 4.0
    # 2-point PLOG, Ea=0 → k_i(T) = A_i (constant in T)
    kin = PlogRate([PlogPoint(1e4, 1e9, 0.0, 0.0), PlogPoint(1e6, 1e7, 0.0, 0.0)])
    @test plog_rate(kin, 1000.0, 1e4) ≈ 1e9            # at P_lo → k_lo
    @test plog_rate(kin, 1000.0, 1e6) ≈ 1e7            # at P_hi → k_hi
    # geometric mid pressure (log-mid): f=0.5 → sqrt(k_lo·k_hi) = 1e8
    @test plog_rate(kin, 1000.0, 1e5) ≈ 1e8
    # clamping
    @test plog_rate(kin, 1000.0, 1e3) ≈ 1e9            # below P_lo → k_lo
    @test plog_rate(kin, 1000.0, 1e7) ≈ 1e7            # above P_hi → k_hi
    # 3-point: mid segment uses points 2,3
    kin3 = PlogRate([PlogPoint(1e4, 1e9, 0.0, 0.0), PlogPoint(1e5, 1e8, 0.0, 0.0), PlogPoint(1e6, 1e7, 0.0, 0.0)])
    @test plog_rate(kin3, 1000.0, 1e5) ≈ 1e8           # exactly at mid point → k_mid
    @test plog_rate(kin3, 1000.0, 3e5) ≈ 1e8 * (1e7 / 1e8)^((log(3e5/1e5)) / log(1e6/1e5))  # bracket [2,3]
end

@testset "Large-mech A: _troe_fcent_plan degenerate detection" begin
    using ChemMechSim: _troe_fcent_plan
    # normal params → all active
    @test _troe_fcent_plan(0.43, 2941.0, 6964.0, 74.0) == (:active, :active, :active)
    # Aramco #1 H2O2: T3=1e-30→:zero, T1=1e30→:one, T2=1e30→:zero
    @test _troe_fcent_plan(0.43, 1e30, 1e30, 1e-30) == (:zero, :one, :zero)
    # Aramco #18: T1=1e-30→:zero(exp(-T/T1)→0), T2=1e30→:zero, T3=-10200→:active(negative, build normally)
    # Return tuple is (term1=T3, term2=T1, term3=T2) → (:active, :zero, :zero)
    @test _troe_fcent_plan(0.5, 1e-30, 1e30, -10200.0) == (:active, :zero, :zero)
    # Aramco #38: T2=1e20→:zero, others normal → (term1=T3:active, term2=T1:active, term3=T2:zero)
    @test _troe_fcent_plan(0.5, 60.79, 1e20, 815.3) == (:active, :active, :zero)
end

@testset "Large-mech T2: M_eff as algebraic variable (state+tearing → observed)" begin
    using OrdinaryDiffEq: ODEFunction
    # H + O2 + M -> HO2 + M ; [M]_eff = sum of all species (alpha=1 default)
    H = SpeciesData(id=1, name="H");  O2  = SpeciesData(id=2, name="O2")
    HO2 = SpeciesData(id=3, name="HO2"); M = SpeciesData(id=4, name="M")
    rxn = ReactionData(reactants=Dict(1 => 1.0, 2 => 1.0), products=Dict(3 => 1.0),
                       kinetics=ThirdBodyArrhenius(ElementaryArrhenius(2.0, 0.0, 0.0),
                                                   Dict(4 => 1.0)))
    mech = Mechanism(species=[H, O2, HO2, M], reactions=[rxn])
    sys = lower_to_mtk(mech)
    # M_eff_1 should be in observed, NOT in unknowns
    obs_names = Set(String(getname(o.lhs)) for o in observed(sys))
    unk_names = Set(String(getname(u)) for u in unknowns(sys))
    @test "M_eff_1" in obs_names
    @test !("M_eff_1" in unk_names)
    # Numeric: at H=1,O2=2,M=3,HO2=0: [M]_eff = 1+2+0+3 = 6 -> rate = 2*1*2*6 = 24
    idx = _state_index(sys); u = zeros(4)
    u[idx["H"]] = 1.0; u[idx["O2"]] = 2.0; u[idx["M"]] = 3.0
    du = zeros(4)
    pvals = [getdefault(p) for p in parameters(sys)]
    ODEFunction(sys)(du, u, pvals, 0.0)
    @test du[idx["H"]]   ≈ -24.0
    @test du[idx["HO2"]] ≈  24.0
    @test du[idx["M"]]   ≈ 0.0
    # Troe falloff: M_eff should also be observed
    troe_rxn = ReactionData(reactants=Dict(1 => 1.0, 2 => 1.0), products=Dict(3 => 1.0),
        kinetics=TroeFalloff(ElementaryArrhenius(1.0e20, -1.4, 0.0),
                             ElementaryArrhenius(1.0e15, -1.0, 0.0),
                             Dict(4 => 1.0), TroeParams(0.5, 1.0e-30, 1.0e30, 1.0e30)))
    troe_mech = Mechanism(species=[H, O2, HO2, M], reactions=[troe_rxn])
    troe_sys = lower_to_mtk(troe_mech)
    troe_obs = Set(String(getname(o.lhs)) for o in observed(troe_sys))
    @test "M_eff_1" in troe_obs
end
