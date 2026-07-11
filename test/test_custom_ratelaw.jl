using Test, ModelingToolkit
using DynamicQuantities
using ChemMechSim
using ChemMechSim: afactor, ktemp, plain, paramspec, body, needs_T, rate_constant

# ---- User-defined custom rate law: scaled Arrhenius k(T) = f·A·T^b·exp(-Ea/RT) ----
# The user writes ONLY this — no edit to lowering/. The framework's generic symbolic_kf
# (Task 6) + generic rate_constant (Task 1) handle the rest.
struct MyArrhenius <: AbstractKinetics
    A::Float64; b::Float64; Ea::Float64; f::Float64   # f = dimensionless scale factor
end

# The formula body (generic; pure arithmetic; the SINGLE definition of the formula).
_my_arrhenius_body(A, b, θ, f, T) = f * A * T^b * exp(-θ / T)

# Declare how each field becomes a parameter (paramspec) + the body + T-dependence.
ChemMechSim.paramspec(kin::MyArrhenius) = (afactor(:A, "", kin.b), plain(:b), ktemp(:Ea, ""), plain(:f))
ChemMechSim.body(kin::MyArrhenius) = _my_arrhenius_body
ChemMechSim.needs_T(kin::MyArrhenius) = true

@testset "Plan A T7: custom rate law (MyArrhenius) end-to-end" begin
    # 1) numeric rate_constant matches the body directly (MTK-free path)
    kin = MyArrhenius(1e9, 0.5, 5000.0, 2.0)
    @test rate_constant(kin, 1000.0) ≈ 2.0 * 1e9 * 1000.0^0.5 * exp(-(5000.0 / 8.314) / 1000.0)

    # 2) build a 1-reaction mechanism A -> B using the custom law, lower it, check structure.
    #    Use the zero-point config (MechanismConfig() = pure-kinetics bare ODE): the custom
    #    species A/B have no NASA7 thermo, so :fixedT (which requires nasa7) would fail.
    #    T is still auto-created as a parameter because MyArrhenius is T-dependent
    #    (needs_T=true -> _needs_T(mech) true -> T param with default 300 K).
    spA = SpeciesData(id=1, name="A"); spB = SpeciesData(id=2, name="B")
    rxn = ReactionData(reactants=Dict(1 => 1.0), products=Dict(2 => 1.0),
                       kinetics=MyArrhenius(1.0, 0.5, 5000.0, 2.0), reverse_policy=Irreversible())
    mech = Mechanism(species=[spA, spB], reactions=[rxn])
    phase = ChemMechSim.ChemPhaseSystem(mech; config=MechanismConfig())
    sys = ChemMechSim.extract_system(phase)
    @test length(unknowns(sys)) == 2                      # A, B concentrations
    # T appears as a parameter (the law is T-dependent); A/Ea materialized as params
    pnames = sort([String(ModelingToolkit.getname(p)) for p in parameters(sys)])
    @test "k_1_A" in pnames                               # A-factor param named per convention
    @test "k_1_theta" in pnames                           # θ = Ea/R
    @test "T" in pnames                                   # T auto-created (needs_T=true)
    # 3) simulate and check A decays (consumed), B grows (produced)
    Tparam = parameters(sys)[findfirst(p -> String(ModelingToolkit.getname(p)) == "T", parameters(sys))]
    Avar = unknowns(sys)[findfirst(s -> String(ModelingToolkit.getname(s)) == "A", unknowns(sys))]
    Bvar = unknowns(sys)[findfirst(s -> String(ModelingToolkit.getname(s)) == "B", unknowns(sys))]
    sol = simulate(phase, (0.0, 1.0); u0=Dict("A" => 1.0, "B" => 0.0),
                   params=[Tparam => 1000.0], reltol=1e-9, abstol=1e-12)
    @test sol(1.0; idxs=Avar) < sol(0.0; idxs=Avar)
    @test sol(1.0; idxs=Bvar) > sol(0.0; idxs=Bvar)

    # 4) symbolic_rate (design §5 protocol): default = symbolic_kf × mass-action(reactants).
    #    Build a minimal RateCtx matching the lowering pipeline's construction.
    t = ModelingToolkit.t
    Avar_sym, Bvar_sym = ModelingToolkit.@variables A(t) B(t)
    cvar_map = Dict(1 => Avar_sym, 2 => Bvar_sym)
    Tp = ChemMechSim.rate_param(:T, 1000.0, u"K")
    tcx = ChemMechSim.make_thermo_ctx(Tp)
    ctx = ChemMechSim.RateCtx(mech, cvar_map, Tp, 1, sum(values(rxn.reactants)),
                              tcx.R, tcx.P_std, tcx.coeff_cache)
    kin2 = rxn.kinetics
    expected = ChemMechSim.symbolic_kf(kin2, ctx) * ChemMechSim._mass_action(rxn.reactants, ctx.cvar)
    @test isa(ChemMechSim.symbolic_rate(kin2, rxn, ctx), ModelingToolkit.Num)
    @test isequal(ChemMechSim.symbolic_rate(kin2, rxn, ctx), expected)
end

# ---- Custom law that overrides symbolic_rate (inhibition factor) — proves the override
# is reachable from _direct_rate dispatch, not silently ignored. Plan A review fix.
struct _InhibitedTest <: AbstractKinetics
    A::Float64; b::Float64; Ea::Float64; K_inh::Float64; inhibitor::Int
end

_inhibited_body(A, b, θ, T) = A * T^b * exp(-θ / T)

ChemMechSim.paramspec(kin::_InhibitedTest) = (afactor(:A, "", kin.b), plain(:b), ktemp(:Ea, ""))
ChemMechSim.body(kin::_InhibitedTest) = _inhibited_body
ChemMechSim.needs_T(kin::_InhibitedTest) = true

# Override symbolic_rate: default kf×mass-action multiplied by 1/(1 + K_inh·[I])
function ChemMechSim.symbolic_rate(kin::_InhibitedTest, rx::ReactionData, ctx::ChemMechSim.RateCtx)
    default = ChemMechSim.symbolic_kf(kin, ctx) * ChemMechSim._mass_action(rx.reactants, ctx.cvar)
    return default / (1 + kin.K_inh * ctx.cvar[kin.inhibitor])
end

@testset "Plan A T7 fix: symbolic_rate override is honored" begin
    # Build a 2-species mechanism: A -> B with species 2 (B) acting as the inhibitor.
    spA = SpeciesData(id=1, name="A"); spB = SpeciesData(id=2, name="B")
    kin = _InhibitedTest(1.0, 0.5, 5000.0, 3.0, 2)  # K_inh=3, inhibitor=species 2
    rxn = ReactionData(reactants=Dict(1 => 1.0), products=Dict(2 => 1.0),
                       kinetics=kin, reverse_policy=Irreversible())
    mech = Mechanism(species=[spA, spB], reactions=[rxn])

    # Build a minimal RateCtx matching the lowering pipeline's construction.
    t = ModelingToolkit.t
    Avar_sym, Bvar_sym = ModelingToolkit.@variables A(t) B(t)
    cvar_map = Dict(1 => Avar_sym, 2 => Bvar_sym)
    Tp = ChemMechSim.rate_param(:T, 1000.0, u"K")
    tcx = ChemMechSim.make_thermo_ctx(Tp)
    ctx = ChemMechSim.RateCtx(mech, cvar_map, Tp, 1, sum(values(rxn.reactants)),
                              tcx.R, tcx.P_std, tcx.coeff_cache)

    # 1) symbolic_rate override DIFFERS from the plain default (inhibition factor present).
    plain_default = ChemMechSim.symbolic_kf(kin, ctx) * ChemMechSim._mass_action(rxn.reactants, ctx.cvar)
    overridden = ChemMechSim.symbolic_rate(kin, rxn, ctx)
    @test !isequal(overridden, plain_default)

    # 2) The _direct_rate fallback now routes through symbolic_rate (not inline kf×ma).
    #    So the rate produced by the dispatch fallback equals the overridden symbolic_rate.
    fallback_rate = ChemMechSim._direct_rate(kin, rxn, mech, cvar_map, Tp, 1, ctx)
    @test isequal(fallback_rate, overridden)

    # 3) The overridden rate contains the inhibitor concentration symbol (B), proving the
    #    inhibition factor is actually present in the expression.
    overridden_str = string(overridden)
    @test occursin("B", overridden_str)
end
