# Rate-law lowering (§5.2, §5.4, §3.3): turn an AbstractKinetics rate law into a symbolic
# rate expression. Two paths share the same unit-bearing k and the same @species:
#   - catalyst_lowering  : plain elementary Arrhenius via Catalyst.Reaction + oderatelaw
#                          (combinatoric_ratelaw=false), spec §3.3 layer 2;
#   - direct_mtk_lowering: everything else (third-body, Troe, Lindemann, ...), building the
#                          symbolic rate directly, spec §3.3 layer 3.
# lower_reaction dispatches between them. _direct_kf returns the effective forward rate
# constant EXCLUDING the mass-action term (so thermo.jl's ThermoReverse can form kr=kf/Kc).

"Symbolic rate constant k(T) for an ElementaryArrhenius law, as a unit-bearing parameter.
 `order` = Σ reactant stoichiometry (for unit derivation). Creates A (and θ, T if needed)."
function _arrhenius_k_param(kin::ElementaryArrhenius, order::Real, nameprefix::AbstractString, T)
    b = kin.b
    A = rate_param(Symbol(nameprefix, "_A"), kin.A, _k_unit(order, b))
    iszero(b) && iszero(kin.Ea) && return A              # constant rate, no T dependence
    iszero(kin.Ea) && return A * T^b                      # Ea=0, b≠0: power-law k=A·T^b, no θ
    # general: k = A·T^b·exp(-θ/T), θ = Ea/R (K) so the exponent is dimensionless
    θ = rate_param(Symbol(nameprefix, "_theta"), kin.Ea / R_GAS, u"K")
    return A * T^b * exp(-θ / T)
end

"True iff rate law `kin` needs a temperature symbol (T-dependent Arrhenius).
 Other kinetics types arrive in Phase 2.5b and declare their own needs there."
_is_T_dependent(kin::ElementaryArrhenius) = !(iszero(kin.b) && iszero(kin.Ea))
_is_T_dependent(kin::ThirdBodyArrhenius) = _is_T_dependent(kin.base)
_is_T_dependent(kin::AbstractFalloff) = true
_is_T_dependent(kin::AbstractKinetics) = false

"True iff any reaction in `mech` needs a T parameter (forward kinetics or reverse)."
_needs_T(mech::Mechanism) =
    any(_is_T_dependent(rx.kinetics) || _reverse_needs_T(rx.reverse_policy) for rx in mech.reactions)
_reverse_needs_T(::ThermoReverse) = true
_reverse_needs_T(::ReverseRatePolicy) = false

"Mass-action product ∏ c[sid]^ν over a stoichiometry map."
function _mass_action(stoich::Dict{SpeciesID,Float64}, cvar)
    ma = 1.0
    for (sid, nu) in stoich
        ma = ma * cvar[sid]^nu
    end
    return ma
end

"Whether a reaction lowers via the Catalyst mass-action backend (spec §3.3/§5.4).
 True for plain elementary Arrhenius (mass-action); false for rate types Catalyst
 does not represent natively (third-body/falloff/PLOG/Chebyshev arrive in 2.5b)."
catalyst_native(rx::ReactionData, config::MechanismConfig) =
    rx.kinetics isa ElementaryArrhenius

"Symbolic NET rate for one reaction (forward minus reverse). Irreducible elementary
 reactions may go via the Catalyst path; everything else (and all reversible reactions)
 use the direct path. `j` is the reaction index (for naming its rate parameters)."
function lower_reaction(rx::ReactionData, mech::Mechanism, cvar, T, config::MechanismConfig, j::Int)
    rx.reverse_policy isa Irreversible ||
        return _net_rate(rx, mech, cvar, T, j)            # ThermoReverse (Task 6)
    return catalyst_native(rx, config) ? catalyst_lowering(rx, mech, cvar, T, j) :
                                         direct_mtk_lowering(rx, mech, cvar, T, j)
end

"Direct-MTK lowering path: build the symbolic rate by dispatching on kinetics type."
direct_mtk_lowering(rx::ReactionData, mech::Mechanism, cvar, T, j::Int) =
    _direct_rate(rx.kinetics, rx, mech, cvar, T, j)

"Elementary Arrhenius forward rate: k(T)·∏ reactants. k is a unit-bearing rate_param.
 Thin wrapper over `_direct_kf` (DRY): rate = kf · mass_action(reactants)."
_direct_rate(kin::ElementaryArrhenius, rx, mech, cvar, T, j) =
    _direct_kf(kin, rx, mech, cvar, T, j) * _mass_action(rx.reactants, cvar)

"Third-body enhanced rate (spec §5.2): k_base(T)·∏ reactants·[M]_eff. The third body is
 NOT a reactant — it enters via the efficiencies map. k_base carries a unit one order higher
 than the elementary base (the [M]_eff factor adds one concentration).
 Thin wrapper over `_direct_kf` (DRY): rate = kf · mass_action(reactants)."
_direct_rate(kin::ThirdBodyArrhenius, rx, mech, cvar, T, j) =
    _direct_kf(kin, rx, mech, cvar, T, j) * _mass_action(rx.reactants, cvar)

"Troe falloff forward rate (spec §5.2, §3.4 #2). k_blend = kinf·(Pr/(1+Pr))·F_Troe, with
 Pr = k0·[M]_eff/kinf (dimensionless). kinf carries the high-pressure (Σν-reactant) unit;
 k0 carries one order higher. Verified 2026-06-26: log10/10^x/exp of symbolic Nums survive
 mtkCompile and the dimension check passes (Pr dimensionless)."
function _direct_rate(kin::TroeFalloff, rx, mech, cvar, T, j)
    T === nothing && error("_direct_rate(TroeFalloff): falloff is T-dependent but no T parameter exists.")
    return _direct_kf(kin, rx, mech, cvar, T, j) * _mass_action(rx.reactants, cvar)
end

"Lindemann falloff forward rate (spec §5.2). F≡1 (no center broadening), so the rate is
 kinf·(Pr/(1+Pr))·∏reactants. Dispatched like TroeFalloff; _direct_kf provides the effective
 forward rate constant so ThermoReverse can compute kr = kf/Kc consistently."
function _direct_rate(kin::LindemannFalloff, rx, mech, cvar, T, j)
    T === nothing && error("_direct_rate(LindemannFalloff): falloff is T-dependent but no T parameter exists.")
    return _direct_kf(kin, rx, mech, cvar, T, j) * _mass_action(rx.reactants, cvar)
end

# —— Effective forward rate constant (for ThermoReverse: kr = kf/Kc) ————————————
# `_direct_kf` returns the rate-constant factor EXCLUDING the reactant mass-action term,
# so `_reverse_rate(::ThermoReverse)` can compute kr = kf/Kc consistently.
# DRY: `_direct_rate = _direct_kf * _mass_action(rx.reactants, cvar)` for each kinetics type.
# Phase 5a T5fix: previously _net_rate hardcoded an ElementaryArrhenius guard; now any
# kinetics with a _direct_kf method is accepted (ThirdBody, Troe, Elementary).

"_direct_kf for ElementaryArrhenius: k(T) = A·T^b·exp(-θ/T) with the reactant-order unit."
_direct_kf(kin::ElementaryArrhenius, rx, mech, cvar, T, j) =
    _arrhenius_k_param(kin, sum(values(rx.reactants)), "k_$j", T)

"_direct_kf for ThirdBodyArrhenius: k_base(T)·[M]_eff (order = Σ reactants + 1 for [M]_eff)."
function _direct_kf(kin::ThirdBodyArrhenius, rx, mech, cvar, T, j)
    order = sum(values(rx.reactants)) + 1                  # +1 for [M]_eff
    return _arrhenius_k_param(kin.base, order, "k_$j", T) * _meff(mech, kin.efficiencies, cvar)
end

"_direct_kf for TroeFalloff: kinf·(Pr/(1+Pr))·F_Troe with Pr = k0·[M]_eff/kinf."
function _direct_kf(kin::TroeFalloff, rx, mech, cvar, T, j)
    T === nothing && error("_direct_kf(TroeFalloff): falloff is T-dependent but no T parameter exists.")
    base_order = sum(values(rx.reactants))
    kinf = _arrhenius_k_param(kin.high_rate, base_order,     "k_$j" * "_high", T)
    k0   = _arrhenius_k_param(kin.low_rate,  base_order + 1, "k_$j" * "_low",  T)
    meff = _meff(mech, kin.efficiencies, cvar)
    Pr   = k0 * meff / kinf
    F    = _troe_F(kin.troe, Pr, T, j)
    return kinf * (Pr / (1 + Pr)) * F
end

"_direct_kf for LindemannFalloff: kinf·Pr/(1+Pr) with Pr = k0·[M]_eff/kinf (F≡1, no Troe center
 broadening). kinf carries the high-pressure (Σν-reactant) unit; k0 carries one order higher."
function _direct_kf(kin::LindemannFalloff, rx, mech, cvar, T, j)
    T === nothing && error("_direct_kf(LindemannFalloff): falloff is T-dependent but no T parameter exists.")
    base_order = sum(values(rx.reactants))
    kinf = _arrhenius_k_param(kin.high_rate, base_order,     "k_$j" * "_high", T)
    k0   = _arrhenius_k_param(kin.low_rate,  base_order + 1, "k_$j" * "_low",  T)
    meff = _meff(mech, kin.efficiencies, cvar)
    Pr   = k0 * meff / kinf
    return kinf * (Pr / (1 + Pr))
end

_direct_kf(kin::AbstractKinetics, rx, mech, cvar, T, j) =
    error("_direct_kf: not implemented for $(typeof(kin)); arrives in Phase 6.")

"Troe center-broadening factor F (TroeParams α, T1, T2, T3). T1/T2/T3 are temperatures
 (K); under units they MUST be K-params so T/T1 etc. are dimensionless — a bare Float64
 would make exp(-T/T3) dimensional and fail the dim check (verified 2026-06-26: bare-T3
 → ValidationError; K-param T3 → passes)."
function _troe_F(tp::TroeParams, Pr, T, j)
    α = tp.α
    T1 = rate_param(Symbol("k_", j, "_troeT1"), tp.T1, u"K")
    T2 = rate_param(Symbol("k_", j, "_troeT2"), tp.T2, u"K")
    T3 = rate_param(Symbol("k_", j, "_troeT3"), tp.T3, u"K")
    Fcent = (1 - α) * exp(-T / T3) + α * exp(-T / T1) + exp(-T2 / T)
    lFc = log10(Fcent); lPr = log10(Pr)
    c = -0.4 - 0.67 * lFc; N = 0.75 - 1.27 * lFc
    f1 = lPr + c; f2 = N - 0.14 * f1
    return 10^(lFc / (1 + (f1 / f2)^2))
end

"Effective third-body concentration [M]_eff = Σ_i α_i·[X_i] over all species (default α=1)."
function _meff(mech::Mechanism, efficiencies::Dict{SpeciesID,Float64}, cvar)
    m = 0.0
    for sp in mech.species
        alpha = get(efficiencies, sp.id, 1.0)
        m += alpha * cvar[sp.id]
    end
    return m
end

# Fallback for kinetics types not yet unit-aware (third-body/Troe/etc. arrive in Tasks 2-4).
_direct_rate(kin::AbstractKinetics, rx, mech, cvar, T, j) =
    error("_direct_rate: unit-aware lowering for $(typeof(kin)) arrives in a later task.")

"Catalyst mass-action lowering path (spec §5.4). Builds a Catalyst.Reaction on the
 shared @species with the SAME unit-bearing k, then reads its rate law via oderatelaw."
function catalyst_lowering(rx::ReactionData, mech::Mechanism, cvar, T, j::Int)
    kin = rx.kinetics
    kin isa ElementaryArrhenius ||
        error("catalyst_lowering: only ElementaryArrhenius is Catalyst-native so far.")
    order = sum(values(rx.reactants))
    k = _arrhenius_k_param(kin, order, "k_$j", T)
    subs       = [cvar[sid] for sid in keys(rx.reactants)]
    substoich  = collect(values(rx.reactants))
    prods      = [cvar[sid] for sid in keys(rx.products)]
    prodstoich = collect(values(rx.products))
    crate = Catalyst.Reaction(k, subs, prods, substoich, prodstoich)
    return Catalyst.oderatelaw(crate; combinatoric_ratelaw=false)
end
