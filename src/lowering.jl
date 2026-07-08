# Lowering pipeline: Mechanism + config → MTK ODESystem (spec §5.4).
# Phase 2.5b: unit-aware lowering (§5.6). @species carry [unit=conc], T [unit=K],
# and each reaction's rate constant k is a unit-bearing parameter (stoichiometrically
# derived unit), so MTK's dimension check fires at System construction.
# The Catalyst mass-action backend (catalyst_lowering via oderatelaw) shares the
# same unit-bearing k. Constraint-layer assembly (append_constraint_layers!) is a
# stub (Phase 4).

using DynamicQuantities     # for u"..." unit literals in rate-param construction

# R_GAS now lives in src/data/types.jl (moved in Task 5; visible here because
# types.jl is included before lowering.jl in ChemMechSim.jl).

"Attach a DynamicQuantities unit to a symbolic variable/parameter (the @species/@parameters
 macros reject interpolated names with [unit=...], so units are attached via setmetadata)."
_attach_unit(sym, unit) = ModelingToolkit.setmetadata(sym, ModelingToolkit.VariableUnit, unit)

"Build a rate-constant parameter `name` with `default` value and a derived `unit` (§5.6.5).
 The @parameters macro only accepts a LITERAL default with interpolation, so create with a
 placeholder then setdefault + setmetadata."
function rate_param(name::Symbol, default, unit)
    kp = only(@parameters ($(name)) = 1.0)
    kp = ModelingToolkit.setdefault(kp, default)
    return _attach_unit(kp, unit)
end

"Expected unit of a rate constant for overall reaction order `order` (Σ reactant stoich,
 incl. the third-body/[M] factor where applicable) and Arrhenius exponent `b`:
 [k] = conc^(1-order)·s⁻¹ ; the A-factor absorbs T^b -> [A] = [k] / K^b."
_k_unit(order, b) = ChemUnits.conc^(1 - order) * u"s^-1" / (u"K"^b)

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

# Lowering-scoped singletons for the thermo constants R and P° (shared across all K_c uses, so
# every reaction's (P°/RT)^Δν factor references the SAME parameter — no duplicate-name params).
# Reset at the start of each lower_to_mtk call.
const _PSTD_PARAM = Ref{Any}(nothing)
const _RGAS_PARAM = Ref{Any}(nothing)
_p_std_param() = _PSTD_PARAM[] === nothing ? (_PSTD_PARAM[] = rate_param(:P_std, P_STD, u"Pa")) : _PSTD_PARAM[]
_r_param()     = _RGAS_PARAM[] === nothing ? (_RGAS_PARAM[] = rate_param(:R_gas, R_GAS, u"J/(mol*K)")) : _RGAS_PARAM[]

# Per-species NASA7 coefficient cache (Phase 4a). cp/R, h/RT and g/RT for one species share
# ONE coefficient set; without this, calling each separately creates duplicate-name params
# (sp{sid}_a1l, …). Reset at the start of each lower_to_mtk call.
const _COEFF_CACHE = Ref{Dict{Int,Any}}(Dict{Int,Any}())

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
    Fcent = (1 - α) * exp(-T / T3) + α * exp(-T / T1) + exp(-T / T2)
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

# ThermoReverse net rate: forward minus reverse (Task 6, §3.4 #4).

"Net rate = forward − reverse for a reversible reaction (direct path). ExplicitReverse uses the
 general forward dispatch (_direct_rate) since its reverse is independent of the forward k_f.
 ThermoReverse needs the forward k_f (for k_r = k_f/K_c); `_direct_kf` provides the per-kinetics
 effective rate constant (elementary, ThirdBody, or Troe) so K_c-based reverse works for any
 forward kinetics type. Phase 5a T5fix removed the elementary-only guard that previously blocked
 ThirdBody/Troe under ThermoReverse."
function _net_rate(rx::ReactionData, mech, cvar, T, j)
    policy = rx.reverse_policy
    if policy isa ExplicitReverse
        fwd = _direct_rate(rx.kinetics, rx, mech, cvar, T, j)
        return fwd - _reverse_rate(policy, rx, mech, cvar, T, j)
    end
    if policy isa ThermoReverse
        kf = _direct_kf(rx.kinetics, rx, mech, cvar, T, j)
        fwd = kf * _mass_action(rx.reactants, cvar)
        return fwd - _reverse_rate(policy, rx, mech, cvar, T, kf)
    end
    error("_net_rate: unsupported policy $(typeof(policy))")
end

_reverse_rate(::Irreversible, rx, mech, cvar, T, kf) = 0.0
function _reverse_rate(::ThermoReverse, rx, mech, cvar, T, kf)
    T === nothing &&
        error("_reverse_rate(ThermoReverse): K_c(T) needs a T parameter, but none exists.")
    Kc = _equilibrium_constant(mech, rx, T)
    return (kf / Kc) * _mass_action(rx.products, cvar)
end

"Reverse rate for an ExplicitReverse policy: kr(T)·∏products, kr from the policy's own rate law.
 Phase 3 supports policy.rate::ElementaryArrhenius (the common explicit-reverse form)."
function _reverse_rate(policy::ExplicitReverse, rx::ReactionData, mech, cvar, T, j)
    policy.rate isa ElementaryArrhenius ||
        error("_reverse_rate(ExplicitReverse): non-Arrhenius reverse rate ($(typeof(policy.rate))) " *
              "deferred; use ElementaryArrhenius.")
    order = sum(values(rx.products))                       # reverse "reactants" = forward products
    kr = _arrhenius_k_param(policy.rate, order, "k_$(j)_rev", T)   # _rev suffix avoids name clash
    return kr * _mass_action(rx.products, cvar)
end

"Equilibrium constant K_c(T) = exp(-Δg°/RT)·(P°/(R·T))^Δν (spec §3.4 #4, §4.2).
 Δν = Σν_products − Σν_reactants. Δν=0 → factor is 1 (the existing behavior; current tests unaffected).
 Δg°/RT is dimensionless and (P°/RT)^Δν carries the concentration-basis unit — both pass the dim check."
function _equilibrium_constant(mech::Mechanism, rx::ReactionData, T)
    dnu = sum(values(rx.products)) - sum(values(rx.reactants))
    base = exp(-_delta_g_over_RT(mech, rx, T))
    iszero(dnu) && return base
    return base * (_p_std_param() / (_r_param() * T))^dnu
end

function _delta_g_over_RT(mech::Mechanism, rx::ReactionData, T)
    g = 0.0
    for (sid, nu) in rx.products;  g += nu * _g_over_RT(_thermo_of(mech, sid), T, sid); end
    for (sid, nu) in rx.reactants; g -= nu * _g_over_RT(_thermo_of(mech, sid), T, sid); end
    return g
end

"Dimensionless g/RT from NASA7 thermo. For plain Real T (e.g. the numeric K_c test)
 delegates directly to the data-layer g_over_RT. For a symbolic/unit-bearing T (the
 K-param in lowering) the NASA7 coefficients must carry unit metadata so each polynomial
 term is dimensionless under MTK's dim check: a2..a5 absorb powers of K^-n (so a_i·T^i is
 dimensionless), a6 has unit K (so a6/T is dimensionless), a1/a7 are dimensionless. Tmid
 also needs unit K for the range comparison. Coefficients are created as unit-bearing
 rate_params (default = stored value, metadata unit) so MTK's dim check validates but the
 generated code uses plain Float64s. `sid` names params uniquely per species. Verified
 2026-06-26: K_c = exp(-Δg°/RT) is dimensionless and passes the dim check. NOTE: Num <: Real
 in Julia, so the symbolic method is dispatched via T::Num (more specific than Real)."
_g_over_RT(m::NASA7, T::Real, sid) = g_over_RT(m, T)

"Build NASA7's 7 unit-bearing symbolic coefficients for species `sid`, selected by
 `ifelse(T <= Tmid, lo, hi)`. a_i·T^(i-1) is dimensionless (i=1..5), a6∈K makes a6/T and a6
 dimensionless via the 1/T and (h/RT) terms, a1/a7 are dimensionless, Tmid∈K makes the range
 test unit-consistent, Tref=1 K makes log(T/Tref) dimensionless. Cached per sid within one
 lower_to_mtk call so cp/R, h/RT, g/RT share one coefficient set (no duplicate-name params).
 Returns ((a1..a7), Tmid_K, T_ref). Verified 2026-07-02."
function _nasa7_coeffs_sym(m::NASA7, T, sid)
    c = get(_COEFF_CACHE[], sid, nothing)
    c === nothing || return c
    Tmid_K = rate_param(Symbol("Tmid_sp", sid), m.Tmid, u"K")
    T_ref  = rate_param(Symbol("Tref_sp", sid), 1.0, u"K")
    lo, hi = m.low_coeffs, m.high_coeffs
    a1 = ifelse(T <= Tmid_K, _sp(sid, :a1l, lo[1], u"1"),    _sp(sid, :a1h, hi[1], u"1"))
    a2 = ifelse(T <= Tmid_K, _sp(sid, :a2l, lo[2], u"K^-1"), _sp(sid, :a2h, hi[2], u"K^-1"))
    a3 = ifelse(T <= Tmid_K, _sp(sid, :a3l, lo[3], u"K^-2"), _sp(sid, :a3h, hi[3], u"K^-2"))
    a4 = ifelse(T <= Tmid_K, _sp(sid, :a4l, lo[4], u"K^-3"), _sp(sid, :a4h, hi[4], u"K^-3"))
    a5 = ifelse(T <= Tmid_K, _sp(sid, :a5l, lo[5], u"K^-4"), _sp(sid, :a5h, hi[5], u"K^-4"))
    a6 = ifelse(T <= Tmid_K, _sp(sid, :a6l, lo[6], u"K"),    _sp(sid, :a6h, hi[6], u"K"))
    a7 = ifelse(T <= Tmid_K, _sp(sid, :a7l, lo[7], u"1"),    _sp(sid, :a7h, hi[7], u"1"))
    c = ((a1, a2, a3, a4, a5, a6, a7), Tmid_K, T_ref)
    _COEFF_CACHE[][sid] = c
    return c
end

"Symbolic dimensionless cp/R for a unit-bearing T (energy equation, Phase 4a)."
function _cp_over_R(m::NASA7, T::Num, sid)
    (a1, a2, a3, a4, a5, a6, a7), _, _ = _nasa7_coeffs_sym(m, T, sid)
    return a1 + a2 * T + a3 * T^2 + a4 * T^3 + a5 * T^4
end

"Symbolic dimensionless h/RT for a unit-bearing T (energy equation, Phase 4a)."
function _h_over_RT(m::NASA7, T::Num, sid)
    (a1, a2, a3, a4, a5, a6, a7), _, _ = _nasa7_coeffs_sym(m, T, sid)
    return a1 + a2 * T / 2 + a3 * T^2 / 3 + a4 * T^3 / 4 + a5 * T^4 / 5 + a6 / T
end

"Dimensionless g/RT from NASA7 thermo for a symbolic/unit-bearing T (the K-param in lowering).
 Refactored Phase 4a to share `_nasa7_coeffs_sym` with cp/R and h/RT. The bare `ifelse`
 LOWERS to a correct runtime `if (T <= Tmid) … else …` branch and the dimension check passes
 (verified 2026-06-29: distinct low/high coeffs give K_c=2 below Tmid, K_c=5 above, exact match)."
function _g_over_RT(m::NASA7, T::Num, sid)
    (a1, a2, a3, a4, a5, a6, a7), _, T_ref = _nasa7_coeffs_sym(m, T, sid)
    h_RT = a1 + a2 * T / 2 + a3 * T^2 / 3 + a4 * T^3 / 4 + a5 * T^4 / 5 + a6 / T
    s_R  = a1 * log(T / T_ref) + a2 * T + a3 * T^2 / 2 + a4 * T^3 / 3 + a5 * T^4 / 4 + a7
    return h_RT - s_R
end
_g_over_RT(m::ThermoModel, T, sid) = error("_g_over_RT: thermo model $(typeof(m)) unsupported; only NASA7.")

"Species-keyed unit-bearing parameter for NASA7 coefficients (rate_param wrapper)."
_sp(sid, tag, val, unit) = rate_param(Symbol("sp", sid, "_", tag), val, unit)

_species_by_id(mech::Mechanism, sid::SpeciesID) = species_by_id(mech, sid)
function _thermo_of(mech::Mechanism, sid::SpeciesID)
    th = _species_by_id(mech, sid).thermo
    th === nothing && error("_thermo_of: species id $sid has no thermo; ThermoReverse needs NASA7 on all species.")
    return th
end

"net rate of change Σⱼ netstoichⱼᵢ·rateⱼ for the species with id `sid`."
function _species_rhs(sid::SpeciesID, mech::Mechanism, rates)
    rhs = 0.0
    for (j, rx) in enumerate(mech.reactions)
        net = get(rx.products, sid, 0.0) - get(rx.reactants, sid, 0.0)
        iszero(net) || (rhs += net * rates[j])
    end
    return rhs
end

"True iff `config` is lowerable: concentration basis; energy ∈ {:isothermal,:adiabatic};
 constraint ∈ {:none,:constant_volume,:constant_pressure}; :adiabatic allows const-V or const-P,
 :isothermal allows any constraint; eos ∈ {:off,:ideal_gas}; :constant_pressure REQUIRES :ideal_gas
 (V is defined by the EOS). Phase 4b adds const-P (path A, pure ODE)."
_lowerable(c::MechanismConfig) =
    c.state_basis === :concentration &&
    c.energy in (:isothermal, :adiabatic) &&
    (c.energy === :adiabatic ? c.constraint in (:constant_volume, :constant_pressure)
                             : c.constraint in (:none, :constant_volume, :constant_pressure)) &&
    (c.constraint === :constant_pressure ? c.eos === :ideal_gas : c.eos in (:off, :ideal_gas))

"Lower a Mechanism into a structural_simplify'd MTK ODESystem (unit-aware, zero-point).
 Species get [unit=conc]; T [unit=K] is created iff any reaction is T-dependent or
 ThermoReverse. Each reaction's rate constant is a unit-bearing parameter (default = stored
 value), so MTK's dimension check fires at System construction (§5.6)."
function lower_to_mtk(mech::Mechanism; config::MechanismConfig=MechanismConfig())
    _PSTD_PARAM[] = nothing        # reset lowering-scoped thermo-constant singletons (Task 4)
    _RGAS_PARAM[] = nothing
    _COEFF_CACHE[] = Dict{Int,Any}()   # per-species NASA7 coeff cache (Phase 4a)
    _lowerable(config) ||
        error("lower_to_mtk: config not supported in Phase 4a. Supported: energy∈{:isothermal,:adiabatic}, " *
              "state_basis=:concentration, eos∈{:off,:ideal_gas}; constraint must be :constant_volume for " *
              ":adiabatic; non-concentration bases arrive in Phase 4b+. Got: " *
              "energy=$(config.energy) constraint=$(config.constraint) eos=$(config.eos) " *
              "basis=$(config.state_basis). Use MechanismConfig() (:kinetic), convenience_config(:fixedT), " *
              "or convenience_config(:adiabatic_constV).")
    config.constraint === :constant_pressure && return _lower_constP(mech, config)
    t = ModelingToolkit.t
    D = ModelingToolkit.D
    cvars = [_attach_unit(only(@species ($(Symbol(sp.name)))(t)), ChemUnits.conc)
             for sp in mech.species]
    cvar = Dict(mech.species[i].id => cvars[i] for i in eachindex(mech.species))
    is_adiabatic = config.energy === :adiabatic
    # T exists iff any reaction is T-dependent, EOS needs it, OR the energy layer makes it a state.
    # Under :adiabatic T is a STATE (@variables); otherwise a unit-bearing parameter.
    needs_T = _needs_T(mech) || config.eos === :ideal_gas || is_adiabatic
    Tparam = if !needs_T
        nothing
    elseif is_adiabatic
        _attach_unit(only(@variables T(t)), u"K")     # temperature STATE
    else
        rate_param(:T, 300.0, u"K")                    # temperature parameter (isothermal)
    end
    rates = [lower_reaction(rx, mech, cvar, Tparam, config, j)
             for (j, rx) in enumerate(mech.reactions)]
    eqs = [D(cvars[i]) ~ _species_rhs(mech.species[i].id, mech, rates)
           for i in eachindex(mech.species)]
    # Constraint-layer assembly (energy layer adds the const-V dT/dt under :adiabatic; spec §5.4).
    eqs = append_constraint_layers!(eqs, mech, config, cvar, Tparam, rates)
    if config.eos === :ideal_gas
        return _lower_with_eos(eqs, t, cvars, Tparam, is_adiabatic)
    end
    @named raw = System(eqs, t)          # auto-discovers states [c₁..cₙ, T] and RHS params
    return mtkcompile(raw)
end

"Build the system with EOS observed P ~ (Σc)·R·T. Under :adiabatic T is a STATE (included in
 `states`); under :isothermal T is a parameter retained via the observed-param fix (Phase 3).
 R appears in the energy-equation RHS under :adiabatic so it is retained automatically."
function _lower_with_eos(eqs, t, cvars, Tparam, is_adiabatic::Bool)
    Rparam = _r_param()                            # shared with K_c / energy eq (memoized singleton)
    Pvar = _attach_unit(only(@variables P(t)), ChemUnits.press)
    obs = [Pvar ~ sum(cvars) * Rparam * Tparam]
    states = is_adiabatic ? [cvars; Tparam] : cvars   # T is a state under :adiabatic
    @named _tmp = System(eqs, t)                   # auto-discover RHS params
    rhsparams = ModelingToolkit.parameters(_tmp)
    rhsnames = Set(ModelingToolkit.getname(p) for p in rhsparams)
    extras = Any[]
    ModelingToolkit.getname(Rparam) in rhsnames || push!(extras, Rparam)
    is_adiabatic || (ModelingToolkit.getname(Tparam) in rhsnames || push!(extras, Tparam))  # T param only when isothermal
    @named raw = System(eqs, t, states, [rhsparams; extras]; observed=obs)
    return mtkcompile(raw)
end

"Lower under :constant_pressure — path A, PURE ODE (probed 2026-07-03 P1/P2).
 Moles `nᵢ` are the states; `V ~ (Σn)RT/P` and concentrations `cᵢ ~ nᵢ/V` are observed (structural_simplify
 flattens them). The rate laws consume `cvar` (concentrations) UNCHANGED — only the outer assembly differs
 from the concentration-state path. :adiabatic adds the enthalpy energy equation via append_constraint_layers!."
function _lower_constP(mech::Mechanism, config::MechanismConfig)
    t = ModelingToolkit.t; D = ModelingToolkit.D
    is_adiabatic = config.energy === :adiabatic
    cvars = [_attach_unit(only(@species ($(Symbol(sp.name)))(t)), ChemUnits.conc) for sp in mech.species]
    cvar  = Dict(mech.species[i].id => cvars[i] for i in eachindex(mech.species))
    nvars = [_attach_unit(only(@species ($(Symbol("n_", sp.name)))(t)), ChemUnits.mol) for sp in mech.species]
    nvar  = Dict(mech.species[i].id => nvars[i] for i in eachindex(mech.species))
    Pparam = rate_param(:P, P_STD, ChemUnits.press)
    Tsym   = is_adiabatic ? _attach_unit(only(@variables T(t)), u"K") : rate_param(:T, 300.0, u"K")
    Rparam = _r_param()
    Vvar   = _attach_unit(only(@variables V(t)), ChemUnits.vol)
    rates  = [lower_reaction(rx, mech, cvar, Tsym, config, j) for (j, rx) in enumerate(mech.reactions)]
    eqs = Equation[D(nvars[i]) ~ Vvar * _species_rhs(mech.species[i].id, mech, rates)
              for i in eachindex(mech.species)]
    push!(eqs, Vvar ~ sum(nvars) * Rparam * Tsym / Pparam)            # EOS → observed V
    for i in eachindex(mech.species)
        push!(eqs, cvars[i] ~ nvars[i] / Vvar)                        # observed concentrations (rate-law input)
    end
    eqs = append_constraint_layers!(eqs, mech, config, cvar, Tsym, rates; nvar=nvar, Vvar=Vvar)
    @named raw = System(eqs, t)
    return mtkcompile(raw)
end

# —— Constraint-layer assembly (Phase 4a: energy layer; EOS-as-DAE + const-P arrive in 4b) ——

"Append energy/reactor constraint layers to the equation set (spec §5.4). Phase 4a: the energy
 layer (:adiabatic) adds the const-V energy ODE for T. `cvar`/`T`/`rates` are the shared species
 vars, the temperature symbol, and the per-reaction symbolic net rates from lower_to_mtk."
function append_constraint_layers!(eqs, mech, config, cvar, T, rates; nvar=nothing, Vvar=nothing)
    config.energy === :adiabatic || return eqs
    if config.constraint === :constant_pressure
        push!(eqs, _energy_ode_constP(mech, nvar, Vvar, T, rates))    # Task 2
    else
        push!(eqs, _energy_ode_constV(mech, cvar, T, rates))
    end
    return eqs
end

"Constant-pressure adiabatic energy equation (spec §5.3/§11 Phase 4; probed 2026-07-03 P2):
 dT/dt = -V·Σⱼ rⱼ·Δh̄ⱼ / Σᵢ nᵢ·cpᵢ, with Δh̄ⱼ = Σ_prod ν·h̄ − Σ_react ν·h̄ and cpᵢ = (cp/R)·R (ideal gas).
 All species must carry NASA7 thermo (spec §5.3.4 — clear error otherwise). H = Σnᵢh̄ᵢ(T) is conserved."
function _energy_ode_constP(mech::Mechanism, nvar, Vvar, T, rates)
    D = ModelingToolkit.D
    R = _r_param()
    for sp in mech.species
        sp.thermo isa NASA7 ||
            error("_energy_ode_constP: species $(sp.name) (id $(sp.id)) has no NASA7 thermo; " *
                  ":adiabatic requires NASA7 thermo on all species (spec §5.3.4). " *
                  "Use energy=:isothermal or provide NASA7 thermo.")
    end
    cp_sum = sum(nvar[sp.id] * _cp_over_R(sp.thermo, T, sp.id) * R for sp in mech.species)   # [J/K]
    src = 0.0
    for (j, rx) in enumerate(mech.reactions)
        delta_h = 0.0
        for (sid, nu) in rx.products
            delta_h += nu * _h_over_RT(_species_by_id(mech, sid).thermo, T, sid) * R * T
        end
        for (sid, nu) in rx.reactants
            delta_h -= nu * _h_over_RT(_species_by_id(mech, sid).thermo, T, sid) * R * T
        end
        src += rates[j] * (-delta_h)                                  # Σⱼ rⱼ·(-Δh̄ⱼ)  [J/(m³·s)]
    end
    return D(T) ~ Vvar * src / cp_sum
end

"Constant-volume adiabatic energy equation (spec §5.3, §11 Phase 4; verified 2026-07-02):
 dT/dt = -Σⱼ rⱼ·Δūⱼ / Σᵢ cᵢ·cvᵢ, with ūᵢ=(h/RT-1)·R·T and cvᵢ=(cp/R-1)·R (ideal gas).
 All species must carry NASA7 thermo (spec §5.3.4 — clear error otherwise)."
function _energy_ode_constV(mech::Mechanism, cvar, T, rates)
    D = ModelingToolkit.D
    R = _r_param()
    for sp in mech.species
        sp.thermo isa NASA7 ||
            error("_energy_ode_constV: species $(sp.name) (id $(sp.id)) has no NASA7 thermo; " *
                  ":adiabatic requires NASA7 thermo on all species (spec §5.3.4). " *
                  "Use energy=:isothermal or provide NASA7 thermo.")
    end
    # Σᵢ cᵢ·cvᵢ  [J/(m³·K)]
    cv_sum = sum(cvar[sp.id] * (_cp_over_R(sp.thermo, T, sp.id) - 1) * R for sp in mech.species)
    # -Σⱼ rⱼ·Δūⱼ  [J/(m³·s)],  Δūⱼ = Σ_products ν·ū − Σ_reactants ν·ū
    src = 0.0
    for (j, rx) in enumerate(mech.reactions)
        delta_u = 0.0
        for (sid, nu) in rx.products
            th = _species_by_id(mech, sid).thermo
            delta_u += nu * (_h_over_RT(th, T, sid) - 1) * R * T
        end
        for (sid, nu) in rx.reactants
            th = _species_by_id(mech, sid).thermo
            delta_u -= nu * (_h_over_RT(th, T, sid) - 1) * R * T
        end
        src += rates[j] * (-delta_u)
    end
    return D(T) ~ src / cv_sum
end
