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
    if iszero(b) && iszero(kin.Ea)
        return A                                   # constant rate, no T dependence
    end
    # T-dependent: k = A·T^b·exp(-θ/T), θ = Ea/R (K) so the exponent is dimensionless
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

"Elementary Arrhenius forward rate: k(T)·∏ reactants. k is a unit-bearing rate_param."
function _direct_rate(kin::ElementaryArrhenius, rx, mech, cvar, T, j)
    order = sum(values(rx.reactants))
    k = _arrhenius_k_param(kin, order, "k_$j", T)
    return k * _mass_action(rx.reactants, cvar)
end

"Third-body enhanced rate (spec §5.2): k_base(T)·∏ reactants·[M]_eff. The third body is
 NOT a reactant — it enters via the efficiencies map. k_base carries a unit one order higher
 than the elementary base (the [M]_eff factor adds one concentration)."
function _direct_rate(kin::ThirdBodyArrhenius, rx, mech, cvar, T, j)
    base_order = sum(values(rx.reactants))
    order = base_order + 1                              # +1 for the [M]_eff factor
    k = _arrhenius_k_param(kin.base, order, "k_$j", T)
    return k * _mass_action(rx.reactants, cvar) * _meff(mech, kin.efficiencies, cvar)
end

"Troe falloff forward rate (spec §5.2, §3.4 #2). k_blend = kinf·(Pr/(1+Pr))·F_Troe, with
 Pr = k0·[M]_eff/kinf (dimensionless). kinf carries the high-pressure (Σν-reactant) unit;
 k0 carries one order higher. Verified 2026-06-26: log10/10^x/exp of symbolic Nums survive
 mtkCompile and the dimension check passes (Pr dimensionless)."
function _direct_rate(kin::TroeFalloff, rx, mech, cvar, T, j)
    T === nothing && error("_direct_rate(TroeFalloff): falloff is T-dependent but no T parameter exists.")
    base_order = sum(values(rx.reactants))
    kinf = _arrhenius_k_param(kin.high_rate, base_order,     "k_$j" * "_high", T)
    k0   = _arrhenius_k_param(kin.low_rate,  base_order + 1, "k_$j" * "_low",  T)
    meff = _meff(mech, kin.efficiencies, cvar)
    Pr   = k0 * meff / kinf
    F    = _troe_F(kin.troe, Pr, T, j)
    k    = kinf * (Pr / (1 + Pr)) * F
    return k * _mass_action(rx.reactants, cvar)
end

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

"Net rate = forward - reverse for a reversible reaction (direct path)."
function _net_rate(rx::ReactionData, mech, cvar, T, j)
    rx.kinetics isa ElementaryArrhenius ||
        error("_net_rate: reverse rates for non-elementary kinetics ($(typeof(rx.kinetics))) " *
              "are not supported in this spike; use Irreversible or an explicit reverse.")
    order = sum(values(rx.reactants))
    kf = _arrhenius_k_param(rx.kinetics, order, "k_$j", T)
    fwd = kf * _mass_action(rx.reactants, cvar)
    return fwd - _reverse_rate(rx.reverse_policy, rx, mech, cvar, T, kf)
end

_reverse_rate(::Irreversible, rx, mech, cvar, T, kf) = 0.0
function _reverse_rate(::ThermoReverse, rx, mech, cvar, T, kf)
    T === nothing &&
        error("_reverse_rate(ThermoReverse): K_c(T) needs a T parameter, but none exists.")
    Kc = _equilibrium_constant(mech, rx, T)
    return (kf / Kc) * _mass_action(rx.products, cvar)
end

"Equilibrium constant K_c(T) = exp(-Δg°/RT) from NASA7 thermo (§3.4 #4). Δg°/RT is
 dimensionless, so exp is dimensionless — passes the dim check under units (verified
 2026-06-26). The general (P°/RT)^Δν factor for Δν≠0 is deferred."
_equilibrium_constant(mech::Mechanism, rx::ReactionData, T) = exp(-_delta_g_over_RT(mech, rx, T))

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

"Known limitation (review I1): piecewise NASA7 — the low/high coefficient range switch at
 Tmid — is NOT robustly supported for symbolic T. This method builds both branches eagerly
 via Symbolics `ifelse`, which does not reliably lower as an ODE rate expression for real
 codegen; with distinct low/high coeffs across Tmid it can silently pick the wrong branch
 or fail. The Task 6 tests pass only because the test species use IDENTICAL low/high coeff
 sets (`NASA7(base, base, …)`), so both branches coincide. Real-species mechanisms with
 distinct low/high NASA7 coeffs must revisit this (deferred — the §3.4 spike uses
 constructed identical-coeff species). The implementation assumes a single representative
 coefficient set; no attempt is made here to fix the range switching, which is a
 fundamental symbolic-T limitation out of scope for the spike."
function _g_over_RT(m::NASA7, T::Num, sid)
    Tmid_K = rate_param(Symbol("Tmid_sp", sid), m.Tmid, u"K")  # K-typed so T <= Tmid is unit-consistent
    T_ref = rate_param(Symbol("Tref_sp", sid), 1.0, u"K")     # 1 K reference so log(T/T_ref) is dimensionless
    lo, hi = m.low_coeffs, m.high_coeffs
    # Coeffs as unit-bearing params: a_i·T^(i-1) dimensionless for i=1..5; a6·(1/T) dimensionless
    a1 = ifelse(T <= Tmid_K, _sp(sid, :a1l, lo[1], u"1"),       _sp(sid, :a1h, hi[1], u"1"))
    a2 = ifelse(T <= Tmid_K, _sp(sid, :a2l, lo[2], u"K^-1"),    _sp(sid, :a2h, hi[2], u"K^-1"))
    a3 = ifelse(T <= Tmid_K, _sp(sid, :a3l, lo[3], u"K^-2"),    _sp(sid, :a3h, hi[3], u"K^-2"))
    a4 = ifelse(T <= Tmid_K, _sp(sid, :a4l, lo[4], u"K^-3"),    _sp(sid, :a4h, hi[4], u"K^-3"))
    a5 = ifelse(T <= Tmid_K, _sp(sid, :a5l, lo[5], u"K^-4"),    _sp(sid, :a5h, hi[5], u"K^-4"))
    a6 = ifelse(T <= Tmid_K, _sp(sid, :a6l, lo[6], u"K"),       _sp(sid, :a6h, hi[6], u"K"))
    a7 = ifelse(T <= Tmid_K, _sp(sid, :a7l, lo[7], u"1"),       _sp(sid, :a7h, hi[7], u"1"))
    h_RT = a1 + a2 * T / 2 + a3 * T^2 / 3 + a4 * T^3 / 4 + a5 * T^4 / 5 + a6 / T
    s_R  = a1 * log(T / T_ref) + a2 * T + a3 * T^2 / 2 + a4 * T^3 / 3 + a5 * T^4 / 4 + a7
    return h_RT - s_R
end
_g_over_RT(m::ThermoModel, T, sid) = error("_g_over_RT: thermo model $(typeof(m)) unsupported; only NASA7.")

"Species-keyed unit-bearing parameter for NASA7 coefficients (rate_param wrapper)."
_sp(sid, tag, val, unit) = rate_param(Symbol("sp", sid, "_", tag), val, unit)

_species_by_id(mech::Mechanism, sid::SpeciesID) = mech.species[sid]
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

"True iff `config` is the :kinetic zero-point (the only config lowered so far)."
function _is_zero_point(c::MechanismConfig)
    return c.energy === :isothermal && c.constraint === :none && c.eos === :off &&
           c.thermo_data === :none && c.reverse_rate === :irreversible &&
           c.state_basis === :concentration
end

"Lower a Mechanism into a structural_simplify'd MTK ODESystem (unit-aware, zero-point).
 Species get [unit=conc]; T [unit=K] is created iff any reaction is T-dependent or
 ThermoReverse. Each reaction's rate constant is a unit-bearing parameter (default = stored
 value), so MTK's dimension check fires at System construction (§5.6)."
function lower_to_mtk(mech::Mechanism; config::MechanismConfig=MechanismConfig())
    _is_zero_point(config) ||
        error("lower_to_mtk: only the :kinetic zero-point config (MechanismConfig()) is supported so far; " *
              "energy/EOS/thermo layers arrive in later phases.")
    t = ModelingToolkit.t
    D = ModelingToolkit.D
    cvars = [_attach_unit(only(@species ($(Symbol(sp.name)))(t)), ChemUnits.conc)
             for sp in mech.species]
    cvar = Dict(mech.species[i].id => cvars[i] for i in eachindex(mech.species))
    Tparam = _needs_T(mech) ? rate_param(:T, 300.0, u"K") : nothing
    rates = [lower_reaction(rx, mech, cvar, Tparam, config, j)
             for (j, rx) in enumerate(mech.reactions)]
    eqs = [D(cvars[i]) ~ _species_rhs(mech.species[i].id, mech, rates)
           for i in eachindex(mech.species)]
    @named raw = System(eqs, t)          # dimension check fires here (ValidationError on mismatch)
    return mtkcompile(raw)
end

# —— Constraint-layer assembly (stub until Phase 4) ——

"Append energy/EOS/reactor constraint layers to the equation set. (stub — Phase 4)"
append_constraint_layers!(eqs, mech, config) =
    error("append_constraint_layers!: not implemented; see the Phase 4 plan.")
