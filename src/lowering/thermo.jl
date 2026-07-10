# Thermodynamic lowering (§3.4 #4, §4.2, Phase 4a): NASA7 symbolic cp/h/g for a unit-bearing
# temperature, the equilibrium constant K_c(T) = exp(-Δg°/RT)·(P°/RT)^Δν, and the reverse-rate
# policy machinery (_net_rate + _reverse_rate for Irreversible / ThermoReverse / ExplicitReverse).
# _direct_kf (kinetics.jl) supplies the forward rate constant so ThermoReverse can form kr=kf/Kc.
# The per-species NASA7 coefficient set is cached in _COEFF_CACHE (state.jl) so cp/R, h/RT, g/RT
# share one set of parameters per lower_to_mtk call.

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

"Species-keyed unit-bearing parameter for NASA7 coefficients (rate_param wrapper)."
_sp(sid, tag, val, unit) = rate_param(Symbol("sp", sid, "_", tag), val, unit)

_species_by_id(mech::Mechanism, sid::SpeciesID) = species_by_id(mech, sid)
function _thermo_of(mech::Mechanism, sid::SpeciesID)
    th = _species_by_id(mech, sid).thermo
    th === nothing && error("_thermo_of: species id $sid has no thermo; ThermoReverse needs NASA7 on all species.")
    return th
end
