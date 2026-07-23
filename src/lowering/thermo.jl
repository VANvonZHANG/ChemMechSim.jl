# Thermodynamic lowering (§3.4 #4, §4.2, Phase 4a): NASA7 symbolic cp/h/g for a unit-bearing
# temperature, the equilibrium constant K_c(T) = exp(-Δg°/RT)·(P°/RT)^Δν, and the reverse-rate
# policy machinery (_net_rate + _reverse_rate for Irreversible / ThermoReverse / ExplicitReverse).
# symbolic_kf (kinetics.jl) supplies the forward rate constant so ThermoReverse can form kr=kf/Kc.
# The per-species NASA7 coefficient set is cached in ctx.coeff_cache (state.jl RateCtx/ThermoCtx)
# so cp/R, h/RT, g/RT share one set of parameters per lower_to_mtk call.

using ModelingToolkit: @register_symbolic, @register_derivative
using DynamicQuantities: Quantity, ustrip

# —— Opaque K_c call node (spec §5.2) ——————————————————————————————
# ThermoReverse's kr = kf/K_c inlines a NASA7 Gibbs polynomial at scale (Aramco 479 reverse
# rxn → build_function / dual-RHS compile explosion). Emit K_c's NASA7 part (exp(-Δg°/RT)) as a
# registered call node with a registered analytic ∂/∂T (sin→cos pattern), mirroring plog_kf.
#
# SPLIT DESIGN (opaque NASA7 + inlined P°/RT factor):
#   K_c(T) = exp(-Δg°/RT) · (P°/(R·T))^Δν
#           └── keq(T,id) ──┘   └── inlined symbolic ──┘
# The NASA7 polynomial `exp(-Δg°/RT)` is the part that blows up at scale — it lives in the
# opaque call node. The (P°/RT)^Δν factor is a trivial 1-term expression; it is inlined with
# proper symbolic units (P_std, R_gas, T params from ctx). This split is REQUIRED for the unit
# check: MTK's get_unit probes the registered function ONCE with id=1 (the unitless Quantity
# for the literal arg) and assigns that unit to EVERY call node, so the call node's unit must
# be INVARIANT across reactions. exp(-Δg°/RT) is always dimensionless → invariant. Putting
# conc^Δν inside the opaque node would make the unit rxn-specific → MTK gets it wrong for any
# reaction whose dnu ≠ KEQ_TABLE[1].dnu. The inlined (P°/RT)^Δν carries the conc^Δν unit via
# standard symbolic unit arithmetic (Pa / (J/(mol·K) · K) → mol/m^3 = conc).
const KEQ_TABLE    = Dict{Int, KcData}()       # id → KcData (NASA7 + stoich)
const KEQ_NEXT_ID  = Ref(0)

"Numeric exp(-Δg°/RT) for the registered call node; looks up reaction by id.
 This is the DIMENSIONLESS NASA7 part of K_c; the (P°/RT)^Δν factor is inlined by _reverse_rate."
keq(T, rxid::Int)    = exp(-_delta_g_over_RT_data(KEQ_TABLE[rxid], T))
keq(T::Real, rxid::Int) = exp(-_delta_g_over_RT_data(KEQ_TABLE[rxid], T))
"Numeric ∂/∂T of exp(-Δg°/RT) for the registered call node."
keq_dT(T, rxid::Int) = _keq_dimless_dT(KEQ_TABLE[rxid], T)
keq_dT(T::Real, rxid::Int) = _keq_dimless_dT(KEQ_TABLE[rxid], T)

# Data-layer Δg°/RT (sum of ν·g_over_RT) and its T-derivative. Used by the opaque keq node so
# the lowering layer does NOT re-invoke the symbolic NASA7 polynomial (the whole point).
function _delta_g_over_RT_data(kcd::KcData, T)
    g = 0.0
    for (th, nu) in kcd.prod;  g += nu * g_over_RT(th, T); end
    for (th, nu) in kcd.react; g -= nu * g_over_RT(th, T); end
    return g
end
# Analytic ∂(exp(-Δg°/RT))/∂T via the Gibbs-Helmholtz relation ∂(g/RT)/∂T = -(h/RT)/T:
#   ∂(exp(-g/RT))/∂T = -exp(-g/RT) · ∂(g/RT)/∂T = exp(-g/RT) · (h/RT)/T.
function _keq_dimless_dT(kcd::KcData, T)
    g_RT = _delta_g_over_RT_data(kcd, T)
    h_RT = 0.0
    for (th, nu) in kcd.prod;  h_RT += nu * h_over_RT(th, T); end
    for (th, nu) in kcd.react; h_RT -= nu * h_over_RT(th, T); end
    return exp(-g_RT) * h_RT / T
end

# Quantity-dispatch: MTK's check_units walks the call node with Quantity args (incl. a unitless
# Quantity for the literal rxid). ustrip → Float64 method → result is dimensionless (no re-wrap
# needed for keq; keq_dT gets a 1/K unit). rxid may arrive as Int (numeric call) or unitless
# Quantity (unit check). The returned unit is INVARIANT across reactions (always dimensionless
# for keq, always 1/K for keq_dT) — required because MTK probes with id=Quantity(1.0) and
# applies the result to every call node.
_keq_id_int(id::Int)      = id
_keq_id_int(id::Real)     = Int(id)
_keq_id_int(id::Quantity) = Int(ustrip(id))
keq(T::Real, rxid::Real)    = keq(T, _keq_id_int(rxid))
keq_dT(T::Real, rxid::Real) = keq_dT(T, _keq_id_int(rxid))
keq(T::Quantity, rxid::Int)      = keq(ustrip(T), rxid)                  # dimensionless Quantity
keq(T::Quantity, rxid::Real)     = keq(ustrip(T), _keq_id_int(rxid))     # dimensionless Quantity
keq(T::Quantity, rxid::Quantity) = keq(ustrip(T), _keq_id_int(rxid))     # dimensionless Quantity
keq(T::Quantity, rxid)           = keq(ustrip(T), _keq_id_int(rxid))     # dimensionless Quantity
keq_dT(T::Quantity, rxid::Int)      = keq_dT(ustrip(T), rxid) / u"K"              # 1/K unit
keq_dT(T::Quantity, rxid::Real)     = keq_dT(ustrip(T), _keq_id_int(rxid)) / u"K" # 1/K unit
keq_dT(T::Quantity, rxid::Quantity) = keq_dT(ustrip(T), _keq_id_int(rxid)) / u"K" # 1/K unit
keq_dT(T::Quantity, rxid)           = keq_dT(ustrip(T), _keq_id_int(rxid)) / u"K" # 1/K unit

@register_symbolic keq(T, rxid::Int)
@register_symbolic keq_dT(T, rxid::Int)
@register_derivative keq(T, rxid) 1 keq_dT(T, rxid)     # ∂/∂T  (rxid literal Int — no rule)

# ThermoReverse net rate: forward minus reverse (Task 6, §3.4 #4).

"Net rate = forward − reverse for a reversible reaction (direct path). ExplicitReverse uses the
 general forward dispatch (_direct_rate) since its reverse is independent of the forward k_f.
 ThermoReverse needs the forward k_f (for k_r = k_f/K_c); `symbolic_kf` provides the per-kinetics
 effective rate constant (elementary, ThirdBody, or Troe) so K_c-based reverse works for any
 forward kinetics type. Phase 5a T5fix removed the elementary-only guard that previously blocked
 ThirdBody/Troe under ThermoReverse."
function _net_rate(rx::ReactionData, mech, cvar, T, j, ctx)
    policy = rx.reverse_policy
    if policy isa ExplicitReverse
        fwd = _direct_rate(rx.kinetics, rx, mech, cvar, T, j, ctx)
        return fwd - _reverse_rate(policy, rx, mech, cvar, T, j, ctx)
    end
    if policy isa ThermoReverse
        kf = symbolic_kf(rx.kinetics, ctx)
        fwd = kf * _mass_action(rx.reactants, cvar)
        return fwd - _reverse_rate(policy, rx, mech, cvar, T, kf, ctx)
    end
    error("_net_rate: unsupported policy $(typeof(policy))")
end

_reverse_rate(::Irreversible, rx, mech, cvar, T, kf, ctx) = 0.0
function _reverse_rate(::ThermoReverse, rx, mech, cvar, T, kf, ctx)
    T === nothing &&
        error("_reverse_rate(ThermoReverse): K_c(T) needs a T parameter, but none exists.")
    id = (KEQ_NEXT_ID[] += 1)
    KEQ_TABLE[id] = KcData(rx, mech)
    dnu = sum(values(rx.products)) - sum(values(rx.reactants))
    # K_c(T) = exp(-Δg°/RT) · (P°/(R·T))^Δν. The NASA7 polynomial exp(-Δg°/RT) is the OPAQUE
    # call node (the part that blows up at scale); the (P°/RT)^Δν factor is inlined symbolically
    # so the conc^Δν unit is derived by standard symbolic unit arithmetic (MTK's per-call-node
    # unit probe is invariant — see keq docstring above). For Δν=0 the factor is 1 (skipped).
    Kc = iszero(dnu) ? keq(T, id) : keq(T, id) * (ctx.P_std / (ctx.R * T))^dnu
    return (kf / Kc) * _mass_action(rx.products, cvar)
end

"Reverse rate for an ExplicitReverse policy: kr(T)·∏products, kr from the policy's own rate law.
 Phase 3 supports policy.rate::ElementaryArrhenius (the common explicit-reverse form)."
function _reverse_rate(policy::ExplicitReverse, rx::ReactionData, mech, cvar, T, j, ctx)
    policy.rate isa ElementaryArrhenius ||
        error("_reverse_rate(ExplicitReverse): non-Arrhenius reverse rate ($(typeof(policy.rate))) " *
              "deferred; use ElementaryArrhenius.")
    order = sum(values(rx.products))                       # reverse "reactants" = forward products
    kin = policy.rate
    b, Ea = kin.b, kin.Ea
    prefix = "k_$(j)_rev"                                   # _rev suffix avoids name clash
    A = rate_param(Symbol(prefix, "_A"), kin.A, _k_unit(order, b))
    kr = if iszero(b) && iszero(Ea)
        A                                                   # constant rate, no T dependence
    elseif iszero(Ea)
        A * T^b                                             # power-law k=A·T^b, no θ
    else
        θ = rate_param(Symbol(prefix, "_theta"), Ea / R_GAS, u"K")
        _arrhenius_body(A, b, θ, T)                         # full: A·T^b·exp(-θ/T)
    end
    return kr * _mass_action(rx.products, cvar)
end

"LEGACY numeric inlined K_c(T) = exp(-Δg°/RT)·(P°/(R·T))^Δν (spec §3.4 #4, §4.2).
 Kept as a numeric-test helper ONLY — _reverse_rate(ThermoReverse) now emits an opaque keq(T,id)
 call node and the data-layer `equilibrium_constant(KcData, T)` is the single source of truth.
 Tests in test/test_thermo.jl use this for the cross-check `equilibrium_constant ≈ _equilibrium_constant`.
 Δν = Σν_products − Σν_reactants. Δν=0 → factor is 1 (the existing behavior; current tests unaffected).
 Δg°/RT is dimensionless and (P°/RT)^Δν carries the concentration-basis unit — both pass the dim check."
function _equilibrium_constant(mech::Mechanism, rx::ReactionData, T, ctx)
    dnu = sum(values(rx.products)) - sum(values(rx.reactants))
    base = exp(-_delta_g_over_RT(mech, rx, T, ctx))
    iszero(dnu) && return base
    return base * (ctx.P_std / (ctx.R * T))^dnu
end

"LEGACY numeric ∂g°/RT for the legacy `_equilibrium_constant` (see its docstring).
 Kept for the legacy numeric-only K_c helper; the opaque keq call node does NOT use it."
function _delta_g_over_RT(mech::Mechanism, rx::ReactionData, T, ctx)
    g = 0.0
    for (sid, nu) in rx.products;  g += nu * _g_over_RT(_thermo_of(mech, sid), T, sid, ctx); end
    for (sid, nu) in rx.reactants; g -= nu * _g_over_RT(_thermo_of(mech, sid), T, sid, ctx); end
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
_g_over_RT(m::NASA7, T::Real, sid, ctx) = g_over_RT(m, T)

"Dimensionless g/RT from NASA7 thermo for a symbolic/unit-bearing T (the K-param in lowering).
 Materializes unit-bearing coefficients (cached in ctx.coeff_cache via _nasa7_coeffs_sym), then
 calls the data-layer _nasa7_h and _nasa7_s bodies (single polynomial definition; T2 dedup).
 The s/R log term needs a dimensionless argument for MTK's dim check, so log(T) from _nasa7_s
 is replaced with log(T/T_ref) where T_ref = 1 K (numerically identical; verified 2026-06-29)."
function _g_over_RT(m::NASA7, T::Num, sid, ctx)
    coeffs, _, T_ref = _nasa7_coeffs_sym(m, T, sid, ctx)
    a1 = coeffs[1]
    # h/RT via data body; s/R via data body, fixing the log argument for the dim check:
    #   _nasa7_s uses a1*log(T); replace with a1*log(T/T_ref) so log is dimensionless.
    return _nasa7_h(coeffs, T) - (_nasa7_s(coeffs, T) - a1*log(T) + a1*log(T / T_ref))
end
_g_over_RT(m::ThermoModel, T, sid, ctx) = error("_g_over_RT: thermo model $(typeof(m)) unsupported; only NASA7.")

"Build NASA7's 7 unit-bearing symbolic coefficients for species `sid`, selected by
 `ifelse(T <= Tmid, lo, hi)`. a_i·T^(i-1) is dimensionless (i=1..5), a6∈K makes a6/T and a6
 dimensionless via the 1/T and (h/RT) terms, a1/a7 are dimensionless, Tmid∈K makes the range
 test unit-consistent, Tref=1 K makes log(T/Tref) dimensionless. Cached per sid within one
 lower_to_mtk call so cp/R, h/RT, g/RT share one coefficient set (no duplicate-name params).
 Returns ((a1..a7), Tmid_K, T_ref). Verified 2026-07-02."
function _nasa7_coeffs_sym(m::NASA7, T, sid, ctx)
    c = get(ctx.coeff_cache, sid, nothing)
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
    ctx.coeff_cache[sid] = c
    return c
end

"Symbolic dimensionless cp/R for a unit-bearing T (energy equation, Phase 4a).
 Calls the data-layer _nasa7_cp body (single polynomial definition; T2 dedup)."
function _cp_over_R(m::NASA7, T::Num, sid, ctx)
    coeffs, _, _ = _nasa7_coeffs_sym(m, T, sid, ctx)
    return _nasa7_cp(coeffs, T)
end

"Symbolic dimensionless h/RT for a unit-bearing T (energy equation, Phase 4a).
 Calls the data-layer _nasa7_h body (single polynomial definition; T2 dedup)."
function _h_over_RT(m::NASA7, T::Num, sid, ctx)
    coeffs, _, _ = _nasa7_coeffs_sym(m, T, sid, ctx)
    return _nasa7_h(coeffs, T)
end

"Species-keyed unit-bearing parameter for NASA7 coefficients (rate_param wrapper)."
_sp(sid, tag, val, unit) = rate_param(Symbol("sp", sid, "_", tag), val, unit)

_species_by_id(mech::Mechanism, sid::SpeciesID) = species_by_id(mech, sid)
function _thermo_of(mech::Mechanism, sid::SpeciesID)
    th = _species_by_id(mech, sid).thermo
    th === nothing && error("_thermo_of: species id $sid has no thermo; ThermoReverse needs NASA7 on all species.")
    return th
end
