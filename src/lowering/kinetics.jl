# Rate-law lowering (§5.2, §5.4, §3.3): turn an AbstractKinetics rate law into a symbolic
# rate expression. Two paths share the same unit-bearing k and the same @species:
#   - catalyst_lowering  : plain elementary Arrhenius via Catalyst.Reaction + oderatelaw
#                          (combinatoric_ratelaw=false), spec §3.3 layer 2;
#   - direct_mtk_lowering: everything else (third-body, Troe, Lindemann, ...), building the
#                          symbolic rate directly, spec §3.3 layer 3.
# lower_reaction dispatches between them. symbolic_kf returns the effective forward rate
# constant EXCLUDING the mass-action term (so thermo.jl's ThermoReverse can form kr=kf/Kc).

using ModelingToolkit: @register_symbolic, @register_derivative
using DynamicQuantities: Quantity, ustrip

# —— Opaque PLOG call node (spec §5.2) ——————————————————————————————
# PLOG rate is emitted as a call to a registered Julia function (not an inlined ifelse tree),
# with registered analytic derivatives (∂k/∂T, ∂k/∂P) — the sin→cos pattern. This keeps
# symbolic differentiation (calculate_jacobian) working AND cheap. The PlogRate params live
# in a session table keyed by a globally-unique id (safe across multiple lowerings).
#
# The opaque function deliberately bypasses MTK's *inner* symbolic unit check on the PLOG
# interpolation body (architecture decision — see chemmechsim-plog-jacobian-rootcause memory).
# The unit BOUNDARY is preserved: a Quantity-dispatch on plog_kf/dT/dP re-wraps the Float64
# result with the reaction's rate-constant unit (conc^(1-order)·s⁻¹), so MTK's check_units
# derives the correct call-node unit and the equation-level dim check still fires.
const PLOG_TABLE     = Dict{Int, PlogRate}()           # id → PlogRate (params)
const PLOG_UNIT_TABLE = Dict{Int, Any}()               # id → output Quantity unit (conc^(1-order)·s⁻¹)
const PLOG_NEXT_ID   = Ref(0)

"Numeric PLOG k(T,P) for the registered call node; looks up reaction by id."
plog_kf(T, P, id::Int)    = plog_rate(PLOG_TABLE[id], T, P)
plog_kf(T::Real, P::Real, id::Int) = plog_rate(PLOG_TABLE[id], T, P)
"Numeric ∂k/∂T for the registered call node."
plog_kf_dT(T, P, id::Int) = plog_dkdT(PLOG_TABLE[id], T, P)
plog_kf_dT(T::Real, P::Real, id::Int) = plog_dkdT(PLOG_TABLE[id], T, P)
"Numeric ∂k/∂P for the registered call node."
plog_kf_dP(T, P, id::Int) = plog_dkdP(PLOG_TABLE[id], T, P)
plog_kf_dP(T::Real, P::Real, id::Int) = plog_dkdP(PLOG_TABLE[id], T, P)

# Quantity-dispatch: extract SI values, call the Float64 method, re-wrap with the stored
# output unit. Invoked by MTK's check_units (get_unit walks the call node and fires this with
# unit-bearing Quantity args for ALL parameters, including the literal id which arrives as a
# unitless Quantity). T is in K, P is in Pa (both SI base), so ustrip gives exactly what
# plog_rate expects. id may arrive as Int (numeric call) or unitless Quantity (unit check).
_id_int(id::Int)       = id
_id_int(id::Real)      = Int(id)
_id_int(id::Quantity)  = Int(ustrip(id))
plog_kf(T::Real, P::Real, id::Real)    = plog_kf(T, P, _id_int(id))
plog_kf_dT(T::Real, P::Real, id::Real) = plog_kf_dT(T, P, _id_int(id))
plog_kf_dP(T::Real, P::Real, id::Real) = plog_kf_dP(T, P, _id_int(id))
plog_kf(T::Quantity, P::Quantity, id::Int)    = plog_kf(ustrip(T), ustrip(P), id)    * PLOG_UNIT_TABLE[id]
plog_kf_dT(T::Quantity, P::Quantity, id::Int) = plog_kf_dT(ustrip(T), ustrip(P), id) * PLOG_UNIT_TABLE[id] / u"K"
plog_kf_dP(T::Quantity, P::Quantity, id::Int) = plog_kf_dP(ustrip(T), ustrip(P), id) * PLOG_UNIT_TABLE[id] / u"Pa"
plog_kf(T::Quantity, P::Quantity, id)    = plog_kf(ustrip(T), ustrip(P), _id_int(id))    * PLOG_UNIT_TABLE[_id_int(id)]
plog_kf_dT(T::Quantity, P::Quantity, id) = plog_kf_dT(ustrip(T), ustrip(P), _id_int(id)) * PLOG_UNIT_TABLE[_id_int(id)] / u"K"
plog_kf_dP(T::Quantity, P::Quantity, id) = plog_kf_dP(ustrip(T), ustrip(P), _id_int(id)) * PLOG_UNIT_TABLE[_id_int(id)] / u"Pa"

@register_symbolic plog_kf(T, P, id::Int)
@register_symbolic plog_kf_dT(T, P, id::Int)
@register_symbolic plog_kf_dP(T, P, id::Int)
@register_derivative plog_kf(T, P, id) 1 plog_kf_dT(T, P, id)     # ∂/∂T
@register_derivative plog_kf(T, P, id) 2 plog_kf_dP(T, P, id)     # ∂/∂P  (id is a literal Int — no rule)

"Explicit per-type T-dependence (migrated verbatim from _is_T_dependent; Brusselator-safe).
 True iff rate law `kin` needs a temperature symbol (T-dependent Arrhenius)."
needs_T(kin::ElementaryArrhenius) = !(iszero(kin.b) && iszero(kin.Ea))
needs_T(kin::ThirdBodyArrhenius) = needs_T(kin.base)
needs_T(kin::AbstractFalloff) = true

"True iff any reaction in `mech` needs a T parameter (forward kinetics or reverse)."
_needs_T(mech::Mechanism) =
    any(needs_T(rx.kinetics) || _reverse_needs_T(rx.reverse_policy) for rx in mech.reactions)
_reverse_needs_T(::ThermoReverse) = true
_reverse_needs_T(::ReverseRatePolicy) = false

"True iff any reaction in `mech` needs pressure (P-needing kinetics law, e.g. PLOG)."
_needs_P(mech::Mechanism) = any(needs_P(rx.kinetics) for rx in mech.reactions)

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
 does not represent natively (third-body/falloff/PLOG/Chebyshev/etc.)."
catalyst_native(rx::ReactionData, config::MechanismConfig) =
    rx.kinetics isa ElementaryArrhenius

"Symbolic NET rate for one reaction (forward minus reverse). Irreducible elementary
 reactions may go via the Catalyst path; everything else (and all reversible reactions)
 use the direct path. `j` is the reaction index (for naming its rate parameters)."
function lower_reaction(rx::ReactionData, mech::Mechanism, cvar, T, config::MechanismConfig, j::Int, ctx::RateCtx)
    rx.reverse_policy isa Irreversible ||
        return _net_rate(rx, mech, cvar, T, j, ctx)            # ThermoReverse (Task 6)
    return catalyst_native(rx, config) ? catalyst_lowering(rx, mech, cvar, T, j, ctx) :
                                         direct_mtk_lowering(rx, mech, cvar, T, j, ctx)
end

"Direct-MTK lowering path: build the symbolic rate by dispatching on kinetics type."
direct_mtk_lowering(rx::ReactionData, mech::Mechanism, cvar, T, j::Int, ctx::RateCtx) =
    _direct_rate(rx.kinetics, rx, mech, cvar, T, j, ctx)

"Elementary Arrhenius forward rate: k(T)·∏ reactants. k is a unit-bearing rate_param.
 Thin wrapper over `symbolic_kf` (DRY): rate = kf · mass_action(reactants)."
_direct_rate(kin::ElementaryArrhenius, rx, mech, cvar, T, j, ctx::RateCtx) =
    symbolic_kf(kin, ctx) * _mass_action(rx.reactants, cvar)

"Third-body enhanced rate (spec §5.2): k_base(T)·∏ reactants·[M]_eff. The third body is
 NOT a reactant — it enters via the efficiencies map. k_base carries a unit one order higher
 than the elementary base (the [M]_eff factor adds one concentration).
 Thin wrapper over `symbolic_kf` (DRY): rate = kf · mass_action(reactants)."
_direct_rate(kin::ThirdBodyArrhenius, rx, mech, cvar, T, j, ctx::RateCtx) =
    symbolic_kf(kin, ctx) * _mass_action(rx.reactants, cvar)

"Troe falloff forward rate (spec §5.2, §3.4 #2). k_blend = kinf·(Pr/(1+Pr))·F_Troe, with
 Pr = k0·[M]_eff/kinf (dimensionless). kinf carries the high-pressure (Σν-reactant) unit;
 k0 carries one order higher. Verified 2026-06-26: log10/10^x/exp of symbolic Nums survive
 mtkCompile and the dimension check passes (Pr dimensionless)."
function _direct_rate(kin::TroeFalloff, rx, mech, cvar, T, j, ctx::RateCtx)
    T === nothing && error("_direct_rate(TroeFalloff): falloff is T-dependent but no T parameter exists.")
    return symbolic_kf(kin, ctx) * _mass_action(rx.reactants, cvar)
end

"Lindemann falloff forward rate (spec §5.2). F≡1 (no center broadening), so the rate is
 kinf·(Pr/(1+Pr))·∏reactants. Dispatched like TroeFalloff; symbolic_kf provides the effective
 forward rate constant so ThermoReverse can compute kr = kf/Kc consistently."
function _direct_rate(kin::LindemannFalloff, rx, mech, cvar, T, j, ctx::RateCtx)
    T === nothing && error("_direct_rate(LindemannFalloff): falloff is T-dependent but no T parameter exists.")
    return symbolic_kf(kin, ctx) * _mass_action(rx.reactants, cvar)
end

# Fallback for custom kinetics types: route through symbolic_rate (design §5 protocol),
# which defaults to symbolic_kf × mass-action but can be overridden by laws with extra
# concentration-dependent factors (e.g. inhibition). Built-in types (ElementaryArrhenius,
# ThirdBodyArrhenius, TroeFalloff, LindemannFalloff) have explicit _direct_rate methods
# above and never reach this fallback.
_direct_rate(kin::AbstractKinetics, rx, mech, cvar, T, j, ctx::RateCtx) =
    symbolic_rate(rx.kinetics, rx, ctx)

# —— Symbolic k_f(T) via data-layer bodies (Task 4: eliminate inline formulas) ——————————
# Each built-in symbolic_kf method materializes unit-bearing params via _aparam/_kparam/
# _tvparam (same names as the former _arrhenius_k_param/_troe_F), then calls the data-layer
# body _arrhenius_body / _troe_F_body. The symbolic expressions are IDENTICAL to the former
# inline formulas (behavior-preserving). The param unit derivation must match the former
# code: ElementaryArrhenius uses ctx.order (Σ reactants); ThirdBody uses ctx.order+1 (for
# [M]_eff); Troe/Lindemann kinf uses ctx.order, k0 uses ctx.order+1.

"Symbolic k_f(T) for ElementaryArrhenius. Preserves the three regimes of the former
 _arrhenius_k_param: constant (b=Ea=0, no T ref → not T-dependent), power-law (Ea=0),
 full Arrhenius. Param names k_{j}_A / k_{j}_theta unchanged."
function symbolic_kf(kin::ElementaryArrhenius, ctx::RateCtx)
    A = _aparam(ctx, "", kin.A, kin.b)
    iszero(kin.b) && iszero(kin.Ea) && return A               # constant rate
    iszero(kin.Ea) && return A * ctx.T^kin.b                   # power-law k = A·T^b, no θ
    θ = _kparam(ctx, "", kin.Ea)
    return _arrhenius_body(A, kin.b, θ, ctx.T)                 # full: A·T^b·exp(-θ/T)
end

"ThirdBody k_f = k_base(T)·[M]_eff. k_base via _arrhenius_body; [M]_eff = Σ α_i·c_i.
 Base A-factor unit uses ctx.order+1 (the +1 for the [M]_eff concentration factor)."
function symbolic_kf(kin::ThirdBodyArrhenius, ctx::RateCtx)
    # order for the base A-factor includes the +1 [M]_eff concentration
    A = rate_param(Symbol("k_", ctx.j, "_A"), kin.base.A, _k_unit(ctx.order + 1, kin.base.b))
    base = iszero(kin.base.b) && iszero(kin.base.Ea) ? A :
           iszero(kin.base.Ea) ? A * ctx.T^kin.base.b :
           _arrhenius_body(A, kin.base.b, _kparam(ctx, "", kin.base.Ea), ctx.T)
    return base * _meff(ctx, kin.efficiencies)
end

"Troe k_f = kinf·(Pr/(1+Pr))·F, Pr = k0·[M]_eff/kinf. kinf/k0 via _arrhenius_dimless_body
 (T_ref-normalized to avoid DynamicQuantities FixedRational rounding on T^(b0-binf) in the
 ratio k0/kinf; see PLOG docstring for details); F via _troe_F_from_fcent with Fcent built
 per _troe_fcent_plan (short-circuits sentinel Troe params so pathological symbolic
 exp(-T/1e-30) is never constructed). kinf A-factor uses ctx.order (high-pressure limit =
 Σ reactant order); k0 uses ctx.order+1 (low-pressure limit adds one [M]_eff concentration)."
function symbolic_kf(kin::TroeFalloff, ctx::RateCtx)
    ctx.T === nothing && error("symbolic_kf(TroeFalloff): falloff is T-dependent but ctx.T is nothing.")
    T_ref = _tvparam(ctx, "_Tref", 1.0)
    kinf = _arrhenius_dimless_body(_aparam(ctx, "_high", kin.high_rate.A, 0.0),
                                    kin.high_rate.b, _kparam(ctx, "_high", kin.high_rate.Ea),
                                    ctx.T, T_ref)
    k0   = _arrhenius_dimless_body(rate_param(Symbol("k_", ctx.j, "_low_A"), kin.low_rate.A,
                                               _k_unit(ctx.order + 1, 0.0)),
                                    kin.low_rate.b, _kparam(ctx, "_low", kin.low_rate.Ea),
                                    ctx.T, T_ref)
    meff = _meff(ctx, kin.efficiencies)
    Pr   = k0 * meff / kinf
    # Fcent with degenerate-term short-circuit (data-layer plan; avoids pathological symbolic exp)
    α, tp = kin.troe.α, kin.troe
    plan = _troe_fcent_plan(α, tp.T1, tp.T2, tp.T3)
    t1 = plan[1] == :active ? (1 - α) * exp(-ctx.T / _tvparam(ctx, "T3", tp.T3)) :
         plan[1] == :one    ? (1 - α) : 0.0
    t2 = plan[2] == :active ? α * exp(-ctx.T / _tvparam(ctx, "T1", tp.T1)) :
         plan[2] == :one    ? α : 0.0
    t3 = plan[3] == :active ? exp(-_tvparam(ctx, "T2", tp.T2) / ctx.T) :
         plan[3] == :one    ? 1.0 : 0.0
    F = _troe_F_from_fcent(t1 + t2 + t3, Pr)
    return kinf * (Pr / (1 + Pr)) * F
end

"Lindemann k_f = kinf·Pr/(1+Pr) (F≡1). Same as Troe minus the center-broadening factor.
 kinf A-factor uses ctx.order; k0 uses ctx.order+1. Uses _arrhenius_dimless_body (same
 FixedRational-avoidance rationale as Troe)."
function symbolic_kf(kin::LindemannFalloff, ctx::RateCtx)
    ctx.T === nothing && error("symbolic_kf(LindemannFalloff): falloff is T-dependent but ctx.T is nothing.")
    T_ref = _tvparam(ctx, "_Tref", 1.0)
    kinf = _arrhenius_dimless_body(_aparam(ctx, "_high", kin.high_rate.A, 0.0),
                                    kin.high_rate.b, _kparam(ctx, "_high", kin.high_rate.Ea),
                                    ctx.T, T_ref)
    k0   = _arrhenius_dimless_body(rate_param(Symbol("k_", ctx.j, "_low_A"), kin.low_rate.A,
                                               _k_unit(ctx.order + 1, 0.0)),
                                    kin.low_rate.b, _kparam(ctx, "_low", kin.low_rate.Ea),
                                    ctx.T, T_ref)
    meff = _meff(ctx, kin.efficiencies)
    Pr   = k0 * meff / kinf
    return kinf * (Pr / (1 + Pr))
end

"Symbolic PLOG forward rate constant k(T,P) — OPAQUE call node (not inlined ifelse).
 Registers this reaction's PlogRate in PLOG_TABLE under a fresh id and emits plog_kf(T,P,id).
 ∂k/∂T and ∂k/∂P are supplied by @register_derivative (analytic) so calculate_jacobian
 differentiates through the call node cheaply. Requires ctx.P ≠ nothing.
 The output unit (conc^(1-order)·s⁻¹) is stored in PLOG_UNIT_TABLE so the Quantity-dispatch
 on plog_kf re-wraps the result with the correct rate-constant unit (MTK's check_units then
 derives the call-node unit correctly — the boundary unit is preserved even though the
 opaque function bypasses the inner symbolic unit check on the interpolation body)."
function symbolic_kf(kin::PlogRate, ctx::RateCtx)
    ctx.P === nothing && error("symbolic_kf(PlogRate): pressure-dependent rate needs ctx.P " *
                               "(a config with eos=:ideal_gas); got nothing.")
    id = (PLOG_NEXT_ID[] += 1)
    PLOG_TABLE[id] = kin
    PLOG_UNIT_TABLE[id] = _k_unit(ctx.order, 0.0)      # b=0: PLOG normalizes T via T_ref internally (matches old inliner)
    return plog_kf(ctx.T, ctx.P, id)
end

# —— generic paramspec-driven symbolic_kf (the L2 default for custom / paramspec laws) ——
# A kinetics law that declares paramspec + body (no explicit symbolic_kf) lowers via this:
# materialize each (field, role, tag) into a unit-bearing param (or plain value), then call
# body(vals..., ctx.T). Built-in laws override symbolic_kf per-type (Task 4) and bypass this.

"Materialize one paramspec entry (role + value) into a symbolic param or plain value, per role."
materialize(role::AFactor, ctx, tag, A) = _aparam(ctx, tag, A, role.b)   # b carried by AFactor for unit
materialize(::KTemp,       ctx, tag, Ea) = _kparam(ctx, tag, Ea)
materialize(::KValue,      ctx, tag, T)  = _tvparam(ctx, tag, T)
materialize(::Plain,       ctx, _,   v)  = v                               # plain value, no param

"Generic symbolic k_f for any law declaring paramspec + body. Driven entirely by the role table."
function symbolic_kf(kin::AbstractKinetics, ctx)
    spec = paramspec(kin)
    vals = ntuple(i -> materialize(spec[i][2], ctx, spec[i][3], getfield(kin, spec[i][1])), length(spec))
    return body(kin)(vals..., ctx.T)
end

"Default full forward rate = symbolic_kf × mass-action(reactants). Laws with extra
 concentration-dependent factors (e.g. inhibition, [M]_eff beyond the rate constant)
 override this with an explicit method. Default protocol entry (design §5)."
symbolic_rate(kin::AbstractKinetics, rx::ReactionData, ctx::RateCtx) =
    symbolic_kf(kin, ctx) * _mass_action(rx.reactants, ctx.cvar)

"Effective third-body concentration [M]_eff = Σ_i α_i·[X_i] over all species (default α=1).
 Emits M_eff_j as an algebraic variable (state+algebraic pattern, §7.1): the equation
 M_eff_j ~ Σα·c is registered in ctx.meff_eqs, and M_eff_j (a single symbol) is returned
 for use in the rate. MTK tearing eliminates M_eff_j → observed at compile time. This avoids
 inlining the N_species-term sum into every rate, so MTK's dim-check processes one symbol
 per rate (fast) instead of an N-term expression (hangs for Aramco's 581 species)."
function _meff(ctx::RateCtx, efficiencies::Dict{SpeciesID,Float64})
    t = ModelingToolkit.t
    Mvar = _attach_unit(only(@variables ($(Symbol("M_eff_", ctx.j)))(t)), ChemUnits.conc)
    s = 0.0
    for sp in ctx.mech.species
        alpha = get(efficiencies, sp.id, 1.0)
        s += alpha * ctx.cvar[sp.id]
    end
    push!(ctx.meff_eqs, Mvar ~ s)           # register the algebraic equation
    return Mvar
end

"Catalyst mass-action lowering path (spec §5.4). Builds a Catalyst.Reaction on the
 shared @species with the SAME unit-bearing k, then reads its rate law via oderatelaw."
function catalyst_lowering(rx::ReactionData, mech::Mechanism, cvar, T, j::Int, ctx::RateCtx)
    kin = rx.kinetics
    kin isa ElementaryArrhenius ||
        error("catalyst_lowering: only ElementaryArrhenius is Catalyst-native so far.")
    k = symbolic_kf(kin, ctx)
    subs       = [cvar[sid] for sid in keys(rx.reactants)]
    substoich  = collect(values(rx.reactants))
    prods      = [cvar[sid] for sid in keys(rx.products)]
    prodstoich = collect(values(rx.products))
    crate = Catalyst.Reaction(k, subs, prods, substoich, prodstoich)
    return Catalyst.oderatelaw(crate; combinatoric_ratelaw=false)
end
