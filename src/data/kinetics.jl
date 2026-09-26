# Kinetics (rate-law) type hierarchy. Pure Julia (no MTK/unit dependency).
#
# Type family is AbstractKinetics (NOT AbstractRateLaw) to match the CHEMKIN/
# Cantera "kinetics" vocabulary and to echo the ReactionData.kinetics field name.
# Rate EVALUATION is deferred — these structs only carry parameters.
# Falloff forms (Troe/SRI/Lindemann) use SEPARATE concrete types because their
# parameter sets differ (Troe: 4 params, SRI: 3 params, Lindemann: none), rather
# than a single type with a polymorphic `form` field.

# —— Falloff center-broadening parameter packs ——

"""
    TroeParams(α, T1, T2, T3)

Troe center-broadening parameters: blending function Fcent is built from `α` (dimensionless)
and the three temperatures `T1`, `T2`, `T3` [K]. Carried by `TroeFalloff`.
"""
struct TroeParams
    α::Float64
    T1::Float64
    T2::Float64
    T3::Float64
end

"""
    SRIParams(a, b, c)

SRI center-broadening parameters (dimensionless `a`, `b` and temperature `c` [K]).
Carried by `SRIFalloff`.
"""
struct SRIParams
    a::Float64
    b::Float64
    c::Float64
end

# —— Kinetics hierarchy ——

"Abstract parent of all rate-law (kinetics) models."
abstract type AbstractKinetics end

# Basic elementary reaction: Arrhenius k(T) = A·T^b·exp(-Ea/RT)
struct ElementaryArrhenius <: AbstractKinetics
    A::Float64
    b::Float64
    Ea::Float64
end

# Third-body enhanced: H + O2 + M → HO2 + M ; [M]_eff = Σ α_i [X_i]
struct ThirdBodyArrhenius <: AbstractKinetics
    base::ElementaryArrhenius
    efficiencies::Dict{SpeciesID,Float64}
end

# Falloff: low/high-pressure limits + center broadening
"""
    AbstractFalloff

Common supertype of the pressure-falloff rate laws: `TroeFalloff`, `SRIFalloff`,
`LindemannFalloff`. Each blends a low- and high-pressure Arrhenius limit over the
reduced pressure Pr = [M]_eff / k₀-derived scale, with optional center broadening.
"""
abstract type AbstractFalloff <: AbstractKinetics end

"""
    TroeFalloff(low_rate, high_rate, efficiencies, troe)

Pressure-dependent falloff with Troe center broadening: `low_rate`/`high_rate` are the
`ElementaryArrhenius` limits, `efficiencies` the third-body enhancement map (as in
`ThirdBodyArrhenius`), and `troe::TroeParams` the broadening parameters. Lowered
symbolically via `symbolic_kf`; numeric evaluation via `rate_constant`.
"""
struct TroeFalloff <: AbstractFalloff
    low_rate::ElementaryArrhenius
    high_rate::ElementaryArrhenius
    efficiencies::Dict{SpeciesID,Float64}
    troe::TroeParams
end

"""
    SRIFalloff(low_rate, high_rate, efficiencies, sri)

As `TroeFalloff` but with SRI center broadening (`sri::SRIParams`).
"""
struct SRIFalloff <: AbstractFalloff
    low_rate::ElementaryArrhenius
    high_rate::ElementaryArrhenius
    efficiencies::Dict{SpeciesID,Float64}
    sri::SRIParams
end

"""
    LindemannFalloff(low_rate, high_rate, efficiencies)

Plain Lindemann-Hinshelwood falloff — low/high `ElementaryArrhenius` limits plus the
third-body `efficiencies` map, with no center broadening.
"""
struct LindemannFalloff <: AbstractFalloff   # no extra center-broadening params
    low_rate::ElementaryArrhenius
    high_rate::ElementaryArrhenius
    efficiencies::Dict{SpeciesID,Float64}
end

# —— PLOG (pressure-dependent Arrhenius) ——————————————————————————
# log-log linear interpolation in k between discrete pressure points (CHEMKIN/Cantera).
# Implemented in the multiplicative form k = k_lo·(k_hi/k_lo)^f to avoid taking log of a
# dimensioned k (fails MTK's dim check). See docs/.../2026-07-11-phase6-plog-design.md §1.

"One PLOG pressure point: (P [Pa], A, b, Ea [J/mol])."
struct PlogPoint
    P::Float64
    A::Float64
    b::Float64
    Ea::Float64
end

"PLOG rate law: N pressure points (sorted ascending by P). k(T,P) log-log interpolates."
struct PlogRate <: AbstractKinetics
    points::Vector{PlogPoint}
end

"One interpolation segment (multiplicative form, dimensionless ratio → passes dim check)."
_plog_interp_segment(k_lo, k_hi, f) = k_lo * (k_hi / k_lo)^f

"Numeric PLOG rate constant k(T,P) — MTK-free standalone eval (Cantera comparison, plots, tests).
 Uses P_STD as the dimensionless-reference scaffold (its value cancels in the ratios).
 Groups same-pressure points (sum k_i(T) at each unique P) before log-log interpolation —
 Cantera semantics. Allocation-free bracket walk: only the (at most two) groups bracketing
 P are evaluated; the old evaluate-all-channels path computed and discarded the rest."
function plog_rate(kin::PlogRate, T::Real, P::Real)
    log_P = log(P / P_STD)
    st, i0, i1, j0, j1, lp_lo, lp_hi = _plog_bracket(kin, log_P)
    k_lo = _plog_group_k(kin, i0, i1, T)
    (st === :lo) && return k_lo
    k_hi = _plog_group_k(kin, j0, j1, T)
    (st === :hi) && return k_hi
    f = (log_P - lp_lo) / (lp_hi - lp_lo)
    return _plog_interp_segment(k_lo, k_hi, f)      # k_lo·(k_hi/k_lo)^f — multiplicative form
end

"""
    _plog_bracket(kin, log_P) -> (state, i0, i1, j0, j1, lp_lo, lp_hi)

Single allocation-free walk over `kin.points` finding the pressure group bracketing
`log_P = log(P/P_STD)`. Points with exactly equal log(P/P_STD) form one group (their rates
sum — Cantera same-pressure semantics); groups are strictly increasing because the parser
sorts points by P (cantera_yaml.jl).

- `:lo`      — `log_P <= lp` of the first group (low clamp, exact node, or the degenerate
              single-group case): only `(i0, i1)` is meaningful.
- `:between` — strictly inside: lo group `i0:i1`, hi group `j0:j1`. At `log_P == lp_hi`
              the caller computes f = 1.0 exactly, reproducing the old segment-fold value
              at nodes (NOT the raw group sum).
- `:hi`      — `log_P` above the last group (high clamp): only `(j0, j1)` is meaningful.
"""
function _plog_bracket(kin::PlogRate, log_P::Real)
    pts = kin.points
    n = length(pts)
    gs = 1                        # current group: channels gs:ge, log-pressure lp
    ge = 1
    lp = log(pts[1].P / P_STD)
    pgs = pge = 0                 # next-lower group: channels pgs:pge, log-pressure plp
    plp = 0.0
    have_prev = false
    while true
        while ge < n && log(pts[ge + 1].P / P_STD) == lp
            ge += 1               # extend group over same-pressure points
        end
        if log_P <= lp
            return (have_prev ? (:between, pgs, pge, gs, ge, plp, lp)
                              : (:lo, gs, ge, gs, ge, lp, lp))
        end
        have_prev = true
        pgs, pge, plp = gs, ge, lp
        ge == n && return (:hi, gs, ge, gs, ge, lp, lp)
        gs = ge + 1
        ge = gs
        lp = log(pts[gs].P / P_STD)
    end
end

"Sum of Arrhenius k(T) over channels i:j (one pressure group), accumulated in channel
order — the same order (and thus the same Float64 rounding) as the old grouped-sum path."
function _plog_group_k(kin::PlogRate, i::Int, j::Int, T::Real)
    s = _arrhenius_body(kin.points[i].A, kin.points[i].b, kin.points[i].Ea / R_GAS, T)
    for m in (i + 1):j
        s += _arrhenius_body(kin.points[m].A, kin.points[m].b, kin.points[m].Ea / R_GAS, T)
    end
    return s
end

"Arrhenius kᵢ(T)=A·T^b·exp(-θ/T) and its T-derivative kᵢ'=kᵢ·(b/T+θ/T²). Returns (kᵢ, kᵢ')."
_arrhenius_k_dkT(A, b, θ, T) = (k = A*T^b*exp(-θ/T); (k, k*(b/T + θ/T^2)))

"∂k/∂T for PLOG (analytic, MTK-free, generic over Real). Same pressure-grouping + log-log
 interpolation structure as plog_rate, with each group using Σkᵢ' instead of Σkᵢ.
 In-segment (f is P-only): ∂k/∂T = k·[(1-f)(k_lo'/k_lo) + f(k_hi'/k_hi)]. Clamps → endpoint
 group's Σkᵢ'. Allocation-free (shares _plog_bracket with plog_rate)."
function plog_dkdT(kin::PlogRate, T::Real, P::Real)
    log_P = log(P / P_STD)
    st, i0, i1, j0, j1, lp_lo, lp_hi = _plog_bracket(kin, log_P)
    k_lo, dk_lo = _plog_group_k_dkT(kin, i0, i1, T)
    (st === :lo) && return dk_lo
    k_hi, dk_hi = _plog_group_k_dkT(kin, j0, j1, T)
    (st === :hi) && return dk_hi
    f = (log_P - lp_lo) / (lp_hi - lp_lo)
    seg_k = k_lo^(1 - f) * k_hi^f                 # power form — verbatim from the old path
    return seg_k * ((1 - f) * dk_lo / k_lo + f * dk_hi / k_hi)
end

"Per-group sums of (Σk, Σk′) over channels i:j, accumulated in channel order (same
rounding as the old separately-summed ks/dks arrays)."
function _plog_group_k_dkT(kin::PlogRate, i::Int, j::Int, T::Real)
    k, dk = _arrhenius_k_dkT(kin.points[i].A, kin.points[i].b, kin.points[i].Ea / R_GAS, T)
    for m in (i + 1):j
        km, dkm = _arrhenius_k_dkT(kin.points[m].A, kin.points[m].b, kin.points[m].Ea / R_GAS, T)
        k += km; dk += dkm
    end
    return (k, dk)
end

"∂k/∂P for PLOG (analytic, MTK-free). In-segment:
 ∂k/∂P = k_lo^(1-f)·k_hi^f · ln(k_hi/k_lo)·(1/P)/(lp_hi−lp_lo). Clamps → 0 (k constant
 w.r.t. P outside range). Allocation-free (shares _plog_bracket with plog_rate)."
function plog_dkdP(kin::PlogRate, T::Real, P::Real)
    log_P = log(P / P_STD)
    st, i0, i1, j0, j1, lp_lo, lp_hi = _plog_bracket(kin, log_P)
    (st === :lo || st === :hi) && return 0.0
    k_lo = _plog_group_k(kin, i0, i1, T)
    k_hi = _plog_group_k(kin, j0, j1, T)
    f = (log_P - lp_lo) / (lp_hi - lp_lo)
    seg_k = k_lo^(1 - f) * k_hi^f                 # power form — verbatim from the old path
    return seg_k * log(k_hi / k_lo) * (1 / P) / (lp_hi - lp_lo)
end

struct ChebyshevRate <: AbstractKinetics end

# —— generic formula bodies (pure arithmetic; MTK-free; Real and symbolic Num both work) ——
# These are the SINGLE definition of each formula. The lowering layer calls them with
# unit-bearing symbolic params; the numeric path (rate_constant) calls them with Float64.

"Arrhenius k(T) = A·T^b·exp(-θ/T), θ = Ea/R. Generic over T (Real or symbolic Num)."
_arrhenius_body(A, b, θ, T) = A * T^b * exp(-θ / T)

"T_ref-normalized Arrhenius body: k = A·(T/T_ref)^b·exp(-θ/T). Used by falloff lowering
 (Troe/Lindemann) and the former PLOG inliner to avoid DynamicQuantities FixedRational
 rounding on T^(b_i − b_j) in k-ratios: T^b_i and T^b_j use independently-rounded rationals
 that DON'T exactly cancel (e.g. 3.817→24047/6300, 4.149→20911/5040, but the difference
 0.332→4183/12600 ≠ 789/25200 = 20911/5040−24047/6300). Normalizing by T_ref makes each T^b
 factor dimensionless, so all k_i share the same unit regardless of b_i. The A-factor carries
 the rate-constant unit directly (conc^(1-order)·s⁻¹), NOT the b-dependent _k_unit."
_arrhenius_dimless_body(A, b, θ, T, T_ref) = A * (T / T_ref)^b * exp(-θ / T)

"Troe Fcent term degeneracy plan (pure Float64, MTK-free). Returns (t1,t2,t3) ∈ {:zero,:one,:active}
 for term1=(1-α)exp(-T/T3), term2=α·exp(-T/T1), term3=exp(-T2/T). Sentinel thresholds are
 float-dynamic-range limits (1e20/1e-20), mechanism-independent. Negative x is :active (normal)."
function _troe_fcent_plan(α::Float64, T1::Float64, T2::Float64, T3::Float64)
    (_exp_negT_over_x_form(T3), _exp_negT_over_x_form(T1), _exp_neg_x_over_T_form(T2))
end
# exp(-T/x): |x|≤1e-20→:zero, |x|≥1e20→:one, else :active
_exp_negT_over_x_form(x) = abs(x) ≤ 1e-20 ? :zero : (abs(x) ≥ 1e20 ? :one : :active)
# exp(-x/T): |x|≥1e20→:zero, |x|≤1e-20→:one, else :active
_exp_neg_x_over_T_form(x) = abs(x) ≥ 1e20 ? :zero : (abs(x) ≤ 1e-20 ? :one : :active)

"Troe F given Fcent and reduced pressure Pr (the post-Fcent formula). Generic (Real/Num)."
function _troe_F_from_fcent(Fcent, Pr)
    lFc = log10(Fcent); lPr = log10(Pr)
    c = -0.4 - 0.67 * lFc; N = 0.75 - 1.27 * lFc
    f1 = lPr + c; f2 = N - 0.14 * f1
    return 10^(lFc / (1 + (f1 / f2)^2))
end

"Troe center-broadening factor F. Naive Fcent (no degenerate handling) — kept for numeric/simple
 callers. Lowering's symbolic_kf(::TroeFalloff) instead builds Fcent via _troe_fcent_plan (to
 short-circuit sentinel params) then calls _troe_F_from_fcent."
_troe_F_body(α, T1, T2, T3, Pr, T) =
    _troe_F_from_fcent((1 - α) * exp(-T / T3) + α * exp(-T / T1) + exp(-T2 / T), Pr)

# —— param-role types (MTK-free markers; materialize/numeric_value dispatch on them) ——
# Roles describe how a struct field becomes a rate parameter: its unit role + naming.
abstract type ParamRole end
"struct AFactor carries the T^b exponent `b` for A-factor unit derivation ([A] = conc^(1-order)·s⁻¹/K^b)."
struct AFactor <: ParamRole; b::Float64; end
struct KTemp    <: ParamRole; end   # activation energy → θ = Ea/R, unit K
struct KValue   <: ParamRole; end   # plain temperature value (Troe T1/T2/T3), unit K
struct Plain    <: ParamRole; end   # dimensionless plain value (exponent b, scale f), no param

# numeric evaluation rules (MTK-free; live in data so generic rate_constant is pure Julia)
numeric_value(::AFactor, A)  = A
numeric_value(::KTemp,   Ea) = Ea / R_GAS
numeric_value(::KValue,   T) = T
numeric_value(::Plain,    v) = v

# role-table helpers (build (field::Symbol, role::ParamRole, tag) triples for paramspec)
afactor(f, tag, b) = (f, AFactor(b), tag)
ktemp(f, tag)      = (f, KTemp(),    tag)
kvalue(f, tag)     = (f, KValue(),   tag)
plain(f)           = (f, Plain(),    nothing)

# —— per-type declarations (user/framework provides these for each kinetics type) ——
# paramspec(kin) :: NTuple of (field, role, tag); body(kin) :: function(vals..., T);
# needs_T(kin) :: Bool. Generic rate_constant/symbolic_kf (lowering) drive off these.
# No generic fallback for paramspec/body: a law MUST declare them to use the generic path,
# OR provide its own explicit symbolic_kf/rate_constant (built-in laws do — see Task 4).
# Declare the function names (no methods) so external code can extend via Module.func(...).
function paramspec end
function body end

"Generic numeric k_f(T) for a kinetics law that declares paramspec + body. Pure-Real, MTK-free."
function rate_constant(kin::AbstractKinetics, T::Real)
    spec = paramspec(kin)
    vals = ntuple(i -> numeric_value(spec[i][2], getfield(kin, spec[i][1])), length(spec))
    return body(kin)(vals..., T)
end

# Generic needs_T fallback: a law without an explicit needs_T method is assumed T-dependent
# (safe default; built-in laws and custom laws declare the precise method).
needs_T(::AbstractKinetics) = true

"Default: a kinetics law is pressure-independent. PLOG (and future P-dependent laws) override."
needs_P(::AbstractKinetics) = false

# —— PLOG symbolic lowering moved to src/lowering/kinetics.jl (opaque call node) ——
# PLOG k(T,P) is now emitted as a registered Julia call `plog_kf(T,P,id)` with analytic
# ∂k/∂T, ∂k/∂P (the sin→cos pattern), so calculate_jacobian stays cheap. The numeric
# helpers plog_rate / plog_dkdT / plog_dkdP (allocation-free bracket walk, 2026-09-16:
# the old Any[]-grouping path allocated ~4.7 KB per call and dominated FFCM/Aramco warm
# solves) / _arrhenius_dimless_body all stay here (data layer, MTK-free).

"PLOG is pressure-dependent."
needs_P(kin::PlogRate) = true
