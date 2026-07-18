# Kinetics (rate-law) type hierarchy. Pure Julia (no MTK/unit dependency).
#
# Type family is AbstractKinetics (NOT AbstractRateLaw) to match the CHEMKIN/
# Cantera "kinetics" vocabulary and to echo the ReactionData.kinetics field name.
# Rate EVALUATION is deferred — these structs only carry parameters.
# Falloff forms (Troe/SRI/Lindemann) use SEPARATE concrete types because their
# parameter sets differ (Troe: 4 params, SRI: 3 params, Lindemann: none), rather
# than a single type with a polymorphic `form` field.

# —— Falloff center-broadening parameter packs ——

struct TroeParams
    α::Float64
    T1::Float64
    T2::Float64
    T3::Float64
end

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
abstract type AbstractFalloff <: AbstractKinetics end

struct TroeFalloff <: AbstractFalloff
    low_rate::ElementaryArrhenius
    high_rate::ElementaryArrhenius
    efficiencies::Dict{SpeciesID,Float64}
    troe::TroeParams
end

struct SRIFalloff <: AbstractFalloff
    low_rate::ElementaryArrhenius
    high_rate::ElementaryArrhenius
    efficiencies::Dict{SpeciesID,Float64}
    sri::SRIParams
end

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

"Full PLOG interpolation with low/high clamping. ks = N k_i(T) values; log_P = ln(P/P_ref);
 log_Pi = N ln(P_i/P_ref) (ascending). Folds ifelse from the high end (Num-safe: builds one
 nested ifelse expression when called with symbolic inputs)."
function _plog_interpolate(ks, log_P, log_Pi)
    n = length(ks)
    n == length(log_Pi) || error("_plog_interpolate: ks ($(length(ks))) / log_Pi ($(length(log_Pi))) length mismatch")
    n == 1 && return ks[1]                              # degenerate single-point → constant rate
    result = ks[n]                                      # high clamp
    for i in (n - 1):-1:1
        f = (log_P - log_Pi[i]) / (log_Pi[i + 1] - log_Pi[i])
        seg = _plog_interp_segment(ks[i], ks[i + 1], f)
        # log_P ≤ log_Pi[i]   → low-side clamp ks[i]
        # log_Pi[i] < log_P ≤ log_Pi[i+1] → this segment
        # else → whatever we'd built above
        result = ifelse(log_P <= log_Pi[i], ks[i],
                        ifelse(log_P <= log_Pi[i + 1], seg, result))
    end
    return result
end

"Group points by unique log-pressure (after sort), summing ks within each group. Returns
 (unique_ks, unique_log_Pi) for downstream _plog_interpolate. Pure arithmetic, generic.
 Matches Cantera's same-pressure PLOG semantics (rates sum at each pressure, then interpolate).
 Assumes ks/log_Pi are already sorted ascending by log_Pi (the parser sorts by P before building
 PlogRate). Exact Float64 equality on log_Pi is safe: Pa-converted atm values from the same YAML
 entry produce bit-identical log_Pi (same Float64 multiply + log call)."
function _plog_sum_at_pressures(ks, log_Pi)
    n = length(ks)
    n == length(log_Pi) || error("_plog_sum_at_pressures: length mismatch")
    out_ks = Any[]; out_lp = Any[]
    i = 1
    while i ≤ n
        j = i
        s = ks[i]
        while j + 1 ≤ n && log_Pi[j + 1] == log_Pi[i]   # same pressure (exact-equal: log_Pi are log(P/P_STD), Float64)
            j += 1; s += ks[j]
        end
        push!(out_ks, s); push!(out_lp, log_Pi[i])
        i = j + 1
    end
    return (out_ks, out_lp)
end

"Numeric PLOG rate constant k(T,P) — MTK-free standalone eval (Cantera comparison, plots, tests).
 Uses P_STD as the dimensionless-reference scaffold (its value cancels in the ratios).
 Groups same-pressure points (sum k_i(T) at each unique P) before interpolating — Cantera semantics."
function plog_rate(kin::PlogRate, T::Real, P::Real)
    ks = [_arrhenius_body(p.A, p.b, p.Ea / R_GAS, T) for p in kin.points]
    log_Pi = [log(p.P / P_STD) for p in kin.points]      # may have duplicates (helper groups)
    s_ks, s_lp = _plog_sum_at_pressures(ks, log_Pi)
    return _plog_interpolate(s_ks, log(P / P_STD), s_lp)
end

"Arrhenius kᵢ(T)=A·T^b·exp(-θ/T) and its T-derivative kᵢ'=kᵢ·(b/T+θ/T²). Returns (kᵢ, kᵢ')."
_arrhenius_k_dkT(A, b, θ, T) = (k = A*T^b*exp(-θ/T); (k, k*(b/T + θ/T^2)))

"∂k/∂T for PLOG (analytic, MTK-free, generic over Real). Same pressure-grouping + log-log
 interpolation structure as plog_rate, with each point using kᵢ' instead of kᵢ. In-segment
 (f is P-only): ∂k/∂T = k·[(1-f)(k_lo'/k_lo) + f(k_hi'/k_hi)]. Low/high clamps → endpoint kᵢ'."
function plog_dkdT(kin::PlogRate, T::Real, P::Real)
    kd = [_arrhenius_k_dkT(p.A, p.b, p.Ea / R_GAS, T) for p in kin.points]
    ks  = [first(x) for x in kd]
    dks = [last(x)  for x in kd]
    logPi = [log(p.P / P_STD) for p in kin.points]
    s_ks,  s_lp = _plog_sum_at_pressures(ks,  logPi)
    s_dks, _   = _plog_sum_at_pressures(dks, logPi)
    return _plog_interp_derivT(s_ks, s_dks, log(P / P_STD), s_lp)
end

"∂k/∂P for PLOG (analytic, MTK-free). In-segment: ∂k/∂P = k·ln(k_hi/k_lo)·(1/P)/(logPᵢ₊₁−logPᵢ).
 Clamps → 0 (k constant w.r.t. P outside range)."
function plog_dkdP(kin::PlogRate, T::Real, P::Real)
    ks = [_arrhenius_body(p.A, p.b, p.Ea / R_GAS, T) for p in kin.points]
    logPi = [log(p.P / P_STD) for p in kin.points]
    s_ks, s_lp = _plog_sum_at_pressures(ks, logPi)
    return _plog_interp_derivP(s_ks, log(P / P_STD), s_lp, P)
end

"Segment-fold for ∂k/∂T (ks=N summed k, dks=N summed k'). Folds high→low like _plog_interpolate."
function _plog_interp_derivT(ks, dks, log_P, log_Pi)
    n = length(ks)
    n == 1 && return dks[1]
    out = dks[n]                                  # high clamp
    for i in (n - 1):-1:1
        f = (log_P - log_Pi[i]) / (log_Pi[i + 1] - log_Pi[i])
        klo, khi = ks[i], ks[i + 1]
        seg_k = klo^(1 - f) * khi^f
        seg_d = seg_k * ((1 - f) * dks[i] / klo + f * dks[i + 1] / khi)
        out = ifelse(log_P <= log_Pi[i], dks[i],
                     ifelse(log_P <= log_Pi[i + 1], seg_d, out))
    end
    return out
end

"Segment-fold for ∂k/∂P (ks=N summed k). Clamps → 0."
function _plog_interp_derivP(ks, log_P, log_Pi, P)
    n = length(ks)
    n == 1 && return 0.0
    out = 0.0                                     # high clamp
    for i in (n - 1):-1:1
        klo, khi = ks[i], ks[i + 1]
        f = (log_P - log_Pi[i]) / (log_Pi[i + 1] - log_Pi[i])
        seg_k = klo^(1 - f) * khi^f
        seg_d = seg_k * log(khi / klo) * (1 / P) / (log_Pi[i + 1] - log_Pi[i])
        out = ifelse(log_P <= log_Pi[i], 0.0,
                     ifelse(log_P <= log_Pi[i + 1], seg_d, out))
    end
    return out
end

struct ChebyshevRate <: AbstractKinetics end

# —— generic formula bodies (pure arithmetic; MTK-free; Real and symbolic Num both work) ——
# These are the SINGLE definition of each formula. The lowering layer calls them with
# unit-bearing symbolic params; the numeric path (rate_constant) calls them with Float64.

"Arrhenius k(T) = A·T^b·exp(-θ/T), θ = Ea/R. Generic over T (Real or symbolic Num)."
_arrhenius_body(A, b, θ, T) = A * T^b * exp(-θ / T)

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

# —— PLOG symbolic lowering (MTK-free composition) —————————————————————
# A grid law needing a hand-written symbolic_kf (not the paramspec materializer).
# MTK-free: calls module-scope _aparam/_kparam (lowering/units.jl, resolved at call time)
# + pure arithmetic; the Num values flow through via Base dispatch. No `using MTK`.
# Colocated with PlogRate (the user's "PLOG is a special reaction" decision; spec §2/§3).

"Symbolic PLOG forward rate constant k(T,P). Materializes 2N params (A_i, θ_i per point;
 P_i and b_i are plain Float64), builds each k_i(T) via _arrhenius_dimless_body, groups same-pressure
 points (sum k_i(T) at each unique P — Cantera semantics), then log-log interpolates via
 _plog_interpolate using ctx.P (pressure symbol). Requires ctx.P ≠ nothing.
 Uses _arrhenius_dimless_body (T_ref-normalized) instead of _arrhenius_body.

 PLOG uses a T_ref-normalized Arrhenius body: k_i = A_i·(T/T_ref)^b_i·exp(-θ_i/T) instead of
 the generic A·T^b. Different PLOG points have different b_i, and in the interpolation ratio
 (k_hi/k_lo)^f the T^b dimensions must cancel. DynamicQuantities stores exponents as
 FixedRational{Int32,25200}, so T^b_i and T^b_j use independently-rounded rationals that DON'T
 exactly cancel in the ratio (e.g. 3.817→24047/6300, 4.149→20911/5040, but the difference
 0.332→4183/12600 ≠ 789/25200 = 20911/5040−24047/6300). Using (T/T_ref)^b makes each T^b factor
 dimensionless, so all k_i share the same unit regardless of b_i. The A-factor carries the
 rate-constant unit directly (conc^(1-order)·s⁻¹), NOT the b-dependent _k_unit.
 The same issue affects Troe/Lindemann: Pr = k0·M/kinf involves a ratio of Arrhenius
 expressions with different b values."
_arrhenius_dimless_body(A, b, θ, T, T_ref) = A * (T / T_ref)^b * exp(-θ / T)

function symbolic_kf(kin::PlogRate, ctx)
    ctx.P === nothing && error("symbolic_kf(PlogRate): pressure-dependent rate needs ctx.P " *
                               "(a config with eos=:ideal_gas); got nothing.")
    n = length(kin.points)
    T_ref = _tvparam(ctx, "_Tref", 1.0)
    ks = ntuple(i -> let p = kin.points[i]
        _arrhenius_dimless_body(_aparam(ctx, "_p$i", p.A, 0.0), p.b,
                                _kparam(ctx, "_p$i", p.Ea), ctx.T, T_ref)
    end, n)
    log_Pi = ntuple(i -> log(kin.points[i].P / P_STD), n)   # plain Float64, may have dups
    s_ks, s_lp = _plog_sum_at_pressures(collect(ks), collect(log_Pi))
    return _plog_interpolate(s_ks, log(ctx.P / ctx.P_std), s_lp)
end

"PLOG is pressure-dependent."
needs_P(kin::PlogRate) = true
