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

# Pressure-dependent: PLOG (discrete pressure points), Chebyshev (T-P grid).
# Minimal concrete subtypes — full fields/eval deferred to the Phase 6 plan.
struct PlogRate <: AbstractKinetics end
struct ChebyshevRate <: AbstractKinetics end

# —— generic formula bodies (pure arithmetic; MTK-free; Real and symbolic Num both work) ——
# These are the SINGLE definition of each formula. The lowering layer calls them with
# unit-bearing symbolic params; the numeric path (rate_constant) calls them with Float64.

"Arrhenius k(T) = A·T^b·exp(-θ/T), θ = Ea/R. Generic over T (Real or symbolic Num)."
_arrhenius_body(A, b, θ, T) = A * T^b * exp(-θ / T)

"Troe center-broadening factor F (α, T1, T2, T3, reduced pressure Pr, T). Generic over T."
function _troe_F_body(α, T1, T2, T3, Pr, T)
    Fcent = (1 - α) * exp(-T / T3) + α * exp(-T / T1) + exp(-T2 / T)
    lFc = log10(Fcent); lPr = log10(Pr)
    c = -0.4 - 0.67 * lFc; N = 0.75 - 1.27 * lFc
    f1 = lPr + c; f2 = N - 0.14 * f1
    return 10^(lFc / (1 + (f1 / f2)^2))
end

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
