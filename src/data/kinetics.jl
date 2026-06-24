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
